#!/usr/bin/env python3
"""p33 —— 把候选「**质量不是货币，支撑结构才是**」的量级测扎实（精确匹配质量 + N 上下文 + CI）

为什么需要这一步（p31/p32 的遗留缺陷）：
  · p32 用"就近取 n、质量相对误差 <=1.35x"做匹配 ⇒ **质量并没有真正匹配**；
  · N=9 个上下文 ⇒ 中质量段（0.1-0.4）的翻转率 CI 互相重叠，无法判定；
  · 未按上下文类型分层报告 ⇒ 无法排除"只在合成重复文本上成立"。

S3b 给的**冻结杀线**（`notes/S3B_NUMERICAL_DRIFT_2026-09-15.md`）：
  在匹配质量下，若 TV 比 <=3x ⇒ 杀（不优于 2604.13206 已测的稠密方向天花板）；
  且必须同时看到 top-2 margin 的**差异性位移**。

本探针做法：
  对每个 (上下文, 模式, 目标质量 M) 用**二分法**求 n，使丢弃质量落在 M 的 +-2% 内；
  记录 TV、是否翻转、top2 margin 及扰动后的 d_gap。
  报告：分层的翻转率（Wilson CI）、精确匹配点上的 TV 比（bootstrap CI）、margin 位移。

三类上下文（零成本）：synth 重复合成 / natural 仓内真实文本 / random 随机 token（平坦对照）。
每完成一个上下文即落盘（共享机会被抢也不丢数据）。
"""
import argparse, glob, json, math, os, random, sys, torch, torch.nn.functional as F
import transformers.models.qwen3.modeling_qwen3 as M
from transformers import AutoModelForCausalLM, AutoTokenizer

STATE = {"n": None, "mode": "off", "dropped": []}
_ORIG = M.eager_attention_forward

def _patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    key_states = M.repeat_kv(key, module.num_key_value_groups)
    value_states = M.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : key_states.shape[-2]]
    w = F.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    n, mode = STATE["n"], STATE["mode"]
    if mode != "off" and n is not None and w.shape[-1] > n:
        if mode == "topn":
            keep_v, keep_i = w.topk(n, dim=-1)
        elif mode == "randn":
            allowed = w > 0
            rnd = torch.rand_like(w).masked_fill(~allowed, -1.0)
            keep_i = rnd.topk(n, dim=-1).indices
            keep_v = w.gather(-1, keep_i)
        elif mode == "window":
            keep_i = torch.arange(w.shape[-1] - n, w.shape[-1], device=w.device)
            keep_i = keep_i.view(1, 1, 1, n).expand(*w.shape[:-1], n)
            keep_v = w.gather(-1, keep_i)
        else:
            raise SystemExit("bad mode " + mode)
        STATE["dropped"].append(float((1.0 - keep_v.sum(-1)).mean()))
        sp = torch.zeros_like(w).scatter_(-1, keep_i, keep_v)
        w = sp / sp.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.matmul(w, value_states).transpose(1, 2).contiguous(), w

M.eager_attention_forward = _patched
M.ALL_ATTENTION_FUNCTIONS["eager"] = _patched

def tv(p, q):
    return float(0.5 * (p - q).abs().sum())

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (ph, max(0.0, c - hw), min(1.0, c + hw))

def run_once(model, ids, mode, n, base_p, base_top1, base_gap):
    STATE.update(n=n, mode=mode, dropped=[])
    with torch.no_grad():
        lg = model(ids).logits[0, -1].float()
    q = F.softmax(lg, dim=-1)
    dm = sum(STATE["dropped"]) / max(1, len(STATE["dropped"]))
    v = lg.topk(2).values
    return {"n": n, "dropped_mass": dm, "tv": tv(base_p, q),
            "flip": int(int(lg.argmax()) != base_top1),
            "gap": float(v[0] - v[1]), "d_gap": float(v[0] - v[1]) - base_gap}

def bisect_n(model, ids, mode, target, L, base_p, base_top1, base_gap, tol=0.02, iters=9):
    """寻找 n 使 dropped_mass 最接近 target（质量随 n 增大而下降）"""
    lo, hi = 2, L - 1
    best = None
    for _ in range(iters):
        mid = (lo + hi) // 2
        r = run_once(model, ids, mode, mid, base_p, base_top1, base_gap)
        if best is None or abs(r["dropped_mass"] - target) < abs(best["dropped_mass"] - target):
            best = r
        if r["dropped_mass"] > target:
            lo = mid + 1                      # 丢太多 => 保留更多
        else:
            hi = mid - 1
        if hi <= lo:
            break
    ok = abs(best["dropped_mass"] - target) <= tol * target
    best["mass_ok"] = int(ok)
    best["mass_target"] = target
    return best

def build_contexts(tok, L, root, n_per_type):
    out = []
    synth = ("The history of computing began with mechanical calculators and evolved through "
             "vacuum tubes, transistors, integrated circuits, and parallel processors. ") * 80
    for s in range(n_per_type):
        out.append(("synth", s, tok(synth, return_tensors="pt", truncation=True, max_length=L)["input_ids"][0]))
    files = sorted(glob.glob(os.path.join(root, "notes", "*.md"))) + \
            sorted(glob.glob(os.path.join(root, "probes", "**", "*.py"), recursive=True)) + \
            sorted(glob.glob(os.path.join(root, "results", "**", "*.md"), recursive=True))
    rng = random.Random(7)
    rng.shuffle(files)
    got = 0
    for fn in files:
        if got >= n_per_type:
            break
        try:
            txt = open(fn, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=L)["input_ids"][0]
        if ids.numel() >= L:
            out.append(("natural", got, ids)); got += 1
    V = tok.vocab_size
    for s in range(n_per_type):
        ids = torch.tensor([rng.randrange(1000, V) for _ in range(L)], dtype=torch.long)
        out.append(("random", s, ids))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/user/models/Qwen3-4B")
    ap.add_argument("--root", default="/home/user/myCCFA")
    ap.add_argument("--ctx-tokens", type=int, default=512)
    ap.add_argument("--per-type", type=int, default=30)
    ap.add_argument("--targets", default="0.1,0.2,0.4,0.6")
    ap.add_argument("--modes", default="topn,randn,window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/user/ccfa_logs/p33/scale.jsonl")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                attn_implementation="eager").to("cuda").eval()
    targets = [float(x) for x in a.targets.split(",")]
    modes = a.modes.split(",")
    ctxs = build_contexts(tok, a.ctx_tokens, a.root, a.per_type)
    print("contexts:", len(ctxs), "by type:", {t: sum(1 for c in ctxs if c[0] == t) for t in ("synth", "natural", "random")}, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "a")

    for ci, (ctype, cseed, ids0) in enumerate(ctxs):
        ids = ids0.unsqueeze(0).to("cuda")
        L = ids.shape[1]
        STATE.update(n=None, mode="off", dropped=[])
        with torch.no_grad():
            base = model(ids).logits[0, -1].float()
        base_p = F.softmax(base, dim=-1)
        b1 = int(base.argmax())
        bv = base.topk(2).values
        bgap = float(bv[0] - bv[1])
        for mode in modes:
            for M_ in targets:
                r = bisect_n(model, ids, mode, M_, L, base_p, b1, bgap)
                r.update(ctx_type=ctype, ctx_seed=cseed, ctx_idx=ci, L=L, mode=mode, gap_base=bgap)
                fout.write(json.dumps(r) + "\n"); fout.flush()
        print("  ctx %d/%d %s/%d L=%d done" % (ci + 1, len(ctxs), ctype, cseed, L), flush=True)
    fout.close()
    print("P33_DONE ->", a.out)

if __name__ == "__main__":
    sys.exit(main())
