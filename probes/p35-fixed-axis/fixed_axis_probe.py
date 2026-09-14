#!/usr/bin/env python3
"""p35 —— 修正 p31–p33 的**匹配轴缺陷**，并在长上下文与更大模型上重测。

## 为什么要重测（自我审计发现的真实缺陷）

p28–p33 的 `dropped_mass` 是对**全部 query** 取平均：
    STATE["dropped"].append(float((1.0 - keep_v.sum(-1)).mean()))     # mean over (B,H,Lq)
但被测量的输出只有**最后一个 query** 的 logits。

而三种结构在**早期 query** 上的行为不对称（因果掩码下）：
  · `window`：窗口 = 行的最后 n 个索引；早期 query 的窗口与 allowed 交集为空
              ⇒ keep_v.sum()=0 ⇒ 该 query 记 dropped = **1.0**（高估）
  · `topn`  ：top-k 会选到权重为 0 的已掩码位置 ⇒ keep_v.sum()=1 ⇒ dropped = **0**（低估）
  · `randn` ：`.topk` 在 masked_fill(-1) 上取，允许位置不够时补到 -1.0 项、权重为 0
              ⇒ dropped = **0**（低估）
⇒ 按**全局平均**匹配时，window 的"真实"（末 query）丢弃质量**低于**目标，**自动获得虚假优势**。
   这与我测到的方向一致 ⇒ **必须重测，不能沿用 p31–p33 的量级**。

## 本探针的修正
1. **只记末 query 的丢弃质量**（`dropped_last`，对 (B,H) 取均值），并**以它为匹配轴**；
2. 同时记全局均值 `dropped_global`，用于量化两轴差多少（把缺陷本身也变成数据）；
3. **分块计算注意力**（query 块 Bq），内存 O(Bq·L) ⇒ 可上 L=4096/16384（eager 全矩阵在 L=4096 需 ~2 GB/层，不可行）；
4. 另加 `norenorm` 对照（不重新归一化 —— 真实 mask 式稀疏核的另一种做法）。

判据（先用 L=512 与 p33 同条件复算，再上长上下文）：
  · 若修正后 **window/randn 的 TV 比落到 <=3x** ⇒ **p33 的结论被自己的匹配缺陷解释掉 ⇒ 候选死**；
  · 若仍 >3x（CI 下界）⇒ 结论在正确匹配轴上成立，继续做长上下文与更大模型。
"""
import argparse, glob, json, math, os, random, sys, torch, torch.nn.functional as F
import transformers.models.qwen3.modeling_qwen3 as M
from transformers import AutoModelForCausalLM, AutoTokenizer

STATE = {"n": None, "mode": "off", "renorm": True, "drop_last": [], "drop_global": []}
_ORIG = M.eager_attention_forward
BQ = 256

def _sparse(q, k, v, mask, scaling, mode, n, renorm):
    """分块注意力 + 结构选择。q,k,v: (B,H,L,D)；mask: (B,1,Lq,Lk) 加性或 None。"""
    Lq = q.shape[-2]
    outs = []
    for i in range(0, Lq, BQ):
        qb = q[:, :, i:i + BQ]
        w = torch.matmul(qb, k.transpose(-1, -2)) * scaling
        if mask is not None:
            w = w + mask[:, :, i:i + BQ, :k.shape[-2]]
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        if mode != "off" and n is not None and w.shape[-1] > n:
            if mode == "topn":
                keep_v, keep_i = w.topk(n, dim=-1)
            elif mode == "randn":
                allowed = w > 0
                rnd = torch.rand_like(w).masked_fill(~allowed, -1.0)
                keep_i = rnd.topk(n, dim=-1).indices
                keep_v = w.gather(-1, keep_i)
            elif mode == "window":
                # 与 p28-p33 定义一致：取行的最后 n 个**索引**（因果掩码下早期 query 会取空）
                keep_i = torch.arange(w.shape[-1] - n, w.shape[-1], device=w.device)
                keep_i = keep_i.view(1, 1, 1, n).expand(*w.shape[:-1], n)
                keep_v = w.gather(-1, keep_i)
            else:
                raise SystemExit("bad mode " + mode)
            last = w.shape[-2] - 1                       # 本块的最后一行 = 真正的末 query（仅最后一块）
            if i + BQ >= Lq:
                STATE["drop_last"].append(float((1.0 - keep_v[:, :, last, :].sum(-1)).mean()))
            STATE["drop_global"].append(float((1.0 - keep_v.sum(-1)).mean()))
            sp = torch.zeros_like(w).scatter_(-1, keep_i, keep_v)
            if renorm:
                sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-12)
            w = sp
        outs.append(torch.matmul(w, v))
    return torch.cat(outs, dim=2)

def _patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    k = M.repeat_kv(key, module.num_key_value_groups)
    v = M.repeat_kv(value, module.num_key_value_groups)
    out = _sparse(query, k, v, attention_mask, scaling, STATE["mode"],
                  STATE["n"], STATE["renorm"])
    return out.transpose(1, 2).contiguous(), None

M.eager_attention_forward = _patched
M.ALL_ATTENTION_FUNCTIONS["eager"] = _patched

def tv(p, q):
    return float(0.5 * (p - q).abs().sum())

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (ph, max(0.0, c - hw), min(1.0, c + hw))

def run_once(model, ids, mode, n, base_p, b1, renorm=True):
    STATE.update(n=n, mode=mode, renorm=renorm, drop_last=[], drop_global=[])
    with torch.no_grad():
        lg = model(ids).logits[0, -1].float()
    q = F.softmax(lg, dim=-1)
    dl = sum(STATE["drop_last"]) / max(1, len(STATE["drop_last"]))
    dg = sum(STATE["drop_global"]) / max(1, len(STATE["drop_global"]))
    return {"n": n, "drop_last": dl, "drop_global": dg, "tv": tv(base_p, q),
            "flip": int(int(lg.argmax()) != b1), "gap": float(lg.topk(2).values[0] - lg.topk(2).values[1])}

def bisect(model, ids, mode, target, L, base_p, b1, renorm, iters=13):
    """按 **末 query 丢弃质量** 匹配（质量随 n 增大而下降）"""
    lo, hi, best = 2, L - 1, None
    for _ in range(iters):
        mid = (lo + hi) // 2
        r = run_once(model, ids, mode, mid, base_p, b1, renorm)
        if best is None or abs(r["drop_last"] - target) < abs(best["drop_last"] - target):
            best = r
        if r["drop_last"] > target:
            lo = mid + 1
        else:
            hi = mid - 1
        if hi <= lo:
            break
    best["mass_ok"] = int(abs(best["drop_last"] - target) <= 0.03 * target)
    best["mass_target"] = target
    return best

def build_contexts(tok, L, root, per_type):
    out = []
    synth = ("The history of computing began with mechanical calculators and evolved through "
             "vacuum tubes, transistors, integrated circuits, and parallel processors. ") * 200
    for s in range(per_type):
        out.append(("synth", s, tok(synth, return_tensors="pt", truncation=True, max_length=L)["input_ids"][0]))
    files = sorted(glob.glob(os.path.join(root, "notes", "*.md"))) + \
            sorted(glob.glob(os.path.join(root, "results", "**", "*.md"), recursive=True)) + \
            sorted(glob.glob(os.path.join(root, "probes", "**", "*.py"), recursive=True))
    rng = random.Random(11); rng.shuffle(files)
    got = 0
    for fn in files:
        if got >= per_type:
            break
        try:
            txt = open(fn, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=L)["input_ids"][0]
        if ids.numel() >= L:
            out.append(("natural", got, ids)); got += 1
    V = tok.vocab_size
    for s in range(per_type):
        out.append(("random", s, torch.tensor([rng.randrange(1000, V) for _ in range(L)], dtype=torch.long)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TARGET", "/root/autodl-tmp/models/Qwen3-4B"))
    ap.add_argument("--root", default="/root/myCCFA")
    ap.add_argument("--ctx-tokens", type=int, default=512)
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--targets", default="0.1,0.2,0.4,0.6")
    ap.add_argument("--modes", default="topn,randn,window")
    ap.add_argument("--renorm", type=int, default=1)
    ap.add_argument("--out", default="/root/ccfa_results/p35/scale.jsonl")
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                attn_implementation="eager").to("cuda").eval()
    targets = [float(x) for x in a.targets.split(",")]
    ctxs = build_contexts(tok, a.ctx_tokens, a.root, a.per_type)
    print("L=%d contexts=%d %s" % (a.ctx_tokens, len(ctxs),
          {t: sum(1 for c in ctxs if c[0] == t) for t in ("synth", "natural", "random")}), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "a")
    for ci, (ctype, cseed, ids0) in enumerate(ctxs):
        ids = ids0.unsqueeze(0).to("cuda")
        L = ids.shape[1]
        STATE.update(n=None, mode="off", renorm=True, drop_last=[], drop_global=[])
        with torch.no_grad():
            base = model(ids).logits[0, -1].float()
        base_p = F.softmax(base, dim=-1); b1 = int(base.argmax())
        for mode in a.modes.split(","):
            for M_ in targets:
                r = bisect(model, ids, mode, M_, L, base_p, b1, bool(a.renorm))
                r.update(ctx_type=ctype, ctx_seed=cseed, L=L, mode=mode, renorm=int(a.renorm))
                fout.write(json.dumps(r) + "\n"); fout.flush()
        print("  ctx %d/%d %s/%d L=%d done" % (ci + 1, len(ctxs), ctype, cseed, L), flush=True)
    fout.close()
    print("P35_DONE ->", a.out)

if __name__ == "__main__":
    sys.exit(main())
