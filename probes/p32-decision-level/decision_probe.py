#!/usr/bin/env python3
"""p32 —— 按质量匹配的**决策层**检验：结构效应到底能不能到决策层？以及"可匹配质量带"是否被上下文锁死？

p31 的结论（soft 层）：质量匹配后 oracle 漂移比 random 小 2.4–4.0× ⇒ 结构效应存活，但量级是 ~3×。
p31 的**构造性障碍**：oracle 在任何 n 下最多只能丢 ~11% 质量（注意力极度集中），
   于是三 mode 的共同质量带只到 0.23，而 argmax 翻转在该带内罕见
   ⇒ 决策层检验可能**欠功率**，且欠功率程度**取决于上下文的注意力集中度**。

本探针同时判两件事：
  A. **集中度是否依赖上下文**：若换上下文后 oracle 能丢更多质量，则可匹配带变宽 ⇒ 决策层检验可行；
     若所有上下文都锁死在 ~0.1，则"按质量匹配的决策层检验"在本平台上**构造上不可行**。
  B. **匹配质量下的翻转率**（含 Wilson 95% CI，N=contexts 数）：三种结构的翻转率是否可区分。

上下文三类（均为零成本、无需语料下载）：
  synth    —— 与 p28/p31 相同的重复合成文本（已知极度集中）
  natural  —— 本项目自己的仓内文本（真实代码/散文的 token 分布）
  random   —— 随机 token id（注意力最平坦的对照）
"""
import argparse, json, math, os, random, sys, torch, torch.nn.functional as F
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
        return (0.0, 0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (ph, max(0.0, c - hw), min(1.0, c + hw))

NATURAL_FILES = ["notes/FILTER_3AXIS.md", "notes/OCCUPANCY_LEDGER.md", "probes/p31-matched-mass/matched_mass_probe.py"]

def build_contexts(tok, L, root):
    ctxs = []
    synth = ("The history of computing began with mechanical calculators and evolved through "
             "vacuum tubes, transistors, integrated circuits, and parallel processors. ") * 60
    for s in range(4):
        ids = tok(synth, return_tensors="pt", truncation=True, max_length=L)["input_ids"][0]
        ctxs.append(("synth", s, ids))
    for s, fn in enumerate(NATURAL_FILES):
        p = os.path.join(root, fn)
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8", errors="ignore").read()
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=L)["input_ids"][0]
        if ids.numel() >= L // 2:
            ctxs.append(("natural", s, ids))
    rng = random.Random(1234)
    V = tok.vocab_size
    for s in range(4):
        ids = torch.tensor([rng.randrange(1000, V) for _ in range(L)], dtype=torch.long)
        ctxs.append(("random", s, ids))
    return ctxs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/user/models/Qwen3-4B")
    ap.add_argument("--root", default="/home/user/myCCFA")
    ap.add_argument("--ctx-tokens", type=int, default=1024)
    ap.add_argument("--ns", default="16,32,64,96,128,192,256,320,384,448,512,640,768,896,960,990,1010")
    ap.add_argument("--modes", default="topn,randn,window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/user/ccfa_logs/p32/decision.json")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                attn_implementation="eager").to("cuda").eval()
    nsl = [int(x) for x in a.ns.split(",")]
    modes = a.modes.split(",")
    ctxs = build_contexts(tok, a.ctx_tokens, a.root)
    print("contexts:", [(c[0], c[1], int(c[2].numel())) for c in ctxs], flush=True)

    allrows = []
    for ctype, cseed, ids in ctxs:
        ids = ids.unsqueeze(0).to("cuda")
        L = ids.shape[1]
        STATE.update(n=None, mode="off", dropped=[])
        with torch.no_grad():
            base = model(ids).logits[0, -1].float()
        p = F.softmax(base, dim=-1)
        b1 = int(base.argmax())
        for mode in modes:
            for n in nsl:
                if n >= L:
                    continue
                STATE.update(n=n, mode=mode, dropped=[])
                with torch.no_grad():
                    lg = model(ids).logits[0, -1].float()
                q = F.softmax(lg, dim=-1)
                dm = sum(STATE["dropped"]) / max(1, len(STATE["dropped"]))
                t1 = int(lg.argmax())
                v = lg.topk(2).values
                allrows.append({"ctx_type": ctype, "ctx_seed": cseed, "L": L, "mode": mode, "n": n,
                                "dropped_mass": dm, "tv": tv(p, q), "flip": int(t1 != b1),
                                "gap": float(v[0] - v[1]),
                                "gap_base": float(base.topk(2).values[0] - base.topk(2).values[1])})
        print("  done ctx %s/%d L=%d" % (ctype, cseed, L), flush=True)

    # ---- A. 集中度是否依赖上下文：每个 (ctx,mode) 能达到的最大丢弃质量 ----
    print("\n== A. 各上下文下 oracle 能达到的最大丢弃质量（>0.25 才说明可匹配带够宽）==", flush=True)
    for ctype in ("synth", "natural", "random"):
        for mode in modes:
            mx = [r["dropped_mass"] for r in allrows if r["ctx_type"] == ctype and r["mode"] == mode]
            if mx:
                print("  %-8s %-7s max_dropped=%.4f" % (ctype, mode, max(mx)), flush=True)

    # ---- B. 按质量匹配的翻转率 ----
    print("\n== B. 匹配质量下的翻转率（就近取 n，质量相对误差 <=1.35x；Wilson 95% CI）==", flush=True)
    targets = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60]
    summ = {}
    for M_ in targets:
        line = "  mass~%.2f : " % M_
        for mode in modes:
            ks, ns_ = 0, 0
            tvs = []
            for ctype in ("synth", "natural", "random"):
                for cseed in set(r["ctx_seed"] for r in allrows if r["ctx_type"] == ctype):
                    cand = [r for r in allrows if r["ctx_type"] == ctype and r["ctx_seed"] == cseed
                            and r["mode"] == mode and r["dropped_mass"] > 0]
                    if not cand:
                        continue
                    best = min(cand, key=lambda r: abs(math.log(r["dropped_mass"] / M_)))
                    if abs(math.log(best["dropped_mass"] / M_)) <= math.log(1.35):
                        ks += best["flip"]; ns_ += 1; tvs.append(best["tv"])
                pass
            ph, lo, hi = wilson(ks, ns_)
            summ.setdefault(mode, []).append({"mass": M_, "k": ks, "n": ns_, "rate": ph, "ci": [lo, hi],
                                              "mean_tv": (sum(tvs) / len(tvs)) if tvs else None})
            line += "%-7s %d/%-2d=%.2f [%.2f,%.2f] tv=%.3f   " % (
                mode, ks, ns_, ph, lo, hi, (sum(tvs) / len(tvs)) if tvs else float("nan"))
        print(line, flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"rows": allrows, "summary": summ}, open(a.out, "w"), indent=1)
    print("\nDECISION_DONE ->", a.out)

if __name__ == "__main__":
    sys.exit(main())
