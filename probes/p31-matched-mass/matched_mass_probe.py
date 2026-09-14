#!/usr/bin/env python3
"""p31 —— **按「丢弃质量」匹配**重测 p28 的结构性结论（对我自己头条数字的对照复核）

**为什么要做这件事（S3b 提出的、我认为成立的质疑）**：
p28 的头条是"同一 n 下 oracle 与 random 的 TV 差 3–31×"。但那是**按 n 匹配，不是按丢弃质量匹配**：
n=256 时 oracle 丢 0.0258、random 丢 0.4097 —— **质量差 16 倍**。
按质量横看，p28 自己的数据已经显示两mode**区间重叠**：
  oracle dropped 0.0880 -> TV 0.0880 (TV/dropped = 1.00)
  randn  dropped 0.4097 -> TV 0.8060 (TV/dropped = 1.97)
  randn  dropped 0.6204 -> TV 0.1926 (TV/dropped = 0.31)
⇒ **"440× vs 1323× 的跨度"可能主要是质量效应，不是结构效应。** 本探针直接判它。

**同时记录已发表论文认定的真判据**（S3b 查到的两篇独立工作都把 flip 归因于 margin 而非累积量级）：
最终层 top-1 与 top-2 的 logit 间距 `gap`，以及扰动后 `gap` 的变化 `d_gap`。
若结构的效应只体现在 TV 上、而不体现在 `d_gap` 的分布上 ⇒ **结构对"决策"没有额外信息**，本切片也死。

判据（开跑前冻结）：
  · **存活**：在质量匹配的重叠区内，oracle 与 randn 的 TV 比 > 3×，**且** `|d_gap|` 分布显著不同。
  · **死亡**：质量匹配后 TV 比 < 1.5×，**或** TV 比 > 3× 但 `d_gap` 分布重合（=只有软指标差异、无决策差异）。
"""
import argparse, json, math, sys, torch, torch.nn.functional as F
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
        if mode == "topn":                      # oracle：保留权重最大的 n 个
            keep_v, keep_i = w.topk(n, dim=-1)
        elif mode == "randn":                   # random：在允许位置里随机取 n 个
            allowed = w > 0
            rnd = torch.rand_like(w).masked_fill(~allowed, -1.0)
            keep_i = rnd.topk(n, dim=-1).indices
            keep_v = w.gather(-1, keep_i)
        elif mode == "window":                  # recency（StreamingLLM 式）：保留最近 n 个
            keep_i = torch.arange(w.shape[-1] - n, w.shape[-1], device=w.device)
            keep_i = keep_i.view(1, 1, 1, n).expand(*w.shape[:-1], n)
            keep_v = w.gather(-1, keep_i)
        else:
            raise SystemExit("bad mode " + mode)
        STATE["dropped"].append(float((1.0 - keep_v.sum(-1)).mean()))
        sp = torch.zeros_like(w).scatter_(-1, keep_i, keep_v)
        w = sp / sp.sum(-1, keepdim=True).clamp_min(1e-12)
    out = torch.matmul(w, value_states).transpose(1, 2).contiguous()
    return out, w

M.eager_attention_forward = _patched
M.ALL_ATTENTION_FUNCTIONS["eager"] = _patched

def tv(p, q):
    return float(0.5 * (p - q).abs().sum())

def top2_stats(lg):
    v, i = lg.topk(2)
    return int(i[0]), int(i[1]), float(v[0]), float(v[0] - v[1])

def interp_tv(rows, mass):
    """在 log10(mass) 上线性插值 TV；返回 None 若质量出界"""
    pts = sorted([(r["dropped_mass"], r["tv"]) for r in rows if r["dropped_mass"] > 0])
    if len(pts) < 2 or not (pts[0][0] <= mass <= pts[-1][0]):
        return None
    for (m0, t0), (m1, t1) in zip(pts, pts[1:]):
        if m0 <= mass <= m1:
            if m0 == m1:
                return t0
            f = (math.log10(mass) - math.log10(m0)) / (math.log10(m1) - math.log10(m0))
            return t0 + f * (t1 - t0)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/user/models/Qwen3-4B")
    ap.add_argument("--ns", default="16,32,48,64,96,128,192,256,384,512,704,896,940,960,980,990,"
                                   "1000,1005,1010,1015,1020,1025,1030,1035")
    ap.add_argument("--modes", default="topn,randn,window")
    ap.add_argument("--ctx-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/user/ccfa_logs/p31/matched_mass.json")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                attn_implementation="eager").to("cuda").eval()
    text = ("The history of computing began with mechanical calculators and evolved through "
            "vacuum tubes, transistors, integrated circuits, and parallel processors. ") * 40
    ids = tok(text, return_tensors="pt", truncation=True, max_length=a.ctx_tokens)["input_ids"].to("cuda")
    L = ids.shape[1]
    nsl = [int(x) for x in a.ns.split(",")]
    modes = a.modes.split(",")
    print("ctx tokens:", L, "| modes:", modes, flush=True)

    STATE.update(n=None, mode="off", dropped=[])
    with torch.no_grad():
        base = model(ids).logits[0, -1].float()
    p = F.softmax(base, dim=-1)
    b1, b2, bv1, bgap = top2_stats(base)
    print("BASE top1=%d top2=%d gap=%.4f" % (b1, b2, bgap), flush=True)

    # 决定性自检：mode=off 必须逐位复现（p28 已证本平台前向是确定的）
    with torch.no_grad():
        chk = model(ids).logits[0, -1].float()
    print("DETERMINISM mode=off TV=%.9f (必须为 0)" % tv(p, F.softmax(chk, dim=-1)), flush=True)

    out = {"ctx": L, "base": {"top1": b1, "top2": b2, "gap": bgap}, "modes": {}}
    for mode in modes:
        rows = []
        for n in nsl:
            if n >= L:
                continue
            STATE.update(n=n, mode=mode, dropped=[])
            with torch.no_grad():
                lg = model(ids).logits[0, -1].float()
            q = F.softmax(lg, dim=-1)
            dm = sum(STATE["dropped"]) / max(1, len(STATE["dropped"]))
            t = tv(p, q)
            t1, t2, v1, gap = top2_stats(lg)
            rows.append({"n": n, "dropped_mass": dm, "tv": t,
                         "tv_per_dropped": (t / dm) if dm > 1e-9 else None,
                         "top1": t1, "top2": t2, "gap": gap, "d_gap": gap - bgap,
                         "argmax_same": t1 == b1})
            print("  [%-6s] n=%-5d dropped=%.6f TV=%.6f TV/dropped=%-8s top1=%-6d gap=%8.4f d_gap=%9.4f same=%s"
                  % (mode, n, dm, t, ("%.3f" % (t / dm)) if dm > 1e-9 else "n/a", t1, gap, gap - bgap,
                     rows[-1]["argmax_same"]), flush=True)
        out["modes"][mode] = rows

    # ---- 质量匹配比较 ----
    print("\n== 按丢弃质量匹配的 TV 比较（log10(mass) 线性插值）==", flush=True)
    allm = []
    for m, rows in out["modes"].items():
        ms = [r["dropped_mass"] for r in rows if r["dropped_mass"] > 0]
        if ms:
            allm.append((max(min(ms), 1e-7), max(ms)))
    lo = max(x[0] for x in allm)
    hi = min(x[1] for x in allm)
    print("三mode 共同质量区间: [%.6g, %.6g]" % (lo, hi), flush=True)
    matched = []
    if hi > lo:
        for k in range(11):
            M_ = 10 ** (math.log10(lo) + (math.log10(hi) - math.log10(lo)) * k / 10)
            rec = {"mass": M_}
            for m, rows in out["modes"].items():
                rec[m] = interp_tv(rows, M_)
            matched.append(rec)
            tvv = {m: rec[m] for m in out["modes"]}
            if tvv.get("topn") and tvv.get("randn"):
                print("  mass=%.6g  " % M_ + "  ".join("%s TV=%s" % (m, ("%.6f" % v) if v is not None else "n/a")
                                                       for m, v in tvv.items())
                      + "   oracle/random=%.2fx" % (tvv["topn"] / max(tvv["randn"], 1e-12)), flush=True)
    out["matched_mass"] = matched

    # ---- 决策判据：d_gap 在质量匹配点上的比较 ----
    print("\n== 决策判据 d_gap（扰动后 top1-top2 间距变化）在质量匹配点上的比较 ==", flush=True)
    for mode, rows in out["modes"].items():
        near = sorted(rows, key=lambda r: abs(math.log10(max(r["dropped_mass"], 1e-9)) - math.log10(lo)))[:3]
        print("  [%s] 最靠近质量下界的 3 点: " % mode +
              " | ".join("m=%.4g TV=%.4f d_gap=%+.4f same=%s" % (r["dropped_mass"], r["tv"], r["d_gap"],
                                                                  r["argmax_same"]) for r in near), flush=True)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nMATCHED_DONE ->", a.out)

if __name__ == "__main__":
    sys.exit(main())
