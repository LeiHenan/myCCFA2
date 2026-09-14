#!/usr/bin/env python3
"""p28 —— verify 侧 KV 稀疏「能不能用一个运行期可校验的量预测输出漂移」

核心可证伪问题：`d_TV(p, p~)` 是「丢弃质量」的可用函数，还是取决于「丢了哪部分质量」？
做法：在 HF 的 eager attention 上注入 top-n / random-n 保留，比较**最终 next-token 分布**与精确值。
  · 若 TV 只由丢弃质量决定 ⇒ 运行期证书可行（用 dropped mass 当判据）；
  · 若同一 n 下 oracle 与 random 的 TV 差很多（或 TV/dropped_mass 跨 n 变化数量级）⇒ 证书不可行。
"""
import argparse, json, sys, torch, torch.nn.functional as F
import transformers.models.qwen3.modeling_qwen3 as M
from transformers import AutoModelForCausalLM, AutoTokenizer

STATE = {"n": None, "mode": "off", "dropped": [], "kept": []}
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
        else:  # randn：在**允许的位置**（权重>0）里随机取 n 个
            allowed = w > 0
            rnd = torch.rand_like(w).masked_fill(~allowed, -1.0)
            keep_i = rnd.topk(n, dim=-1).indices
            keep_v = w.gather(-1, keep_i)
        STATE["dropped"].append(float((1.0 - keep_v.sum(-1)).mean()))
        STATE["kept"].append(int(n))
        sp = torch.zeros_like(w).scatter_(-1, keep_i, keep_v)
        w = sp / sp.sum(-1, keepdim=True).clamp_min(1e-12)
    out = torch.matmul(w, value_states).transpose(1, 2).contiguous()
    return out, w

M.eager_attention_forward = _patched
M.ALL_ATTENTION_FUNCTIONS["eager"] = _patched

def tv(p, q):
    return float(0.5 * (p - q).abs().sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/user/models/Qwen3-4B")
    ap.add_argument("--ns", default="1024,512,256,128,64,32,16")
    ap.add_argument("--mode", default="topn", choices=["topn", "randn", "off"])
    ap.add_argument("--ctx-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                attn_implementation="eager").to("cuda").eval()
    text = ("The history of computing began with mechanical calculators and evolved through "
            "vacuum tubes, transistors, integrated circuits, and parallel processors. ") * 40
    ids = tok(text, return_tensors="pt", truncation=True, max_length=a.ctx_tokens)["input_ids"].to("cuda")
    L = ids.shape[1]
    print("ctx tokens:", L, "| mode:", a.mode, flush=True)

    STATE.update(n=None, mode="off", dropped=[], kept=[])
    with torch.no_grad():
        base = model(ids).logits[0, -1].float()
    p = F.softmax(base, dim=-1)

    rows = []
    for n in [int(x) for x in a.ns.split(",")]:
        if n >= L:
            continue
        STATE.update(n=n, mode=a.mode, dropped=[], kept=[])
        with torch.no_grad():
            lg = model(ids).logits[0, -1].float()
        q = F.softmax(lg, dim=-1)
        dm = sum(STATE["dropped"]) / max(1, len(STATE["dropped"]))
        t = tv(p, q)
        rows.append({"n": n, "dropped_mass": dm, "tv": t,
                     "tv_per_dropped": (t / dm) if dm > 1e-9 else None,
                     "argmax_same": int(base.argmax()) == int(lg.argmax())})
        print("  n=%-5d dropped=%.6f  TV=%.6f  TV/dropped=%s  argmax_same=%s"
              % (n, dm, t, ("%.3f" % (t / dm)) if dm > 1e-9 else "n/a", rows[-1]["argmax_same"]), flush=True)
    json.dump({"mode": a.mode, "ctx": L, "rows": rows}, open("/tmp/drift_%s.json" % a.mode, "w"), indent=1)
    if rows:
        ratios = [r["tv_per_dropped"] for r in rows if r["tv_per_dropped"]]
        if ratios:
            print("  TV/dropped 的跨度: min=%.3f max=%.3f  (max/min=%.1fx)"
                  % (min(ratios), max(ratios), max(ratios)/max(min(ratios), 1e-9)))
    print("DRIFT_DONE")

if __name__ == "__main__":
    sys.exit(main())
