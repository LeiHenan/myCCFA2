#!/usr/bin/env python
"""C-1j: locate the "third cost" -- decompose per-step cost into FIXED vs PER-DRAFT.

C-1i left one unresolved item and it sits directly on my hardest claim:
    hybrid: A_actual/A* = 2.35, i.e. the acceptance model PREDICTS A WIN,
            yet measured speedup is 0.794x (a loss).
So on hybrid there is a cost component that is neither acceptance nor the
average per-step multiplier. This probe tries to find its SHAPE by fitting the
per-step cost as a function of the draft length K:

    t_step(K) = t_fixed + K * c_verify        (+ draft cost, see below)

How to separate the two:
  * sweep K and record, from the engine's own per-request accumulator,
        steps(K)          -- number of verify steps
        A(K)              -- mean accepted length per step
        draft_tokens(K)   -- proposed drafts
    plus wall time, so
        t_step(K) = wall_time(K) / steps(K)
  * fit t_step vs K:
        slope   ~ cost per extra draft token  (verification/attention growth)
        intercept ~ fixed per-step cost (scheduler, metadata, graph, state)
  * compare the fit on dense vs hybrid. If hybrid's SLOPE is much larger, the
    cost is in verifying more tokens (attention over draft positions). If
    hybrid's INTERCEPT is much larger, it is fixed per-step overhead.

That distinction is exactly what decides whether the finding is about
linear-attention kernels (slope) or about the hybrid step machinery (intercept).

Gate: whichever term dominates on hybrid, report its magnitude and check it is
stable across the two family members we can run (Qwen3.5; Falcon is blocked by
the upstream #47635 crash).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

SYSTEM = (
    "You are a meticulous software engineering assistant embedded in a large "
    "codebase. Always answer concisely and cite the relevant module. Follow the "
    "team conventions: prefer explicit names, avoid global state, and never "
    "silently swallow errors. "
)


def build_prompts(tok, n, prefix):
    filler = tok.encode(
        "The inventory service exposes a small HTTP API and a set of pure "
        "helpers, described below in detail for onboarding purposes. ",
        add_special_tokens=False)
    out = []
    for c in range(n):
        ids = tok.encode(SYSTEM, add_special_tokens=False)
        while len(ids) < prefix:
            ids = ids + filler
        ids = ids[:prefix]
        out.append(ids + tok.encode(f"\nUser: t{c}\nAssistant:",
                                    add_special_tokens=False))
    return out


def metrics_from(outs):
    acc = steps = den = 0
    seen = 0
    for o in outs:
        for so in o.outputs:
            m = getattr(so, "spec_decode_metrics", None)
            if m is None:
                continue
            seen += 1
            h = list(getattr(m, "histogram", []) or [])
            acc += sum(j * c for j, c in enumerate(h))
            steps += sum(h)
            den += int(getattr(m, "num_draft_tokens", 0) or 0)
    if not seen or not steps:
        return {}
    return {"n": seen, "accepted": acc, "steps": steps, "draft_tokens": den,
            "A": 1 + acc / steps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--hybrid", default="/root/autodl-tmp/models/Qwen3.5-4B")
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--ks", default="1,2,3,4,6,8")
    ap.add_argument("--targets", default="dense,hybrid")
    ap.add_argument("--out", default="/root/autodl-tmp/c1j_decompose.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)
    rows = {}

    for target in [t.strip() for t in a.targets.split(",") if t.strip()]:
        path = a.dense if target == "dense" else a.hybrid
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        prompts = [{"prompt_token_ids": p}
                   for p in build_prompts(tok, a.n, a.prefix)]

        for K in [int(x) for x in a.ks.split(",") if x.strip()]:
            kw = dict(model=path, gpu_memory_utilization=a.gpu_util,
                      max_model_len=a.max_len, enable_prefix_caching=True,
                      enforce_eager=True, dtype="bfloat16",
                      language_model_only=True,
                      per_request_spec_decode_metrics="detailed",
                      speculative_config={
                          "method": "ngram", "num_speculative_tokens": K,
                          "prompt_lookup_max": 6, "prompt_lookup_min": 2})
            try:
                llm = LLM(**kw)
            except Exception as e:  # noqa: BLE001
                print(f"[{target} K={K}] ENGINE FAILED {type(e).__name__}",
                      flush=True)
                rows[f"{target}|K={K}"] = {"error": str(e)[:150]}
                continue

            walls, mets = [], []
            for _ in range(a.passes):
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
                walls.append(time.perf_counter() - t0)
                mets.append(metrics_from(outs))
            wall = st.median(walls)
            merged = {}
            for m in mets:
                for kk, vv in m.items():
                    merged.setdefault(kk, []).append(vv)
            merged = {kk: st.median(vv) for kk, vv in merged.items()}
            steps = merged.get("steps")
            row = {"K": K, "wall_s": round(wall, 4),
                   "A": round(merged.get("A", float("nan")), 4) if merged else None,
                   "steps": steps,
                   "draft_tokens": merged.get("draft_tokens"),
                   "t_step_ms": round(wall / steps * 1000, 4) if steps else None,
                   "tokens_per_s": round(
                       sum(a.gen for _ in prompts) / wall, 1) if wall else None}
            rows[f"{target}|K={K}"] = row
            print(f"[{target} K={K}] wall={row['wall_s']:.3f}s steps={steps} "
                  f"A={row['A']} t_step={row['t_step_ms']}ms", flush=True)
            del llm
            import torch
            torch.cuda.empty_cache(); time.sleep(2)

    print("\n=== per-step cost vs K (t_step = wall/steps) ===")
    print(f"{'cell':<14}{'A':>7}{'steps':>7}{'t_step_ms':>11}{'drafts':>8}")
    fits = {}
    for target in [t.strip() for t in a.targets.split(",") if t.strip()]:
        pts = []
        for K in [int(x) for x in a.ks.split(",") if x.strip()]:
            r = rows.get(f"{target}|K={K}") or {}
            if r.get("t_step_ms") and r.get("A"):
                print(f"{target + '|K=' + str(K):<14}{r['A']:>7.2f}{r['steps']:>7}"
                      f"{r['t_step_ms']:>11.3f}{(r['draft_tokens'] or 0):>8}")
                pts.append((K, r["t_step_ms"]))
        if len(pts) >= 3:
            # ordinary least squares on K
            n = len(pts)
            sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
            sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
            den = n * sxx - sx * sx
            slope = (n * sxy - sx * sy) / den if den else float("nan")
            inter = (sy - slope * sx) / n if n else float("nan")
            fits[target] = {"intercept_ms": round(inter, 4),
                            "slope_ms_per_draft": round(slope, 4),
                            "n_points": n}
            print(f"  {target}: t_step ≈ {inter:.3f} + {slope:.3f}*K  (ms)")

    print("\n=== interpretation ===")
    if len(fits) >= 2:
        d, h = fits.get("dense"), fits.get("hybrid")
        if d and h:
            print(f"  intercept: dense {d['intercept_ms']} vs hybrid {h['intercept_ms']} "
                  f"-> x{h['intercept_ms'] / d['intercept_ms']:.2f}"
                  if d['intercept_ms'] else "")
            print(f"  slope    : dense {d['slope_ms_per_draft']} vs "
                  f"hybrid {h['slope_ms_per_draft']} "
                  f"-> x{h['slope_ms_per_draft'] / d['slope_ms_per_draft']:.2f}"
                  if d['slope_ms_per_draft'] else "")
            print("  (larger ratio tells you whether hybrid's extra cost is FIXED "
                  "per-step or GROWS with draft length)")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "rows": rows, "fits": fits}, f,
                  indent=2, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
