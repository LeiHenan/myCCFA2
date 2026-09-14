#!/usr/bin/env python
"""Direction B feasibility: does an acceleration technique change the DISTRIBUTION
of per-request latency (not just its mean)?

Motivation. Every acceleration paper reports a MEAN speedup (e.g. "2.3x faster").
SLOs, however, are contracted on PERCENTILES. A technique that improves the mean
while widening the spread makes p99 worse and can break an SLO even as the
benchmark number improves. Nobody reports the per-request speedup distribution.

Here we use prompt-lookup / n-gram spec decoding because it needs NO draft model,
so it runs on the box as-is. Output length is pinned with ignore_eos, so decode
cost is comparable across arms.

Measured per arm: mean, p50, p95, p99 latency, and the per-request speedup
distribution when the two arms are paired on identical prompts.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

import torch


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--gpu-util", type=float, default=0.45)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=32)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    def build(spec: bool):
        kw = dict(model=a.model, gpu_memory_utilization=a.gpu_util,
                  max_model_len=a.max_len, enable_prefix_caching=False,
                  enforce_eager=True, dtype="bfloat16")
        if spec:
            kw["speculative_config"] = {
                "method": "ngram",
                "num_speculative_tokens": 5,
                "prompt_lookup_max": 4,
                "prompt_lookup_min": 2,
            }
        return LLM(**kw)

    prompts = []
    base = [1000 + (i * 37) % 20000 for i in range(a.prompt)]
    for k in range(a.n_prompts):
        p = list(base)
        p[-8:] = [(20000 + k * 11 + j) % 30000 for j in range(8)]
        prompts.append(p)

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)
    inputs = [{"prompt_token_ids": p} for p in prompts]

    def run(llm):
        # warm up shapes, then measure per-prompt
        llm.generate(inputs[:2], sampling_params=sp, use_tqdm=False)
        lat = []
        for one in inputs:
            t0 = time.perf_counter()
            llm.generate([one], sampling_params=sp, use_tqdm=False)
            lat.append(time.perf_counter() - t0)
        return lat

    print(f"model={a.model} prompts={a.n_prompts} prompt_tok={a.prompt} gen={a.gen}")
    print("building baseline engine...")
    l0 = build(False)
    lat0 = run(l0)
    del l0
    torch.cuda.empty_cache()

    print("building ngram-spec engine...")
    l1 = build(True)
    lat1 = run(l1)

    def summary(name, xs):
        return {"arm": name, "n": len(xs),
                "mean_ms": round(st.mean(xs) * 1000, 1),
                "p50_ms": round(pct(xs, .50) * 1000, 1),
                "p95_ms": round(pct(xs, .95) * 1000, 1),
                "p99_ms": round(pct(xs, .99) * 1000, 1),
                "cv": round(st.pstdev(xs) / st.mean(xs), 4) if st.mean(xs) else None}

    s0, s1 = summary("baseline", lat0), summary("ngram_spec", lat1)
    print()
    hdr = f"{'arm':<14}{'mean_ms':>10}{'p50_ms':>10}{'p95_ms':>10}{'p99_ms':>10}{'CV':>9}"
    print(hdr); print("-" * len(hdr))
    for s in (s0, s1):
        print(f"{s['arm']:<14}{s['mean_ms']:>10}{s['p50_ms']:>10}"
              f"{s['p95_ms']:>10}{s['p99_ms']:>10}{s['cv']:>9}")

    ratio_mean = s1["mean_ms"] / s0["mean_ms"]
    ratio_p99 = s1["p99_ms"] / s0["p99_ms"]
    per_req = [b / s for b, s in zip(lat0, lat1)]
    v = {
        "mean_speedup": round(1 / ratio_mean, 3),
        "p99_speedup": round(1 / ratio_p99, 3),
        "mean_vs_p99_gap": round(ratio_p99 / ratio_mean, 3),
        "per_request_speedup": {
            "min": round(min(per_req), 3), "p50": round(pct(per_req, .5), 3),
            "max": round(max(per_req), 3),
            "spread": round(max(per_req) / min(per_req), 2),
        },
        "cv_baseline": s0["cv"], "cv_spec": s1["cv"],
    }
    print("\n--- verdict ---")
    print(json.dumps(v, indent=2))
    if abs(v["mean_vs_p99_gap"] - 1.0) > 0.15:
        print("\n*** MEAN AND TAIL DISAGREE (>15%): a mean-only speedup number would")
        print("    misrepresent the SLO-relevant behaviour. Direction B looks real. ***")
    else:
        print("\nmean and tail agree within 15% here: weaker support for direction B")
        print("(widen the workload mix / raise concurrency before concluding).")
    with open("/root/autodl-tmp/e4_tail_result.json", "w") as f:
        json.dump({"config": vars(a), "baseline": s0, "spec": s1, "verdict": v}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
