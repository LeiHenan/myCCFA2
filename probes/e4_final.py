#!/usr/bin/env python
"""E4 (final design): mean-vs-tail under acceleration, with an in-process noise floor.

Lessons from the failed attempts (all my own bugs, kept here as a record):
  - 3 engines at 0.28 GPU_UTIL: the 3rd found 0 KV blocks (too little per-engine).
  - 3 engines at 0.42: total > GPU.
  - Deleting the 2nd engine did NOT return its memory (`empty_cache()` races with the
    engine's own shutdown), so building a 3rd in the same process failed.
  => one process must not build more than TWO engines.

Design used here (one process, exactly two engines):
    engine A  baseline
    engine B  ngram-spec
    Interleave per prompt, alternating order, so drift hits both equally.
    Then re-run the SAME prompts through engine A a second time -> A vs A' is the
    in-process run-to-run noise floor for the very statistics we compare.

    Optionally (--control-only) run a SECOND process that builds two IDENTICAL
    baseline engines to get a cross-engine noise floor.

Reported: mean/p50/p95/p99 latency per arm, the B/A ratio of each statistic, the
per-request ratio distribution, and whether the tail/mean divergence exceeds the
noise floor.
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
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def stats(xs):
    return {"n": len(xs), "mean_ms": st.mean(xs) * 1000, "p50_ms": pct(xs, .50) * 1000,
            "p95_ms": pct(xs, .95) * 1000, "p99_ms": pct(xs, .99) * 1000,
            "cv": (st.pstdev(xs) / st.mean(xs)) if st.mean(xs) else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--prompt", type=int, default=384)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--control-only", action="store_true",
                    help="run two IDENTICAL baseline engines (cross-engine noise floor)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=7919,
                    help="prompt-tail seed; vary to make independent replicates")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    def build(spec: bool):
        kw = dict(model=a.model, gpu_memory_utilization=a.gpu_util,
                  max_model_len=a.max_len, enable_prefix_caching=False,
                  enforce_eager=True, dtype="bfloat16")
        if spec:
            kw["speculative_config"] = {"method": "ngram",
                                        "num_speculative_tokens": 5,
                                        "prompt_lookup_max": 4, "prompt_lookup_min": 2}
        return LLM(**kw)

    base_ids = [1000 + ((i * 37 + a.seed) % 20000) for i in range(a.prompt)]
    inputs = []
    for k in range(a.n_prompts):
        p = list(base_ids)
        p[-12:] = [20000 + (k * a.seed + j * 13) % 40000 for j in range(12)]
        inputs.append({"prompt_token_ids": p})
    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True,
                        seed=a.seed)

    def timed(eng, one):
        t0 = time.perf_counter()
        eng.generate([one], sampling_params=sp, use_tqdm=False)
        return time.perf_counter() - t0

    def interleaved(e1, e2, label):
        x, y = [], []
        for i, one in enumerate(inputs):
            if i % 2 == 0:
                x.append(timed(e1, one)); y.append(timed(e2, one))
            else:
                y.append(timed(e2, one)); x.append(timed(e1, one))
            if (i + 1) % 50 == 0:
                print(f"  [{label}] {i+1}/{len(inputs)}", flush=True)
        return x, y

    print(f"tag={a.tag} control_only={a.control_only} prompts={a.n_prompts} "
          f"prompt_tok={a.prompt} gen={a.gen} gpu_util={a.gpu_util}", flush=True)

    print("building engine 1...", flush=True)
    e1 = build(False)
    print("building engine 2...", flush=True)
    e2 = build(False if a.control_only else True)
    for e in (e1, e2):
        e.generate(inputs[:3], sampling_params=sp, use_tqdm=False)

    print("interleaved measurement...", flush=True)
    X, Y = interleaved(e1, e2, "1v2")

    print("re-running the same prompts through engine 1 (in-process noise floor)...",
          flush=True)
    X2 = [timed(e1, one) for one in inputs]

    sX, sY, sX2 = stats(X), stats(Y), stats(X2)

    def ratio(p, q):
        return {k: (p[k + "_ms"] / q[k + "_ms"] if k != "cv" else p["cv"] / q["cv"])
                for k in ("mean", "p50", "p95", "p99", "cv")}

    r_main = ratio(sY, sX)        # engine2 / engine1
    r_noise = ratio(sX2, sX)      # same engine, second pass  -> noise floor

    per_req = [y / x for x, y in zip(X, Y)]
    pr = {"min": min(per_req), "p05": pct(per_req, .05), "p50": pct(per_req, .50),
          "p95": pct(per_req, .95), "max": max(per_req),
          "frac_slower": sum(1 for v in per_req if v > 1.0) / len(per_req)}

    print()
    hdr = f"{'metric':<10}{'eng1':>11}{'eng2':>11}{'ratio':>9}   {'noise':>8}"
    print(hdr); print("-" * len(hdr))
    for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
        print(f"{k:<10}{sX[k]:>11.1f}{sY[k]:>11.1f}{sY[k]/sX[k]:>9.3f}   "
              f"{sX2[k]/sX[k]:>8.3f}")
    print(f"{'cv':<10}{sX['cv']:>11.4f}{sY['cv']:>11.4f}{sY['cv']/sX['cv']:>9.3f}   "
          f"{sX2['cv']/sX['cv']:>8.3f}")

    if not a.control_only:
        print("\nper-request eng2/eng1 latency ratio (<1 = faster):")
        print(f"  min {pr['min']:.3f} p05 {pr['p05']:.3f} p50 {pr['p50']:.3f} "
              f"p95 {pr['p95']:.3f} max {pr['max']:.3f}  frac_slower={pr['frac_slower']:.3f}")

    tv = r_main["p95"] / r_main["mean"]
    nv = r_noise["p95"] / r_noise["mean"]
    v = {"tag": a.tag, "control_only": a.control_only,
         "ratios_main": r_main, "ratios_noise": r_noise,
         "tail_vs_mean": tv, "tail_vs_mean_noise": nv,
         "tail_effect_exceeds_noise": abs(tv - 1) > abs(nv - 1),
         "per_request": pr if not a.control_only else None}
    print("\n--- verdict ---")
    print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x)
                      for k, x in v.items() if k != "per_request"}, indent=2))
    if not a.control_only:
        print(json.dumps({"per_request": {k: round(x, 4) for k, x in pr.items()}},
                         indent=2))
    out = f"/root/autodl-tmp/e4_final_{a.tag}.json"
    with open(out, "w") as f:
        json.dump({"config": vars(a), "eng1": sX, "eng2": sY, "eng1_pass2": sX2,
                   "verdict": v}, f, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
