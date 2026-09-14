#!/usr/bin/env python
"""E4-BATCH: independent-seed replications of the mean-vs-tail divergence.

Why this exists: a single E4 run (n=40, both engines, two passes) takes ~5-6 min,
and the box kills persistent background processes, so one SSH command window only
fits one run. This driver chains MANY seeds in ONE process invocation so a single
long-lived job (the harness background job) can collect a real sample.

Per seed it runs probes/e4_final.py unchanged (same controlled design: interleaved
A/B, order-alternated, in-process noise floor, pinned output length) and then
aggregates the per-seed verdicts:

    for each seed:  mean_ratio, p95_ratio, p99_ratio, cv_ratio, tail_vs_mean
    aggregate:      median + range across seeds, and how many seeds satisfy
                    tail_vs_mean > 1.15 AND tail_effect_exceeds_noise

That aggregate is the robustness evidence the proposal's E2 stage demands.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time


def run_seed(seed: int, args) -> dict | None:
    tag = f"b{seed}"
    cmd = [sys.executable, os.path.join(args.workdir, "e4_final.py"),
           "--n-prompts", str(args.n_prompts),
           "--gpu-util", str(args.gpu_util),
           "--max-len", str(args.max_len),
           "--gen", str(args.gen),
           "--prompt", str(args.prompt),
           "--seed", str(seed),
           "--tag", tag]
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0",
               VLLM_ATTENTION_BACKEND="FLASH_ATTN")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=args.workdir, env=env,
                       capture_output=True, text=True, timeout=args.timeout)
    dt = time.time() - t0
    jf = os.path.join(args.workdir, f"e4_final_{tag}.json")
    if not os.path.exists(jf):
        tail = "\n".join((r.stderr or "").strip().splitlines()[-6:])
        print(f"[seed {seed}] FAILED in {dt:.0f}s rc={r.returncode}\n{tail}",
              flush=True)
        return None
    with open(jf) as f:
        d = json.load(f)
    v = d["verdict"]
    out = {"seed": seed, "seconds": round(dt, 1),
           "ratios_main": v["ratios_main"], "ratios_noise": v["ratios_noise"],
           "tail_vs_mean": v["tail_vs_mean"],
           "tail_vs_mean_noise": v["tail_vs_mean_noise"],
           "exceeds_noise": v["tail_effect_exceeds_noise"],
           "per_request": v.get("per_request")}
    print(f"[seed {seed}] {dt:.0f}s  mean={v['ratios_main']['mean']:.3f} "
          f"p95={v['ratios_main']['p95']:.3f} p99={v['ratios_main']['p99']:.3f} "
          f"tv m={v['tail_vs_mean']:.3f} noise={v['tail_vs_mean_noise']:.3f} "
          f"exceeds={v['tail_effect_exceeds_noise']}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/root/autodl-tmp")
    ap.add_argument("--seeds", default="7919,104729,224737,611953,1000003,2000003")
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="/root/autodl-tmp/e4_batch_summary.json")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    print(f"E4-BATCH: {len(seeds)} seeds {seeds}", flush=True)
    rows = []
    for s in seeds:
        try:
            r = run_seed(s, a)
        except subprocess.TimeoutExpired:
            print(f"[seed {s}] TIMEOUT after {a.timeout}s", flush=True)
            r = None
        if r:
            rows.append(r)

    if not rows:
        print("NO SUCCESSFUL SEEDS", flush=True)
        return 1

    def col(k, src="ratios_main"):
        return [r[src][k] for r in rows]

    agg = {
        "n_seeds": len(rows),
        "seeds": [r["seed"] for r in rows],
        "mean_ratio": {"median": st.median(col("mean")), "min": min(col("mean")),
                       "max": max(col("mean"))},
        "p95_ratio": {"median": st.median(col("p95")), "min": min(col("p95")),
                      "max": max(col("p95"))},
        "p99_ratio": {"median": st.median(col("p99")), "min": min(col("p99")),
                      "max": max(col("p99"))},
        "tail_vs_mean": {"median": st.median([r["tail_vs_mean"] for r in rows]),
                         "min": min(r["tail_vs_mean"] for r in rows),
                         "max": max(r["tail_vs_mean"] for r in rows)},
        "noise_tail_vs_mean": {"median": st.median([r["tail_vs_mean_noise"] for r in rows]),
                               "max": max(r["tail_vs_mean_noise"] for r in rows)},
        "frac_exceeds_noise": sum(1 for r in rows if r["exceeds_noise"]) / len(rows),
        "frac_tail_vs_mean_gt_1.15": sum(
            1 for r in rows if r["tail_vs_mean"] > 1.15) / len(rows),
        "frac_slower_median": st.median(
            [r["per_request"]["frac_slower"] for r in rows
             if r.get("per_request")] or [float("nan")]),
        "per_seed": rows,
    }
    print("\n=== AGGREGATE ===")
    print(json.dumps({k: v for k, v in agg.items() if k != "per_seed"},
                     indent=2, default=str))
    with open(a.out, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"wrote {a.out}")

    crit = (agg["frac_exceeds_noise"] >= 0.8
            and agg["tail_vs_mean"]["median"] > 1.15)
    print("\n=== VERDICT (proposal E2 gate) ===")
    print(f"  median tail_vs_mean = {agg['tail_vs_mean']['median']:.3f} (need > 1.15)")
    print(f"  frac exceeding noise = {agg['frac_exceeds_noise']:.2f} (need >= 0.80)")
    print("  => " + ("PASS: the effect replicates across seeds."
                     if crit else "FAIL: effect does not replicate robustly."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
