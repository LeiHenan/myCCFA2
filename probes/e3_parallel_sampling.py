#!/usr/bin/env python
"""Does parallel sampling (n>1) waste prefill, and can KV be shared across samples?

Why ask: every engine's headline efficiency claim rests on batching *distinct*
requests. Parallel sampling creates n sequences that share a byte-identical
prompt -- the best possible case for prefix reuse -- yet engines typically treat
them as one request with n children, so whether the prompt KV is computed once or
n times is an implementation detail nobody has measured publicly.

Design (offline vLLM API, deterministic output length via ignore_eos):
  A) N separate requests, n=1        -> N independent prefills (baseline)
  B) ONE request with n=N            -> does the engine share one prefill?
  C) A repeated (cache warm)         -> prefix-cache effect

Metric: total wall time, plus prefill-proxy = tokens computed / request.
For a fair comparison output length is pinned, so decode cost is identical across
arms; any difference is prefill.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--gpu-util", type=float, default=0.45)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_len, enable_prefix_caching=True,
              enforce_eager=True, dtype="bfloat16")

    ids = [(1000 + (i * 37) % 20000) for i in range(a.prompt)]
    N = a.n
    # pin output length so decode cost is identical across arms
    sp1 = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.gen,
                         n=1, ignore_eos=True, seed=0)
    spN = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.gen,
                         n=N, ignore_eos=True, seed=0)

    def timeit(fn):
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            out = fn()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts), out

    def arm_A():
        return llm.generate([{"prompt_token_ids": list(ids)}] * N,
                            sampling_params=sp1, use_tqdm=False)

    def arm_B():
        return llm.generate({"prompt_token_ids": list(ids)},
                            sampling_params=spN, use_tqdm=False)

    def arm_C():
        return llm.generate([{"prompt_token_ids": list(ids)}],
                            sampling_params=sp1, use_tqdm=False)

    print(f"model={a.model} prompt_tokens={a.prompt} n={N} gen={a.gen} reps={a.reps}")
    print(f"{'arm':<40}{'median_s':>10}{'seqs':>7}{'tokens_out':>12}")
    print("-" * 70)

    res = {}
    for name, fn in (("A: N separate requests (n=1)", arm_A),
                     ("B: ONE request with n=N", arm_B),
                     ("C: single n=1 (warm)", arm_C)):
        dt, out = timeit(fn)
        nseq = sum(len(o.outputs) for o in out)
        ntok = sum(len(x.token_ids) for o in out for x in o.outputs)
        res[name] = {"median_s": round(dt, 4), "seqs": nseq, "tok": ntok}
        print(f"{name:<40}{dt:>10.4f}{nseq:>7}{ntok:>12}")

    A, B = res["A: N separate requests (n=1)"], res["B: ONE request with n=N"]
    print("\n--- comparison (identical output work: same N sequences, same length) ---")
    if B["median_s"] > 0:
        print(f"  n=N vs N separate: {A['median_s']/B['median_s']:.3f}x "
              f"({'n=N faster' if A['median_s']>B['median_s'] else 'no gain from n=N'})")
    verdict = {
        "A_s": A["median_s"], "B_s": B["median_s"],
        "speedup_of_n_over_separate": round(A["median_s"] / B["median_s"], 3) if B["median_s"] else None,
        "same_output_volume": A["tok"] == B["tok"],
        "prefill_shared": A["median_s"] > B["median_s"] * 1.05,
    }
    print(json.dumps(verdict, indent=2))
    if not verdict["same_output_volume"]:
        print("\nWARNING: arms produced different token volumes; comparison confounded.")
    elif verdict["prefill_shared"]:
        print("\n=> n=N DOES share prefill (engine handles it); no waste to recover.")
    else:
        print("\n=> n=N shows no prefill advantage: the engine may be recomputing the")
        print("   prompt per sample. Worth a direct prefill measurement.")
    with open("/root/autodl-tmp/e3_parallel_result.json", "w") as f:
        json.dump({"config": vars(a), "arms": res, "verdict": verdict}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
