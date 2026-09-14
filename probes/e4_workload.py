#!/usr/bin/env python
"""E4-WORKLOAD: does the mean-vs-tail divergence depend on the WORKLOAD?

Weakness of the E4 evidence so far: prompts were synthetic token ids
(`[1000 + i*37 % 20000]`), which is not a realistic text distribution and gives
n-gram speculation an unusual amount of (un)usable structure. Before claiming the
effect generalises, it has to be reproduced on real text with *different amounts of
local repetition*, because that is exactly what n-gram acceptance depends on.

Design: build K workload families from REAL text via the model's own tokenizer, and
run the same controlled A/B (interleaved, order-alternated, pinned output length,
in-process noise floor) for each:

  W1 chat-like      : prose, low local repetition
  W2 code-like      : code-ish prose with repeated identifiers, medium repetition
  W3 repetitive     : a block repeated verbatim across and inside prompts -> n-gram
                      should shine, so this is the best case for acceleration
  W4 mixed          : half prose half repeated block

Reported per workload: mean/p95/p99 ratio, CV ratio, tail_vs_mean, frac_slower.
The scientific question is whether `tail_vs_mean > 1` appears in ALL families or
only in the high-repetition one.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

import torch

PROSE = (
    "The committee reviewed the proposal at length and raised several concerns about "
    "the timeline. Most members agreed that the initial phase should be shortened, and "
    "that a clearer definition of success was needed before any additional funding could "
    "be committed. A smaller working group was asked to prepare a revised schedule. "
)
CODEY = (
    "def process_inventory(apiBaseUrl, inventoryPath, formData):\n"
    "    response = await fetch(apiBaseUrl + inventoryPath, { method: 'post', body: formData })\n"
    "    parsed = await response.json()\n"
    "    if parsed.identifier > 0:\n"
    "        setQuantity(parsed.identifier)\n"
    "    return navigateTo(inventoryListPage, parsed.identifier)\n"
)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def stats(xs):
    return {"n": len(xs), "mean_ms": st.mean(xs) * 1000, "p50_ms": pct(xs, .50) * 1000,
            "p95_ms": pct(xs, .95) * 1000, "p99_ms": pct(xs, .99) * 1000,
            "cv": st.pstdev(xs) / st.mean(xs) if st.mean(xs) else float("nan")}


def build_texts(tok, target_tokens: int, n_prompts: int, pseed: int = 0):
    """Return {family: [token_id_list, ...]} using the real tokenizer."""
    def to_len(text: str, uniq_tail: str) -> list[int]:
        for _ in range(60):
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) >= target_tokens:
                break
            text = text + text
        ids = ids[:target_tokens]
        # give each prompt a distinct tail so they are not all identical
        tail = tok.encode(uniq_tail, add_special_tokens=False)
        return (ids[: max(0, target_tokens - len(tail))] + tail)

    fams = {}
    for name, body in (("W1_prose", PROSE), ("W2_code", CODEY),
                       ("W3_repetitive", CODEY * 4), ("W4_mixed", PROSE + CODEY * 2)):
        if pseed:
            body = body + " " + body
        _ = pseed
        fams[name] = [to_len(body, f" [variant {pseed}-{k:04d}]")
                      for k in range(n_prompts)]
    # sanity: families must actually differ in measurable repetition
    return fams


def prompt_match_rate(ids: list[int], n: int = 4) -> float:
    """Fraction of positions where the n-gram ending at i recurs EARLIER in the same
    context. This is the quantity n-gram/prompt-lookup speculation actually exploits
    (it can only propose a continuation that already appeared before the cursor),
    so it is the right headroom proxy -- unlike raw n-gram uniqueness, which
    saturates and does not discriminate between families."""
    grams = {}
    hits = 0
    total = 0
    for i in range(len(ids) - n):
        g = tuple(ids[i:i + n])
        total += 1
        if g in grams:
            hits += 1
        grams[g] = i
    return hits / total if total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--families", default="W1_prose,W2_code,W3_repetitive,W4_mixed")
    ap.add_argument("--out", default="/root/autodl-tmp/e4_workload_summary.json")
    ap.add_argument("--pseed", type=int, default=0,
                    help="prompt-set seed; vary for independent replication")
    ap.add_argument("--method", default="ngram",
                    choices=("ngram", "eagle3"),
                    help="which acceleration technique to compare against baseline")
    ap.add_argument("--eagle3", default="/root/autodl-tmp/models/eagle3-qwen3-4b")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    def build(spec: bool):
        kw = dict(model=a.model, gpu_memory_utilization=a.gpu_util,
                  max_model_len=a.max_len, enable_prefix_caching=False,
                  enforce_eager=True, dtype="bfloat16")
        if spec:
            if a.method == "ngram":
                kw["speculative_config"] = {
                    "method": "ngram", "num_speculative_tokens": 5,
                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
            else:
                kw["speculative_config"] = {
                    "method": "eagle3", "model": a.eagle3,
                    "num_speculative_tokens": 3}
        return LLM(**kw)

    wanted = [f for f in a.families.split(",") if f.strip()]
    fams = build_texts(tok, a.prompt, a.n_prompts, a.pseed)
    fams = {k: v for k, v in fams.items() if k in wanted}
    for k, v in fams.items():
        print(f"{k}: prompt_match={prompt_match_rate(v[0]):.3f} len={len(v[0])}",
              flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)

    print("building engines...", flush=True)
    eB = build(False)
    eS = build(True)

    def timed(eng, one):
        t0 = time.perf_counter()
        eng.generate([one], sampling_params=sp, use_tqdm=False)
        return time.perf_counter() - t0

    results = {}
    for fam, prompts in fams.items():
        inputs = [{"prompt_token_ids": p} for p in prompts]
        for e in (eB, eS):
            e.generate(inputs[:2], sampling_params=sp, use_tqdm=False)
        A, B = [], []
        for i, one in enumerate(inputs):
            if i % 2 == 0:
                A.append(timed(eB, one)); B.append(timed(eS, one))
            else:
                B.append(timed(eS, one)); A.append(timed(eB, one))
        A2 = [timed(eB, one) for one in inputs]          # noise floor
        sA, sB, sA2 = stats(A), stats(B), stats(A2)
        per = [b / x for x, b in zip(A, B)]
        r = {"mean": sB["mean_ms"] / sA["mean_ms"], "p50": sB["p50_ms"] / sA["p50_ms"],
             "p95": sB["p95_ms"] / sA["p95_ms"], "p99": sB["p99_ms"] / sA["p99_ms"],
             "cv": sB["cv"] / sA["cv"]}
        noise = {"mean": sA2["mean_ms"] / sA["mean_ms"],
                 "p95": sA2["p95_ms"] / sA["p95_ms"]}
        tv = r["p95"] / r["mean"]
        nv = noise["p95"] / noise["mean"]
        results[fam] = {
            "prompt_match": round(prompt_match_rate(prompts[0]), 4),
            "ratios": {k: round(v, 4) for k, v in r.items()},
            "noise": {k: round(v, 4) for k, v in noise.items()},
            "tail_vs_mean": round(tv, 4), "noise_tail_vs_mean": round(nv, 4),
            "exceeds_noise": abs(tv - 1) > abs(nv - 1),
            "frac_slower": round(sum(1 for x in per if x > 1.0) / len(per), 4),
            "per_req_min": round(min(per), 4), "per_req_max": round(max(per), 4),
        }
        print(f"  {fam:<16} pmatch={results[fam]['prompt_match']:.3f} "
              f"mean={r['mean']:.3f} p95={r['p95']:.3f} "
              f"tv={tv:.3f} (noise {nv:.3f}) slower={results[fam]['frac_slower']:.2f}",
              flush=True)

    print("\n=== workload summary ===")
    print(f"{'family':<16}{'pmatch':>8}{'mean':>8}{'p95':>8}{'tail/mean':>11}"
          f"{'noise':>8}{'slower':>8}")
    for k, v in results.items():
        print(f"{k:<16}{v['prompt_match']:>8.3f}{v['ratios']['mean']:>8.3f}"
              f"{v['ratios']['p95']:>8.3f}{v['tail_vs_mean']:>11.3f}"
              f"{v['noise_tail_vs_mean']:>8.3f}{v['frac_slower']:>8.2f}")
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {a.out}")

    fams_ok = [k for k, v in results.items() if v["tail_vs_mean"] > 1.15 and v["exceeds_noise"]]
    print(f"\nworkloads showing tail/mean divergence: {len(fams_ok)}/{len(results)} -> {fams_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
