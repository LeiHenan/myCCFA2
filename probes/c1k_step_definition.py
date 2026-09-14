#!/usr/bin/env python
"""C-1k: settle the STEP-COUNT definition, then refit the per-draft marginal cost.

Why this probe exists (self-audit of C-1j)
------------------------------------------
C-1j fitted      t_step(K) = intercept + slope*K
using            steps = sum(histogram)
and reported    dense slope 1.264 / hybrid slope 7.075 ms per draft token.

A self-consistency check then failed:
        A * steps  should equal total generated tokens (gen*n = 384)
        measured   73-162 tokens   -> denominator low by ~5x

Reading the engine source explains it.  In vllm/v1/core/sched/scheduler.py the
per-request accumulator is fed only under

    if scheduled_spec_token_ids and (generated_token_ids or ...):
        request.spec_decode_metrics.observe(...)

i.e. histogram counts **VERIFY** steps (steps that carried scheduled drafts AND
produced output), not engine iterations.  Iterations that only propose drafts,
or that prefill, never touch it.

Decisive symptom in the C-1j data: dense K=1 reported A = 1.49, i.e. a single
step supposedly accepted 1.49 tokens while drafting at most 1.  That cannot
happen per iteration, so `sum(histogram)` cannot be iterations there either.
It also reported 1.32 draft tokens per verify step at K=1 -- the schedule
evidently carries more drafts than one pass proposes.

So this probe measures the iteration count FOUR independent ways:
    steps_eng   = delta(vllm:spec_decode_num_drafts)     [iteration counter]
    steps_tok   = sum of vllm:iteration_tokens_total histogram
                  ("Histogram of number of tokens per engine_step")
    steps_ident = total_output_tokens / A_engine          [identity, by def.]
    steps_hist  = sum(histogram)                          [old, suspect]

and then refits t_step = wall / steps.
Whichever definition satisfies  A * steps == total_tokens  is the real one.

n=1 on purpose: one sequence, so an engine step == that sequence's step and the
batching question cannot muddy the accounting.

Also runs a no-spec arm so the prefill/step baseline is measurable.
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


def per_request(outs):
    """Per-request histogram roll-up (the C-1j definition)."""
    acc = steps = drafts = seen = 0
    for o in outs:
        for so in o.outputs:
            m = getattr(so, "spec_decode_metrics", None)
            if m is None:
                continue
            seen += 1
            h = list(getattr(m, "histogram", []) or [])
            acc += sum(j * c for j, c in enumerate(h))
            steps += sum(h)
            drafts += int(getattr(m, "num_draft_tokens", 0) or 0)
    if not seen or not steps:
        return {}
    return {"n": seen, "acc_hist": acc, "steps_hist": steps,
            "drafts_hist": drafts, "A_hist": 1 + acc / steps}


def reset_engine_metrics():
    try:
        from vllm.v1.metrics.reader import get_metrics
    except Exception:
        return False
    try:
        for m in get_metrics():
            if m.name == "vllm:spec_decode_num_drafts":
                m.value = 0
    except Exception:
        pass
    return True


def engine_counters():
    """Read cumulative engine counters; return {} if unavailable."""
    try:
        from vllm.v1.metrics.reader import get_metrics
    except Exception:
        return {}
    out = {}
    try:
        ms = get_metrics()
    except Exception:
        return {}
    for m in ms:
        n = getattr(m, "name", "")
        v = getattr(m, "value", None)
        if v is None:
            continue
        if n == "vllm:spec_decode_num_drafts":
            out["drafts"] = out.get("drafts", 0) + v
        elif n == "vllm:spec_decode_num_draft_tokens":
            out["draft_tokens"] = out.get("draft_tokens", 0) + v
        elif n == "vllm:spec_decode_num_accepted_tokens":
            out["accepted"] = out.get("accepted", 0) + v
    # iteration_tokens is a Histogram: use its _count (number of observations)
    for m in ms:
        n = getattr(m, "name", "")
        if n == "vllm:iteration_tokens_total_count":
            out["iter"] = out.get("iter", 0) + getattr(m, "value", 0)
        elif n == "vllm:iteration_tokens_total_sum":
            out["iter_tokens"] = out.get("iter_tokens", 0) + getattr(m, "value", 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--hybrid", default="/root/autodl-tmp/models/Qwen3.5-4B")
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--ks", default="0,1,2,3,4,6,8")
    ap.add_argument("--targets", default="dense,hybrid")
    ap.add_argument("--out", default="/root/autodl-tmp/c1k_step_definition.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)
    rows = {}
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]

    print(f"engine metrics readable: {reset_engine_metrics()}", flush=True)

    for target in targets:
        path = a.dense if target == "dense" else a.hybrid
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        prompts = [{"prompt_token_ids": p}
                   for p in build_prompts(tok, a.n, a.prefix)]
        total_out = a.gen * a.n

        for K in ks:
            kw = dict(model=path, gpu_memory_utilization=a.gpu_util,
                      max_model_len=a.max_len, enable_prefix_caching=True,
                      enforce_eager=True, dtype="bfloat16",
                      language_model_only=True,
                      per_request_spec_decode_metrics="detailed")
            if K > 0:
                kw["speculative_config"] = {
                    "method": "ngram", "num_speculative_tokens": K,
                    "prompt_lookup_max": 6, "prompt_lookup_min": 2}
            try:
                llm = LLM(**kw)
            except Exception as e:  # noqa: BLE001
                print(f"[{target} K={K}] ENGINE FAILED {type(e).__name__}: "
                      f"{str(e)[:120]}", flush=True)
                rows[f"{target}|K={K}"] = {"error": str(e)[:200]}
                continue

            walls, prs, engs = [], [], []
            for _ in range(a.passes):
                reset_engine_metrics()
                c0 = engine_counters()
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
                w = time.perf_counter() - t0
                c1 = engine_counters()
                walls.append(w)
                prs.append(per_request(outs))
                engs.append({k: c1.get(k, 0) - c0.get(k, 0) for k in
                             ("drafts", "draft_tokens", "accepted", "iter",
                              "iter_tokens")})

            wall = st.median(walls)
            pr = {}
            for m in prs:
                for kk, vv in m.items():
                    pr.setdefault(kk, []).append(vv)
            pr = {kk: st.median(vv) for kk, vv in pr.items()}
            en = {}
            for m in engs:
                for kk, vv in m.items():
                    en.setdefault(kk, []).append(vv)
            en = {kk: st.median(vv) for kk, vv in en.items()}

            steps_hist = pr.get("steps_hist")
            A_hist = pr.get("A_hist")
            acc_eng = en.get("accepted")
            drafts_eng = en.get("drafts")
            A_eng = (1 + acc_eng / drafts_eng) if drafts_eng else None

            def tstep(s):
                return round(wall / s * 1000, 4) if s else None

            row = {
                "K": K, "wall_s": round(wall, 4), "total_out": total_out,
                "steps_hist": steps_hist, "steps_eng": drafts_eng,
                "A_hist": round(A_hist, 4) if A_hist else None,
                "drafts_hist": pr.get("drafts_hist"),
                "drafts_eng": drafts_eng, "accepted_eng": acc_eng,
                "draft_tokens_eng": en.get("draft_tokens"),
                "iter_hist": en.get("iter"), "iter_tokens": en.get("iter_tokens"),
                "A_eng": round(A_eng, 4) if A_eng else None,
                "t_step_ms_hist": tstep(steps_hist),
                "t_step_ms_eng": tstep(drafts_eng),
                "t_step_ms_iter": tstep(en.get("iter")),
                "t_step_ms_ident": tstep(total_out / A_eng) if A_eng else None,
                "tokens_per_s": round(total_out / wall, 1) if wall else None,
                # identity check: does A * steps reproduce the real token count?
                "ident_hist": round(A_hist * steps_hist, 1) if (A_hist and steps_hist) else None,
                "ident_eng": round(A_eng * drafts_eng, 1) if (A_eng and drafts_eng) else None,
                "ident_iter": round(A_eng * en.get("iter", 0), 1) if A_eng else None,
            }
            rows[f"{target}|K={K}"] = row
            print(f"[{target} K={K}] wall={wall:.3f}s tok/s={row['tokens_per_s']} "
                  f"| hist: steps={steps_hist} A={row['A_hist']} "
                  f"->ident={row['ident_hist']} "
                  f"| eng: iters={drafts_eng} A={row['A_eng']} "
                  f"->ident={row['ident_eng']} "
                  f"| iter_metric={en.get('iter')} "
                  f"(need {total_out})", flush=True)
            del llm
            import torch
            torch.cuda.empty_cache()
            time.sleep(2)

    print("\n=== which step definition satisfies  A * steps == total tokens? ===")
    print(f"{'cell':<13}{'total':>6}{'hist':>7}{'i_hist':>8}{'eng':>6}{'i_eng':>7}"
          f"{'iter':>6}{'i_iter':>8}")
    for target in targets:
        for K in ks:
            r = rows.get(f"{target}|K={K}") or {}
            if r.get("error"):
                print(f"{target}|K={K:<8} ENGINE FAILED")
                continue
            print(f"{target + '|K=' + str(K):<13}{r['total_out']:>6}"
                  f"{str(r['steps_hist']):>7}{str(r['ident_hist']):>8}"
                  f"{str(r['steps_eng']):>6}"
                  f"{str(r['ident_eng']):>7}{str(r['iter_hist']):>6}"
                  f"{str(r['ident_iter']):>8}")

    print("\n=== refit t_step = intercept + slope*K under each definition ===")
    for defn in ("t_step_ms_hist", "t_step_ms_eng", "t_step_ms_iter"):
        print(f"-- {defn}")
        for target in targets:
            pts = [(r["K"], r[defn]) for r in
                   (rows.get(f"{target}|K={K}") or {} for K in ks)
                   if r.get(defn) and r.get("K")]
            if len(pts) < 3:
                continue
            n = len(pts)
            sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
            sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
            den = n * sxx - sx * sx
            slope = (n * sxy - sx * sy) / den if den else float("nan")
            inter = (sy - slope * sx) / n if n else float("nan")
            print(f"   {target:<7} t_step = {inter:7.3f} + {slope:6.3f}*K   "
                  f"(n={n})")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "rows": rows}, f, indent=2, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
