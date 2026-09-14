#!/usr/bin/env python
"""C-1e BREAK-EVEN: isolate the COST of speculation from the DATA (acceptance).

Method adopted from the maintainer-adjacent analysis on vLLM #54691
(hongboshi1234, 2026-09-12):

    run each config twice -- once normally, once with rejection sampling forced to
    accept NOTHING -- then
        break_even_acceptance = t_zero_accept / t_no_spec
    i.e. the acceptance rate you would NEED just to come out even. >1 means
    speculation cannot pay off even with a perfect drafter, regardless of data.

Why this is the right next experiment for my open question:
  I have shown (C-1d) that on hybrid Qwen3.5-4B, ngram speculation is a 2.79x net
  loss WHILE its acceptance is HIGHER than dense's (0.687 vs 0.573). Acceptance
  therefore cannot explain it. Break-even removes acceptance from the picture
  entirely and measures the pure per-step cost structure.

Pre-declared gate:
  * If break_even(hybrid) >> break_even(dense) -> the cost structure is the cause,
    and it is architecture-dependent, not data-dependent. (supports a real finding)
  * If break_even(hybrid) ~= break_even(dense) -> the K=3 penalty is data-driven
    after all and C-1d's acceptance numbers need re-examination.

Also sweeps context length, because #54691 reports the effect is context-dependent.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

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
        ids = ids + tok.encode(f"\nUser: task {c}\nAssistant:",
                               add_special_tokens=False)
        out.append(ids)
    return out


def zero_accept_spec_config(K):
    """SpeculativeConfig fields that force every draft to be REJECTED.

    These live on SpeculativeConfig (config/speculative.py:511-527), NOT on
    SamplingParams:
        rejection_sample_method: "synthetic"
        synthetic_acceptance_rates: zeros
    Then exactly one token is emitted per step while the full draft+verify still
    runs, so wall time is pure cost -- which is what makes break-even measurable
    without trusting any acceptance number.
    """
    return {"method": "ngram", "num_speculative_tokens": K,
            "prompt_lookup_max": 4, "prompt_lookup_min": 2,
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_rates": [0.0] * max(1, K)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--hybrid", default="/root/autodl-tmp/models/Qwen3.5-4B")
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prefixes", default="512,2048")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--targets", default="dense,hybrid")
    ap.add_argument("--out", default="/root/autodl-tmp/c1e_breakeven.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sp_normal = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)

    results = {}
    for target in [t.strip() for t in a.targets.split(",") if t.strip()]:
        path = a.dense if target == "dense" else a.hybrid
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        for P in [int(x) for x in a.prefixes.split(",") if x.strip()]:
            prompts = [{"prompt_token_ids": p}
                       for p in build_prompts(tok, a.n, P)]

            def build(mode, K):
                """mode: 'off' | 'spec' | 'zero'"""
                kw = dict(model=path, gpu_memory_utilization=a.gpu_util,
                          max_model_len=a.max_len, enable_prefix_caching=True,
                          enforce_eager=True, dtype="bfloat16",
                          language_model_only=True)
                if mode == "spec":
                    kw["speculative_config"] = {
                        "method": "ngram", "num_speculative_tokens": K,
                        "prompt_lookup_max": 4, "prompt_lookup_min": 2}
                elif mode == "zero":
                    kw["speculative_config"] = zero_accept_spec_config(K)
                return LLM(**kw)

            def timeit(llm, sp):
                xs = []
                for _ in range(a.passes):
                    t0 = time.perf_counter()
                    outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
                    dt = time.perf_counter() - t0
                    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
                    xs.append(ntok / dt if dt > 0 else 0.0)
                return st.median(xs)

            try:
                base = build("off", 0)
                t_base = timeit(base, sp_normal)
                del base
                import torch
                torch.cuda.empty_cache(); time.sleep(3)

                spec = build("spec", a.k)
                t_spec = timeit(spec, sp_normal)
                del spec
                torch.cuda.empty_cache(); time.sleep(3)

                z = build("zero", a.k)
                t_zero = timeit(z, sp_normal)
                del z
                torch.cuda.empty_cache(); time.sleep(3)
            except Exception as e:  # noqa: BLE001
                print(f"[{target} P={P}] FAILED {type(e).__name__}: {str(e)[:140]}",
                      flush=True)
                results[f"{target}|P={P}"] = {"error": str(e)[:200]}
                continue

            be = (t_base / t_zero) if t_zero else None
            # t_zero already holds ~1 token/step; convert to per-token seconds
            row = {
                "tok_per_s_no_spec": round(t_base, 1),
                "tok_per_s_spec": round(t_spec, 1),
                "spec_overhead_x": round(t_base / t_spec, 3) if t_spec else None,
                "tok_per_s_zero_accept": round(t_zero, 1) if t_zero else None,
                "break_even_acceptance": round(be, 3) if be else None,
            }
            results[f"{target}|P={P}"] = row
            print(f"[{target} P={P}] no_spec={row['tok_per_s_no_spec']:8.1f} "
                  f"spec={row['tok_per_s_spec']:8.1f} "
                  f"zero_acc={row['tok_per_s_zero_accept']} "
                  f"break_even={row['break_even_acceptance']}", flush=True)

    print("\n=== BREAK-EVEN SUMMARY (break_even = acceptance needed just to tie) ===")
    print(f"{'cell':<18}{'no_spec':>9}{'spec':>9}{'zero_acc':>10}{'break_even':>12}")
    for k, v in results.items():
        if "error" in v:
            print(f"{k:<18}{'FAIL':>9}")
            continue
        print(f"{k:<18}{v['tok_per_s_no_spec']:>9.1f}{v['tok_per_s_spec']:>9.1f}"
              f"{(v['tok_per_s_zero_accept'] or -1):>10.1f}"
              f"{(v['break_even_acceptance'] or -1):>12.3f}")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "results": results}, f, indent=2, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
