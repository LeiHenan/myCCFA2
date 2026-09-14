#!/usr/bin/env python
"""C-1i: is the acceptance->speedup model actually predictive on hybrid?

The contradiction I have to resolve:
  C-1d: hybrid mean_accept_len = 3.06 (higher than dense 2.72) yet hybrid speedup
        is far WORSE (0.73 vs 1.45).
  C-1e: hybrid break-even A* = 2.23 -> with A_actual = 3.06 the model predicts a
        WIN, contradicting C-1d.

The model both C-1d and C-1e implicitly assume is
        speedup  ~=  A_actual / A*
where A_actual = mean accepted length per step, A* = per-step token cost ratio
(measured via the zero-accept trick).

This probe TESTS that model instead of assuming it. For each cell it reports,
from a single engine instance and a single batch:
    A_actual      (from the engine's own per-request histogram)
    steps/token   (from the histogram's step count vs emitted tokens)
    speedup       (vs a no-spec baseline measured in the same rig)
    A*            (from a zero-accept engine)
and then compares  observed speedup  vs  A_actual / A*.

Gate:
  * if predicted/observed within ~15% -> the model holds, and the C-1d/C-1e
    contradiction means one of the two measurements was mis-normalised
  * if predicted/observed is far off -> the simple acceptance model is WRONG on
    hybrid, which is itself the finding (something other than acceptance and
    per-step cost governs the outcome)
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
    acc = den = steps = 0
    seen = 0
    for o in outs:
        for so in o.outputs:
            m = getattr(so, "spec_decode_metrics", None)
            if m is None:
                continue
            seen += 1
            h = list(getattr(m, "histogram", []) or [])
            nd = int(getattr(m, "num_draft_tokens", 0) or 0)
            acc += sum(j * c for j, c in enumerate(h))
            steps += sum(h)
            den += nd
    if not seen or not steps:
        return {}
    return {"n_with_metrics": seen, "accepted": acc, "steps": steps,
            "draft_tokens": den,
            "A_actual": round(1 + acc / steps, 4),
            "accept_rate": round(acc / den, 4) if den else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--hybrid", default="/root/autodl-tmp/models/Qwen3.5-4B")
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--targets", default="dense,hybrid")
    ap.add_argument("--out", default="/root/autodl-tmp/c1i_model.json")
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

        def build(mode):
            kw = dict(model=path, gpu_memory_utilization=a.gpu_util,
                      max_model_len=a.max_len, enable_prefix_caching=True,
                      enforce_eager=True, dtype="bfloat16",
                      language_model_only=True)
            if mode in ("spec", "zero"):
                sc = {"method": "ngram", "num_speculative_tokens": a.k,
                      "prompt_lookup_max": 4, "prompt_lookup_min": 2}
                if mode == "zero":
                    sc["rejection_sample_method"] = "synthetic"
                    sc["synthetic_acceptance_rates"] = [0.0] * a.k
                kw["speculative_config"] = sc
                kw["per_request_spec_decode_metrics"] = "detailed"
            return LLM(**kw)

        def run(llm):
            tps, mets = [], []
            for _ in range(a.passes):
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
                dt = time.perf_counter() - t0
                toks = sum(len(o.outputs[0].token_ids) for o in outs)
                tps.append(toks / dt if dt else 0.0)
                mets.append(metrics_from(outs))
            med_tps = st.median(tps)
            merged = {}
            for m in mets:
                for kk, vv in m.items():
                    merged.setdefault(kk, []).append(vv)
            merged = {kk: st.median(vv) for kk, vv in merged.items()}
            return med_tps, merged

        for mode in ("off", "spec", "zero"):
            llm = build(mode)
            tps, met = run(llm)
            rows[f"{target}|{mode}"] = {"tok_per_s": round(tps, 1), **met}
            print(f"[{target}|{mode}] {tps:8.1f} tok/s  {met}", flush=True)
            del llm
            import torch
            torch.cuda.empty_cache(); time.sleep(2)

    print("\n=== does  speedup ~= A_actual / A*  hold? ===")
    print(f"{'target':<8}{'no_spec':>9}{'spec':>9}{'zero':>9}"
          f"{'obs_sp':>8}{'A_act':>7}{'A*':>7}{'pred':>7}{'pred/obs':>9}")
    verdicts = {}
    for target in [t.strip() for t in a.targets.split(",") if t.strip()]:
        o = rows.get(f"{target}|off", {}).get("tok_per_s")
        s = rows.get(f"{target}|spec", {}).get("tok_per_s")
        z = rows.get(f"{target}|zero", {}).get("tok_per_s")
        A = rows.get(f"{target}|spec", {}).get("A_actual")
        if not (o and s and z and A):
            print(f"{target:<8} incomplete")
            continue
        obs = s / o
        Astar = o / z
        pred = A / Astar
        verdicts[target] = {"obs_speedup": round(obs, 3), "A_actual": A,
                            "A_star": round(Astar, 3), "pred_speedup": round(pred, 3),
                            "pred_over_obs": round(pred / obs, 3)}
        print(f"{target:<8}{o:>9.1f}{s:>9.1f}{z:>9.1f}{obs:>8.3f}{A:>7.2f}"
              f"{Astar:>7.3f}{pred:>7.3f}{pred / obs:>9.3f}")

    print()
    for t, v in verdicts.items():
        r = v["pred_over_obs"]
        ok = abs(r - 1) <= 0.15
        print(f"{t}: observed={v['obs_speedup']}  predicted={v['pred_speedup']}  "
              f"ratio={r}  -> {'MODEL HOLDS' if ok else 'MODEL FAILS'}")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "rows": rows, "verdicts": verdicts},
                  f, indent=2, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
