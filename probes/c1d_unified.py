#!/usr/bin/env python
"""C-1d UNIFIED RIG: measure throughput AND acceptance in ONE experiment.

Why this file exists -- my own two rigs disagreed by 3-20x:

  rig              dense K=3      hybrid K=3
  offline LLM API    1192.5          222.6     (C-1b: hybrid 2.4-2.8x SLOWER)
  OpenAI server        60.94          68.25    (C-1c: hybrid slightly FASTER)

Same box, same model, same K, same corpus. Both cannot be true, so at least one rig
has a systematic bias. Most likely cause (untested until now): the offline rig
submits 48 requests in ONE batched generate() call and gets continuous batching,
while the server rig issued 48 sequential HTTP requests with no batching.

This rig fixes the measurement problem by removing the comparison entirely:
it uses the OFFLINE API (same as C-1b, so throughput is directly comparable to it)
and reads ACCEPTANCE from the per-request metrics the engine attaches to each
output -- enabled with observability_config.per_request_spec_decode_metrics.

So: one rig, one batch shape, both quantities, both targets. That is the only way
the "hybrid acceptance is higher yet hybrid is slower" tension can be settled.

Gate (pre-declared):
  * if hybrid is still markedly slower here AND its acceptance is higher -> the
    C-1b effect is real and the cause is NOT rejection (real finding)
  * if hybrid is NOT slower here -> C-1b was a rig artifact -> archive C-1
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
TURNS = [
    "Summarise what the inventory module is responsible for.",
    "Which function would I change to add a retry on the POST?",
    "What could go wrong if the quantity is negative?",
    "Suggest a unit test for the navigation step.",
    "Rewrite the error handling to log and re-raise.",
    "What is the risk of calling the API twice for the same event?",
]


def build_prompts(tok, n_convs, prefix):
    filler = tok.encode(
        "The inventory service exposes a small HTTP API and a set of pure "
        "helpers, described below in detail for onboarding purposes. ",
        add_special_tokens=False)
    out = []
    for c in range(n_convs):
        sys_ids = tok.encode(SYSTEM, add_special_tokens=False)
        while len(sys_ids) < prefix:
            sys_ids = sys_ids + filler
        sys_ids = sys_ids[:prefix]
        acc = list(sys_ids)
        for k, t in enumerate(TURNS):
            acc = acc + tok.encode(f"\nUser: {t}\nAssistant:",
                                   add_special_tokens=False)
            out.append(list(acc))
            acc = acc + tok.encode(
                " Understood: the module validates input, posts to the API, "
                f"parses the response and navigates (turn {k}, variant {c}).",
                add_special_tokens=False)
    return out


def grab_metrics(outs):
    """Compute acceptance EXACTLY from the engine's per-request accumulator.

    RequestSpecDecodeMetrics (vllm/v1/metrics/stats.py:301) holds
        histogram[j] = number of verify steps that accepted exactly j draft tokens
        num_draft_tokens = total proposed draft tokens
    so accepted = sum(j * histogram[j]) and
       accept_rate = accepted / num_draft_tokens
       mean_accept_len = 1 + accepted / num_steps
    summed over every request in the batch (more robust than one request).
    """
    num = den = steps = acc = 0
    seen = 0
    for o in outs:
        for so in o.outputs:
            m = getattr(so, "spec_decode_metrics", None)
            if m is None:
                continue
            seen += 1
            h = list(getattr(m, "histogram", []) or [])
            nd = int(getattr(m, "num_draft_tokens", 0) or 0)
            a = sum(j * c for j, c in enumerate(h))
            s_ = sum(h)
            acc += a; den += nd; steps += s_
            num += len(h)
    if not seen or not den:
        return {}
    return {"requests_with_metrics": seen,
            "accept_rate": round(acc / den, 4),
            "mean_accept_len": round(1 + acc / steps, 4) if steps else None,
            "num_draft_tokens_total": den,
            "num_accepted_total": acc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--hybrid", default="/root/autodl-tmp/models/Qwen3.5-4B")
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--n-convs", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--ks", default="0,3")
    ap.add_argument("--targets", default="dense,hybrid")
    ap.add_argument("--out", default="/root/autodl-tmp/c1d_unified.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)
    results = {}

    for target in [t.strip() for t in a.targets.split(",") if t.strip()]:
        path = a.dense if target == "dense" else a.hybrid
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        prompts = [{"prompt_token_ids": p}
                   for p in build_prompts(tok, a.n_convs, a.prefix)]

        for K in [int(x) for x in a.ks.split(",") if x.strip() != ""]:
            label = f"{target}|K={K}"
            # EngineArgs exposes this as a TOP-LEVEL field (arg_utils.py:705),
            # not via an observability_config object.
            kw = dict(model=path, gpu_memory_utilization=a.gpu_util,
                      max_model_len=a.max_len, enable_prefix_caching=True,
                      enforce_eager=True, dtype="bfloat16",
                      language_model_only=True)
            if K > 0:
                kw["per_request_spec_decode_metrics"] = "detailed"
                kw["speculative_config"] = {
                    "method": "ngram", "num_speculative_tokens": K,
                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
            try:
                llm = LLM(**kw)
            except Exception as e:  # noqa: BLE001
                print(f"[{label}] ENGINE FAILED: {type(e).__name__}: {str(e)[:150]}",
                      flush=True)
                results[label] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                continue

            tps_runs, lens, mets = [], [], []
            for _ in range(a.passes):
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
                dt = time.perf_counter() - t0
                L = [len(o.outputs[0].token_ids) for o in outs]
                ntok = sum(L)
                tps_runs.append(ntok / dt if dt > 0 else 0.0)
                lens.append(st.median(L))
                mets.append(grab_metrics(outs))
            tps = st.median(tps_runs)
            merged: dict = {}
            for m in mets:
                for k, v in m.items():
                    merged.setdefault(k, []).append(v)
            merged = {k: st.median(v) for k, v in merged.items()}
            results[label] = {
                "tok_per_s": round(tps, 2),
                "tok_per_s_runs": [round(x, 1) for x in tps_runs],
                "out_len_median": st.median(lens),
                "metrics": merged,
            }
            print(f"[{label}] {tps:8.1f} tok/s  out_len={st.median(lens)}  {merged}",
                  flush=True)
            del llm
            import torch
            torch.cuda.empty_cache()
            time.sleep(3)

    print("\n=== UNIFIED SUMMARY ===")
    print(f"{'cell':<16}{'tok/s':>10}{'out_len':>9}{'acc_rate':>10}{'acc_len':>9}")
    for label, v in results.items():
        if "error" in v:
            print(f"{label:<16}{'FAIL':>10}")
            continue
        m = v.get("metrics", {})
        ar = m.get("draft_acceptance_rate")
        al = m.get("mean_acceptance_length")
        print(f"{label:<16}{v['tok_per_s']:>10.1f}{v['out_len_median']:>9}"
              f"{(f'{ar:.4f}' if isinstance(ar, float) else 'n/a'):>10}"
              f"{(f'{al:.3f}' if isinstance(al, float) else 'n/a'):>9}")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "results": results}, f, indent=2, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
