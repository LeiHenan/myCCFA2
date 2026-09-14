#!/usr/bin/env python
"""C-1b: verify the hybrid speculation collapse is real (not a silent-no-op artifact).

Finding under test (from c1_hybrid.json):
    dense Qwen3-4B  : no-spec 1196.9 -> +ngram 1231.9 tok/s   (+3%)
    hybrid Qwen3.5-4B: no-spec  615.4 -> +ngram  209.7 tok/s  (-66%)

Three artifact hypotheses must be killed before this can be claimed:

  H1  "the drafter silently did nothing" -- then ngram==no-spec in throughput and
      acceptance would be ~0. But ngram was SLOWER, which a no-op cannot explain;
      still, we read the engine's own spec-decode counters to confirm drafting runs.
  H2  "the cost scales with num_speculative_tokens because the draft step is just
      expensive on hybrid" -- test by sweeping K. If throughput degrades
      monotonically in K on hybrid but is flat on dense, H2 is supported and the
      mechanism is per-K draft cost, not acceptance.
  H3  "different acceptance" -- measured via the engine's counters.

Design: same corpus/design as c1_spec_prefix (real text, multi-turn, shared prefix,
pinned output length, prefix caching ON), sweeping K on both targets, and scraping
the engine's SpecDecoding log line (Mean acceptance length / Avg Draft acceptance
rate) to get acceptance directly.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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


class SpecCapture(logging.Handler):
    """Capture vLLM's own SpecDecoding metrics log lines."""

    PAT = re.compile(
        r"Mean acceptance length:\s*([0-9.]+).*?"
        r"Avg Draft acceptance rate:\s*([0-9.]+)%", re.S)

    def __init__(self):
        super().__init__()
        self.mean_accept_len: list[float] = []
        self.accept_rate: list[float] = []
        self.lines: list[str] = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if "Mean acceptance length" in msg:
            m = self.PAT.search(msg)
            if m:
                self.mean_accept_len.append(float(m.group(1)))
                self.accept_rate.append(float(m.group(2)))
                self.lines.append(msg[:300])


def build_conversations(tok, n_convs, pseed, prefix_tokens):
    out = []
    filler = tok.encode(
        "The inventory service exposes a small HTTP API and a set of pure "
        "helpers, described below in detail for onboarding purposes. ",
        add_special_tokens=False)
    for c in range(n_convs):
        sys_ids = tok.encode(SYSTEM, add_special_tokens=False)
        while len(sys_ids) < prefix_tokens:
            sys_ids = sys_ids + filler
        sys_ids = sys_ids[:prefix_tokens]
        convo, acc = [], list(sys_ids)
        for k, t in enumerate(TURNS):
            acc = acc + tok.encode(f"\nUser: {t}\nAssistant:", add_special_tokens=False)
            convo.append(list(acc))
            acc = acc + tok.encode(
                " Understood: the module validates input, posts to the API, "
                f"parses the response and navigates (turn {k}, variant {pseed}).",
                add_special_tokens=False)
        out.append(convo)
    return out


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
    ap.add_argument("--ks", default="0,1,3,5,7")
    ap.add_argument("--targets", default="dense,hybrid")
    ap.add_argument("--out", default="/root/autodl-tmp/c1b_result.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # attach a capture handler to vLLM's spec-decode logger
    cap = SpecCapture()
    for name in ("vllm.v1.spec_decode.metrics", "vllm.spec_decode.metrics",
                 "vllm.v1.spec_decode.metrics"):
        lg = logging.getLogger(name)
        lg.addHandler(cap)
        lg.setLevel(logging.INFO)
    logging.getLogger().addHandler(cap)

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)
    results: dict = {}

    for target in [t.strip() for t in a.targets.split(",") if t.strip()]:
        path = a.dense if target == "dense" else a.hybrid
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        convs = build_conversations(tok, a.n_convs, 0, a.prefix)
        flat = [{"prompt_token_ids": p} for c in convs for p in c]

        for K in [int(x) for x in a.ks.split(",") if x.strip() != ""]:
            label = f"{target}|K={K}"
            kw = dict(model=path, gpu_memory_utilization=a.gpu_util,
                      max_model_len=a.max_len, enable_prefix_caching=True,
                      enforce_eager=True, dtype="bfloat16",
                      language_model_only=True)
            if K > 0:
                kw["speculative_config"] = {
                    "method": "ngram", "num_speculative_tokens": K,
                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
            try:
                llm = LLM(**kw)
            except Exception as e:  # noqa: BLE001
                print(f"[{label}] ENGINE FAILED: {type(e).__name__}: {str(e)[:160]}",
                      flush=True)
                results[label] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                continue

            cap.mean_accept_len.clear(); cap.accept_rate.clear()
            tps_runs = []
            for p in range(a.passes):
                t0 = time.perf_counter()
                outs = llm.generate(flat, sampling_params=sp, use_tqdm=False)
                dt = time.perf_counter() - t0
                ntok = sum(len(o.outputs[0].token_ids) for o in outs)
                tps_runs.append(ntok / dt if dt > 0 else 0.0)
            tps = st.median(tps_runs)
            mal = st.median(cap.mean_accept_len) if cap.mean_accept_len else None
            acc = st.median(cap.accept_rate) if cap.accept_rate else None
            results[label] = {
                "tok_per_s": round(tps, 2),
                "tok_per_s_runs": [round(x, 1) for x in tps_runs],
                "mean_accept_len": mal, "accept_rate_pct": acc,
                "spec_log_lines": len(cap.lines),
            }
            print(f"[{label}] {tps:8.1f} tok/s   accept_len={mal}  "
                  f"accept_rate={acc}%  (runs={[round(x,1) for x in tps_runs]})",
                  flush=True)
            del llm
            import torch
            torch.cuda.empty_cache()
            time.sleep(3)

    print("\n=== summary ===")
    print(f"{'target|K':<14}{'tok/s':>10}{'acc_len':>10}{'acc_rate%':>11}")
    for label, v in results.items():
        if "error" in v:
            print(f"{label:<14}{'FAIL':>10}")
            continue
        print(f"{label:<14}{v['tok_per_s']:>10.1f}"
              f"{(v['mean_accept_len'] if v['mean_accept_len'] is not None else -1):>10.2f}"
              f"{(v['accept_rate_pct'] if v['accept_rate_pct'] is not None else -1):>11.1f}")

    # H2 test: does throughput degrade monotonically in K on hybrid?
    for target in [t.strip() for t in a.targets.split(",") if t.strip()]:
        seq = [(int(k), results.get(f"{target}|K={k}", {}).get("tok_per_s"))
               for k in a.ks.split(",") if k.strip() != ""]
        seq = [(k, v) for k, v in seq if v]
        if len(seq) >= 3:
            mono = sum(1 for i in range(1, len(seq)) if seq[i][1] < seq[i - 1][1])
            print(f"\n{target}: tok/s by K = {seq}")
            print(f"  monotone-decreasing steps: {mono}/{len(seq)-1}")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "results": results}, f, indent=2, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
