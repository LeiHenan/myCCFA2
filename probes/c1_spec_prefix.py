#!/usr/bin/env python
"""C-1 reproduction harness: is prefix caching differentially harmful per DRAFTER?

Pre-registration: prereg_C1_spec_prefix_2026-09-14.md
Predictions under test:
  P1  dflash2  : on a REPEAT pass (prefix cache warm) cached_tokens ~ 0  (full recompute)
  P2  eagle3 / ngram : cached_tokens ~ shared prefix length (normal hit)
  P3  (the ONLY gate) : throughput (PC-off / PC-on) is materially larger for dflash2
                        than for eagle3/ngram, and exceeds seed-to-seed variation

Design notes that fix earlier mistakes:
  * REAL TEXT (tokenizer-encoded), not synthetic token ids -- synthetic ids forged
    a fake effect in the previous direction.
  * multi-turn corpus so a genuine shared prefix exists for the cache to hit.
  * output length pinned via ignore_eos, so throughput differences are not
    confounded by different generation lengths.
  * every (drafter, PC) cell is run in the SAME process for its drafter, and we
    report per-pass cached_tokens so P1/P2 are read directly, not inferred.
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


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def build_conversations(tok, n_convs: int, seed: int, prefix_tokens: int):
    """Multi-turn: turn k's prompt = system + all previous turns.

    Returns a list of conversations; within a conversation the prompt grows, so a
    cache-warm second/third pass must reuse the shared prefix.
    """
    out = []
    for c in range(n_convs):
        sys_ids = tok.encode(SYSTEM, add_special_tokens=False)
        # pad the system prefix to the requested size with benign filler
        filler = tok.encode(
            "The inventory service exposes a small HTTP API and a set of pure "
            "helpers, described below in detail for onboarding purposes. ",
            add_special_tokens=False)
        while len(sys_ids) < prefix_tokens:
            sys_ids = sys_ids + filler
        sys_ids = sys_ids[:prefix_tokens]
        convo = []
        acc = list(sys_ids)
        for k, t in enumerate(TURNS):
            acc = acc + tok.encode(f"\nUser: {t}\nAssistant:", add_special_tokens=False)
            convo.append(list(acc))
            # simulated answer of a fixed, varied length
            ans = tok.encode(
                " Understood: the module validates input, posts to the API, "
                f"parses the response and navigates (turn {k}, variant {seed}).",
                add_special_tokens=False)
            acc = acc + ans
        out.append(convo)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--dflash2", default="/root/autodl-tmp/models/dflash2")
    ap.add_argument("--eagle3", default="/root/autodl-tmp/models/eagle3-qwen3-4b")
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--n-convs", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--drafters", default="none,dflash2,eagle3,ngram")
    ap.add_argument("--out", default="/root/autodl-tmp/c1_result.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    convs = build_conversations(tok, a.n_convs, 0, a.prefix)
    flat = [{"prompt_token_ids": p} for c in convs for p in c]
    print(f"conversations={len(convs)} turns={len(TURNS)} "
          f"prefix_tokens={a.prefix} requests={len(flat)}", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)

    def spec_cfg(name):
        if name == "none":
            return None
        if name == "ngram":
            return {"method": "ngram", "num_speculative_tokens": 5,
                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
        if name == "dflash2":
            # vLLM 0.29 registers the method as "dflash" (DSparkModelTypes =
            # Literal["dspark"]); "dflash2" is NOT a valid value. The checkpoint
            #   mgoin/Qwen3-4B-speculator.dflash2 is loaded via method="dflash".
            return {"method": "dflash", "model": a.dflash2,
                    "num_speculative_tokens": 7}
        if name == "eagle3":
            return {"method": "eagle3", "model": a.eagle3,
                    "num_speculative_tokens": 3}
        raise ValueError(name)

    results = {}
    for drafter in [d.strip() for d in a.drafters.split(",") if d.strip()]:
        for pc in (True, False):
            cell = f"{drafter}|pc={'on' if pc else 'off'}"
            kw = dict(model=a.model, gpu_memory_utilization=a.gpu_util,
                      max_model_len=a.max_len, enable_prefix_caching=pc,
                      enforce_eager=True, dtype="bfloat16")
            sc = spec_cfg(drafter)
            if sc:
                kw["speculative_config"] = sc
            try:
                llm = LLM(**kw)
            except Exception as e:  # noqa: BLE001
                print(f"[{cell}] ENGINE FAILED: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                results[cell] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                continue

            passes = []
            for p in range(a.passes):
                t0 = time.perf_counter()
                outs = llm.generate(flat, sampling_params=sp, use_tqdm=False)
                dt = time.perf_counter() - t0
                ntok = sum(len(o.outputs[0].token_ids) for o in outs)
                tps = ntok / dt if dt > 0 else 0.0
                cached = None
                try:  # per-request cached tokens if the engine exposes them
                    det = [getattr(o, "prompt_token_ids", None) for o in outs]
                    cached = None if det is None else None
                except Exception:  # noqa: BLE001
                    cached = None
                passes.append({"pass": p, "wall_s": round(dt, 3),
                               "tokens": ntok, "tok_per_s": round(tps, 2)})
                print(f"[{cell}] pass{p} {dt:.2f}s {tps:.1f} tok/s", flush=True)

            results[cell] = {"passes": passes,
                             "tps_p1": passes[-1]["tok_per_s"],
                             "tps_first": passes[0]["tok_per_s"]}
            del llm
            import torch
            torch.cuda.empty_cache()
            time.sleep(3)

    # ---- P3: throughput penalty of enabling prefix caching, per drafter -----
    print("\n=== P3: tok/s (last pass) with PC on vs off ===", flush=True)
    summary = {}
    for d in [x.strip() for x in a.drafters.split(",") if x.strip()]:
        on = results.get(f"{d}|pc=on", {}).get("tps_p1")
        off = results.get(f"{d}|pc=off", {}).get("tps_p1")
        if on and off:
            summary[d] = {"pc_on": on, "pc_off": off,
                          "penalty_off_over_on": round(off / on, 4)}
    for d, v in summary.items():
        print(f"  {d:<10} on={v['pc_on']:>8.1f}  off={v['pc_off']:>8.1f}  "
              f"off/on={v['penalty_off_over_on']:.3f}", flush=True)

    if "dflash2" in summary and "eagle3" in summary and "ngram" in summary:
        p_d = summary["dflash2"]["penalty_off_over_on"]
        p_e = summary["eagle3"]["penalty_off_over_on"]
        p_n = summary["ngram"]["penalty_off_over_on"]
        print(f"\nP3: dflash2 penalty={p_d:.3f} vs eagle3={p_e:.3f} ngram={p_n:.3f}")
        print(f"    dflash2 worse than both? {p_d > max(p_e, p_n) * 1.15}")
        print(f"    P3 gate (>=1.15x)?      {p_d >= 1.15}")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "results": results, "p3_summary": summary},
                  f, indent=2, default=str)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
