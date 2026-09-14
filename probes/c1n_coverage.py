#!/usr/bin/env python
"""C-1n: why did speculative decoding only buy 1.15x?  Decompose the REAL steps.

Where this comes from (second self-audit, same day)
--------------------------------------------------
C-1j used `steps = sum(histogram)` and got nonsense, so C-1l/C-1m replaced it
with a patched EngineCore step counter (ground truth).  That counter reports
88-99 engine steps for a 96-token generation, while the histogram reports 6-21.

First reading: "the histogram undercounts by 4-9x".  That is WRONG.  The
histogram counts exactly the steps that DRAFTED, and the engine's own counter
proves it:

    vllm:spec_decode_num_draft_tokens / K  ==  sum(histogram)      (exact)

for every cell measured (dense K=2/3/4: 22/2 = 33/3 = 44/4 = 11 = hist_window).

So the real structure of a decode step is:

    draft steps     : proposed K tokens, verified, accepted A_cond tokens
    non-draft steps : the ngram proposer found no match, so 1 plain token

and only the former are in the histogram.  With M = fraction of decode steps
that managed to draft, the achievable throughput multiplier is

    speedup = (1 - M) * 1  +  M * A_cond

NOT A alone.  This probe measures M directly and checks that formula against the
measured end-to-end speedup.  If it holds, then the reason speculation is weak
here is PROPOSER COVERAGE (M), not acceptance quality (A) and not architecture.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

STEP_FILE = os.environ.get("C1N_STEP_FILE", "/tmp/c1n_steps.txt")

if os.environ.get("C1N_PATCH") == "1":
    try:
        import vllm.v1.engine.core as _core

        if not getattr(_core.EngineCore, "_c1n_patched", False):
            _os, _oswbq = _core.EngineCore.step, _core.EngineCore.step_with_batch_queue

            def _tick():
                try:
                    with open(STEP_FILE, "a") as f:
                        f.write("1")
                except Exception:
                    pass

            def _step(self, *a, **k):
                _tick(); return _os(self, *a, **k)

            def _swbq(self, *a, **k):
                _tick(); return _oswbq(self, *a, **k)

            _core.EngineCore.step = _step
            _core.EngineCore.step_with_batch_queue = _swbq
            _core.EngineCore._c1n_patched = True
    except Exception as _e:  # noqa: BLE001
        print(f"[c1n] patch failed: {_e}", flush=True)


def nsteps():
    try:
        return os.path.getsize(STEP_FILE)
    except OSError:
        return 0


def reset_steps():
    try:
        open(STEP_FILE, "w").close()
    except OSError:
        pass


def snapshot():
    try:
        from vllm.v1.metrics.reader import get_metrics_snapshot
        ms = get_metrics_snapshot()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for m in ms:
        n = getattr(m, "name", "")
        if not n:
            continue
        v = getattr(m, "value", None)
        if v is None:
            v = getattr(m, "count", None)
        if v is None:
            continue
        try:
            out[n] = out.get(n, 0.0) + float(v)
        except (TypeError, ValueError):
            pass
    return out


def norm(n):
    return n[:-6] if n.endswith("_total") else n


def g(d, *names):
    for nm in names:
        if nm in d:
            return d[nm]
    return 0.0


def detail(outs):
    """Per-request histogram + (when available) the ordered per-step arrays."""
    acc = steps = drafts = 0
    per_acc, per_dft = [], []
    for o in outs:
        for so in o.outputs:
            m = getattr(so, "spec_decode_metrics", None)
            if m is None:
                continue
            h = list(getattr(m, "histogram", []) or [])
            acc += sum(j * c for j, c in enumerate(h))
            steps += sum(h)
            drafts += int(getattr(m, "num_draft_tokens", 0) or 0)
            per_acc += list(getattr(m, "per_step_accepted", []) or [])
            per_dft += list(getattr(m, "per_step_drafted", []) or [])
    return {"hist_steps": steps, "hist_acc": acc, "hist_drafts": drafts,
            "per_step_accepted": per_acc, "per_step_drafted": per_dft}


SYSTEM = ("You are a meticulous software engineering assistant embedded in a "
          "large codebase. Always answer concisely and cite the relevant "
          "module. Follow the team conventions: prefer explicit names, avoid "
          "global state, and never silently swallow errors. ")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--hybrid", default="/root/autodl-tmp/models/Qwen3.5-4B")
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--ks", default="0,2,3,4")
    ap.add_argument("--methods", default="ngram",
                    help="comma list of proposer methods to compare coverage")
    ap.add_argument("--draft-model", default="",
                    help="draft model path for eagle3-style methods")
    ap.add_argument("--targets", default="dense,hybrid")
    ap.add_argument("--out", default="/root/autodl-tmp/c1n_coverage.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    total_out = a.gen * a.n
    rows = {}

    for target in targets:
        path = a.dense if target == "dense" else a.hybrid
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        prompts = [{"prompt_token_ids": p}
                   for p in build_prompts(tok, a.n, a.prefix)]
        # one no-spec reference per target
        ref = None
        for meth in methods:
            for K in ks:
                if K == 0 and meth != methods[0]:
                    continue
                kw = dict(model=path, gpu_memory_utilization=a.gpu_util,
                          max_model_len=a.max_len, enable_prefix_caching=True,
                          enforce_eager=True, dtype="bfloat16",
                          language_model_only=True, disable_log_stats=False)
                tag = f"{target}|{meth}|K={K}"
                if K > 0:
                    sc = {"method": meth, "num_speculative_tokens": K}
                    if meth == "ngram":
                        sc.update(prompt_lookup_max=6, prompt_lookup_min=2)
                    elif a.draft_model:
                        sc["model"] = a.draft_model
                    kw["speculative_config"] = sc
                    kw["per_request_spec_decode_metrics"] = "detailed"
                try:
                    llm = LLM(**kw)
                except Exception as e:  # noqa: BLE001
                    print(f"[{tag}] ENGINE FAILED {type(e).__name__}: "
                          f"{str(e)[:110]}", flush=True)
                    rows[tag] = {"error": str(e)[:200]}
                    continue

                ps = []
                for _ in range(a.passes):
                    reset_steps()
                    c0 = snapshot()
                    t0 = time.perf_counter()
                    outs = llm.generate(prompts, sampling_params=sp,
                                        use_tqdm=False)
                    w = time.perf_counter() - t0
                    c1 = snapshot()
                    ps.append({
                        "wall": w, "file": nsteps(),
                        "dtok": g(c1, norm("vllm:spec_decode_num_draft_tokens")) -
                                g(c0, norm("vllm:spec_decode_num_draft_tokens")),
                        "atok": g(c1, norm("vllm:spec_decode_num_accepted_tokens")) -
                                g(c0, norm("vllm:spec_decode_num_accepted_tokens")),
                        **detail(outs)})

                warm = ps[-1]
                wall = st.median([q["wall"] for q in ps[1:]] or [ps[-1]["wall"]])
                fsteps = st.median([q["file"] for q in ps[1:]] or [ps[-1]["file"]])
                dps = warm["hist_steps"]            # drafting steps
                A_cond = (1 + warm["hist_acc"] / dps) if dps else None
                # every drafting step proposes exactly K -> verify the identity
                ident_ok = (warm["dtok"] == K * dps) if (K > 0 and dps) else None
                M = (dps / fsteps) if fsteps else None
                pred = ((1 - M) * 1 + M * A_cond) if (M and A_cond) else None
                tps = total_out / wall if wall else None
                if K == 0:
                    ref = tps
                row = {
                    "target": target, "method": meth, "K": K,
                    "wall_s": round(wall, 4), "tokens_per_s": round(tps, 2),
                    "steps_total": fsteps, "steps_drafting": dps,
                    "steps_nondraft": fsteps - dps,
                    "draft_tokens": warm["dtok"], "accepted": warm["hist_acc"],
                    "A_cond": round(A_cond, 4) if A_cond else None,
                    "coverage_M": round(M, 4) if M else None,
                    "dtok_eq_K_times_drafting": ident_ok,
                    "predicted_multiplier": round(pred, 4) if pred else None,
                    "measured_multiplier": (round(tps / ref, 4)
                                            if (ref and K > 0) else None),
                    "ms_per_step": round(wall / fsteps * 1000, 3) if fsteps else None,
                    "per_step_accepted": (warm["per_step_accepted"][:40]
                                          if warm["per_step_accepted"] else None),
                    "per_step_drafted": (warm["per_step_drafted"][:40]
                                         if warm["per_step_drafted"] else None),
                }
                rows[tag] = row
                print(f"[{tag}] tok/s={row['tokens_per_s']:<6} "
                      f"steps={fsteps:.0f} (draft {dps} / plain {fsteps-dps:.0f}) "
                      f"M={row['coverage_M']} A_cond={row['A_cond']} "
                      f"dtok={warm['dtok']} K*drafts={K*dps} ident={ident_ok} "
                      f"| pred={row['predicted_multiplier']} "
                      f"meas={row['measured_multiplier']}", flush=True)
                del llm
                import torch
                torch.cuda.empty_cache()
                time.sleep(2)

    print("\n=== does  speedup = (1-M) + M*A_cond  explain the measurement? ===")
    print(f"{'cell':<20}{'steps':>6}{'draft':>6}{'M':>7}{'A_cond':>8}"
          f"{'pred':>7}{'meas':>7}{'err%':>7}")
    for tag, r in rows.items():
        if r.get("error") or r["K"] == 0:
            continue
        err = (r["measured_multiplier"] / r["predicted_multiplier"] - 1) * 100 \
            if (r.get("measured_multiplier") and r.get("predicted_multiplier")) else None
        print(f"{tag:<20}{r['steps_total']:>6.0f}{r['steps_drafting']:>6}"
              f"{str(r['coverage_M']):>7}{str(r['A_cond']):>8}"
              f"{str(r['predicted_multiplier']):>7}{str(r['measured_multiplier']):>7}"
              f"{(f'{err:+.1f}' if err is not None else 'n/a'):>7}")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "rows": rows}, f, indent=2, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
