#!/usr/bin/env python
"""C-1l: GROUND-TRUTH engine step count, then refit the per-draft marginal cost.

Motivation (self-audit chain)
-----------------------------
C-1j used  steps = sum(histogram)  as the denominator of t_step, getting
dense slope 1.264 / hybrid 7.075 ms per extra draft token.

C-1k falsified that denominator three ways:
  1. identity  A * steps = generated tokens  fails badly
     (48-token generations report 3-6 steps, i.e. A*steps = 7-12 tokens);
  2. the engine source only feeds the per-request accumulator under
     `if scheduled_spec_token_ids and generated_token_ids` -- VERIFY steps;
  3. at K=1 the reported A was 1.49, i.e. more tokens accepted than drafted.

So the per-request histogram is a *window*, not a total.  This probe therefore
stops trusting it and instruments the engine itself:

    EngineCore.step / step_with_batch_queue  ->  append a tick to a counter file

That is the true number of engine iterations (each iteration = one scheduler
schedule() + one model execute + one update_from_output, for the whole batch).
With n=1 exactly one sequence is in flight, so iterations == that sequence's
steps.

Cross-checks (all reported per cell):
    steps_file   : ticks from the patched loop          <- GROUND TRUTH
    steps_iter   : vllm:iteration_tokens_total count    ("tokens per engine_step")
    steps_drafts : vllm:spec_decode_num_drafts          (one inc per iteration)
    steps_hist   : sum(per-request histogram)           (the C-1j denominator)
    identity     : A_eng * steps_eng == generated tokens

K=0 (no speculation) is a valid arm here: the counter file and the iteration
metric do not need spec decoding, so the baseline is measurable.

Then refit   t_step(K) = intercept + slope*K   under each definition and report
which one is self-consistent.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

STEP_FILE = os.environ.get("C1L_STEP_FILE", "/tmp/c1l_steps.txt")

# --- engine instrumentation -------------------------------------------------
# Must run in BOTH the parent and the spawned EngineCore process, so it is
# executed at import time, guarded by the env var.
if os.environ.get("C1L_PATCH") == "1":
    try:
        import vllm.v1.engine.core as _core

        if not getattr(_core.EngineCore, "_c1l_patched", False):
            _orig_step = _core.EngineCore.step
            _orig_swbq = _core.EngineCore.step_with_batch_queue

            def _tick():
                try:
                    with open(STEP_FILE, "a") as f:
                        f.write("1")
                except Exception:
                    pass

            def _step(self, *a, **k):
                _tick()
                return _orig_step(self, *a, **k)

            def _step_swbq(self, *a, **k):
                _tick()
                return _orig_swbq(self, *a, **k)

            _core.EngineCore.step = _step
            _core.EngineCore.step_with_batch_queue = _step_swbq
            _core.EngineCore._c1l_patched = True
    except Exception as _e:  # noqa: BLE001
        print(f"[c1l] engine patch failed: {_e}", flush=True)


def steps_file_count() -> int:
    try:
        return os.path.getsize(STEP_FILE)
    except OSError:
        return 0


def reset_step_file():
    try:
        open(STEP_FILE, "w").close()
    except OSError:
        pass


def snapshot():
    """name -> summed value over all label sets (Counter/Gauge/Histogram)."""
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
        if hasattr(m, "value"):
            v = m.value
        elif hasattr(m, "count"):          # Histogram: observation count
            v = m.count
        else:
            continue
        try:
            out[n] = out.get(n, 0.0) + float(v)
        except (TypeError, ValueError):
            pass
    return out


def norm(name: str) -> str:
    """Prometheus appends _total for counters; accept both spellings."""
    return name[:-6] if name.endswith("_total") else name


def get(d: dict, *names):
    for nm in names:
        if nm in d:
            return d[nm]
    return 0.0


def per_request_hist(outs):
    acc = steps = drafts = 0
    for o in outs:
        for so in o.outputs:
            m = getattr(so, "spec_decode_metrics", None)
            if m is None:
                continue
            h = list(getattr(m, "histogram", []) or [])
            acc += sum(j * c for j, c in enumerate(h))
            steps += sum(h)
            drafts += int(getattr(m, "num_draft_tokens", 0) or 0)
    return acc, steps, drafts


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
    ap.add_argument("--ks", default="0,1,2,3,4,6,8")
    ap.add_argument("--targets", default="dense,hybrid")
    ap.add_argument("--out", default="/root/autodl-tmp/c1l_ground_truth.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)
    rows = {}
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]

    s0 = snapshot()
    print(f"[c1l] patched={os.environ.get('C1L_PATCH')} metrics_keys="
          f"{len(s0)} stepfile={STEP_FILE} "
          f"iter_metric_present="
          f"{norm('vllm:iteration_tokens_total') in {norm(k) for k in s0}}",
          flush=True)

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
                      language_model_only=True)
            if K > 0:
                kw["speculative_config"] = {
                    "method": "ngram", "num_speculative_tokens": K,
                    "prompt_lookup_max": 6, "prompt_lookup_min": 2}
                kw["per_request_spec_decode_metrics"] = "detailed"
            try:
                llm = LLM(**kw)
            except Exception as e:  # noqa: BLE001
                print(f"[{target} K={K}] ENGINE FAILED {type(e).__name__}: "
                      f"{str(e)[:110]}", flush=True)
                rows[f"{target}|K={K}"] = {"error": str(e)[:200]}
                continue

            walls, hists = [], []
            for p in range(a.passes):
                reset_step_file()
                c0 = snapshot()
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
                w = time.perf_counter() - t0
                c1 = snapshot()
                fsteps = steps_file_count()
                d_iter = get(c1, norm("vllm:iteration_tokens_total")) - \
                    get(c0, norm("vllm:iteration_tokens_total"))
                d_drafts = get(c1, norm("vllm:spec_decode_num_drafts")) - \
                    get(c0, norm("vllm:spec_decode_num_drafts"))
                d_acc = get(c1, norm("vllm:spec_decode_num_accepted_tokens")) - \
                    get(c0, norm("vllm:spec_decode_num_accepted_tokens"))
                d_dtok = get(c1, norm("vllm:spec_decode_num_draft_tokens")) - \
                    get(c0, norm("vllm:spec_decode_num_draft_tokens"))
                acc_h, steps_h, drafts_h = per_request_hist(outs)
                # first pass pays JIT/compile; keep it but also record later ones
                walls.append((p, w))
                hists.append({"p": p, "file": fsteps, "iter": d_iter,
                              "drafts": d_drafts, "acc": d_acc, "dtok": d_dtok,
                              "hist_steps": steps_h, "hist_acc": acc_h,
                              "hist_drafts": drafts_h})
            # use the last pass (warm) for timing; report all for transparency
            warm = [w for p, w in walls if p == max(p for p, _ in walls)][0]
            wall = st.median([w for p, w in walls[1:]]) if len(walls) > 1 else warm
            h = hists[-1]
            A_eng = (1 + h["acc"] / h["drafts"]) if h["drafts"] else None
            A_hist = (1 + h["hist_acc"] / h["hist_steps"]) if h["hist_steps"] else None

            def tstep(s):
                return round(wall / s * 1000, 4) if s else None

            row = {
                "K": K, "wall_s": round(wall, 4), "total_out": total_out,
                "steps_file": h["file"], "steps_iter": h["iter"],
                "steps_drafts": h["drafts"], "steps_hist": h["hist_steps"],
                "A_eng": round(A_eng, 4) if A_eng else None,
                "A_hist": round(A_hist, 4) if A_hist else None,
                "acc_eng": h["acc"], "draft_tokens_eng": h["dtok"],
                "t_step_ms_file": tstep(h["file"]),
                "t_step_ms_iter": tstep(h["iter"]),
                "t_step_ms_hist": tstep(h["hist_steps"]),
                "ident_file": round(A_eng * h["file"], 1) if A_eng else None,
                "ident_iter": round(A_eng * h["iter"], 1) if A_eng else None,
                "ident_hist": round(A_hist * h["hist_steps"], 1) if A_hist else None,
                "tokens_per_s": round(total_out / wall, 1) if wall else None,
                "all_passes": [{"p": p, "wall": round(w, 4)} for p, w in walls],
            }
            rows[f"{target}|K={K}"] = row
            print(f"[{target} K={K}] tok/s={row['tokens_per_s']:<6} "
                  f"wall={wall:.3f} | steps: file={h['file']} "
                  f"iter={h['iter']:.0f} drafts={h['drafts']:.0f} "
                  f"hist={h['hist_steps']} | A_eng={row['A_eng']} "
                  f"A_hist={row['A_hist']} | ident file={row['ident_file']} "
                  f"iter={row['ident_iter']} hist={row['ident_hist']} "
                  f"(true {total_out}) | t_step file={row['t_step_ms_file']} "
                  f"iter={row['t_step_ms_iter']}", flush=True)
            del llm
            import torch
            torch.cuda.empty_cache()
            time.sleep(2)

    print("\n=== which step count satisfies  A * steps == generated tokens? ===")
    print(f"{'cell':<13}{'true':>6}{'file':>6}{'i_file':>8}{'iter':>6}"
          f"{'i_iter':>8}{'hist':>6}{'i_hist':>8}")
    for target in targets:
        for K in ks:
            r = rows.get(f"{target}|K={K}") or {}
            if r.get("error"):
                print(f"{target}|K={K:<9} ENGINE FAILED")
                continue
            print(f"{target + '|K=' + str(K):<13}{r['total_out']:>6}"
                  f"{r['steps_file']:>6}{str(r['ident_file']):>8}"
                  f"{r['steps_iter']:>6.0f}{str(r['ident_iter']):>8}"
                  f"{r['steps_hist']:>6}{str(r['ident_hist']):>8}")

    print("\n=== refit  t_step = intercept + slope*K ===")
    for defn in ("t_step_ms_file", "t_step_ms_iter", "t_step_ms_hist"):
        print(f"-- {defn}")
        for target in targets:
            pts = [(r["K"], r[defn]) for r in
                   (rows.get(f"{target}|K={K}") or {} for K in ks)
                   if r.get(defn) and r.get("K") is not None]
            if len(pts) < 3:
                continue
            n = len(pts)
            sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
            sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
            den = n * sxx - sx * sx
            slope = (n * sxy - sx * sy) / den if den else float("nan")
            inter = (sy - slope * sx) / n if n else float("nan")
            print(f"   {target:<7} t_step = {inter:7.3f} + {slope:6.3f}*K  (n={n})")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "rows": rows}, f, indent=2, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
