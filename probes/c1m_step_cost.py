#!/usr/bin/env python
"""C-1m: the fixed per-decode-step cost of a hybrid stack, measured exactly.

What C-1l established (ground truth)
------------------------------------
Patching EngineCore.step / step_with_batch_queue and ticking a file gives the
true engine-iteration count.  For a 96-token generation it reports 88-99 steps
(and with no speculation at all, exactly 96) -- i.e. it is right to within the
prefill overhead.  Under the same conditions the per-request histogram reports
6-21 "steps".  So:

    sum(histogram) is a WINDOW, not a step count, and every t_step built on it
    (C-1j, and the C-1e A* normalisation) is invalid.

Refitting with the true counter reversed the C-1j conclusion: the per-draft
SLOPE is not larger on hybrid (dense +0.53, hybrid +0.33 ms/draft).  What is
larger on hybrid is the INTERCEPT -- the cost of a decode step that does not
depend on how many draft tokens ride along:

    dense   t_step = 21.3 + 0.53*K   ms
    hybrid  t_step = 33.5 + 0.33*K   ms      -> intercept x1.57

This probe measures that fixed per-step cost cleanly:

  * `disable_log_stats` must be forced False -- vllm/entrypoints/llm.py:228
    otherwise sets it True, which is why the Prometheus counters read 0.
  * the step file (patched loop) is the primary step count; the
    vllm:iteration_tokens_total observation count is the cross-check.
  * both a K=0 (no speculation) and a K>0 arm, so "cost per step" is defined
    without reference to any acceptance model.
  * the same prompt/prefix on both models, so the comparison is architecture
    only.

Reads out: ms per engine step, tokens per step, and tok/s, for each
(architecture, K).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# Optional local workaround for upstream #47635 (Falcon-H1 only): forward the
# draft depth into mamba2_state_shape so speculation can initialize at all.
if os.environ.get("C1M_FALCON_PATCH") == "1":
    try:
        import sys
        sys.path.insert(0, "/root/autodl-tmp")
        import falcon_patch  # noqa: F401  (applies the classmethod patch)
    except Exception as _e:  # noqa: BLE001
        print(f"[c1m] falcon_patch failed: {_e}", flush=True)

STEP_FILE = os.environ.get("C1M_STEP_FILE", "/tmp/c1m_steps.txt")

if os.environ.get("C1M_PATCH") == "1":
    try:
        import vllm.v1.engine.core as _core

        if not getattr(_core.EngineCore, "_c1m_patched", False):
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
            _core.EngineCore._c1m_patched = True
    except Exception as _e:  # noqa: BLE001
        print(f"[c1m] engine patch failed: {_e}", flush=True)


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
        elif hasattr(m, "count"):
            v = m.count
        else:
            continue
        try:
            out[n] = out.get(n, 0.0) + float(v)
        except (TypeError, ValueError):
            pass
    return out


def norm(n: str) -> str:
    return n[:-6] if n.endswith("_total") else n


def g(d, *names):
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
    A = (1 + acc / steps) if steps else None
    return {"hist_acc": acc, "hist_steps": steps, "hist_drafts": drafts,
            "A_hist": round(A, 4) if A else None}


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
    ap.add_argument("--ks", default="0,2,3,4")
    ap.add_argument("--targets", default="dense,hybrid")
    ap.add_argument("--models", default="",
                    help="comma list of model paths, overrides dense/hybrid")
    ap.add_argument("--prefix-caching", type=int, default=1,
                    help="0 disables prefix caching (needed to dodge vLLM #47635)")
    ap.add_argument("--out", default="/root/autodl-tmp/c1m_step_cost.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True)
    rows = {}
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    total_out = a.gen * a.n

    s0 = snapshot()
    keys = {norm(k) for k in s0}
    print(f"[c1m] patched={os.environ.get('C1M_PATCH')} metric_keys={len(s0)} "
          f"iter_metric={norm('vllm:iteration_tokens_total') in keys} "
          f"drafts_metric={norm('vllm:spec_decode_num_drafts') in keys}",
          flush=True)

    paths = {}
    if a.models:
        for spec in a.models.split(","):
            if "=" in spec:
                nm, pth = spec.split("=", 1)
                paths[nm.strip()] = pth.strip()
        targets = list(paths)

    for target in targets:
        path = paths.get(target) or (a.dense if target == "dense" else a.hybrid)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        prompts = [{"prompt_token_ids": p}
                   for p in build_prompts(tok, a.n, a.prefix)]

        for K in ks:
            kw = dict(model=path, gpu_memory_utilization=a.gpu_util,
                      max_model_len=a.max_len,
                      enable_prefix_caching=bool(a.prefix_caching),
                      enforce_eager=True, dtype="bfloat16",
                      language_model_only=True,
                      disable_log_stats=False)
            if K > 0:
                kw["speculative_config"] = {
                    "method": "ngram", "num_speculative_tokens": K,
                    "prompt_lookup_max": 6, "prompt_lookup_min": 2}
                kw["per_request_spec_decode_metrics"] = "detailed"
            try:
                llm = LLM(**kw)
            except Exception as e:  # noqa: BLE001
                print(f"[{target} K={K}] ENGINE FAILED {type(e).__name__}: "
                      f"{str(e)[:130]}", flush=True)
                rows[f"{target}|K={K}"] = {"error": str(e)[:250]}
                continue

            passes = []
            for p in range(a.passes):
                reset_step_file()
                c0 = snapshot()
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
                w = time.perf_counter() - t0
                c1 = snapshot()
                passes.append({
                    "p": p, "wall": w,
                    "file": steps_file_count(),
                    "iter": g(c1, norm("vllm:iteration_tokens_total")) -
                            g(c0, norm("vllm:iteration_tokens_total")),
                    "drafts": g(c1, norm("vllm:spec_decode_num_drafts")) -
                              g(c0, norm("vllm:spec_decode_num_drafts")),
                    "dtok": g(c1, norm("vllm:spec_decode_num_draft_tokens")) -
                            g(c0, norm("vllm:spec_decode_num_draft_tokens")),
                    "atok": g(c1, norm("vllm:spec_decode_num_accepted_tokens")) -
                            g(c0, norm("vllm:spec_decode_num_accepted_tokens")),
                    **per_request_hist(outs)})

            warm = passes[-1]
            walls = [q["wall"] for q in passes[1:]] or [passes[-1]["wall"]]
            wall = st.median(walls)
            fsteps = st.median([q["file"] for q in passes[1:]])
            isteps = st.median([q["iter"] for q in passes[1:]])
            dtok = warm["dtok"]
            atok = warm["atok"]
            drafts = warm["drafts"]
            A_eng = (1 + atok / drafts) if drafts else None
            acc_rate = (atok / dtok) if dtok else None
            denom = fsteps or isteps
            row = {
                "K": K, "wall_s": round(wall, 4), "total_out": total_out,
                "steps_file": fsteps, "steps_iter": isteps,
                "steps_hist": warm["hist_steps"],
                "ms_per_step": round(wall / denom * 1000, 4) if denom else None,
                "tok_per_step": round(total_out / denom, 4) if denom else None,
                "tokens_per_s": round(total_out / wall, 1) if wall else None,
                "A_eng": round(A_eng, 4) if A_eng else None,
                "accept_rate": round(acc_rate, 4) if acc_rate else None,
                "draft_tokens": dtok, "accepted_tokens": atok,
                "A_hist": warm["A_hist"], "steps_hist_window": warm["hist_steps"],
                "ident_file": round(A_eng * fsteps, 1) if (A_eng and fsteps) else None,
                "passes": [{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in q.items()} for q in passes],
            }
            rows[f"{target}|K={K}"] = row
            print(f"[{target} K={K}] pc={a.prefix_caching} tok/s={row['tokens_per_s']:<6} "
                  f"steps_file={fsteps} steps_iter={isteps} "
                  f"hist_window={warm['hist_steps']} "
                  f"ms/step={row['ms_per_step']} "
                  f"tok/step={row['tok_per_step']} "
                  f"A_eng={row['A_eng']} acc={row['accept_rate']} "
                  f"ident={row['ident_file']} (true {total_out})", flush=True)
            del llm
            import torch
            torch.cuda.empty_cache()
            time.sleep(2)

    print("\n=== per (arch,K): ms per engine step (file counter) ===")
    print(f"{'cell':<13}{'tok/s':>8}{'steps':>7}{'ms/step':>9}{'tok/step':>9}"
          f"{'A_eng':>7}{'acc':>7}")
    for target in targets:
        for K in ks:
            r = rows.get(f"{target}|K={K}") or {}
            if r.get("error"):
                print(f"{target}|K={K:<9} ENGINE FAILED")
                continue
            print(f"{target + '|K=' + str(K):<13}{r['tokens_per_s']:>8}"
                  f"{r['steps_file']:>7.0f}{r['ms_per_step']:>9.2f}"
                  f"{r['tok_per_step']:>9.3f}"
                  f"{str(r['A_eng']):>7}{str(r['accept_rate']):>7}")

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "rows": rows}, f, indent=2, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
