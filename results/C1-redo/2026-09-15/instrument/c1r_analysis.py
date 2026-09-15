#!/usr/bin/env python
"""C1 redo -- adjudicate prereg_C1_v2_2026-09-15.md against the raw launch JSONs.

Reads  results/raw/manifest_<grid>.json and every results/raw/<run_id>.json.
Writes results/analysis_<grid>.json and prints a human-readable verdict.

Nothing here is allowed to silently degrade: if an instrument is missing the
corresponding prediction is reported as INSTRUMENT_UNAVAILABLE, never as a
null result.  (That conflation is exactly what sank the first C-1 attempt.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

RAW = "/root/autodl-tmp/results/raw"


def load(grid):
    """Scope strictly to the runs named in this grid's manifest.

    Raw JSONs from every grid live in one directory, so globbing alone would mix
    e.g. the dense and hybrid grids into one analysis.
    """
    mpath = os.path.join(RAW, f"manifest_{grid}.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else None
    want = None
    if man:
        want = {l["run_id"] for l in man.get("launches", []) if l.get("run_id")}
    runs = {}
    for p in glob.glob(os.path.join(RAW, f"*.json")):
        if os.path.basename(p).startswith("manifest"):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "run_id" not in d or "arms" not in d:
            continue
        if want is not None and d["run_id"] not in want:
            continue
        runs[d["run_id"]] = d
    return runs, man


def warm_arms(run):
    out = defaultdict(dict)   # (seed, conc) -> {phase: arm}
    for a in run.get("arms", []):
        tag = a.get("tag", "")
        phase, c, s = tag.split("|")
        out[(int(s[1:]), int(c[1:]))][phase] = a
    return out


def noise_floor(runs):
    """sigma from identical warm replicates sent back-to-back on one engine."""
    cells = []
    for rid, run in runs.items():
        if run.get("engine_status") != "OK":
            continue
        for key, phases in warm_arms(run).items():
            seq = [phases[p]["decode_tps"] for p in sorted(phases)
                   if p.startswith("warm") and phases[p].get("decode_tps")]
            if len(seq) >= 3:
                cells.append({"run_id": rid, "seed": key[0], "conc": key[1],
                              "values": seq, "mean": st.mean(seq),
                              "stdev": st.stdev(seq)})
    if not cells:
        return None
    sigmas = [c["stdev"] / c["mean"] for c in cells if c["mean"]]
    return {
        "n_cells": len(cells),
        "sigma_rel_median": st.median(sigmas),
        "sigma_rel_max": max(sigmas),
        "cells": cells,
        "theta": max(1.05, 1 + 3 * st.median(sigmas)),
    }


def p2_accounting(runs):
    """P2': did the cache actually engage, and does the counter say so?"""
    rows = []
    for rid, run in runs.items():
        if run.get("engine_status") != "OK":
            continue
        pc = run.get("prefix_caching")
        for key, phases in sorted(warm_arms(run).items()):
            cold = phases.get("cold", {})
            warm = [phases[p] for p in sorted(phases) if p.startswith("warm")]
            w = warm[0] if warm else {}
            rows.append({
                "run_id": rid, "drafter": run.get("drafter"),
                "pc": pc, "seed": key[0], "conc": key[1],
                "cold_cached": cold.get("cached_tokens_sum"),
                "warm_cached": w.get("cached_tokens_sum"),
                "warm_hits_delta": w.get("cache_hits_delta"),
                "cold_hits_delta": cold.get("cache_hits_delta"),
                "n_req": w.get("n_requests"),
                # This vLLM build does not return
                # usage.prompt_tokens_details.cached_tokens, so the per-request
                # instrument is unavailable; P2' therefore rests on the engine's
                # own vllm:prefix_cache_hits_total delta, which IS exposed.
                "instrument": (w.get("cached_tokens_instrument")
                               or ("metrics:prefix_cache_hits_total"
                                   if w.get("cache_hits_delta") is not None else None)),
            })
    # gate
    gate = {"pc_on_warm_cells": 0, "pc_on_warm_hits_positive": 0,
            "pc_off_warm_cells": 0, "pc_off_warm_hits_zero": 0,
            "instrument_present": False}
    for r in rows:
        if r["instrument"]:
            gate["instrument_present"] = True
            gate.setdefault("instruments", set()).add(r["instrument"])
        hits = r["warm_hits_delta"]
        if r["pc"]:
            gate["pc_on_warm_cells"] += 1
            if hits is not None and hits > 0:
                gate["pc_on_warm_hits_positive"] += 1
        else:
            gate["pc_off_warm_cells"] += 1
            if hits is not None and hits == 0:
                gate["pc_off_warm_hits_zero"] += 1
    gate["instruments"] = sorted(gate.get("instruments", set()))
    gate["P2_PASS"] = (gate["pc_on_warm_cells"] > 0
                       and gate["pc_on_warm_hits_positive"] == gate["pc_on_warm_cells"]
                       and gate["pc_off_warm_cells"] > 0
                       and gate["pc_off_warm_hits_zero"] == gate["pc_off_warm_cells"])
    return {"rows": rows, "gate": gate}


def _seq(corr_entry, instrument):
    recs = sorted(corr_entry["requests"], key=lambda r: r["idx"])
    if instrument == "token_ids":
        return [tuple(r["token_ids"]) if r.get("token_ids") else None for r in recs]
    return [r.get("text") for r in recs]


def p1_correctness(runs):
    """P1': greedy output must be identical across cache state / PC setting."""
    by_cell = defaultdict(dict)     # (target, drafter, k) -> run_id -> corr list
    for rid, run in runs.items():
        if run.get("engine_status") != "OK" or "correctness" not in run:
            continue
        key = (run["target"], run["drafter"], run["k"])
        by_cell[key][run["prefix_caching"]] = run

    out = []
    for key, pair in sorted(by_cell.items()):
        if 1 not in pair or 0 not in pair:
            out.append({"cell": key, "status": "INCOMPLETE_PAIR",
                        "have_pc": sorted(pair)})
            continue
        for on, off in ((pair[1], pair[0]),):
            instr = on.get("correctness_instrument") or off.get("correctness_instrument")
            if not instr:
                out.append({"cell": key, "status": "INSTRUMENT_UNAVAILABLE"})
                continue
            within, cross, total = 0, 0, 0
            details = []
            on_c = {(c["seed"], c["rep"]): c for c in on["correctness"]}
            off_c = {(c["seed"], c["rep"]): c for c in off["correctness"]}
            for kk in sorted(set(on_c) & set(off_c)):
                a = _seq(on_c[kk], instr)
                b = _seq(off_c[kk], instr)
                for i, (x, y) in enumerate(zip(a, b)):
                    if x is None or y is None:
                        continue
                    total += 1
                    if x != y:
                        cross += 1
                        if len(details) < 8:
                            details.append({"seed": kk[0], "rep": kk[1], "idx": i,
                                            "pc_on": list(x)[:12], "pc_off": list(y)[:12]})
            # cache transparency WITHIN the pc-on run: cold rep1 vs warm rep2
            for sd in sorted({c["seed"] for c in on["correctness"]}):
                e1 = next((c for c in on["correctness"] if c["seed"] == sd and c["rep"] == 1), None)
                e2 = next((c for c in on["correctness"] if c["seed"] == sd and c["rep"] == 2), None)
                if not e1 or not e2:
                    continue
                s1, s2 = _seq(e1, instr), _seq(e2, instr)
                for x, y in zip(s1, s2):
                    if x is not None and y is not None and x != y:
                        within += 1
            out.append({"cell": key, "instrument": instr, "n_compared": total,
                        "cross_pc_divergences": cross,
                        "within_pc_on_divergences": within,
                        "examples": details,
                        "status": "OK"})
    n_ok = [c for c in out if c.get("status") == "OK"]
    n_div = [c for c in n_ok if c["cross_pc_divergences"] or c["within_pc_on_divergences"]]
    none_div = [c for c in n_ok if c["cell"][1] == "none"
                and (c["cross_pc_divergences"] or c["within_pc_on_divergences"])]
    return {"cells": out,
            "verdict": ("INSTRUMENT_UNAVAILABLE" if not n_ok else
                        "S2_DEVICE_BROKEN" if none_div else
                        "P1_DIVERGENCE" if n_div else "P1_HELD"),
            "n_cells": len(n_ok), "n_divergent_cells": len(n_div),
            "none_arm_divergent": len(none_div)}


def p3_cost(runs, theta):
    """P3': decode-phase throughput, PC on vs off, on warm arms only."""
    agg = defaultdict(lambda: defaultdict(list))
    for rid, run in runs.items():
        if run.get("engine_status") != "OK":
            continue
        key = (run["target"], run["drafter"], run["k"])
        for (seed, conc), phases in warm_arms(run).items():
            vals = [phases[p]["decode_tps"] for p in sorted(phases)
                    if p.startswith("warm") and phases[p].get("decode_tps")]
            if vals:
                agg[key][run["prefix_caching"]].append({"seed": seed, "conc": conc,
                                                        "tps": st.mean(vals)})
    rows = []
    for key, d in sorted(agg.items()):
        on, off = d.get(1, []), d.get(0, [])
        if not on or not off:
            rows.append({"cell": key, "status": "INCOMPLETE_PAIR"})
            continue
        mo, mf = st.mean(x["tps"] for x in on), st.mean(x["tps"] for x in off)
        ratio = mo / mf
        # per-seed direction
        so = defaultdict(list); sf = defaultdict(list)
        for x in on: so[x["seed"]].append(x["tps"])
        for x in off: sf[x["seed"]].append(x["tps"])
        seed_ratios = {s: st.mean(so[s]) / st.mean(sf[s])
                       for s in sorted(set(so) & set(sf))}
        same_dir = all(r > 1 for r in seed_ratios.values()) or \
                   all(r < 1 for r in seed_ratios.values())
        rows.append({
            "cell": key, "pc_on_tps": mo, "pc_off_tps": mf, "ratio_on_over_off": ratio,
            "exceeds_theta": abs(ratio - 1) > (theta - 1),
            "seed_ratios": seed_ratios, "direction_consistent": same_dir,
            "significant": abs(ratio - 1) > (theta - 1) and same_dir,
        })
    return {"theta": theta, "cells": rows,
            "any_significant": [r["cell"] for r in rows if r.get("significant")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="core-dense")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs, man = load(a.grid)
    print(f"loaded {len(runs)} launch JSONs for grid={a.grid}")
    failed = {k: v.get("engine_error") for k, v in runs.items()
              if v.get("engine_status") != "OK"}
    if failed:
        print(f"ENGINE FAILURES ({len(failed)}):")
        for k, v in failed.items():
            print(f"   {k}: {str(v)[:110]}")

    nf = noise_floor(runs)
    theta = nf["theta"] if nf else 1.15
    p2 = p2_accounting(runs)
    p1 = p1_correctness(runs)
    p3 = p3_cost(runs, theta)

    print("\n=== NOISE FLOOR (identical warm replicates, same engine) ===")
    if nf:
        print(f"  cells={nf['n_cells']}  sigma/mu median={nf['sigma_rel_median']:.4f} "
              f"max={nf['sigma_rel_max']:.4f}  -> theta={nf['theta']:.4f}"
              f"   (S3 sigma/mu>15%? {nf['sigma_rel_median'] > 0.15})")
    else:
        print("  UNAVAILABLE")

    print("\n=== P2' cache accounting gate ===")
    print(f"  {p2['gate']}")

    print("\n=== P1' correctness invariant ===")
    print(f"  verdict={p1['verdict']} cells={p1['n_cells']} "
          f"divergent={p1['n_divergent_cells']} none_arm_divergent={p1['none_arm_divergent']}")
    for c in p1["cells"]:
        if c.get("status") == "OK" and (c["cross_pc_divergences"] or c["within_pc_on_divergences"]):
            print(f"    DIVERGENT {c['cell']}: cross={c['cross_pc_divergences']} "
                  f"within={c['within_pc_on_divergences']} / {c['n_compared']}")

    print("\n=== P3' decode throughput, PC on vs off (warm) ===")
    print(f"  theta={p3['theta']:.4f}")
    for r in p3["cells"]:
        if r.get("status"):
            print(f"  {str(r['cell']):44s} {r['status']}")
        else:
            print(f"  {str(r['cell']):44s} on={r['pc_on_tps']:8.2f} off={r['pc_off_tps']:8.2f} "
                  f"ratio={r['ratio_on_over_off']:.4f} "
                  f"{'SIGNIFICANT' if r['significant'] else ''}")
    print(f"\n  significant cells: {p3['any_significant']}")

    res = {"grid": a.grid, "n_runs": len(runs), "engine_failures": failed,
           "noise_floor": nf, "p1_correctness": p1, "p2_accounting": p2,
           "p3_cost": p3, "manifest": man}
    out = a.out or f"/root/autodl-tmp/results/analysis_{a.grid}.json"
    json.dump(res, open(out, "w"), indent=2, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
