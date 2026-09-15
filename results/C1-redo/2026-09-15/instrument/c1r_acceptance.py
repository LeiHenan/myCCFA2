#!/usr/bin/env python
"""C1 redo -- acceptance-rate instrument (the DIRECT test of the upstream mechanism).

WHY THIS EXISTS, and why P1' alone is not enough:

The pre-registration's P1' was "PC-on and PC-off greedy output must be identical".
That is a real invariant and it is worth testing, but it CANNOT detect the specific
upstream mechanism this re-run was chasing.  Reason: speculative decoding is
LOSSLESS.  The target model verifies every draft, so a corrupted drafter changes
only WHICH drafts get proposed, never the emitted token.  A drafter reading
uninitialised KV could therefore corrupt acceptance while leaving the output
bit-identical -- exactly the case P1' would pass while the bug is present.

The quantity that the upstream claim actually predicts is the ACCEPTANCE RATE on a
genuine cache hit.  So this script reports, per (target, drafter, K, seed, conc):

    acceptance_rate   = accepted_tokens / draft_tokens
    mean_accept_len   = 1 + accepted_tokens / drafts
    drafts, draft_tokens, accepted_tokens

and contrasts three cache states with the SAME prompts and the SAME drafter:
    cold (PC on, first send) / warm (PC on, cache hit) / PC off (no cache at all)

If the cache corrupts the drafter's context, the warm arm is where it must show up.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

RAW = "/root/autodl-tmp/results/raw"


def load(grid=None):
    want = None
    if grid:
        mpath = os.path.join(RAW, f"manifest_{grid}.json")
        if os.path.exists(mpath):
            m = json.load(open(mpath))
            want = {l["run_id"] for l in m.get("launches", []) if l.get("run_id")}
    runs = {}
    for p in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        if os.path.basename(p).startswith("manifest"):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not d.get("run_id") or d.get("engine_status") != "OK":
            continue
        if want is not None and d["run_id"] not in want:
            continue
        runs[d["run_id"]] = d
    return runs


def acc_from(arm):
    m = arm.get("metrics_delta") or {}
    drafts = m.get("vllm:spec_decode_num_drafts_total")
    dtok = m.get("vllm:spec_decode_num_draft_tokens_total")
    atok = m.get("vllm:spec_decode_num_accepted_tokens_total")
    if drafts is None or dtok in (None, 0):
        return None
    return {
        "drafts": drafts, "draft_tokens": dtok, "accepted_tokens": atok,
        "acceptance_rate": (atok / dtok) if atok is not None else None,
        "mean_accept_len": (1 + atok / drafts) if (atok is not None and drafts) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=None,
                    help="restrict to the runs named in manifest_<grid>.json")
    ap.add_argument("--out", default="/root/autodl-tmp/results/acceptance.json")
    a = ap.parse_args()
    runs = load(a.grid)

    rows = []
    for rid, run in runs.items():
        if run.get("drafter") in (None, "none"):
            continue
        for arm in run.get("arms", []):
            tag = arm.get("tag", "")
            try:
                phase, c, s = tag.split("|")
            except ValueError:
                continue
            acc = acc_from(arm)
            if not acc:
                continue
            rows.append({
                "target": run["target"], "drafter": run["drafter"], "k": run["k"],
                "pc": int(run["prefix_caching"]), "seed": int(s[1:]),
                "conc": int(c[1:]), "phase": phase,
                "hits": arm.get("cache_hits_delta"),
                "decode_tps": arm.get("decode_tps"),
                **acc,
            })

    print(f"# acceptance rows: {len(rows)} from {len(runs)} runs\n")

    # ---- contrast 1: PC on warm vs PC off warm, same drafter -------------
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["phase"].startswith("warm"):
            agg[(r["target"], r["drafter"], r["k"])][r["pc"]].append(r)

    print("## A. 暖臂接受率：PC on vs PC off（同一 drafter、同一 prompt）\n")
    print("| target | drafter | K | PC-on acc | PC-off acc | on/off | PC-on mean len | PC-off mean len | draft_tokens(on/off) |")
    print("|---|---|---|---|---|---|---|---|---|")
    out = {"warm_pc_contrast": []}
    for key, d in sorted(agg.items()):
        on, off = d.get(1, []), d.get(0, [])
        if not on or not off:
            continue
        ao = st.mean(x["acceptance_rate"] for x in on if x["acceptance_rate"] is not None)
        af = st.mean(x["acceptance_rate"] for x in off if x["acceptance_rate"] is not None)
        lo = st.mean(x["mean_accept_len"] for x in on if x["mean_accept_len"])
        lf = st.mean(x["mean_accept_len"] for x in off if x["mean_accept_len"])
        to = sum(x["draft_tokens"] for x in on)
        tf = sum(x["draft_tokens"] for x in off)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {ao:.4f} | {af:.4f} | "
              f"{ao/af:.4f} | {lo:.3f} | {lf:.3f} | {to:.0f}/{tf:.0f} |")
        # per-seed direction
        so = defaultdict(list); sf = defaultdict(list)
        for x in on:
            if x["acceptance_rate"] is not None:
                so[x["seed"]].append(x["acceptance_rate"])
        for x in off:
            if x["acceptance_rate"] is not None:
                sf[x["seed"]].append(x["acceptance_rate"])
        sr = {s: st.mean(so[s]) / st.mean(sf[s]) for s in sorted(set(so) & set(sf)) if st.mean(sf[s])}
        out["warm_pc_contrast"].append({
            "cell": key, "pc_on_acc": ao, "pc_off_acc": af, "ratio": ao / af,
            "pc_on_len": lo, "pc_off_len": lf, "seed_ratios": sr,
            "direction_consistent": (all(v > 1 for v in sr.values())
                                     or all(v < 1 for v in sr.values())) if sr else None,
        })

    # ---- contrast 2: within PC on, cold vs warm (the cache-hit path) -----
    print("\n## B. PC 开启时：冷臂 vs 暖臂接受率（缓存命中路径，上游 bug 应在此显形）\n")
    print("| target | drafter | K | cold acc | warm acc | warm/cold | cold hits | warm hits |")
    print("|---|---|---|---|---|---|---|---|")
    byc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["pc"] == 1:
            byc[(r["target"], r["drafter"], r["k"])][r["phase"]].append(r)
    out["cold_warm_contrast"] = []
    for key, d in sorted(byc.items()):
        cold = d.get("cold", [])
        warm = [x for p, v in d.items() if p.startswith("warm") for x in v]
        if not cold or not warm:
            continue
        ac = st.mean(x["acceptance_rate"] for x in cold if x["acceptance_rate"] is not None)
        aw = st.mean(x["acceptance_rate"] for x in warm if x["acceptance_rate"] is not None)
        hc = st.mean(x["hits"] for x in cold if x["hits"] is not None)
        hw = st.mean(x["hits"] for x in warm if x["hits"] is not None)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {ac:.4f} | {aw:.4f} | {aw/ac:.4f} | "
              f"{hc:.0f} | {hw:.0f} |")
        out["cold_warm_contrast"].append({
            "cell": key, "cold_acc": ac, "warm_acc": aw, "ratio": aw / ac,
            "cold_hits_mean": hc, "warm_hits_mean": hw})

    json.dump(out, open(a.out, "w"), indent=2, default=str)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
