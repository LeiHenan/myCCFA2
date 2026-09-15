#!/usr/bin/env python
"""C1 redo -- consolidate every grid's adjudication into one markdown report.

Reads results/analysis_<grid>.json for each grid that has been analysed and
emits a single markdown document with the pre-registered predictions answered
one by one, plus the raw numbers behind each answer.

Deliberately explicit about what could NOT be measured: the failure mode being
guarded against is the v1 habit of reporting an unmeasured quantity as if it
were a null result.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

RES = "/root/autodl-tmp/results"


def load_analyses():
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, "analysis_*.json"))):
        grid = os.path.basename(p)[len("analysis_"):-len(".json")]
        try:
            out[grid] = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            print(f"skip {p}: {e}")
    return out


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "report_data.md"))
    a = ap.parse_args()
    A = load_analyses()
    L = []
    w = L.append

    w("# C1 重做：数据汇总（自动生成）\n")
    w(f"grids analysed: {', '.join(sorted(A)) or '(none)'}\n")

    # ---------- engine failures ----------
    w("\n## 0. 装置失败（与「效应为零」严格区分）\n")
    any_fail = False
    for g, d in sorted(A.items()):
        for rid, err in (d.get("engine_failures") or {}).items():
            any_fail = True
            w(f"- `{rid}` ({g}): {str(err)[:200]}")
    if not any_fail:
        w("- 无")

    # ---------- noise floor ----------
    w("\n## 1. 噪声底（同一引擎内的相同暖臂重复）\n")
    w("| grid | cells | σ/μ 中位 | σ/μ 最大 | θ = max(1.05, 1+3σ/μ) | S3 (σ/μ>15%) |")
    w("|---|---|---|---|---|---|")
    thetas = []
    for g, d in sorted(A.items()):
        nf = d.get("noise_floor")
        if not nf:
            w(f"| {g} | 0 | n/a | n/a | 1.05 (fallback) | n/a |")
            continue
        thetas.append(nf["theta"])
        w(f"| {g} | {nf['n_cells']} | {fmt(nf['sigma_rel_median'],4)} | "
          f"{fmt(nf['sigma_rel_max'],4)} | {fmt(nf['theta'],4)} | "
          f"{'YES' if nf['sigma_rel_median']>0.15 else 'no'} |")
    theta = max(thetas) if thetas else 1.15

    # ---------- P2' ----------
    w("\n## 2. P2' 命中记账闸门（S1）\n")
    w("| grid | PC-on 暖格 | 其中命中>0 | PC-off 暖格 | 其中命中=0 | 仪器 | P2_PASS |")
    w("|---|---|---|---|---|---|---|")
    for g, d in sorted(A.items()):
        gt = d["p2_accounting"]["gate"]
        w(f"| {g} | {gt['pc_on_warm_cells']} | {gt['pc_on_warm_hits_positive']} | "
          f"{gt['pc_off_warm_cells']} | {gt['pc_off_warm_hits_zero']} | "
          f"{', '.join(gt.get('instruments') or [])} | {gt['P2_PASS']} |")

    # ---------- P1' ----------
    w("\n## 3. P1' 正确性不变量：PC on 与 PC off 的贪心输出必须一致\n")
    w("| grid | verdict | cells | 分歧格数 | none 臂分歧 | 仪器 |")
    w("|---|---|---|---|---|---|")
    for g, d in sorted(A.items()):
        p1 = d["p1_correctness"]
        instr = {c.get("instrument") for c in p1["cells"] if c.get("instrument")}
        w(f"| {g} | **{p1['verdict']}** | {p1['n_cells']} | {p1['n_divergent_cells']} | "
          f"{p1['none_arm_divergent']} | {', '.join(sorted(i for i in instr if i)) or 'n/a'} |")
    for g, d in sorted(A.items()):
        for c in d["p1_correctness"]["cells"]:
            if c.get("status") != "OK" and c.get("status") != "INSTRUMENT_UNAVAILABLE":
                w(f"- {g} {c['cell']}: {c['status']}")
            if c.get("status") == "OK" and (c["cross_pc_divergences"]
                                            or c["within_pc_on_divergences"]):
                w(f"- **DIVERGENT** {g} {c['cell']}: cross-PC={c['cross_pc_divergences']}, "
                  f"within-PC-on={c['within_pc_on_divergences']}, n={c['n_compared']}")

    # ---------- P3' ----------
    w(f"\n## 4. P3' 解码期吞吐：PC on vs off（暖臂），θ = {fmt(theta,4)}\n")
    w("| target | drafter | K | PC-on tok/s | PC-off tok/s | on/off | 超 θ | seed 方向一致 | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for g, d in sorted(A.items()):
        for r in d["p3_cost"]["cells"]:
            if r.get("status"):
                w(f"| {r['cell'][0]} | {r['cell'][1]} | {r['cell'][2]} | - | - | - | - | - | {r['status']} |")
                continue
            t, dr, k = r["cell"]
            w(f"| {t} | {dr} | {k} | {fmt(r['pc_on_tps'],2)} | {fmt(r['pc_off_tps'],2)} | "
              f"**{fmt(r['ratio_on_over_off'],4)}** | {r['exceeds_theta']} | "
              f"{r['direction_consistent']} | "
              f"{'**显著**' if r['significant'] else '不显著'} |")

    # ---------- prefill share ----------
    w("\n## 5. 预填充占比（证明主判据不再被预填充支配）\n")
    w("| grid | 臂 | TTFT p50 (s) | 解码段 (s) | 预填充占比 |")
    w("|---|---|---|---|---|")
    for p in sorted(glob.glob(os.path.join(RES, "raw", "*.json"))):
        if os.path.basename(p).startswith("manifest"):
            continue
        try:
            run = json.load(open(p))
        except Exception:
            continue
        if run.get("engine_status") != "OK":
            continue
        worst = None
        for arm in run.get("arms", []):
            per = [r for r in arm.get("per_request", []) if r.get("ttft_s") and r.get("decode_s")]
            if not per:
                continue
            ttft = st.median(r["ttft_s"] for r in per)
            dec = st.median(r["decode_s"] for r in per)
            share = ttft / (ttft + dec) if (ttft + dec) else None
            if worst is None or share > worst[1]:
                worst = (arm["tag"], share, ttft, dec)
        if worst:
            w(f"| {run.get('target')}/{run.get('drafter')}/pc{int(run.get('prefix_caching'))} "
              f"| {worst[0]} | {fmt(worst[2],4)} | {fmt(worst[3],3)} | "
              f"**{fmt(worst[1]*100,1)}%** |")

    txt = "\n".join(L) + "\n"
    open(a.out, "w").write(txt)
    print(txt)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
