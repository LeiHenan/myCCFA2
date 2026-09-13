#!/usr/bin/env python3
"""bs×depth×γ 网格判定器（v2：稳态口径 + 显著性检验）

判据（预登记见 notes/prereg/p06-frontier.md）：
  **① 稳定交叉反转**：不同 (bs,ctx) 下 `argmax_depth` 不同，且差距可分辨
  **② 内点最优**：中等深度同时优于更浅与更深
  **③（决定性）**：`argmax_depth` 是否随 γ 移动（深度与长度是否**耦合**）
  **⑦（归因）**：显存充裕时仍反转 ⇒ 计算代价；消失 ⇒ 显存挤占

⚠️ v2 的两处修正（2026-09-13 审计发现，见 decision #65/#66）：
  1. **稳态口径**：同一格前几次重复存在冷启动爬升，且**爬升幅度随深度不同**
     （实测 bs=1：d5 爬 11–13%，d1/d3 只爬 6–7%；bs≥8 无爬升）
     ⇒ 只报"全部重复的均值"会**系统性低估深 drafter**。故同时报 `steady`（后一半重复均值），
     并以 steady 做 argmax 判定。
  2. **显著性**：不再用"差距 > 2×极差"的启发式，改用 **Welch t 检验**（不等方差、小样本）报 p 值。

用法：
  python analyze_bs_grid.py --dir <含 bs<N>/ 的结果根> [--out <目录>] [--stat mean|steady]
  python analyze_bs_grid.py --selftest
"""

import argparse
import collections
import glob
import json
import math
import os
import re
import statistics as st
import sys

FNAME = re.compile(r"d(\d+)_g(\d+)_ctx(\d+)_r(\d+)\.bench\.json$")


def load(root: str):
    """→ {(bs, depth, gamma, ctx): {rep: tok_s}}"""
    per = collections.defaultdict(dict)
    for sub in sorted(glob.glob(os.path.join(root, "bs*"))):
        m2 = re.match(r"bs(\d+)$", os.path.basename(sub))
        if not m2:
            continue
        bs = int(m2.group(1))
        for f in glob.glob(os.path.join(sub, "*.bench.json")):
            m = FNAME.search(os.path.basename(f))
            if not m:
                continue
            dep, gam, ctx, rep = (int(x) for x in m.groups())
            try:
                v = json.load(open(f)).get("output_throughput")
            except Exception:
                continue
            if isinstance(v, (int, float)):
                per[(bs, dep, gam, ctx)][rep] = float(v)
    return per


def welch(a, b):
    """Welch t 检验 → (t, 近似自由度, 双尾 p)（p 用正态近似，小样本偏乐观）。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan"), float("nan")
    va, vb = st.variance(a), st.variance(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return float("nan"), float("nan"), float("nan")
    t = (st.mean(a) - st.mean(b)) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df, math.erfc(abs(t) / math.sqrt(2))


def cell_stats(rep_map: dict, mode: str = "steady"):
    reps = sorted(rep_map)
    allv = [rep_map[r] for r in reps]
    if not allv:
        return None
    half = max(1, len(allv) // 2)
    steady = allv[-half:]                      # 后一半 = 稳态
    use = steady if mode == "steady" else allv
    return {
        "n": len(allv), "mean": st.mean(allv), "steady": st.mean(steady), "use": st.mean(use),
        "p50": st.median(allv), "min": min(allv), "max": max(allv),
        "ramp_pct": (allv[-1] - allv[0]) / allv[0] * 100 if allv[0] else float("nan"),
        "vals": allv, "steady_vals": steady,
    }


def analyze(root: str, mode: str = "steady") -> dict:
    per = load(root)
    if not per:
        return {"error": f"{root} 下没有可用 bench.json"}
    cells = {k: cell_stats(v, mode) for k, v in per.items()}
    bs_all = sorted({k[0] for k in cells})
    ctx_all = sorted({k[3] for k in cells})
    gam_all = sorted({k[2] for k in cells})
    dep_all = sorted({k[1] for k in cells})

    rows, verdicts = [], []
    for ctx in ctx_all:
        for bs in bs_all:
            argmax = {}
            for g in gam_all:
                cand = [(cells[(bs, d, g, ctx)]["use"], d)
                        for d in dep_all if (bs, d, g, ctx) in cells]
                if not cand:
                    continue
                cand.sort(reverse=True)
                top, d_best = cand[0]
                d_2nd = cand[1][1] if len(cand) > 1 else None
                gap = (top - cand[1][0]) / top * 100 if len(cand) > 1 else float("nan")
                t = df = p = float("nan")
                ramp_a = ramp_b = float("nan")
                if d_2nd is not None:
                    a = cells[(bs, d_best, g, ctx)]["steady_vals"]
                    b = cells[(bs, d_2nd, g, ctx)]["steady_vals"]
                    t, df, p = welch(a, b)
                    ramp_a = cells[(bs, d_best, g, ctx)]["ramp_pct"]
                    ramp_b = cells[(bs, d_2nd, g, ctx)]["ramp_pct"]
                argmax[g] = d_best
                rows.append({
                    "ctx": ctx, "bs": bs, "gamma": g, "depth_star": d_best,
                    "tok_s": round(top, 1),
                    "tok_s_2nd": round(cand[1][0], 1) if len(cand) > 1 else None,
                    "gap_pct": round(gap, 2),
                    "p_value": round(p, 5) if p == p else None,
                    "ramp_top_pct": round(ramp_a, 1) if ramp_a == ramp_a else None,
                    "ramp_2nd_pct": round(ramp_b, 1) if ramp_b == ramp_b else None,
                    "interior_optimum": d_best not in (dep_all[0], dep_all[-1]),
                    "significant": bool(p == p and p < 0.05),
                })
            if len(gam_all) >= 2 and len(set(argmax.values())) > 1:
                shape = " | ".join(f"γ{g}→d{argmax[g]}" for g in sorted(argmax))
                verdicts.append(f"✅ 判据③ 成立 @ctx={ctx},bs={bs}：{shape}（深度与长度**耦合**）")
            elif len(argmax) >= 2:
                shape = " | ".join(f"γ{g}→d{argmax[g]}" for g in sorted(argmax))
                verdicts.append(f"❌ 判据③ 不成立 @ctx={ctx},bs={bs}：{shape}（各 γ 下最优深度相同）")
    return {"cells": cells, "rows": rows, "verdicts": verdicts, "mode": mode,
            "axes": {"bs": bs_all, "ctx": ctx_all, "gamma": gam_all, "depth": dep_all}}


def summarize(res: dict) -> str:
    if "error" in res:
        return res["error"]
    L = [f"# bs×depth×γ 网格判定（口径：{'稳态（后一半重复）' if res['mode']=='steady' else '全部重复均值'}）", ""]
    L += res["verdicts"] or ["（无判据③ 结论：网格里 γ 只有一档）"]
    L += ["", "## 逐格（每 (bs,γ) 取 tok/s 最优深度）", "",
          "| ctx | bs | γ | depth* | tok/s(稳态) | 次优 | 差距 | Welch p | 显著 | 冷启动爬升(最优/次优) | 内点最优 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append("| {ctx} | {bs} | {gamma} | **d{depth_star}** | {tok_s} | {tok_s_2nd} | {gap_pct}% | "
                 "{p_value} | {sig} | {ramp_top_pct}% / {ramp_2nd_pct}% | {io} |".format(
                     sig="✔" if r["significant"] else "✘",
                     io="✔" if r["interior_optimum"] else "—", **r))
    L += ["",
          "> `depth*` = 该 (ctx,bs,γ) 下**稳态** tok/s 最大的深度；显著性 = Welch t 检验 p<0.05（正态近似，小样本偏乐观）。",
          "> `冷启动爬升` = 该格 r1→rlast 变化率；**深度之间爬升差异大时必须看稳态列**（decision #65/#66）。",
          "> `内点最优` = 最优深度既非最浅也非最深（判据②）。"]
    return "\n".join(L) + "\n"


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # bs=1：γ=3 → d3 最优、γ=7 → d5 最优（移动 ⇒ 判据③ 成立）
        # bs=32：两个 γ 都是 d3 最优（不移动 ⇒ 不成立）；并给 d5 注入冷启动爬升
        table = {
            1: {3: {1: 70.0, 3: 90.0, 5: 80.0}, 7: {1: 72.0, 3: 90.0, 5: 105.0}},
            32: {3: {1: 700.0, 3: 1000.0, 5: 500.0}, 7: {1: 720.0, 3: 1010.0, 5: 520.0}},
        }
        for bs, per_g in table.items():
            sub = os.path.join(td, f"bs{bs}")
            os.makedirs(sub)
            for g, per_d in per_g.items():
                for d, base in per_d.items():
                    for r in (1, 2, 3, 4, 5):
                        v = base + (r - 3) * 0.3
                        if d == 5:
                            v *= 0.85 + 0.03 * r       # d5 冷启动爬升
                        json.dump({"output_throughput": v},
                                  open(os.path.join(sub, f"d{d}_g{g}_ctx4096_r{r}.bench.json"), "w"))
        res = analyze(td, "steady")
        vd = res["verdicts"]
        assert any("✅ 判据③ 成立 @ctx=4096,bs=1" in v for v in vd), vd
        assert any("❌ 判据③ 不成立 @ctx=4096,bs=32" in v for v in vd), vd
        r1 = [r for r in res["rows"] if r["bs"] == 1 and r["gamma"] == 7][0]
        assert r1["depth_star"] == 5, r1
        assert r1["ramp_top_pct"] is not None
        txt = summarize(res)
        assert "Welch p" in txt and "稳态" in txt
        print("selftest ✔ 加载/稳态口径/argmax/判据③/Welch/爬升记录/summary")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--stat", choices=["mean", "steady"], default="steady")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.dir:
        ap.error("需要 --dir，或 --selftest")
    res = analyze(a.dir, a.stat)
    txt = summarize(res)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "bs_grid.md"), "w", encoding="utf-8").write(txt)
        keys = ["ctx", "bs", "gamma", "depth_star", "tok_s", "tok_s_2nd", "gap_pct",
                "p_value", "significant", "ramp_top_pct", "ramp_2nd_pct", "interior_optimum"]
        with open(os.path.join(a.out, "bs_grid.csv"), "w", encoding="utf-8") as fh:
            fh.write(",".join(keys) + "\n")
            for r in res["rows"]:
                fh.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"\n已写出 {a.out}/bs_grid.md 与 bs_grid.csv")
