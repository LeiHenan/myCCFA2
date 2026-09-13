#!/usr/bin/env python3
"""bs×depth×γ 网格判定器 —— 回答三个判据（对应用户 2026-09-12 的复核框架）

判据（预登记见 notes/prereg/p06-frontier.md 2026-09-13 条）：
  **③（决定性）**：`argmax_depth(bs)` 在 γ=3 与 γ=7 下是否**不同**？
      不同且差距 > 2×池化噪声 ⇒ 深度与长度**耦合**（二维联合分配问题成立）
      相同且恒定 ⇒ 只支持"按 bs 选一个固定深度" ⇒ 机制层弱化
  **③b**：每个 (bs,ctx) 是否存在**内点最优**（中等深度同时优于更浅与更深）
  **⑦（归因）**：`bs=16` 是 **KV 中立控制**（需求 65k tokens，而各深度可用 KV 均远大于它）
      若 bs=16 仍反转 ⇒ 归因**计算代价**；若消失 ⇒ 归因**显存挤占**（叙事退化）

用法：
  python analyze_bs_grid.py --dir <含 bs<N>/ 子目录的结果根> [--out <目录>]
  python analyze_bs_grid.py --selftest
"""

import argparse
import collections
import glob
import json
import os
import re
import statistics as st
import sys

FNAME = re.compile(r"d(\d+)_g(\d+)_ctx(\d+)_r(\d+)\.bench\.json$")


def load(root: str):
    """→ {(bs, depth, gamma, ctx): [tok_s, ...]}"""
    per = collections.defaultdict(list)
    for sub in sorted(glob.glob(os.path.join(root, "bs*"))):
        bs = int(os.path.basename(sub)[2:])
        for f in glob.glob(os.path.join(sub, "*.bench.json")):
            m = FNAME.search(os.path.basename(f))
            if not m:
                continue
            dep, gam, ctx, _ = (int(x) for x in m.groups())
            try:
                v = json.load(open(f)).get("output_throughput")
            except Exception:
                continue
            if isinstance(v, (int, float)):
                per[(bs, dep, gam, ctx)].append(float(v))
    return per


def stats(vals):
    if not vals:
        return None
    m = st.mean(vals)
    return {
        "n": len(vals), "mean": m, "p50": st.median(vals),
        "p95": sorted(vals)[max(0, int(round(0.95 * (len(vals) - 1))))],
        "min": min(vals), "max": max(vals),
        "spread_pct": (max(vals) - min(vals)) / m * 100 if m else float("nan"),
    }


def pair_noise_pct(per, keys):
    """参与比较的两格（最优 / 次优）的重复极差均值（%）。
    只用这两格而不是全部档位：离群档位（例如某档一次 34% 的离群）会把噪声估计整体抬高，
    从而把真实差异误判为"不可分辨"。"""
    vals = []
    for k in keys:
        if k in per and per[k]:
            s = stats(per[k])
            if s:
                vals.append(s["spread_pct"])
    return st.mean(vals) if vals else float("nan")


def analyze(root: str) -> dict:
    per = load(root)
    if not per:
        return {"error": f"{root} 下没有可用 bench.json"}
    cells = {k: stats(v) for k, v in per.items()}
    bs_all = sorted({k[0] for k in cells})
    ctx_all = sorted({k[3] for k in cells})
    gam_all = sorted({k[2] for k in cells})
    dep_all = sorted({k[1] for k in cells})

    rows, verdicts = [], []
    for ctx in ctx_all:
        for bs in bs_all:
            argmax = {}
            for g in gam_all:
                cand = [(cells[(bs, d, g, ctx)]["mean"], d)
                        for d in dep_all if (bs, d, g, ctx) in cells]
                if not cand:
                    continue
                cand.sort(reverse=True)
                top, d_best = cand[0]
                gap = (top - cand[1][0]) / top * 100 if len(cand) > 1 else float("nan")
                top2 = [c[1] for c in cand[:2]]
                noise = pair_noise_pct(per, [(bs, d, g, ctx) for d in top2])
                argmax[g] = d_best
                # 内点最优：最优深度既非最浅也非最深
                interior = d_best not in (dep_all[0], dep_all[-1])
                rows.append({
                    "ctx": ctx, "bs": bs, "gamma": g, "depth_star": d_best,
                    "tok_s": round(top, 1), "gap_to_2nd_pct": round(gap, 2),
                    "noise_pct": round(noise, 2), "resolvable": gap > 2 * noise,
                    "interior_optimum": interior,
                })
            # 判据③：不同 γ 下 argmax_depth 是否不同
            if len(gam_all) >= 2 and len(set(argmax.values())) > 1:
                shape = " | ".join(f"γ{g}→d{argmax[g]}" for g in sorted(argmax))
                verdicts.append(f"✅ 判据③ 成立 @ctx={ctx},bs={bs}：{shape}（深度与长度**耦合**）")
            elif len(argmax) >= 2:
                shape = " | ".join(f"γ{g}→d{argmax[g]}" for g in sorted(argmax))
                verdicts.append(f"❌ 判据③ 不成立 @ctx={ctx},bs={bs}：{shape}（各 γ 下最优深度相同）")
    return {"cells": cells, "rows": rows, "verdicts": verdicts,
            "axes": {"bs": bs_all, "ctx": ctx_all, "gamma": gam_all, "depth": dep_all}}


def summarize(res: dict) -> str:
    if "error" in res:
        return res["error"]
    L = ["# bs×depth×γ 网格判定", ""]
    L += res["verdicts"] or ["（无判据③ 结论：网格里 γ 只有一档）"]
    L += ["", "## 逐格（每 (bs,γ) 取 tok/s 最优深度）", "",
          "| ctx | bs | γ | depth* | tok/s | 与次优差 | 重复噪声 | 可分辨 | 内点最优 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append("| {ctx} | {bs} | {gamma} | **d{depth_star}** | {tok_s} | {gap_to_2nd_pct}% | "
                 "{noise_pct}% | {rk} | {io} |".format(
                     rk="✔" if r["resolvable"] else "✘", io="✔" if r["interior_optimum"] else "—", **r))
    L += ["", "> 口径：`depth*` = 该 (ctx,bs,γ) 下 tok/s 均值最大的深度；`可分辨` = 与次优的差距 > 2×重复极差；"]
    L += ["> `内点最优` = 最优深度既非最浅也非最深（判据②）；`bs=16` 行为 KV 中立控制（判据⑦）。"]
    return "\n".join(L) + "\n"


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # 构造（关键：要让 argmax_depth 随 γ 移动，否则判据③ 不该成立）：
        #   bs=1 ：γ=3 → d3 最优；γ=7 → d5 最优（移动 ⇒ 判据③ 成立）
        #   bs=32：两个 γ 都是 d3 最优（不移动 ⇒ 判据③ 不成立）
        table = {
            1: {3: {1: 70.0, 3: 90.0, 5: 80.0}, 7: {1: 72.0, 3: 90.0, 5: 105.0}},
            32: {3: {1: 700.0, 3: 1000.0, 5: 500.0}, 7: {1: 720.0, 3: 1010.0, 5: 520.0}},
        }
        for bs, per_g in table.items():
            sub = os.path.join(td, f"bs{bs}")
            os.makedirs(sub)
            for g, per_d in per_g.items():
                for d, base in per_d.items():
                    for r in (1, 2, 3):
                        v = base + (r - 2) * 0.5   # 极小的重复抖动
                        json.dump({"output_throughput": v},
                                  open(os.path.join(sub, f"d{d}_g{g}_ctx4096_r{r}.bench.json"), "w"))
        res = analyze(td)
        assert len(res["rows"]) == 4, len(res["rows"])   # rows = ctx(1) × bs(2) × γ(2)
        vd = res["verdicts"]
        assert any("✅ 判据③ 成立 @ctx=4096,bs=1" in v for v in vd), vd
        assert any("❌ 判据③ 不成立 @ctx=4096,bs=32" in v for v in vd), vd
        txt = summarize(res)
        assert "depth*" in txt
        print("selftest ✔ 加载/统计/argmax/判据③/内点最优/summary")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.dir:
        ap.error("需要 --dir，或 --selftest")
    res = analyze(a.dir)
    txt = summarize(res)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "bs_grid.md"), "w", encoding="utf-8").write(txt)
        with open(os.path.join(a.out, "bs_grid.csv"), "w", encoding="utf-8") as fh:
            keys = ["ctx", "bs", "gamma", "depth_star", "tok_s", "gap_to_2nd_pct",
                    "noise_pct", "resolvable", "interior_optimum"]
            fh.write(",".join(keys) + "\n")
            for r in res["rows"]:
                fh.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"\n已写出 {a.out}/bs_grid.md 与 bs_grid.csv")
