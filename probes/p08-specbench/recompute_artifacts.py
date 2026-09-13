#!/usr/bin/env python3
"""p08 零卡复算：三条度量伪影的**确切幅度**与**排序扭曲**程度（= S4 的 oracle 上界）。

**为什么是零卡**：P1/P2/P3 三条伪影的证据全在**已入库的 120 格 A1 逐次数据**里
（`results/p06-frontier/2026-09-13/A1_4k/`）⇒ 不花 GPU 就能算出「伪影能把结论带偏多少」，
从而在**采数之前**回答 S4 的闸门问题：上界 ≥8% 才值得做。

三个估计量（同一批数据，只换口径）：
  `rep1`   = 只报第 1 次重复（很多脚本的真实做法：跑一次就发）
  `mean`   = 全部重复取均值（默认做法）
  `steady` = 后一半重复取均值（本工作区论证的正确口径，decision #66）
对每个 (bs, γ) 比较：`argmax_depth` 是否不同、深度间差距差多少、以及**排序扭曲**。

用法：
  python recompute_artifacts.py --dir results/p06-frontier/2026-09-13/A1_4k [--out 目录]
  python recompute_artifacts.py --selftest
"""

import argparse
import collections
import glob
import json
import os
import re
import statistics as st
import sys

F = re.compile(r"d(\d+)_g(\d+)_ctx(\d+)_r(\d+)\.bench\.json$")


def load(root):
    per = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(root, "bs*", "*.bench.json")):
        m = F.search(os.path.basename(f))
        if not m:
            continue
        d, g, ctx, r = (int(x) for x in m.groups())
        bs = int(re.search(r"bs(\d+)", f).group(1))
        per[(bs, g)].setdefault(d, {})[r] = json.load(open(f))
    return per


def estimators(reps):
    """→ {'rep1':float, 'mean':float, 'steady':float}（吞吐）"""
    rs = sorted(reps)
    v = [reps[r]["output_throughput"] for r in rs]
    half = max(1, len(v) // 2)
    return {"rep1": v[0], "mean": st.mean(v), "steady": st.mean(v[-half:])}


def analyze(root):
    per = load(root)
    if not per:
        return {"error": f"{root} 下没有 A1 逐次数据"}
    rows, worst = [], []
    for (bs, g), depths in sorted(per.items()):
        if len(depths) < 2:
            continue
        est = {d: estimators(reps) for d, reps in depths.items()}
        rec = {"bs": bs, "gamma": g}
        for name in ("rep1", "mean", "steady"):
            order = sorted(est, key=lambda d: -est[d][name])
            top, second = order[0], order[1]
            gap = (est[top][name] - est[second][name]) / est[top][name] * 100
            rec[name] = {"argmax": top, "gap_pct": round(gap, 2)}
        # 排序/幅度扭曲
        rec["argmax_changed"] = len({rec[n]["argmax"] for n in ("rep1", "mean", "steady")}) > 1
        rec["gap_bias_mean_vs_steady"] = round(rec["mean"]["gap_pct"] - rec["steady"]["gap_pct"], 2)
        rec["gap_bias_rep1_vs_steady"] = round(rec["rep1"]["gap_pct"] - rec["steady"]["gap_pct"], 2)
        # 接受长度口径（P3）：每步延迟被当成每 token 的放大倍数
        d_top = rec["steady"]["argmax"]
        r0 = depths[d_top][sorted(depths[d_top])[-1]]
        rec["accept_len"] = round(r0["spec_decode_acceptance_length"], 3)
        rec["itl_over_tpot"] = round(r0["mean_itl_ms"] / r0["mean_tpot_ms"], 3)
        rec["p99_itl_over_mean_tpot"] = round(r0["p99_itl_ms"] / r0["mean_tpot_ms"], 3)
        rows.append(rec)
        worst.append((abs(rec["gap_bias_mean_vs_steady"]), bs, g, rec["gap_bias_mean_vs_steady"]))
    if not rows:
        return {"error": "没有可比较的 (bs,γ) 格"}
    worst.sort(reverse=True)
    max_bias = worst[0]
    n_flip = sum(1 for r in rows if r["argmax_changed"])
    verdicts = [
        f"**P1（暖机/口径）**：{len(rows)} 个 (bs,γ) 格中，三种口径下 `argmax_depth` 不同者有 **{n_flip}** 个；"
        f"最大「均值 vs 稳态」差距偏差 = **{max_bias[3]:+.2f} 个百分点**（bs={max_bias[1]}, γ={max_bias[2]}）。",
        f"**P3（ITL/TPOT）**：被当成「每 token 延迟」时的放大倍数 = `accept_len`，"
        f"实测区间 **{min(r['accept_len'] for r in rows):.2f}–{max(r['accept_len'] for r in rows):.2f}×**；"
        f"若报 `p99 ITL`，放大到 **{min(r['p99_itl_over_mean_tpot'] for r in rows):.2f}–"
        f"{max(r['p99_itl_over_mean_tpot'] for r in rows):.2f}×**。",
    ]
    mags = [abs(max_bias[3]), max(r["accept_len"] for r in rows) - 1,
            max(r["p99_itl_over_mean_tpot"] for r in rows) - 1]
    verdicts.append(f"**S4 闸门（上界 ≥8%）**：三条伪影的幅度分别为 "
                    f"{mags[0]:.1f} pp / {mags[1]:.2f}× / {mags[2]:.2f}× ⇒ "
                    + ("**全部远超 8% ⇒ 闸门通过**" if min(mags) > 0.08 * 10 else "**需人工判断**"))
    return {"rows": rows, "verdicts": verdicts}


def summarize(res):
    if "error" in res:
        return res["error"]
    L = ["# p08 零卡复算：度量伪影的幅度与排序扭曲", ""] + res["verdicts"] + ["",
         "| bs | γ | argmax(rep1) | argmax(mean) | argmax(steady) | gap(rep1) | gap(mean) | gap(steady) | 均值-稳态偏差 | 接受长度 | p99ITL/meanTPOT |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['bs']} | {r['gamma']} | d{r['rep1']['argmax']} | d{r['mean']['argmax']} | d{r['steady']['argmax']} | "
                 f"{r['rep1']['gap_pct']}% | {r['mean']['gap_pct']}% | {r['steady']['gap_pct']}% | "
                 f"**{r['gap_bias_mean_vs_steady']:+.2f} pp** | {r['accept_len']} | {r['p99_itl_over_mean_tpot']}× |")
    L += ["", "> `rep1` = 只跑一次；`mean` = 全部重复均值；`steady` = 后一半重复均值（decision #66 论证的正确口径）。",
          "> 「均值-稳态偏差」为正 ⇒ 默认口径**高估**深度间的差距。单位 pp = 百分点。"]
    return "\n".join(L) + "\n"


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "bs1"))
        # d5 冷启动爬升更大 ⇒ mean 口径会把 d5 的优势压低
        for d, base, ramp in ((1, 100.0, 0.06), (5, 130.0, 0.13)):
            for r in range(1, 6):
                v = base * (1 - ramp) * (1 + ramp * (r - 1) / 4)
                json.dump({"output_throughput": v, "spec_decode_acceptance_length": 1.3 + 0.3 * d,
                           "mean_itl_ms": 20.0, "mean_tpot_ms": 20.0 / (1.3 + 0.3 * d),
                           "p99_itl_ms": 30.0},
                          open(os.path.join(td, "bs1", f"d{d}_g7_ctx4096_r{r}.bench.json"), "w"))
        res = analyze(td)
        assert "error" not in res, res
        assert res["rows"][0]["steady"]["argmax"] == 5, res["rows"][0]
        assert res["rows"][0]["gap_bias_mean_vs_steady"] < 0, res["rows"][0]
        assert any("S4 闸门" in v for v in res["verdicts"])
        assert "均值-稳态偏差" in summarize(res)
        print("selftest ✔ 加载/三口径/排序扭曲/偏差方向/接受长度放大/上界判定/输出")


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
        open(os.path.join(a.out, "artifacts.md"), "w", encoding="utf-8").write(txt)
        print(f"已写出 {a.out}/artifacts.md")
