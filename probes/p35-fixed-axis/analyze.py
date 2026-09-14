#!/usr/bin/env python3
"""p35 分析：**修正轴**（末 query 丢弃质量）上的 TV 比与翻转率，并量化旧轴的缺陷幅度。

冻结判据（p33 之前就定的，不得事后移动）：
  · 匹配质量下 window/randn 的 TV 比 <=3x ⇒ 候选死（3x = 已发表稠密方向天花板 2604.13206）
  · 翻转率 CI 重叠 ⇒ 不能主张
"""
import json, math, random, sys
from collections import defaultdict

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (ph, max(0.0, c - hw), min(1.0, c + hw))

def med(v):
    s = sorted(v); return s[len(s) // 2] if s else float("nan")

def boot_ratio(xs, ys, B=4000, seed=0):
    rng = random.Random(seed)
    if len(xs) < 3 or len(ys) < 3:
        return (float("nan"),) * 3
    pt = med(xs) / max(med(ys), 1e-12)
    out = []
    for _ in range(B):
        a = [xs[rng.randrange(len(xs))] for _ in xs]
        b = [ys[rng.randrange(len(ys))] for _ in ys]
        out.append(med(a) / max(med(b), 1e-12))
    out.sort()
    return pt, out[int(0.025 * B)], out[int(0.975 * B)]

path = sys.argv[1] if len(sys.argv) > 1 else "/root/ccfa_results/p35/L512.jsonl"
raw = [json.loads(l) for l in open(path)]
rows = [r for r in raw if r.get("mass_ok")]
print("总格数 %d | mass_ok 合格 %d (%.1f%%) | **匹配轴 = drop_last（末 query）**"
      % (len(raw), len(rows), 100 * len(rows) / max(1, len(raw))))
bt = defaultdict(list)
for r in rows:
    bt[(r["ctx_type"], r["mode"], r["mass_target"])].append(r)
print("合格格分布:", {k: len(v) for k, v in sorted(bt.items())})

print("\n== A. 旧轴缺陷幅度（同一 drop_last 下 drop_global 是它的几倍）==")
print("  %-9s %-8s %-9s %s" % ("上下文", "mode", "中位 last", "中位 global / last"))
for ct in ("synth", "natural", "random"):
    for mode in ("topn", "randn", "window"):
        g = [r for r in rows if r["ctx_type"] == ct and r["mode"] == mode]
        if g:
            print("  %-9s %-8s %-9.4f **%.2fx**" % (ct, mode, med([r["drop_last"] for r in g]),
                                                    med([r["drop_global"] for r in g]) / max(med([r["drop_last"] for r in g]), 1e-9)))

print("\n== B. 各格翻转率（Wilson 95% CI）与 TV ==")
for ct in ("synth", "natural", "random"):
    print("-- %s --" % ct)
    print("  %-6s %-8s %-14s %-18s %-10s %s" % ("质量", "mode", "flip", "CI", "中位TV", "中位drop_last"))
    for M_ in sorted(set(r["mass_target"] for r in rows)):
        for mode in ("topn", "randn", "window"):
            g = bt.get((ct, mode, M_))
            if not g:
                continue
            k = sum(x["flip"] for x in g); n = len(g)
            ph, lo, hi = wilson(k, n)
            print("  %-6.2f %-8s %-14s [%.2f,%.2f]%-6s %-10.4f %.4f"
                  % (M_, mode, "%d/%d=%.2f" % (k, n, ph), lo, hi, "",
                     med([x["tv"] for x in g]), med([x["drop_last"] for x in g])))
    print()

print("== C. 修正轴上的 TV 比（median 比 + bootstrap 95% CI）==")
for ct in ("synth", "natural", "random"):
    for M_ in sorted(set(r["mass_target"] for r in rows)):
        w = [x["tv"] for x in bt.get((ct, "window", M_), [])]
        r_ = [x["tv"] for x in bt.get((ct, "randn", M_), [])]
        t_ = [x["tv"] for x in bt.get((ct, "topn", M_), [])]
        parts = []
        if len(w) >= 3 and len(r_) >= 3:
            pt, lo, hi = boot_ratio(r_, w)
            parts.append("randn/window=%8.1fx [%.1f,%.1f] %s" % (pt, lo, hi, "过3x" if lo > 3 else "未过3x"))
        if len(t_) >= 3 and len(r_) >= 3:
            pt, lo, hi = boot_ratio(r_, t_)
            parts.append("randn/topn=%6.1fx [%.1f,%.1f]" % (pt, lo, hi))
        if parts:
            print("  %-8s mass=%.2f : %s" % (ct, M_, "  |  ".join(parts)))

print("\n== D. 判据核对（合并三上下文，只用合格格）==")
for M_ in sorted(set(r["mass_target"] for r in rows)):
    w = [x["tv"] for r in rows if r["mass_target"] == M_ and r["mode"] == "window" for x in [r]]
    r_ = [x["tv"] for r in rows if r["mass_target"] == M_ and r["mode"] == "randn" for x in [r]]
    if len(w) >= 3 and len(r_) >= 3:
        pt, lo, hi = boot_ratio(r_, w)
        print("  mass=%.2f  randn/window = %8.1fx  CI [%.1f, %.1f]  -> %s"
              % (M_, pt, lo, hi, "**过 3x 天花板**" if lo > 3 else "**未过 3x ⇒ 按冻结判据应杀**"))
