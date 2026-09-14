#!/usr/bin/env python3
"""p33 分析：分层翻转率（Wilson CI）、精确匹配质量上的 TV 比（bootstrap CI）、margin 位移。

冻结判据（S3B_NUMERICAL_DRIFT §4，不得事后移动）：
  · 匹配质量下 TV 比 <=3x ⇒ 杀（不优于 2604.13206 的稠密方向天花板 3x）
  · 翻转率 CI 重叠 ⇒ 不能主张
  · 必须同时看到 top-2 margin 的差异性位移
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

def boot_ratio(xs, ys, B=4000, seed=0):
    """median(xs)/median(ys) 的 bootstrap CI（按上下文重采样需配对，这里用独立重采样，保守）"""
    rng = random.Random(seed)
    if len(xs) < 3 or len(ys) < 3:
        return (float("nan"),) * 3
    def med(v): 
        s = sorted(v); return s[len(s)//2]
    pt = med(xs) / max(med(ys), 1e-12)
    out = []
    for _ in range(B):
        a = [xs[rng.randrange(len(xs))] for _ in xs]
        b = [ys[rng.randrange(len(ys))] for _ in ys]
        out.append(med(a) / max(med(b), 1e-12))
    out.sort()
    return pt, out[int(0.025 * B)], out[int(0.975 * B)]

raw = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "/home/user/ccfa_logs/p33/scale.jsonl")]
rows = [r for r in raw if r.get("mass_ok")]
print("总格数: %d | mass_ok 合格: %d (%.1f%%) | **以下只用合格格**" % (
    len(raw), len(rows), 100 * len(rows) / len(raw)))
import collections
print("合格格按类型/mode:", dict(collections.Counter((r["ctx_type"], r["mode"]) for r in rows)))

byt = defaultdict(list)
for r in rows:
    byt[(r["ctx_type"], r["mode"], r["mass_target"])].append(r)

print("\n== 分层翻转率（Wilson 95% CI）与 TV ==")
for ct in ("synth", "natural", "random"):
    print("-- 上下文类型: %s --" % ct)
    print("  %-8s %-7s %-13s %-9s %-9s %s" % ("质量", "mode", "flip rate", "CI", "median TV", "median d_gap"))
    for M_ in sorted(set(r["mass_target"] for r in rows)):
        for mode in ("topn", "randn", "window"):
            g = byt.get((ct, mode, M_))
            if not g:
                continue
            k = sum(x["flip"] for x in g); n = len(g)
            ph, lo, hi = wilson(k, n)
            tvs = sorted(x["tv"] for x in g); dg = sorted(x["d_gap"] for x in g)
            print("  %-8.2f %-7s %-13s [%.2f,%.2f]  %-9.4f %+.3f" % (
                M_, mode, "%d/%d=%.2f" % (k, n, ph), lo, hi, tvs[len(tvs)//2], dg[len(dg)//2]))
    print()

print("== 精确匹配质量上的 TV 比（median 比 + bootstrap 95% CI）==")
for ct in ("synth", "natural", "random"):
    for M_ in sorted(set(r["mass_target"] for r in rows)):
        w = [x["tv"] for x in byt.get((ct, "window", M_), [])]
        r_ = [x["tv"] for x in byt.get((ct, "randn", M_), [])]
        t_ = [x["tv"] for x in byt.get((ct, "topn", M_), [])]
        line = "  %-8s mass=%.2f : " % (ct, M_)
        if w and r_:
            pt, lo, hi = boot_ratio(w, r_)
            line += "window/randn=%.2fx [%.2f,%.2f]  " % (pt, lo, hi)
        if t_ and r_:
            pt, lo, hi = boot_ratio(t_, r_)
            line += "topn/randn=%.2fx [%.2f,%.2f]" % (pt, lo, hi)
        print(line)

print("\n== 判据核对（只用 natural+random，synth 的中位 TV 为 0 会使比值退化）==")
for M_ in sorted(set(r["mass_target"] for r in rows)):
    w = [x["tv"] for ct in ("natural", "random") for x in byt.get((ct, "window", M_), [])]
    r_ = [x["tv"] for ct in ("natural", "random") for x in byt.get((ct, "randn", M_), [])]
    t_ = [x["tv"] for ct in ("natural", "random") for x in byt.get((ct, "topn", M_), [])]
    if w and r_:
        pt, lo, hi = boot_ratio(w, r_)
        # 比值 <1 表示 window 漂移更小 => 倒数才是"差多少倍"
        print("  mass=%.2f  window/randn=%.3fx [%.3f,%.3f] => randn 漂移是 window 的 %.1f 倍 [%.1f,%.1f]  -> %s"
              % (M_, pt, lo, hi, 1/max(pt,1e-9), 1/max(hi,1e-9), 1/max(lo,1e-9),
                 "越过 3x 天花板" if 1/max(hi,1e-9) > 3 else "未过 3x"))
    if t_ and r_:
        pt, lo, hi = boot_ratio(t_, r_)
        print("  mass=%.2f  topn/randn  =%.3fx [%.3f,%.3f] => randn 漂移是 topn 的 %.1f 倍" % (
            M_, pt, lo, hi, 1/max(pt,1e-9)))
