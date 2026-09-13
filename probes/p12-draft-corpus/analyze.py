#!/usr/bin/env python3
"""p12 分析器：把逐请求 JSONL 折成「rep 曲线 + 冷启动罚金 + 两条路径交叉验证」。

用法：
  python analyze.py --in /root/ccfa_results/.../smoke.jsonl
  python analyze.py --in a.jsonl b.jsonl ... --selftest

设计要点（每条都对应一条已踩过的坑）：
- **rep 边界显式**：同一 serve 内 idx 从 0 重新计（决策 #93：trace 序号必须与 rep 边界对齐）。
- **两条独立路径**：客户端读出的 `spec_accept_length`（响应体）与
  `accept_len_from_counters`（`/get_internal_state` 累计计数器**做差**）必须一致（决策 #66）。
- **不算 gauge**：本分析器不碰任何跑完归零的量。
- 缺字段不伪造：记 None 并在摘要里报「未采到」。
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def by_tag(rows):
    d = defaultdict(list)
    for r in rows:
        d[r.get("tag", "?")].append(r)
    return d


def fmean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None


def fmt(x, nd=4):
    return "未采到" if x is None else f"{x:.{nd}f}"


def rep_curve(rows):
    """→ {rep: {'acc': 均值, 'n': 条数, 'cum_acc': 均值}}，按 (rep, idx) 排序保证边界正确。"""
    rs = sorted(rows, key=lambda r: (r.get("rep", 0), r.get("idx", 0)))
    per = defaultdict(list)
    for r in rs:
        if r.get("err"):
            continue
        per[r.get("rep", 0)].append(r)
    out = {}
    for rep, rs2 in sorted(per.items()):
        out[rep] = {
            "n": len(rs2),
            "acc": fmean([r.get("spec_accept_length") for r in rs2]),
            "rate": fmean([r.get("spec_accept_rate") for r in rs2]),
            "cum_acc": fmean([r.get("accept_len_from_counters") for r in rs2]),
            "wall": fmean([r.get("wall_s") for r in rs2]),
            "cached": fmean([r.get("cached_tokens") for r in rs2]),
            "comp": fmean([r.get("completion_tokens") for r in rs2]),
        }
    return out


def cross_check(rows):
    """两条路径的一致性：|resp − counters| / resp。"""
    pairs = [(r.get("spec_accept_length"), r.get("accept_len_from_counters"))
             for r in rows if not r.get("err")]
    pairs = [(a, b) for a, b in pairs if a and b]
    if not pairs:
        return None, 0
    rel = [abs(a - b) / a for a, b in pairs]
    return st.fmean(rel), len(pairs)


def first_request_effect(rows):
    """同 serve 内第 1 次 vs 稳态（后半 reps）的接受长度差 —— 会话内累积的效应量。"""
    curve = rep_curve(rows)
    if len(curve) < 2:
        return None
    reps = sorted(curve)
    first = curve[reps[0]]["acc"]
    tail = [curve[r]["acc"] for r in reps[len(reps) // 2:] if curve[r]["acc"] is not None]
    if first is None or not tail:
        return None
    steady = st.fmean(tail)
    return {"rep0": first, "steady": steady,
            "delta_abs": steady - first,
            "delta_pct": 100.0 * (steady - first) / first if first else None}


def report(rows, label=""):
    print(f"===== {label}  rows={len(rows)}  err={sum(1 for r in rows if r.get('err'))}")
    keys = sorted({k for r in rows for k in r.get("meta_keys", []) or []})
    if keys:
        print("  meta_info 可见字段：", ", ".join(keys))
    curve = rep_curve(rows)
    print("  rep |    n |  acc_len | acc_rate | cum_acc | wall_s | cached | comp")
    for rep, v in curve.items():
        print(f"  {rep:3d} | {v['n']:4d} | {fmt(v['acc']):>8s} | {fmt(v['rate']):>8s} | "
              f"{fmt(v['cum_acc']):>7s} | {fmt(v['wall'],3):>6s} | {fmt(v['cached'],1):>6s} | {fmt(v['comp'],1):>5s}")
    rel, n = cross_check(rows)
    print(f"  交叉验证（响应体 vs 累计计数器做差）：n={n} 平均相对差={fmt(rel)}")
    fe = first_request_effect(rows)
    if fe:
        print(f"  会话内第 1 rep vs 稳态：{fe['rep0']:.4f} → {fe['steady']:.4f} "
              f"(+{fe['delta_abs']:.4f}, +{fe['delta_pct']:.1f}%)")
    return curve


def selftest():
    rows = []
    # 假数据：rep0 acc=1.5，rep1-2 acc=3.0；计数器做差路径与响应体略有偏差
    for rep in range(3):
        for idx in range(4):
            acc = 1.5 if rep == 0 else 3.0
            rows.append({"tag": "t", "rep": rep, "idx": idx, "err": None,
                         "spec_accept_length": acc, "spec_accept_rate": acc / 8,
                         "accept_len_from_counters": acc * 1.01,
                         "completion_tokens": 128, "wall_s": 1.0, "cached_tokens": 0,
                         "meta_keys": ["spec_accept_length", "spec_accept_rate"]})
    rep_curve(rows)
    rel, n = cross_check(rows)
    assert n == 12 and rel is not None and 0.009 < rel < 0.011, (n, rel)
    fe = first_request_effect(rows)
    assert abs(fe["rep0"] - 1.5) < 1e-9 and abs(fe["steady"] - 3.0) < 1e-9, fe
    assert abs(fe["delta_pct"] - 100.0) < 1e-6, fe
    # 单 rep 时不应给出效应（样本不足）
    assert first_request_effect(rows[:4]) is None
    # 错误行必须被排除
    rows.append({"tag": "t", "rep": 0, "idx": 9, "err": "boom", "spec_accept_length": 99.0})
    c = rep_curve(rows)
    assert abs(c[0]["acc"] - 1.5) < 1e-9, c
    print("selftest ✔ rep 边界/两路径交叉/首 rep 效应/错误行排除")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="*", default=[])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.inputs:
        ap.error("需要 --in <jsonl>...")
    rows = load(a.inputs)
    for tag, rs in sorted(by_tag(rows).items()):
        report(rs, label=tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
