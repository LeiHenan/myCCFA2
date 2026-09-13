#!/usr/bin/env python3
"""p12 判据统计器：按预登记 C0–C4 逐条算数并给判定（不依赖 analyze.py）。

用法：
  python verdict.py --dir /root/ccfa_results/2026-09-13/p12_corpus
  python verdict.py --selftest

对应 `candidates/sglang-draft-corpus/50_prereg.md`（冻结于 commit 207de66，数据采集前）：
  A_control_r{1,2,3}.jsonl  空语料首通 ×3
  B_treat_r{1,2,3}.jsonl    预置语料首通 ×3
  C_steady.jsonl            空语料 4 遍（第 4 遍为饱和）

判据：
  C0  accept_control_first <= 0.5 * accept_steady 且三进程同向
  C1  accept<=8、rate∈[0,1]、cached_tokens==0 全部无违例
  C2  recovery >= 0.80
  C3  effect = treat_mean - control_mean >= 3 * sigma(control 三进程)
  C4  wall_treat <= wall_control（方向一致性）
"""
import argparse
import glob
import json
import os
import statistics as st
import sys


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if not r.get("err")]


def first_pass(path):
    """该进程第 1 遍的 32 条（A/B 臂只有 1 遍；C 臂取 rep==0）。"""
    rs = [r for r in load(path) if r.get("rep", 0) == 0]
    return rs


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return st.fmean(vals) if vals else None


def total_wall(rows):
    vals = [r["wall_s"] for r in rows if r.get("wall_s") is not None]
    return sum(vals) if vals else None


def verdict(d, verbose=True):
    A = sorted(glob.glob(os.path.join(d, "A_control_r*.jsonl")))
    B = sorted(glob.glob(os.path.join(d, "B_treat_r*.jsonl")))
    C = os.path.join(d, "C_steady.jsonl")
    out = {}
    if verbose:
        print(f"A 臂文件 {len(A)} 个 | B 臂文件 {len(B)} 个 | C 臂 {'有' if os.path.exists(C) else '缺'}")
    if len(A) < 2 or len(B) < 2 or not os.path.exists(C):
        print("数据不足，无法判定")
        return None

    a_means, a_walls = [], []
    for f in A:
        rs = first_pass(f)
        a_means.append(mean_of(rs, "spec_accept_length"))
        a_walls.append(total_wall(rs))
    b_means, b_walls = [], []
    for f in B:
        rs = first_pass(f)
        b_means.append(mean_of(rs, "spec_accept_length"))
        b_walls.append(total_wall(rs))

    c_rows = load(C)
    reps = sorted({r.get("rep", 0) for r in c_rows})
    c_curve = {rp: mean_of([r for r in c_rows if r.get("rep", 0) == rp], "spec_accept_length")
               for rp in reps}
    c_walls = {rp: total_wall([r for r in c_rows if r.get("rep", 0) == rp]) for rp in reps}
    steady = c_curve[reps[-1]]
    steady_wall = c_walls[reps[-1]]

    control = st.fmean(a_means)
    treat = st.fmean(b_means)
    sigma = st.stdev(a_means) if len(a_means) > 1 else 0.0
    effect = treat - control
    recovery = (treat - control) / (steady - control) if steady != control else float("nan")

    # C1 语法/口径自洽（跨所有臂、所有行）
    viol, cached_nonzero, n_rows = 0, 0, 0
    for f in A + B + [C]:
        for r in load(f):
            n_rows += 1
            al, ar = r.get("spec_accept_length"), r.get("spec_accept_rate")
            if al is not None and al > 8.0 + 1e-9:
                viol += 1
            if ar is not None and not (-1e-9 <= ar <= 1 + 1e-9):
                viol += 1
            if r.get("cached_tokens"):
                cached_nonzero += 1

    c0 = control <= 0.5 * steady and all(m < steady for m in a_means)
    c1 = (viol == 0 and cached_nonzero == 0)
    c2 = recovery >= 0.80
    c3 = effect >= 3 * sigma
    c4 = (st.fmean(b_walls) <= st.fmean(a_walls)) and effect > 0

    if verbose:
        print("\n=== 逐进程首通接受长度 ===")
        for f, m in zip(A, a_means):
            print(f"  A {os.path.basename(f):22s} {m:.4f}")
        for f, m in zip(B, b_means):
            print(f"  B {os.path.basename(f):22s} {m:.4f}")
        print("\n=== C 臂 rep 曲线（饱和证据）===")
        for rp in reps:
            print(f"  rep{rp}  accept={c_curve[rp]:.4f}  wall={c_walls[rp]:.3f}s")
        print("\n=== 统计 ===")
        print(f"  control_first = {control:.4f}  (σ = {sigma:.4f}, n={len(a_means)})")
        print(f"  treat_first   = {treat:.4f}  (n={len(b_means)})")
        print(f"  steady        = {steady:.4f} (rep{reps[-1]}); rep{reps[-2]}={c_curve[reps[-2]]:.4f}")
        print(f"  cold_penalty  = {(steady - control) / steady * 100:.1f}%")
        print(f"  effect        = {effect:+.4f} token  ({effect / sigma if sigma else float('inf'):.1f}σ)")
        print(f"  recovery      = {recovery * 100:.1f}%")
        print(f"  wall: control={st.fmean(a_walls):.2f}s  treat={st.fmean(b_walls):.2f}s  steady={steady_wall:.2f}s")
        print(f"\n=== 判据 ===")
        print(f"  C0 冷启动可复现     : {'✅' if c0 else '❌'}  (control {control:.3f} <= 0.5×steady {0.5 * steady:.3f})")
        print(f"  C1 口径自洽         : {'✅' if c1 else '❌'}  (rows={n_rows}, 违例={viol}, cached!=0={cached_nonzero})")
        print(f"  C2 recovery >= 80%  : {'✅' if c2 else '❌'}  ({recovery * 100:.1f}%)")
        print(f"  C3 效应 >= 3σ       : {'✅' if c3 else '❌'}  ({effect:.3f} vs 3σ={3 * sigma:.3f})")
        print(f"  C4 wall 方向一致    : {'✅' if c4 else '❌'}")
        go = c0 and c1 and c2 and c3
        print(f"\n  ⇒ 判定：{'**Go**' if go else '**未达 Go**'}" +
              ("" if go else "（按预登记 §杀判据 处置）"))
    out = dict(a_means=a_means, b_means=b_means, c_curve=c_curve, control=control, treat=treat,
               steady=steady, sigma=sigma, effect=effect, recovery=recovery,
               c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, n_rows=n_rows, viol=viol,
               cached_nonzero=cached_nonzero, a_walls=a_walls, b_walls=b_walls,
               steady_wall=steady_wall)
    return out


def selftest():
    import tempfile
    td = tempfile.mkdtemp()
    def write(name, means, wall=1.0, reps=1):
        with open(os.path.join(td, name), "w", encoding="utf-8") as fh:
            for rp in range(reps):
                for i, m in enumerate(means):
                    fh.write(json.dumps({"rep": rp, "idx": i, "err": None,
                                         "spec_accept_length": m, "spec_accept_rate": 0.5,
                                         "cached_tokens": 0, "wall_s": wall}) + "\n")
    write("A_control_r1.jsonl", [1.8] * 32)
    write("A_control_r2.jsonl", [1.9] * 32)
    write("A_control_r3.jsonl", [1.85] * 32)
    write("B_treat_r1.jsonl", [7.6] * 32)
    write("B_treat_r2.jsonl", [7.7] * 32)
    write("B_treat_r3.jsonl", [7.65] * 32)
    write("C_steady.jsonl", [1.8] * 32, reps=1)
    # C 臂续写 rep1..rep3（每次 append，均值 7.0 / 7.7 / 7.78 ⇒ 稳态 = 最后一个 rep）
    for rp, mv in enumerate([7.0, 7.7, 7.78], start=1):
        with open(os.path.join(td, "C_steady.jsonl"), "a", encoding="utf-8") as fh:
            for i in range(32):
                fh.write(json.dumps({"rep": rp, "idx": i, "err": None,
                                     "spec_accept_length": mv, "spec_accept_rate": 0.5,
                                     "cached_tokens": 0, "wall_s": 1.0}) + "\n")
    o = verdict(td, verbose=False)
    assert o["c0"] and o["c1"] and o["c2"] and o["c3"] and o["c4"], o
    assert abs(o["steady"] - 7.78) < 1e-9, o["steady"]
    assert o["recovery"] > 0.97, o["recovery"]  # treat 7.65 / control 1.85 / steady 7.78 ⇒ 0.978
    # 反向：干预无效 ⇒ C2/C3 失败
    write("B_treat_r1.jsonl", [1.8] * 32); write("B_treat_r2.jsonl", [1.85] * 32)
    write("B_treat_r3.jsonl", [1.82] * 32)
    o2 = verdict(td, verbose=False)
    assert not o2["c2"], o2["recovery"]
    # 违例检测
    write("A_control_r1.jsonl", [9.0] * 32)
    o3 = verdict(td, verbose=False)
    assert not o3["c1"] and o3["viol"] > 0, o3
    print("selftest ✔ control/treat/steady 统计 · recovery · C0–C4 · 越界检测")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/ccfa_results/2026-09-13/p12_corpus")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return 0 if verdict(a.dir) else 1


if __name__ == "__main__":
    sys.exit(main())
