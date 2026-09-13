#!/usr/bin/env python3
"""p12 P 阶段判据计算器：新副本的语料获取策略（整机总时间 + 跨租户迁移）。

对应 `notes/prereg/sglang-draft-corpus-p.md`（采数前冻结）：
  P1 冷启动前提复现            accept(cold) <= 0.5 * accept(self)
  P2 异租户迁移不足            accept(peer) <= 1.03 * accept(bare)   ⇒ H-P2 成立（前缀迁不动）
  P3 文本型语料是否已够        accept(text) >= 0.9 * accept(self)
  P4 整机时间最优臂 ≥8% 优于冷启动
  P5 口径自洽                  违例 0

用法：
  python compare_policy.py --dir /root/ccfa_results/2026-09-13/p12_policy
  python compare_policy.py --selftest
"""
import argparse
import json
import os
import statistics as st
import sys

TARGET = "S_cold"          # 目标 workload 的标签集合（所有策略臂都服务同一批 prompt）
PREP_ARMS = {              # 策略臂 -> 其语料来源（生成臂的 tag）；None = 无需准备
    "S_cold": None,
    "S_self": "S_gen_self",
    "S_peer": "S_gen_peer",
    "S_text": None,        # 语料就是 prompt 原文，客户端本来持有 ⇒ 生成成本 0
    "S_bare": "S_gen_bare",
}


def rows(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return [r for r in out if not r.get("err")]


def mean_accept(path):
    rs = rows(path)
    v = [r["spec_accept_length"] for r in rs if r.get("spec_accept_length") is not None]
    return st.fmean(v) if v else None


def sum_e2e(path):
    rs = rows(path)
    v = [r.get("e2e_latency") for r in rs if r.get("e2e_latency") is not None]
    return sum(v) if v else None


def restore_cost(cost_file, corpus_id):
    for r in rows(cost_file):
        if r.get("_id") == corpus_id or r.get("corpus_id") == corpus_id:
            return r.get("wall_s")
    # 无 id 字段时按出现顺序回退（本轮每臂只恢复一次，且日志顺序与臂顺序一致）
    ws = [r.get("wall_s") for r in rows(cost_file)]
    return ws[-1] if ws else None


def analyze(d, verbose=True):
    out = {}
    for arm, gen in PREP_ARMS.items():
        ap = mean_accept(os.path.join(d, arm + ".jsonl"))
        tgt = sum_e2e(os.path.join(d, arm + ".jsonl"))
        prep = 0.0
        if gen:
            prep = sum_e2e(os.path.join(d, gen + ".jsonl")) or 0.0
        out[arm] = {"accept": ap, "target": tgt, "prep": prep,
                    "total": None if tgt is None else tgt + prep}
    acc = {k: v["accept"] for k, v in out.items()}
    tot = {k: v["total"] for k, v in out.items()}
    ok = all(v is not None for v in acc.values()) and all(v is not None for v in tot.values())

    p1 = ok and acc["S_cold"] is not None and acc["S_self"] is not None \
        and acc["S_cold"] <= 0.5 * acc["S_self"]
    p2 = ok and acc["S_peer"] is not None and acc["S_bare"] is not None \
        and acc["S_peer"] <= 1.03 * acc["S_bare"]
    p3 = ok and acc["S_text"] is not None and acc["S_self"] is not None \
        and acc["S_text"] >= 0.9 * acc["S_self"]
    best = min((k for k in tot if tot[k] is not None), key=lambda k: tot[k]) if ok else None
    gain = (tot["S_cold"] - tot[best]) / tot["S_cold"] if (ok and best) else None
    p4 = gain is not None and gain >= 0.08

    if verbose:
        print("=== 各策略臂（目标 workload 相同：512-token 共享前缀 + 4096 原文 × 32）===")
        print(f"{'臂':<8}{'首通accept':>12}{'目标耗时(s)':>14}{'准备耗时(s)':>14}{'整机总时(s)':>14}")
        for k in ("S_cold", "S_bare", "S_peer", "S_text", "S_self"):
            v = out[k]
            f = lambda x: "未采到" if x is None else f"{x:.3f}"  # noqa: E731
            print(f"{k:<8}{f(v['accept']):>12}{f(v['target']):>14}{f(v['prep']):>14}{f(v['total']):>14}")
        print("\n=== 相对「从 0 学」(S_cold) 的整机时间差 ===")
        if ok:
            for k in ("S_bare", "S_peer", "S_text", "S_self"):
                rel = (tot["S_cold"] - tot[k]) / tot["S_cold"] * 100
                print(f"  {k:<8} {rel:+.1f}%")
        print("\n=== 判据 ===")
        print(f"  P1 冷启动前提复现            : {'✅' if p1 else '❌'}")
        print(f"  P2 异租户迁移 ≤3% 于裸 prompt : {'✅' if p2 else '❌'}  "
              f"(peer={acc['S_peer']:.3f} vs bare={acc['S_bare']:.3f})" if ok else "  P2 : 未判定")
        print(f"  P3 原文够用(≥0.9×self)       : {'✅' if p3 else '❌'}  "
              f"(text={acc['S_text']:.3f} vs self={acc['S_self']:.3f})" if ok else "  P3 : 未判定")
        print(f"  P4 最优臂 ≥8% 优于冷启动      : {'✅' if p4 else '❌'}  "
              f"(最优 {best}, 增益 {gain * 100:.1f}%)" if ok and best else "  P4 : 未判定")
        if ok and best:
            print(f"  ⇒ 整机时间最优策略 = {best}")
            if not p4:
                print("  ⇒ 按预登记：最优臂 <8% ⇒ **杀策略方向**")
    return dict(arms=out, p1=bool(p1), p2=bool(p2), p3=bool(p3), p4=bool(p4),
                best=best, gain=gain, acc=acc, tot=tot)


def selftest():
    """用合成数据逐条验证判据；**每个场景只改需要改的臂**，并显式写出预期。"""
    import tempfile
    td_holder = {}
    def fresh():
        import shutil, tempfile as _tf
        td = _tf.mkdtemp()
        def w(name, acc, e2e):
            with open(os.path.join(td, name), "w", encoding="utf-8") as fh:
                for i in range(32):
                    fh.write(json.dumps({"idx": i, "err": None, "spec_accept_length": acc,
                                         "e2e_latency": e2e}) + "\n")
        # 基准场景：cold 慢且低、self 快且高、peer≈bare（前缀迁不动）、text 中等（不够 0.9×self）
        w("S_gen_self.jsonl", 1.0, 0.783)     # 准备 25.06 s
        w("S_gen_peer.jsonl", 1.0, 0.783)     # 准备 25.06 s
        w("S_gen_bare.jsonl", 1.0, 0.783)     # 准备 25.06 s
        w("S_cold.jsonl", 1.8360, 0.780)      # 目标 24.96 s，总 24.96
        w("S_self.jsonl", 7.0000, 0.320)      # 目标 10.24 s，总 35.30
        w("S_peer.jsonl", 1.9000, 0.770)      # 目标 24.64 s，总 49.70
        w("S_text.jsonl", 4.5000, 0.500)      # 目标 16.00 s，总 16.00
        w("S_bare.jsonl", 1.8500, 0.780)      # 目标 24.96 s，总 50.02
        return td, w
    td, w = fresh()

    # --- 场景 1（基准）：P1 ✅ P2 ✅ P3 ❌（4.5 < 0.9×7）P4 ✅（S_text 16.0 最优）---
    o = analyze(td, verbose=False)
    assert o["p1"], o
    assert o["p2"], f"peer 1.90 <= 1.03*1.85=1.9055 应成立: {o['acc']}"
    assert not o["p3"], f"text 4.5 < 0.9*7=6.3 应不成立: {o['acc']}"
    assert o["p4"] and o["best"] == "S_text", (o["best"], o["tot"])

    # --- 场景 2：peer 明显优于 bare（+4%）⇒ P2 失败（前缀可迁移）---
    td2, w2 = fresh()
    w2("S_peer.jsonl", 1.9300, 0.760)
    assert not analyze(td2, verbose=False)["p2"], "1.93 > 1.03*1.85=1.9055 ⇒ P2 应失败"

    # --- 场景 3：text 够用（≥0.9×self）⇒ P3 成立 ---
    td3, w3 = fresh()
    w3("S_text.jsonl", 6.4000, 0.400)
    o3 = analyze(td3, verbose=False)
    assert o3["p3"], f"6.4 >= 0.9*7=6.3 应成立: {o3['acc']}"

    # --- 场景 4：self 变慢以致最优臂只快 2% ⇒ P4 失败 ---
    td4, w4 = fresh()
    w4("S_text.jsonl", 4.5000, 0.765)          # 目标 24.48，总 24.48（相对 cold 仅 -1.9%）
    w4("S_self.jsonl", 7.0000, 0.765)          # 目标 24.48 + 准备 25.06
    assert not analyze(td4, verbose=False)["p4"], "最优臂增益 <8% ⇒ P4 应失败"

    # --- 场景 5：cold 不再明显慢 ⇒ P1 失败 ---
    td5, w5 = fresh()
    w5("S_cold.jsonl", 6.0000, 0.780)
    assert not analyze(td5, verbose=False)["p1"], "cold 6.0 > 0.5*7=3.5 ⇒ P1 应失败"
    print("selftest ✔ 整机总时间/五臂统计/P1–P4/五个场景（含 4 个反例）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/ccfa_results/2026-09-13/p12_policy")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    analyze(a.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
