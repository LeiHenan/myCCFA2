#!/usr/bin/env python3
"""p12 R 阶段判据计算器：恢复成本 vs 生成成本、恢复等价性、合并语料收益。

对应 `notes/prereg/sglang-draft-corpus-r.md`（采数前冻结）：
  R1  cost_restore < 0.15 × cost_gen
  R2  |accept_restored − accept_regen| / accept_regen < 1%
  R3  accept_combo ≥ accept_restored − 1%
  跨进程  |accept_xproc − accept_restored| / accept_restored < 1%

用法：
  python compare_restore.py --dir /root/ccfa_results/2026-09-13/p12_restore
  python compare_restore.py --selftest
"""
import argparse
import glob
import json
import os
import statistics as st
import sys


def load_rows(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return [r for r in out if not r.get("err")]


def accept(path):
    rs = load_rows(path)
    if not rs:
        return None
    return st.fmean([r["spec_accept_length"] for r in rs if r.get("spec_accept_length") is not None])


def cost_gen(path):
    """生成一遍的 GPU 忙时 = 32 请求 e2e_latency 之和。"""
    rs = load_rows(path)
    vals = [r.get("e2e_latency") for r in rs if r.get("e2e_latency") is not None]
    return sum(vals) if vals else None


def cost_restore(cost_file, corpus_id):
    for r in load_rows(cost_file):
        if r.get("success") and (not corpus_id or corpus_id in str(r.get("_id", corpus_id))):
            pass
    rows = load_rows(cost_file)
    return [r["wall_s"] for r in rows if r.get("wall_s") is not None]


def rel(a, b):
    return None if (a is None or b in (None, 0)) else abs(a - b) / b


def analyze(d, verbose=True):
    g = lambda name: os.path.join(d, name)  # noqa: E731
    ap_gen = accept(g("R_gen.jsonl"))
    ap_cold = accept(g("R_cold.jsonl"))
    ap_regen = accept(g("R_regen.jsonl"))
    ap_docs = accept(g("R_restore_docs.jsonl"))
    ap_combo = accept(g("R_restore_combo.jsonl"))
    ap_xproc = accept(g("R_restore_xproc.jsonl"))
    cg = cost_gen(g("R_gen.jsonl"))
    costs = cost_restore(g("restore_cost.jsonl"), None)

    r1 = (costs and cg and max(costs) < 0.15 * cg)
    r2 = rel(ap_docs, ap_regen)
    r3 = (ap_combo is not None and ap_docs is not None and ap_combo >= ap_docs * 0.99)
    rxp = rel(ap_xproc, ap_docs)
    cold_ok = ap_cold is not None and abs(ap_cold - 1.8360) < 0.01

    if verbose:
        print("=== 首通接受长度（同 prompt 集同序、串行、每臂独立全新进程）===")
        print(f"  R_gen（无语料，生成用）      {ap_gen!r}")
        print(f"  R_cold（冷启动对照）         {ap_cold!r}")
        print(f"  R_regen（重新生成语料）      {ap_regen!r}")
        print(f"  R_restore_docs（落盘恢复）   {ap_docs!r}")
        print(f"  R_restore_combo（+prompt）   {ap_combo!r}")
        print(f"  R_restore_xproc（另一进程）  {ap_xproc!r}")
        print("\n=== 代价 ===")
        print(f"  cost_gen（32 请求 GPU 忙时之和）{cg if cg is None else round(cg, 3)} s")
        print(f"  cost_restore 各次             {[round(c, 4) for c in costs]} s")
        if costs and cg:
            print(f"  恢复/生成 比值                {max(costs) / cg * 100:.3f}%")
        print("\n=== 判据 ===")
        print(f"  R1 恢复 < 15% 生成成本   : {'✅' if r1 else '❌'}")
        print(f"  R2 恢复 vs 生成 相对差   : {'未判定' if r2 is None else f'{r2 * 100:.3f}%'} "
              f"{'✅' if (r2 is not None and r2 < 0.01) else '❌'}")
        print(f"  R3 合并 >= 仅续写 −1%    : {'✅' if r3 else '❌'}")
        print(f"  跨进程 相对差            : {'未判定' if rxp is None else f'{rxp * 100:.3f}%'} "
              f"{'✅' if (rxp is not None and rxp < 0.01) else '❌'}")
        print(f"  冷启动对照复现 1.8360    : {'✅' if cold_ok else '❌'}  (实测 {ap_cold})")
        if not cold_ok:
            print("  ⇒ 按预登记：R_cold 未复现 ⇒ 本轮作废")
        elif not r1:
            print("  ⇒ 按预登记：恢复不比生成便宜 ⇒ **杀 R 方向**")
        elif r2 is not None and r2 < 0.01 and rxp is not None and rxp < 0.01:
            print("  ⇒ 恢复路径成立：更便宜且等价（含跨进程可移植）")
    return dict(ap_gen=ap_gen, ap_cold=ap_cold, ap_regen=ap_regen, ap_docs=ap_docs,
                ap_combo=ap_combo, ap_xproc=ap_xproc, cost_gen=cg, costs=costs,
                r1=bool(r1), r2=r2, r3=bool(r3), rxp=rxp, cold_ok=bool(cold_ok))


def selftest():
    import tempfile
    td = tempfile.mkdtemp()
    def w(name, acc, e2e=0.78):
        with open(os.path.join(td, name), "w", encoding="utf-8") as fh:
            for i in range(32):
                fh.write(json.dumps({"idx": i, "err": None, "spec_accept_length": acc,
                                     "e2e_latency": e2e}) + "\n")
    w("R_gen.jsonl", 1.8360); w("R_cold.jsonl", 1.8360)
    w("R_regen.jsonl", 6.6787); w("R_restore_docs.jsonl", 6.6787)
    w("R_restore_combo.jsonl", 6.7000); w("R_restore_xproc.jsonl", 6.6787)
    with open(os.path.join(td, "restore_cost.jsonl"), "w", encoding="utf-8") as fh:
        for ws in (1.2, 3.5, 1.1):
            fh.write(json.dumps({"success": True, "wall_s": ws}) + "\n")
    o = analyze(td, verbose=False)
    assert o["cold_ok"] and o["r1"] and o["r2"] == 0 and o["r3"], o
    assert o["rxp"] == 0, o["rxp"]
    assert abs(o["cost_gen"] - 0.78 * 32) < 1e-6, o["cost_gen"]
    # 反例：恢复很贵 ⇒ R1 失败
    with open(os.path.join(td, "restore_cost.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"success": True, "wall_s": 30.0}) + "\n")
    assert not analyze(td, verbose=False)["r1"]
    # 反例：恢复不等价 ⇒ R2 失败
    w("R_restore_docs.jsonl", 5.0)
    assert analyze(td, verbose=False)["r2"] > 0.01
    print("selftest ✔ R1/R2/R3/跨进程/冷启动复现/两个反例")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/ccfa_results/2026-09-13/p12_restore")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    analyze(a.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
