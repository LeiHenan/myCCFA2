#!/usr/bin/env python3
"""p20 DSD 网格判定 —— 读 summary.tsv + *.specdiff.json，按预登记（含"采数前修订 #1"）逐条判 P1a/P1b/P2/P3/P4/P5。

判定规则（来自 notes/prereg/p20-dsd-concurrency.md）：
  · 主指标 `output_throughput`；**逐次重复上报**，冷（rep1）与热（rep≥2）分开。
  · 噪声：同格重复的极差；**k=3 规则** —— 效应须 ≥3× 重复极差才称效应（E1：噪声 >8% ⇒ 提高重复次数）。
  · **P1a**（K 匹配、最干净）：thr(A2_static_k3) vs thr(A6_table_const3) —— 两者 K 都是 3，唯一差异是有没有表。
        相对差 ≥8% 且 A6 更慢 ⇒ 表的存在本身有代价。
  · **P1b**：thr(A1_nospec) vs thr(A4_table_allk0) —— 表存在但全 K=0 vs 完全不投机。
  · **P2**：thr(A2)/thr(A1) —— 投机净收益；<1 记为"投不划算"的独立复现。
  · **P3**：A4 的 num_drafts 增量必须 = 0（证明全 K=0 表确实零草稿）；≠0 ⇒ 本轮作废。
  · **P4**：thr(A1) vs thr(A3_static_k1) —— 最小投机（K=1）净效应（原 P4 口径已随修订 #1 改变）。
  · **P5**：A5（表 [[1,3,3],[4,8192,0]]）在并发 8 下应落在 bs≥4 档 ⇒ K=0 ⇒ num_drafts ≈ 0，吞吐应贴近 A1/A4；
        若反而 ≈ A2/A6 ⇒ 表未被真正查询（对照不成立，本轮作废）。

用法： python3 analyze_dsd.py --dir /root/ccfa_results/2026-09-14/p20_dsd_v2 [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

THRESH = 8.0  # %


def load_tsv(path: str) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            for k in ("rep",):
                row[k] = int(row[k])
            for k in ("out_tput", "req_tput", "med_ttft_ms", "med_tpot_ms", "med_itl_ms", "med_e2el_ms"):
                try:
                    row[k] = float(row[k])
                except ValueError:
                    row[k] = None
            arms.setdefault(row["tag"], []).append(row)
    for v in arms.values():
        v.sort(key=lambda r: r["rep"])
    return arms


def load_specdiff(d: str) -> dict[str, dict]:
    out = {}
    for f in glob.glob(os.path.join(d, "*.specdiff.json")):
        tag = os.path.basename(f)[: -len(".specdiff.json")]
        try:
            out[tag] = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return out


def warm(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["rep"] >= 2] or rows


def rel(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100.0


def spread_pct(vals: list[float]) -> float | None:
    if len(vals) < 2 or not min(vals):
        return None
    return (max(vals) - min(vals)) / min(vals) * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()

    tsv = os.path.join(a.dir, "summary.tsv")
    if not os.path.exists(tsv):
        print(f"找不到 {tsv}", file=sys.stderr)
        return 2
    arms = load_tsv(tsv)
    spec = load_specdiff(a.dir)

    print("=" * 100)
    print("逐臂逐次（主指标 output_throughput tok/s）")
    print("=" * 100)
    print(f"{'臂':16s} {'rep1(冷)':>10s} {'rep2':>10s} {'rep3':>10s} {'热 p50':>10s} {'热极差%':>9s} {'冷/热%':>8s}")
    stats = {}
    for tag in sorted(arms):
        rows = arms[tag]
        o = [r["out_tput"] for r in rows if r["out_tput"] is not None]
        w = [r["out_tput"] for r in warm(rows) if r["out_tput"] is not None]
        cold = o[0] if o else None
        wm = st.median(w) if w else None
        sp = spread_pct(w)
        stats[tag] = {"cold": cold, "warm": wm, "warm_vals": w, "spread_pct": sp}
        cells = [f"{v:10.3f}" if v is not None else f"{'—':>10s}" for v in (o + [None, None])[:3]]
        print(f"{tag:16s} {cells[0]} {cells[1]} {cells[2]} "
              f"{(f'{wm:10.3f}' if wm else '—'):>10s} "
              f"{(f'{sp:9.2f}' if sp is not None else '—'):>9s} "
              f"{(f'{rel(cold, wm):8.2f}' if rel(cold, wm) is not None else '—'):>8s}")

    def wm(tag):
        return stats.get(tag, {}).get("warm")

    print()
    print("=" * 100)
    print("预登记判据")
    print("=" * 100)
    verdicts = {}

    def noise_of(*tags):
        vals = [stats[t]["spread_pct"] for t in tags if t in stats and stats[t]["spread_pct"] is not None]
        return max(vals) if vals else None

    # ---- P1a
    d = rel(wm("A6_table_const3"), wm("A2_static_k3"))
    n = noise_of("A2_static_k3", "A6_table_const3")
    if d is None:
        verdicts["P1a"] = "无法判定（缺臂）"
    else:
        hit = d <= -THRESH
        resolvable = (n is None) or (abs(d) >= 3 * n)
        verdicts["P1a"] = (f"{d:+.2f}%  (A6 相对 A2{'更慢' if d < 0 else '更快'})；"
                           f"噪声={n if n is None else round(n, 2)}%；"
                           f"{'≥8% 且方向为 A6 更慢 ⇒ **H1 成立**' if hit else '<8% 或方向相反 ⇒ **H1 否证（K 匹配）**'}"
                           f"{'' if resolvable else '；但效应未过 3× 噪声 ⇒ 不可分辨'}")
    # ---- P1b
    d = rel(wm("A4_table_allk0"), wm("A1_nospec"))
    n = noise_of("A1_nospec", "A4_table_allk0")
    if d is None:
        verdicts["P1b"] = "无法判定（缺臂）"
    else:
        hit = d <= -THRESH
        verdicts["P1b"] = (f"{d:+.2f}%  (A4 相对 A1{'更慢' if d < 0 else '更快'})；噪声={n if n is None else round(n, 2)}%；"
                           f"{'≥8% 且 A4 更慢 ⇒ **零草稿表相对不投机也有代价（H1 的第二种形态）**' if hit else '<8% ⇒ **零草稿表相对不投机无可测代价**'}")
    # ---- P2
    d = rel(wm("A2_static_k3"), wm("A1_nospec"))
    verdicts["P2"] = (f"thr(A2)/thr(A1) = {(wm('A2_static_k3') / wm('A1_nospec')):.4f}"
                      if wm("A2_static_k3") and wm("A1_nospec") else "无法判定"
                      )
    # ---- P3 / P5 (drafts from specdiff)
    def drafts(tag):
        s = spec.get(tag) or {}
        return s.get("num_drafts")
    for pid, tag, expect_zero in (("P3", "A4_table_allk0", True), ("P5", "A5_table_switch", True)):
        dv = drafts(tag)
        if dv is None:
            verdicts[pid] = f"{tag} 无 specdiff（无法判定）"
        elif expect_zero:
            verdicts[pid] = (f"{tag} num_drafts={dv} ⇒ "
                             + ("**符合预测（0 草稿）**" if dv == 0 else "**不符（表中 K 未被遵守或表未被查询）**"))
        else:
            verdicts[pid] = f"{tag} num_drafts={dv}"
    # P3 的对照有效性另需 A2 有草稿
    verdicts["P3 对照有效性"] = (f"A2 num_drafts={drafts('A2_static_k3')}（应 ≫0，否则整个投机路径没跑）")
    # ---- P4
    d = rel(wm("A3_static_k1"), wm("A1_nospec"))
    verdicts["P4"] = (f"{d:+.2f}%  (A3 静态 K=1 相对不投机)" if d is not None else "无法判定")
    # ---- 附带
    for pair in (("A5_table_switch", "A6_table_const3"), ("A6_table_const3", "A2_static_k3")):
        d = rel(wm(pair[0]), wm(pair[1]))
        if d is not None:
            verdicts[f"附带 {pair[0]}/{pair[1]}"] = f"{d:+.2f}%"

    for k, v in verdicts.items():
        print(f"  {k:14s}: {v}")

    print()
    print("=" * 100)
    print("投机计数（/metrics 做差）")
    print("=" * 100)
    for tag in sorted(spec):
        s = spec[tag]
        print(f"  {tag:16s} drafts={s.get('num_drafts')} draft_tokens={s.get('num_draft_tokens')} "
              f"accepted={s.get('num_accepted_tokens')} accept_len={s.get('accept_len')} "
              f"accept_rate={s.get('accept_rate')}")

    # 噪声预警（E1 触发检查）
    print()
    loud = {t: s["spread_pct"] for t, s in stats.items() if s["spread_pct"] and s["spread_pct"] > THRESH}
    if loud:
        print(f"⚠️ E1 触发：以下臂热重复极差 >{THRESH}% ⇒ 预登记要求把重复提高到 5 次：{loud}")
    else:
        print(f"✅ 无臂热重复极差 >{THRESH}%（E1 未触发）")

    if a.json:
        json.dump({"arms": stats, "verdicts": verdicts, "specdiff": spec},
                  open(a.json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\n已写 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
