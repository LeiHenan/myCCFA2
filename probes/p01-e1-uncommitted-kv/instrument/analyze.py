#!/usr/bin/env python3
"""E1 分析器：把 $E1_LOG 的 JSONL 变成 R_byte / R_reserve / H 与决定表。

用法：
    python analyze.py --log /tmp/e1_metrics.jsonl --out results/p01-e1-uncommitted-kv/2026-09-12

输出：out/ratio.csv、out/summary.md（含判据逐条对照）
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict


def pct(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return vals[f] if f == c else vals[f] + (vals[c] - vals[f]) * (k - f)


def load(path):
    specs, allocs = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            (specs if r.get("ev") == "spec" else allocs).append(r)
    return specs, allocs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    specs, allocs = load(a.log)
    if not specs:
        raise SystemExit("日志里没有 spec 记录 —— 确认 E1_LOG 已设置、且确实跑了投机配置")

    rows = []
    # ---- 令牌口径：R_byte = rejected / committed(ctx) ----
    for r in specs:
        ctx = r.get("ctx") or 0
        rej = r.get("rejected") or 0
        draft = r.get("draft") or 0
        rows.append(
            {
                "kind": "spec",
                "req": r.get("req"),
                "ctx": ctx,
                "draft": draft,
                "accepted": r.get("accepted"),
                "rejected": rej,
                "invalid": r.get("invalid"),
                "R_byte": (rej / ctx) if ctx > 0 else None,
                "R_analytic": (draft / ctx) if ctx > 0 else None,
                "R_reserve": None,
                "H": 0,  # 源码事实：reject 在同一步回滚 num_computed_tokens
            }
        )

    # ---- 预留口径：R_reserve = blocks*block_size / committed ----
    by_req_alloc = defaultdict(list)
    for r in allocs:
        by_req_alloc[r.get("req")].append(r)
        ctx = r.get("ctx") or 0
        blocks = r.get("blocks")
        rows.append(
            {
                "kind": "alloc",
                "req": r.get("req"),
                "ctx": ctx,
                "draft": None,
                "accepted": None,
                "rejected": None,
                "invalid": None,
                "R_byte": None,
                "R_analytic": None,
                "R_reserve": ((blocks * a.block_size) / ctx) if (blocks and ctx > 0) else None,
                "H": None,
                "new_tokens": r.get("new_tokens"),
                "blocks": blocks,
            }
        )

    spec_rows = [r for r in rows if r["kind"] == "spec"]
    alloc_rows = [r for r in rows if r["kind"] == "alloc"]

    rb = [r["R_byte"] for r in spec_rows]
    ra = [r["R_analytic"] for r in spec_rows]
    rr = [r["R_reserve"] for r in alloc_rows]

    # 逐请求峰值（p95 单请求口径）
    per_req = defaultdict(list)
    for r in spec_rows:
        per_req[r["req"]].append(r["R_byte"])

    summary = {
        "records": {"spec": len(spec_rows), "alloc": len(alloc_rows), "requests": len(per_req)},
        "R_byte": {"p50": pct(rb, 0.50), "p95": pct(rb, 0.95), "max": max([v for v in rb if v is not None], default=float("nan"))},
        "R_analytic_gamma_over_ctx": {"p50": pct(ra, 0.50), "p95": pct(ra, 0.95)},
        "R_reserve": {"p50": pct(rr, 0.50), "p95": pct(rr, 0.95)},
        "ratio_reserve_over_byte_p95": (
            pct(rr, 0.95) / pct(rb, 0.95) if pct(rb, 0.95) and pct(rb, 0.95) == pct(rb, 0.95) and pct(rb, 0.95) > 0 else None
        ),
        "ctx_range": {
            "min": min((r["ctx"] for r in spec_rows), default=None),
            "max": max((r["ctx"] for r in spec_rows), default=None),
        },
        "gamma_observed": sorted({r["draft"] for r in spec_rows if r["draft"]}),
    }

    with open(os.path.join(a.out, "ratio.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    verdict = (
        "**杀（#1 归档）** —— 全部可行点 p95 < 5%"
        if (summary["R_byte"]["p95"] == summary["R_byte"]["p95"] and summary["R_byte"]["p95"] < 0.05)
        else "**灰区** —— 补测峰值占用与释放延迟后再判"
        if (summary["R_byte"]["p95"] == summary["R_byte"]["p95"] and summary["R_byte"]["p95"] < 0.10)
        else "**活** —— 存在 ≥10% 的点，进入机制设计"
    )

    md = f"""# E1 结果 {a.tag}

## 一、量 / 区间 / 基线
- 量：`R_byte`（令牌口径 = rejected/committed）、`R_reserve`（预留口径 = blocks×block_size/committed）、`H`（驻留步数）
- 区间：ctx {summary['ctx_range']['min']} – {summary['ctx_range']['max']}；观测 γ = {summary['gamma_observed']}
- 基线：部署中实际在跑的投机配置

## 二、数字
| 指标 | p50 | p95 | max |
|---|---|---|---|
| `R_byte`（实测） | {summary['R_byte']['p50']:.5f} | {summary['R_byte']['p95']:.5f} | {summary['R_byte']['max']:.5f} |
| `R_analytic ≈ γ/ctx` | {summary['R_analytic_gamma_over_ctx']['p50']:.5f} | {summary['R_analytic_gamma_over_ctx']['p95']:.5f} | — |
| `R_reserve` | {summary['R_reserve']['p50']:.5f} | {summary['R_reserve']['p95']:.5f} | — |

- `R_reserve / R_byte`（p95）= {summary['ratio_reserve_over_byte_p95']}
- 记录数：spec {summary['records']['spec']} ｜ alloc {summary['records']['alloc']} ｜ 请求数 {summary['records']['requests']}

## 三、判据对照（`notes/prereg/p01-e1-uncommitted-kv.md`）
- [ ] 全部 HBM 可行点 p95 < 5% ⇒ 杀
- [ ] 某可行点 p95 ≥ 10% ⇒ 活
- [ ] 5% ≤ p95 < 10% ⇒ 灰区
- **本结果初判：{verdict}**

## 四、结论与后续
（写 3 行：判据落在哪一档 / 与解析值 γ/ctx 的差额说明了什么 / 下一步动作）

## 五、必须回答的三个陷阱
1. 定额之差：`R_reserve / R_byte` 是否 >2？（>2 说明真正的可优化对象是**保守预留**而非"未提交 KV"）
2. `H`：源码显示 reject 在同一步回滚 `num_computed_tokens` ⇒ 预期 H≈0；若实测出现 H≥1，说明存在额外驻留窗口，需单独记录。
3. 无效草稿：`invalid`（grammar 无效）是否显著？它应从"提出的草稿数"里扣除。
"""
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n已写出：{a.out}/ratio.csv、{a.out}/summary.md")


if __name__ == "__main__":
    main()
