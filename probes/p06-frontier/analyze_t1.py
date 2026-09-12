#!/usr/bin/env python3
"""T1 分析：聚合 sweep 结果，画 ridge，判定主假设（drafter capacity ⟂ draft length）。

输入（二选一）：
  --csv  FILE   列含 depth,gamma,ctx,tok_s[,accepted_len]
  --dir  DIR    目录内的 *.bench.json + *.metrics（尽力解析：文件名 d<d>_g<g>_ctx<c>_r<r>）

输出：<out>/t1.csv、<out>/matrix.csv、<out>/ridge.csv、<out>/summary.md
判定（逐 ctx）：γ*(depth) = argmax_γ 平均 tok/s；spread = max γ* − min γ*
  spread == 0                     ⇒ 强可分离（**正交成立**）
  0 < spread 且 γ* 随 depth 单调   ⇒ **斜 ridge** ⇒ 走 H1b（预测式 horizon，且必须打赢已 ship 适配器）
  其它（非单调）                   ⇒ 不可判定 ⇒ 补重复 / 扩 bs 再判
用法：python analyze_t1.py --csv ... --out ... ｜ python analyze_t1.py --selftest
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "depth": int(r["depth"]), "gamma": int(r["gamma"]), "ctx": int(r["ctx"]),
                "tok_s": float(r.get("tok_s") or "nan"),
                "accepted_len": float(r["accepted_len"]) if r.get("accepted_len") else None,
            })
    return rows


def parse_bench_json(p: Path):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    for k in ("output_throughput", "request_throughput", "tokens_per_second"):
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    return None


def parse_metrics(p: Path):
    """尽力从 /metrics 里取 (accepted, draft) 计数。"""
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    def total(name):
        acc = 0.0
        for m in re.finditer(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", t, re.M):
            try:
                acc += float(m.group(1))
            except ValueError:
                pass
        return acc
    a = total("vllm:spec_decode_num_accepted_tokens_total") or total("vllm:spec_decode_num_accepted_tokens")
    d = total("vllm:spec_decode_num_draft_tokens_total") or total("vllm:spec_decode_num_draft_tokens")
    return {"accepted": a, "draft": d} if (a or d) else None


def load_dir(dirp: Path):
    per = defaultdict(lambda: {"tok_s": [], "accepted": [], "draft": []})
    for f in sorted(dirp.glob("*.bench.json")):
        m = re.match(r"d(\d+)_g(\d+)_ctx(\d+)_r(\d+)", f.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        v = parse_bench_json(f)
        if v is not None:
            per[key]["tok_s"].append(v)
    for f in sorted(dirp.glob("*.metrics")):
        m = re.match(r"d(\d+)_g(\d+)_ctx(\d+)_r(\d+)", f.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        mtr = parse_metrics(f)
        if mtr:
            per[key]["accepted"].append(mtr["accepted"])
            per[key]["draft"].append(mtr["draft"])
    rows = []
    for (d, g, c), v in per.items():
        acc = None
        if v["draft"] and sum(v["draft"]) > 0:
            acc = sum(v["accepted"]) / sum(v["draft"]) * max(g, 1)
        rows.append({"depth": d, "gamma": g, "ctx": c,
                     "tok_s": mean(v["tok_s"]), "accepted_len": acc})
    return rows


def ridge(rows):
    """逐 ctx：γ*(depth) 与 spread。"""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not math.isnan(r["tok_s"]):
            by[r["ctx"]][(r["depth"], r["gamma"])].append(r["tok_s"])
    out = []
    for ctx, cells in sorted(by.items()):
        depths = sorted({d for d, _ in cells})
        gammas = sorted({g for _, g in cells})
        gstar = {}
        for d in depths:
            best, bg = None, None
            for g in gammas:
                m = mean(cells.get((d, g), []))
                if not math.isnan(m) and (best is None or m > best):
                    best, bg = m, g
            if bg is not None:
                gstar[d] = bg
        if not gstar:
            continue
        vals = list(gstar.values())
        spread = max(vals) - min(vals)
        seq = [gstar[d] for d in sorted(gstar)]
        mono = all(b >= a for a, b in zip(seq, seq[1:])) or all(b <= a for a, b in zip(seq, seq[1:]))
        verdict = ("正交成立（强可分离）" if spread == 0 else
                   ("斜 ridge ⇒ 走 H1b（预测式 horizon，须打赢已 ship 适配器）" if mono else
                    "不可判定（非单调）⇒ 补重复/扩 bs"))
        out.append({"ctx": ctx, "gstar": gstar, "spread": spread, "monotone": mono, "verdict": verdict})
    return out


def selftest():
    # 正交：γ* 与 depth 无关
    orth = [{"depth": d, "gamma": g, "ctx": 4096, "tok_s": 100 - abs(g - 3) - 0.1 * d, "accepted_len": g / 2}
            for d in (1, 3, 5) for g in (1, 3, 7)]
    r = ridge(orth)
    assert r and r[0]["spread"] == 0 and "正交成立" in r[0]["verdict"], r
    # 斜 ridge：depth 越大，最优 γ 越大（单调）
    slant = [{"depth": d, "gamma": g, "ctx": 4096,
              "tok_s": 100 - abs(g - d) * 5, "accepted_len": g / 2}
             for d in (1, 3, 5) for g in (1, 3, 7)]
    r = ridge(slant)
    assert r and r[0]["spread"] > 0 and r[0]["monotone"] and "H1b" in r[0]["verdict"], r
    # 非单调
    noisy = [{"depth": 1, "gamma": 1, "ctx": 4096, "tok_s": 10, "accepted_len": 1},
             {"depth": 1, "gamma": 3, "ctx": 4096, "tok_s": 9, "accepted_len": 1},
             {"depth": 3, "gamma": 1, "ctx": 4096, "tok_s": 8, "accepted_len": 1},
             {"depth": 3, "gamma": 3, "ctx": 4096, "tok_s": 11, "accepted_len": 1},
             {"depth": 5, "gamma": 1, "ctx": 4096, "tok_s": 12, "accepted_len": 1},
             {"depth": 5, "gamma": 3, "ctx": 4096, "tok_s": 7, "accepted_len": 1}]
    r = ridge(noisy)
    assert r and "不可判定" in r[0]["verdict"], r
    print("selftest ✔ 三种情形（正交 / 斜 ridge / 非单调）判定均正确")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--dir"); ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rows = load_csv(a.csv) if a.csv else load_dir(Path(a.dir))
    if not rows:
        sys.exit("没有可用数据（检查 --csv/--dir 与文件名格式 d<d>_g<g>_ctx<c>_r<r>）")
    out = Path(a.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "t1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["depth", "gamma", "ctx", "tok_s", "accepted_len"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["ctx"], r["depth"], r["gamma"])):
            w.writerow(r)
    res = ridge(rows)
    with open(out / "ridge.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ctx", "depth", "gstar", "spread", "monotone", "verdict"])
        for r in res:
            for d, g in sorted(r["gstar"].items()):
                w.writerow([r["ctx"], d, g, r["spread"], r["monotone"], r["verdict"]])
    lines = ["# T1 结果（自动生成骨架）", "",
             "## ridge（每 ctx：γ*(depth)）", "", "| ctx | " + " | ".join(f"d{d}" for d in sorted({d for r in res for d in r['gstar']})) + " | spread | 判定 |",
             "|---|" + "---|" * (len({d for r in res for d in r['gstar']}) + 2)]
    for r in res:
        ds = sorted(r["gstar"])
        lines.append("| " + str(r["ctx"]) + " | " + " | ".join(str(r["gstar"][d]) for d in ds) +
                     f" | {r['spread']} | {r['verdict']} |")
    lines += ["", "## 必答项（手填）",
              "- ① 与已 ship 控制器的差异：", "- ② 为什么不能把 per-batch K 查表扩到深度轴：",
              "- ③ adaptive-steps-only vs +depth 增量（T2）：", "- ④ 动机抗辩（vs PRISM）：",
              "- ⑤ 截断下界的不对称性说明：", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出 {out}/t1.csv、ridge.csv、summary.md")
    for r in res:
        print(f"  ctx={r['ctx']}: γ*={r['gstar']} spread={r['spread']} → {r['verdict']}")


if __name__ == "__main__":
    main()
