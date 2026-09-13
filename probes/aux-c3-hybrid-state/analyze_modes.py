#!/usr/bin/env python3
"""aux-c3 第 2 步判定器：三种 mamba 缓存模式 × 两个显存预算 → 容量/吞吐/复用

判据（预登记见 `notes/prereg/aux-c3-hybrid-state.md`，**写在采数之前**）：
  1. **容量代价**：`Maximum concurrency` 在模式间差 ≥8%
  2. **吞吐代价**：`none` vs `align` 的 tok/s 差 ≥8%（且 > 3× 重复极差才算）
  3. **复用收益**：重复前缀的 TTFT / prefix-cache 命中计数
  4. **粒度权衡**：`all` 的 block 从 default 降到 128 时容量单调下降（验证 `检查点开销/KV = L*/b`）
  杀判据：三模式吞吐差 <8% 且容量差 <8% ⇒ 杀 P1/P3。

用法：
  python analyze_modes.py --dir <run_modes.sh 的 OUT 目录>
  python analyze_modes.py --selftest
"""

import argparse
import collections
import glob
import json
import os
import re
import statistics as st
import sys

BENCH = re.compile(r"^(u[\d.]+)_(none|align|all)(?:_b(\d+))?_d_\2(?:_b\d+)?_c(\d+)_r(\d+)\.bench\.json$")


def load(root):
    """→ {(util, mode, block, conc): {rep: record}}；block='default' 表示未显式设置"""
    per = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(root, "bs*", "*.bench.json"))):
        b = os.path.basename(f)
        m = re.match(r"^u?([\d.]+)_(none|align|all)(?:_b(\d+))?_d_\2(?:_b(\d+))?_c(\d+)_r(\d+)\.bench\.json$", b)
        if not m:
            continue
        util, mode, blk1, blk2, conc, rep = m.groups()   # 文件名带 u 前缀，CSV 里不带 ⇒ 统一去掉
        per[(util, mode, blk1 or blk2 or "default", int(conc))][int(rep)] = json.load(open(f))
    return per


def preemptions(root):
    out = {}
    for f in glob.glob(os.path.join(root, "bs*", "*.metrics")):
        m = re.match(r"^u?([\d.]+)_(none|align|all)(?:_b(\d+))?_d_\2(?:_b(\d+))?_c(\d+)_r(\d+)\.metrics$",
                     os.path.basename(f))
        if not m:
            continue
        util, mode, blk1, blk2, conc, _ = m.groups()
        mm = re.search(r"^vllm:num_preemptions_total\{[^}]*\}\s+([\d.eE+]+)", open(f, errors="ignore").read(), re.M)
        if mm:
            k = (util, mode, blk1 or blk2 or "default", int(conc))
            out[k] = max(out.get(k, 0.0), float(mm.group(1)))
    return out


def stat(recs, key):
    vs = [r.get(key) for r in recs if isinstance(r.get(key), (int, float))]
    if not vs:
        return None
    return {"mean": st.mean(vs), "min": min(vs), "max": max(vs),
            "spread_pct": (max(vs) - min(vs)) / st.mean(vs) * 100 if st.mean(vs) else float("nan"), "n": len(vs)}


def analyze(root):
    per = load(root)
    if not per:
        return {"error": f"{root}/bs*/ 下没有 bench.json"}
    pre = preemptions(root)
    rows = []
    for (util, mode, block, conc), reps in sorted(per.items()):
        recs = [reps[r] for r in sorted(reps)]
        tok = stat(recs, "output_throughput")
        ttft = stat(recs, "mean_ttft_ms")
        # 命中计数（counter）：从 .metrics 取最后一次
        rows.append({
            "util": util, "mode": mode, "block": block, "conc": conc, "n": len(recs),
            "tok_s": round(tok["mean"], 1) if tok else None,
            "tok_spread_pct": round(tok["spread_pct"], 1) if tok else None,
            "ttft_ms": round(ttft["mean"], 1) if ttft else None,
            "preempt": pre.get((util, mode, block, conc)),
        })
    # 容量读数来自 summary.csv（serve 日志）
    cap = {}
    csv = os.path.join(root, "summary.csv")
    if os.path.exists(csv):
        for ln in open(csv, encoding="utf-8").read().splitlines()[1:]:
            p = ln.split(",")
            if len(p) >= 6 and p[4]:
                cap[(p[1], p[2], p[3])] = (int(p[4]), float(p[5]) if p[5] else None)

    # 判定
    verdicts = []
    def gap(a, b):
        if not a or not b or not a["tok_s"] or not b["tok_s"]:
            return None
        return (a["tok_s"] / b["tok_s"] - 1) * 100

    for util in sorted({r["util"] for r in rows}):
        for conc in sorted({r["conc"] for r in rows}):
            sel = {r["mode"]: r for r in rows if r["util"] == util and r["conc"] == conc and r["block"] == "default"}
            if "none" in sel and "align" in sel:
                d = gap(sel["align"], sel["none"])
                noise = max(sel["none"]["tok_spread_pct"] or 0, sel["align"]["tok_spread_pct"] or 0)
                resolvable = d is not None and abs(d) > 3 * noise
                verdicts.append(f"util={util} bs={conc}: align vs none 吞吐差 **{d:+.1f}%**（噪声 {noise:.1f}%，"
                                f"{'可分辨' if resolvable else '**不可分辨**'}）")
                if resolvable and abs(d) >= 8:
                    verdicts.append(f"  ⇒ ✅ 判据 2 成立（≥8% 且可分辨）：P1/P3 有吞吐证据")
    caps = sorted({k[1] for k in cap})
    for util in sorted({k[0] for k in cap}):
        line = []
        for mode in caps:
            for blk in ("default", "1024", "128"):
                if (util, mode, blk) in cap:
                    line.append(f"{mode}/{blk}: {cap[(util, mode, blk)][1]}x ({cap[(util, mode, blk)][0]} tok)")
        if line:
            verdicts.append(f"util={util} 容量（最大并发）: " + " ｜ ".join(line))
    bad = [r for r in rows if r.get("preempt")]
    verdicts.append("⚠️ 存在 preemption>0 的格：" + str([(r["util"], r["mode"], r["block"], r["conc"], r["preempt"]) for r in bad])
                    if bad else "✅ 全部格 preemption = 0（吞吐主张可采纳）")
    return {"rows": rows, "cap": cap, "verdicts": verdicts}


def summarize(res):
    if "error" in res:
        return res["error"]
    L = ["# aux-c3 第 2 步判定：mamba 缓存模式 × 显存预算", ""] + res["verdicts"] + ["",
         "| util | mode | block | bs | n | tok/s | 重复极差 | TTFT(ms) | preempt |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['util']} | {r['mode']} | {r['block']} | {r['conc']} | {r['n']} | **{r['tok_s']}** | "
                 f"{r['tok_spread_pct']}% | {r['ttft_ms']} | {r['preempt']} |")
    L += ["", "> 判据与杀判据见 `notes/prereg/aux-c3-hybrid-state.md`（采数前登记）。",
          "> `preempt` 取自 `.metrics` 的 `num_preemptions_total`（**计数器**；gauge 跑完归零不可用）。"]
    return "\n".join(L) + "\n"


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # align 比 none 快 20%（可分辨）⇒ 判据 2 成立
        cases = {("0.30", "none", None): 100.0, ("0.30", "align", None): 120.0}
        for (util, mode, blk), base in cases.items():
            d = os.path.join(td, f"bs32")
            os.makedirs(d, exist_ok=True)
            for r in (1, 2, 3):
                v = base + (r - 2) * 1.0
                name = f"{util}_{mode}_d_{mode}_c32_r{r}"
                json.dump({"output_throughput": v, "mean_ttft_ms": 100.0},
                          open(os.path.join(d, name + ".bench.json"), "w"))
                open(os.path.join(d, name + ".metrics"), "w").write(
                    'vllm:num_preemptions_total{engine="0"} 0.0\n')
        open(os.path.join(td, "summary.csv"), "w").write(
            "tag,util,mode,block,kv_tokens,max_conc,tok_s,mean_ttft_ms,accept\n"
            "u0.30_none,0.30,none,default,300000,73.2,100.0,100,\n"
            "u0.30_align,0.30,align,default,250000,61.0,120.0,100,\n")
        res = analyze(td)
        assert any("+20.0%" in v for v in res["verdicts"]), res["verdicts"]
        assert any("判据 2 成立" in v for v in res["verdicts"]), res["verdicts"]
        assert any("preemption = 0" in v for v in res["verdicts"]), res["verdicts"]
        assert any("容量（最大并发）" in v for v in res["verdicts"]), res["verdicts"]
        assert "tok/s" in summarize(res)
        print("selftest ✔ 加载/重复极差/align-vs-none 判定/容量读数/preemption 前置条件/输出")


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
        open(os.path.join(a.out, "modes.md"), "w", encoding="utf-8").write(txt)
        keys = ["util", "mode", "block", "conc", "n", "tok_s", "tok_spread_pct", "ttft_ms", "preempt"]
        with open(os.path.join(a.out, "modes.csv"), "w", encoding="utf-8") as fh:
            fh.write(",".join(keys) + "\n")
            for r in res["rows"]:
                fh.write(",".join(str(r.get(k)) for k in keys) + "\n")
        print(f"已写出 {a.out}/modes.md 与 modes.csv")
