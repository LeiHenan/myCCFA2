#!/usr/bin/env python3
"""p10 判定器：算出前缀复用的**生效点（onset）**，并交叉验证。

判据见 `notes/prereg/prefix-reuse-lag.md`（**采数前**）：
  C1 复现：第 2 次命中增量为 0、第 3 次 > 0
  C2 交叉验证：命中出现的那个 rep，mean_ttft_ms 相对前一 rep 下降 ≥30%
  C3 依赖关系：onset 随 并发{1,32} 与 池大小{0.30,0.95} 的变化
  C4 可回收性：存在不干预负载的手段把 onset 从 3 提到 2（由后续实验判定；本器只报事实）

数据：`run_onset.sh` 产出的 `bs<N>/<tag>_c<N>_r<R>.{pre,post}.metrics` 与 `.bench.json`。
**计数器必须做差**（跨 rep 累计），并且每个 rep 用**该 rep 自己的 pre→post 增量**。

用法：
  python analyze_onset.py --dir <OUT> [--out 目录]
  python analyze_onset.py --selftest
"""

import argparse
import glob
import json
import os
import re
import statistics as st
import sys

F = re.compile(r"^(u[\d.]+)_(off|d5g3)_c(\d+)_r(\d+)\.bench\.json$")


def metric(path, name):
    if not path or not os.path.exists(path):
        return None
    m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}}\s+([\d.eE+]+)", open(path, errors="ignore").read(), re.M)
    return float(m.group(1)) if m else None


def analyze(root):
    cells = {}
    for f in sorted(glob.glob(os.path.join(root, "bs*", "*.bench.json"))):
        m = F.match(os.path.basename(f))
        if not m:
            continue
        util, spec, conc, rep = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        base = f[: -len(".bench.json")]
        d = json.load(open(f))
        hits_pre = metric(base + ".pre.metrics", "vllm:prefix_cache_hits_total")
        hits_post = metric(base + ".post.metrics", "vllm:prefix_cache_hits_total")
        q_pre = metric(base + ".pre.metrics", "vllm:prefix_cache_queries_total")
        q_post = metric(base + ".post.metrics", "vllm:prefix_cache_queries_total")
        cells[(util, spec, conc, rep)] = {
            "tok_s": d.get("output_throughput"),
            "ttft": d.get("mean_ttft_ms"),
            "hits_delta": (hits_post - hits_pre) if (hits_post is not None and hits_pre is not None) else None,
            "q_delta": (q_post - q_pre) if (q_post is not None and q_pre is not None) else None,
        }
    if not cells:
        return {"error": f"{root}/bs*/ 下没有 u*_*.bench.json"}

    groups = {}
    for (util, spec, conc, rep), c in cells.items():
        groups.setdefault((util, spec, conc), {})[rep] = c

    rows, verdicts = [], []
    for key in sorted(groups):
        util, spec, conc = key
        reps = groups[key]
        seq = [reps[r] for r in sorted(reps)]
        onset = next((i + 1 for i, c in enumerate(seq) if (c["hits_delta"] or 0) > 0), None)
        ttfts = [c["ttft"] for c in seq]
        toks = [c["tok_s"] for c in seq]
        rows.append({"util": util, "spec": spec, "conc": conc, "n": len(seq),
                     "onset": onset,
                     "hits": [c["hits_delta"] for c in seq],
                     "q": [c["q_delta"] for c in seq],
                     "tok_s": [None if t is None else round(t, 1) for t in toks],
                     "ttft": [None if t is None else round(t, 1) for t in ttfts]})
        # C1 / C2
        if onset == 2 or (onset == 3 and (seq[1]["hits_delta"] or 0) == 0):
            verdicts.append(f"**C1 复现** util={util} spec={spec} bs={conc}：命中增量 = {[c['hits_delta'] for c in seq]}"
                            f" ⇒ **第 2 次仍零命中、onset={onset}**" if onset else
                            f"**C1 失败** util={util} spec={spec} bs={conc}：第 2 次即命中 ⇒ 判死信号")
        if onset and onset > 1:
            pre_t, hit_t = ttfts[onset - 2], ttfts[onset - 1]
            if pre_t and hit_t:
                drop = (1 - hit_t / pre_t) * 100
                verdicts.append(f"**C2 交叉验证** util={util} spec={spec} bs={conc}：onset 前 TTFT {pre_t:.0f} ms → "
                                f"onset 时 {hit_t:.0f} ms（**{drop:+.1f}%**）⇒ "
                                + ("✅ TTFT 阶梯成立（真复用）" if drop >= 30 else "⚠️ TTFT 未显著下降 ⇒ 疑似指标口径问题"))
        if len(toks) >= 3 and all(toks):
            verdicts.append(f"**吞吐** util={util} spec={spec} bs={conc}：" + " / ".join(f"{t:.0f}" for t in toks)
                            + f"（极差 {max(toks) / min(toks):.2f}×）")
    return {"rows": rows, "verdicts": verdicts}


def summarize(res):
    if "error" in res:
        return res["error"]
    L = ["# p10 前缀复用 onset 判定", ""] + res["verdicts"] + ["",
         "| util | spec | bs | reps | **onset** | 命中增量（逐次） | 查询增量（逐次） | 吞吐（逐次） | TTFT（逐次） |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['util']} | {r['spec']} | {r['conc']} | {r['n']} | **{r['onset']}** | "
                 f"{r['hits']} | {r['q']} | {r['tok_s']} | {r['ttft']} |")
    L += ["", "> `onset` = 第几次重复首次出现命中增量 > 0（1 = 首次就命中，None = 全程未命中）。",
          "> 计数器为**增量**（每 rep 自己的 pre→post），避免累计值陷阱（decision #76）。",
          "> 判据见 `notes/prereg/prefix-reuse-lag.md`（采数前登记）。"]
    return "\n".join(L) + "\n"


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "bs32")
        os.makedirs(d)
        def w(tag, r, hits_pre, hits_post, tok, ttft, q_pre=0.0, q_post=131328.0):
            b = os.path.join(d, f"{tag}_c32_r{r}")
            json.dump({"output_throughput": tok, "mean_ttft_ms": ttft}, open(b + ".bench.json", "w"))
            open(b + ".pre.metrics", "w").write(
                f'vllm:prefix_cache_hits_total{{engine="0"}} {hits_pre}\n'
                f'vllm:prefix_cache_queries_total{{engine="0"}} {q_pre}\n')
            open(b + ".post.metrics", "w").write(
                f'vllm:prefix_cache_hits_total{{engine="0"}} {hits_post}\n'
                f'vllm:prefix_cache_queries_total{{engine="0"}} {q_post}\n')
        # onset=3：第 1、2 次零命中，第 3 次命中，TTFT 大幅下降、吞吐翻倍
        w("u0.95_d5g3", 1, 0, 0, 700, 300)
        w("u0.95_d5g3", 2, 0, 0, 700, 300)
        w("u0.95_d5g3", 3, 0, 130560, 1900, 90)
        res = analyze(td)
        assert "error" not in res, res
        r = res["rows"][0]
        assert r["onset"] == 3, r
        assert any("C1 复现" in v for v in res["verdicts"]), res["verdicts"]
        assert any("C2 交叉验证" in v and "✅" in v for v in res["verdicts"]), res["verdicts"]
        assert "onset" in summarize(res)
        print("selftest ✔ 增量解析/onset 判定/C1/C2 交叉验证/吞吐汇总/输出")


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
        open(os.path.join(a.out, "onset.md"), "w", encoding="utf-8").write(txt)
        print(f"已写出 {a.out}/onset.md")
