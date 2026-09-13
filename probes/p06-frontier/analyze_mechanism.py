#!/usr/bin/env python3
"""机制层分解器 —— 把 depth 的吞吐差异拆成「接受长度」与「每步代价」两个因子。

**为什么需要它（2026-09-13 设计审计，decision #66）**：
A 系列此前只读了 `output_throughput` 一个字段 ⇒ 只能回答"谁快"，不能回答**为什么快**，
而用户的两层判据里 ⑦ 明确要求"可归因机制，不是显存/排队假象"。
重读 `bench.json` 发现机制数据**当时就已经采到了但从没被分析**：
  - `spec_decode_acceptance_length`（平均接受长度）
  - `spec_decode_per_position_acceptance_rates`（逐位置接受率，list[γ]）
  - `mean_itl_ms`（**每步**间隔）⇒ `step_ms = mean_itl_ms` = **每个 verify 步的代价**
  - 另存 `.metrics` 里的 `num_preemptions_total`（**计数器**，唯一在跑完后仍有意义的字段）
    与 serve log 里的 `GPU KV cache size`（启动期容量）

**恒等式**（2026-09-13 在 A1 的 18 个格上**实测核对过**，不是假设）：
  - 投机解码下 `mean_itl_ms` 是**每步**延迟（一步可吐出 accept_len 个 token）：
    `mean_tpot_ms ≈ mean_itl_ms / accept_len`（实测 18/18 格吻合到 <1%：d1γ3 bs=1 → 14.1/1.33 = 10.6 = tpot）
  - ⇒ 单序列 `tok/s ≈ 1000·accept_len / step_ms = 1000 / tpot_ms`
  - ⇒ **`Δln(tok/s) = Δln(accept_len) − Δln(step_ms)`**：优势来自"接受更长"还是"每步更便宜"，可**逐项归因**。
  ⚠️ 初版把 `step_ms` 写成 `itl × accept_len`（把 ITL 误当每 token 延迟）⇒ 自测当场抓到，已改正。

用法：
  python analyze_mechanism.py --dir <结果根> [--serve-log-glob '*.serve.log'] [--out 目录]
  python analyze_mechanism.py --selftest
"""

import argparse
import collections
import glob
import json
import math
import os
import re
import statistics as st
import sys

FNAME = re.compile(r"d(\d+)_g(\d+)_ctx(\d+)_r(\d+)\.bench\.json$")
KVLINE = re.compile(r"GPU KV cache size:\s*([\d,]+) tokens")


def _f(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def load_bench(root):
    """→ {(bs, depth, gamma, ctx): {rep: record}}"""
    per = collections.defaultdict(dict)
    for sub in sorted(glob.glob(os.path.join(root, "bs*"))):
        m2 = re.match(r"bs(\d+)$", os.path.basename(sub))
        if not m2:
            continue
        bs = int(m2.group(1))
        for f in sorted(glob.glob(os.path.join(sub, "*.bench.json"))):
            m = FNAME.search(os.path.basename(f))
            if not m:
                continue
            dep, gam, ctx, rep = (int(x) for x in m.groups())
            try:
                d = json.load(open(f))
            except Exception:
                continue
            per[(bs, dep, gam, ctx)][rep] = d
    return per


def kv_capacity(serve_log_glob):
    """serve log → {(depth, gamma): 可用 KV tokens}（文件名形如 d5_g7.serve.log）"""
    out = {}
    for f in glob.glob(serve_log_glob):
        m = re.search(r"d(\d+)_g(\d+)\.serve\.log$", os.path.basename(f))
        if not m:
            continue
        try:
            txt = open(f, errors="ignore").read()
        except Exception:
            continue
        hits = KVLINE.findall(txt)
        if hits:
            out[(int(m.group(1)), int(m.group(2)))] = int(hits[-1].replace(",", ""))
    return out


def preemptions(root):
    """→ (bs, depth, gamma, ctx) -> 该格累计 preemption 次数（计数器取各 rep 最大值即可）。

    ⚠️ `.metrics` 是**跑完之后**抓的 ⇒ `kv_cache_usage_perc` 等 gauge 一律已归零、不可用；
    只有 `*_total` 计数器有意义。首次审计就把这一点记进 decision #66。
    """
    out = {}
    for sub in sorted(glob.glob(os.path.join(root, "bs*"))):
        m2 = re.match(r"bs(\d+)$", os.path.basename(sub))
        if not m2:
            continue
        bs = int(m2.group(1))
        for f in glob.glob(os.path.join(sub, "*.metrics")):
            m = re.match(r"d(\d+)_g(\d+)_ctx(\d+)_r(\d+)\.metrics$", os.path.basename(f))
            if not m:
                continue
            dep, gam, ctx, _ = (int(x) for x in m.groups())
            try:
                txt = open(f, errors="ignore").read()
            except Exception:
                continue
            mm = re.search(r"^vllm:num_preemptions_total\{[^}]*\}\s+([\d.eE+]+)", txt, re.M)
            if mm:
                v = float(mm.group(1))
                k = (bs, dep, gam, ctx)
                out[k] = max(out.get(k, 0.0), v)
    return out


def cell(rep_map, mode="steady"):
    """把一个格的多条 bench.json 聚合成机制指标。"""
    recs = [rep_map[r] for r in sorted(rep_map)]
    if not recs:
        return None
    half = max(1, len(recs) // 2)
    use = recs[-half:] if mode == "steady" else recs

    def avg(key):
        vs = [_f(r.get(key)) for r in use]
        vs = [v for v in vs if v is not None]
        return st.mean(vs) if vs else None

    acc = avg("spec_decode_acceptance_length")
    itl = avg("mean_itl_ms")
    tpot = avg("mean_tpot_ms")
    # 逐位置接受率：按 γ 对齐后逐位置取均值
    curves = [r.get("spec_decode_per_position_acceptance_rates") for r in use]
    curves = [c for c in curves if isinstance(c, list) and c]
    perpos = [st.mean(c[i] for c in curves if i < len(c))
              for i in range(max(len(c) for c in curves))] if curves else []
    # step_ms = mean_itl_ms（每步延迟，实测 tpot = itl/accept_len）
    steps = [_f(r["mean_itl_ms"]) for r in use if _f(r.get("mean_itl_ms"))]
    return {
        "n": len(use),
        "tok_s": avg("output_throughput"),
        "accept_len": acc,
        "itl_ms": itl,
        "tpot_ms": tpot,
        "step_ms": st.mean(steps) if steps else None,
        "drafts": avg("spec_decode_num_drafts"),
        "accept_rate_pct": avg("spec_decode_acceptance_rate"),
        "ttft_ms": avg("mean_ttft_ms"),
        "perpos": perpos,
        "reps_tok_s": [_f(r.get("output_throughput")) for r in recs],
    }


def analyze(root, mode="steady", serve_log_glob=None):
    per = load_bench(root)
    if not per:
        return {"error": f"{root} 下没有可用 bench.json"}
    cells = {k: cell(v, mode) for k, v in per.items()}
    pre = preemptions(root)
    kv = kv_capacity(serve_log_glob) if serve_log_glob else {}
    bs_all = sorted({k[0] for k in cells})
    gam_all = sorted({k[2] for k in cells})
    ctx_all = sorted({k[3] for k in cells})
    dep_all = sorted({k[1] for k in cells})

    rows, attrib = [], []
    for ctx in ctx_all:
        for bs in bs_all:
            for g in gam_all:
                have = [d for d in dep_all if (bs, d, g, ctx) in cells]
                if len(have) < 2:
                    continue
                base_d = have[0]
                base = cells[(bs, base_d, g, ctx)]
                for d in have:
                    c = cells[(bs, d, g, ctx)]
                    r = {
                        "ctx": ctx, "bs": bs, "gamma": g, "depth": d,
                        "tok_s": round(c["tok_s"], 1) if c["tok_s"] else None,
                        "accept_len": round(c["accept_len"], 3) if c["accept_len"] else None,
                        "step_ms": round(c["step_ms"], 3) if c["step_ms"] else None,
                        "itl_ms": round(c["itl_ms"], 3) if c["itl_ms"] else None,
                        "tpot_ms": round(c["tpot_ms"], 3) if c["tpot_ms"] else None,
                        "ttft_ms": round(c["ttft_ms"], 1) if c["ttft_ms"] else None,
                        "preempt": pre.get((bs, d, g, ctx)),
                        "kv_cap": kv.get((d, g)),
                        "n": c["n"],
                    }
                    if d != base_d and base["accept_len"] and base["step_ms"] \
                            and c["accept_len"] and c["step_ms"]:
                        r["d_accept_pct"] = round((c["accept_len"] / base["accept_len"] - 1) * 100, 1)
                        r["d_step_pct"] = round((c["step_ms"] / base["step_ms"] - 1) * 100, 1)
                        r["d_tok_pct"] = round((c["tok_s"] / base["tok_s"] - 1) * 100, 1)
                        la = math.log(c["accept_len"] / base["accept_len"])
                        lsp = math.log(c["step_ms"] / base["step_ms"])
                        tot = la - lsp
                        r["accept_share_pct"] = round(la / tot * 100, 1) if abs(tot) > 1e-9 else None
                        r["net_is_accept_driven"] = (la > 0 and abs(la) > abs(lsp)) or (la > 0 > lsp)
                    rows.append(r)
                    if d != base_d:
                        attrib.append(r)

    # 判定：优势是否由"接受更长"驱动；显存是否被挤占
    verdicts = []
    if attrib:
        acc_driven = [r for r in attrib if r.get("d_tok_pct", 0) > 0]
        bad = [r for r in attrib if r.get("preempt")]
        if acc_driven:
            ok = [r for r in acc_driven if r.get("net_is_accept_driven")]
            verdicts.append(
                f"⑦ 归因：{len(ok)}/{len(acc_driven)} 个'更深更快'的格里，优势由**接受长度**增长主导"
                f"（`Δln(tok/s)=Δln(接受长度)−Δln(每步代价)`）⇒ "
                + ("**计算代价侧的真实机制**，不是显存挤占。" if len(ok) == len(acc_driven)
                   else "存在混合/反常格，见下表。"))
        if bad:
            verdicts.append(f"⚠️ ⑦ 违例：{len(bad)} 个格出现 **preemption>0** ⇒ 该格结论受显存挤占污染，不可作机制证据。")
        else:
            verdicts.append("⑦ 前置条件：全部格 **preemption = 0** ⇒ 无 KV 驱逐 ⇒ 结论可归因于计算，而非显存/排队。")
    return {"cells": cells, "rows": rows, "verdicts": verdicts, "mode": mode, "kv": kv,
            "axes": {"bs": bs_all, "ctx": ctx_all, "gamma": gam_all, "depth": dep_all}}


def summarize(res):
    if "error" in res:
        return res["error"]
    L = ["# 机制层分解：接受长度 vs 每步代价", ""]
    L += res["verdicts"]
    L += ["", "| ctx | bs | γ | d | tok/s | 接受长度 | 每步 (ms) | ITL (ms) | Δ接受 | Δ每步 | Δtok/s | 接受占比 | preempt |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append("| {ctx} | {bs} | {gamma} | d{depth} | {tok_s} | {accept_len} | {step_ms} | {itl_ms} | "
                 "{da} | {ds} | {dt} | {sh} | {pe} |".format(
                     da=f"{r['d_accept_pct']:+}%" if r.get("d_accept_pct") is not None else "—",
                     ds=f"{r['d_step_pct']:+}%" if r.get("d_step_pct") is not None else "—",
                     dt=f"{r['d_tok_pct']:+}%" if r.get("d_tok_pct") is not None else "—",
                     sh=f"{r['accept_share_pct']}%" if r.get("accept_share_pct") is not None else "—",
                     pe=("0" if r.get("preempt") == 0 else (r.get("preempt") or "?")), **r))
    L += ["", "## 逐位置接受率（depth 的作用集中在后半段）", ""]
    for ctx in res["axes"]["ctx"]:
        for bs in res["axes"]["bs"]:
            for g in res["axes"]["gamma"]:
                parts = []
                for d in res["axes"]["depth"]:
                    c = res["cells"].get((bs, d, g, ctx))
                    if c and c["perpos"]:
                        parts.append(f"d{d}[" + " ".join(f"{x:.2f}" for x in c["perpos"]) + "]")
                if parts:
                    L.append(f"- ctx={ctx} bs={bs} γ={g}: " + " ｜ ".join(parts))
    L += ["", "> `Δ接受/Δ每步/Δtok/s` 均相对**该格最浅 depth**；`接受占比 = Δln(接受)/(Δln(接受)−Δln(每步))`，",
          "> 越接近 100% 说明优势越纯粹来自接受长度；若某格靠'每步更便宜'取胜则是另一个故事。",
          "> `preempt` 取自 `.metrics` 的 `num_preemptions_total`（**计数器**；gauge 类字段跑完后已归零，不可用）。"]
    return "\n".join(L) + "\n"


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # d1: 接受 1.5、每步 10ms → 150 tok/s ; d5: 接受 3.0、每步 12ms → 250 tok/s
        # ⇒ 接受 +100%、每步 +20%、tok/s +66.7% ⇒ 优势由接受主导
        for bs in (1, 32):
            sub = os.path.join(td, f"bs{bs}")
            os.makedirs(sub)
            for d, (acc, step) in {1: (1.5, 10.0), 5: (3.0, 12.0)}.items():
                for r in (1, 2, 3, 4, 5):
                    itl = step                      # 实测：itl = 每步延迟
                    tok = bs * 1000.0 * acc / step
                    json.dump({
                        "output_throughput": tok,
                        "spec_decode_acceptance_length": acc,
                        "spec_decode_acceptance_rate": (acc - 1) / 7 * 100,
                        "mean_itl_ms": itl, "mean_tpot_ms": 1000.0 * itl / acc / 1000.0,
                        "mean_ttft_ms": 100.0,
                        "spec_decode_per_position_acceptance_rates": [0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1],
                    }, open(os.path.join(sub, f"d{d}_g7_ctx4096_r{r}.bench.json"), "w"))
                    open(os.path.join(sub, f"d{d}_g7_ctx4096_r{r}.metrics"), "w").write(
                        'vllm:num_preemptions_total{engine="0"} 0.0\n'
                        'vllm:kv_cache_usage_perc{engine="0"} 0.0\n')
        open(os.path.join(td, "d5_g7.serve.log"), "w").write(
            "INFO GPU KV cache size: 203,980 tokens, Maximum concurrency for 40,960 tokens per request: 4.98x\n")
        res = analyze(td, "steady", os.path.join(td, "*.serve.log"))
        assert any("真实机制" in v for v in res["verdicts"]), res["verdicts"]
        assert any("preemption = 0" in v for v in res["verdicts"]), res["verdicts"]
        r5 = [r for r in res["rows"] if r["depth"] == 5][0]
        assert abs(r5["d_accept_pct"] - 100.0) < 0.01, r5
        assert abs(r5["d_step_pct"] - 20.0) < 0.01, r5
        assert r5["accept_share_pct"] > 100, r5          # 接受涨 100%，每步涨 20%
        assert res["kv"][(5, 7)] == 203980, res["kv"]
        txt = summarize(res)
        assert "逐位置接受率" in txt and "接受占比" in txt
        print("selftest ✔ 加载/接受长度/每步代价/逐位置曲线/preemption/KV 容量/归因判定/summary")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--stat", choices=["mean", "steady"], default="steady")
    ap.add_argument("--serve-log-glob")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.dir:
        ap.error("需要 --dir，或 --selftest")
    g = a.serve_log_glob or os.path.join(a.dir, "*.serve.log")
    res = analyze(a.dir, a.stat, g)
    txt = summarize(res)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "mechanism.md"), "w", encoding="utf-8").write(txt)
        keys = ["ctx", "bs", "gamma", "depth", "tok_s", "accept_len", "step_ms", "itl_ms",
                "tpot_ms", "d_accept_pct", "d_step_pct", "d_tok_pct", "accept_share_pct",
                "preempt", "kv_cap", "n"]
        with open(os.path.join(a.out, "mechanism.csv"), "w", encoding="utf-8") as fh:
            fh.write(",".join(keys) + "\n")
            for r in res["rows"]:
                fh.write(",".join(str(r.get(k)) for k in keys) + "\n")
        print(f"\n已写出 {a.out}/mechanism.md 与 mechanism.csv")
