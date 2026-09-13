#!/usr/bin/env python3
"""p09 一致性判定器：把预登记的四条断言 C1–C4 真的算出来。

判据见 `notes/prereg/engine-metric-conformance.md`（**采数前**写下）：
  C1 计数类指标对必须一致（<1%）；
  C2 延迟指标属「每 token」还是「每步」—— 由与 `accept_len` 的比值判定；
  C3 两引擎的「接受长度」必须同定义（相对差 ≥8% ⇒ 定义不同）；
  C4 吞吐计入口径（仅输出 vs 输入+输出）必须声明。

数据来源（都由 `run_conformance.sh` 产出）：
  `bs<N>/<engine>_<spec>_c<N>_r<R>.bench.json`   —— **同一个客户端**对两家引擎的读数
  `<engine>_<spec>.windows`                      —— 每个 bench 的 START/END 时间戳
  `<engine>_<spec>.scrapes`                      —— **运行中**轮询到的 /metrics（SGLang 的
                                                    accept 是 gauge，跑完归零 ⇒ 必须运行中取，decision #66）

用法：
  python analyze_conformance.py --dir <OUT> [--discover] [--out 目录]
  python analyze_conformance.py --selftest
"""

import argparse
import glob
import json
import os
import re
import statistics as st
import sys

WIN = re.compile(r"^(\d+\.\d+)\s+(START|END)\s+(\S+)$")


def load_bench(root):
    """→ {(engine, spec, conc, rep): bench.json}"""
    out = {}
    for f in glob.glob(os.path.join(root, "bs*", "*.bench.json")):
        m = re.match(r"^(vllm|sglang)_(off|ngram)_c(\d+)_r(\d+)\.bench\.json$", os.path.basename(f))
        if m:
            out[(m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))] = json.load(open(f))
    return out


def load_windows(root):
    """→ {(engine, spec): [(rep_tag, start, end)]}"""
    out = {}
    for f in glob.glob(os.path.join(root, "*.windows")):
        m = re.match(r"^(vllm|sglang)_(off|ngram)\.windows$", os.path.basename(f))
        if not m:
            continue
        eng, spec = m.group(1), m.group(2)
        starts, spans = {}, []
        for ln in open(f, encoding="utf-8"):
            mm = WIN.match(ln.strip())
            if not mm:
                continue
            ts, kind, tag = float(mm.group(1)), mm.group(2), mm.group(3)
            if kind == "START":
                starts[tag] = ts
            elif tag in starts:
                spans.append((tag, starts.pop(tag), ts))
        out[(eng, spec)] = sorted(spans, key=lambda x: x[1])
    return out


def load_scrapes(root):
    """→ {(engine, spec): [(ts, {metric: value})]}"""
    out = {}
    for f in glob.glob(os.path.join(root, "*.scrapes")):
        m = re.match(r"^(vllm|sglang)_(off|ngram)\.scrapes$", os.path.basename(f))
        if not m:
            continue
        samples, ts, cur = [], None, {}
        for ln in open(f, encoding="utf-8", errors="ignore"):
            if ln.startswith("### "):
                if ts is not None and cur:
                    samples.append((ts, cur))
                try:
                    ts = float(ln[4:].strip())
                except ValueError:
                    ts = None
                cur = {}
                continue
            mm = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([\d.eE+-]+)$", ln.strip())
            if mm and ts is not None:
                try:
                    cur[mm.group(1)] = float(mm.group(3))
                except ValueError:
                    pass
        if ts is not None and cur:
            samples.append((ts, cur))
        out[(m.group(1), m.group(2))] = samples
    return out


def pick(samples, t, before=True):
    """取时间上最接近 t 的样本（before=True: 不晚于 t）。"""
    cand = [s for s in samples if (s[0] <= t if before else s[0] >= t)]
    if not cand:
        return None
    return min(cand, key=lambda s: abs(s[0] - t))[1]


def series(engine, spec, tag, benches, wins, scraps):
    """取出一个 bench 窗口内、以及相对**上一次**窗口的引擎侧指标。"""
    eng, sp = engine, spec
    span = next((w for w in wins.get((eng, sp), []) if w[0] == tag), None)
    samples = scraps.get((eng, sp), [])
    if not span or not samples:
        return {}
    _, t0, t1 = span
    cur = pick(samples, t1, before=True) or {}
    prev = pick(samples, t0, before=True) or {}
    return {"cur": cur, "prev": prev}


def delta(cur, prev, key):
    if key in cur and key in prev:
        return cur[key] - prev[key]
    return cur.get(key)


def analyze(root, discover=False):
    benches = load_bench(root)
    wins = load_windows(root)
    scraps = load_scrapes(root)
    if not benches:
        return {"error": f"{root}/bs*/ 下没有 vllm_*/sglang_* 的 bench.json"}

    names = sorted({k for s in scraps.values() for _, m in s for k in m})
    found = {eng: sorted({k for (e, _), ss in scraps.items() if e == eng for _, m in ss for k in m})
             for eng in ("vllm", "sglang")}
    if discover:
        return {"discover": True, "found": found, "names": names,
                "benches": sorted(f"{k[0]}/{k[1]}/c{k[2]}/r{k[3]}" for k in benches)}

    rows, verdicts = [], []
    for key in sorted(benches):
        eng, spec, conc, rep = key
        b = benches[key]
        tag = f"{eng}_{spec}_c{conc}_r{rep}"
        s = series(eng, spec, tag, benches, wins, scraps)
        cur, prev = s.get("cur", {}), s.get("prev", {})
        row = {"engine": eng, "spec": spec, "conc": conc, "rep": rep,
               "in_tok": b.get("total_input_tokens"), "out_tok": b.get("total_output_tokens"),
               "completed": b.get("completed"),
               "tok_s": b.get("output_throughput"),
               "ttft": b.get("mean_ttft_ms"), "itl": b.get("mean_itl_ms"), "tpot": b.get("mean_tpot_ms"),
               "accept_client": b.get("spec_decode_acceptance_length"),
               "acc_rate_client": b.get("spec_decode_acceptance_rate"),
               # 引擎侧：vLLM 计数器（增量）/ SGLang gauge
               "drafts": delta(cur, prev, "vllm:spec_decode_num_drafts_total"),
               "accepted": delta(cur, prev, "vllm:spec_decode_num_accepted_tokens_total"),
               "sg_accept": cur.get("sglang:spec_accept_length"),
               "sg_accept_rate": cur.get("sglang:spec_accept_rate"),
               "eng_prompt_tok": delta(cur, prev, "vllm:prompt_tokens_total") or cur.get("sglang:prompt_tokens_total"),
               "eng_gen_tok": delta(cur, prev, "vllm:generation_tokens_total") or cur.get("sglang:generation_tokens_total"),
               "eng_reqs": delta(cur, prev, 'vllm:request_success_total{engine="0",finished_reason="length",model_name="' + str(cur.get("__model__", "")) + '"}'),
               }
        if row["drafts"]:
            row["accept_engine_vllm"] = 1 + (row["accepted"] or 0) / row["drafts"]
        rows.append(row)

    # ---- C3：接受长度是否同定义
    c3 = []
    for conc in sorted({r["conc"] for r in rows}):
        v = [r for r in rows if r["engine"] == "vllm" and r["spec"] == "ngram" and r["conc"] == conc
             and r.get("accept_engine_vllm")]
        g = [r for r in rows if r["engine"] == "sglang" and r["spec"] == "ngram" and r["conc"] == conc
             and r.get("sg_accept")]
        if v and g:
            av = st.mean(r["accept_engine_vllm"] for r in v)
            ag = st.mean(r["sg_accept"] for r in g)
            d = (av / ag - 1) * 100 if ag else float("nan")
            c3.append((conc, av, ag, d))
            verdicts.append(f"**C3 @bs={conc}**：vLLM 接受长度 **{av:.3f}**（由 `1+accepted/drafts` 增量推出）"
                            f" vs SGLang `spec_accept_length` **{ag:.3f}** ⇒ 相对差 **{d:+.1f}%** ⇒ "
                            + ("**定义不同（判据成立，≥8%）**" if abs(d) >= 8 else "视为可比（<8%）"))
        elif v and not g:
            verdicts.append(f"**C3 @bs={conc}**：SGLang 侧**取不到** `spec_accept_length`（gauge 可能未采到）⇒ 不可判定")
    if not c3 and not any("C3" in v for v in verdicts):
        verdicts.append("**C3 不可判定**：任一引擎缺接受长度来源")

    # ---- C2：ITL 与 TPOT 的比值（每步 vs 每 token）
    for eng in ("vllm", "sglang"):
        for spec in ("off", "ngram"):
            rs = [r for r in rows if r["engine"] == eng and r["spec"] == spec and r["itl"] and r["tpot"]]
            if not rs:
                continue
            ratios = [r["itl"] / r["tpot"] for r in rs]
            accs = [r.get("accept_engine_vllm") or r.get("sg_accept") for r in rs]
            accs = [a for a in accs if a]
            verdicts.append(f"**C2** {eng}/{spec}：`itl/tpot` = **{st.mean(ratios):.2f}×**"
                            + (f"，同格接受长度 = {st.mean(accs):.2f} ⇒ "
                               + ("**itl 是「每步」**（与 accept_len 同量级 ⇒ 与 TPOT 不可直接并列）"
                                  if abs(st.mean(ratios) - st.mean(accs)) < 0.35 * st.mean(accs)
                                  else "**itl 与 accept_len 不成比例 ⇒ 更像「每 token」**")
                               if accs else "（无接受长度对照）"))

    # ---- C1：计数类一致性（用**同一个客户端**的读数作参照）
    v_off = [r for r in rows if r["engine"] == "vllm" and r["spec"] == "off" and r["conc"] == 32]
    g_off = [r for r in rows if r["engine"] == "sglang" and r["spec"] == "off" and r["conc"] == 32]
    if v_off and g_off:
        for field in ("in_tok", "out_tok", "completed"):
            a, b = st.mean(r[field] for r in v_off if r[field] is not None), st.mean(r[field] for r in g_off if r[field] is not None)
            if a and b:
                d = (a / b - 1) * 100
                verdicts.append(f"**C1** 客户端读数 `{field}` @bs32/off：vLLM {a:.0f} vs SGLang {b:.0f} ⇒ **{d:+.2f}%** ⇒ "
                                + ("✅ 一致（<1%）" if abs(d) < 1 else "⚠️ 不一致（>1%）⇒ 两家对同一负载的 token 计数口径不同"))

    # ---- C4：吞吐口径
    v32 = [r for r in rows if r["engine"] == "vllm" and r["spec"] == "off" and r["conc"] == 32]
    g32 = [r for r in rows if r["engine"] == "sglang" and r["spec"] == "off" and r["conc"] == 32]
    if v32 and g32:
        vo = st.mean(r["tok_s"] for r in v32 if r["tok_s"])
        go = st.mean(r["tok_s"] for r in g32 if r["tok_s"])
        verdicts.append(f"**C4** 客户端 `output_throughput` @bs32/off：vLLM **{vo:.1f}** vs SGLang **{go:.1f}** tok/s"
                        f"（比值 {vo / go:.2f}×）—— 注意这是**同一客户端**算出的，"
                        "差异只可能来自**引擎回报的 usage 口径**或真实性能 ⇒ 必须配合 `/metrics` 的 token 计数才能定性")

    return {"rows": rows, "verdicts": verdicts, "found": found}


def summarize(res):
    if "error" in res:
        return res["error"]
    if res.get("discover"):
        L = ["# p09 指标发现（两引擎 /metrics 里实际存在的指标名）", ""]
        for eng, names in res["found"].items():
            L += [f"## {eng}（{len(names)} 个）", "", "```", "\n".join(names), "```", ""]
        return "\n".join(L) + "\n"
    L = ["# p09 跨引擎一致性判定（C1–C4）", ""] + res["verdicts"] + ["",
         "| 引擎 | 投机 | bs | rep | 输入tok | 输出tok | tok/s | TTFT | ITL | TPOT | 客户端accept | 引擎侧accept |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        eng_acc = r.get("accept_engine_vllm") or r.get("sg_accept")
        L.append(f"| {r['engine']} | {r['spec']} | {r['conc']} | {r['rep']} | {r['in_tok']} | {r['out_tok']} | "
                 f"{None if r['tok_s'] is None else round(r['tok_s'], 1)} | "
                 f"{None if r['ttft'] is None else round(r['ttft'], 1)} | "
                 f"{None if r['itl'] is None else round(r['itl'], 2)} | "
                 f"{None if r['tpot'] is None else round(r['tpot'], 2)} | {r['accept_client']} | "
                 f"{None if eng_acc is None else round(eng_acc, 3)} |")
    L += ["", "> 判据见 `notes/prereg/engine-metric-conformance.md`（采数前登记）。",
          "> 引擎侧数值取自**运行中**的 `/metrics` 轮询；vLLM 用**增量**（累计计数器），SGLang 用窗口内 gauge。"]
    return "\n".join(L) + "\n"


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "bs32"))
        # 构造：vLLM accept=2.8（增量），SGLang spec_accept_length=2.0 ⇒ C3 差 40% ⇒ 成立
        for eng, acc in (("vllm", 2.8), ("sglang", 2.0)):
            for r in (1, 2, 3):
                json.dump({"total_input_tokens": 131328, "total_output_tokens": 4096, "completed": 32,
                           "output_throughput": 2000.0, "mean_ttft_ms": 100.0,
                           "mean_itl_ms": 34.0, "mean_tpot_ms": 12.0},
                          open(os.path.join(td, "bs32", f"{eng}_ngram_c32_r{r}.bench.json"), "w"))
        # windows：两个窗口，模拟增量
        with open(os.path.join(td, "vllm_ngram.windows"), "w") as fh:
            fh.write("100.0 START vllm_ngram_c32_r1\n110.0 END vllm_ngram_c32_r1\n")
        with open(os.path.join(td, "sglang_ngram.windows"), "w") as fh:
            fh.write("100.0 START sglang_ngram_c32_r1\n110.0 END sglang_ngram_c32_r1\n")
        # scrapes
        with open(os.path.join(td, "vllm_ngram.scrapes"), "w") as fh:
            fh.write("### 99.0\nvllm:spec_decode_num_drafts_total{engine=\"0\"} 0.0\n"
                     "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\"} 0.0\n"
                     "### 109.0\nvllm:spec_decode_num_drafts_total{engine=\"0\"} 1000.0\n"
                     "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\"} 1800.0\n")
        with open(os.path.join(td, "sglang_ngram.scrapes"), "w") as fh:
            fh.write("### 99.0\nsglang:spec_accept_length 0.0\n### 109.0\nsglang:spec_accept_length 2.0\n")
        res = analyze(td)
        assert "error" not in res, res
        assert any("C3" in v and "定义不同" in v for v in res["verdicts"]), res["verdicts"]
        assert any("C2" in v for v in res["verdicts"]), res["verdicts"]
        d = analyze(td, discover=True)
        assert "vllm" in d["found"] and d["found"]["vllm"], d
        assert "C1–C4" in summarize(res)
        print("selftest ✔ 窗口/增量/gauge 解析/C2 比值/C3 定义差判定/发现模式/输出")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.dir:
        ap.error("需要 --dir，或 --selftest")
    res = analyze(a.dir, a.discover)
    txt = summarize(res)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        name = "discover.md" if a.discover else "conformance.md"
        open(os.path.join(a.out, name), "w", encoding="utf-8").write(txt)
        print(f"已写出 {a.out}/{name}")
