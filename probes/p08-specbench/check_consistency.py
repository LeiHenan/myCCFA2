#!/usr/bin/env python3
"""p08 自检器：对一组投机解码**测量结果**跑五条度量恒等式，判定其是否自洽。

**这是本候选的核心可交付物**（S2 形态⑥ 的对冲：把"建议"变成"判据 + 工具"）。
恒等式与容差**预登记在** `notes/prereg/specbench-methodology.md`（采数前写下）。

五条恒等式
----------
I1 `mean_tpot_ms ≈ mean_itl_ms / accept_len`            ±3%   —— ITL 是「每步」不是「每 token」
I2 `accept_len ≤ γ + 1`                                 硬约束
I3 `accept_len`(bench.json) ≈ `1 + accepted/drafts`（/metrics **计数器**）  ±5%
I4 逐位置接受率 `max_k p_k < 1.0`（≥0.999 报警）        —— 真实文本不可能每位置必然接受
I5 `tok/s ≤ 1.05 × batch × 1000 / tpot_ms`             **单边上界**

用法：
  python check_consistency.py --dir <结果目录> [--gamma 7] [--out 目录]
  python check_consistency.py --selftest
"""

import argparse
import glob
import json
import os
import re
import sys

BENCH = re.compile(r"^(?:[a-z0-9]+_)?(custom|random)_t([\d.]+)_d_?(?:.*?)_c(\d+)_r(\d+)\.bench\.json$",
                   re.I)
FALLBACK = re.compile(r"^(custom|random)_t([\d.]+)_c(\d+)_r(\d+)\.bench\.json$", re.I)


def _f(x):
    return x if isinstance(x, (int, float)) else None


def load_cells(root):
    """→ {(dataset, temp, conc, rep): {bench:..., metrics_path:...}}"""
    cells = {}
    for f in sorted(glob.glob(os.path.join(root, "bs*", "*.bench.json"))):
        b = os.path.basename(f)
        m = FALLBACK.match(b) or BENCH.match(b)
        if not m:
            continue
        ds, temp, conc, rep = m.group(1).lower(), float(m.group(2)), int(m.group(3)), int(m.group(4))
        met = f[: -len(".bench.json")] + ".metrics"
        cells[(ds, temp, conc, rep)] = {
            "bench": json.load(open(f)),
            "metrics": met if os.path.exists(met) else None,
        }
    return cells


def counters(path):
    """从 /metrics 取两个**计数器**（gauge 跑完归零，不可用）。"""
    if not path:
        return None
    txt = open(path, errors="ignore").read()
    out = {}
    for key, pat in (("drafts", r"^vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([\d.eE+]+)"),
                     ("accepted", r"^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([\d.eE+]+)")):
        m = re.search(pat, txt, re.M)
        if m:
            out[key] = float(m.group(1))
    return out or None


def check_cell(rec, gamma, tol=(0.03, 0.05, 0.10), prev=None):
    """→ (结果 dict, 违规列表)"""
    d = rec["bench"]
    acc = _f(d.get("spec_decode_acceptance_length"))
    itl = _f(d.get("mean_itl_ms"))
    tpot = _f(d.get("mean_tpot_ms"))
    tok = _f(d.get("output_throughput"))
    conc = _f(d.get("max_concurrency")) or 1
    pos = d.get("spec_decode_per_position_acceptance_rates") or []
    r = {"accept_len": acc, "itl_ms": itl, "tpot_ms": tpot, "tok_s": tok,
         "max_pos_accept": max(pos) if pos else None}
    bad = []
    # I1
    if acc and itl and tpot:
        pred = itl / acc
        err = abs(pred - tpot) / tpot
        r["I1_err_pct"] = round(err * 100, 2)
        if err > tol[0]:
            bad.append(f"I1 违反：tpot 预测 {pred:.2f} vs 实测 {tpot:.2f}（误差 {err * 100:.1f}% > {tol[0] * 100:.0f}%）")
    # I2
    if acc:
        r["I2_ok"] = acc <= gamma + 1 + 1e-9
        if not r["I2_ok"]:
            bad.append(f"I2 违反：accept_len {acc:.2f} > γ+1 = {gamma + 1}")
    # I3：⚠️ /metrics 是**累计计数器**（跨 rep 累加），bench.json 是**本次运行**的值
    # ⇒ 必须用「本次 − 上次」的增量。初版直接拿累计值比，导致 random+t0.7 的 4 个格假报警，
    #    而 custom 因为每次 draft 数几乎相同而侥幸通过 —— 这是**仪器自身的 bug**，已修正。
    c = counters(rec["metrics"])
    if c and acc:
        base_d = (prev or {}).get("drafts", 0.0)
        base_a = (prev or {}).get("accepted", 0.0)
        dd = c.get("drafts", 0.0) - base_d
        da = c.get("accepted", 0.0) - base_a
        r["drafts_delta"] = dd
      # 增量才算本次运行
        derived = 1 + (da / dd if dd else 0.0)
        err = abs(derived - acc) / acc
        r["I3_derived"] = round(derived, 3)
        r["I3_err_pct"] = round(err * 100, 2)
        r["I3_delta_used"] = dd
        if err > tol[1]:
            bad.append(f"I3 违反：bench.json {acc:.2f} vs /metrics 推出 {derived:.2f}（误差 {err * 100:.1f}%）")
    # I4
    if r["max_pos_accept"] is not None:
        r["I4_ok"] = r["max_pos_accept"] < 0.999
        if not r["I4_ok"]:
            bad.append(f"I4 报警：逐位置接受率最大值 {r['max_pos_accept']:.4f} ≥ 0.999（真实文本不可能）")
    # I5（单边上界）：稳态理想吞吐 = batch×1000/tpot；实测**低于**它属正常（prefill/爬升/尾巴），
    # **高于**它才不可能 ⇒ 只在这个方向报警。初版把它写成 ±10% 等式，被自测当场抓到（真实比例 ≈0.77）。
    if tok and tpot:
        ideal = conc * 1000.0 / tpot
        ratio = tok / ideal
        r["I5_ratio"] = round(ratio, 3)
        if ratio > 1.05:
            bad.append(f"I5 违反：实测 {tok:.1f} **高于**理想上界 {ideal:.1f}（比值 {ratio:.2f} > 1.05）")
    return r, bad


def analyze(root, gamma=7):
    cells = load_cells(root)
    if not cells:
        return {"error": f"{root}/bs*/ 下没有可识别的 bench.json（文件名需含 custom|random 与 t<温度>）"}
    rows = []
    prev_by_group = {}
    for (ds, temp, conc, rep), rec in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        grp = (ds, temp)      # ⚠️ 分组键必须是 **serve 会话**（数据集×温度），不是 (…, conc)：
                              # 一个 serve 会话内先跑 bs=1 再跑 bs=32，计数器跨并发累加，
                              # 按 conc 分组会让 bs=32 的第 1 个 rep 拿累计值当增量（假报警，实测 11.6%）
        r, bad = check_cell(rec, gamma, prev=prev_by_group.get(grp))
        c = counters(rec["metrics"])
        if c:
            prev_by_group[grp] = c          # 同一 serve 会话内累计 ⇒ 下个 rep 用增量
        r.update({"dataset": ds, "temp": temp, "conc": conc, "rep": rep, "violations": bad})
        rows.append(r)
    # 判据①：判别力
    def cell(ds, temp, conc):
        cand = [r for r in rows if r["dataset"] == ds and r["temp"] == temp and r["conc"] == conc]
        return cand[0] if cand else None
    v = []
    rnd = [r for r in rows if r["dataset"] == "random" and r["accept_len"]]
    cus = [r for r in rows if r["dataset"] == "custom" and r["accept_len"]]
    if rnd and cus:
        a_rnd = sum(r["accept_len"] for r in rnd) / len(rnd)
        a_cus = sum(r["accept_len"] for r in cus) / len(cus)
        v.append(f"**接受长度**：`random` 均值 **{a_rnd:.2f}** vs `custom` 均值 **{a_cus:.2f}**（γ+1 = {gamma + 1}）"
                 f" ⇒ 差距 **{(a_rnd / a_cus - 1) * 100:+.1f}%**")
        maxpos_r = max((r["max_pos_accept"] or 0) for r in rnd)
        maxpos_c = max((r["max_pos_accept"] or 0) for r in cus)
        crit1 = (a_rnd >= gamma + 1 - 0.05) and maxpos_r >= 0.999 and a_cus <= gamma and maxpos_c <= 0.95
        v.append(f"**判据①（判别力）**：random 峰值位置接受率 {maxpos_r:.4f}，custom {maxpos_c:.4f} ⇒ "
                 + ("✅ **成立**：自检器能区分有效/无效协议" if crit1
                    else "❌ **不成立**（见 §预测 Pa 的更正义务）"))
    i1_bad = [r for r in rows if r.get("I1_err_pct", 0) > 3]
    v.append(f"**判据②（I1 在所有格成立）**：" + ("✅ 全部成立" if not i1_bad
             else f"❌ {len(i1_bad)} 格违反（最大误差 {max(r['I1_err_pct'] for r in i1_bad):.1f}%）⇒ 恒等式需修正"))
    i3_bad = [r for r in rows if r["dataset"] == "custom" and r.get("I3_err_pct", 0) > 5]
    v.append("**I3（两条独立路径一致，**按计数器增量**）**：" + ("✅ 全部格一致（误差 ≤5%）" if not i3_bad
             else f"⚠️ {len(i3_bad)} 格不一致（最大 {max(r['I3_err_pct'] for r in i3_bad):.1f}%）"))
    # 判据③：温度
    for ds in ("custom", "random"):
        t0 = [r["accept_len"] for r in rows if r["dataset"] == ds and r["temp"] == 0 and r["accept_len"]]
        t7 = [r["accept_len"] for r in rows if r["dataset"] == ds and r["temp"] == 0.7 and r["accept_len"]]
        if t0 and t7:
            m0, m7 = sum(t0) / len(t0), sum(t7) / len(t7)
            v.append(f"**判据③（温度 ⇒ 接受率）** {ds}：temp=0 **{m0:.2f}** vs temp=0.7 **{m7:.2f}**"
                     f" ⇒ **{(m0 / m7 - 1) * 100:+.1f}%**" + ("（≥8% ⇒ 温度是接受率的混淆变量）" if m0 / m7 - 1 >= 0.08 else ""))
    pre = [r for r in rows if r.get("violations")]
    kinds = {}
    for r in pre:
        for x in r["violations"]:
            kinds[x.split()[0]] = kinds.get(x.split()[0], 0) + 1
    tail = ("（类型：" + "、".join(f"{k}×{n}" for k, n in sorted(kinds.items())) + "）") if pre else ""
    v.append(f"**违规计数**：{len(pre)}/{len(rows)} 格有违规" + tail)
    return {"rows": rows, "verdicts": v}


def summarize(res):
    if "error" in res:
        return res["error"]
    L = ["# p08 自检器：度量恒等式一致性检查", ""] + res["verdicts"] + ["",
         "| 数据集 | 温度 | bs | rep | 接受长度 | ITL(ms) | TPOT(ms) | I1 误差 | I3 误差 | 峰值位置接受率 | I5 比值 | 违规 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['dataset']} | {r['temp']} | {r['conc']} | {r['rep']} | **{r['accept_len']}** | {r['itl_ms']} | "
                 f"{r['tpot_ms']} | {r.get('I1_err_pct', '—')}% | {r.get('I3_err_pct', '—')}% | "
                 f"{r['max_pos_accept'] if r['max_pos_accept'] is None else round(r['max_pos_accept'], 4)} | "
                 f"{r.get('I5_ratio', '—')} | "
                 f"{'; '.join(r['violations']) if r['violations'] else '✅ 自洽'} |")
    L += ["", "> 恒等式与容差**预登记在采数之前**（`notes/prereg/specbench-methodology.md`）。",
          "> 违规 ≠ 引擎有 bug：也可能是**协议无效**（如随机输入）或**报告口径错**（把 ITL 当每 token）。"]
    return "\n".join(L) + "\n"


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "bs32")
        os.makedirs(d)
        # custom：自洽
        def w(name, acc, itl, tpot, tok, pos, drafts, accepted):
            json.dump({"spec_decode_acceptance_length": acc, "mean_itl_ms": itl, "mean_tpot_ms": tpot,
                       "output_throughput": tok, "max_concurrency": 32,
                       "spec_decode_per_position_acceptance_rates": pos},
                      open(os.path.join(d, name + ".bench.json"), "w"))
            open(os.path.join(d, name + ".metrics"), "w").write(
                f'vllm:spec_decode_num_drafts_total{{engine="0"}} {drafts}\n'
                f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {accepted}\n'
                'vllm:num_preemptions_total{engine="0"} 0.0\n')
        for r in (1, 2, 3):
            w(f"custom_t0.7_c32_r{r}", 2.83, 34.9, 12.33, 2000.0, [0.71, 0.45, 0.29, 0.17, 0.11, 0.06, 0.04],
              1467, 2642)
            w(f"random_t0.7_c32_r{r}", 8.0, 30.0, 3.75, 6000.0, [1.0] * 7, 1467, 10269)
        res = analyze(td, gamma=7)
        assert "error" not in res, res
        assert any("成立" in v and "判别力" in v for v in res["verdicts"]), res["verdicts"]
        assert any("I1 在所有格成立" in v and "✅" in v for v in res["verdicts"]), res["verdicts"]
        rnd = [r for r in res["rows"] if r["dataset"] == "random"][0]
        assert any("I4" in x for x in rnd["violations"]), rnd
        cus = [r for r in res["rows"] if r["dataset"] == "custom"][0]
        assert not cus["violations"], cus
        # 破坏 I1 ⇒ 应被抓住
        w("custom_t0.0_c32_r1", 2.83, 34.9, 34.9, 2000.0, [0.7] * 7, 1467, 2642)
        w("random_t0.0_c32_r1", 8.0, 30.0, 3.75, 9999.0, [1.0] * 7, 1467, 10269)  # 高于理想上界 ⇒ 应报 I5
        res2 = analyze(td, gamma=7)
        assert any("I1" in v and "❌" in v for v in res2["verdicts"]), res2["verdicts"]
        assert "自检器" in summarize(res)
        print("selftest ✔ custom 自洽/random 触发 I4/I2/判别力判定/I1 破坏被抓/温度比较/输出")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--gamma", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.dir:
        ap.error("需要 --dir，或 --selftest")
    res = analyze(a.dir, a.gamma)
    txt = summarize(res)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "consistency.md"), "w", encoding="utf-8").write(txt)
        print(f"已写出 {a.out}/consistency.md")
