#!/usr/bin/env python3
"""T0 判定器 —— 读 run_t0.sh 的产物，输出三结局（A/B/C）判定与 summary.md。

用法：
  python analyze_t0.py --dir results/p06-frontier/<date>/T0 [--drafter-root upstream/drafters]
  python analyze_t0.py --selftest

指标口径（vLLM v0.29 `vllm/v1/spec_decode/metrics.py` 的 PromQL 注释）：
  接受率        = num_accepted_tokens / num_draft_tokens
  平均接受长度  = num_accepted_tokens / num_drafts
注意：`--kv-cache-dtype` 合法值是 auto/float16/bfloat16/fp8*（**没有** fp16）。
"""

import argparse
import json
import re
import sys
from pathlib import Path

METRIC_ALIASES = {
    "drafts": ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts"),
    "draft_tokens": ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_draft_tokens"),
    "accepted": ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"),
}
# T0 是"旋钮灵不灵"的粗判，不是预登记判据；此阈值仅用于把"几乎不动"标出来供人判断
HEURISTIC_SPREAD = 0.05


def parse_metrics(text: str) -> dict:
    out = {}
    for key, names in METRIC_ALIASES.items():
        total = 0.0
        found = False
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for name in names:
                m = re.match(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9eE.+-]+)\s*$", line.strip())
                if m:
                    total += float(m.group(1))
                    found = True
        out[key] = total if found else None
    return out


def variant_layers(drafter_root: Path, d: int):
    cfg = drafter_root / f"d{d}" / "config.json"
    if not cfg.exists():
        return None
    try:
        c = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return None
    tlc = c.get("transformer_layer_config")
    if isinstance(tlc, dict) and isinstance(tlc.get("num_hidden_layers"), int):
        return tlc["num_hidden_layers"]
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        if isinstance(c.get(k), int):
            return c[k]
    return None


def first_text(prompt_json: Path):
    if not prompt_json.exists():
        return None
    try:
        j = json.loads(prompt_json.read_text(encoding="utf-8"))
        return (j.get("choices") or [{}])[0].get("text")
    except Exception:
        return None


def bench_throughput(bench_json: Path):
    if not bench_json.exists():
        return None
    try:
        j = json.loads(bench_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    for k in ("output_throughput", "total_token_throughput", "request_throughput"):
        if isinstance(j.get(k), (int, float)):
            return float(j[k]), k
    return None


def collect(out_dir: Path, drafter_root: Path | None) -> list[dict]:
    rows = []
    for metrics in sorted(out_dir.glob("d*.metrics")):
        d = int(re.match(r"d(\d+)\.metrics", metrics.name).group(1))
        m = parse_metrics(metrics.read_text(encoding="utf-8", errors="ignore"))
        acc, drafts, dtok = m["accepted"], m["drafts"], m["draft_tokens"]
        row = {
            "d": d,
            "accepted": acc,
            "drafts": drafts,
            "draft_tokens": dtok,
            "accept_len": (1.0 + acc / drafts) if (acc is not None and drafts) else None,
            "accept_rate": (acc / dtok) if (acc is not None and dtok) else None,
            "tok_s": None,
            "text": first_text(out_dir / f"d{d}.prompt.json"),
            "cfg_layers": variant_layers(drafter_root, d) if drafter_root else None,
            "served": (out_dir / f"d{d}.bench.json").exists(),
        }
        bt = bench_throughput(out_dir / f"d{d}.bench.json")
        if bt:
            row["tok_s"] = bt[0]
        rows.append(row)
    return rows


def judge(rows: list[dict]) -> tuple[str, list[str]]:
    notes: list[str] = []
    live = [r for r in rows if r["accept_len"] is not None]
    if len(rows) < 2 or len(live) < 2:
        return "无法判定", ["成功产出指标的档位不足 2 个 ⇒ 先看各档 d*.serve.log 的报错"]

    if any(r["cfg_layers"] is not None and r["cfg_layers"] != r["d"] for r in rows):
        notes.append("⚠️ 有变体的 config 层数与目录名不符 ⇒ 变体生成或路径有问题")

    lens = [r["accept_len"] for r in live]
    spread = max(lens) - min(lens)
    mean = sum(lens) / len(lens)
    rel = spread / mean if mean else 0.0
    notes.append(f"平均接受长度 {min(lens):.3f}–{max(lens):.3f}（相对跨度 {rel:.1%}，展示阈值 {HEURISTIC_SPREAD:.0%}）")
    toks = [r["tok_s"] for r in live if r["tok_s"]]
    if toks:
        notes.append(f"tok/s {min(toks):.1f}–{max(toks):.1f}")

    texts = [r["text"] for r in live if r["text"]]
    # ⚠️ 更正（2026-09-12 实测）：投机解码是**无损**的 ⇒ 各档 greedy 输出**本就应当逐字相同**。
    #    早期版本把"输出全同"判为"截断未生效"，是**假警报**：真正的判别量是接受长度/tok/s 是否随 d 变动。
    if len(texts) >= 3 and len(set(texts)) == 1:
        notes.append("lossless 检查通过：各档 greedy 输出逐字相同（投机解码无损，这是**预期**行为）")
    elif len(texts) >= 2:
        notes.append(f"⚠️ 各档输出不一致（{len(set(texts))} 种）⇒ 检查是否有档位加载失败或非贪心采样")

    # 单调性：接受长度是否随深度单调不减
    seq = [(r["d"], r["accept_len"]) for r in sorted(live, key=lambda x: x["d"])]
    mono = all(b[1] >= a[1] - 1e-9 for a, b in zip(seq, seq[1:]))
    notes.append("接受长度随深度" + ("单调不减" if mono else "非单调（可能单峰，需看 T1 的 γ 维）"))

    if rel < HEURISTIC_SPREAD:
        return "C. 灰区（能跑但接受长度几乎不随深度变）", notes
    return "A. 旋钮可操作（进 T1）", notes


def write_summary(out_dir: Path, rows: list[dict], verdict: str, notes: list[str]) -> Path:
    lines = ["# T0 — 深度旋钮验证（结果）", "", f"**判定**：{verdict}", ""]
    lines += [f"- {n}" for n in notes]
    lines += ["", "| depth | config 层数 | 平均接受长度 | 接受率 | tok/s | greedy 输出（前 40 字） |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        def fmt(x, p=3):
            return "—" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))
        txt = (r["text"] or "").replace("|", "/").replace("\n", " ")[:40]
        lines.append(
            f"| {r['d']} | {fmt(r['cfg_layers'])} | {fmt(r['accept_len'])} | "
            f"{fmt(r['accept_rate'])} | {fmt(r['tok_s'], 1)} | {txt} |"
        )
    lines += [
        "",
        "> 口径：**平均接受长度 = 1 + `num_accepted_tokens / num_drafts`**（与 vLLM 日志的 `Mean acceptance length` 一致；1.00 = 一个草稿都没被接受）；接受率 = `num_accepted_tokens / num_draft_tokens`。",
        "> 判据见 `probes/p06-frontier/T0-runbook.md` §4；上表的展示阈值**不是**预登记判据，仅用于标出'几乎不动'。",
    ]
    p = out_dir / "summary.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "d1.metrics").write_text(
            "# TYPE vllm:spec_decode_num_drafts counter\n"
            'vllm:spec_decode_num_drafts{model_name="m"} 100.0\n'
            'vllm:spec_decode_num_draft_tokens{model_name="m"} 700.0\n'
            'vllm:spec_decode_num_accepted_tokens{model_name="m"} 200.0\n', encoding="utf-8")
        (root / "d3.metrics").write_text(
            'vllm:spec_decode_num_drafts{model_name="m"} 100.0\n'
            'vllm:spec_decode_num_draft_tokens{model_name="m"} 700.0\n'
            'vllm:spec_decode_num_accepted_tokens{model_name="m"} 400.0\n', encoding="utf-8")
        for d in (1, 3):
            (root / f"d{d}.prompt.json").write_text(json.dumps({"choices": [{"text": f"out{d}"}]}), encoding="utf-8")
            (root / f"d{d}.bench.json").write_text(json.dumps({"output_throughput": 10.0 * d}), encoding="utf-8")
        rows = collect(root, None)
        # 口径：accept_len = 1 + accepted/drafts（与 vLLM 的 Mean acceptance length 一致）
        assert rows[0]["accept_len"] == 3.0 and abs(rows[0]["accept_rate"] - 2 / 7) < 1e-9, rows[0]
        assert rows[1]["accept_len"] == 5.0, rows[1]
        v, n = judge(rows)
        assert v.startswith("A."), (v, n)
        p = write_summary(root, rows, v, n)
        assert p.exists() and "T0 — 深度旋钮验证" in p.read_text(encoding="utf-8")
        # 全同文本 ⇒ 应判为 lossless 通过（而非 B/C 边界）
        for d in (1, 3):
            (root / f"d{d}.prompt.json").write_text(json.dumps({"choices": [{"text": "same"}]}), encoding="utf-8")
        (root / "d5.metrics").write_text((root / "d1.metrics").read_text(encoding="utf-8"), encoding="utf-8")
        (root / "d5.prompt.json").write_text(json.dumps({"choices": [{"text": "same"}]}), encoding="utf-8")
        rows2 = collect(root, None)
        v2, n2 = judge(rows2)
        assert "lossless 检查通过" in " ".join(n2), n2          # 输出相同 = 无损，属预期
        assert not v2.startswith("B/"), v2                       # 不得再据此判成 B/C 边界
        # 接受长度不随深度变 ⇒ C（灰区）
        (root / "d3.metrics").write_text((root / "d1.metrics").read_text(encoding="utf-8"), encoding="utf-8")
        v3, _ = judge(collect(root, None))
        assert v3.startswith("C."), v3
        print("selftest ✔ 指标解析 / 接受长度口径 / lossless 预期 / A 与 C 判定 / summary 落盘")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--drafter-root")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.dir:
        ap.error("需要 --dir，或 --selftest")
    d = Path(a.dir)
    rows = collect(d, Path(a.drafter_root) if a.drafter_root else None)
    if not rows:
        sys.exit(f"{d} 下没有 d*.metrics —— 服务可能都没起来")
    verdict, notes = judge(rows)
    p = write_summary(d, rows, verdict, notes)
    print(f"判定：{verdict}")
    for n in notes:
        print("  -", n)
    print(f"已写 {p}")
