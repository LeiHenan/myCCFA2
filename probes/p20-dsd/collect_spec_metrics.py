#!/usr/bin/env python3
"""p20 采集器：从 vLLM 的 Prometheus `/metrics` 取投机解码计数器并**做差**。

为什么必须这样做（口径教训，2026-09-14）：
  `--per-request-spec-decode-metrics {none,summary,detailed}` 的出口**不是响应体**，而是 `/metrics`；
  且只有 `vllm:spec_decode*_total` 这类**累计计数器**有效，必须**跑前跑后做差**
  （依据：`vllm/benchmarks/serve.py:189-241` 的 `fetch_spec_decode_metrics`，它自己也这么做）。
  我的 p20 冒烟曾在响应体里找这些字段 ⇒ 找不到，属**口径错误**，已记入 decision log。

采集字段（前缀 `vllm:spec_decode`，仅 `_total`）：
  num_drafts / num_draft_tokens / num_accepted_tokens / num_accepted_tokens_per_pos{position="k"}

用法：
  python collect_spec_metrics.py --base http://127.0.0.1:32080 fetch          # 单次抓取（调试）
  python collect_spec_metrics.py --base ... --before b.json --after a.json diff
  python collect_spec_metrics.py --selftest
"""
import argparse
import json
import re
import sys
import urllib.request

PREFIX = "vllm:spec_decode"


def fetch(base, timeout=60):
    """→ {'counters': {name: value}, 'per_pos': {pos: value}, 'found': bool, 'raw_n': int}"""
    with urllib.request.urlopen(base + "/metrics", timeout=timeout) as r:
        text = r.read().decode()
    counters, per_pos, found = {}, {}, False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith(PREFIX):
            continue
        parts = line.split(None, 1)
        name = parts[0].split("{")[0]
        if not name.endswith("_total"):
            continue
        found = True
        try:
            val = float(parts[-1])
        except (ValueError, IndexError):
            continue
        if "num_accepted_tokens_per_pos" in name:
            m = re.search(r'position="(\d+)"', line)
            if m:
                per_pos[int(m.group(1))] = per_pos.get(int(m.group(1)), 0.0) + val
        else:
            counters[name] = counters.get(name, 0.0) + val
    return {"counters": counters, "per_pos": per_pos, "found": found, "raw_n": len(text.splitlines())}


def diff(before, after):
    """→ 派生量：drafts / draft_tokens / accepted / 接受长度 / 逐位置接受率"""
    c0, c1 = before["counters"], after["counters"]
    d = {k: c1.get(k, 0.0) - c0.get(k, 0.0) for k in set(c0) | set(c1)}
    num_drafts = d.get(f"{PREFIX}_num_drafts_total", 0.0)
    draft_tok = d.get(f"{PREFIX}_num_draft_tokens_total", 0.0)
    accepted = d.get(f"{PREFIX}_num_accepted_tokens_total", 0.0)
    per_pos = {p: after["per_pos"].get(p, 0.0) - before["per_pos"].get(p, 0.0)
               for p in set(before["per_pos"]) | set(after["per_pos"])}
    per_pos_rate = {p: (v / num_drafts if num_drafts else None) for p, v in sorted(per_pos.items())}
    return {"delta": d,
            "num_drafts": num_drafts, "num_draft_tokens": draft_tok,
            "num_accepted_tokens": accepted,
            "accept_len": (accepted / num_drafts if num_drafts else None),
            "accept_rate": (accepted / draft_tok if draft_tok else None),
            "per_pos": per_pos, "per_pos_rate": per_pos_rate}


def selftest():
    # 用假 /metrics 文本验证解析（含标签、注释、非 _total 行、多行累加）
    global urllib
    fake = """# HELP vllm:spec_decode
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{model_name="q3"} 100.0
vllm:spec_decode_num_draft_tokens_total{model_name="q3"} 700.0
vllm:spec_decode_num_accepted_tokens_total{model_name="q3"} 350.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="q3",position="0"} 90.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="q3",position="1"} 50.0
vllm:spec_decode_num_drafts 5.0
vllm:other_metric_total 999.0
"""
    import tempfile, os
    class _R:
        def __init__(self, t): self.t = t.encode()
        def read(self): return self.t
        def __enter__(self): return self
        def __exit__(self, *a): return False
    real = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: _R(fake)
    try:
        b = fetch("http://x")
        # 假文本是 8 行内容 + 末尾换行 ⇒ splitlines() 得 9（断言按真实行为写）
        assert b["found"] and b["raw_n"] == 9, b
        assert b["counters"]["vllm:spec_decode_num_drafts_total"] == 100.0, b["counters"]
        assert b["per_pos"] == {0: 90.0, 1: 50.0}, b["per_pos"]
        assert "vllm:other_metric_total" not in b["counters"]
        # 做差：after = before + 增量
        fake2 = fake.replace("100.0", "140.0").replace("700.0", "980.0") \
                   .replace("350.0", "560.0").replace("90.0", "120.0").replace("50.0", "80.0")
        urllib.request.urlopen = lambda *a, **k: _R(fake2)
        a = fetch("http://x")
        d = diff(b, a)
        assert d["num_drafts"] == 40.0 and d["num_draft_tokens"] == 280.0, d
        assert d["num_accepted_tokens"] == 210.0, d
        assert abs(d["accept_len"] - 5.25) < 1e-9, d["accept_len"]
        assert abs(d["accept_rate"] - 0.75) < 1e-9, d["accept_rate"]
        assert d["per_pos_rate"][0] == 30.0 / 40.0, d["per_pos_rate"]
    finally:
        urllib.request.urlopen = real
    print("selftest ✔ /metrics 解析（含标签/注释/非_total 过滤）+ 做差 + 接受长度/率 + 逐位置")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["fetch", "diff"])
    ap.add_argument("--base", default="http://127.0.0.1:32080")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("需要子命令 fetch/diff，或 --selftest")
    if a.cmd == "fetch":
        d = fetch(a.base)
        print(json.dumps(d, ensure_ascii=False)[:800])
        if a.out:
            json.dump(d, open(a.out, "w"))
        return 0 if d["found"] else 2
    b = json.load(open(a.before)); c = json.load(open(a.after))
    d = diff(b, c)
    print(json.dumps({k: v for k, v in d.items() if k != "delta"}, ensure_ascii=False, indent=2))
    if a.out:
        json.dump(d, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
