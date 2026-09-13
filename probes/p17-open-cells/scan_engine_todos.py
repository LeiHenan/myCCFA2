#!/usr/bin/env python3
"""从已装引擎源码里挖「上游自认未做」的格子 —— OPEN 格最硬的一手证据。

为什么这样做：本工作区两轮实证表明"从源码找痛点"会撞上"维护者已知并已处理"；
但**反向使用**同一手段是可靠的：**上游自己写的 TODO / "not supported" / 显式 fallback**，
恰恰是"他们知道但还没做"的清单。这是 OPEN 格的定义性证据（不是我的推测）。

用法：
  python scan_engine_todos.py --root /root/ccfa_venv/.../vllm --tag vllm [--selftest]
  python scan_engine_todos.py --root .../sglang --tag sglang --topics spec,kv,prefix
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

PATTERNS = [
    (r"\bTODO\b[(: ]", "TODO"),
    (r"\bFIXME\b[(: ]", "FIXME"),
    (r"\bXXX\b[(: ]", "XXX"),
    (r"not\s+supported", "not supported"),
    (r"not\s+implemented", "not implemented"),
    (r"unsupported", "unsupported"),
    (r"does\s+not\s+support", "does not support"),
    (r"will\s+be\s+supported\s+in\s+the\s+future", "future work(verbatim)"),
    (r"for\s+now", "for now"),
    (r"fallback", "fallback"),
    (r"workaround", "workaround"),
    (r"raise\s+NotImplementedError", "NotImplementedError"),
    (r"raise\s+ValueError\([^)]*not\s+support", "raise:not supported"),
]
TOPIC_KWS = {
    "spec": ["speculat", "draft", "eagle", "medusa", "ngram", "mtp", "propos"],
    "kv": ["kv_cache", "kv cache", "paged", "block_table", "cache_engine", "prefix_cache",
           "radix", "hicache", "kv_quant", "kv_cache_dtype"],
    "sched": ["schedul", "chunked", "prefill", "decode", "batching", "preempt"],
    "quant": ["quant", "fp8", "int4", "awq", "gptq"],
}
MAX_LINE = 200


def scan(root, topics=None):
    hits = defaultdict(list)
    counts = Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "test", "tests", "csrc")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    for ln, line in enumerate(fh, 1):
                        low = line.lower()
                        for pat, label in PATTERNS:
                            if re.search(pat, line, re.IGNORECASE):
                                counts[label] += 1
                                topics_hit = [t for t, kws in TOPIC_KWS.items()
                                              if any(k in low for k in kws)]
                                key = topics_hit[0] if topics_hit else "other"
                                if topics and key not in topics:
                                    break
                                hits[key].append((os.path.relpath(p, root), ln,
                                                  label, line.strip()[:MAX_LINE]))
                                break
            except Exception:  # noqa: BLE001
                continue
    return hits, counts


def selftest():
    import tempfile
    td = tempfile.mkdtemp()
    os.makedirs(os.path.join(td, "srt"))
    with open(os.path.join(td, "srt", "a.py"), "w") as fh:
        fh.write("# TODO: support kv_cache quantization\n")
        fh.write("x = 1  # not supported for spec decoding\n")
        fh.write("# unrelated comment\n")
    with open(os.path.join(td, "srt", "b.py"), "w") as fh:
        fh.write("raise NotImplementedError('prefix cache for now')\n")
    hits, counts = scan(td)
    assert counts["TODO"] == 1 and counts["not supported"] == 1, counts
    # 设计说明：**逐行只记第一个命中的模式**（避免同一行重复计数）⇒ 该行记到 "for now" 而非
    # "NotImplementedError"（两者都命中）。断言按真实行为写，并单独验证 NotImplementedError 能被识别。
    assert counts["for now"] == 1, counts
    tot = sum(len(v) for v in hits.values())
    assert tot == 3, (tot, dict(hits))
    import re as _re
    assert any(_re.search(pat, "raise NotImplementedError('x')", _re.IGNORECASE)
               for pat, lab in PATTERNS if lab == "NotImplementedError")
    kv = [h for k, v in hits.items() for h in v if "kv" == k]
    assert len(kv) == 1, kv
    print("selftest ✔ 模式匹配/主题分类/计数（3 条命中，kv 类 1 条）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=False)
    ap.add_argument("--tag", default="engine")
    ap.add_argument("--topics")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.root, "需要 --root"
    topics = set(a.topics.split(",")) if a.topics else None
    hits, counts = scan(a.root, topics)
    lines = [f"# {a.tag} 未做清单扫描（root={a.root}）", ""]
    lines.append("## 命中总数")
    for k, v in counts.most_common():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 按主题分组"]
    for topic in sorted(hits):
        lines += [f"### {topic}（{len(hits[topic])} 条）", ""]
        for f, ln, label, txt in hits[topic][:60]:
            lines.append(f"- `{f}:{ln}` **[{label}]** {txt}")
        lines.append("")
    out = a.out or f"/tmp/{a.tag}_todos.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{a.tag}: 命中 {sum(len(v) for v in hits.values())} 条（{len(hits)} 个主题）-> {out}")
    for topic in sorted(hits):
        print(f"  {topic}: {len(hits[topic])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
