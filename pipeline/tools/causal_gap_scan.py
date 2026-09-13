#!/usr/bin/env python3
"""从三态地图里挖出**「上游断言了因果、但没有做匹配对照」的裂缝**（FILTER 附录 A.6 的机会类型）。

**为什么这是一个独立的机会类型（而不是又一份清单）**：
三态地图的 B 段是靠"有人还在问"建起来的，而"有人还在问"最容易长出 PR ⇒ 按 B 段找候选会被引向最热闹的地方
（本轮 7 条线索、7 次 S3b 全部命中 OPEN PR 已实证）。但 PR 修的是**功能**，几乎没人去**复核因果**。
于是存在一类结构性缺口：**上游写了"The trigger is X" / "the real bug is Y" / "this is due to Z"，
却没有把 X（或 Y、Z）单独变过一次的匹配对照**。这类缺口：
  · 不占位（没有 PR 在做"复核因果"这件事）；
  · **便宜**——匹配对照通常是分钟级实验，不是多周工程；
  · 形状正好是 FILTER §3 要的"**条件性差异轴**"（换一个变量，结论不同）；
  · 且**已被验证过一次**：我用 K 匹配对照纠正了 vLLM #48494 的"The presence of the batch-size table is the trigger"
    （实测表的存在本身 −0.51%，真正花钱的是 K=0 档 −31.7%）。

**本脚本只做机械筛选**：找出**同时含因果标记与量/性能标记**的句子，交给人工做两步判断——
  ① 该断言**点名了哪个变量**？（必须是一个**可单独变动**的配置/机制，而不是"整体更慢"）
  ② 引文里**有没有**一个只变那个变量的对照？（若对照已存在于引文中 ⇒ 不是裂缝，丢弃）

用法：
  causal_gap_scan.py --map KV_CACHE_GAP_MAP.md --map spec-decode-gap-map-2026-09-13.md
  causal_gap_scan.py --map X.md --max 40
  causal_gap_scan.py --selftest
"""
from __future__ import annotations

import argparse
import re
import sys

# 因果断言标记：上游"点名了一个原因"
CAUSAL = re.compile(
    r"(the (?:real )?(?:bug|trigger|cause|reason|culprit|source|problem|issue) (?:is|was|here)\b"
    r"|is (?:the|a) (?:real )?(?:trigger|cause|culprit|root cause)\b"
    r"|(?:is|was|are|were) (?:caused|triggered|driven|due) (?:by|to)\b"
    r"|because of\b|attribut\w+ to\b|stem(?:s|med) from\b|root cause\b"
    r"|the reason (?:is|was|being)\b|that is why\b|which is why\b|comes from\b"
    r"|(?:is|are) the (?:deciding|dominant|binding) factor\b"
    r"|confirm\w* (?:that|the)\b.*\b(?:cause|trigger|root)\b)",
    re.I,
)
# 量/性能标记：断言必须带数字或性能后果，否则不值得复核
# ⚠️ 坑（自测抓到过，且这是**同类第二次**）：`\b` 收尾**不能**用在 `%` / `×` 这类**非单词字符**单位后面 ——
# `131%` 后面是空格，两侧都不是 word char ⇒ `\b` 永不成立 ⇒ 整个 QUANT 静默失效、扫描返回 0 条。
# 所以 `\b` 只加在**以字母结尾**的单位上（x\b / ms\b / GiB\b …），`%` 与 `×` 不加。
QUANT = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:%|×|x\b|tok/s|tokens/s|t/s|ms\b|GiB\b|MiB\b|GB\b|MB\b|tokens?\b)"
    r"|\b(?:throughput|latency|tpot|ttft|itl|hit rate|acceptance|slowdown|regression|penalt|collapse|slower|faster)\w*)",
    re.I,
)
# "对照的痕迹"：如果句子里已经出现只变一个变量的迹象，降低优先级
CONTROL_HINT = re.compile(
    r"(control\b|holding\b|everything else (?:identical|equal)|same (?:config|prompts|model)\b"
    r"|only (?:the )?\w+ (?:changed|differs|differ)|other\w* (?:identical|equal)"
    r"|A/B\b|ablation|isolat\w+|matched\b)",
    re.I,
)
# 明确点名"可单独变动的变量"的线索词（用于排序：命中越多越值得复核）。
# ⚠️ 坑（自测抓到）：**不能**把 `[A-Z][A-Z0-9_]{3,}` 放进一个 `re.I` 的正则里 ——
# 开了 re.I 之后它会退化成"**任意长度≥4 的词**"，于是 `Flash` / `block` / `from` / `homogeneous` 全被当成"变量名"，
# 把排序信号彻底污染。所以拆成**大小写敏感**的全大写识别器 + **大小写不敏感**的已知标识符识别器。
NAMED_VAR_CS = re.compile(r"[A-Z][A-Z0-9_]{3,}")
NAMED_VAR_CI = re.compile(
    r"(`[a-z_][a-z0-9_]*`|--[a-z][a-z0-9-]+"
    r"|num_speculative_tokens\w*|cudagraph\w*|block[_ ]?size\w*|page[_ ]?size\w*"
    r"|prefix cach\w*|batch[_ ]size\w*|lcm\w*)",
    re.I,
)


def named_vars(text: str) -> list[str]:
    vals = {m.group(0) for m in NAMED_VAR_CS.finditer(text)}
    vals |= {m.group(0) for m in NAMED_VAR_CI.finditer(text)}
    return sorted(vals)[:4]


def split_sentences(text: str) -> list[str]:
    # 保留 markdown 结构但把长段落切句；数字里的句点不切
    text = re.sub(r"\n+", " \n ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[`\"'])", text)
    out = []
    for p in parts:
        for ln in p.split(" \n "):
            ln = ln.strip()
            if ln:
                out.append(ln)
    return out


def scan(text: str) -> list[dict]:
    """扫**1–2 句的滑窗**，不是单句。

    为什么必须用滑窗（自测抓到这个真缺陷）：上游的写法经常是
        "The presence of the batch-size table is the trigger; K=0 tiers are not required. This cost 12-25% throughput."
    —— **因果在前一句、数字在后一句**。只扫单句会**结构性漏掉**这类最典型的断言。
    """
    sents = split_sentences(text)
    best: dict[str, dict] = {}
    for i, s in enumerate(sents):
        for w in (s, s + " " + sents[i + 1] if i + 1 < len(sents) else s):
            if not (CAUSAL.search(w) and QUANT.search(w)):
                continue
            key = re.sub(r"\s+", " ", s).strip()      # 以**首句**为键，保留最短窗口
            cand = {
                "sentence": re.sub(r"\s+", " ", w).strip(),
                "has_control": bool(CONTROL_HINT.search(w)),
                "named_vars": named_vars(w),
                "n_sent": 1 if w == s else 2,
            }
            if key not in best or len(cand["sentence"]) < len(best[key]["sentence"]):
                best[key] = cand
    return list(best.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="append", required=False)
    ap.add_argument("--max", type=int, default=25)
    ap.add_argument("--include-with-control", action="store_true",
                    help="也打印句内已带对照痕迹的（默认排除，因为它们多半不是裂缝）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.map:
        print("需要 --map", file=sys.stderr)
        return 2

    for path in a.map:
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as e:
            print(f"跳过 {path}: {e}", file=sys.stderr)
            continue
        hits = scan(text)
        if not a.include_with_control:
            hits = [h for h in hits if not h["has_control"]]
        # 优先：点名了可单独变动的变量
        hits.sort(key=lambda h: (-len(h["named_vars"]),))
        print("=" * 100)
        print(f"# {path} —— 因果断言（带量/性能后果）{len(hits)} 条，显示前 {a.max}")
        print("=" * 100)
        for i, h in enumerate(hits[: a.max], 1):
            mark = "·" if h["named_vars"] else " "
            print(f"\n[{i}]{mark} vars={h['named_vars']}")
            print(f"    {h['sentence'][:400]}")
    print("\n# 人工两步：① 断言点名了哪个**可单独变动**的变量？② 引文里有没有只变它的对照？")
    print("#   若②为否 ⇒ 这是一个「匹配对照裂缝」：可用分钟级实验复核，且没有 PR 在做这件事。")
    print("#   ⚠️ 本脚本不做判断，只做机械筛选；不得据其输出直接宣布'上游错了'。")
    return 0


def selftest() -> int:
    md = """
- The real bug is 131% pool overflow from the homogeneous lcm=256 physical block layout vs DSv4-Flash's
  {256, 64, 4, 8}-block-size KV groups, so some eviction every alloc cycle is mandatory.
- The presence of the batch-size table is the trigger; K=0 tiers are not required. This cost 12-25% throughput.
- Holding everything else identical and changing only `cudagraph_mode`, the 8% TPOT regression is caused by the graph capture path.
- We saw a large slowdown in this workload.
- The cause of the collapse is unknown.
"""
    hits = scan(md)
    joined = " || ".join(h["sentence"] for h in hits)
    assert "131% pool overflow" in joined, hits            # 因果 + 数字
    assert "presence of the batch-size table" in joined    # 因果 + 数字（这是已被我复核掉的那条）
    # 该条**确实含因果标记**（"is caused by"）**且**含对照痕迹 ⇒ 必须出现，并被标为 has_control。
    # （曾经的写法用一条**没有因果标记**的句子来测 has_control —— 那是**测试用例本身错了**，不是代码错。）
    ctrl = [h for h in hits if h["has_control"]]
    assert any("cudagraph_mode" in h["sentence"] for h in ctrl), "应把带对照痕迹的那条标出来"
    assert not any("cause of the collapse is unknown" in h["sentence"] for h in hits), "无数量的纯因果不应入选"
    assert not any("large slowdown in this workload" in h["sentence"] for h in hits), "无因果标记不应入选"
    v = [h for h in hits if "131% pool overflow" in h["sentence"]][0]["named_vars"]
    assert any("lcm" in x.lower() for x in v), f"应认出 lcm=256 这个被点名的变量：{v}"
    assert "from" not in [x.lower() for x in v] and "block" not in [x.lower() for x in v], \
        f"大小写敏感修复后不应再把普通词当变量名：{v}"
    print("selftest OK —— 命中", len(hits), "条；带对照痕迹", len(ctrl), "条")
    for h in hits:
        print(f"   ctrl={h['has_control']} vars={h['named_vars']} :: {h['sentence'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
