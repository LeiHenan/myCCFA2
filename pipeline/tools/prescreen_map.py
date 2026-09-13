#!/usr/bin/env python3
"""把三态地图（A 已闭合 / B 开放 / C 试过被放弃 / D 硬件排除）**初筛成一张可读的 TSV**。

⚠️ 这是**分诊助手，不是判据**。它只做 FILTER §0 里能机械化的那一半：
   「本机够不够得着」（硬件/模型族）＋「是不是优化类」＋「有没有明写被占位」。
   真正的 §1–§7（差异轴条件性、整机账、可持续性、S3b 三问）**必须人判**——
   判据文件是 `notes/FILTER_3AXIS.md`，本脚本不得替代它。

为什么要它：三张地图合起来 400+ 个格子，靠肉眼扫会漏；而"漏掉一个已判死的格子再重做一遍"
是本工作区已经犯过的错（decision 第 2 轮重复发现深度轴，见 `notes/OCCUPANCY_LEDGER.md` 的由来）。

**格式前提（重要）**：本工具按"**每个格子 = 一个 `### B<数字>. 标题` 块**"来解析，
即 `KV_CACHE_GAP_MAP.md` 与 `INFERENCE_ACCEL_GAP_MAP.md` 的形态。
`spec-decode-gap-map-2026-09-13.md` 的 `###` 是**主题分组**（A1–A6 / B1–… / C1–…），
格子是组内的**表格行** ⇒ 对它只能做到"组级"分诊（实测 20 组），这是格式差异、不是 bug。

用法：
  prescreen_map.py --map KV_CACHE_GAP_MAP.md
  prescreen_map.py --map INFERENCE_ACCEL_GAP_MAP.md --only-reachable
  prescreen_map.py --selftest
"""
from __future__ import annotations

import argparse
import re
import sys

# ---- FILTER §0 的机械化部分：本机（单卡 RTX PRO 6000 96GB sm120 / Qwen3-4B dense）够不着的东西
HARDWARE_PAT = re.compile(
    r"\b(multi[- ]?node|multi[- ]?gpu|tensor[- ]parallel|tp\s*=?\s*[2-9]|tp[2-9]\b|"
    r"nvlink|infiniband|\bib\b|rdma|nixl|mnnvl|cluster|disaggregat|"
    r"[2-9]\s*[x×]\s*(h100|h200|a100|b200|b300|4090|3090)|"
    r"(h100|h200|a100|b200|b300)\s*[x×]\s*[2-9]|"
    r"\b(pcp|dcp|context parallel|sequence parallel|expert parallel|pipeline parallel)\b|"
    r"\bdp\b.*\brank|data[- ]parallel)\b",
    re.I,
)
# 模型族够不着：我们的目标是 dense 全注意力 Qwen3-4B
MODEL_PAT = re.compile(
    r"\b(hybrid|mamba|gdn|gated.?delta|linear attention|ssm|recurrent state|"
    r"\bmla\b|deepseek|nope|latent attention|moe|mixture.of.experts)\b", re.I,
)
# 非优化类（正确性/可观测性/文档）
# 注意：不能用 `\b` 收尾去匹配**前缀型**词干（如 non-determin -> non-deterministic 处 \b 不成立）。
# 自测抓过这个 bug：`\bnon-determin\b` 永远匹配不到 "non-deterministic"。
CORRECT_PAT = re.compile(
    r"(\bnan\b|\bcorrupt\w*|\bwrong\b|\bincorrect\w*|\bcrash\w*|\bassert\w*|"
    r"\bdeadlock\w*|\bhang\w*|\bdata ?loss\b|\bdouble.?free\w*|\bmemory leak\w*|"
    r"\bcorrectness\b|\bnon-?determin\w*|\bdeterminis\w*|\breproducib\w*|"
    r"\blogprob\w*|\bobservab\w*|\bundocumented\b|\btypo\w*|\bdoc(?:s|umentation)\b)",
    re.I,
)
# 已有 OPEN 占位者的信号
OCCUPIED_PAT = re.compile(r"\b(OPEN|open PR|still open|awaiting|needs[- ]rebase|unmerged|WIP|RFC)\b")


def parse_items(text: str) -> list[dict]:
    """抽出 B/C/D 段里以 `###`/`####` 开头、且标题带 B#/C#/D# 编号的格子及其块文本。"""
    lines = text.splitlines()
    # 只认**带数字编号**的格子标题（`### B12. xxx`）；这样 `## B. OPEN` 这种段标题不会被当成格子。
    hdr = re.compile(r"^#{2,4}\s+([BCD]\d+(?:\.\d+)*)\.\s+(\S.*)$")
    items: list[dict] = []
    cur: dict | None = None
    for i, ln in enumerate(lines):
        m = hdr.match(ln)
        if m:
            if cur:
                cur["block"] = "\n".join(cur.pop("_buf")).strip()
                items.append(cur)
            cur = {"id": m.group(1), "title": m.group(2).strip(), "_buf": [], "line": i + 1}
            continue
        if cur is not None:
            if ln.startswith("## "):
                cur["block"] = "\n".join(cur.pop("_buf")).strip()
                items.append(cur)
                cur = None
                continue
            cur["_buf"].append(ln)
    if cur:
        cur["block"] = "\n".join(cur.pop("_buf")).strip()
        items.append(cur)
    return items


def classify(it: dict) -> dict:
    blob = it["title"] + "\n" + it.get("block", "")
    hw = bool(HARDWARE_PAT.search(blob))
    mdl = bool(MODEL_PAT.search(blob))
    corr = bool(CORRECT_PAT.search(blob))
    occ = bool(OCCUPIED_PAT.search(blob))
    # 可达性：硬件与模型族都够得着
    reachable = not hw and not mdl
    # 分诊桶
    if not reachable:
        bucket = "0-够不着"
    elif corr:
        bucket = "1-正确性/可观测(非优化类)"
    else:
        bucket = "2-可能候选"
    return dict(it, hw=hw, model_family=mdl, correctness=corr, occupied=occ,
                reachable=reachable, bucket=bucket)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=False)
    ap.add_argument("--only-reachable", action="store_true")
    ap.add_argument("--bucket", help="只打印某个桶（前缀匹配）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.map:
        print("需要 --map", file=sys.stderr)
        return 2

    text = open(a.map, encoding="utf-8").read()
    items = [classify(x) for x in parse_items(text)]
    if a.only_reachable:
        items = [x for x in items if x["reachable"]]
    if a.bucket:
        items = [x for x in items if x["bucket"].startswith(a.bucket)]

    order = {"2-可能候选": 0, "1-正确性/可观测(非优化类)": 1, "0-够不着": 2}
    items.sort(key=lambda x: (order.get(x["bucket"], 9), x["id"]))

    print(f"# {a.map} —— 共 {len(items)} 个格子（分诊，非判定）")
    print("id\ttitle\thw\tmodel\tcorr\toccupied\tbucket\tline")
    for x in items:
        print("\t".join([x["id"], x["title"][:110], str(x["hw"]), str(x["model_family"]),
                         str(x["correctness"]), str(x["occupied"]), x["bucket"], str(x["line"])]))
    print()
    from collections import Counter
    print("# 分桶统计:", dict(Counter(x["bucket"] for x in items)))
    print("# ⚠️ 分诊只代表『本机够得着 + 不是明显的正确性类』；§1–§7 与 S3b 三问仍须逐条人判。")
    return 0


def selftest() -> int:
    md = """## B. OPEN
### B1. Hybrid mamba prefix caching on TP8 across NVLink
- WHO: someone
### B2. Sampling kernel overhead on a single GPU
- WHO: someone
### B3. Non-deterministic greedy output after cache reuse
- WHO: someone
### B4. RDMA KV transfer descriptor explosion
- WHO: someone
"""
    items = [classify(x) for x in parse_items(md)]
    got = {x["id"]: x["bucket"] for x in items}
    assert len(items) == 4, items
    assert got["B1"] == "0-够不着", got            # mamba + TP8 + NVLink
    assert got["B2"] == "2-可能候选", got           # 单卡采样开销
    assert got["B3"] == "1-正确性/可观测(非优化类)", got
    assert got["B4"] == "0-够不着", got            # RDMA
    print("selftest OK", got)
    return 0


if __name__ == "__main__":
    sys.exit(main())
