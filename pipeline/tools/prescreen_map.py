#!/usr/bin/env python3
"""把三态地图（A 已闭合 / B 开放 / C 试过被放弃 / D 硬件排除 / **E 从未被讨论过**）**初筛成一张可读的 TSV**。

⚠️ 这是**分诊助手，不是判据**。它只做 FILTER §0 里能机械化的那一半：
   「本机够不够得着」（硬件/模型族）＋「是不是优化类」＋「有没有明写被占位」。
   真正的 §1–§7（差异轴条件性、整机账、可持续性、S3b 三问）**必须人判**——
   判据文件是 `notes/FILTER_3AXIS.md`（含**附录 A 事后修订**），本脚本不得替代它。

**分桶顺序直接来自 `goal-a729588f` rev8 的候选池优先级**（不是我自己发明的顺序）：
   ① **C 段且标了 `REASON-MAY-HAVE-EXPIRED`** —— 失败理由依赖某个已改变的约束（FILTER §1a）
   ② **E 段且标了 `NO-ACTIVE-WORK-FOUND`** —— 从未被任何 issue/PR 讨论过（唯一结构上可能空闲的来源）
   ③ **够不着**（硬件 >1 卡 / 多节点 / 模型族是 hybrid-mamba-MLA-MoE 等）
   ④ **已被占位/已解决**（S3b 标了 `OCCUPIED(...)` / `SOLVED(...)`）—— 只用于排除
   ⑤ **正确性/可观测类**（不是优化类）
   ⑥ **其余**（未标过期的 C 段 + 其余 B 段；**降权**：B 段 ≈ 在飞工作的快照，见 OCCUPANCY_LEDGER §七）

**格式前提**：按"**每个格子 = 一个 `### B<数字>. 标题` 块**"解析，
即 `KV_CACHE_GAP_MAP.md` 与（升级后的）`INFERENCE_ACCEL_GAP_MAP.md` 的形态。
`spec-decode-gap-map-2026-09-13.md` 的 `###` 是**主题分组**（A1–A6 / B1–… / C1–…），
格子是组内的**表格行** ⇒ 对它只能做到"组级"分诊，这是格式差异、不是 bug。

用法：
  prescreen_map.py --map KV_CACHE_GAP_MAP.md
  prescreen_map.py --map INFERENCE_ACCEL_GAP_MAP.md --bucket 1
  prescreen_map.py --map X.md --only-reachable
  prescreen_map.py --selftest
"""
from __future__ import annotations

import argparse
import re
import sys

# ---- FILTER §0 的机械化部分：本机（单卡 RTX PRO 6000 96GB sm120 / Qwen3-4B dense）够不着的东西
HARDWARE_PAT = re.compile(
    r"\b(multi[- ]?node|multi[- ]?gpu|tensor[- ]?parallel|tp\s*=?\s*[2-9]|tp[2-9]\b|"
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
OCCUPIED_PAT = re.compile(r"\b(OPEN|open PR|still open|awaiting|needs[- ]rebase|unmerged|WIP|RFC)\b")

# ---- 新版地图的结构性标记（rev8 要求地图必须带上它们）
S3B_OCCUPIED = re.compile(r"OCCUPIED\s*\(([^)]*)\)", re.I)
S3B_SOLVED = re.compile(r"SOLVED\s*\(([^)]*)\)", re.I)
S3B_NOACTIVE = re.compile(r"NO-ACTIVE-WORK-FOUND(?:\s*\(([^)]*)\))?", re.I)
EXPIRED = re.compile(r"REASON-MAY-HAVE-EXPIRED", re.I)


def parse_items(text: str) -> list[dict]:
    """抽出 B/C/D/**E** 段里以 `###`/`####` 开头、且标题带编号的格子及其块文本。"""
    lines = text.splitlines()
    # 只认**带数字编号**的格子标题（`### B12. xxx`）；这样 `## B. OPEN` 这种段标题不会被当成格子。
    hdr = re.compile(r"^#{2,4}\s+([BCDE]\d+(?:\.\d+)*)\.\s+(\S.*)$")
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
    sec = it["id"][0]

    mo, ms, mn = S3B_OCCUPIED.search(blob), S3B_SOLVED.search(blob), S3B_NOACTIVE.search(blob)
    if mo:
        s3b = "OCCUPIED"
    elif ms:
        s3b = "SOLVED"
    elif mn:
        s3b = "NO-ACTIVE-WORK-FOUND"
    else:
        s3b = None
    expired = bool(EXPIRED.search(blob))

    reachable = not hw and not mdl
    if not reachable:
        bucket = "3-够不着"
    elif s3b in ("OCCUPIED", "SOLVED"):
        bucket = "4-已被占位/已解决(仅用于排除)"
    elif sec == "C" and expired:
        bucket = "1-C且失败理由可能已过期"          # rev8 候选池 ①
    elif sec == "E" and s3b == "NO-ACTIVE-WORK-FOUND":
        bucket = "2-E且无人做过"                      # rev8 候选池 ②
    elif corr:
        bucket = "5-正确性/可观测(非优化类)"
    else:
        bucket = "6-其余(B段与非过期C段,降权)"
    return dict(it, hw=hw, model_family=mdl, correctness=corr, occupied=occ,
                section=sec, s3b=s3b, expired=expired,
                reachable=reachable, bucket=bucket)


BUCKET_ORDER = {
    "1-C且失败理由可能已过期": 0,
    "2-E且无人做过": 1,
    "3-够不着": 2,
    "4-已被占位/已解决(仅用于排除)": 3,
    "5-正确性/可观测(非优化类)": 4,
    "6-其余(B段与非过期C段,降权)": 5,
}


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

    items.sort(key=lambda x: (BUCKET_ORDER.get(x["bucket"], 9), x["id"]))

    print(f"# {a.map} —— 共 {len(items)} 个格子（分诊，非判定）")
    print("id\tsection\ts3b\texpired\thw\tmodel\tcorr\tbucket\ttitle\tline")
    for x in items:
        print("\t".join([x["id"], x["section"], str(x["s3b"]), str(x["expired"]),
                         str(x["hw"]), str(x["model_family"]), str(x["correctness"]),
                         x["bucket"], x["title"][:100], str(x["line"])]))
    print()
    from collections import Counter
    print("# 分桶统计:", dict(Counter(x["bucket"] for x in items)))
    print("# ⚠️ 分诊只代表『本机够得着 + 不是明显的正确性类』；§1–§7 与 S3b 三问仍须逐条人判。")
    if not any(x["section"] == "E" for x in items):
        print("# ⚠️ 本图**没有 E 段**（从未被讨论过/负面证据）⇒ 缺 rev8 候选池 ② 的唯一来源。")
    if not any(x["s3b"] for x in items):
        print("# ⚠️ 本图**没有任何 S3b 状态标签** ⇒ 无法据以排除占位者，须自行补做 S3b。")
    return 0


def selftest() -> int:
    md = """## B. OPEN
### B1. Hybrid mamba prefix caching on TP8 across NVLink
- WHO: someone
### B2. Sampling kernel overhead on a single GPU
- WHO: someone
  - S3b: NO-ACTIVE-WORK-FOUND (queries: "sampling overhead" on GitHub, arXiv)
### B3. Non-deterministic greedy output after cache reuse
- WHO: someone
### B4. RDMA KV transfer descriptor explosion
- WHO: someone
### B5. Some cudagraph memory knob
- WHO: someone
  - S3b: OCCUPIED (open PR #12345)
## C. ABANDONED
### C1. Fused RMSNorm attempt dropped because sm90 had no kernel
- STATED REASON: "..."
  - REASON-MAY-HAVE-EXPIRED (sm120 now has the kernel)
### C2. Some other dropped thing
- STATED REASON: "..."
## E. NEVER-DISCUSSED
### E1. Vocab-parallel top-k fallback on this backend
- NO-ACTIVE-WORK-FOUND (queries: "vocab parallel topk", "logits processor fallback")
### E2. Something someone already filed
- OCCUPIED (open issue #999)
"""
    items = [classify(x) for x in parse_items(md)]
    got = {x["id"]: (x["bucket"], x["s3b"]) for x in items}
    assert len(items) == 9, (len(items), [x["id"] for x in items])
    assert got["B1"][0] == "3-够不着", got["B1"]              # mamba + TP8 + NVLink
    assert got["B3"][0] == "5-正确性/可观测(非优化类)", got["B3"]
    assert got["B4"][0] == "3-够不着", got["B4"]              # RDMA
    assert got["B5"][0] == "4-已被占位/已解决(仅用于排除)", got["B5"]
    assert got["C1"][0] == "1-C且失败理由可能已过期", got["C1"]  # rev8 池 ①
    assert got["E1"][0] == "2-E且无人做过", got["E1"]            # rev8 池 ②
    assert got["E2"][0] == "4-已被占位/已解决(仅用于排除)", got["E2"]
    # rev8：B 段不因"没搜到"升权（B 段本身就被降权）
    assert got["B2"][0] == "6-其余(B段与非过期C段,降权)", got["B2"]
    print("selftest OK")
    for k in sorted(got):
        print(f"   {k:4s} {got[k][0]:32s} s3b={got[k][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
