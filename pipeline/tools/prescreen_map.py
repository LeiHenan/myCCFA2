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

# ---- FILTER §0 的机械化部分：**按硬件画像分档**
# 教训（2026-09-15）：本文件原先硬编码"单卡 RTX PRO 6000 96GB sm120"，那是 2026-09-13 那台 AutoDL 机，
# **该机已关机**。现役 schoolserver 是 8x RTX 4090 24GB sm89。硬件档写死会让整个"够不着"桶
# 在两个方向上同时错：把 multi-GPU 判死（现机有 8 卡），却放过多卡内存够不着的条目。
# ⇒ 硬件画像必须**显式分档**，且分档本身可自测。
PROFILES = {
    # 现役（默认）
    "ada-8x4090": dict(
        desc="8x RTX 4090 24GB sm89, PCIe-only 无 P2P(host-staged), 2x NUMA(4+4), "
             "driver 550.67 CUDA 12.4, 192 核共享, /home 仅 ~67-79G 可用, vLLM 不可用, HF transformers 可用",
        multi_gpu=True, nvlink=False, ib=False, max_card_gb=24, engines=("hf-transformers",)),
    # 已关机的旧机（保留以便复述旧结论时对照）
    "blackwell-1x-pro6000": dict(
        desc="1x RTX PRO 6000 Blackwell 96GB sm120, driver 580.82.09 CUDA 13, vLLM 0.29 + SGLang 0.5.19",
        multi_gpu=False, nvlink=False, ib=False, max_card_gb=96, engines=("vllm", "sglang", "hf-transformers")),
}
PROFILE = PROFILES["ada-8x4090"]      # 由 --profile 覆盖

# ① 互连/规模：**任何机型都没有** ⇒ 永久硬杀
FABRIC_PAT = re.compile(
    r"\b(multi[- ]?node|infiniband|\bib\b|rdma|nixl|mnnvl|nvlink|nvswitch|cluster|disaggregat\w*)\b", re.I)
# ② 多卡并行：**只在画像没有多卡时**才是硬杀
MULTIGPU_PAT = re.compile(
    r"\b(multi[- ]?gpu|tensor[- ]?parallel|tp\s*=?\s*[2-9]|tp[2-9]\b|"
    r"[2-9]\s*[x×]\s*(h100|h200|a100|b200|b300|4090|3090)|"
    r"(h100|h200|a100|b200|b300)\s*[x×]\s*[2-9]|"
    r"pcp|dcp|context parallel|sequence parallel|expert parallel|pipeline parallel|"
    r"data[- ]parallel|\bdp\b.*\brank)\b", re.I)
# ③ 单卡容量：**旧画像缺的那一半**。命中 80/96/141GB 级卡或显存下限描述 ⇒ 24GB 卡够不着
CAPACITY_PAT = re.compile(
    r"\b(80\s*gb|96\s*gb|141\s*gb|h100|h200|a100|b200|b300|mi300|h800|"
    r"[2-9][0-9]\s*gb\s*(?:card|gpu|vram|memory)|(?:card|gpu|vram|memory)\s*[2-9][0-9]\s*gb)\b", re.I)
# ④ 引擎能力：**不是自动杀**，只作提示列（地图通篇都在提引擎名，自动杀会过度杀伤）
ENGINE_NEEDS = {"vllm": re.compile(r"\bvllm\b", re.I), "sglang": re.compile(r"\bsglang\b", re.I)}

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


def classify(it: dict, profile: dict | None = None) -> dict:
    pf = profile or PROFILE
    blob = it["title"] + "\n" + it.get("block", "")
    fabric = bool(FABRIC_PAT.search(blob))
    multigpu = bool(MULTIGPU_PAT.search(blob))
    capacity = bool(CAPACITY_PAT.search(blob))
    hw_blocks = []
    if fabric:
        hw_blocks.append("fabric")                      # 无该互连：永久
    if multigpu and not pf["multi_gpu"]:
        hw_blocks.append("multi-gpu")                   # 画像只有单卡
    if capacity and pf["max_card_gb"] < 40:
        hw_blocks.append("capacity>%dG" % pf["max_card_gb"])
    hw = bool(hw_blocks)
    needs_engines = [e for e, pat in ENGINE_NEEDS.items() if pat.search(blob)]
    engine_missing = [e for e in needs_engines if e not in pf["engines"]]
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
    return dict(it, hw=hw, hw_why="+".join(hw_blocks), model_family=mdl, correctness=corr,
                occupied=occ, engine_hint=("有引擎依赖:" + ",".join(engine_missing)) if engine_missing else "",
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
    ap.add_argument("--profile", default="ada-8x4090", choices=sorted(PROFILES))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    global PROFILE
    PROFILE = PROFILES[a.profile]
    if a.selftest:
        return selftest()
    if not a.map:
        print("需要 --map", file=sys.stderr)
        return 2

    text = open(a.map, encoding="utf-8").read()
    print("# 硬件画像: %s -- %s" % (a.profile, PROFILE["desc"]))
    items = [classify(x, PROFILE) for x in parse_items(text)]
    if a.only_reachable:
        items = [x for x in items if x["reachable"]]
    if a.bucket:
        items = [x for x in items if x["bucket"].startswith(a.bucket)]

    items.sort(key=lambda x: (BUCKET_ORDER.get(x["bucket"], 9), x["id"]))

    print(f"# {a.map} —— 共 {len(items)} 个格子（分诊，非判定）")
    print("id\tsection\ts3b\texpired\thw\thw_why\tmodel\tcorr\tengine\tbucket\ttitle\tline")
    for x in items:
        print("\t".join([x["id"], x["section"], str(x["s3b"]), str(x["expired"]),
                         str(x["hw"]), x.get("hw_why", ""), str(x["model_family"]), str(x["correctness"]),
                         x.get("engine_hint", ""), x["bucket"], x["title"][:100], str(x["line"])]))
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
    assert got["B1"][0] == "3-够不着", got["B1"]              # mamba + TP8 + NVLink（NVLink 永久杀）
    assert got["B3"][0] == "5-正确性/可观测(非优化类)", got["B3"]
    assert got["B4"][0] == "3-够不着", got["B4"]              # RDMA：永久硬杀
    assert got["B5"][0] == "4-已被占位/已解决(仅用于排除)", got["B5"]
    assert got["C1"][0] == "1-C且失败理由可能已过期", got["C1"]  # rev8 池 ①
    assert got["E1"][0] == "2-E且无人做过", got["E1"]            # rev8 池 ②
    assert got["E2"][0] == "4-已被占位/已解决(仅用于排除)", got["E2"]
    # rev8：B 段不因"没搜到"升权（B 段本身就被降权）
    assert got["B2"][0] == "6-其余(B段与非过期C段,降权)", got["B2"]
    # ---- 画像分档的真断言（2026-09-15 修系统性偏差时加）
    ada, bw = PROFILES["ada-8x4090"], PROFILES["blackwell-1x-pro6000"]
    t_tp = "### B9. Tensor parallel all-reduce chunking on 4 GPUs"
    t_cap = "### B10. A 70B model served on one H100 80 GB card"
    i_tp = classify(parse_items(t_tp)[0], ada)
    i_cap = classify(parse_items(t_cap)[0], ada)
    # 现役是 8 卡 ⇒ TP 类**不再**是硬杀（这正是原版把整桶判死的地方）
    assert not i_tp["hw"], ("ada 画像不该把 TP 判为够不着", i_tp["hw_why"])
    # 现役单卡 24G ⇒ 80G 卡的需求**必须**被杀（原版漏掉的那一半）
    assert i_cap["hw"] and "capacity" in i_cap["hw_why"], ("ada 画像必须杀掉 80G 卡需求", i_cap["hw_why"])
    # 旧画像下 TP 才是硬杀（保证对照仍可复述旧结论）
    i_tp_old = classify(parse_items(t_tp)[0], bw)
    assert i_tp_old["hw"] and "multi-gpu" in i_tp_old["hw_why"], i_tp_old["hw_why"]
    print("selftest OK  | 画像分档断言: ada 下 TP 可达、80G 卡被杀；blackwell-1x 下 TP 被杀")
    for k in sorted(got):
        print(f"   {k:4s} {got[k][0]:32s} s3b={got[k][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
