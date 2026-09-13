#!/usr/bin/env python3
"""选题 pipeline 的公共库 —— 供 `check.py`（校验）与 `new_candidate.py`（脚手架）共用。

唯一事实源是 `pipeline/gates.json`：本文件不重复定义任何判据，只负责读取、生成骨架、解析产物。
这样"文档里写的"与"机器检查的"不会漂移。
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
GATES = os.path.join(HERE, "gates.json")

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)
BOX_RE = re.compile(r"^-\s+\[( |x|X)\]\s+(.*)$", re.M)
PAIN_RE = re.compile(r"^#{2,3}\s*(P\d+[^\n]*)$", re.M)
# 必须 re.M（痛点块是多行字符串）；`\*{0,2}` 兼容 `- **证据**：` 这类加粗写法
EVID_RE = re.compile(r"证据\*{0,2}\s*[：:]\s*(.*)$", re.M)


def load_gates(path=GATES):
    with open(path, encoding="utf-8") as fh:
        g = json.load(fh)
    g["_by_id"] = {s["id"]: s for s in g["stages"]}
    return g


def stage_index(sid):
    return int(sid[1:])


def skeleton(stage, slug="", title=""):
    """按 gates.json 生成一个阶段的填写骨架（小节 + 可勾选清单）。"""
    head = f"# {stage['id']} {stage['name']}"
    if title:
        head += f" — {title}"
    L = [head, "",
         f"**候选**：`{slug}` ｜ **成本上限**：{stage.get('cost', '—')}",
         f"**目标**：{stage['goal']}", ""]
    for s in stage["sections"]:
        L += [f"## {s}", "", "（填写）", ""]
    L += ["## 完成清单", "",
          "> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。", ""]
    L += [f"- [ ] {c}" for c in stage["checklist"]]
    L += ["", f"> **杀出口**：{stage.get('kill', '无')}", ""]
    return "\n".join(L)


def parse_artifact(path):
    """→ (原文, 小节名集合, [(勾选项文本, 是否已勾)])"""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    sections = {m.group(1).strip() for m in SECTION_RE.finditer(text)}
    boxes = [(m.group(2).strip(), m.group(1).lower() == "x") for m in BOX_RE.finditer(text)]
    return text, sections, boxes


def match_box(item, boxes):
    """清单项与产物里的勾选行做前缀匹配（允许产物里在清单项后补写说明）。"""
    key = " ".join(item.split())
    for txt, checked in boxes:
        if " ".join(txt.split()).startswith(key):
            return checked
    return None


def pain_blocks(text):
    """切出痛点条目（`### P1 ...` 形式）→ [(标题, 该条正文)]"""
    hits = list(PAIN_RE.finditer(text))
    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        out.append((m.group(1).strip(), text[m.start():end]))
    return out


def evidence_ok(block):
    """该痛点是否有可用证据：含链接/命令/路径，或显式标注 L0 + 未取证。"""
    m = EVID_RE.search(block)
    if not m:
        return False, "缺『证据：』字段"
    val = m.group(1).strip()
    if not val or val in {"-", "—", "无"}:
        return False, "证据字段为空"
    if "L0" in block and "未取证" in block:
        return True, "L0 + 未取证（允许，但不进入下一阶段）"
    if re.search(r"https?://|`[^`]+`|\$ \S+|/\w+/\S+", val):
        return True, "含链接/命令/路径"
    return False, f"证据不可复现：{val[:40]}"


def scan_banned(text, cfg):
    """→ (blocking 列表, warning 列表)；同行的『豁免：』可解除 blocking。"""
    block, warn = [], []
    for i, line in enumerate(text.splitlines(), 1):
        for w in cfg.get("banned_words_block", []):
            if w in line and cfg.get("banned_words_block_exempt_marker", "豁免：") not in line:
                block.append((i, w, line.strip()[:80]))
        for w in cfg.get("banned_words_warn", []):
            if w in line:
                warn.append((i, w, line.strip()[:80]))
    return block, warn
