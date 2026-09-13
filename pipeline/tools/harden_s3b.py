#!/usr/bin/env python3
"""按第 6 轮的教训加固 S3b 判据：**"开关存在" ≠ "问题已解决"**（幂等）。

教训来源（2026-09-13，goal-ebcec24d 第 3→6 轮）：
第 3 轮我把候选 `batch-invariance-cost` 判死，理由是"两家引擎都已 ship 专用开关"。
第 6 轮一份跨引擎检索显示该推理**不成立**：上游自己在追踪 issue（vLLM #27433 **OPEN**，93 条评论）、
有**open issues 报告"开了开关仍然发散"**（#47069 FA3/sm_90、#51187 高并发、#56370、vllm-ascend #14884）、
修复 PR 仍在陆续 merge（#48391 RMSNorm block size pin）、且文档与代码互相矛盾（envs.py 说 sm≥9.0，docs 说 8.0+）。
⇒ 隐含前提"开关存在 ⇒ 已解决"是错的。**判据必须多问一步**：
   ① 开关在**我这代硬件/这个负载形态**上是否被验证有效？
   ② 是否仍有**未关闭**的同类失败报告？
   ③ **文档与实现是否一致**？
本脚本把这三问写进 gates.json 的 S3b，并把它加进 OCCUPANCY_LEDGER 的维护规则。

用法：python pipeline/tools/harden_s3b.py [--check]
"""
import argparse
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GATES = os.path.join(REPO, "pipeline", "gates.json")

ITEM = ("**若最近邻是「已 ship 的开关/特性」**：不得据『开关存在』判死 —— 必须再核三件事"
        "（① 该开关在**本代硬件与本负载形态**上是否被验证有效？② 是否仍有**未关闭**的同类失败报告？"
        "③ **文档与实现是否一致**？），并把三条的取证写进判定矩阵；"
        "命中任一『失效/未关闭/不一致』⇒ 该对象记 ⚠ 而非 🔴")
KILL_ADD = ("；**v1.1.2 追加**：若最近邻只是『已 ship 一个开关』而**未验证其在本代硬件/本负载上有效**、"
            "或**仍有未关闭的同类失败报告**、或**文档与实现不一致** ⇒ **不得据此判死**，须降级为 ⚠ 并继续取证")
NOTE = ("；**v1.1.2 追加**：判据来源 = 2026-09-13 第 3→6 轮的自我更正 —— 第 3 轮把 `batch-invariance-cost` 判死的理由是"
        "『两家引擎都已 ship 专用开关』，而第 6 轮的跨引擎检索证明该推理不成立（vLLM #27433 仍 OPEN／"
        "open issues 报告开了开关仍发散／修复 PR 仍在 merge／envs.py 与 docs 的算力下限互相矛盾）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    g = json.load(io.open(GATES, encoding="utf-8"), object_pairs_hook=collections.OrderedDict)
    g.pop("_by_id", None)
    s3b = [s for s in g["stages"] if s["id"] == "S3b"][0]
    changed = []
    if ITEM not in s3b["checklist"]:
        s3b["checklist"].insert(1, ITEM)
        changed.append("S3b: 插入『已 ship 开关的三问』判据")
    if "v1.1.2" not in s3b.get("kill", ""):
        s3b["kill"] = s3b["kill"] + KILL_ADD
        changed.append("S3b: kill 出口加限定")
    if "v1.1.2" not in s3b.get("evidence_from_repo", ""):
        s3b["evidence_from_repo"] = s3b.get("evidence_from_repo", "") + NOTE
        changed.append("S3b: evidence 追加")
    g["version"] = "1.1.2"
    g["updated"] = "2026-09-13"
    if a.check:
        print("checks:", changed or "已是最新")
        return 0
    io.open(GATES, "w", encoding="utf-8").write(json.dumps(g, ensure_ascii=False, indent=2) + "\n")
    # OCCUPANCY_LEDGER 的维护规则同步
    led = os.path.join(REPO, "notes", "OCCUPANCY_LEDGER.md")
    if os.path.exists(led):
        s = io.open(led, encoding="utf-8").read()
        add = ("5. **🔴 的判定标准（v1.1.2 收紧）**：只有『已 ship **且** 在本代硬件/本负载上被验证有效 **且** 无未关闭的同类失败报告』"
               "才可标 🔴；若只是『有个开关』而有效性未验、或仍有 open 失败报告、或文档与实现不一致 ⇒ 记 **⚠**。"
               "本条来自 2026-09-13 的自我更正：第 3 轮据『两家都 ship 了开关』把 `batch-invariance-cost` 判死，"
               "而检索显示 vLLM #27433 仍 OPEN、多个 open issue 报告开了开关仍发散、修复 PR 仍在 merge。\n")
        if "v1.1.2 收紧" not in s:
            assert "4. **不再新增条目" in s or "**维护规则**" in s, "ledger 锚点未命中"
            s = s.replace("**维护规则**：", "**维护规则**：\n\n" + add + "\n**其余**：")
            io.open(led, "w", encoding="utf-8").write(s)
            changed.append("OCCUPANCY_LEDGER: 加 🔴 判定标准")
    print("hardened:", changed or "已是最新（幂等）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
