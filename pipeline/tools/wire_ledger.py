#!/usr/bin/env python3
"""把 OCCUPANCY_LEDGER 接进 S3b 判据与 PIPELINE.md（幂等，可重复跑）。

为什么：第 2 轮「重复发现 4 天前已判死的深度轴」的根因是**没有先查负面结果库存**。
本脚本确保「先查库存」是 S3b 的**第 0 步**，而不是又一份没人看的文档。

用法：python pipeline/tools/wire_ledger.py [--check]
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
PIPE = os.path.join(REPO, "PIPELINE.md")

ITEM = ("**先查 `notes/OCCUPANCY_LEDGER.md`（已知被占图）**：任何方向动手前 grep 该表；"
        "命中 🔴 已解决时必须指出差异轴是**条件性**的且上游未覆盖该条件，否则直接放弃"
        "（避免重复发现已被否证的轴）")
NOTE = ("；**v1.1 追加**：`notes/OCCUPANCY_LEDGER.md` 把 27 类问题类的占位者集中成机器可读库存"
        "（含墓碑 GPU·h），直接来自 2026-09-13 第 2 轮「重复发现 4 天前已判死的深度轴」这一实证。")
PIPE_OLD = ("- 动作：① 用**问题类**描述候选（不是某个引擎的功能名）；"
            "② 跑 `python pipeline/solved_scan.py --keywords … --repos …` 生成待查清单；")
PIPE_NEW = ("- 动作：⓪ **先查 [`notes/OCCUPANCY_LEDGER.md`](notes/OCCUPANCY_LEDGER.md)**"
            "（已知被占图：27 类问题类 + 墓碑 GPU·h）——本工作区实证：第 2 轮**重复发现了 4 天前已判死的深度轴**，"
            "根因就是没先查库存；① 用**问题类**描述候选（不是某个引擎的功能名）；"
            "② 跑 `python pipeline/solved_scan.py --keywords … --repos …` 生成待查清单；")


def wire(check=False):
    g = json.load(io.open(GATES, encoding="utf-8"), object_pairs_hook=collections.OrderedDict)
    g.pop("_by_id", None)
    s3b = [s for s in g["stages"] if s["id"] == "S3b"][0]
    changed = []
    if ITEM not in s3b["checklist"]:
        s3b["checklist"].insert(0, ITEM)
        changed.append("gates.json: S3b checklist 插入第 0 步")
    if "OCCUPANCY_LEDGER" not in s3b.get("evidence_from_repo", ""):
        s3b["evidence_from_repo"] = (s3b.get("evidence_from_repo", "") + NOTE)
        changed.append("gates.json: S3b evidence_from_repo 追加")
    g["version"] = "1.1.1"
    g["updated"] = "2026-09-13"
    if check:
        print("checks:", changed or "已是最新")
        return 0
    io.open(GATES, "w", encoding="utf-8").write(json.dumps(g, ensure_ascii=False, indent=2) + "\n")
    s = io.open(PIPE, encoding="utf-8").read()
    if PIPE_NEW not in s:
        assert PIPE_OLD in s, "PIPELINE.md 锚点未命中"
        io.open(PIPE, "w", encoding="utf-8").write(s.replace(PIPE_OLD, PIPE_NEW))
        changed.append("PIPELINE.md: S3b 动作插入第 0 步")
    print("wired:", changed or "已是最新（幂等）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    return wire(a.check)


if __name__ == "__main__":
    sys.exit(main())
