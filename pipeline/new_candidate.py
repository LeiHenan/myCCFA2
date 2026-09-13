#!/usr/bin/env python3
"""选题 pipeline 脚手架 —— 建一个候选方向的档案目录，并生成各阶段的填写骨架。

用法：
  python pipeline/new_candidate.py --slug kv-reserve-accounting --title "KV 预留记账"
  python pipeline/new_candidate.py --slug demo --title demo --stages S0,S1   # 只建前两阶段
  python pipeline/new_candidate.py --selftest

产物：`candidates/<slug>/00_intake.md … 80_close.md`（**内容全部来自 gates.json 的骨架**，所以
清单与小节永远和校验器一致），外加 `candidates/README.md` 的候选登记行提示。
已存在的文件不会被覆盖（避免抹掉已写内容）。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

REPO = os.path.dirname(lib.HERE)
CAND = os.path.join(REPO, "candidates")


def _shown(path):
    """仓库内用相对路径，仓库外保留绝对路径（避免 `../../..` 噪音）。"""
    rel = os.path.relpath(path, REPO)
    return rel if not rel.startswith("..") else path


def build(slug, title, stages=None, dest=None):
    gates = lib.load_gates()
    d = dest or os.path.join(CAND, slug)
    os.makedirs(d, exist_ok=True)
    made, kept = [], []
    for st in gates["stages"]:
        if stages and st["id"] not in stages:
            continue
        path = os.path.join(d, st["artifact"])
        if os.path.exists(path):
            kept.append(st["artifact"])
            continue
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(lib.skeleton(st, slug=slug, title=title))
        made.append(st["artifact"])
    return d, made, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--title", default="")
    ap.add_argument("--stages", help="逗号分隔，如 S0,S1；默认全建")
    ap.add_argument("--dest")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d, made, kept = build("demo", "演示", dest=os.path.join(td, "demo"))
            gates = lib.load_gates()
            assert len(made) == len(gates["stages"]), made
            # 骨架必须能通过"小节齐全 + 清单齐全（全未勾）"的结构检查
            for st in gates["stages"]:
                text, secs, boxes = lib.parse_artifact(os.path.join(d, st["artifact"]))
                missing = [s for s in st["sections"] if s not in secs]
                assert not missing, (st["id"], missing)
                for item in st["checklist"]:
                    assert lib.match_box(item, boxes) is False, (st["id"], item)
            # 幂等：再跑一次不覆盖
            _, made2, kept2 = build("demo", "演示", dest=d)
            assert not made2 and len(kept2) == len(gates["stages"]), (made2, kept2)
            # 子集
            d2, made3, _ = build("demo2", "演示2", stages={"S0", "S1"}, dest=os.path.join(td, "demo2"))
            assert len(made3) == 2, made3
            print("selftest ✔ 全阶段骨架/小节齐全/清单可匹配/幂等不覆盖/子集选择")
        return 0
    if not a.slug:
        ap.error("需要 --slug，或 --selftest")
    stages = {s.strip() for s in a.stages.split(",")} if a.stages else None
    d, made, kept = build(a.slug, a.title or a.slug, stages, a.dest)
    print(f"档案目录：{d}")
    print(f"已生成 {len(made)} 个：{', '.join(made) if made else '（无）'}")
    if kept:
        print(f"已存在未覆盖：{', '.join(kept)}")
    print("\n下一步：")
    print(f"  1) 填 {os.path.join(d, '00_intake.md')}（约束必须量化）")
    print(f"  2) python pipeline/check.py --dir {_shown(d)} --through S0")
    print("  3) 按 PIPELINE.md 的『Agent 操作协议』逐阶段推进；每阶段结束记一行 notes/decision_log.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
