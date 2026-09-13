#!/usr/bin/env python3
"""判据闸门视图 —— 打印**某一道闸门**的完整判据、当前勾选状态与阻塞项。

用法：
  python pipeline/gate.py S3b --dir candidates/<slug>
  python pipeline/gate.py S3b                        # 只看判据（不指定档案）
  python pipeline/gate.py --list                     # 所有闸门一行一条
  python pipeline/gate.py --selftest

退出码：0 = 该闸门无阻塞；1 = 有阻塞或未指定档案；2 = 未知闸门 id。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402


def render(sid, dossier=None):
    g = lib.load_gates()
    st = g["_by_id"].get(sid)
    if st is None:
        print(f"未知闸门：{sid}；可选：{', '.join(s['id'] for s in g['stages'])}")
        return 2
    print(f"=== {st['id']} {st['name']} ===")
    print(f"产物：{st['artifact']}   成本：{st.get('cost', '—')}")
    print(f"目标：{st['goal']}\n")
    print("必填小节：")
    for s in st["sections"]:
        print(f"  · {s}")
    print("\n判据清单：")
    checked = {}
    path = os.path.join(dossier, st["artifact"]) if dossier else None
    if path and os.path.exists(path):
        _, secs, boxes = lib.parse_artifact(path)
        for item in st["checklist"]:
            v = lib.match_box(item, boxes)
            checked[item] = ("[x]" if v else "[ ]") if v is not None else "[缺]"
    for item in st["checklist"]:
        mark = checked.get(item, "[?]")
        print(f"  {mark} {item}")
    if path and not os.path.exists(path):
        print(f"\n⚠️ 档案里还没有 {st['artifact']}（用 pipeline/new_candidate.py 生成骨架）")
    print(f"\n杀出口：{st.get('kill', '无')}")
    print(f"依据出处：{st.get('evidence_from_repo', '—')}")
    if path and os.path.exists(path):
        todo = [k for k, v in checked.items() if v != "[x]"]
        if todo:
            print(f"\n❌ 该闸门未通过：{len(todo)} 条未满足")
            return 1
        print("\n✅ 该闸门已满足")
    return 0


def selftest():
    g = lib.load_gates()
    ids = [s["id"] for s in g["stages"]]
    assert "S3b" in ids, ids
    order = [lib.stage_index(i) for i in ids]
    assert order == sorted(order), order
    assert lib.stage_index("S3") < lib.stage_index("S3b") < lib.stage_index("S4")
    assert lib.stage_index("S0") == 0 and lib.stage_index("S8") == 8
    st = g["_by_id"]["S3b"]
    assert len(st["checklist"]) >= 6 and st["artifact"] == "35_solved_check.md"
    assert render("S3b") in (0, 1)
    assert render("S99") == 2
    print("selftest ✔ S3b 存在/排序 S3<S3b<S4/清单≥6/render 三种返回码")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?")
    ap.add_argument("--dir")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    g = lib.load_gates()
    if a.list or not a.stage:
        for s in g["stages"]:
            print(f"{s['id']:<5}{s['name']:<34}{s['artifact']:<22}{s.get('cost', '—')}")
        return 0
    return render(a.stage, a.dir)


if __name__ == "__main__":
    sys.exit(main())
