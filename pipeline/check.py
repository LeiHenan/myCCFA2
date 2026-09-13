#!/usr/bin/env python3
"""选题 pipeline 校验器 —— 在"声称某阶段完成"之前，用机器把清单核一遍。

用法：
  python pipeline/check.py --list                       # 只打印判据表
  python pipeline/check.py --dir candidates/<slug>      # 校验档案（未开始的阶段不算失败）
  python pipeline/check.py --dir candidates/<slug> --through S5   # 声称 S0–S5 已完成 ⇒ 缺项即失败
  python pipeline/check.py --dir candidates/<slug> --repo .        # 额外核对 prereg 提升与决策日志
  python pipeline/check.py --selftest

退出码：0 = 无阻塞项；1 = 有阻塞项（逐条列出）。**不要用"看起来差不多"绕过它。**
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402


def check(dossier, gates, through=None, repo=None, verbose=True):
    cfg = gates["global_invariants"]
    rows, blocking = [], []
    for st in gates["stages"]:
        sid, idx = st["id"], lib.stage_index(st["id"])
        path = os.path.join(dossier, st["artifact"])
        claimed = through is not None and idx <= lib.stage_index(through)
        if not os.path.exists(path):
            state = "未开始" + ("（已声称完成 ⇒ 失败）" if claimed else "")
            if claimed:
                blocking.append(f"{sid} 声称完成但缺 {st['artifact']}")
            rows.append((sid, st["name"], state, "—", "—"))
            continue
        text, sections, boxes = lib.parse_artifact(path)
        missing = [s for s in st["sections"] if s not in sections]
        todo, absent, matched = [], [], set()
        for item in st["checklist"]:
            key = " ".join(item.split())
            hit = None
            for txt, checked in boxes:
                if " ".join(txt.split()).startswith(key) and txt not in matched:
                    hit = (txt, checked)
                    break
            if hit is None:
                absent.append(item)
            else:
                matched.add(hit[0])
                if not hit[1]:
                    todo.append(item)
        # 散落项：不在判据清单里、且未勾选 —— 声称完成时说明还留着没做完的事
        stray = [txt for txt, checked in boxes if not checked and txt not in matched]
        if missing:
            blocking.append(f"{sid} 缺小节：{', '.join(missing)}")
        if absent:
            blocking.append(f"{sid} 清单项未出现在产物里（{len(absent)} 条）：{absent[0][:36]}…")
        state = "完成" if not (missing or todo or absent or stray) else (
            f"进行中（{len(todo) + len(stray)} 项未勾）" if not (missing or absent) else "不合格")
        if claimed and state != "完成":
            blocking.append(f"{sid} 声称完成但状态为『{state}』")
        if claimed and stray:
            blocking.append(f"{sid} 声称完成但留有 {len(stray)} 个未勾选散落项：{stray[0][:36]}…")
        rows.append((sid, st["name"], state,
                     "—" if not missing else str(len(missing)),
                     "—" if not (todo or absent or stray) else f"{len(todo)}+{len(absent)}+{len(stray)}"))
        # 全局禁词
        blk, warn = lib.scan_banned(text, cfg)
        for ln, w, snippet in blk:
            blocking.append(f"{sid} {st['artifact']}:{ln} 出现『{w}』（需补完或写『豁免：<理由>』）— {snippet}")
        for ln, w, snippet in warn:
            rows.append(("", "", f"⚠️ 措辞过虚（{w}）", f"{st['artifact']}:{ln}", snippet[:30]))

    # 痛点证据强度（只有在"已写了痛点"或"声称 S1 完成"时才作为判据；空骨架不算失败）
    p = os.path.join(dossier, "10_pains.md")
    if os.path.exists(p):
        text, _, _ = lib.parse_artifact(p)
        pains = lib.pain_blocks(text)
        s1_claimed = through is not None and lib.stage_index(through) >= 1
        if not pains:
            if s1_claimed:
                blocking.append("S1 声称完成但没有痛点条目（用 `### P1 …` 形式逐条编号）")
            else:
                rows.append(("", "", "痛点尚未录入（骨架为空）", "—", "—"))
        for title, block in pains:
            ok, why = lib.evidence_ok(block)
            if not ok:
                blocking.append(f"S1 痛点『{title[:30]}』证据不合格：{why}")
        if pains:
            rows.append(("", "", f"痛点条目 {len(pains)} 条（证据已逐条核）", "—", "—"))

    # 仓库级不变量
    if repo:
        slug = os.path.basename(os.path.normpath(dossier))
        if cfg.get("require_prereg_promotion") and os.path.exists(os.path.join(dossier, "50_prereg.md")):
            reg = os.path.join(repo, "notes", "prereg", f"{slug}.md")
            if not os.path.exists(reg):
                blocking.append(f"S5 已写 prereg 但未提升为 notes/prereg/{slug}.md（预登记的效力来自仓库级冻结副本）")
        if cfg.get("require_decision_log_row"):
            dl = os.path.join(repo, "notes", "decision_log.md")
            if os.path.exists(dl) and slug not in open(dl, encoding="utf-8").read():
                rows.append(("", "", f"⚠️ 决策日志没有提到 `{slug}`", "notes/decision_log.md", "结题时补"))
    if verbose:
        print(f"{'阶段':<5}{'名称':<26}{'状态':<22}{'缺小节':<8}{'未完成'}")
        print("-" * 84)
        for sid, name, state, m, t in rows:
            print(f"{sid:<5}{name:<26}{state:<22}{m:<8}{t}")
        print()
    return rows, blocking


def selftest():
    import json
    import tempfile
    gates = lib.load_gates()
    with tempfile.TemporaryDirectory() as td:
        dossier = os.path.join(td, "demo")
        os.makedirs(os.path.join(dossier))
        # 用脚手架生成 S0–S4 并全部勾选 + 填证据
        for st in gates["stages"][:5]:
            txt = lib.skeleton(st, slug="demo")
            txt = txt.replace("- [ ]", "- [x]")
            if st["id"] == "S1":
                txt += ("\n### P1 例子痛点\n- 现象：demo\n- 谁在疼：demo\n- 量级：L2\n"
                        "- 证据：`python -c 'print(1)'`\n- 反证：无\n")
            open(os.path.join(dossier, st["artifact"]), "w", encoding="utf-8").write(txt)

        _, blk = check(dossier, gates, through="S4", verbose=False)
        assert not blk, f"应通过，实际阻塞：{blk}"

        # 1) 取消勾选 → 失败
        p = os.path.join(dossier, "20_screen.md")
        open(p, "a", encoding="utf-8").write("\n- [ ] 新增未完成项\n")
        _, blk = check(dossier, gates, through="S4", verbose=False)
        assert any("S2" in b for b in blk), blk
        open(p, "w", encoding="utf-8").write(
            lib.skeleton(gates["_by_id"]["S2"], slug="demo").replace("- [ ]", "- [x]"))

        # 2) 缺小节 → 失败
        p = os.path.join(dossier, "30_occupancy.md")
        open(p, "w", encoding="utf-8").write("- [x] " + gates["_by_id"]["S3"]["checklist"][0])
        _, blk = check(dossier, gates, through="S4", verbose=False)
        assert any("缺小节" in b for b in blk), blk

        # 3) 禁词 → 失败；加豁免 → 通过
        p = os.path.join(dossier, "30_occupancy.md")
        t = lib.skeleton(gates["_by_id"]["S3"], slug="demo").replace("- [ ]", "- [x]") + "\nTODO 补实验\n"
        open(p, "w", encoding="utf-8").write(t)
        _, blk = check(dossier, gates, through="S4", verbose=False)
        assert any("TODO" in b for b in blk), blk
        open(p, "w", encoding="utf-8").write(t.replace("TODO 补实验", "TODO 补实验（豁免：已另开 issue 跟踪）"))
        _, blk = check(dossier, gates, through="S4", verbose=False)
        assert not any("TODO" in b for b in blk), blk

        # 4) 证据不合格 → 失败
        p = os.path.join(dossier, "10_pains.md")
        open(p, "w", encoding="utf-8").write(
            lib.skeleton(gates["_by_id"]["S1"], slug="demo").replace("- [ ]", "- [x]")
            + "\n### P1 空证据\n- 现象：x\n- 谁在疼：y\n- 量级：L2\n- 证据：\n- 反证：z\n")
        _, blk = check(dossier, gates, through="S4", verbose=False)
        assert any("证据" in b for b in blk), blk

        # 5) prereg 未提升 → 失败
        open(os.path.join(dossier, "50_prereg.md"), "w", encoding="utf-8").write(
            lib.skeleton(gates["_by_id"]["S5"], slug="demo").replace("- [ ]", "- [x]"))
        repo = os.path.join(td, "repo")
        os.makedirs(os.path.join(repo, "notes", "prereg"))
        open(os.path.join(repo, "notes", "decision_log.md"), "w", encoding="utf-8").write("| 1 | demo |\n")
        _, blk = check(dossier, gates, through="S5", repo=repo, verbose=False)
        assert any("提升" in b for b in blk), blk
        open(os.path.join(repo, "notes", "prereg", "demo.md"), "w", encoding="utf-8").write("x")
        _, blk = check(dossier, gates, through="S5", repo=repo, verbose=False)
        assert not any("提升" in b for b in blk), blk
        print("selftest ✔ 骨架生成/小节比对/勾选比对/禁词与豁免/证据核查/prereg 提升/决策日志/退出码路径")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--repo")
    ap.add_argument("--through")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    gates = lib.load_gates()
    if a.selftest:
        selftest()
        return 0
    if a.list:
        print(f"gates.json v{gates['version']}（{gates['updated']}）\n")
        for st in gates["stages"]:
            print(f"{st['id']} {st['name']}  → {st['artifact']}  [{st.get('cost','—')}]")
            print(f"    目标：{st['goal']}")
            print(f"    杀出口：{st.get('kill','无')}")
        return 0
    if not a.dir:
        ap.error("需要 --dir，或 --list / --selftest")
    if not os.path.isdir(a.dir):
        print(f"目录不存在：{a.dir}")
        return 1
    _, blocking = check(a.dir, gates, a.through, a.repo)
    if blocking:
        print(f"❌ 阻塞项 {len(blocking)} 条：")
        for b in blocking:
            print(f"  - {b}")
        return 1
    print("✅ 无阻塞项" + (f"（已核到 {a.through}）" if a.through else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
