#!/usr/bin/env python3
"""Per-file tally of primary-quote re-location for the verify reports.

Primary quote field per section:
  CLOSED -> QUOTE | OPEN -> ASKING QUOTE | ABANDONED -> STATED REASON |
  HARDWARE-RULED-OUT -> HARDWARE QUOTE
Verdicts: MATCH (continuous run), FRAG (all fragments found; composite),
CODE (all words present after deleting source syntax markers), FAIL.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import check
from vcode import check as codecheck, check_raw as rawcheck
from vcheck import parse_blocks, urls_of
from vfrag import fragments

PRIMARY = {"CLOSED": "QUOTE", "OPEN": "ASKING QUOTE", "ABANDONED": "STATED REASON",
           "HARDWARE-RULED-OUT": "HARDWARE QUOTE", "NEVER-DISCUSSED": "GAP"}


def judge(urls, frag):
    for u in urls:
        try:
            r = check(u, frag)
        except Exception:
            r = {"verdict": "ERR"}
        if r.get("verdict") in ("MATCH", "MATCH-FRAGMENTS"):
            return r["verdict"]
    for u in urls:
        try:
            if rawcheck(u, frag).get("verdict") == "MATCH-RAW":
                return "RAW"
        except Exception:
            pass
    for u in urls:
        try:
            if codecheck(u, frag).get("verdict") == "MATCH-CODE-NORMALISED":
                return "CODE"
        except Exception:
            pass
    return "FAIL"


def main():
    grand = {}
    for fn in ("S5.md", "S6.md", "S7.md", "S8.md"):
        if len(sys.argv) > 1 and fn != sys.argv[1]:
            continue
        tally = {"checked": 0, "MATCH": 0, "FRAG": 0, "RAW": 0, "CODE": 0, "FAIL_NOQUOTE": 0}
        fails = []
        for b in parse_blocks(fn):
            tally["checked"] += 1
            field = PRIMARY.get(b["section"], "QUOTE")
            q = b["fields"].get(field, "")
            if not q.strip():
                tally["FAIL_NOQUOTE"] += 1
                fails.append((b["id"], field, "NO-QUOTE-FIELD"))
                continue
            us = urls_of(b)
            frags = fragments(q)
            if not frags:
                frags = [q]
            verdicts = [judge(us, f) for f in frags]
            if all(v == "MATCH" for v in verdicts):
                v = "MATCH"
            elif all(v in ("MATCH", "MATCH-FRAGMENTS", "RAW", "CODE") for v in verdicts):
                v = "FRAG" if "MATCH-FRAGMENTS" in verdicts else (
                    "CODE" if "CODE" in verdicts else ("RAW" if "RAW" in verdicts else "FRAG"))
            else:
                v = "FAIL"
            tally[v] = tally.get(v, 0) + 1
            if v == "FAIL":
                fails.append((b["id"], field, [x for x, vv in zip([f[:60] for f in frags], verdicts)
                                               if vv == "FAIL"][:1]))
        relocated = tally["MATCH"] + tally["FRAG"] + tally["RAW"] + tally["CODE"]
        print(f"### {fn}: checked={tally['checked']} relocated={relocated} "
              f"(MATCH={tally['MATCH']} FRAG={tally['FRAG']} RAW={tally['RAW']} CODE={tally['CODE']}) "
              f"FAIL={tally.get('FAIL',0)} NOQUOTE={tally['FAIL_NOQUOTE']}")
        for f in fails:
            print("   FAIL:", f[0], f[1], f[2])
        grand[fn] = tally
    print(json.dumps(grand, indent=1))


if __name__ == "__main__":
    main()
