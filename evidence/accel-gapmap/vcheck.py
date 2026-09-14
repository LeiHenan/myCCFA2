#!/usr/bin/env python3
"""Adversarial per-file quote/state checker for the accel gap map.

  python3 vcheck.py S5.md            # check every quote field of every block
  python3 vcheck.py S5.md A5.1 C5.1  # only those blocks

For each block it collects the URL(s) and every field whose name looks like a quote
field, then runs the project's own mechanical checker (verify.check) on each.
Continuation lines are folded into the preceding field (verify.py's parser drops them).
"""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import check

BLOCK = re.compile(r"^###\s+([ABCDE][0-9.]*?)\s*\|\s*(CLOSED|OPEN|ABANDONED|HARDWARE-RULED-OUT|NEVER-DISCUSSED)\s*(.*)$")
FIELD = re.compile(r"^([A-Z][A-Z0-9 _\(\)/\.\->\+]*?):\s?(.*)$")
QUOTEY = re.compile(r"^(QUOTE|QUOTE ?[0-9]|ASKING QUOTE|STATED REASON|HARDWARE QUOTE|DETAIL|CONTEXT|NOTE|"
                    r"SECOND QUOTE|THIRD QUOTE|FOURTH QUOTE|FIFTH QUOTE|SECOND HARDWARE QUOTE|"
                    r"THIRD HARDWARE QUOTE|FOURTH HARDWARE QUOTE|FIFTH HARDWARE QUOTE|MEASUREMENT|"
                    r"SOURCE FACT|RESIDUAL|CORROBORATION|EXTERNAL CORROBORATION|RIG CONSEQUENCE)$")


def parse_blocks(path):
    blocks, cur = [], None
    lastfield = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = BLOCK.match(line)
        if m:
            if cur:
                blocks.append(cur)
            cur = {"id": m.group(1), "section": m.group(2), "fields": {}, "order": []}
            lastfield = None
            continue
        if cur is None:
            continue
        if line.startswith("#") and not line.startswith("###"):
            lastfield = None
            continue
        f = FIELD.match(line)
        if f and not line.startswith("###"):
            k = f.group(1).strip()
            cur["fields"][k] = f.group(2).strip()
            cur["order"].append(k)
            lastfield = k
        elif line.strip() and lastfield:
            cur["fields"][lastfield] = (cur["fields"][lastfield] + " " + line.strip()).strip()
    if cur:
        blocks.append(cur)
    return blocks


def urls_of(b):
    raw = b["fields"].get("URL", "")
    return [u.strip() for u in re.split(r"\s*;\s*", raw) if u.strip().startswith("http")]


def main():
    path = sys.argv[1]
    want = set(sys.argv[2:])
    blocks = parse_blocks(path)
    for b in blocks:
        if want and b["id"] not in want:
            continue
        us = urls_of(b)
        print(f"\n##### {b['id']} | {b['section']} | urls={len(us)}")
        for u in us:
            print("   URL", u)
        for k in b["order"]:
            if not QUOTEY.match(k):
                continue
            q = b["fields"][k]
            if len(q) < 20:
                print(f"   [{k}] SHORT/EMPTY: {q!r}")
                continue
            results = []
            for u in us:
                try:
                    r = check(u, q)
                except Exception as e:
                    r = {"verdict": "ERR", "err": str(e)[:120]}
                results.append((u, r))
            best = max(results, key=lambda t: (t[1].get("verdict") == "MATCH",
                                               t[1].get("verdict") == "MATCH-FRAGMENTS",
                                               t[1].get("ratio", 0))) if results else (None, {})
            v = best[1]
            flag = v.get("verdict")
            print(f"   [{k}] {flag} ratio={v.get('ratio')} variant={v.get('variant')} "
                  f"n_frag={v.get('n_fragments')} url={best[0]}")
            if flag in ("NEAR", "MISMATCH", "ERR", "EMPTY"):
                print(f"        missing_tail: {v.get('missing_tail','')!r}")
                print(f"        per-url: " + " | ".join(f"{u.split('/')[-1]}:{r.get('verdict')}:{r.get('ratio')}"
                                                          for u, r in results))
    print("\nTOTAL BLOCKS:", len([b for b in blocks if not want or b['id'] in want]))


if __name__ == "__main__":
    main()
