#!/usr/bin/env python3
"""S10: print a compact digest of fetched gh pages: title, state, and all comments after #1."""
import glob, re, sys

for path in sorted(sys.argv[1:] or glob.glob("/tmp/s10_fetch/*.txt")):
    t = open(path, encoding="utf-8", errors="replace").read()
    title = re.search(r"^TITLE: (.*)$", t, re.M)
    sr = re.search(r"^STATE_REASON: (.*)$", t, re.M)
    state = set(re.findall(r"^STATE-TOKEN: (.*)$", t, re.M))
    labs = re.search(r"^LABELS: (.*)$", t, re.M)
    ca = re.search(r"^CLOSED_AT: (.*)$", t, re.M)
    ma = re.search(r"^MERGED_AT: (.*)$", t, re.M)
    parts = re.split(r"^--- COMMENT \d+ ---$", t, flags=re.M)
    print("=" * 100)
    print(f"FILE {path}")
    print("TITLE:", (title.group(1) if title else "?")[:200])
    print("STATE:", ",".join(sorted(state)) or "?", "| REASON:", sr.group(1) if sr else "-",
          "| CLOSED:", ca.group(1)[:60] if ca else "-", "| MERGED:", ma.group(1)[:60] if ma else "-")
    print("LABELS:", labs.group(1) if labs else "-")
    for i, body in enumerate(parts[1:], 1):
        if i == 1:
            continue
        b = body.split("=== PAGE TEXT")[0].strip()
        if not b:
            continue
        print(f"\n  --- C{i} --- {b[:2600]}")
    print()
