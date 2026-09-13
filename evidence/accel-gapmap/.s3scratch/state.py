#!/usr/bin/env python3
import re, sys, os
DATE = re.compile(r'^[A-Z][a-z]{2} \d{1,2}, \d{4}$')
for path in sys.argv[1:]:
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="replace")]
    t = "\n".join(lines)
    title = (re.search(r'^TITLE: (.*)$', t, re.M) or [None,'?'])[1] or '?'
    num = (re.search(r'#(\d+)', title) or [None,'?'])[1]
    dates = []
    merged_any = False
    for i, l in enumerate(lines):
        if re.search(r'merged \d+ commits? into', l):
            merged_any = True
            for j in range(i, min(len(lines), i+6)):
                if DATE.match(lines[j]):
                    dates.append(lines[j]); break
        if re.search(r'closed this as (not planned|completed)', l, re.I):
            for j in range(max(0,i-6), min(len(lines), i+6)):
                if DATE.match(lines[j]):
                    dates.append("CLOSE:"+lines[j]); break
    st = re.findall(r'STATE-TOKEN: (merged|closed|open|draft)', t)
    print(f"#{num:<7} merged={'YES' if merged_any else 'NO ':3}  dates={sorted(set(dates))[:3]}")
    print(f"    {title[:150]}")
