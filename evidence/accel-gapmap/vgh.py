#!/usr/bin/env python3
"""Dump authoritative state for every GitHub URL referenced by the S files."""
import re, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vstate import report

HERE = os.path.dirname(os.path.abspath(__file__))
urls = []
for fn in ("S5.md", "S6.md", "S7.md", "S8.md"):
    txt = open(os.path.join(HERE, fn), encoding="utf-8").read()
    for m in re.finditer(r"https://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/(?:pull|issues)/\d+", txt):
        u = m.group(0)
        if u not in urls:
            urls.append(u)
print("TOTAL GH URLS:", len(urls), flush=True)
for u in urls:
    try:
        report(u)
    except Exception as e:
        print("ERR", u, repr(e)[:200], flush=True)
