#!/usr/bin/env python3
"""Read back every arXiv ID cited by the S files and print its title."""
import re, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import arxiv

HERE = os.path.dirname(os.path.abspath(__file__))
ids = []
for fn in ("S5.md", "S6.md", "S7.md", "S8.md"):
    txt = open(os.path.join(HERE, fn), encoding="utf-8").read()
    for m in re.finditer(r"(?:arXiv:|arxiv\.org/abs/)(\d{4}\.\d{4,5})", txt):
        if m.group(1) not in ids:
            ids.append(m.group(1))
print("TOTAL ARXIV IDS:", len(ids), flush=True)
for i in ids:
    try:
        t = arxiv(i)
        title = ""
        for line in t.splitlines():
            if line.startswith("TITLE:"):
                title = line[6:].strip()
        print(f"{i}\t{title}", flush=True)
    except Exception as e:
        print(f"{i}\tERR {e!r}", flush=True)
