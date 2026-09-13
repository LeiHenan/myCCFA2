#!/usr/bin/env python3
"""Readback a list of arXiv ids with fetch.py arxiv; cache per-id output.

Usage: python3 d5rb.py id1 id2 ...   (writes .s13_scratch/rb/<id>.txt)
"""
import os, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
import fetch as F

OUT = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/.s13_scratch/rb"
os.makedirs(OUT, exist_ok=True)


def one(aid):
    dest = os.path.join(OUT, aid + ".txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        txt = open(dest, encoding="utf-8").read()
    else:
        txt = F.arxiv(aid)
        open(dest, "w", encoding="utf-8").write(txt)
    t = re.search(r"^TITLE: (.*)$", txt, re.M)
    ok = bool(t) and "DID NOT PARSE" not in txt
    return aid, ("OK  " + (t.group(1)[:95] if t else "?")) if ok else ("FAIL " + txt[:120].replace("\n", " "))


def main():
    ids = sys.argv[1:] or [l.strip() for l in sys.stdin if l.strip()]
    with ThreadPoolExecutor(max_workers=5) as ex:
        for aid, st in ex.map(one, ids):
            print(aid, "|", st, flush=True)


if __name__ == "__main__":
    main()
