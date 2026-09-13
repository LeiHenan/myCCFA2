#!/usr/bin/env python3
"""Batch-grab many arXiv ids: write per-id sentence hits into .s13_scratch/hits/<id>.txt

Usage: python3 d5batch.py id1 id2 ...   (or reads ids from stdin if no args)
"""
import os, re, sys, json
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/.s13_scratch")
import fetch as F
from d5grab import PATS, sentences, get_html

OUT = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/.s13_scratch/hits"
os.makedirs(OUT, exist_ok=True)


def one(aid):
    dest = os.path.join(OUT, aid + ".txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return aid, "cached"
    url, h = get_html(aid)
    if not h:
        open(dest, "w").write("NO_HTML\n")
        return aid, "NO_HTML"
    txt = F.strip_html(h)
    sents = [s for s in sentences(txt) if PATS.search(s) and 30 < len(s) < 800]
    # also grab abstract region sentences for headline context
    head = txt[:3000]
    lines = ["HTML_URL: " + url, "TEXT_LEN: %d" % len(txt), "=== HEAD ===", head, "=== HITS ==="]
    for i, s in enumerate(sents, 1):
        lines.append(f"[{i}] {s}")
    open(dest, "w", encoding="utf-8").write("\n".join(lines))
    return aid, f"{len(sents)} hits"


def main():
    ids = sys.argv[1:] or [l.strip() for l in sys.stdin if l.strip()]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for aid, st in ex.map(one, ids):
            print(aid, st, flush=True)


if __name__ == "__main__":
    main()
