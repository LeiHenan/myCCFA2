#!/usr/bin/env python3
"""Print context windows around broad GPU-count patterns in an arXiv paper's HTML text.

Usage: python3 d5scan.py <arxiv_id> [win]
"""
import re, sys, os
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/.s13_scratch")
import fetch as F
from d5grab import get_html

BROAD = re.compile(
    r"(single[\s\-]?gpu|single[\s\-]?device|single[\s\-]?card|single[\s\-]?node|"
    r"\bone gpu\b|\b1 gpu\b|\b1×|\b1 ?x ?(a100|h100|h800|l40s|a800|v100|4090|h20)|"
    r"one (a100|h100|h800|l40s|a800|v100|rtx|nvidia)|"
    r"tp\s*=\s*1|tensor[\s\-]?parallel[\s\-]?(size|degree|ism)?\s*(=|of|is|to)?\s*1\b|"
    r"on (a|one) single (gpu|device|card)|per[\s\-]?gpu|"
    r"\b[24816]\s*(x|×)\s*(a100|h100|h800|l40s|a800|v100|4090|h20|rtx)|"
    r"multi[\s\-]?gpu|multi[\s\-]?node|8 gpus|four gpus|two gpus|2 gpus|4 gpus)", re.I)


def main():
    aid = sys.argv[1]
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 260
    url, h = get_html(aid)
    if not h:
        print("NO_HTML", aid)
        return
    txt = re.sub(r"\s+", " ", F.strip_html(h))
    print("### HTML:", url, "len:", len(txt))
    seen = set()
    n = 0
    for m in BROAD.finditer(txt):
        a = max(0, m.start() - win)
        b = min(len(txt), m.end() + win)
        seg = txt[a:b]
        key = seg[:80]
        if key in seen:
            continue
        seen.add(key)
        n += 1
        print(f"--- [{n}] ...{seg}...")


if __name__ == "__main__":
    main()
