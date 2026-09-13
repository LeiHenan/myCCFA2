#!/usr/bin/env python3
"""Grab an arXiv paper's full text HTML (ar5iv/arxiv native) via fetch.py's curl
and print sentences mentioning GPU counts.

Usage:
  python3 d5grab.py <arxiv_id>            # print matching sentences + header
  python3 d5grab.py <arxiv_id> --raw      # print whole text
"""
import os, re, subprocess, sys, hashlib, html as htmlmod

CACHE = "/tmp/accelsrc"
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
import fetch as F  # noqa

PATS = re.compile(
    r"(single[\s\-]?GPU|single GPU|one GPU|a single A100|single A100|single H100|single H800|"
    r"1[\s×xX*]*\s*(A100|H100|H800|L40S|A800|RTX|V100|GPU)|TP\s*=\s*1|tensor[\s\-]parallel[\s\-]?(size|degree)?\s*=\s*1|"
    r"single[\s\-]device|single card|one device)", re.I)


def get_html(aid):
    for suffix in ["v1", "v2", "v3", "v4", ""]:
        url = f"https://arxiv.org/html/{aid}{suffix}"
        h = F.curl(url)
        if h and "<html" in h.lower() and len(h) > 5000 and "not found" not in h[:2000].lower():
            return url, h
    return None, ""


def sentences(text):
    # split on sentence enders followed by space+capital or newline
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def main():
    aid = sys.argv[1]
    url, h = get_html(aid)
    if not h:
        print("NO_HTML for", aid)
        return
    txt = F.strip_html(h)
    if "--raw" in sys.argv:
        print(txt)
        return
    print("HTML_URL:", url)
    print("TEXT_LEN:", len(txt))
    hits = 0
    for s in sentences(txt):
        if PATS.search(s) and 40 < len(s) < 700:
            hits += 1
            print(f"  [{hits}] {s}")
    if not hits:
        print("  (no GPU-count sentence found)")


if __name__ == "__main__":
    main()
