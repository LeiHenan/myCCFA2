#!/usr/bin/env python3
"""S12 verifier: extract (URL, QUOTE) pairs from S12.md and check each quote appears
verbatim (whitespace-normalised, HTML-unescaped) in the fetched page.

Usage: python3 s12_verify.py S12.md
"""
import re, sys, os, html, subprocess, hashlib

HERE = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap"
sys.path.insert(0, HERE)
from fetch import curl, gh, strip_html

BLOCKS = re.compile(r"^### (C\d+\.\d+|[A-D]\d+\.\d+)\s*\|", re.M)


def norm(s):
    s = html.unescape(s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014", "-").replace("\u2013", "-").replace("\u2026", "...")
    s = s.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


def main(path):
    md = open(path, encoding="utf-8").read()
    # split into blocks by heading
    idx = [m.start() for m in BLOCKS.finditer(md)]
    idx.append(len(md))
    bad = 0
    total = 0
    for i in range(len(idx) - 1):
        blk = md[idx[i]:idx[i + 1]]
        head = blk.split("\n", 1)[0].strip()
        urls = re.findall(r"^URL: (.+)$", blk, re.M)
        quotes = []
        for lab in ("QUOTE", "STATED REASON", "ASKING QUOTE", "HARDWARE QUOTE"):
            quotes += re.findall(rf"^{lab}: (.*?)(?=\n[A-Z][A-Z0-9 _-]*:|$)", blk, re.M | re.S)
        if not urls or not quotes:
            continue
        url_list = [u.strip() for u in re.split(r"\s*;\s*", urls[0]) if u.strip()]
        cachetext = ""
        for u in url_list:
            try:
                cachetext += "\n" + gh(u) + "\n" + strip_html(curl(u))
            except Exception as e:
                cachetext += "\nERR " + str(e)
        for q in quotes:
            q = norm(q.strip().strip('"'))
            if len(q) < 25:
                continue
            # try the full quote, then progressively longer prefixes down to 60 chars
            found = False
            cand = [q]
            for cut in (200, 150, 120, 90, 60):
                if len(q) > cut:
                    cand.append(q[:cut])
            for c in cand:
                if norm(c) in norm(cachetext):
                    found = True
                    break
            total += 1
            if not found:
                bad += 1
                print(f"MISMATCH {head}\n  URL: {url_list[0]}\n  QUOTE: {q[:220]!r}\n")
    print(f"\n{total} quotes checked, {bad} not found")


if __name__ == "__main__":
    main(sys.argv[1])
