#!/usr/bin/env python3
"""S3: spaced arXiv search with retry-on-empty. Usage: python3 s3_q.py q1 q2 ..."""
import re, sys, html, time, urllib.parse
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl

def search(q, size=50, sort="relevance", tries=6):
    url = ("https://arxiv.org/search/?searchtype=all&query=" + urllib.parse.quote(q) +
           f"&size={size}&sortBy={sort}")
    for k in range(tries):
        h = curl(url)
        if len(h) > 20000:
            break
        time.sleep(20)
    out = []
    for m in re.finditer(r'<li class="arxiv-result">(.*?)</li>', h, re.S):
        blk = m.group(1)
        idm = re.search(r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})', blk)
        tm = re.search(r'<p class="title is-5 mathjax">(.*?)</p>', blk, re.S)
        dm = re.search(r'Submitted</span>\s*([^;<]*)', blk)
        if not idm: continue
        title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", tm.group(1)))).strip() if tm else "?"
        out.append((idm.group(1), (dm.group(1).strip() if dm else "?"), title))
    return out, url

if __name__ == "__main__":
    for q in sys.argv[1:]:
        try:
            rows, url = search(q)
        except Exception as e:
            print(f"### QUERY: {q} EXC {e}"); continue
        print(f"\n### QUERY: {q} -> {len(rows)}")
        for i, d, t in rows:
            print(f"{i}\t{d[:20]}\t{t}")
        sys.stdout.flush(); time.sleep(12)
