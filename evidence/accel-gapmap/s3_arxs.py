#!/usr/bin/env python3
"""S3 helper: arXiv web-search (HTML) -> candidate list. Usage:
   python3 s3_arxs.py "<query>" [size] [sortBy]
   sortBy: relevance | submittedDate-desc | announcementDate-desc  (default submittedDate-desc)
"""
import re, sys, html, urllib.parse
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl

def search(q, size=50, sort="submittedDate-desc"):
    url = ("https://arxiv.org/search/?searchtype=all&query=" + urllib.parse.quote(q) +
           f"&size={size}&sortBy={sort}")
    h = curl(url)
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
    q = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    sort = sys.argv[3] if len(sys.argv) > 3 else "submittedDate-desc"
    rows, url = search(q, size, sort)
    print(f"### QUERY: {q}  ({sort}) -> {len(rows)}  {url}")
    for i, d, t in rows:
        print(f"{i}\t{d[:20]}\t{t}")
