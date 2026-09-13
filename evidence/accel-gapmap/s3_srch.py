#!/usr/bin/env python3
"""S3 helper: arXiv API search -> compact candidate list (id | date | title)."""
import re, sys, html
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl

def search(q, n=40, sort="relevance"):
    import urllib.parse
    url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q) +
           f"&start=0&max_results={n}&sortBy={sort}&sortOrder=descending")
    t = curl(url)
    ents = re.findall(r"<entry>(.*?)</entry>", t, re.S)
    out = []
    for e in ents:
        idm = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
        tm = re.search(r"<title>(.*?)</title>", e, re.S)
        dm = re.search(r"<published>([^<]+)</published>", e)
        if not idm or not tm: continue
        title = re.sub(r"\s+", " ", html.unescape(tm.group(1))).strip()
        out.append((idm.group(1), dm.group(1)[:10] if dm else "?", title))
    return out

if __name__ == "__main__":
    sort = "relevance"
    args = sys.argv[1:]
    if args and args[0].startswith("sort="):
        sort = args[0][5:]; args = args[1:]
    for q in args:
        rows = search(q, 40, sort)
        print(f"\n### QUERY: {q}  -> {len(rows)}")
        for i, d, t in rows:
            print(f"{i}\t{d}\t{t}")
