#!/usr/bin/env python3
"""S3: Semantic Scholar discovery -> arXiv ids. Usage: s3_s2.py "query" ..."""
import sys, json, time, urllib.parse, re
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl

def q(query, limit=20):
    u = ("https://api.semanticscholar.org/graph/v1/paper/search?query=" + urllib.parse.quote(query) +
         f"&fields=title,abstract,externalIds,year,venue&limit={limit}")
    for _ in range(5):
        h = curl(u)
        try:
            j = json.loads(h)
        except Exception:
            time.sleep(8); continue
        if "data" in j:
            return j["data"]
        time.sleep(10)
    return []

if __name__ == "__main__":
    for query in sys.argv[1:]:
        rows = q(query)
        print(f"\n### S2: {query} -> {len(rows)}")
        for r in rows:
            ax = (r.get("externalIds") or {}).get("ArXiv")
            if not ax: continue
            print(f"{ax}\t{r.get('year')}\t{(r.get('title') or '')[:110]}")
        sys.stdout.flush(); time.sleep(5)
