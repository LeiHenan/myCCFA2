#!/usr/bin/env python3
"""E2 batch GitHub issue/PR search for the never-discussed test."""
import sys, re, html, urllib.parse
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl


def rows(repo, q, kind="issues"):
    url = (f"https://github.com/{repo}/{kind}?q=" +
           urllib.parse.quote(q, safe="") + "&state=all")
    h = curl(url)
    out = []
    for m in re.finditer(r'href="/' + re.escape(repo) + r'/(issues|pull)/(\d+)"[^>]*>(.*?)</a>', h, re.S):
        kind_, num, inner = m.group(1), m.group(2), m.group(3)
        txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", inner))).strip()
        if len(txt) < 8:
            continue
        if txt.lower() in ("issues", "pull requests", "labels", "milestones"):
            continue
        out.append((num, kind_, txt[:190]))
    seen, res = set(), []
    for num, kind_, txt in out:
        if num in seen:
            continue
        seen.add(num)
        res.append((num, kind_, txt))
    return url, res


if __name__ == "__main__":
    # args: repo q1 | q2 | q3 ...
    repo = sys.argv[1]
    for q in sys.argv[2:]:
        try:
            url, r = rows(repo, q)
        except Exception as e:
            print(f"\n### REPO {repo} QUERY: {q} -> ERROR {e}")
            continue
        print(f"\n### QUERY: {q}  -> {len(r)} rows")
        for num, k, t in r:
            print(f"  {k}/{num}\t{t}")
