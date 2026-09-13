#!/usr/bin/env python3
"""S12 helper: parse ANY GitHub listing/search page (React-embedded JSON payload).

Usage: python3 s12_list.py '<url>' [--excerpts]
Prints: number <TAB> state <TAB> title <TAB> url
"""
import sys, re, json, html
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl


def entries(h):
    seen = {}
    # primary: JSON objects that carry both a url and a title
    for m in re.finditer(
            r'"title":"((?:[^"\\]|\\.)*)","titleHTML"', h):
        pass
    # general: find url first then nearest preceding title
    for m in re.finditer(r'"url":"(https://github\.com/([^/"]+)/([^/"]+)/(issues|pull)/(\d+))"', h):
        url, owner, repo, kind, num = m.groups()
        seg = h[max(0, m.start() - 3000):m.start() + 4000]
        t = re.findall(r'"title":"((?:[^"\\]|\\.)*)"', seg)
        title = ""
        for cand in reversed(t):
            try:
                cand = json.loads('"' + cand + '"')
            except Exception:
                continue
            if len(cand) > 8 and not cand.startswith("Issues") and "·" not in cand:
                title = cand
                break
        st = re.findall(r'"state":"([A-Za-z_]+)"', seg)
        seen[num] = (num, st[-1] if st else "?", title, url)
    return list(seen.values())


if __name__ == "__main__":
    url = sys.argv[1]
    h = curl(url)
    rows = entries(h)
    print(f"### {url} -> {len(rows)} rows")
    for num, st, title, u in sorted(rows, key=lambda r: -int(r[0])):
        print(f"{num}\t{st}\t{title}\n\t{u}")
