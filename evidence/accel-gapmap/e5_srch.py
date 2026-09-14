#!/usr/bin/env python3
"""E5 helper: repo-scoped GitHub issues/pulls search (github.com/search is HTTP 429 here).

Usage:  python3 e5_srch.py <repo> <query terms> [<query terms> ...]
Prints per query: number | state | title | url
"""
import re, sys, urllib.parse
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl, strip_html

ANCHOR = re.compile(
    r'href="(/[^/]+/[^/]+/(?:issues|pull)/(\d+))"[^>]*data-testid="issue-pr-title-link"[^>]*>(.*?)</a>',
    re.S,
)


def search(repo, terms, kind="issues"):
    url = f"https://github.com/{repo}/{kind}?q=" + urllib.parse.quote(terms)
    h = curl(url)
    ms = list(ANCHOR.finditer(h))
    out = []
    for idx, m in enumerate(ms):
        href, num, raw = m.group(1), m.group(2), m.group(3)
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", raw)).strip()
        end = ms[idx + 1].start() if idx + 1 < len(ms) else min(len(h), m.end() + 4000)
        win = strip_html(h[m.end():end])
        mm = re.search(r"(opened|was closed|was merged)[^\n]*", win)
        state = mm.group(0).strip() if mm else ""
        out.append((num, state, title[:160], "https://github.com" + href))
    return url, out


if __name__ == "__main__":
    repo = sys.argv[1]
    for q in sys.argv[2:]:
        url, rows = search(repo, q)
        print(f"\n### [{repo}] {q}  -> {len(rows)} rows")
        for num, state, title, u in rows:
            print(f"{num}\t{state}\t{title}")
