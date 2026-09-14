#!/usr/bin/env python3
"""S12e: parse a repo-scoped GitHub issues list page (fast, not rate-limited) into rows.

usage: s12e_list.py '<repo issues list url>' [more urls...]
prints: number | state-ish | labels | title | url
"""
import sys, re, json, html
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl


def parse(url):
    h = curl(url)
    rows = []
    idx = h.find('"search":{"edges":[')
    if idx < 0:
        # fall back: graphql payload variant
        idx = h.find('"edges":[{"node":{"__typename":"Issue"')
        if idx < 0:
            return rows, h
        start = h.find("[", idx)
    else:
        start = h.find("[", idx)
    dec = json.JSONDecoder()
    try:
        arr, _ = dec.raw_decode(h[start:])
    except Exception as e:
        return rows, h
    for e in arr:
        n = e.get("node", {})
        if not n:
            continue
        title = re.sub(r"<[^>]+>", "", n.get("titleHtml", "") or "")
        title = html.unescape(title).strip()
        labs = []
        for le in (n.get("labels") or {}).get("edges", []) or []:
            labs.append(le["node"].get("name", ""))
        num = n.get("number")
        kind = "/pull/" if n.get("__isIssueOrPullRequest") == "PullRequest" else "/issues/"
        rows.append((num, n.get("state", "?"), ",".join(labs), title,
                     f"https://github.com{url.split('github.com')[1].split('/issues')[0]}{kind}{num}"))
    return rows, h


if __name__ == "__main__":
    for url in sys.argv[1:]:
        rows, h = parse(url)
        print(f"### {url}  -> {len(rows)} rows (html {len(h)})")
        for num, state, labs, title, u in rows:
            print(f"{num}\t{state}\t{labs}\t{title}\n\t{u}")
