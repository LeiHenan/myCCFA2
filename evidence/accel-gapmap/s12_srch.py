#!/usr/bin/env python3
"""S12 helper v2: parse a GitHub repo-scoped issue/PR listing page."""
import re, sys, html
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl


def listing(url):
    h = curl(url)
    out = []
    chunks = re.split(r'<a id="issue_(\d+)_link"', h)
    for i in range(1, len(chunks) - 1, 2):
        num = chunks[i]
        body = chunks[i + 1]
        href = re.search(r'href="(/[^"]+)"', body)
        title = re.search(r'>(.*?)</a>', body, re.S)
        title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", title.group(1)))).strip() if title else "?"
        tail = body[:2500]
        dates = re.findall(r'<relative-time datetime="([^"]+)"', tail)
        if "was closed" in tail:
            state = "CLOSED"
        elif "was merged" in tail:
            state = "MERGED"
        else:
            state = "OPEN"
        out.append((num, state, dates[:1], title, "https://github.com" + (href.group(1) if href else "")))
    return out


if __name__ == "__main__":
    url = sys.argv[1]
    rows = listing(url)
    print(f"### {url}  -> {len(rows)} rows")
    for num, state, dates, title, u in rows:
        print(f"{state}\t{num}\t{dates}\t{title[:130]}")
