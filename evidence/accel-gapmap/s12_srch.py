#!/usr/bin/env python3
"""S12 helper: fetch a GitHub search-results page and list (num, title, dates)."""
import re, sys, html
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl

def listing(url):
    h = curl(url)
    out = []
    for m in re.finditer(
        r'<a id="issue_(\d+)_link"[^>]*href="(/[^"]+)"[^>]*>(.*?)</a>(.{0,2500}?)(?=<!-- Issue title column -->|</div>\s*</div>\s*</div>)',
        h, re.S):
        num, href, title, tail = m.group(1), m.group(2), m.group(3), m.group(4)
        title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", title))).strip()
        dates = re.findall(r'<relative-time datetime="([^"]+)"', tail)
        state = "closed" if "was closed" in tail or "was merged" in tail else "open"
        out.append((num, title, dates[:1], state, "https://github.com" + href))
    return out

if __name__ == "__main__":
    url = sys.argv[1]
    rows = listing(url)
    print(f"### {url}  -> {len(rows)} rows")
    for num, title, dates, state, u in rows:
        print(f"{num}\t{dates}\t{state}\t{title}")
