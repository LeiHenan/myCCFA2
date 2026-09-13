#!/usr/bin/env python3
"""S12 helper: parse a GitHub GLOBAL search page (issues+PRs) into rows."""
import re, sys, html
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl

def listing(url):
    h = curl(url)
    out = []
    parts = h.split('Result-module__Result__')
    for p in parts[1:]:
        m = re.search(r'href="(/[^"/]+/[^"/]+/(?:issues|pull)/(\d+))"', p)
        if not m:
            continue
        href, num = m.group(1), m.group(2)
        t = re.search(r'href="/[^"]*/(?:issues|pull)/\d+"><span[^>]*>(.*?)</span></a>', p, re.S)
        title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", t.group(1)))).strip() if t else "?"
        c = re.search(r'Content-module__Content[^>]*>(.*?)</div>', p, re.S)
        exc = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c.group(1)))).strip() if c else ""
        repo = re.search(r'href="(/[^"/]+/[^"/]+)"', p)
        out.append((num, repo.group(1) if repo else "?", title, exc[:400], "https://github.com" + href))
    return out

if __name__ == "__main__":
    url = sys.argv[1]
    rows = listing(url)
    print(f"### {url}  -> {len(rows)} rows")
    for num, repo, title, exc, u in rows:
        print(f"{num}\t{repo}\t{title}\n\tEXC: {exc}\n\t{u}")
