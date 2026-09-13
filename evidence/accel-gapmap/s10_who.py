#!/usr/bin/env python3
"""S10: extract (author login, body) pairs from the cached GitHub HTML for an issue/PR page.

Usage: python3 s10_who.py <issue-or-pr-url>
"""
import hashlib, json, os, re, sys
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl, strip_html

url = sys.argv[1]
h = curl(url)
key = hashlib.sha1(url.encode()).hexdigest()
print("CACHE:", os.path.join("/tmp/accelsrc", key + ".html"))

logins = re.findall(r'"login":"([A-Za-z0-9_.\-]+)"', h)
print("LOGINS_IN_ORDER:", " | ".join(dict.fromkeys(logins))[:1200])

# comment bodies with their preceding author marker
bodies = re.findall(r'"body":"((?:[^"\\]|\\.)*)"', h)
authors = re.findall(r'"author":\{"[^}]*?"login":"([^"]+)"', h)
print()
print("BODY_COUNT:", len(bodies), "AUTHOR_COUNT:", len(authors))
for i, (a, b) in enumerate(zip(authors, bodies)):
    try:
        txt = json.loads('"' + b + '"')
    except Exception:
        txt = b
    txt = re.sub(r"\s+", " ", txt)[:400]
    print(f"[{i}] @{a}: {txt}")
