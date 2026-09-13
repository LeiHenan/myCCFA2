#!/usr/bin/env python3
"""S10: pair each comment body with the nearest PRECEDING author login in the HTML payload."""
import hashlib, json, os, re, sys
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl

url = sys.argv[1]
h = curl(url)

auths = [(m.start(), m.group(1)) for m in
         re.finditer(r'"author":\{"[^{}]*?"login":"([^"]+)"', h)]
bodies = [(m.start(), m.group(1)) for m in
          re.finditer(r'"body":"((?:[^"\\]|\\.)*)"', h)]

print("URL:", url)
seen = set()
for pos, raw in bodies:
    prev = [a for p, a in auths if p < pos]
    who = prev[-1] if prev else "?"
    try:
        txt = json.loads('"' + raw + '"')
    except Exception:
        txt = raw
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt or txt in seen:
        continue
    seen.add(txt)
    print(f"\n--- @{who} ---\n{txt[:3000]}")
