#!/usr/bin/env python3
"""S10 fetch batch: fetch.py gh on each URL in argv, save to /tmp/s10_fetch/<kind><num>.txt"""
import os, re, sys
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import gh

OUT = "/tmp/s10_fetch"
os.makedirs(OUT, exist_ok=True)

for url in sys.argv[1:]:
    m = re.search(r"/(issues|pull|discussions)/(\d+)", url)
    tag = (m.group(1)[:4] + m.group(2)) if m else re.sub(r"\W+", "_", url)[-40:]
    path = os.path.join(OUT, tag + ".txt")
    if os.path.exists(path) and os.path.getsize(path) > 400:
        print(f"SKIP(cached) {tag} {url}")
        continue
    try:
        txt = gh(url)
    except Exception as e:
        print(f"FAIL {url}: {e}")
        continue
    open(path, "w", encoding="utf-8").write(f"URL: {url}\n" + txt)
    n = txt.count("--- COMMENT ")
    print(f"OK {tag} comments={n} bytes={len(txt)}  {url}")
    sys.stdout.flush()
