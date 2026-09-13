#!/usr/bin/env python3
"""S12 batch fetcher: fetch a list of GitHub URLs via fetch.py gh, store in /tmp/s12/."""
import os, subprocess, sys, hashlib
OUT = "/tmp/s12"
os.makedirs(OUT, exist_ok=True)
urls = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
for u in urls:
    key = hashlib.sha1(u.encode()).hexdigest()[:16]
    dest = os.path.join(OUT, key + ".txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        continue
    try:
        p = subprocess.run(["python3", "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/fetch.py", "gh", u],
                           capture_output=True, timeout=180)
        txt = p.stdout.decode("utf-8", "replace")
    except Exception as e:
        txt = "FETCH ERROR " + str(e)
    with open(dest, "w") as f:
        f.write("URL: " + u + "\n" + txt)
    print(u, len(txt), flush=True)
