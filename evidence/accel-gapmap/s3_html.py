#!/usr/bin/env python3
"""S3: fetch arXiv HTML full text (arxiv.org/html/<id>vN) -> text file for grep."""
import sys, os, re, subprocess
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl, strip_html
OUT = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/.s3scratch/html"
os.makedirs(OUT, exist_ok=True)
for aid in sys.argv[1:]:
    got = False
    for v in ("v1", "v2", "v3", "v4", ""):
        u = f"https://arxiv.org/html/{aid}{v}"
        h = curl(u)
        if len(h) > 20000 and "no HTML" not in h[:3000]:
            t = strip_html(h)
            open(os.path.join(OUT, aid + ".txt"), "w").write(t)
            print(f"{aid}\tOK\t{v}\tchars={len(t)}")
            got = True; break
    if not got:
        print(f"{aid}\tNO-HTML")
    sys.stdout.flush()
