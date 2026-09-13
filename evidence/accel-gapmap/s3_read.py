#!/usr/bin/env python3
"""S3: readback arXiv IDs via fetch.py arxiv, save full output, print TITLE/DATELINE."""
import subprocess, sys, os, re, time
OUT = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/.s3scratch/reads"
os.makedirs(OUT, exist_ok=True)
for aid in sys.argv[1:]:
    dest = os.path.join(OUT, aid.replace("/", "_") + ".txt")
    if not os.path.exists(dest):
        r = subprocess.run(["python3", "fetch.py", "arxiv", aid], capture_output=True, text=True, timeout=180)
        open(dest, "w").write(r.stdout)
        time.sleep(1)
    t = open(dest).read()
    tm = re.search(r"^TITLE: (.*)$", t, re.M)
    dm = re.search(r"^DATELINE: (.*)$", t, re.M)
    ok = "ARXIV_ID_READBACK: " + aid in t
    print(f"{aid}\t{'OK' if ok else 'MISSING'}\t{(dm.group(1) if dm else '?')}\t{(tm.group(1) if tm else 'NO-TITLE')}")
    sys.stdout.flush()
