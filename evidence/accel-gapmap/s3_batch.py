import subprocess, sys, time
QS = sys.argv[1:]
for q in QS:
    for sort in ("submittedDate-desc",):
        try:
            r = subprocess.run([sys.executable, "s3_arxs.py", q, "50", sort],
                               capture_output=True, text=True, timeout=180)
            print(r.stdout); print(r.stderr[:300])
        except Exception as e:
            print(f"### QUERY: {q} FAILED {e}")
        sys.stdout.flush(); time.sleep(2)
