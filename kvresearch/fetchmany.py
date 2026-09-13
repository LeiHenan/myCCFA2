#!/usr/bin/env python3
"""Fetch many issue pages and dump title/date/state/body into a JSON file."""
import sys, os, json, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghpage import read

targets = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    repo, num = line.split()
    targets.append((repo, num))

out = []
def work(t):
    repo, num = t
    try:
        r = read(repo, num)
    except Exception as e:
        r = {"repo": repo, "number": num, "error": str(e)}
    r["url"] = f"https://github.com/{repo}/issues/{num}"
    return r

with cf.ThreadPoolExecutor(max_workers=5) as ex:
    for r in ex.map(work, targets):
        out.append(r)
        print(f"{r.get('number')} {r.get('state_marker')} {r.get('date')} {str(r.get('title'))[:80]} body_len={r.get('body_len')}", flush=True)

json.dump(out, open(sys.argv[2], "w"), indent=1)
print("WROTE", sys.argv[2], len(out))
