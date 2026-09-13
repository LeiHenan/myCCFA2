#!/usr/bin/env python3
"""Fetch PR pages and classify merged vs closed-unmerged."""
import sys, os, json, re, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghpage import read, fetch

def pr_state(repo, num):
    t = fetch(f"https://github.com/{repo}/pull/{num}?plain=1")
    if len(t) < 2000:
        return {"repo": repo, "number": num, "err": "fetch_failed"}
    st = None
    m = re.search(r'data-status="(pullMerged|pullClosed|pullOpen)"', t)
    if m:
        st = m.group(1)
    title = None
    m2 = re.search(r'<script type="application/json" data-target="react-app.embeddedData">(.*?)</script>', t, re.S)
    date = None
    author = None
    if m2:
        try:
            d = json.loads(m2.group(1))
            pt = d.get("title") or ""
            mm = re.match(r"(.*?) by ([\w.\-]+) · (?:Pull Request|Issue) #\d+ · ", pt, re.S)
            sd = d["payload"].get("structured_data") or {}
            if mm:
                title = mm.group(1).strip()
                author = mm.group(2)
            else:
                title = sd.get("headline")
                a = sd.get("author")
                author = a.get("name") if isinstance(a, dict) else a
            date = (sd.get("datePublished") or "")[:10]
        except Exception:
            pass
    return {"repo": repo, "number": num, "url": f"https://github.com/{repo}/pull/{num}",
            "pr_state": st, "title": title, "date": date, "author": author,
            "closed_unmerged": st == "pullClosed"}

if __name__ == "__main__":
    tgt = []
    for line in open(sys.argv[1]):
        line = line.strip()
        if line and not line.startswith("#"):
            a = line.split()
            tgt.append((a[0], a[1]))
    res = []
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        for r in ex.map(lambda x: pr_state(*x), tgt):
            res.append(r)
            print(f"{r.get('number')} {r.get('pr_state')} {r.get('date')} {str(r.get('title'))[:85]}", flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
