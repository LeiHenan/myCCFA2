#!/usr/bin/env python3
"""S12 S3B probe: for each (repo, keyword) pair, look for OPEN PRs and OPEN issues
on that topic, using repo-scoped listing pages (explicit state markers).

Usage: python3 s12_s3b.py probes.tsv   where each line is  <repo>\t<keyword>[\t<label>]
Prints:  LABEL \t REPO \t KEYWORD \t n_open \t urls
"""
import sys
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from s12_srch import listing
from urllib.parse import quote

if __name__ == "__main__":
    for line in open(sys.argv[1]):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        label, repo, kw = parts[0], parts[1], parts[2]
        hits, urls = [], []
        for kind, path, qual in (("PR", "pulls", "is:pr"), ("issue", "issues", "is:issue")):
            u = f"https://github.com/{repo}/{path}?q=" + quote(f"{qual} is:open {kw}")
            urls.append(u)
            try:
                for num, state, dates, title, link in listing(u):
                    if state == "OPEN":
                        hits.append(f"{kind}#{num} {title[:100]}")
            except Exception as e:
                hits.append(f"ERR {e}")
        print(f"{label}\t{repo}\t{kw}\tOPEN={len(hits)}\t{urls[0]} ; {urls[1]}")
        for h in hits[:5]:
            print(f"\t  -> {h}")
        sys.stdout.flush()
