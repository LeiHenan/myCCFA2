#!/usr/bin/env python3
"""Scan page dumps for negative-result phrases and print verbatim context."""
import os, re, sys, glob

PAGES = "/Users/leihenan/Desktop/myProject/kv_discovery/pages"
PHRASES = [
    "won't fix", "wontfix", "not planned", "by design", "out of scope",
    "we decided against", "no longer pursuing", "no plans to", "stale",
    "not worth", "does not help", "no benefit", "no gain", "no improvement",
    "regression", "reverted", "revert", "closing this", "close this",
    "closing as", "not going to", "abandon", "depriorit", "not pursuing",
    "decided not", "won't be", "will not be", "dropping", "dropped",
    "superseded", "no longer", "doesn't help", "didn't help", "not helpful",
    "won't merge", "closing", "not merge", "no measurable", "not worth the",
    "low effort", "agent generated", "slop", "closed-as-slop",
    "not supported", "unsupported", "design decision", "intentional",
    "working as intended", "expected behavior", "cannot reproduce",
    "not reproducible", "duplicate", "invalid",
]

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(glob.glob(os.path.join(PAGES, "*.txt")))
    for f in files:
        name = os.path.basename(f)
        if only and only not in name:
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        low = txt.lower()
        hits = []
        for ph in PHRASES:
            for m in re.finditer(re.escape(ph.lower()), low):
                s = max(0, m.start() - 350)
                e = min(len(txt), m.end() + 350)
                ctx = txt[s:e].replace("\n", " ")
                ctx = re.sub(r"\s+", " ", ctx)
                hits.append((ph, ctx))
        if hits:
            print(f"\n\n############### {name} ({len(hits)} hits) ###############")
            seen = set()
            for ph, ctx in hits:
                k = ctx[:120]
                if k in seen:
                    continue
                seen.add(k)
                print(f"  <{ph}> ...{ctx}...")

main()
