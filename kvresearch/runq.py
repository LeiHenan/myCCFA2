#!/usr/bin/env python3
"""Run many queries with a polite delay, print compact lines."""
import sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghsearch import search

def run(queries, kind=None, state=None, sort="created", order="desc", page=1, tag=""):
    allq = []
    for q in queries:
        try:
            nodes, total, full = search(q, sort=sort, order=order, page=page, state=state, kind=kind)
        except Exception as e:
            print("ERR", q, e); time.sleep(6); continue
        print(f"\n=== [{tag}] {full}  (n={len(nodes)}) ===")
        for it in nodes:
            flag = ""
            if it["stateReason"] == "NOT_PLANNED": flag = "***NOT_PLANNED***"
            elif "stale" in it["labels"]: flag = "***STALE***"
            elif "wontfix" in it["labels"]: flag = "***WONTFIX***"
            elif it["stateReason"] == "COMPLETED": flag = "completed"
            elif it["state"] == "CLOSED": flag = "closed?"
            elif it["state"] == "MERGED": flag = "merged"
            print(f"{it['number']:>7} {str(it['state']):<8} {str(it['stateReason']):<12} {it['type'][:2]} {it['createdAt']} "
                  f"{'/'.join(it['labels'])[:34]:<34} {it['title'][:78]} {flag}")
        allq.append(full)
        time.sleep(6)
    return allq

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind")
    ap.add_argument("--state")
    ap.add_argument("--sort", default="created")
    ap.add_argument("--order", default="desc")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("queries", nargs="+")
    a = ap.parse_args()
    run(a.queries, kind=a.kind, state=a.state, sort=a.sort, order=a.order, page=a.page, tag=a.tag)
