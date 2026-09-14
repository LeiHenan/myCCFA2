#!/usr/bin/env python3
"""E5 parallel batch search driver."""
import sys, concurrent.futures as cf
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from e5_batch import QUERIES
from e5_srch import search


def run(item):
    repo, q = item
    try:
        url, rows = search(repo, q)
    except Exception as e:
        return f"\n### [{repo}] {q}  -> ERROR {e}\n"
    out = [f"\n### [{repo}] {q}  -> {len(rows)} rows"]
    for num, state, title, u in rows:
        out.append(f"{num}\t{state}\t{title}\t{u}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(run, QUERIES):
            print(res, flush=True)
