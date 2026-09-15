#!/usr/bin/env python
"""C1 redo -- fetch every checkpoint the corrected grid needs, from hf-mirror.

Design notes:
  * resumable (`curl -C -`) and size-verified against the API's blob sizes, so a
    truncated 4 GB shard can never masquerade as a complete one;
  * explicit per-repo excludes, because some repos carry 90 GB of training jsonl
    or duplicate `.pt` weights that this experiment will never open;
  * priority-ordered: the dense target and its drafters come first so the
    instrument-validation run can start while the rest is still landing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = "/root/autodl-tmp/models"
WORKERS = 4

# (repo, local name, substrings to skip)  -- order == priority
JOBS = [
    ("Qwen/Qwen3-4B", "Qwen3-4B", []),
    ("AngelSlim/Qwen3-4B_eagle3", "eagle3-qwen3-4b", []),
    ("mgoin/Qwen3-4B-speculator.dflash2", "dflash-qwen3-4b", []),
    ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", []),
    ("yuyijiong/Qwen3.5-4B-Eagle3", "eagle3-qwen3.5-4b", []),
    ("shanjiaz/qwen3.5-4b-speculator.dflash", "dflash-qwen3.5-4b",
     ["open-perfectblend", "optimizer_state_dict"]),
    ("tiiuae/Falcon-H1-3B-Instruct", "Falcon-H1-3B-Instruct", []),
]

PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


# hf-mirror returns 403 to python-urllib's default UA but serves curl fine.
UA = {"User-Agent": "curl/8.5.0"}


def list_files(repo: str):
    url = f"{ENDPOINT}/api/models/{repo}?blobs=true"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
    out = []
    for s in d.get("siblings", []):
        out.append((s["rfilename"], s.get("size") or 0))
    return out


def fetch(repo: str, dest: str, fname: str, expected: int) -> str:
    out = os.path.join(dest, fname)
    os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
    have = os.path.getsize(out) if os.path.exists(out) else 0
    if expected and have == expected:
        return f"cached {fname}"
    if expected and have > expected:
        # Two concurrent downloaders were once pointed at the same path and both
        # appended from the same offset.  A resume (curl -C -) would resume from
        # the *inflated* size and make it worse, so restart this file from zero.
        os.remove(out)
        have = 0
    url = f"{ENDPOINT}/{repo}/resolve/main/{fname}"
    # --speed-limit/--speed-time abort a STALLED transfer (observed: a shard sat
    # at 4.17/5.33 GB making zero progress for minutes); without them curl waits
    # forever and the whole queue blocks behind one dead connection.
    cmd = ["curl", "-s", "-L", "--retry", "8", "--retry-delay", "3",
           "--retry-all-errors", "--speed-limit", "50000", "--speed-time", "45",
           "--connect-timeout", "30", "-C", "-", "-o", out, url]
    rc = subprocess.call(cmd)
    got = os.path.getsize(out) if os.path.exists(out) else 0
    if rc != 0:
        return f"FAIL rc={rc} {fname} ({got}/{expected})"
    if expected and got != expected:
        return f"SIZE-MISMATCH {fname} got={got} want={expected}"
    return f"ok {fname} ({got / 1e6:.0f} MB)"


def main() -> int:
    tasks = []
    list_failures = []
    for repo, short, excl in JOBS:
        dest = os.path.join(ROOT, short)
        try:
            files = list_files(repo)
        except Exception as e:  # noqa: BLE001
            log(f"[{repo}] LIST-FAILED {type(e).__name__}: {e}")
            list_failures.append(repo)
            continue
        keep = [(f, s) for f, s in files
                if not any(e and e in f for e in excl)]
        skipped = [f for f, _ in files if any(e and e in f for e in excl)]
        total = sum(s for _, s in keep)
        log(f"[{repo}] -> {dest}  {len(keep)} files, {total / 1e9:.2f} GB"
            + (f"  (skipped {len(skipped)}: {', '.join(skipped)})" if skipped else ""))
        for f, s in keep:
            tasks.append((repo, dest, f, s))

    log(f"\n=== {len(tasks)} files queued, {WORKERS} parallel ===\n")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(fetch, r, d, f, s): (r, f) for r, d, f, s in tasks}
        for fut in futs:
            r, f = futs[fut]
            try:
                msg = fut.result()
            except Exception as e:  # noqa: BLE001
                msg = f"EXC {type(e).__name__}: {e}"
            results.append((r, f, msg))
            log(f"  {msg}")

    bad = [x for x in results if not (x[2].startswith("ok") or x[2].startswith("cached"))]
    log(f"\n=== SUMMARY: {len(results) - len(bad)}/{len(results)} files ok ===")
    for r, f, m in bad:
        log(f"  BAD {r}/{f}: {m}")
    for r in list_failures:
        log(f"  BAD {r}: repo listing failed, nothing downloaded")
    ok = (not bad) and (not list_failures) and bool(results)
    log("ALL_DOWNLOADS_COMPLETE" if ok else "DOWNLOADS_HAVE_FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
