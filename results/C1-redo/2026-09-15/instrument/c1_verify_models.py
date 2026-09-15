#!/usr/bin/env python
"""C1 redo -- verify every checkpoint byte-for-byte, and repair what is broken.

WHY THIS EXISTS
---------------
Two downloader processes once ran concurrently against the same paths and both
resumed from the same offsets.  The resulting files had EXACTLY the expected byte
sizes, so a size check passed -- but the contents were interleaved garbage.  That
went unnoticed until the model itself was asked to generate:

    transformers, greedy, "The capital of France is" -> "IIIIIIIIIIIIIIIIIIII"
    top-5 next tokens: ('II', 11.25), (' II', 6.75), ('III', 6.44) ...

A size check is not an integrity check.  This script compares against the hash the
hub itself publishes:
  * LFS files  -> lfs.sha256, compared directly;
  * git blobs  -> the git object id, recomputed as sha1("blob <len>\\0" + data).
Anything that mismatches is deleted and fetched again from scratch (never resumed),
then re-verified.  A lock file stops two instances from racing again.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = "/root/autodl-tmp/models"
UA = {"User-Agent": "curl/8.5.0"}
LOCK = "/root/autodl-tmp/.verify.lock"

JOBS = [
    ("Qwen/Qwen3-4B", "Qwen3-4B"),
    ("AngelSlim/Qwen3-4B_eagle3", "eagle3-qwen3-4b"),
    ("mgoin/Qwen3-4B-speculator.dflash2", "dflash-qwen3-4b"),
    ("Qwen/Qwen3.5-4B", "Qwen3.5-4B"),
    ("yuyijiong/Qwen3.5-4B-Eagle3", "eagle3-qwen3.5-4b"),
    ("shanjiaz/qwen3.5-4b-speculator.dflash", "dflash-qwen3.5-4b"),
    ("tiiuae/Falcon-H1-3B-Instruct", "Falcon-H1-3B-Instruct"),
]
SKIP = ("open-perfectblend", "optimizer_state_dict")


def list_files(repo):
    req = urllib.request.Request(f"{ENDPOINT}/api/models/{repo}?blobs=true", headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    out = []
    for s in d.get("siblings", []):
        name = s["rfilename"]
        if any(k in name for k in SKIP):
            continue
        out.append({
            "name": name,
            "size": s.get("size") or 0,
            "sha256": (s.get("lfs") or {}).get("sha256"),
            "blob_id": s.get("blobId"),
        })
    return out


def git_blob_sha1(path):
    h = hashlib.sha1()
    size = os.path.getsize(path)
    h.update(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check(repo, short, ent):
    path = os.path.join(ROOT, short, ent["name"])
    if not os.path.exists(path):
        return (repo, ent["name"], "MISSING", None)
    got_size = os.path.getsize(path)
    if ent["size"] and got_size != ent["size"]:
        return (repo, ent["name"], "SIZE", f"{got_size} != {ent['size']}")
    try:
        if ent["sha256"]:
            got = sha256_file(path)
            if got != ent["sha256"]:
                return (repo, ent["name"], "SHA256", f"{got[:16]} != {ent['sha256'][:16]}")
        elif ent["blob_id"]:
            got = git_blob_sha1(path)
            if got != ent["blob_id"]:
                return (repo, ent["name"], "GITBLOB", f"{got[:16]} != {ent['blob_id'][:16]}")
    except Exception as e:  # noqa: BLE001
        return (repo, ent["name"], "HASH_ERROR", str(e)[:120])
    return (repo, ent["name"], "OK", None)


def fetch_fresh(repo, short, ent):
    """Delete and re-download from zero.  Never resumes: a resume is what produced
    a right-sized-but-wrong file in the first place."""
    dest = os.path.join(ROOT, short, ent["name"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        os.remove(dest)
    url = f"{ENDPOINT}/{repo}/resolve/main/{ent['name']}"
    cmd = ["curl", "-s", "-L", "--retry", "8", "--retry-delay", "3",
           "--retry-all-errors", "--speed-limit", "50000", "--speed-time", "60",
           "--connect-timeout", "30", "-o", dest, url]
    rc = subprocess.call(cmd)
    return rc == 0


def main():
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another verifier/downloader holds the lock; refusing to race")
        return 2

    entries = []
    for repo, short in JOBS:
        try:
            for ent in list_files(repo):
                entries.append((repo, short, ent))
        except Exception as e:  # noqa: BLE001
            print(f"[{repo}] LIST-FAILED {type(e).__name__}: {e}")

    print(f"verifying {len(entries)} files against hub hashes...")
    bad = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for res in pool.map(lambda t: check(*t), entries):
            if res[2] != "OK":
                bad.append(res)
                print(f"  BAD {res[2]:10s} {res[0]}/{res[1]}  {res[3] or ''}")

    print(f"\n{len(entries) - len(bad)}/{len(entries)} verified OK; {len(bad)} need repair")
    if not bad:
        print("ALL_MODELS_VERIFIED")
        return 0

    print("\nre-fetching corrupt/missing files from scratch (serial)...")
    lookup = {(r, s, e["name"]): e for r, s, e in entries}
    for repo, name, kind, detail in bad:
        short = dict((r, s) for r, s in JOBS)[repo]
        ent = lookup[(repo, short, name)]
        print(f"  fetching {repo}/{name} ...", flush=True)
        fetch_fresh(repo, short, ent)

    print("\nre-verifying...")
    still = []
    for repo, name, kind, detail in bad:
        short = dict((r, s) for r, s in JOBS)[repo]
        ent = lookup[(repo, short, name)]
        res = check(repo, short, ent)
        print(f"  {res[2]:10s} {repo}/{name}")
        if res[2] != "OK":
            still.append(res)
    if still:
        print(f"\nSTILL_BROKEN: {len(still)}")
        for s in still:
            print("   ", s)
        return 1
    print("\nALL_MODELS_VERIFIED_AFTER_REPAIR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
