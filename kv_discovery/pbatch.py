#!/usr/bin/env python3
"""Parallel batch-read GitHub pages. Usage: pbatch.py listfile [workers]"""
import sys, os, time, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT = "/Users/leihenan/Desktop/myProject/kv_discovery/pages"
SCRIPT = "/Users/leihenan/Desktop/myProject/kv_discovery/gh_page.py"


def work(item):
    repo, num = item
    dest = os.path.join(OUT, f"{repo.replace('/','__')}__{num}.txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        return f"SKIP {repo}#{num}"
    try:
        p = subprocess.run(["python3", SCRIPT, repo, num],
                           capture_output=True, timeout=240)
        txt = p.stdout.decode("utf-8", "replace")
        err = p.stderr.decode("utf-8", "replace")
        with open(dest, "w") as f:
            f.write(txt)
            if err.strip():
                f.write("\n\n##### STDERR #####\n" + err)
        return f"OK {repo}#{num} ({len(txt)}b)"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT {repo}#{num}"
    except Exception as e:
        return f"ERR {repo}#{num}: {e}"


def main():
    listfile = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(OUT, exist_ok=True)
    items = []
    for line in open(listfile):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            items.append((parts[0], parts[1]))
    print(f"{len(items)} items, {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, it): it for it in items}
        for i, fu in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(items)}] {fu.result()}", flush=True)
    print("PBATCH DONE", flush=True)


if __name__ == "__main__":
    main()
