#!/usr/bin/env python3
"""Batch-read GitHub pages for a list of repo#num and write text dumps.

Input file: one "owner/repo NUM" per line.
Output: outdir/<repo>__<num>.txt
"""
import sys, os, time, subprocess

OUT = "/Users/leihenan/Desktop/myProject/kv_discovery/pages"
SCRIPT = "/Users/leihenan/Desktop/myProject/kv_discovery/gh_page.py"


def main():
    listfile = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    items = []
    for line in open(listfile):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            items.append((parts[0], parts[1]))
    print(f"{len(items)} items to read", flush=True)
    for i, (repo, num) in enumerate(items, 1):
        dest = os.path.join(OUT, f"{repo.replace('/','__')}__{num}.txt")
        if os.path.exists(dest) and os.path.getsize(dest) > 500:
            print(f"[{i}/{len(items)}] SKIP (cached) {repo}#{num}", flush=True)
            continue
        try:
            p = subprocess.run(["python3", SCRIPT, repo, num],
                               capture_output=True, timeout=180)
            txt = p.stdout.decode("utf-8", "replace")
            err = p.stderr.decode("utf-8", "replace")
            with open(dest, "w") as f:
                f.write(txt)
                if err.strip():
                    f.write("\n\n##### STDERR #####\n" + err)
            print(f"[{i}/{len(items)}] OK {repo}#{num} ({len(txt)} bytes)", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[{i}/{len(items)}] TIMEOUT {repo}#{num}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(items)}] ERR {repo}#{num}: {e}", flush=True)
        time.sleep(1.2)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
