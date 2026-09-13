#!/usr/bin/env python3
"""Batch runner: fetch many URLs via fetch.py, save each to .s3scratch/raw/<slug>.txt"""
import subprocess, sys, os, re, hashlib, concurrent.futures as cf

BASE = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap"
OUT = os.path.join(BASE, ".s3scratch", "raw")
os.makedirs(OUT, exist_ok=True)

def slug(url):
    s = re.sub(r'^https?://', '', url)
    s = re.sub(r'[^A-Za-z0-9]+', '_', s)
    return s.strip('_')[:120]

def run(spec):
    mode, url = spec
    p = os.path.join(OUT, f"{mode}__{slug(url)}.txt")
    if os.path.exists(p) and os.path.getsize(p) > 200:
        return (url, p, "cached")
    try:
        r = subprocess.run([sys.executable, os.path.join(BASE, "fetch.py"), mode, url],
                           capture_output=True, timeout=300)
        txt = r.stdout.decode("utf-8", "replace")
        err = r.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        txt, err = "", "TIMEOUT"
    open(p, "w", encoding="utf-8").write(txt + ("\n---STDERR---\n" + err if err.strip() else ""))
    return (url, p, f"{len(txt)}b")

if __name__ == "__main__":
    specs = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            specs.append((parts[0], parts[1]))
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for url, p, status in ex.map(run, specs):
            print(f"{status:>10}  {os.path.basename(p)}")
