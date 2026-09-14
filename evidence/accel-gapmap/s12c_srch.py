#!/usr/bin/env python3
"""S12c: robust parallel GitHub search fetcher.
Reads URLs from a file, fetches each (short curl timeout, cached in /tmp/accelsrc), parses with s12_gs.listing.
"""
import concurrent.futures, hashlib, os, subprocess, sys, time
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from s12_gs import listing

CACHE = "/tmp/accelsrc"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
os.makedirs(CACHE, exist_ok=True)
OUT = "/tmp/s12c_search"
os.makedirs(OUT, exist_ok=True)


def fetch(url, tries=3):
    key = hashlib.sha1(url.encode()).hexdigest()
    raw = os.path.join(CACHE, key + ".html")
    if os.path.exists(raw) and os.path.getsize(raw) > 2000:
        return open(raw, "rb").read().decode("utf-8", "replace")
    for i in range(tries):
        p = subprocess.run(
            ["curl", "-sL", "--max-time", "25", "-A", UA, url],
            capture_output=True)
        data = p.stdout.decode("utf-8", "replace")
        if len(data) > 2000:
            open(raw, "w", encoding="utf-8").write(data)
            return data
        time.sleep(1 + i)
    return data


def run(url):
    k = hashlib.sha1(url.encode()).hexdigest()[:16]
    dest = os.path.join(OUT, k + ".txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        return url, "cached"
    try:
        fetch(url)  # warm the /tmp/accelsrc cache
        rows = listing(url)
        body = [f"### {url} -> {len(rows)} rows"]
        for num, repo, title, exc, u in rows:
            body.append(f"{num}\t{repo}\t{title}\n\tEXC: {exc}\n\t{u}")
        out = "\n".join(body)
    except Exception as e:
        out = f"### {url} -> ERROR {e}"
    open(dest, "w").write(out)
    return url, out.split("\n")[0][-60:]


if __name__ == "__main__":
    urls = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(sys.argv[2]) if len(sys.argv) > 2 else 4) as ex:
        for u, n in ex.map(run, urls):
            print(n, u[:120], flush=True)
