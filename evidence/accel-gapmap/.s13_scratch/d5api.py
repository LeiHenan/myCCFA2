#!/usr/bin/env python3
"""arXiv API fetch with retry/backoff + cache; prints ids/titles/abstracts.

Usage: python3 d5api.py "<search_query>" <max_results> [outfile]
"""
import hashlib, os, re, subprocess, sys, time, html

CACHE = "/tmp/accelsrc"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
os.makedirs(CACHE, exist_ok=True)


def fetch(url, tries=8):
    key = hashlib.sha1(("API::" + url).encode()).hexdigest()
    path = os.path.join(CACHE, key + ".atom")
    if os.path.exists(path) and os.path.getsize(path) > 800:
        return open(path, encoding="utf-8").read()
    data = ""
    for i in range(tries):
        p = subprocess.run(["curl", "-sL", "--max-time", "60", "-A", UA, url],
                           capture_output=True)
        data = p.stdout.decode("utf-8", "replace")
        if "<entry" in data or "<feed" in data:
            break
        time.sleep(3 + 2 * i)
    if len(data) > 800:
        open(path, "w", encoding="utf-8").write(data)
    return data


def unesc(s):
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    q = sys.argv[1]
    n = sys.argv[2]
    url = ("http://export.arxiv.org/api/query?search_query=" + q +
           "&start=0&max_results=" + n +
           "&sortBy=submittedDate&sortOrder=descending")
    data = fetch(url)
    if "<entry" not in data:
        print("NO ENTRIES / ERROR:", data[:400])
        return
    entries = re.findall(r"(?s)<entry>(.*?)</entry>", data)
    print("### QUERY:", q, "-> entries:", len(entries))
    for e in entries:
        aid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
        t = re.search(r"(?s)<title>(.*?)</title>", e)
        pub = re.search(r"<published>([^<]+)</published>", e)
        ab = re.search(r"(?s)<summary>(.*?)</summary>", e)
        print("=" * 100)
        print("ID:", aid.group(1) if aid else "?")
        print("DATE:", (pub.group(1)[:10] if pub else "?"))
        print("TITLE:", unesc(t.group(1)) if t else "?")
        print("ABS:", unesc(ab.group(1)) if ab else "?")


if __name__ == "__main__":
    main()
