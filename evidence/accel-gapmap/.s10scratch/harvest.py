#!/usr/bin/env python3
"""Harvest withdrawn/retracted-looking entries from arXiv HTML search result pages.

Usage: python3 harvest.py "<query>" [searchtype] [size]
Prints: ID | TITLE | COMMENTS | ABSTRACT_SNIPPET
"""
import hashlib, html, os, re, subprocess, sys, json

CACHE = "/tmp/accelsrc"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
os.makedirs(CACHE, exist_ok=True)


def curl(url, timeout=90, tries=6):
    key = hashlib.sha1(url.encode()).hexdigest()
    raw = os.path.join(CACHE, key + ".html")
    if os.path.exists(raw) and os.path.getsize(raw) > 500:
        return open(raw, "rb").read().decode("utf-8", "replace")
    data = ""
    for i in range(tries):
        p = subprocess.run(
            ["curl", "-sL", "--retry", "3", "--retry-delay", "2", "--retry-all-errors",
             "--max-time", str(timeout), "-A", UA, url],
            capture_output=True)
        data = p.stdout.decode("utf-8", "replace")
        if len(data) > 500:
            break
        import time
        time.sleep(3 + 2 * i)
    if len(data) > 500:
        open(raw, "w", encoding="utf-8").write(data)
    return data


def strip_html(s):
    s = re.sub(r"(?is)<(script|style|svg|noscript)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</(p|div|li|tr|h[1-6]|pre|blockquote)>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def norm(s):
    return re.sub(r"\s+", " ", strip_html(s)).strip()


def parse(htm):
    out = []
    m = re.search(r"Showing\s+([\d,]+)[–\-]([\d,]+)\s+of\s+([\d,]+)\s+results", strip_html(htm))
    total = m.group(3) if m else "?"
    chunks = htm.split('<li class="arxiv-result">')[1:]
    for li in chunks:
        aid = re.search(r'arxiv\.org/abs/([\d.]+)', li)
        if not aid:
            continue
        aid = aid.group(1)
        t = re.search(r'(?s)<p class="title[^"]*">(.*?)</p>', li)
        title = norm(t.group(1)).replace("Title:", "").strip() if t else "?"
        c = re.search(r'(?s)Comments:</span>(.*?)</p>', li)
        comments = norm(c.group(1)) if c else ""
        ab = re.search(r'(?s)<span class="abstract-full[^"]*"[^>]*>(.*?)</span>\s*(?:<a|</p>|</span>)', li)
        if not ab:
            ab = re.search(r'(?s)<span class="abstract-full[^"]*"[^>]*>(.*)', li)
        abstract = norm(ab.group(1)) if ab else ""
        authors = re.search(r'(?s)<p class="authors">(.*?)</p>', li)
        auth = norm(authors.group(1)).replace("Authors:", "").strip() if authors else ""
        jref = re.search(r'(?s)Journal ref:</span>\s*<span[^>]*>(.*?)</span>', li)
        jr = norm(jref.group(1)) if jref else ""
        out.append(dict(id=aid, title=title, comments=comments, abstract=abstract,
                        authors=auth, jref=jr))
    return total, out


if __name__ == "__main__":
    q = sys.argv[1]
    if q.startswith("http"):
        htm = curl(q, timeout=60)
        total, items = parse(htm)
        print(f"### URL: {q} -> {len(items)} parsed / {total} total  (bytes={len(htm)})")
        for it in items:
            blob = (it["comments"] + " " + it["abstract"]).lower()
            flag = "***" if ("withdraw" in blob or "retract" in blob) else "   "
            print(f"{flag} {it['id']} | {it['title'][:120]}")
            if it["comments"]:
                print(f"      COMMENTS: {it['comments'][:600]}")
            if it["jref"]:
                print(f"      JREF: {it['jref'][:200]}")
        sys.exit(0)
    st = sys.argv[2] if len(sys.argv) > 2 else "all"
    size = sys.argv[3] if len(sys.argv) > 3 else "100"
    from urllib.parse import quote_plus
    url = f"https://arxiv.org/search/?searchtype={st}&query={quote_plus(q)}&size={size}&start=0"
    htm = curl(url)
    total, items = parse(htm)
    print(f"### QUERY: {q}  [searchtype={st}]  -> {len(items)} parsed / {total} total")
    for it in items:
        blob = (it["comments"] + " " + it["abstract"]).lower()
        flag = "***" if ("withdraw" in blob or "retract" in blob) else "   "
        print(f"{flag} {it['id']} | {it['title'][:110]}")
        if it["comments"]:
            print(f"      COMMENTS: {it['comments'][:400]}")
        if it["jref"]:
            print(f"      JREF: {it['jref'][:200]}")
    print()
