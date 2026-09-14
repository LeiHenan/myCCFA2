#!/usr/bin/env python3
"""S12c: robust parallel GitHub issue/PR fetcher (fetch.py gh logic, short timeouts, cached)."""
import concurrent.futures, hashlib, os, subprocess, sys, time, json, re, html

sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import strip_html

CACHE = "/tmp/accelsrc"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
os.makedirs(CACHE, exist_ok=True)
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/s12c_gh"
os.makedirs(OUT, exist_ok=True)


def fetch(url, tries=3):
    key = hashlib.sha1(url.encode()).hexdigest()
    raw = os.path.join(CACHE, key + ".html")
    if os.path.exists(raw) and os.path.getsize(raw) > 2000:
        return open(raw, "rb").read().decode("utf-8", "replace")
    for i in range(tries):
        p = subprocess.run(["curl", "-sL", "--max-time", "30", "-A", UA, url], capture_output=True)
        data = p.stdout.decode("utf-8", "replace")
        if len(data) > 2000:
            open(raw, "w", encoding="utf-8").write(data)
            return data
        time.sleep(1 + i)
    return data


def gh(url):
    h = fetch(url)
    out = []
    m = re.search(r'<title>(.*?)</title>', h, re.S)
    if m:
        out.append("TITLE: " + html.unescape(strip_html(m.group(1))))
    for pat in [r'"state":"(open|closed|merged|draft)"', r'State--(\w+)']:
        for mm in re.findall(pat, h):
            out.append("STATE-TOKEN: " + mm)
        if out and out[-1].startswith("STATE-TOKEN"):
            break
    for lab, pat in [("MERGED_AT", r'"mergedAt":"([^"]+)"'),
                     ("CLOSED_AT", r'"closedAt":"([^"]+)"'),
                     ("CREATED_AT", r'"createdAt":"([^"]+)"')]:
        v = re.findall(pat, h)
        if v:
            out.append(f"{lab}: " + " | ".join(sorted(set(v))[:4]))
    labs = re.findall(r'"label":{"[^}]*?"name":"([^"]+)"', h)
    if labs:
        out.append("LABELS: " + ", ".join(sorted(set(labs))[:25]))
    for cr in set(re.findall(r'"stateReason":"([^"]+)"', h)):
        out.append("STATE_REASON: " + cr)
    bodies = []
    for m in re.finditer(r'<div class="comment-body[^"]*"[^>]*>(.*?)</div>\s*(?=<|$)', h, re.S):
        bodies.append(strip_html(m.group(1)))
    if not bodies:
        for m in re.finditer(r'"body":"((?:[^"\\]|\\.)*)"', h):
            try:
                bodies.append(json.loads('"' + m.group(1) + '"'))
            except Exception:
                pass
    seen = set()
    idx = 0
    for b in bodies:
        b = b.strip()
        if not b or b in seen:
            continue
        seen.add(b)
        idx += 1
        out.append(f"\n--- COMMENT {idx} ---\n{b}")
    if not bodies:
        out.append("\n[NO COMMENT BODIES EXTRACTED]")
    out.append("\n=== PAGE TEXT ===\n" + strip_html(re.sub(r'(?s)<div class="comment-body.*', '', h))[:20000])
    return "\n".join(out)


def run(url):
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    dest = os.path.join(OUT, key + ".txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        return url, "cached"
    try:
        txt = gh(url)
    except Exception as e:
        txt = "FETCH ERROR " + str(e)
    open(dest, "w").write("URL: " + url + "\n" + txt)
    return url, str(len(txt))


if __name__ == "__main__":
    urls = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for u, n in ex.map(run, urls):
            print(n, u[:120], flush=True)
