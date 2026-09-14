#!/usr/bin/env python3
"""S12c-v2: GitHub issue/PR extractor with per-comment AUTHOR attribution.

Parses the embedded JSON payload of the React issue page: for every "body":"..." occurrence,
walk backwards to the nearest "login":"..." to attribute the author.
Also extracts title/state/stateReason/closedAt/mergedAt/createdAt/labels.
"""
import concurrent.futures, hashlib, os, re, subprocess, sys, time, json, html

CACHE = "/tmp/accelsrc"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
os.makedirs(CACHE, exist_ok=True)


def fetch(url, tries=3, minlen=2000):
    key = hashlib.sha1(url.encode()).hexdigest()
    raw = os.path.join(CACHE, key + ".html")
    if os.path.exists(raw) and os.path.getsize(raw) > minlen:
        return open(raw, "rb").read().decode("utf-8", "replace")
    data = ""
    for i in range(tries):
        p = subprocess.run(["curl", "-sL", "--max-time", "30", "-A", UA, url], capture_output=True)
        data = p.stdout.decode("utf-8", "replace")
        if len(data) > minlen:
            open(raw, "w", encoding="utf-8").write(data)
            return data
        time.sleep(1 + i)
    return data


def jstr(h, start):
    """Decode a JSON string literal that starts at h[start] == '\"'."""
    i = start + 1
    buf = []
    while i < len(h):
        c = h[i]
        if c == "\\":
            buf.append(h[i:i + 2]); i += 2; continue
        if c == '"':
            break
        buf.append(c); i += 1
    try:
        return json.loads('"' + "".join(buf) + '"'), i
    except Exception:
        return "".join(buf), i


def extract(url):
    h = fetch(url)
    out = []
    out.append("URL: " + url)
    m = re.search(r'<title>(.*?)</title>', h, re.S)
    if m:
        out.append("TITLE: " + html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1)))).strip())
    for pat, lab in [(r'"state":"(OPEN|CLOSED|MERGED|DRAFT)"', "STATE"),
                     (r'"stateReason":"([^"]*)"', "STATE_REASON"),
                     (r'"mergedAt":"([^"]+)"', "MERGED_AT"),
                     (r'"closedAt":"([^"]+)"', "CLOSED_AT"),
                     (r'"createdAt":"([^"]+)"', "CREATED_AT")]:
        v = sorted(set(re.findall(pat, h)))
        if v:
            out.append(f"{lab}: " + " | ".join(v[:6]))
    labs = re.findall(r'"label":\{"[^}]*?"name":"([^"]+)"', h)
    if labs:
        out.append("LABELS: " + ", ".join(sorted(set(labs))[:20]))
    # body / comment bodies with author attribution
    seen = set()
    idx = 0
    for m in re.finditer(r'"body":"', h):
        s = m.end() - 1
        body, end = jstr(h, s)
        if not body or len(body) < 2:
            continue
        pre = h[max(0, m.start() - 2500):m.start()]
        au = re.findall(r'"login":"([^"]+)"', pre)
        author = au[-1] if au else "?"
        cre = re.findall(r'"createdAt":"([^"]+)"', pre)
        created = cre[-1] if cre else ""
        key = (author, body[:200])
        if key in seen:
            continue
        seen.add(key)
        idx += 1
        out.append(f"\n--- BODY {idx} | author={author} | created={created} ---\n{body}")
    if idx == 0:
        out.append("\n[NO BODIES EXTRACTED]")
    return "\n".join(out)


def run(url, outdir):
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    dest = os.path.join(outdir, key + ".txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        return url, "cached"
    try:
        txt = extract(url)
    except Exception as e:
        txt = "URL: %s\nFETCH ERROR %s" % (url, e)
    open(dest, "w").write(txt + "\n")
    return url, str(len(txt))


if __name__ == "__main__":
    urls = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/s12c_gh2"
    os.makedirs(outdir, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for u, n in ex.map(lambda u: run(u, outdir), urls):
            print(n, u[:120], flush=True)
