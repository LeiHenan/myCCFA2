#!/usr/bin/env python3
"""S12c-pr: GitHub issue/PR extractor that works for BOTH issue and PR pages.

- issues: embedded JSON `"body":"..."` with nearest preceding `"login":"..."`
- PRs (and issues too): `<clipboard-copy value="...">` blocks = raw markdown of each comment,
  with nearest preceding author link for attribution.
Emits: TITLE / STATE / STATE_REASON / timestamps / labels / every comment with author.
"""
import concurrent.futures, hashlib, os, re, subprocess, sys, time, json, html

CACHE = "/tmp/accelsrc"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
os.makedirs(CACHE, exist_ok=True)


def fetch(url, tries=4, minlen=5000):
    key = hashlib.sha1(url.encode()).hexdigest()
    raw = os.path.join(CACHE, key + ".html")
    if os.path.exists(raw) and os.path.getsize(raw) > 2000:
        return open(raw, "rb").read().decode("utf-8", "replace")
    data = ""
    for i in range(tries):
        p = subprocess.run(["curl", "-sL", "--max-time", "30", "-A", UA, url], capture_output=True)
        data = p.stdout.decode("utf-8", "replace")
        if len(data) > minlen:
            open(raw, "w", encoding="utf-8").write(data)
            return data
        time.sleep(2 + 2 * i)
    if data:
        open(raw, "w", encoding="utf-8").write(data)
    return data


def jstr(h, start):
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


def clean(x):
    x = re.sub(r"(?is)<(script|style|svg|noscript)[^>]*>.*?</\1>", " ", x)
    x = re.sub(r"(?is)<br\s*/?>", "\n", x)
    x = re.sub(r"(?is)</(p|div|li|tr|h[1-6]|pre|blockquote|td)>", "\n", x)
    x = re.sub(r"(?s)<[^>]+>", "", x)
    return re.sub(r"[ \t\xa0]+", " ", html.unescape(x)).strip()


def extract(url):
    h = fetch(url)
    out = ["URL: " + url]
    m = re.search(r"<title>(.*?)</title>", h, re.S)
    if m:
        out.append("TITLE: " + re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))).strip())
    for pat, lab in [(r'"state":"(OPEN|CLOSED|MERGED|DRAFT)"', "STATE"),
                     (r'"stateReason":"(?:")?([A-Z_]+)', "STATE_REASON"),
                     (r'"mergedAt":"([^"]+)"', "MERGED_AT"),
                     (r'"closedAt":"([^"]+)"', "CLOSED_AT"),
                     (r'"createdAt":"([^"]+)"', "CREATED_AT")]:
        v = sorted(set(re.findall(pat, h)))
        if v:
            out.append(f"{lab}: " + " | ".join(v[:6]))
    labs = re.findall(r'"label":\{"[^}]*?"name":"([^"]+)"', h)
    if labs:
        out.append("LABELS: " + ", ".join(sorted(set(labs))[:20]))
    # merged / closed / review markers present in the timeline markup
    for pat in ["Closed with unmerged commits", "merged this pull request", "closed this",
                "approved these changes", "reviewed changes", "requested changes",
                "This pull request has been automatically closed", "marked this pull request as ready for review"]:
        c = h.count(pat)
        if c:
            out.append(f"MARKER[{c}]: {pat}")

    seen, idx = set(), 0
    # path 1: clipboard-copy raw markdown (works on PR pages)
    for m in re.finditer(r"<clipboard-copy[^>]*?value=\"(.*?)\"[^>]*>", h, re.S):
        body = html.unescape(m.group(1))
        if len(body.strip()) < 2:
            continue
        pre = h[max(0, m.start() - 6000):m.start()]
        au = re.findall(r'class="author[^"]*"[^>]*href="/([^"/]+)"', pre) or \
             re.findall(r'href="/([^"/]+)"[^>]*class="author', pre) or \
             re.findall(r'data-hovercard-url="/users/([^"/]+)/hovercard"', pre)
        author = au[-1] if au else "?"
        ts = re.findall(r'datetime="([^"]+)"', pre)
        key = body[:200]
        if key in seen:
            continue
        seen.add(key); idx += 1
        out.append(f"\n--- C{idx} | author={author} | ts={ts[-1] if ts else ''} ---\n{body}")
    # path 2: issue JSON bodies
    if idx == 0:
        for m in re.finditer(r'"body":"', h):
            body, _ = jstr(h, m.end() - 1)
            if not body or len(body) < 2:
                continue
            pre = h[max(0, m.start() - 2500):m.start()]
            au = re.findall(r'"login":"([^"]+)"', pre)
            cre = re.findall(r'"createdAt":"([^"]+)"', pre)
            key = body[:200]
            if key in seen:
                continue
            seen.add(key); idx += 1
            out.append(f"\n--- C{idx} | author={au[-1] if au else '?'} | ts={cre[-1] if cre else ''} ---\n{body}")
    if idx == 0:
        out.append("\n[NO BODIES EXTRACTED]")
        out.append("\n=== PAGE TEXT ===\n" + clean(h)[:8000])
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
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/s12c_pr"
    os.makedirs(outdir, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        for u, n in ex.map(lambda u: run(u, outdir), urls):
            print(n, u[:120], flush=True)
