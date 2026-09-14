#!/usr/bin/env python3
"""S12c-paced: fetch GitHub /search and /pulls listing pages slowly (avoid HTTP 429).

Reads URLs from argv[1]; writes rendered listings to /tmp/s12c_paced/<sha>.txt
Detects 429 and backs off.
"""
import hashlib, os, re, subprocess, sys, time, html

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
CACHE = "/tmp/accelsrc"
OUT = "/tmp/s12c_paced"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def raw(url):
    key = hashlib.sha1(url.encode()).hexdigest()
    p = os.path.join(CACHE, key + ".html")
    if os.path.exists(p) and os.path.getsize(p) > 20000:
        return open(p, "rb").read().decode("utf-8", "replace"), "cached"
    return None, "miss"


def get(url, backoff=20):
    h, st = raw(url)
    if h:
        return h, st
    for attempt in range(6):
        pr = subprocess.run(["curl", "-sL", "--max-time", "30", "-A", UA, "-w", "\n%{http_code}",
                             url], capture_output=True)
        txt = pr.stdout.decode("utf-8", "replace")
        code = txt.rsplit("\n", 1)[-1].strip()
        body = txt.rsplit("\n", 1)[0]
        if code == "200" and len(body) > 20000:
            k = hashlib.sha1(url.encode()).hexdigest()
            open(os.path.join(CACHE, k + ".html"), "w", encoding="utf-8").write(body)
            return body, "200"
        time.sleep(backoff * (attempt + 1))
    return body if 'body' in dir() else "", "fail:" + code


def strip_html(s):
    s = re.sub(r"(?is)<(script|style|svg|noscript)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?is)</(p|div|li|tr|h[1-6]|pre|blockquote|span|a)>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", s)


def render(url, h):
    out = ["### " + url]
    if "/search?" in url:
        parts = h.split("Result-module__Result__")
        for p in parts[1:]:
            m = re.search(r'href="(/[^"/]+/[^"/]+/(?:issues|pull)/(\d+))"', p)
            if not m:
                continue
            t = re.search(r'href="/[^"]*/(?:issues|pull)/\d+"><span[^>]*>(.*?)</span></a>', p, re.S)
            title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", t.group(1)))).strip() if t else "?"
            c = re.search(r'Content-module__Content[^>]*>(.*?)</div>', p, re.S)
            exc = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c.group(1)))).strip() if c else ""
            out.append(f"{m.group(2)}\t{title}\n\tEXC: {exc[:400]}\n\thttps://github.com{m.group(1)}")
        if len(out) == 1:
            out.append("[NO RESULTS PARSED] " + strip_html(h)[:600].replace("\n", " "))
    elif "/pulls?" in url or "/issues?" in url:
        for m in re.finditer(r'<a id="issue_(\d+)_link"[^>]*href="(/[^"]+)"[^>]*>(.*?)</a>(.{0,2500}?)(?=<!-- Issue title column -->|</div>\s*</div>\s*</div>)', h, re.S):
            num, href, title, tail = m.group(1), m.group(2), m.group(3), m.group(4)
            title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", title))).strip()
            dates = re.findall(r'<relative-time datetime="([^"]+)"', tail)
            state = "closed" if ("was closed" in tail or "was merged" in tail) else "open"
            au = re.findall(r'class="[^"]*author[^"]*"[^>]*>([^<]+)<', tail)
            out.append(f"{num}\t{dates[:1]}\t{state}\t{title}\tby {au[:1]}\n\thttps://github.com{href}")
        if len(out) == 1:
            out.append("[NO RESULTS PARSED] " + strip_html(h)[:600].replace("\n", " "))
    else:
        out.append(strip_html(h)[:20000])
    return "\n".join(out)


if __name__ == "__main__":
    urls = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    for u in urls:
        k = hashlib.sha1(u.encode()).hexdigest()[:16]
        d = os.path.join(OUT, k + ".txt")
        if os.path.exists(d) and os.path.getsize(d) > 100:
            print("cached", u[:110], flush=True); continue
        h, st = get(u)
        open(d, "w").write(render(u, h))
        print(st, u[:110], flush=True)
        if st not in ("cached",):
            time.sleep(8)
