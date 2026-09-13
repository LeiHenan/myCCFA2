#!/usr/bin/env python3
"""Shared fetch/extract helper for the inference-accel gap map.

Usage:
  python3 fetch.py get <url>                 # raw text (html->text, preserves JSON payload lines)
  python3 fetch.py gh <url>                  # GitHub issue/PR: title, state, timeline, all comment bodies
  python3 fetch.py arxiv <id-or-url>         # arXiv abs page: title, authors, abstract, dates
  python3 fetch.py raw <url>                 # raw bytes to stdout (binary-safe-ish)

Caching: raw HTML cached under /tmp/accelsrc/<sha1>.html ; text under /tmp/accelsrc/<sha1>.txt
Retries: curl -sL --retry 4 --retry-delay 2 --retry-all-errors with a browser UA (the only
reliable path in this environment; bare curl and the web_fetch tool are unreliable here).
"""
import hashlib, html, os, re, subprocess, sys, json

CACHE = "/tmp/accelsrc"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
os.makedirs(CACHE, exist_ok=True)


def curl(url, timeout=90):
    key = hashlib.sha1(url.encode()).hexdigest()
    raw = os.path.join(CACHE, key + ".html")
    if os.path.exists(raw) and os.path.getsize(raw) > 500:
        return open(raw, "rb").read().decode("utf-8", "replace")
    p = subprocess.run(
        ["curl", "-sL", "--retry", "4", "--retry-delay", "2", "--retry-all-errors",
         "--max-time", str(timeout), "-A", UA, url],
        capture_output=True)
    data = p.stdout.decode("utf-8", "replace")
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


def gh(url):
    h = curl(url)
    out = []
    m = re.search(r'<title>(.*?)</title>', h, re.S)
    if m:
        out.append("TITLE: " + html.unescape(strip_html(m.group(1))))
    # state
    for pat in [r'"state":"(open|closed|merged|draft)"', r'State--(\w+)']:
        for mm in re.findall(pat, h):
            out.append("STATE-TOKEN: " + mm)
        if out and out[-1].startswith("STATE-TOKEN"):
            break
    # merged / closed timestamps embedded in payload
    for lab, pat in [("MERGED_AT", r'"mergedAt":"([^"]+)"'),
                     ("CLOSED_AT", r'"closedAt":"([^"]+)"'),
                     ("CREATED_AT", r'"createdAt":"([^"]+)"')]:
        v = re.findall(pat, h)
        if v:
            out.append(f"{lab}: " + " | ".join(sorted(set(v))[:4]))
    # labels
    labs = re.findall(r'"label":{"[^}]*?"name":"([^"]+)"', h)
    if labs:
        out.append("LABELS: " + ", ".join(sorted(set(labs))[:25]))
    # close reason
    for cr in set(re.findall(r'"stateReason":"([^"]+)"', h)):
        out.append("STATE_REASON: " + cr)
    # comment bodies, two independent extraction paths
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
    out.append("\n=== PAGE TEXT (non-comment) ===\n" +
               strip_html(re.sub(r'(?s)<div class="comment-body.*', '', h)))
    return "\n".join(out)


def arxiv(url):
    m = re.search(r'(\d{4}\.\d{4,5})', url)
    aid = m.group(1) if m else url
    h = curl("https://arxiv.org/abs/" + aid)
    out = [f"ARXIV_ID_READBACK: {aid}", f"URL: https://arxiv.org/abs/{aid}"]
    t = re.search(r'(?s)<h1 class="title[^"]*">(.*?)</h1>', h)
    if t:
        out.append("TITLE: " + strip_html(t.group(1)).replace("Title:", "").strip())
    a = re.search(r'(?s)<div class="authors">(.*?)</div>', h)
    if a:
        out.append("AUTHORS: " + re.sub(r"\s+", " ", strip_html(a.group(1))).strip())
    ab = re.search(r'(?s)<blockquote class="abstract[^"]*">(.*?)</blockquote>', h)
    if ab:
        out.append("ABSTRACT: " + strip_html(ab.group(1)).replace("Abstract:", "").strip())
    d = re.search(r'(?s)<div class="dateline">(.*?)</div>', h)
    if d:
        out.append("DATELINE: " + re.sub(r"\s+", " ", strip_html(d.group(1))).strip())
    sub = re.findall(r'(?s)<td class="tablecell ([a-z]+)">(.*?)</td>', h)
    for k, v in sub:
        out.append(f"{k.upper()}: " + re.sub(r"\s+", " ", strip_html(v)).strip())
    if "TITLE:" not in "\n".join(out):
        out.append("[ARXIV PAGE DID NOT PARSE — raw text follows]\n" + strip_html(h)[:4000])
    return "\n".join(out)


if __name__ == "__main__":
    mode, url = sys.argv[1], sys.argv[2]
    if mode == "get":
        print(strip_html(curl(url)))
    elif mode == "raw":
        sys.stdout.write(curl(url))
    elif mode == "gh":
        print(gh(url))
    elif mode == "arxiv":
        print(arxiv(url))
    else:
        sys.exit("unknown mode " + mode)
