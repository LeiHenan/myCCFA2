#!/usr/bin/env python3
"""S12 helper: GitHub issue/PR fetch with per-comment AUTHOR + STATE.

Usage: python3 s12_gh.py <url> [--full]
"""
import sys, re, json, html
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl


def strip(s):
    s = re.sub(r"(?is)<(script|style|svg|noscript)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</(p|div|li|tr|h[1-6]|pre|blockquote|td)>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def main(url):
    h = curl(url)
    out = []
    m = re.search(r"<title>(.*?)</title>", h, re.S)
    if m:
        out.append("TITLE: " + strip(m.group(1)))
    out.append("URL: " + url)
    st = set(re.findall(r'"state":"(open|closed|merged|draft|OPEN|CLOSED|MERGED)"', h))
    # PR header state
    st |= set(re.findall(r'State--(open|closed|merged|draft)', h))
    out.append("STATE-TOKEN: " + (", ".join(sorted(st)) if st else "?"))
    for lab, pat in [("MERGED_AT", r'"mergedAt":"([^"]+)"'),
                     ("CLOSED_AT", r'"closedAt":"([^"]+)"'),
                     ("CREATED_AT", r'"createdAt":"([^"]+)"')]:
        v = sorted(set(re.findall(pat, h)))
        if v:
            out.append(f"{lab}: " + " | ".join(v[:3]))
    for cr in sorted(set(re.findall(r'"stateReason":"([^"]+)"', h))):
        out.append("STATE_REASON: " + cr)
    labs = re.findall(r'"label":\{"[^}]*?"name":"([^"]+)"', h)
    if labs:
        out.append("LABELS: " + ", ".join(sorted(set(labs))[:30]))

    # comments: locate each comment-body, look BACKWARD for author login
    bodies = []
    for m in re.finditer(r'<div class="comment-body[^"]*"[^>]*>(.*?)</div>\s*(?=<|$)', h, re.S):
        pre = h[max(0, m.start() - 20000):m.start()]
        au = re.findall(r'"login":"([^"]+)"', pre)
        name = re.findall(r'data-hovercard-type="user"[^>]*href="/([^"/]+)"', pre)
        author = au[-1] if au else (name[-1] if name else "?")
        bodies.append((author, strip(m.group(1))))
    if not bodies:
        # fallback: embedded JSON payload bodies
        for m in re.finditer(r'"body":"((?:[^"\\]|\\.)*)"', h):
            try:
                b = json.loads('"' + m.group(1) + '"')
            except Exception:
                continue
            pre = h[max(0, m.start() - 4000):m.start()]
            au = re.findall(r'"login":"([^"]+)"', pre)
            bodies.append((au[-1] if au else "?", b))
    seen = set()
    i = 0
    for a, b in bodies:
        b = b.strip()
        if not b or b in seen:
            continue
        seen.add(b)
        i += 1
        out.append(f"\n--- COMMENT {i} [author: {a}] ---\n{b}")
    if not bodies:
        out.append("\n[NO COMMENT BODIES EXTRACTED]")
    return "\n".join(out)


if __name__ == "__main__":
    print(main(sys.argv[1]))
