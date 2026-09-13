#!/usr/bin/env python3
"""Fetch a GitHub issue/PR HTML page and extract readable text (body + comments).

Usage: gh_read.py <url> [--json-out PATH] [--max-chars N]
Prints:
  - header info (title, state, labels if discoverable)
  - stripped text of the embeddedData payload
"""
import sys, json, re, html, subprocess, os, gzip, io

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def fetch(url):
    cmd = ["curl", "-sL", "-m", "60", "--compressed", "-A", UA,
           "-H", "Accept: text/html,application/xhtml+xml",
           "-H", "Accept-Language: en-US,en;q=0.9", url]
    p = subprocess.run(cmd, capture_output=True)
    return p.stdout.decode("utf-8", "replace")


def extract_embedded(page):
    """Return list of (target, json_obj_or_raw)."""
    out = []
    for m in re.finditer(
        r'<script type="application/json" data-target="([^"]+)">(.*?)</script>',
        page, re.S):
        target, raw = m.group(1), m.group(2)
        try:
            out.append((target, json.loads(raw)))
        except Exception:
            out.append((target, raw))
    return out


TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
BR_RE = re.compile(r"<br\s*/?>", re.I)
BLOCK_RE = re.compile(r"</(p|div|li|h[1-6]|tr|pre|blockquote)>", re.I)


def strip_tags(s):
    if s is None:
        return ""
    s = TAG_RE.sub(" ", s)
    s = BR_RE.sub("\n", s)
    s = BLOCK_RE.sub("\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def walk_strings(obj, path=""):
    """Yield (path, string) for every string in nested json."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def main():
    url = sys.argv[1]
    maxchars = 200000
    if "--max-chars" in sys.argv:
        maxchars = int(sys.argv[sys.argv.index("--max-chars") + 1])
    raw_out = None
    if "--raw-out" in sys.argv:
        raw_out = sys.argv[sys.argv.index("--raw-out") + 1]

    page = fetch(url)
    if raw_out:
        with open(raw_out, "w") as f:
            f.write(page)

    if not page:
        print("FETCH FAILED (empty response)")
        return 1

    emb = extract_embedded(page)
    if not emb:
        print("NO embeddedData FOUND. page length =", len(page))
        print("TITLE TAG:", (re.search(r"<title>(.*?)</title>", page, re.S) or [None, "?"])[1])
        return 2

    printed = 0
    for target, obj in emb:
        print(f"\n########## embeddedData target={target} ##########")
        if isinstance(obj, str):
            print(strip_tags(obj)[:maxchars])
            printed += 1
            continue
        for path, s in walk_strings(obj):
            t = strip_tags(s)
            if len(t) < 2:
                continue
            # skip obvious noise
            print(f"--- {path} ---")
            print(t[: maxchars // 3])
            printed += 1
            if printed > 400:
                print("... TRUNCATED ...")
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
