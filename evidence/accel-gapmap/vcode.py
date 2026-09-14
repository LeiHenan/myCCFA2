#!/usr/bin/env python3
"""Secondary (code-aware) verbatim check.

Python source quotes in the gap map join adjacent string literals and comment
lines with the interleaved `"`, `#`, backslashes and indentation elided. That is
a *reconstruction*, not a continuous run, so verify.py returns MISMATCH even
though no word was changed. This checker deletes the characters that Python
syntax/comment markers introduce (`"`, `#`, `\\`, whitespace, and line-continuation
backslashes) from BOTH the page text and the quote and then tests substring
containment. It still FAILS if any word was added, removed or reordered.

  python3 vcode.py <url> "<quote fragment>"
"""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import curl, strip_html, gh
from verify import norm

STRIP = re.compile(r'["#\\\s]')


def code_norm(s):
    s = norm(s)
    return STRIP.sub("", s)


def variants(url):
    out = []
    try:
        out.append(("page-text", code_norm(strip_html(curl(url)))))
    except Exception as e:
        out.append(("ERR", ""))
    # strip_html is destructive on source files: a stray '<' followed much later by a
    # '>' deletes the whole intervening span. raw-text is the un-mangled variant.
    try:
        out.append(("raw-text", code_norm(curl(url))))
    except Exception:
        pass
    if "github.com" in url:
        try:
            out.append(("gh-extract", code_norm(gh(url))))
        except Exception:
            pass
    return out


def check_raw(url, quote):
    """Whitespace-normalised continuous-run check against the UN-stripped page text."""
    q = norm(quote)
    if q and q in norm(curl(url)):
        return {"verdict": "MATCH-RAW", "variant": "raw-text"}
    return {"verdict": "NOT-FOUND-RAW"}


def check(url, quote):
    q = code_norm(quote)
    for name, t in variants(url):
        if q and q in t:
            return {"verdict": "MATCH-CODE-NORMALISED", "variant": name}
    return {"verdict": "NOT-FOUND", "chars": len(q)}


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1], sys.argv[2]), ensure_ascii=False))
