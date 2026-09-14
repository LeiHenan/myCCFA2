#!/usr/bin/env python3
"""Show exactly where a claimed quote diverges from the retrieved page.

  python3 vdiff.py <url> "<quote>"
"""
import difflib, json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import norm, corpus, strip_wrapping_quotes


def main():
    url, quote = sys.argv[1], sys.argv[2]
    q = norm(strip_wrapping_quotes(quote))
    best = None
    for name, text in corpus(url):
        if not text:
            continue
        if q in text:
            print(json.dumps({"verdict": "MATCH", "variant": name}))
            return
        sm = difflib.SequenceMatcher(None, q, text, autojunk=False)
        m = sm.find_longest_match(0, len(q), 0, len(text))
        ratio = m.size / max(1, len(q))
        if best is None or ratio > best[0]:
            best = (ratio, name, m, text)
    ratio, name, m, text = best
    print(json.dumps({"verdict": "NEAR" if ratio > 0.5 else "MISMATCH",
                      "ratio": round(ratio, 3), "variant": name,
                      "matched_head": q[:m.a + m.size][-120:],
                      "missing_tail": q[m.a + m.size:][:200]}))
    lo = max(0, m.b - 100)
    hi = min(len(text), m.b + m.size + 260)
    print("SOURCE WINDOW:", repr(text[lo:hi]))


if __name__ == "__main__":
    main()
