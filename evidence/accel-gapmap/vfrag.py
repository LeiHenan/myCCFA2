#!/usr/bin/env python3
"""Fragment-level adversarial quote checker.

The S*.md STATED_REASON / HARDWARE_QUOTE / QUOTE fields frequently mix verbatim
fragments with the researcher's own bracketed editorial framing ("[Source: ...]")
and use the S7 convention "frag1 ;; [n] frag2". A whole-field check therefore
returns a spurious MISMATCH. This splits a field into candidate verbatim fragments
(on ' ;; ' markers and on quoted spans) and checks each independently.

  python3 vfrag.py S6.md [block-id ...]
"""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import check
from vcode import check as codecheck, check_raw as rawcheck
from vcheck import parse_blocks, urls_of, QUOTEY


def fragments(field):
    """Yield candidate verbatim fragments from a block field."""
    out = []
    for part in re.split(r"\s*;;\s*(?:\[[^\]]*\])?\s*", field):
        part = part.strip()
        if not part:
            continue
        # strip a fully-wrapping pair of quotes
        for m in re.finditer(r'[“"]([^”"]{25,})[”"]', part):
            out.append(m.group(1).strip())
        if not re.search(r'[“"]', part):
            out.append(part)
    seen, res = set(), []
    for f in out:
        f = f.strip().strip('"').strip()
        # drop trailing bracketed annotation glued to the fragment
        f = re.sub(r"\s*\[(Source|verbatim|the author|from the|quoted|footnote)[^\]]*\]\s*$", "", f).strip()
        if len(f) >= 25 and f not in seen:
            seen.add(f)
            res.append(f)
    return res


def main():
    path = sys.argv[1]
    want = set(sys.argv[2:])
    blocks = parse_blocks(path)
    for b in blocks:
        if want and b["id"] not in want:
            continue
        us = urls_of(b)
        print(f"\n##### {b['id']} | {b['section']}")
        for k in b["order"]:
            if not QUOTEY.match(k):
                continue
            field = b["fields"][k]
            if len(field) < 20:
                continue
            frags = fragments(field)
            print(f"  [{k}] {len(frags)} fragment(s)")
            for f in frags:
                results = []
                for u in us:
                    try:
                        r = check(u, f)
                    except Exception as e:
                        r = {"verdict": "ERR", "err": str(e)[:100]}
                    results.append((u, r))
                best = max(results, key=lambda t: (t[1].get("verdict") == "MATCH",
                                                   t[1].get("verdict") == "MATCH-FRAGMENTS",
                                                   t[1].get("ratio", 0)))
                v = best[1]
                code_ok = None
                if v.get("verdict") not in ("MATCH", "MATCH-FRAGMENTS"):
                    for u in us:
                        try:
                            c = codecheck(u, f)
                        except Exception:
                            c = {"verdict": "NOT-FOUND"}
                        if c.get("verdict") == "MATCH-CODE-NORMALISED":
                            code_ok = u
                            break
                raw_ok = None
                if not code_ok and v.get("verdict") not in ("MATCH", "MATCH-FRAGMENTS"):
                    for u in us:
                        try:
                            rr = rawcheck(u, f)
                        except Exception:
                            rr = {"verdict": "NOT-FOUND-RAW"}
                        if rr.get("verdict") == "MATCH-RAW":
                            raw_ok = u
                            break
                mark = "OK " if (v.get("verdict") in ("MATCH", "MATCH-FRAGMENTS")
                                 or code_ok or raw_ok) else "!! "
                tag = "MATCH" if v.get("verdict") in ("MATCH", "MATCH-FRAGMENTS") else (
                    "CODE-NORM" if code_ok else ("RAW" if raw_ok else v.get("verdict")))
                print(f"    {mark}{tag:16s} r={v.get('ratio')} {f[:90]!r}")
                if mark == "!! ":
                    print(f"        tail: {v.get('missing_tail','')!r}")
                    print(f"        per-url: " + " | ".join(
                        f"{u.split('/')[-1][:22]}:{r.get('verdict')}:{r.get('ratio')}"
                        for u, r in results))


if __name__ == "__main__":
    main()
