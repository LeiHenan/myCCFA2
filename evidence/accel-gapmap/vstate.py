#!/usr/bin/env python3
"""Adversarial state reader: parse the authoritative embedded JSON payload of a
GitHub PR/issue page instead of trusting the loose STATE-TOKEN regexes in fetch.py.

  python3 vstate.py <url> [...]

Prints, per URL: number, kind, title, state, stateReason, mergedTime/mergedAt,
closedAt, createdAt, labels, plus any cross-referenced numbers.
"""
import json, re, sys, html as H

sys.path.insert(0, ".")
from fetch import curl

JSON_BLOB = re.compile(
    r'<script type="application/json"[^>]*>(.*?)</script>', re.S)


def payloads(h):
    out = []
    for m in JSON_BLOB.finditer(h):
        try:
            out.append(json.loads(H.unescape(m.group(1))))
        except Exception:
            pass
    # also the big react-app embeddedData may be escaped inside; try raw find
    return out


def walk(o, hits):
    if isinstance(o, dict):
        if "state" in o and ("number" in o or "title" in o):
            hits.append(o)
        for v in o.values():
            walk(v, hits)
    elif isinstance(o, list):
        for v in o:
            walk(v, hits)


def report(url):
    h = curl(url)
    print("=" * 100)
    print("URL:", url, "| bytes:", len(h))
    hits = []
    for p in payloads(h):
        walk(p, hits)
    seen = set()
    for d in hits:
        key = (d.get("number"), d.get("state"), d.get("title"))
        if key in seen:
            continue
        seen.add(key)
        if d.get("number") is None and d.get("title") is None:
            continue
        keep = {k: d.get(k) for k in
                ("number", "state", "stateReason", "mergedTime", "mergedAt",
                 "closedAt", "createdAt", "title", "isDraft", "merged",
                 "isInMergeQueue", "author") if k in d}
        # labels
        labs = d.get("labels")
        if isinstance(labs, list):
            try:
                keep["labels"] = [l.get("name") if isinstance(l, dict) else l for l in labs][:20]
            except Exception:
                pass
        au = keep.pop("author", None)
        if isinstance(au, dict):
            keep["author"] = au.get("login")
        print(json.dumps(keep, ensure_ascii=False)[:900])
    # explicit merged-token evidence
    mt = re.findall(r'"mergedTime":"([^"]+)"', h)
    if mt:
        print("MERGED_TIME_TOKENS:", sorted(set(mt))[:5])
    sr = re.findall(r'"stateReason":"([^"]+)"', h)
    if sr:
        print("STATE_REASON_TOKENS:", sorted(set(sr))[:5])


if __name__ == "__main__":
    for u in sys.argv[1:]:
        try:
            report(u)
        except Exception as e:
            print("ERR", u, repr(e))
