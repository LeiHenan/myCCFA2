#!/usr/bin/env python3
"""V1 pass 3: correct multi-line field handling.

A line starts a new field iff, after removing parenthetical groups, it matches
^[A-Z][A-Z0-9 _()/.\->+]*?:  (no lowercase in the key).  Lines that are quote
continuations (code, prose, tables, bullets) never match.
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify as V

FILES = sys.argv[1:] or ["S1.md", "S2.md", "S3.md", "S4.md"]
BLOCK = re.compile(r"^###\s+([ABCDE][0-9.]*?)\s*\|\s*(CLOSED|OPEN|ABANDONED|HARDWARE-RULED-OUT|NEVER-DISCUSSED)\s*(.*)$")
KEY = re.compile(r"^([A-Z][A-Z0-9 _()/\.\->\+]*?):\s?(.*)$")


def strip_parens(s):
    return re.sub(r"\s*\([^)]*\)", "", s)


def norm_key(k):
    return re.sub(r"\s*\(.*?\)\s*", "", k).strip().replace(" ", "_")


def blocks(fn):
    cur = None
    lastkey = None
    for raw in open(os.path.join(HERE, fn), encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        m = BLOCK.match(line)
        if m:
            if cur:
                yield cur
            cur = {"id": m.group(1), "section": m.group(2), "sub": m.group(3).strip(" ()"),
                   "file": fn, "fields": {}}
            lastkey = None
            continue
        if cur is None:
            continue
        probe = strip_parens(line)
        f = KEY.match(probe) if probe[:1].isupper() else None
        if f:
            k = norm_key(f.group(1))
            cur["fields"].setdefault(k, [])
            cur["fields"][k].append(f.group(2).strip())
            lastkey = k
        elif lastkey and line.strip() and line.strip() != "---":
            cur["fields"][lastkey].append(line.strip())
    if cur:
        yield cur


def flat(cur):
    return {k: "\n".join(v).strip() for k, v in cur["fields"].items()}


def pick(fl, *names):
    for n in names:
        if fl.get(n):
            return fl[n]
    for k in fl:
        for n in names:
            if k.startswith(n):
                return fl[k]
    return None


def quote_field(sec, fl):
    if sec == "OPEN":
        return pick(fl, "ASKING_QUOTE")
    if sec == "ABANDONED":
        return pick(fl, "STATED_REASON")
    if sec == "HARDWARE-RULED-OUT":
        return pick(fl, "HARDWARE_QUOTE")
    if sec == "NEVER-DISCUSSED":
        return pick(fl, "GAP")
    return pick(fl, "QUOTE")


urlre = re.compile(r"https?://[^\s;,]+")
out = open(os.path.join(HERE, "v1_checkall3.jsonl"), "w", encoding="utf-8")
RANK = {"MATCH": 3, "MATCH-FRAGMENTS": 2, "NEAR": 1}
for fn in FILES:
    for b in blocks(fn):
        fl = flat(b)
        q = quote_field(b["section"], fl)
        urls = urlre.findall(fl.get("URL", ""))
        rec = {"id": b["id"], "file": fn, "section": b["section"], "sub": b["sub"],
               "urls": urls, "quote": (q or "")[:1200], "all_keys": sorted(set(fl))}
        if not q:
            rec["verdict"] = "NO-QUOTE"
        elif not urls:
            rec["verdict"] = "NO-URL"
        else:
            best = {"verdict": "MISMATCH", "ratio": 0.0}
            per = []
            for u in urls:
                try:
                    r = V.check(u, q)
                except Exception as e:
                    r = {"verdict": "FETCH-ERROR", "err": str(e)[:120]}
                per.append({"url": u, "v": r.get("verdict"), "ratio": r.get("ratio"),
                            "tail": (r.get("missing_tail") or "")[:150]})
                if RANK.get(r.get("verdict"), 0) > RANK.get(best.get("verdict"), 0) or (
                        RANK.get(r.get("verdict"), 0) == RANK.get(best.get("verdict"), 0)
                        and (r.get("ratio") or 0) > (best.get("ratio") or 0)):
                    best = dict(r)
                    best["url"] = u
            rec.update({"verdict": best.get("verdict"), "variant": best.get("variant"),
                        "ratio": best.get("ratio"), "missing_tail": best.get("missing_tail"),
                        "url_checked": best.get("url"), "per_url": per})
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{rec['verdict']:18s} {rec.get('ratio')!s:6s} {fn:6s} {b['id']:8s} "
              f"{b['section']:18s} {(rec.get('url_checked') or '')[:58]}", flush=True)
out.close()
