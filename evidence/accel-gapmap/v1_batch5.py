#!/usr/bin/env python3
"""V1 pass 5: unescape literal \\n / \\" inside claimed quotes before mechanical relocation.

Researchers wrote many quote fields on a single line using literal backslash-n to mark line
breaks.  verify.py's norm() does not translate those, so it reports MISMATCH on quotes whose
words are all present.  This pass unescapes first and reports both verdicts so the difference
between a rendering artefact and a text edit is visible.
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify as V

FILES = sys.argv[1:] or ["S1.md", "S2.md", "S3.md", "S4.md"]
BLOCK = re.compile(r"^###\s+([ABCDE][0-9.]*?)\s*\|\s*(CLOSED|OPEN|ABANDONED|HARDWARE-RULED-OUT|NEVER-DISCUSSED)\s*(.*)$")
KEY = re.compile(r"^([A-Z][A-Z0-9 _()/\.\->+]*?):\s?(.*)$")
RANK = {"MATCH": 3, "MATCH-FRAGMENTS": 2, "NEAR": 1}
SKIP_MINE = {"S3B", "REASON-MAY-HAVE-EXPIRED", "BORDERLINE_NOTE", "WHY_IT_IS_OUT_OF_REACH",
             "WHAT_IS_MISSING", "AUTHOR_CONTINUATION", "FINAL_RECOMMENDATION", "NOTE2"}


def strip_parens(s):
    return re.sub(r"\s*\([^)]*\)", "", s)


def norm_key(k):
    return re.sub(r"\s*\(.*?\)\s*", "", k).strip().replace(" ", "_")


def unesc(s):
    return (s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
             .replace("\\'", "'"))


def blocks(fn):
    cur = None
    last = None
    for raw in open(os.path.join(HERE, fn), encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        m = BLOCK.match(line)
        if m:
            if cur:
                yield cur
            cur = {"id": m.group(1), "section": m.group(2), "sub": m.group(3).strip(" ()"),
                   "file": fn, "fields": []}
            last = None
            continue
        if cur is None:
            continue
        probe = strip_parens(line)
        f = KEY.match(probe) if probe[:1].isupper() else None
        if f:
            last = [norm_key(f.group(1)), [f.group(2).strip()]]
            cur["fields"].append(last)
        elif last and line.strip() and line.strip() != "---":
            last[1].append(line.strip())
    if cur:
        yield cur


def get_prefix(fl, *prefixes):
    for p in prefixes:
        for k, v in fl:
            if k == p or k.startswith(p):
                return "\n".join(v).strip()
    return None


def primary(sec, fl):
    if sec == "OPEN":
        return get_prefix(fl, "ASKING_QUOTE")
    if sec == "ABANDONED":
        return get_prefix(fl, "STATED_REASON")
    if sec == "HARDWARE-RULED-OUT":
        return get_prefix(fl, "HARDWARE_QUOTE")
    if sec == "NEVER-DISCUSSED":
        return get_prefix(fl, "GAP")
    return get_prefix(fl, "QUOTE")


QUOTED = re.compile(r'"((?:[^"\\]|\\.){40,})"', re.S)
urlre = re.compile(r"https?://[^\s;,]+")
out = open(os.path.join(HERE, "v1_checkall5.jsonl"), "w", encoding="utf-8")


def check_best(urls, q):
    best = {"verdict": "MISMATCH", "ratio": 0.0}
    per = []
    for u in urls:
        try:
            r = V.check(u, q)
        except Exception as e:
            r = {"verdict": "FETCH-ERROR", "err": str(e)[:100]}
        per.append({"url": u[-46:], "v": r.get("verdict"), "ratio": r.get("ratio")})
        if RANK.get(r.get("verdict"), 0) > RANK.get(best.get("verdict"), 0) or (
                RANK.get(r.get("verdict"), 0) == RANK.get(best.get("verdict"), 0)
                and (r.get("ratio") or 0) > (best.get("ratio") or 0)):
            best = dict(r)
            best["url"] = u
    best["per_url"] = per
    return best


for fn in FILES:
    for b in blocks(fn):
        fl = b["fields"]
        urls = urlre.findall(get_prefix(fl, "URL") or "")
        cands = []
        pq = primary(b["section"], fl)
        if pq:
            cands.append(("PRIMARY", pq))
        for k, v in fl:
            if k in SKIP_MINE or k in ("QUESTION", "WHO", "ARTIFACT", "URL", "STATUS",
                                       "EVIDENCE", "WHO_IS_STILL_ASKING"):
                continue
            if k.startswith(("QUOTE", "ASKING_QUOTE", "STATED_REASON", "HARDWARE_QUOTE")):
                continue
            for s in QUOTED.findall("\n".join(v)):
                if len(s) >= 40:
                    cands.append((k, s))
        for tag, q in cands:
            raw_v = check_best(urls, q) if urls else {"verdict": "NO-URL"}
            un_v = check_best(urls, unesc(q)) if urls else {"verdict": "NO-URL"}
            rec = {"id": b["id"], "file": fn, "section": b["section"], "tag": tag,
                   "urls": urls, "quote": q[:600],
                   "verdict": raw_v.get("verdict"), "ratio": raw_v.get("ratio"),
                   "missing_tail": raw_v.get("missing_tail"),
                   "url_checked": raw_v.get("url"),
                   "verdict_unescaped": un_v.get("verdict"), "ratio_unescaped": un_v.get("ratio"),
                   "missing_tail_unescaped": un_v.get("missing_tail"),
                   "url_checked_unescaped": un_v.get("url"),
                   "per_url": un_v.get("per_url")}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{rec['verdict']:9s}->{rec['verdict_unescaped']:9s} "
                  f"{str(rec.get('ratio_unescaped')):6s} {fn:6s} {b['id']:8s} {tag[:20]:20s} "
                  f"{rec['quote'][:60]!r}", flush=True)
out.close()
