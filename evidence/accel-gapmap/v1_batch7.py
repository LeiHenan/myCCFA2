#!/usr/bin/env python3
"""V1 pass 7: adds a RAW corpus variant (no HTML stripping) for raw.githubusercontent /
raw file URLs, because verify.py's page-text variant deletes everything between a stray
'<' and a later '>' in source files, producing false MISMATCHes.
"""
import html as _html, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify as V
from fetch import curl, strip_html, gh

FILES = sys.argv[1:] or ["S1.md", "S2.md", "S3.md", "S4.md"]
BLOCK = re.compile(r"^###\s+([ABCDE][0-9.]*?)\s*\|\s*(CLOSED|OPEN|ABANDONED|HARDWARE-RULED-OUT|NEVER-DISCUSSED)\s*(.*)$")
KEY = re.compile(r"^([A-Z][A-Z0-9 _/.\->+]*?(?:\s*\([^)]*\))?)\s*:\s?(.*)$")
RANK = {"MATCH": 4, "MATCH-FRAGMENTS": 3, "NEAR": 2, "MISMATCH": 1}
SKIP_MINE = {"S3B", "REASON-MAY-HAVE-EXPIRED", "BORDERLINE_NOTE", "WHY_IT_IS_OUT_OF_REACH",
             "WHAT_IS_MISSING", "AUTHOR_CONTINUATION", "FINAL_RECOMMENDATION"}


def norm_key(k):
    return re.sub(r"\s*\(.*?\)\s*", "", k).strip().replace(" ", "_")


def unesc(s):
    return s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")


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
        f = KEY.match(line) if line[:1].isupper() else None
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


def variants(url):
    out = []
    try:
        out.append(("rawbytes", V.norm(_html.unescape(curl(url)))))
    except Exception:
        pass
    if "github.com" in url:
        try:
            out.append(("gh", V.norm(gh(url))))
        except Exception:
            pass
    try:
        out.append(("stripped", V.norm(strip_html(curl(url)))))
    except Exception:
        pass
    return out


def judge(url, q):
    qq = V.norm(V.strip_wrapping_quotes(q))
    if not qq:
        return {"verdict": "EMPTY"}
    best = {"verdict": "MISMATCH", "ratio": 0.0}
    import difflib
    for name, text in variants(url):
        if not text:
            continue
        if qq in text:
            return {"verdict": "MATCH", "variant": name, "ratio": 1.0, "url": url}
        parts = [p for p in re.split(r"\s*(?:\.\.\.|\[\.\.\.\]|…)\s*", qq) if len(p) > 12]
        if len(parts) > 1 and all(p in text for p in parts):
            return {"verdict": "MATCH-FRAGMENTS", "variant": name, "ratio": 1.0, "url": url}
        sm = difflib.SequenceMatcher(None, qq, text, autojunk=False)
        m = sm.find_longest_match(0, len(qq), 0, len(text))
        ratio = m.size / max(1, len(qq))
        if ratio > best.get("ratio", 0):
            best = {"verdict": "NEAR" if ratio > 0.75 else "MISMATCH",
                    "variant": name, "ratio": round(ratio, 3),
                    "missing_head": qq[:m.a][:200],
                    "missing_tail": qq[m.a + m.size:][:200], "url": url}
    return best


QUOTED = re.compile(r'"((?:[^"\\]|\\.){40,})"', re.S)
urlre = re.compile(r"https?://[^\s;,]+")
out = open(os.path.join(HERE, "v1_checkall7.jsonl"), "w", encoding="utf-8")
summ = {}
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
            best = {"verdict": "NO-URL"}
            per = []
            for u in urls:
                r = judge(u, unesc(q))
                per.append({"url": u[-44:], "v": r.get("verdict"), "ratio": r.get("ratio"),
                            "variant": r.get("variant")})
                if RANK.get(r.get("verdict"), 0) > RANK.get(best.get("verdict"), 0) or (
                        RANK.get(r.get("verdict"), 0) == RANK.get(best.get("verdict"), 0)
                        and (r.get("ratio") or 0) > (best.get("ratio") or 0)):
                    best = dict(r)
            rec = {"id": b["id"], "file": fn, "section": b["section"], "tag": tag,
                   "urls": urls, "quote": q[:600], "verdict": best.get("verdict"),
                   "ratio": best.get("ratio"), "variant": best.get("variant"),
                   "missing_head": best.get("missing_head"), "missing_tail": best.get("missing_tail"),
                   "url_checked": best.get("url"), "per_url": per}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{rec['verdict']:18s} {str(rec.get('ratio')):6s} {fn:6s} {b['id']:8s} "
                  f"{tag[:20]:20s} {rec['quote'][:64]!r}", flush=True)
out.close()
