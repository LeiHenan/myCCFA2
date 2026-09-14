#!/usr/bin/env python3
"""V1 pass 2: multi-line-safe block parser; check each block's quote against EVERY url in URL:."""
import json, os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify as V

FILES = sys.argv[1:] or ["S1.md", "S2.md", "S3.md", "S4.md"]

BLOCK = re.compile(r"^###\s+([ABCDE][0-9.]*?)\s*\|\s*(CLOSED|OPEN|ABANDONED|HARDWARE-RULED-OUT|NEVER-DISCUSSED)\s*(.*)$")
NEWFIELD = re.compile(r"^([A-Z][A-Z0-9 _()/\.\->\+]{1,60}?):\s?(.*)$")
# field names that legitimately start a new field (avoid swallowing quote continuation lines)
KNOWN = None


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
        f = NEWFIELD.match(line) if line[:1].isupper() else None
        if f:
            k = norm_key(f.group(1))
            cur["fields"].setdefault(k, [])
            cur["fields"][k].append(f.group(2).strip())
            lastkey = k
        elif lastkey and line.strip():
            cur["fields"][lastkey].append(line.strip())
    if cur:
        yield cur


def flat(cur):
    return {k: "\n".join(v).strip() for k, v in cur["fields"].items()}


def pick(f, *names):
    for n in names:
        if f.get(n):
            return f[n]
    for k in f:
        for n in names:
            if k.startswith(n):
                return f[k]
    return None


def quote_field(sec, f):
    if sec == "OPEN":
        return pick(f, "ASKING_QUOTE", "QUOTE")
    if sec == "ABANDONED":
        return pick(f, "STATED_REASON")
    if sec == "HARDWARE-RULED-OUT":
        return pick(f, "HARDWARE_QUOTE")
    if sec == "NEVER-DISCUSSED":
        return pick(f, "GAP")
    return pick(f, "QUOTE")


urlre = re.compile(r"https?://[^\s;,]+")
out = open(os.path.join(HERE, "v1_checkall2.jsonl"), "w", encoding="utf-8")
for fn in FILES:
    for b in blocks(fn):
        f = flat(b)
        q = quote_field(b["section"], f)
        urls = urlre.findall(f.get("URL", ""))
        rec = {"id": b["id"], "file": fn, "section": b["section"], "sub": b["sub"],
               "urls": urls, "quote": (q or "")[:600],
               "all_keys": sorted(set(f))}
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
                            "tail": (r.get("missing_tail") or "")[:120]})
                rank = {"MATCH": 3, "MATCH-FRAGMENTS": 2, "NEAR": 1}.get(r.get("verdict"), 0)
                if rank > {"MATCH": 3, "MATCH-FRAGMENTS": 2, "NEAR": 1}.get(best.get("verdict"), 0) \
                   or (rank == {"MATCH": 3, "MATCH-FRAGMENTS": 2, "NEAR": 1}.get(best.get("verdict"), 0)
                       and (r.get("ratio") or 0) > (best.get("ratio") or 0)):
                    best = r
                    best["url"] = u
            rec.update({"verdict": best.get("verdict"), "variant": best.get("variant"),
                        "ratio": best.get("ratio"), "missing_tail": best.get("missing_tail"),
                        "url_checked": best.get("url"), "per_url": per})
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{rec['verdict']:18s} {rec.get('ratio')!s:6s} {fn:6s} {b['id']:8s} "
              f"{b['section']:18s} {(rec.get('url_checked') or '')[:60]}", flush=True)
out.close()
