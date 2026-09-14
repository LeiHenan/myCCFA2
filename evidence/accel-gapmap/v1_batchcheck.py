#!/usr/bin/env python3
"""V1 adversarial verifier: mechanically re-locate EVERY claimed quote in S1..S4."""
import json, os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify as V

FILES = sys.argv[1:] or ["S1.md", "S2.md", "S3.md", "S4.md"]
out = open(os.path.join(HERE, "v1_checkall.jsonl"), "w", encoding="utf-8")

BLOCK = re.compile(r"^###\s+([ABCDE][0-9.]*?)\s*\|\s*(CLOSED|OPEN|ABANDONED|HARDWARE-RULED-OUT|NEVER-DISCUSSED)\s*(.*)$")
FIELD = re.compile(r"^([A-Z][A-Z0-9 _\(\)/\.\->\+]*?):\s?(.*)$")


def blocks(fn):
    cur = None
    for line in open(os.path.join(HERE, fn), encoding="utf-8", errors="replace"):
        m = BLOCK.match(line.rstrip("\n"))
        if m:
            if cur:
                yield cur
            cur = {"id": m.group(1), "section": m.group(2), "sub": m.group(3).strip(" ()"),
                   "file": fn, "fields": {}}
            continue
        if cur is None:
            continue
        f = FIELD.match(line)
        if f:
            k = f.group(1).strip().replace(" ", "_")
            cur["fields"][k] = (cur["fields"].get(k, "") + " " + f.group(2).strip()).strip()
        elif line.strip() and not line.startswith("#"):
            cur["fields"]["_extra"] = (cur["fields"].get("_extra", "") + " " + line.strip()).strip()
    if cur:
        yield cur


def quote_field(sec, f):
    if sec == "OPEN":
        return f.get("ASKING_QUOTE") or f.get("QUOTE")
    if sec == "ABANDONED":
        return f.get("STATED_REASON")
    if sec == "HARDWARE-RULED-OUT":
        return f.get("HARDWARE_QUOTE")
    if sec == "NEVER-DISCUSSED":
        return f.get("GAP")
    return f.get("QUOTE")


urlre = re.compile(r"https?://[^\s;,]+")

for fn in FILES:
    for b in blocks(fn):
        q = quote_field(b["section"], b["fields"])
        urls = urlre.findall(b["fields"].get("URL", ""))
        rec = {"id": b["id"], "file": fn, "section": b["section"], "sub": b["sub"],
               "urls": urls, "quote": (q or "")[:400]}
        if not q:
            rec["verdict"] = "NO-QUOTE"
        elif not urls:
            rec["verdict"] = "NO-URL"
        else:
            u = urls[0]
            try:
                r = V.check(u, q)
            except Exception as e:
                r = {"verdict": "FETCH-ERROR", "err": str(e)[:200]}
            rec["verdict"] = r.get("verdict")
            rec["variant"] = r.get("variant")
            rec["ratio"] = r.get("ratio")
            rec["missing_tail"] = r.get("missing_tail")
            rec["url_checked"] = u
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{rec['verdict']:18s} {fn:6s} {b['id']:8s} {b['section']:18s} {(rec.get('url_checked') or rec['urls'][:1])!s:70.70s}", flush=True)
out.close()
