#!/usr/bin/env python3
"""V1: for every block, resolve the page state of every GitHub artifact it cites."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
p5 = importlib.import_module("v1_batch5") if False else None
from fetch import curl, strip_html  # noqa

BLOCK = re.compile(r"^###\s+([ABCDE][0-9.]*?)\s*\|\s*(CLOSED|OPEN|ABANDONED|HARDWARE-RULED-OUT|NEVER-DISCUSSED)\s*(.*)$")
KEY = re.compile(r"^([A-Z][A-Z0-9 _()/\.\->+]*?):\s?(.*)$")
GH = re.compile(r"https?://github\.com/([^/\s;,]+/[^/\s;,]+)/(pull|issues)/(\d+)")
urlre = re.compile(r"https?://[^\s;,]+")


def strip_parens(s):
    return re.sub(r"\s*\([^)]*\)", "", s)


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
            last = [f.group(1).strip().replace(" ", "_"), [f.group(2).strip()]]
            cur["fields"].append(last)
        elif last and line.strip() and line.strip() != "---":
            last[1].append(line.strip())
    if cur:
        yield cur


def get(fl, name):
    for k, v in fl:
        if k == name:
            return "\n".join(v).strip()
    return ""


STATE_CACHE = {}


def state(u):
    if u in STATE_CACHE:
        return STATE_CACHE[u]
    try:
        h = curl(u)
        t = re.sub(r"[ \t]+", " ", strip_html(h))
        m = re.search(r"#\s*\d+\s*\n\s*(Merged|Closed|Open|Draft)\b", t)
        hdr = m.group(1) if m else None
        oc = re.findall(r">\s*(Open|Closed)\s*<", t)
        mc = bool(re.search(r"\bmerged commit [0-9a-f]{7,}", t))
        stale = bool(re.search(r"automatically marked as stale|automatically closed due to inactivity", t, re.I))
        sr = sorted(set(re.findall(r'"stateReason":"([^"]+)"', h)))
        r = {"header": hdr, "openclosed": sorted(set(oc))[:3], "merged_commit": mc,
             "stale_bot": stale, "state_reason": sr}
    except Exception as e:
        r = {"error": str(e)[:120]}
    STATE_CACHE[u] = r
    return r


out = open(os.path.join(HERE, "v1_rowstate.jsonl"), "w", encoding="utf-8")
for fn in ["S1.md", "S2.md", "S3.md", "S4.md"]:
    for b in blocks(fn):
        fl = b["fields"]
        txt = get(fl, "ARTIFACT") + " " + get(fl, "URL")
        arts = []
        seen = set()
        for owner, kind, num in GH.findall(txt):
            key = (owner, kind, num)
            if key in seen:
                continue
            seen.add(key)
            u = f"https://github.com/{owner}/{kind}/{num}"
            arts.append({"u": u, **state(u)})
        rec = {"id": b["id"], "file": fn, "section": b["section"],
               "status_field": get(fl, "STATUS"), "artifacts": arts}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{fn:6s} {b['id']:8s} {b['section']:18s} " +
              " | ".join(f"{a['u'].split('/')[-2]}/{a['u'].split('/')[-1]}="
                         f"{a.get('header') or a.get('openclosed') or '?'}"
                         f"{'/MERGEDCOMMIT' if a.get('merged_commit') else ''}"
                         f"{'/STALE' if a.get('stale_bot') else ''} {a.get('state_reason') or ''}"
                         for a in arts), flush=True)
out.close()
