#!/usr/bin/env python3
"""V1: read the state of every GitHub PR/issue URL cited in S1-S4, from the page itself."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch import curl, strip_html

urls = [u.strip() for u in open(os.path.join(HERE, "v1_ghurls.txt")) if u.strip()]
out = open(os.path.join(HERE, "v1_ghstate.jsonl"), "w", encoding="utf-8")

for u in urls:
    rec = {"url": u}
    try:
        h = curl(u)
        t = re.sub(r"[ \t]+", " ", strip_html(h))
        m = re.search(r"<title>(.*?)</title>", h, re.S)
        rec["title"] = re.sub(r"\s+", " ", strip_html(m.group(1)))[:160] if m else None
        # page-text markers
        mm = re.search(r"#\s*\d+\s*\n\s*(Merged|Closed|Open|Draft)\b", t)
        rec["header_state"] = mm.group(1) if mm else None
        rec["merged_commit"] = bool(re.search(r"\bmerged commit [0-9a-f]{7,}", t))
        rec["closed_this"] = bool(re.search(r"\bclosed this\b", t))
        rec["closed_with_commit"] = bool(re.search(r"\bclosed this as completed\b|\bclosed this\b", t))
        rec["reopened"] = bool(re.search(r"\breopened this\b", t))
        st = re.findall(r'"state":"(open|closed|merged|draft)"', h)
        rec["payload_states"] = sorted(set(st))
        rec["state_reason"] = sorted(set(re.findall(r'"stateReason":"([^"]+)"', h)))
        rec["merged_at"] = sorted(set(re.findall(r'"mergedAt":"([^"]+)"', h)))[:3]
        rec["closed_at"] = sorted(set(re.findall(r'"closedAt":"([^"]+)"', h)))[:3]
        # stale bot text
        rec["stale_bot"] = bool(re.search(r"automatically marked as stale|automatically closed due to inactivity", t, re.I))
        rec["labels"] = sorted(set(re.findall(r'"label":\{"[^}]*?"name":"([^"]+)"', h)))[:20]
    except Exception as e:
        rec["error"] = str(e)[:200]
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.flush()
    print(f"{str(rec.get('header_state')):8s} mc={int(bool(rec.get('merged_commit')))} "
          f"stale={int(bool(rec.get('stale_bot')))} {u[-60:]:60s} {(rec.get('title') or '')[:60]}",
          flush=True)
out.close()
