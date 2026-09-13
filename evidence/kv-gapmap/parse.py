#!/usr/bin/env python3
"""Mechanically parse S*.md evidence files into structured JSONL. No interpretation."""
import re, json, glob, os
from collections import Counter

D = os.path.dirname(os.path.abspath(__file__))

def norm_key(k):
    k = k.strip().strip('*').strip().rstrip(':').lower()
    k = re.sub(r'\s+', ' ', k)
    return k

out = []
for path in sorted(glob.glob(os.path.join(D, "S[0-9]*.md"))):
    if path.endswith(".verify.md"):
        continue
    seg = os.path.basename(path)[:-3]
    txt = open(path, encoding="utf-8", errors="replace").read()
    parts = re.split(r'(?m)^###\s+', txt)
    for p in parts[1:]:
        lines = p.split("\n")
        head = lines[0].strip()
        m = re.match(r'^(CLOSED|OPEN|ABANDONED|HW_RULED_OUT|HW)[-_]?(\d+)?\s*:\s*(.*)$', head)
        if not m:
            continue
        cls, num, title = m.group(1), m.group(2) or "", m.group(3).strip()
        if cls == "HW_RULED_OUT":
            cls = "HW"
        body = "\n".join(lines[1:])
        fields = {}
        cur = None
        for ln in body.split("\n"):
            fm = re.match(r'^\s*[-*]\s*(?:\*\*)?([A-Za-z][A-Za-z0-9 ,\'"/\-()]{0,58}?)(?:\*\*)?\s*:\s*(.*)$', ln)
            if fm:
                cur = norm_key(fm.group(1))
                v = fm.group(2).strip()
                if cur in fields and fields[cur]:
                    fields[cur] += " " + v
                else:
                    fields[cur] = v
            elif cur and ln.strip() and not ln.startswith("#"):
                fields[cur] += " " + ln.strip()
        out.append({"seg": seg, "cls": cls, "num": num, "title": title,
                    "fields": fields, "raw": body.strip()})

with open(os.path.join(D, "_index.jsonl"), "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("blocks parsed:", len(out))
print("classes:", Counter(r["cls"] for r in out))
for cls in ["CLOSED", "OPEN", "ABANDONED", "HW"]:
    c = Counter()
    n = 0
    for r in out:
        if r["cls"] == cls:
            n += 1
            for k in r["fields"]:
                c[k] += 1
    print(f"--- {cls} (n={n}) top fields:")
    for k, v in c.most_common(10):
        print(f"      {v:4d}  {k}")
