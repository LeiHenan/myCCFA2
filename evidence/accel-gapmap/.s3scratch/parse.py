#!/usr/bin/env python3
"""Parse GitHub issue-list search page text -> entry rows (block-based title extraction)."""
import re, sys, os

HDR = re.compile(r'\s*#\s*(\d+)\s+In\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+);\s*·\s*(.*)')

def parse(path):
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="replace")]
    hdrs = [i for i, l in enumerate(lines) if HDR.match(l)]
    rows = []
    prev = -1
    for i in hdrs:
        m = HDR.match(lines[i])
        num, repo, rest = m.group(1), m.group(2), m.group(3).strip()
        block = lines[prev+1:i]
        prev = i
        labels = []
        # status line
        state = "?"
        mst = re.search(r'Status:\s*(.+?)\.?\s*$', rest)
        if "was closed" in rest: state = "CLOSED"
        elif "opened on" in rest: state = "OPEN"
        elif "was merged" in rest: state = "MERGED"
        md = re.search(r'([A-Z][a-z]{2} \d{1,2}, \d{4})', rest)
        date = md.group(1) if md else "?"
        mu = re.match(r'(?:by\s+)?([A-Za-z0-9_-]+)', rest)
        user = mu.group(1) if mu else "?"
        # title = first non-blank line of block
        title = ""
        for b in block:
            if b:
                title = b
                break
        # labels = lines after the title that look like duplicated label text
        for b in block:
            if b.startswith("Status:") or not b or b == title:
                continue
            lb = b.split(" Related to ")[0].strip()
            if lb and lb not in labels:
                labels.append(lb)
        rows.append((num, title, state, date, repo, user, ",".join(labels[:6])))
    return rows

if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = parse(path)
        print(f"\n########## {os.path.basename(path)}  ({len(rows)} rows)")
        for num, title, state, date, repo, user, labels in rows:
            print(f"  {state:7} #{num:>6} {date:>13} [{labels[:40]:40}] {title[:170]}")
