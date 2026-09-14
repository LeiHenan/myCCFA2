#!/usr/bin/env python3
"""S12e dump: GitHub issue/PR -> state, stateReason, labels, comments WITH authors (verbatim bodies)."""
import sys, re, json
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl, strip_html


def dump(url):
    h = curl(url)
    out = []
    t = re.search(r'<title>(.*?)</title>', h, re.S)
    out.append("TITLE: " + strip_html(t.group(1)) if t else "TITLE: ?")
    st = set(re.findall(r'"state":"(OPEN|CLOSED|MERGED|open|closed|merged|draft|DRAFT)"', h))
    out.append("STATE: " + ", ".join(sorted(st)))
    sr = set(re.findall(r'"stateReason":"([^"]+)"', h))
    out.append("STATE_REASON: " + ", ".join(sorted(sr)))
    ma = sorted(set(re.findall(r'"mergedAt":"([^"]+)"', h)))
    ca = sorted(set(re.findall(r'"closedAt":"([^"]+)"', h)))
    cr = sorted(set(re.findall(r'"createdAt":"([^"]+)"', h)))
    if ma: out.append("MERGED_AT: " + " | ".join(ma[:3]))
    if ca: out.append("CLOSED_AT: " + " | ".join(ca[:3]))
    if cr: out.append("CREATED_AT: " + " | ".join(cr[:3]))
    labs = re.findall(r'"label":\{"[^}]*?"name":"([^"]+)"', h)
    if labs: out.append("LABELS: " + ", ".join(sorted(set(labs))[:30]))

    # author -> nearest following body
    items = []
    for m in re.finditer(r'"author":\{"__typename":"(?:User|Bot|Organization|Mannequin)","login":"([^"]+)"', h):
        login = m.group(1)
        seg = h[m.end():m.end() + 40000]
        b = re.search(r'"body":"((?:[^"\\]|\\.)*)"', seg)
        if not b:
            continue
        try:
            body = json.loads('"' + b.group(1) + '"')
        except Exception:
            continue
        body = body.strip()
        if not body:
            continue
        items.append((login, body))
    # dedupe preserving order
    seen = set()
    uniq = []
    for login, body in items:
        k = (login, body[:200])
        if k in seen:
            continue
        seen.add(k)
        uniq.append((login, body))
    out.append(f"COMMENT_COUNT: {len(uniq)}")
    for i, (login, body) in enumerate(uniq, 1):
        out.append(f"\n--- [{i}] @{login} ---\n{body}")
    return "\n".join(out)


if __name__ == "__main__":
    for u in sys.argv[1:]:
        print("#" * 100)
        print("URL:", u)
        try:
            print(dump(u))
        except Exception as e:
            print("ERROR:", e)
