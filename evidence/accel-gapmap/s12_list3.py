#!/usr/bin/env python3
"""S12 helper v3: parse GitHub listing/search pages via embedded JSON nodes.

Usage: python3 s12_list3.py '<url>'
Row: NUM <TAB> STATE <TAB> REASON <TAB> LABELS <TAB> AUTHOR <TAB> TITLE <TAB> URL
"""
import sys, re, json
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl


def find_balanced(s, start):
    depth = 0
    i = start
    instr = False
    esc = False
    while i < len(s):
        c = s[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        else:
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


def nodes(h):
    out = []
    for m in re.finditer(r'\{"__typename":"(?:Issue|PullRequest)","id":"', h):
        b = find_balanced(h, m.start())
        if b is None:
            continue
        try:
            out.append(json.loads(h[m.start():b]))
        except Exception:
            pass
    return out


def row(n):
    num = n.get("number")
    title = re.sub(r"<[^>]+>", "", n.get("titleHTML") or n.get("titleHtml") or n.get("title") or "")
    labs = []
    try:
        for e in n["labels"]["edges"]:
            labs.append(e["node"]["name"])
    except Exception:
        pass
    repo = "NVIDIA/TensorRT-LLM"
    try:
        repo = n["repository"]["owner"]["login"] + "/" + n["repository"]["name"]
    except Exception:
        pass
    kind = "pull" if n.get("__typename") == "PullRequest" else "issues"
    url = n.get("url") or f"https://github.com/{repo}/{kind}/{num}"
    au = ""
    try:
        au = n["author"]["login"]
    except Exception:
        pass
    st = n.get("state")
    if not st:
        if n.get("mergedAt"):
            st = "MERGED"
        elif n.get("closed"):
            st = "CLOSED"
        else:
            st = "OPEN"
    return (f"{num}\t{st}\t{n.get('stateReason')}\t{','.join(labs)}\t{au}\t{title}"
            f"\n\t{url}\tcreated={n.get('createdAt')} closed={n.get('closedAt')}")


if __name__ == "__main__":
    url = sys.argv[1]
    h = curl(url)
    ns = nodes(h)
    seen = set()
    print(f"### {url} -> {len(ns)} raw nodes")
    for n in ns:
        r = row(n)
        if r in seen:
            continue
        seen.add(r)
        print(r)
