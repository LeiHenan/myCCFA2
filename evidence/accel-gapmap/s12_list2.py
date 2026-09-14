#!/usr/bin/env python3
"""S12 helper v2: parse GitHub listing/search pages via embedded JSON edges.

Usage: python3 s12_list2.py '<url>'
Prints one row per hit: NUM <TAB> STATE <TAB> LABELS <TAB> TITLE <TAB> URL
"""
import sys, re, json
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl


def find_balanced(s, start):
    """start points at '{' ; return end index (exclusive) or None."""
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
    for m in re.finditer(r'"search":\{"edges":\[', h):
        i = m.end()
        # walk edges: {"node": {...}} separated by commas
        while True:
            nm = re.compile(r'\{"node":').search(h, i)
            if not nm or nm.start() - i > 40:
                break
            st = h.index("{", nm.end() - 1 + 1 - 1) if False else nm.end()
            # json object starts right after '{"node":'
            b = find_balanced(h, nm.end())
            if b is None:
                break
            try:
                obj = json.loads(h[nm.end():b])
            except Exception:
                break
            out.append(obj)
            i = b
            if not re.match(r'\s*,\s*\{', h[i:i + 20]):
                break
    return out


def row(n):
    num = n.get("number")
    title = n.get("titleHtml") or n.get("title") or ""
    title = re.sub(r"<[^>]+>", "", title)
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
    return f"{num}\t{n.get('state','?')}\t{n.get('stateReason')}\t{','.join(labs)}\t{au}\t{title}\n\t{url}"


if __name__ == "__main__":
    url = sys.argv[1]
    h = curl(url)
    ns = nodes(h)
    print(f"### {url} -> {len(ns)} rows")
    for n in ns:
        print(row(n))
