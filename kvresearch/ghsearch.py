#!/usr/bin/env python3
"""
Scrape GitHub issue/PR search results from the HTML page (zero API quota).
GitHub embeds a JSON payload containing number, title, state, stateReason,
labels, dates, author for every result on the page.

Usage: python3 ghsearch.py "repo:vllm-project/vllm chunked prefill in:title"
       (the string after repo: is passed as the q= parameter verbatim)
"""
import re, sys, json, html, urllib.parse, subprocess, time, os, hashlib

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)


def fetch(url):
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    p = os.path.join(CACHE, key + ".html")
    if os.path.exists(p) and os.path.getsize(p) > 2000:
        return open(p, encoding="utf-8", errors="replace").read()
    r = subprocess.run(["curl", "-sL", "--compressed", "-H",
                        "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36",
                        url], capture_output=True, text=True)
    t = r.stdout
    if len(t) > 2000:
        open(p, "w", encoding="utf-8").write(t)
    return t


def parse_nodes(t):
    """Extract issue/PR nodes from the embedded JSON payload."""
    out = []
    # Each node object has "__typename":"Issue" or "PullRequest" and a "number".
    for m in re.finditer(r'\{"__typename":"(Issue|PullRequest)".*?"__isNode":"(?:Issue|PullRequest)"\}', t, re.S):
        blob = m.group(0)
        try:
            node = json.loads(blob)
        except Exception:
            continue
        num = node.get("number")
        if not num:
            continue
        title = node.get("titleHtml") or node.get("title") or ""
        title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
        labels = [e["node"]["name"] for e in node.get("labels", {}).get("edges", []) if e.get("node")]
        author = (node.get("author") or {}).get("login")
        out.append({
            "number": num,
            "type": node.get("__typename"),
            "title": title,
            "state": node.get("state"),
            "stateReason": node.get("stateReason"),
            "closed": node.get("closed"),
            "closedAt": (node.get("closedAt") or "")[:10],
            "createdAt": (node.get("createdAt") or "")[:10],
            "updatedAt": (node.get("updatedAt") or "")[:10],
            "author": author,
            "labels": labels,
            "repo": ((node.get("repository") or {}).get("owner") or {}).get("login", "") + "/" +
                    ((node.get("repository") or {}).get("name") or ""),
        })
    # dedupe by number keeping richest
    seen = {}
    for n in out:
        seen[n["number"]] = n
    return list(seen.values())


def search(q, sort="created", order="desc", page=1, state=None, kind=None):
    parts = [q]
    if state:
        parts.append("is:" + state)
    if kind == "issue":
        parts.append("is:issue")
    elif kind == "pr":
        parts.append("is:pr")
    full = " ".join(parts)
    m = re.search(r"repo:([\w.\-]+/[\w.\-]+)", full)
    rest = re.sub(r"repo:[\w.\-]+/[\w.\-]+", "", full).strip()
    if m:
        url = ("https://github.com/" + m.group(1) + "/issues?q=" + urllib.parse.quote(rest) +
               "&sort=" + sort + "&order=" + order + "&page=" + str(page))
    else:
        url = ("https://github.com/search?q=" + urllib.parse.quote(full) +
               "&type=issues&sort=" + sort + "&order=" + order + "&p=" + str(page))
    t = fetch(url)
    nodes = parse_nodes(t)
    tc = re.findall(r'"totalCount":(\d+)', t)
    return nodes, (int(tc[0]) if tc else None), full


def main():
    args = sys.argv[1:]
    sort, order, page, state, kind = "created", "desc", 1, None, None
    qs = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--sort":
            sort = args[i + 1]; i += 2
        elif a == "--order":
            order = args[i + 1]; i += 2
        elif a == "--page":
            page = int(args[i + 1]); i += 2
        elif a == "--state":
            state = args[i + 1]; i += 2
        elif a == "--kind":
            kind = args[i + 1]; i += 2
        elif a == "--json":
            i += 1
        else:
            qs.append(a); i += 1
    q = " ".join(qs)
    nodes, total, full = search(q, sort, order, page, state, kind)
    print(json.dumps({"query": full, "total": total, "n": len(nodes), "items": nodes}, indent=1))


if __name__ == "__main__":
    main()
