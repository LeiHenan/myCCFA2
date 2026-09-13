#!/usr/bin/env python3
"""Extract rendered comment/review text from a GitHub PR or issue HTML page.

Strategy:
  * metadata from react-app.embeddedData (pullRequestsLayoutRoute.pullRequest,
    or the GraphQL preloadedQuery node for issues)
  * comment text from the server-rendered timeline markup (balanced <div> scan
    around every element whose class contains "markdown-body")
  * if the page says there are hidden items, follow the timeline_more_items
    partial endpoint.

Usage: gh_page.py <owner/repo> <number> [--max N]
"""
import sys, json, re, subprocess, html, os, hashlib

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
CACHE = "/Users/leihenan/Desktop/myProject/kv_discovery/cache"


def fetch(url, cache=True):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    cp = os.path.join(CACHE, key + ".html")
    if cache and os.path.exists(cp) and os.path.getsize(cp) > 1000:
        return open(cp, encoding="utf-8", errors="replace").read()
    p = subprocess.run(["curl", "-sL", "-m", "90", "--compressed", "-A", UA,
                        "-H", "Accept: text/html,application/xhtml+xml",
                        "-H", "Accept-Language: en-US,en;q=0.9", url],
                       capture_output=True)
    page = p.stdout.decode("utf-8", "replace")
    with open(cp, "w", encoding="utf-8") as f:
        f.write(page)
    return page


def strip_tags(s):
    if not s:
        return ""
    s = re.sub(r"<(script|style|svg|template)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<li[^>]*>", "\n- ", s, flags=re.I)
    s = re.sub(r"</(p|div|li|h[1-6]|tr|pre|blockquote|td)>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def balanced_div(page, start):
    """start = index of the '<div' that opens the target. Return inner html."""
    i = page.find(">", start)
    if i < 0:
        return ""
    depth = 1
    j = i + 1
    pat = re.compile(r"<div\b|</div>", re.I)
    while depth > 0:
        m = pat.search(page, j)
        if not m:
            break
        if m.group(0).lower() == "</div>":
            depth -= 1
            if depth == 0:
                return page[i + 1:m.start()]
        else:
            depth += 1
        j = m.end()
    return page[i + 1:j]


def balanced_any(page, start, tag):
    i = page.find(">", start)
    if i < 0:
        return ""
    depth = 1
    j = i + 1
    pat = re.compile(r"<" + tag + r"\b|</" + tag + r">", re.I)
    while depth > 0:
        m = pat.search(page, j)
        if not m:
            break
        if m.group(0).lower() == f"</{tag}>":
            depth -= 1
            if depth == 0:
                return page[i + 1:m.start()]
        else:
            depth += 1
        j = m.end()
    return page[i + 1:j]


def markdown_blocks(page):
    out = []
    seen = set()
    for m in re.finditer(r'<(\w+)[^>]*class="[^"]*markdown-body[^"]*"[^>]*>', page):
        tag = m.group(1).lower()
        if tag in ("script", "style"):
            continue
        inner = balanced_any(page, m.start(), tag)
        txt = strip_tags(inner)
        if txt and txt not in seen:
            seen.add(txt)
            out.append((m.start(), txt))
    out.sort(key=lambda x: x[0])
    return out


def author_before(page, pos, window=9000):
    seg = page[max(0, pos - window):pos]
    cands = re.findall(r'data-hovercard-url="/users/([A-Za-z0-9_.-]+)/hovercard"', seg)
    if not cands:
        cands = re.findall(r'href="/([A-Za-z0-9_.-]+)"[^>]*class="[^"]*author', seg)
    if not cands:
        cands = re.findall(r'author"[^>]*>\s*([A-Za-z0-9_.-]+)', seg)
    return cands[-1] if cands else "?"


def date_before(page, pos, window=9000):
    seg = page[max(0, pos - window):pos]
    d = re.findall(r'datetime="([0-9T:\-Z\.]+)"', seg)
    if d:
        return d[-1]
    d = re.findall(r'([A-Z][a-z]{2} \d{1,2}, \d{4})', seg)
    return d[-1] if d else "?"


def embedded(page):
    out = []
    for m in re.finditer(
        r'<script type="application/json" data-target="react-app.embeddedData">(.*?)</script>',
        page, re.S):
        try:
            out.append(json.loads(m.group(1)))
        except Exception:
            pass
    return out


def timeline_events(page):
    """Compress each rendered js-timeline-item into a one/two-line summary,
    keeping label-add events (they carry maintainer intent) verbatim."""
    out = []
    parts = page.split("js-timeline-item")
    for p in parts[1:]:
        seg = p[:9000]
        txt = strip_tags(seg)
        txt = re.sub(r"\s*\n\s*", " | ", txt)
        txt = re.sub(r"(\|\s*)+", "| ", txt).strip(" |")
        if not txt:
            continue
        # cut the boilerplate tail
        for stop in ("Copy link", "Copy Markdown"):
            pass
        out.append("EV: " + txt[:600])
    return out


def main():
    owner_repo, number = sys.argv[1], sys.argv[2]
    base = f"https://github.com/{owner_repo}"
    page = fetch(f"{base}/issues/{number}")
    # find real PR url if redirected
    is_pr = "/pull/" in page[:200000] and "pullRequestsLayoutRoute" in page

    print(f"FETCHED: {base}/issues/{number}  (is_pr={is_pr}, {len(page)} bytes)")

    pnode = None
    for b in embedded(page):
        pl = b.get("payload") or {}
        # issue graphql
        for q in pl.get("preloadedQueries", []) or []:
            repo = ((q.get("result") or {}).get("data") or {}).get("repository") or {}
            for k in ("issue", "pullRequest"):
                if isinstance(repo.get(k), dict):
                    pnode = repo[k]
                    print(f"GRAPHQL NODE: {k}")
        lay = pl.get("pullRequestsLayoutRoute") or {}
        if lay.get("pullRequest"):
            pr = lay["pullRequest"]
            print("LAYOUT NODE (PR):")
            for k in ("number", "title", "state", "closedTime", "createdTime",
                      "mergedTime", "mergedBy", "commitsCount", "baseBranch", "headBranch"):
                print(f"  {k} = {pr.get(k)}")
            print(f"  author = {(pr.get('author') or {}).get('login')}")

    if pnode:
        print("GRAPHQL META:")
        for k in ("number", "title", "state", "stateReason", "createdAt",
                  "updatedAt", "closedAt"):
            print(f"  {k} = {pnode.get(k)}")
        print(f"  author = {(pnode.get('author') or {}).get('login')}")
        labs = [((e or {}).get("node") or {}).get("name")
                for e in ((pnode.get("labels") or {}).get("edges") or [])]
        print(f"  labels = {labs}")
        for tk in ("frontTimelineItems", "backTimelineItems"):
            edges = ((pnode.get(tk) or {}).get("edges")) or []
            for e in edges:
                n = (e or {}).get("node") or {}
                tn = n.get("__typename")
                if tn == "IssueComment":
                    print(f"\n[GRAPHQL COMMENT by {(n.get('author') or {}).get('login')} "
                          f"assoc={n.get('authorAssociation')} at {n.get('createdAt')}]")
                    print(n.get("body", ""))
                elif tn == "ClosedEvent":
                    print(f"\n[GRAPHQL CLOSED by {(n.get('actor') or {}).get('login')} "
                          f"at {n.get('createdAt')} reason={n.get('stateReason')}]")
                elif tn == "MergedEvent":
                    print(f"\n[GRAPHQL MERGED at {n.get('createdAt')}]")

    print("\n########## RENDERED TIMELINE EVENTS ##########")
    for ev in timeline_events(page):
        print(ev)

    print("\n########## RENDERED TIMELINE (markdown blocks) ##########")
    for pos, txt in markdown_blocks(page):
        who = author_before(page, pos)
        when = date_before(page, pos)
        print(f"\n[RENDERED BLOCK author={who} date={when}]")
        print(txt[:6000])

    # hidden items?
    mm = re.search(r'(\d+)\s+hidden items', page)
    if mm:
        print(f"\n>>> {mm.group(1)} HIDDEN ITEMS (not rendered)")
        fm = re.search(r'action="([^"]*timeline_more_items[^"]*)"', page)
        if fm:
            more = html.unescape(fm.group(1))
            url = base + more if more.startswith("/") else more
            print(f">>> following: {url}")
            p2 = fetch(url)
            for pos, txt in markdown_blocks(p2):
                print(f"\n[MORE BLOCK author={author_before(p2,pos)} date={date_before(p2,pos)}]")
                print(txt[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
