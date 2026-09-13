#!/usr/bin/env python3
"""Fetch a GitHub issue OR pull request HTML page and extract verbatim
title / state / stateReason / labels / body / comments / reviews / closing remarks.

Works via the react-app.embeddedData -> payload.preloadedQueries[].result GraphQL blob
that GitHub server-renders into the HTML. NO API rate limit applies.

Usage:
  gh_extract.py <owner/repo> <number> [--raw PATH] [--mode full|meta|comments]
"""
import sys, json, re, subprocess, html, os, hashlib

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
CACHE = "/Users/leihenan/Desktop/myProject/kv_discovery/cache"


def fetch(url):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    cp = os.path.join(CACHE, key + ".html")
    if os.path.exists(cp) and os.path.getsize(cp) > 1000:
        return open(cp, encoding="utf-8", errors="replace").read()
    cmd = ["curl", "-sL", "-m", "60", "--compressed", "-A", UA,
           "-H", "Accept: text/html,application/xhtml+xml",
           "-H", "Accept-Language: en-US,en;q=0.9", url]
    p = subprocess.run(cmd, capture_output=True)
    page = p.stdout.decode("utf-8", "replace")
    with open(cp, "w", encoding="utf-8") as f:
        f.write(page)
    return page


def embedded(page, target="react-app.embeddedData"):
    out = []
    for m in re.finditer(
        r'<script type="application/json" data-target="' + re.escape(target) + r'">(.*?)</script>',
        page, re.S):
        try:
            out.append(json.loads(m.group(1)))
        except Exception:
            pass
    return out


def strip_tags(s):
    if not s:
        return ""
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|div|li|h[1-6]|tr|pre|blockquote)>", "\n", s, flags=re.I)
    s = re.sub(r"<li[^>]*>", "\n- ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def find_objects(obj, typename_pred, path="", acc=None):
    if acc is None:
        acc = []
    if isinstance(obj, dict):
        if obj.get("__typename") in typename_pred:
            acc.append((path, obj))
        for k, v in obj.items():
            find_objects(v, typename_pred, f"{path}.{k}", acc)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_objects(v, typename_pred, f"{path}[{i}]", acc)
    return acc


def get_in(obj, path):
    cur = obj
    for part in path:
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def dig_issue_or_pr(payload):
    """Return (kind, node) for the issue/PR GraphQL node if present."""
    for q in payload.get("preloadedQueries", []) or []:
        res = q.get("result") or {}
        repo = get_in(res, ["data", "repository"]) or {}
        for key in ("issue", "pullRequest"):
            if isinstance(repo.get(key), dict):
                return key, repo[key]
    return None, None


def fmt_user(u):
    if not isinstance(u, dict):
        return "?"
    return u.get("login") or u.get("name") or "?"


def main():
    owner_repo = sys.argv[1]
    number = sys.argv[2]
    kind_hint = "pull" if len(sys.argv) > 3 and sys.argv[3] == "--pr" else None
    url = f"https://github.com/{owner_repo}/issues/{number}"
    page = fetch(url)
    if len(page) < 2000:
        print("FETCH FAILED for", url)
        return 1

    blobs = embedded(page)
    if not blobs:
        print("NO embeddedData:", url)
        return 2

    pdf = None
    for b in blobs:
        payload = b.get("payload") if isinstance(b, dict) else None
        if not isinstance(payload, dict):
            continue
        kind, node = dig_issue_or_pr(payload)
        if node:
            print(f"URL: {url}")
            print(f"KIND: {kind}")
            print(f"TITLE: {node.get('title')}")
            print(f"STATE: {node.get('state')}  STATE_REASON: {node.get('stateReason')}")
            print(f"CREATED: {node.get('createdAt')}  CLOSED: {node.get('closedAt')}  UPDATED: {node.get('updatedAt')}")
            print(f"AUTHOR: {fmt_user(node.get('author'))}")
            labs = []
            for e in get_in(node, ["labels", "edges"]) or []:
                ln = (e or {}).get("node") or {}
                labs.append(ln.get("name"))
            print(f"LABELS: {labs}")
            # body
            body = node.get("body") or node.get("bodyHTML")
            if body:
                print(f"\n--- BODY ({fmt_user(node.get('author'))}) ---")
                print(strip_tags(body) if "<" in str(body) else body)
            # timeline
            for tk in ("frontTimelineItems", "backTimelineItems"):
                edges = get_in(node, [tk, "edges"]) or []
                if not edges:
                    continue
                print(f"\n--- TIMELINE ({tk}, {len(edges)} items) ---")
                for e in edges:
                    n = (e or {}).get("node") or {}
                    tn = n.get("__typename")
                    if tn == "IssueComment":
                        print(f"\n[COMMENT by {fmt_user(n.get('author'))} "
                              f"({n.get('authorAssociation')}) at {n.get('createdAt')}]")
                        print(n.get("body", ""))
                    elif tn == "PullRequestReview":
                        print(f"\n[REVIEW by {fmt_user(n.get('author'))} "
                              f"state={n.get('state')} at {n.get('createdAt')}]")
                        print(n.get("body", "") or "(no body)")
                    elif tn == "ClosedEvent":
                        print(f"\n[CLOSED by {fmt_user(n.get('actor'))} at {n.get('createdAt')} "
                              f"stateReason={n.get('stateReason')}]")
                    elif tn in ("ReopenedEvent", "MergedEvent", "CrossReferencedEvent",
                                "ReferencedEvent", "LabeledEvent", "UnlabeledEvent",
                                "AssignedEvent", "ReviewRequestedEvent",
                                "PullRequestCommit", "HeadRefForcePushedEvent",
                                "RenamedTitleEvent", "MilestonedEvent"):
                        extra = ""
                        if tn == "MergedEvent":
                            extra = f" commit={n.get('commit',{}).get('oid') if isinstance(n.get('commit'),dict) else ''}"
                        if tn == "LabeledEvent":
                            extra = f" label={(n.get('label') or {}).get('name')}"
                        if tn in ("CrossReferencedEvent", "ReferencedEvent"):
                            src = n.get("source") or {}
                            extra = f" {src.get('__typename')} #{(src.get('number'))} {src.get('title') or ''}"
                        print(f"[{tn} {n.get('createdAt')} actor={fmt_user(n.get('actor'))}{extra}]")
                    else:
                        print(f"[{tn} {n.get('createdAt')}]")
            pdf = payload
            print("\n" + "=" * 70)
            print("TOP-LEVEL PR/ISSUE NODE KEYS:", sorted(node.keys()))
            return 0

    # fall back to layout route (PRs)
    for b in blobs:
        payload = b.get("payload") if isinstance(b, dict) else None
        if not isinstance(payload, dict):
            continue
        lay = payload.get("pullRequestsLayoutRoute") or {}
        pr = lay.get("pullRequest")
        if pr:
            print(f"URL: {url}\nLAYOUT-ONLY (no GraphQL preload)")
            print(json.dumps(pr, indent=1)[:4000])
            return 0
    print("NO issue/PR node found in embeddedData:", url)
    return 3


if __name__ == "__main__":
    sys.exit(main())
