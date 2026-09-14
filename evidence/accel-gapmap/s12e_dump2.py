#!/usr/bin/env python3
"""S12e dump v2: precise GitHub issue/PR parser.

Extracts, from the embedded React/GraphQL payload:
  - title, state (OPEN/CLOSED/MERGED), stateReason, createdAt/closedAt/mergedAt, labels
  - body of the issue/PR itself (with author)
  - every IssueComment / PullRequestReview / PullRequestReviewComment with author,
    authorAssociation, createdAt, url and VERBATIM body
"""
import sys, re, json
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl, strip_html

NODE_RE = re.compile(r'"__typename":"(IssueComment|PullRequestReview|PullRequestReviewComment|Issue|PullRequest|Commit)"')
BODY_RE = re.compile(r'"body":"((?:[^"\\]|\\.)*)"')


def _decode(m):
    try:
        return json.loads('"' + m.group(1) + '"')
    except Exception:
        return None


def carve(h, node_start, next_start):
    return h[node_start:next_start]


def nearest_login_after(s, upto):
    """last login appearing strictly before index upto"""
    best = None
    for m in re.finditer(r'"login":"([^"]+)"', s[:upto]):
        best = m.group(1)
    return best


def assoc_near(s, upto):
    best = None
    for m in re.finditer(r'"authorAssociation":"([^"]+)"', s[:upto]):
        best = m.group(1)
    return best


def dump(url, maxlen=100000):
    h = curl(url)
    out = []
    t = re.search(r'<title>(.*?)</title>', h, re.S)
    out.append("TITLE: " + (strip_html(t.group(1)) if t else "?"))
    st = set(re.findall(r'"state":"(OPEN|CLOSED|MERGED|open|closed|merged|DRAFT|draft)"', h))
    out.append("STATE: " + ", ".join(sorted(st)))
    sr = set(re.findall(r'"stateReason":"([^"]+)"', h))
    out.append("STATE_REASON: " + ", ".join(sorted(sr)))
    for lab, pat in [("CREATED_AT", r'"createdAt":"([^"]+)"'), ("CLOSED_AT", r'"closedAt":"([^"]+)"'),
                     ("MERGED_AT", r'"mergedAt":"([^"]+)"')]:
        v = sorted(set(re.findall(pat, h)))
        if v:
            out.append(f"{lab}: " + " | ".join(v[:4]))
    labs = re.findall(r'"label":\{"[^}]*?"name":"([^"]+)"', h)
    if labs:
        out.append("LABELS: " + ", ".join(sorted(set(labs))[:30]))
    is_pr = "PullRequest" in "".join(set(re.findall(r'"__isIssueOrPullRequest":"(\w+)"', h)))
    out.append("IS_PR: " + ("YES" if is_pr else "no"))

    # split payload into nodes
    hits = [(m.start(), m.group(1)) for m in NODE_RE.finditer(h)]
    seen = set()
    n = 0
    for i, (pos, kind) in enumerate(hits):
        nxt = hits[i + 1][0] if i + 1 < len(hits) else min(len(h), pos + 300000)
        node = h[pos:nxt]
        bm = BODY_RE.search(node)
        if not bm:
            continue
        body = _decode(bm)
        if not body or not body.strip():
            continue
        if len(body) > maxlen:
            body = body[:maxlen] + "\n[...TRUNCATED BY DUMPER...]"
        login = nearest_login_after(node, bm.start())
        assoc = assoc_near(node, bm.start())
        key = (kind, login, body[:120])
        if key in seen:
            continue
        seen.add(key)
        cid = re.search(r'"url":"(https://github\.com/[^"]*#(?:issuecomment|discussion_r|pullrequestreview)[^"]*)"', node)
        created = re.search(r'"createdAt":"([^"]+)"', node)
        if kind in ("Issue", "PullRequest"):
            continue  # body handled separately below
        out.append(f"\n--- {kind} @{login} assoc={assoc} at={created.group(1) if created else '?'} ---\n{cid.group(1) if cid else ''}\n{body}")

    # the issue/PR body itself: look for the top-level node payload
    m = re.search(r'"issue":\{"author":\{"__typename":"[^"]+","login":"([^"]+)"', h)
    bm = re.search(r'"body":"((?:[^"\\]|\\.)*)"', h)
    if bm:
        b = _decode(bm)
        if b:
            out.append(f"\n=== ISSUE/PR BODY (author {m.group(1) if m else '?'}) ===\n{b}")
    return "\n".join(out)


if __name__ == "__main__":
    for u in sys.argv[1:]:
        print("#" * 100)
        print("URL:", u)
        try:
            print(dump(u))
        except Exception as e:
            print("ERROR:", repr(e))
