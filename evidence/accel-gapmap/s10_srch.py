#!/usr/bin/env python3
"""S10 helper: GitHub server-rendered repo issue/PR LIST pages -> compact candidate rows.

Usage:
  python3 s10_srch.py <repo> <query-string> [more query strings...]
Example:
  python3 s10_srch.py vllm-project/vllm "is:issue benchmark prefix caching"
  python3 s10_srch.py sgl-project/sglang "is:pr is:closed is:unmerged benchmark"

List pages (github.com/<org>/<repo>/issues?q=... and /pulls?q=...) are server-rendered
and DO contain the result list. The global /search?q=...&type=issues page does NOT.
"""
import re, sys, html, urllib.parse
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl


def rows(repo, q, kind="issues", state=""):
    url = (f"https://github.com/{repo}/{kind}?q=" +
           urllib.parse.quote(q, safe="") + "&state=" + (state or "all"))
    h = curl(url)
    out = []
    # primary: anchor tags linking to /issues/N or /pull/N with an aria label / title
    for m in re.finditer(r'href="/' + re.escape(repo) + r'/(issues|pull)/(\d+)"[^>]*>(.*?)</a>', h, re.S):
        kind_, num, inner = m.group(1), m.group(2), m.group(3)
        txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", inner))).strip()
        if len(txt) < 8:
            continue
        if txt.lower() in ("issues", "pull requests", "labels", "milestones"):
            continue
        out.append((num, kind_, txt[:200]))
    seen, res = set(), []
    for num, kind_, txt in out:
        if num in seen:
            continue
        seen.add(num)
        res.append((num, kind_, txt))
    return url, res


if __name__ == "__main__":
    repo = sys.argv[1]
    kind = "issues"
    args = sys.argv[2:]
    if args and args[0].startswith("kind="):
        kind = args[0][5:]; args = args[1:]
    for q in args:
        url, r = rows(repo, q, kind)
        print(f"\n### REPO {repo} kind={kind} QUERY: {q}  -> {len(r)} rows")
        print(f"### URL: {url}")
        for num, k, t in r:
            print(f"{k}/{num}\t{t}")
