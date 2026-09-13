#!/usr/bin/env python3
"""
Read a GitHub issue/PR page WITHOUT the REST API (server-rendered HTML only).
Extracts: title, state, stateReason, author, date, body text, comment count,
labels. Comments themselves are NOT in the initial HTML (React-loaded), so
`comments_available` is reported as False unless found.

Usage: python3 ghpage.py vllm-project/vllm 26133
       python3 ghpage.py --body-only vllm-project/vllm 26133
"""
import re, sys, json, html, os, hashlib, subprocess

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def fetch(url):
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    p = os.path.join(CACHE, key + ".html")
    if os.path.exists(p) and os.path.getsize(p) > 2000:
        return open(p, encoding="utf-8", errors="replace").read()
    r = subprocess.run(["curl", "-sL", "--compressed", "--max-time", "45",
                        "-A", UA, url], capture_output=True, text=True)
    t = r.stdout
    if len(t) > 2000:
        open(p, "w", encoding="utf-8").write(t)
    return t


def strip(h):
    h = re.sub(r"<(script|style|svg)\b.*?</\1>", " ", h, flags=re.S | re.I)
    h = re.sub(r"<br\s*/?>", "\n", h, flags=re.I)
    h = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", h, flags=re.I)
    h = re.sub(r"<[^>]+>", "", h)
    return html.unescape(h)


def read(repo, num):
    url = f"https://github.com/{repo}/issues/{num}?plain=1"
    t = fetch(url)
    out = {"repo": repo, "number": num, "url": f"https://github.com/{repo}/issues/{num}",
           "read_body": False}
    if len(t) < 2000:
        return out
    # state reason
    sr = None
    m = re.search(r'data-status="(issueClosedNotPlanned|issueClosedAsCompleted|pullClosed|pullMerged|issueOpen|pullOpen)"', t)
    if m:
        sr = m.group(1)
    out["state_marker"] = sr
    out["closed_not_planned"] = (sr == "issueClosedNotPlanned")
    # structured data (title, body, date, author)
    try:
        m = re.search(r'<script type="application/json" data-target="react-app.embeddedData">(.*?)</script>', t, re.S)
        d = json.loads(m.group(1))
        sd = d["payload"]["structured_data"]
        out["title"] = sd.get("headline")
        out["date"] = (sd.get("datePublished") or "")[:10]
        out["body"] = sd.get("articleBody") or ""
        a = sd.get("author")
        out["author"] = a.get("name") if isinstance(a, dict) else a
        ist = sd.get("interactionStatistic") or {}
        out["comment_count"] = ist.get("userInteractionCount")
        out["read_body"] = bool(out["body"])
    except Exception as e:
        out["err_structured"] = str(e)
    # title fallback
    if not out.get("title"):
        m = re.search(r'<title>(.*?)</title>', t, re.S)
        if m:
            out["title"] = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    # author fallback
    if not out.get("author"):
        m = re.search(r'data-testid="issue-body-header-author">\s*([\w.\-]+)\s+opened', t)
        if m:
            out["author"] = m.group(1)
    # labels present in html
    out["labels"] = sorted(set(re.findall(r'data-name="([^"]+)"[^>]*class="[^"]*Label', t)))
    # is it a PR?
    out["is_pr"] = ("/pull/" in t[:200000] and "pullRequest" in t) or ('"__typename":"PullRequest"' in t)
    # merged?
    out["merged"] = bool(re.search(r'data-status="pullMerged"', t))
    out["body_len"] = len(out.get("body") or "")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    bodyonly = "--body-only" in args
    args = [a for a in args if not a.startswith("--")]
    repo, num = args[0], args[1]
    r = read(repo, num)
    if bodyonly:
        print(r.get("title"), "|", r.get("date"), "|", r.get("state_marker"))
        print(r.get("body", "")[:6000])
    else:
        print(json.dumps(r, indent=1)[:9000])
