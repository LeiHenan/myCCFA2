#!/usr/bin/env python3
"""
Fetch issue/PR comments via the GitHub REST API (rate limited).
Usage: python3 fetchcomments.py targets.txt out.json
targets.txt lines: owner/repo NUMBER
Handles rate-limit exhaustion gracefully: stops and reports progress so the
caller can resume later (already-fetched items are cached to disk).
"""
import sys, os, json, time, subprocess, hashlib

OUT = sys.argv[2]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apicache")
os.makedirs(CACHE, exist_ok=True)
UA = "dsh-kv-research"


def api(url):
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    p = os.path.join(CACHE, key + ".json")
    if os.path.exists(p) and os.path.getsize(p) > 2:
        return json.load(open(p)), 200
    r = subprocess.run(["curl", "-s", "--max-time", "30", "-A", UA,
                        "-H", "Accept: application/vnd.github+json", url],
                       capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except Exception:
        return {"_parse_error": r.stdout[:200]}, 0
    if isinstance(d, dict) and "rate limit exceeded" in str(d.get("message", "")).lower():
        return d, 403
    open(p, "w").write(json.dumps(d))
    return d, 200


def get_comments(repo, num):
    out = {"repo": repo, "number": num, "comments": [], "issue_meta": {}}
    d, code = api(f"https://api.github.com/repos/{repo}/issues/{num}/comments?per_page=100")
    if code == 403:
        return None
    if isinstance(d, list):
        for c in d:
            out["comments"].append({
                "user": (c.get("user") or {}).get("login"),
                "created_at": (c.get("created_at") or "")[:10],
                "association": c.get("author_association"),
                "body": c.get("body") or "",
            })
    # issue metadata (state, state_reason, labels, dates)
    m, code2 = api(f"https://api.github.com/repos/{repo}/issues/{num}")
    if code2 == 403:
        return None
    if isinstance(m, dict) and "number" in m:
        out["issue_meta"] = {
            "title": m.get("title"), "state": m.get("state"),
            "state_reason": m.get("state_reason"),
            "created_at": (m.get("created_at") or "")[:10],
            "closed_at": (m.get("closed_at") or "")[:10],
            "labels": [l["name"] for l in m.get("labels", [])],
            "user": (m.get("user") or {}).get("login"),
            "comments": m.get("comments"),
            "is_pr": "pull_request" in m,
            "html_url": m.get("html_url"),
        }
    return out


def main():
    tgt = []
    for line in open(sys.argv[1]):
        line = line.strip()
        if line and not line.startswith("#"):
            a = line.split()
            tgt.append((a[0], a[1]))
    results = []
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT))
        except Exception:
            results = []
    done = {(r["repo"], str(r["number"])) for r in results}
    for repo, num in tgt:
        if (repo, str(num)) in done:
            print("skip(done)", repo, num, flush=True)
            continue
        r = get_comments(repo, num)
        if r is None:
            print("RATE_LIMITED after", len(results), "items", flush=True)
            json.dump(results, open(OUT, "w"), indent=1)
            return 2
        results.append(r)
        meta = r.get("issue_meta") or {}
        print(f"OK {repo} {num} state={meta.get('state')}/{meta.get('state_reason')} "
              f"comments={len(r['comments'])} closed={meta.get('closed_at')}", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
        time.sleep(1.0)
    json.dump(results, open(OUT, "w"), indent=1)
    print("DONE", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
