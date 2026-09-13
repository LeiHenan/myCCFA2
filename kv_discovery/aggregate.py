#!/usr/bin/env python3
"""Aggregate all raw search-API JSON into one deduped candidate table."""
import json, glob, os, sys

RAW = "/Users/leihenan/Desktop/myProject/kv_discovery/raw"


def main():
    rows = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        qname = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"# {qname}: PARSE ERROR {e}", file=sys.stderr)
            continue
        items = d.get("items")
        if items is None:
            print(f"# {qname}: {str(d)[:200]}", file=sys.stderr)
            continue
        for it in items:
            repo = it["repository_url"].replace("https://api.github.com/repos/", "")
            num = it["number"]
            pr = it.get("pull_request")
            if pr is None:
                typ = "issue"
                merged = "-"
            else:
                typ = "PR"
                merged = "yes" if pr.get("merged_at") else "no"
            labels = ",".join(l["name"] for l in it.get("labels", []))
            key = f"{repo}#{num}"
            rec = rows.setdefault(key, {
                "key": key, "repo": repo, "num": num, "type": typ,
                "merged": merged, "state": it.get("state"),
                "state_reason": it.get("state_reason"),
                "created": it.get("created_at"), "closed": it.get("closed_at"),
                "labels": labels, "title": it.get("title"),
                "url": it.get("html_url"), "queries": [],
            })
            rec["queries"].append(qname)
    out = sorted(rows.values(), key=lambda r: (r["repo"], -(r["num"] or 0)))
    print(json.dumps(out, indent=1))
    print(f"# TOTAL UNIQUE: {len(out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
