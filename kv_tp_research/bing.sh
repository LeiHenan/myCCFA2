#!/bin/bash
# usage: bing.sh "query" [count]
Q="$1"; N="${2:-1}"
curl -sL --retry 4 --retry-delay 2 --retry-all-errors --max-time 40 \
 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" \
 "https://www.bing.com/search?q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$Q")&count=30&setlang=en&cc=US" \
 | python3 - "$N" <<'PY'
import sys,re,html
raw=sys.stdin.read()
# bing result blocks
items=re.findall(r'<li class="b_algo".*?</li>', raw, re.S)
out=[]
for it in items:
    m=re.search(r'<h2><a href="([^"]+)"[^>]*>(.*?)</a></h2>', it, re.S)
    if not m: continue
    url=html.unescape(m.group(1)); title=html.unescape(re.sub('<[^>]+>','',m.group(2)))
    sn=re.search(r'<p class="[^"]*">(.*?)</p>', it, re.S)
    snip=html.unescape(re.sub('<[^>]+>','',sn.group(1))) if sn else ''
    out.append((title,url,snip[:400]))
for t,u,s in out:
    print(f"* {t}\n  {u}\n  {s}\n")
print(f"[{len(out)} results]")
PY
