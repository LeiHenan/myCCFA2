#!/bin/bash
q="$1"; tag="$2"; lim="${3:-20}"
curl -sL --max-time 50 --retry 3 --retry-delay 2 --retry-all-errors \
 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" \
 "https://api.semanticscholar.org/graph/v1/paper/search?query=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$q")&limit=${lim}&fields=title,year,venue,externalIds,abstract" -o "s2_$tag.json"
echo "=== $tag :: $q :: size=$(wc -c < s2_$tag.json) ==="
python3 - "s2_$tag.json" <<'PY'
import sys,json
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("PARSE FAIL",e, open(sys.argv[1]).read()[:200]); raise SystemExit
for p in d.get('data',[]):
    ax=(p.get('externalIds') or {}).get('ArXiv','-')
    print(f"[{ax}] {p.get('year')} | {p.get('venue') or '-'} | {p.get('title')}")
PY
