#!/bin/bash
# usage: bing.sh "<query>"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
Q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$1")
OUT=$(mktemp)
curl -sL --max-time 40 --retry 3 --retry-delay 2 --retry-all-errors -A "$UA" \
  -H "Accept-Language: en-US,en;q=0.9" \
  "https://www.bing.com/search?q=$Q" -o "$OUT"
python3 "$(dirname "$0")/bparse.py" "$OUT"
rm -f "$OUT"
