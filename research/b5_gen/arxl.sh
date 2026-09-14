#!/bin/bash
# usage: arxl.sh <search_query_urlencoded>
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
curl -sL --max-time 60 --retry 3 --retry-delay 2 --retry-all-errors -A "$UA" "$1" | python3 -c "
import sys,re,html
h=sys.stdin.read()
# strip tags
t=re.sub(r'<script.*?</script>','',h,flags=re.S)
t=re.sub(r'<style.*?</style>','',t,flags=re.S)
t=re.sub(r'<[^>]+>',' ',t)
t=html.unescape(t)
t=re.sub(r'[ \t]+',' ',t)
t=re.sub(r'\n\s*\n+','\n',t)
print(t)
"
