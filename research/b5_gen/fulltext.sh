#!/bin/bash
# usage: fulltext.sh <arxivid> <outbase>
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
ID="$1"; B="$2"
for v in v1 v2 v3; do
  code=$(curl -sL --max-time 60 -A "$UA" "https://arxiv.org/html/${ID}${v}" -o "${B}.html" -w "%{http_code}")
  sz=$(wc -c < "${B}.html")
  if [ "$code" = "200" ] && [ "$sz" -gt 5000 ]; then echo "OK ${ID}${v} sz=$sz"; break; fi
done
python3 - "$B" <<'PY'
import re,html,sys
b=sys.argv[1]
h=open(b+'.html',encoding='utf-8',errors='replace').read()
t=re.sub(r'<script.*?</script>','',h,flags=re.S); t=re.sub(r'<style.*?</style>','',t,flags=re.S)
t=re.sub(r'<[^>]+>',' ',t); t=html.unescape(t); t=re.sub(r'[ \t]+',' ',t); t=re.sub(r'\n\s*\n+','\n',t)
open(b+'.txt','w').write(t)
print('chars',len(t))
PY
