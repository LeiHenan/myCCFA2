#!/usr/bin/env bash
# one.sh <url> : fetch url, log url/code/effective/hash, save html+text
url="$1"
h=$(printf '%s' "$url" | md5)
out="/Users/leihenan/Desktop/myProject/.verify13/pages/$h.html"
res=$(bash /Users/leihenan/Desktop/myProject/.verify13/fetch.sh "$url" "$out")
code=$(printf '%s' "$res" | sed -n 's/.*HTTP=\([0-9]*\).*/\1/p')
eff=$(printf '%s' "$res" | sed -n 's/.*url=\(.*\)$/\1/p')
bytes=$(printf '%s' "$res" | sed -n 's/.*bytes=\([0-9]*\).*/\1/p')
printf '%s\t%s\t%s\t%s\t%s\n' "$url" "$code" "$eff" "$bytes" "$h" >> /Users/leihenan/Desktop/myProject/.verify13/urllog.tsv
# make plain text version for grepping
python3 - "$out" "/Users/leihenan/Desktop/myProject/.verify13/pages/$h.txt" <<'PY'
import sys,re,html
src,dst=sys.argv[1],sys.argv[2]
try:
    raw=open(src,'rb').read().decode('utf-8','replace')
except Exception as e:
    open(dst,'w').write(''); sys.exit(0)
# strip script/style
raw=re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>',' ',raw)
t=re.sub(r'(?s)<[^>]+>',' ',raw)
t=html.unescape(t)
t=re.sub(r'[ \t\xa0]+',' ',t)
t=re.sub(r'\n\s*\n+','\n',t)
open(dst,'w').write(t)
PY
