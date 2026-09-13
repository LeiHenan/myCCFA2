#!/usr/bin/env bash
id="$1"
h="html_$(printf '%s' "$id" | tr '.' '_')"
url="https://arxiv.org/html/${id}v1"
res=$(bash /Users/leihenan/Desktop/myProject/.verify13/fetch.sh "$url" "/Users/leihenan/Desktop/myProject/.verify13/pages/$h.html")
code=$(printf '%s' "$res" | sed -n 's/.*HTTP=\([0-9]*\).*/\1/p')
bytes=$(printf '%s' "$res" | sed -n 's/.*bytes=\([0-9]*\).*/\1/p')
printf '%s\t%s\t%s\n' "$id" "$code" "$bytes" >> /Users/leihenan/Desktop/myProject/.verify13/htmllog.tsv
python3 - "/Users/leihenan/Desktop/myProject/.verify13/pages/$h.html" "/Users/leihenan/Desktop/myProject/.verify13/pages/$h.txt" <<'PY'
import sys,re,html
src,dst=sys.argv[1],sys.argv[2]
raw=open(src,'rb').read().decode('utf-8','replace')
raw=re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>',' ',raw)
t=re.sub(r'(?s)<[^>]+>',' ',raw)
t=html.unescape(t)
t=re.sub(r'[ \t\xa0]+',' ',t)
t=re.sub(r'\n\s*\n+','\n',t)
open(dst,'w').write(t)
PY
