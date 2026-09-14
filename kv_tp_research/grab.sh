#!/bin/bash
# usage: grab.sh <url> <tag>   -> saves stripped text to pages/<tag>.txt
U="$1"; T="$2"
D=/Users/leihenan/Desktop/myProject/kv_tp_research/pages
curl -sL --retry 4 --retry-delay 2 --retry-all-errors --max-time 60 \
 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" "$U" -o "$D/$T.raw"
python3 - "$D/$T.raw" "$D/$T.txt" <<'PY'
import sys,re,html
raw=open(sys.argv[1],encoding='utf-8',errors='replace').read()
raw=re.sub(r'(?is)<(script|style|svg|noscript)[^>]*>.*?</\1>',' ',raw)
txt=html.unescape(re.sub(r'(?s)<[^>]+>',' ',raw))
txt=re.sub(r'[ \t\xa0]+',' ',txt); txt=re.sub(r'\n\s*\n+','\n',txt)
open(sys.argv[2],'w').write(txt)
print("chars:",len(txt))
PY
