#!/usr/bin/env bash
url="$1"
h="x_$(printf '%s' "$url" | md5)"
res=$(bash /Users/leihenan/Desktop/myProject/.verify13/fetch.sh "$url" "/Users/leihenan/Desktop/myProject/.verify13/pages/$h.html")
code=$(printf '%s' "$res" | sed -n 's/.*HTTP=\([0-9]*\).*/\1/p')
bytes=$(printf '%s' "$res" | sed -n 's/.*bytes=\([0-9]*\).*/\1/p')
title=$(python3 -c "
import re,html,sys
try: raw=open('/Users/leihenan/Desktop/myProject/.verify13/pages/$h.html','rb').read().decode('utf-8','replace')
except Exception: print(''); sys.exit()
m=re.search(r'<title>(.*?)</title>', raw, re.S|re.I)
print(html.unescape(re.sub(r'\s+',' ',m.group(1))).strip()[:200] if m else 'NO-TITLE')
")
printf '%s\t%s\t%s\t%s\n' "$url" "$code" "$bytes" "$title" >> /Users/leihenan/Desktop/myProject/.verify13/extralog.tsv
