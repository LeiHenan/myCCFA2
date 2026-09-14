#!/bin/bash
# usage: fetch_abs.sh <arxivid> ; prints title/authors/date/abstract-head
id="$1"
f="abs_${id}.html"
curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" --max-time 60 "https://arxiv.org/abs/${id}" -o "$f" -w "  HTTP:%{http_code} SIZE:%{size_download}\n"
python3 - "$f" "$id" <<'PY'
import re,html,sys
f,i=sys.argv[1],sys.argv[2]
try: s=open(f,encoding='utf-8',errors='replace').read()
except Exception as e: print(i,'READ-FAIL',e); sys.exit()
m=re.search(r'<title>(.*?)</title>',s,re.S)
print('  ARXIV-ID-REQUESTED:',i)
print('  TITLE-READBACK   :',html.unescape(m.group(1)).strip() if m else 'NONE')
au=re.findall(r'<meta name="citation_author" content="(.*?)"',s)
print('  AUTHORS          :',', '.join(html.unescape(a) for a in au) if au else 'NONE')
d=re.findall(r'<meta name="citation_date" content="(.*?)"',s)
print('  DATE             :',d[0] if d else 'NONE')
m=re.search(r'<blockquote class="abstract mathjax">(.*?)</blockquote>',s,re.S)
if m:
    t=re.sub(r'<[^>]+>','',m.group(1)); t=html.unescape(t); t=re.sub(r'\s+',' ',t).strip()
    print('  ABSTRACT(900)    :',t[:900])
PY
