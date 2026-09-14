#!/bin/bash
for id in "$@"; do
  out="fetch/v_$id.html"
  code=$(curl -sL --retry 3 --retry-delay 2 --retry-all-errors --max-time 40 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" -o "$out" -w "%{http_code}" "https://arxiv.org/abs/$id")
  python3 - "$out" "$id" "$code" <<'PY'
import re,sys,html
p,i,code=sys.argv[1],sys.argv[2],sys.argv[3]
t=open(p,encoding='utf-8',errors='ignore').read()
def c(s): return html.unescape(re.sub('<[^>]+>','',s)).strip()
m=re.search(r'<h1 class="title[^"]*">(.*?)</h1>',t,re.S)
ti=c(m.group(1)) if m else 'NOT FOUND'
m=re.search(r'<td class="tablecell comments[^"]*">(.*?)</td>',t,re.S)
co=c(m.group(1))[:120] if m else '-'
m=re.search(r'\[Submitted on ([^\]<]+)\]',t)
sub=m.group(1) if m else '-'
print(f"[{code}] {i} | {ti} | COMMENTS: {co} | SUBMITTED: {sub}")
PY
done
