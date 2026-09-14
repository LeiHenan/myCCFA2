#!/bin/bash
# fetch arxiv abs page, print title + abstract head + comments
# usage: ./ax.sh 2403.09636 [name]
set -u
ID="$1"
NAME="${2:-$1}"
OUT="/Users/leihenan/Desktop/myProject/.s3b_dir34/fetch/${NAME}.html"
code=$(curl -sL --retry 4 --retry-delay 2 --retry-all-errors \
  -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" \
  -o "$OUT" -w "%{http_code}" "https://arxiv.org/abs/${ID}")
echo "### https://arxiv.org/abs/${ID}  [HTTP ${code}]  -> ${OUT}"
python3 - "$OUT" <<'PY'
import re,sys,html
p=sys.argv[1]
try:
    t=open(p,encoding='utf-8',errors='ignore').read()
except Exception as e:
    print("READ FAIL",e); raise SystemExit
def clean(s):
    s=re.sub(r'<[^>]+>','',s)
    return html.unescape(s).strip()
m=re.search(r'<h1 class="title[^"]*">(.*?)</h1>',t,re.S)
print("TITLE:", clean(m.group(1)) if m else "NOT FOUND")
m=re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>',t,re.S)
print("ABSTRACT:", (clean(m.group(1))[:1800] if m else "NOT FOUND"))
m=re.search(r'<td class="tablecell comments[^"]*">(.*?)</td>',t,re.S)
if m: print("COMMENTS:", clean(m.group(1))[:300])
m=re.search(r'<td class="tablecell subjects">(.*?)</td>',t,re.S)
if m: print("SUBJECTS:", clean(m.group(1))[:200])
m=re.search(r'\[Submitted on ([^\]<]+)\]',t)
if m: print("SUBMITTED:", m.group(1))
for pat in ['Withdrawn','withdrawn','Retracted']:
    if pat in t: print("FLAG:",pat,"present in page")
PY
