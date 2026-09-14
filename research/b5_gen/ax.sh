#!/bin/bash
# usage: ax.sh <arxivid>  -> prints TITLE, DATE, ABSTRACT (truncated)
ID="$1"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
HTML=$(curl -sL --max-time 45 --retry 3 --retry-delay 2 --retry-all-errors -A "$UA" "https://arxiv.org/abs/$ID")
echo "$HTML" | python3 -c "
import sys,re,html
h=sys.stdin.read()
m=re.search(r'<meta name=\"citation_title\" content=\"(.*?)\"',h)
print('TITLE:',html.unescape(m.group(1)) if m else 'NONE')
m=re.search(r'<meta name=\"citation_date\" content=\"(.*?)\"',h)
print('DATE:',m.group(1) if m else 'NONE')
m=re.search(r'<meta name=\"citation_online_date\" content=\"(.*?)\"',h)
print('ONLINE:',m.group(1) if m else 'NONE')
m=re.search(r'<meta name=\"citation_abstract\" content=\"(.*?)\" />',h,re.S)
print('ABS:',html.unescape(m.group(1))[:2200] if m else 'NONE')
"
