#!/bin/bash
# usage: axsearch.sh "query" <tag> <max>
q="$1"; tag="$2"; mx="${3:-30}"
curl -sL --max-time 50 --retry 3 --retry-delay 2 --retry-all-errors \
 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" \
 "http://export.arxiv.org/api/query?search_query=${q}&start=0&max_results=${mx}&sortBy=relevance" -o "ax_$tag.xml"
echo "=== $tag : $q ==="
python3 - "ax_$tag.xml" <<'PY'
import sys,re,xml.etree.ElementTree as ET
t=ET.parse(sys.argv[1]); ns={'a':'http://www.w3.org/2005/Atom'}
es=t.getroot().findall('a:entry',ns)
print("n=",len(es))
for e in es:
    idu=e.find('a:id',ns).text
    ti=' '.join(e.find('a:title',ns).text.split())
    pub=e.find('a:published',ns).text[:10]
    print(f"{idu.split('/abs/')[-1]} | {pub} | {ti}")
PY
