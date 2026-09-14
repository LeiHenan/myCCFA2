#!/usr/bin/env python3
import sys,json,urllib.parse,subprocess,re,html
q=sys.argv[1]
lim=sys.argv[2] if len(sys.argv)>2 else "30"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
url="https://huggingface.co/api/papers/search?q="+urllib.parse.quote(q)
out=subprocess.run(["curl","-sL","--max-time","45","--retry","3","--retry-delay","2","-A",UA,url],capture_output=True,text=True).stdout
try:
    d=json.loads(out)
except Exception as e:
    print("PARSE_FAIL",out[:200]); sys.exit()
if isinstance(d,dict):
    print("KEYS",list(d.keys())); d=d.get('papers',d.get('results',[]))
print(f"### QUERY: {q}  -> {len(d)} results")
for it in d[:int(lim)]:
    p=it.get('paper',it)
    pid=p.get('id'); title=p.get('title'); pub=p.get('publishedAt','')[:10]
    abst=(p.get('abstract') or p.get('summary') or '')
    print(f"\n[{pid}] {pub}\n  {title}\n  {(abst[:900]).replace(chr(10),' ')}")
