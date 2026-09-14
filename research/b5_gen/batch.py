import json,urllib.parse,subprocess,sys,time
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
queries=json.load(open(sys.argv[1]))
allp={}
for q in queries:
    url="https://huggingface.co/api/papers/search?q="+urllib.parse.quote(q)
    out=subprocess.run(["curl","-sL","--max-time","45","--retry","3","--retry-delay","2","-A",UA,url],capture_output=True,text=True).stdout
    try: d=json.loads(out)
    except Exception: print("FAIL",q,out[:120]); continue
    if isinstance(d,dict): d=d.get('papers',[])
    for it in d:
        p=it.get('paper',it)
        pid=p.get('id')
        if pid and pid not in allp:
            allp[pid]={'title':p.get('title'),'pub':(p.get('publishedAt') or '')[:10],'abs':(p.get('abstract') or p.get('summary') or ''),'q':q}
    time.sleep(0.6)
json.dump(allp,open(sys.argv[2],'w'),indent=1)
print("total unique papers:",len(allp))
