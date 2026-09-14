import json,sys,time,os,urllib.parse,subprocess
OUT="/Users/leihenan/Desktop/myProject/specsa_recon/raw"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
def get(url,key,tries=6):
    p=os.path.join(OUT,key)
    if os.path.exists(p) and os.path.getsize(p)>50:
        try:
            d=json.load(open(p))
            if 'data' in d: return d
        except Exception: pass
    for i in range(tries):
        r=subprocess.run(["curl","-sL","--max-time","45","-A",UA,url],capture_output=True,text=True)
        t=r.stdout
        if t and 'Too Many Requests' not in t and 'Rate exceeded' not in t:
            try:
                d=json.loads(t); json.dump(d,open(p,'w'),indent=1); return d
            except Exception as e: pass
        time.sleep(6+i*5)
    return None
def search(q,limit=20):
    url="https://api.semanticscholar.org/graph/v1/paper/search?query=%s&fields=title,externalIds,year,venue,abstract,authors&limit=%d"%(urllib.parse.quote(q),limit)
    key="s2_"+urllib.parse.quote(q,safe='').replace('%','')[:80]+".json"
    d=get(url,key)
    if not d: print("!! FAIL",q); return
    print("=== %s  (total=%s)"%(q,d.get('total')))
    for p in d.get('data',[]):
        ex=p.get('externalIds') or {}
        print("  | %s || %s || %s || arXiv:%s DOI:%s"%(p.get('title'),p.get('year'),(p.get('venue') or '')[:40],ex.get('ArXiv'),ex.get('DOI')))
    sys.stdout.flush()
    time.sleep(5)
if __name__=="__main__":
    qs=json.load(open(sys.argv[1]))
    for q in qs: search(q)
