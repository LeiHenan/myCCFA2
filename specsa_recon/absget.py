import sys,re,html,os,subprocess,time,json
OUT="/Users/leihenan/Desktop/myProject/specsa_recon/raw"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
def fetch(i):
    p=os.path.join(OUT,"abs_%s.html"%i)
    if not (os.path.exists(p) and os.path.getsize(p)>5000):
        subprocess.run(["curl","-sL","--max-time","40","--retry","2","--retry-delay","2","--retry-all-errors","-A",UA,"https://arxiv.org/abs/%s"%i,"-o",p])
    if not os.path.exists(p) or os.path.getsize(p)<5000: return None
    t=open(p,encoding='utf-8',errors='replace').read()
    def m(pat):
        r=re.search(pat,t,re.S)
        return html.unescape(r.group(1)).strip() if r else None
    return {"id":i,
      "title":m(r'<meta name="citation_title" content="([^"]+)"'),
      "date":m(r'<meta name="citation_date" content="([^"]+)"'),
      "abstract":m(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>') or m(r'<meta name="citation_abstract" content="([^"]+)"'),
      "comments":m(r'<td class="tablecell comments[^"]*">(.*?)</td>'),
      "journal":m(r'<td class="tablecell journal-ref[^"]*">(.*?)</td>')}
res=[]
for i in sys.argv[1:]:
    d=fetch(i)
    if d:
        d['abstract']=re.sub(r'<[^>]+>','',d['abstract'] or '')
        d['abstract']=re.sub(r'\s+',' ',d['abstract']).replace('Abstract:','').strip()
        d['comments']=re.sub(r'<[^>]+>','',d['comments'] or '').strip()
        res.append(d)
        print("### %s | %s | %s | COMMENTS: %s"%(d['id'],d['title'],d['date'],d['comments']))
        print("ABS:",d['abstract'][:2500]); print()
    else:
        print("### %s | FETCH FAILED"%i)
    time.sleep(1)
json.dump(res,open(os.path.join(OUT,"meta_%s.json"%sys.argv[1].replace('.','_')),'w'),indent=1)
