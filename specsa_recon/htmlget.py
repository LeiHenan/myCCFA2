import sys,os,subprocess,re,time,html
OUT="/Users/leihenan/Desktop/myProject/specsa_recon/raw"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
def get(url,name):
    p=os.path.join(OUT,name)
    if not (os.path.exists(p) and os.path.getsize(p)>5000):
        subprocess.run(["curl","-sL","--max-time","50","--retry","3","--retry-delay","2","--retry-all-errors","-A",UA,url,"-o",p])
    return p if os.path.exists(p) else None
def txt(p):
    t=open(p,encoding='utf-8',errors='replace').read()
    t=re.sub(r'<script.*?</script>','',t,flags=re.S)
    t=re.sub(r'<style.*?</style>','',t,flags=re.S)
    t=re.sub(r'<[^>]+>',' ',t)
    t=html.unescape(t)
    return re.sub(r'\s+',' ',t)
if __name__=="__main__":
    mode=sys.argv[1]
    for spec in sys.argv[2:]:
        url,name=spec.split("::")
        p=get(url,name)
        if not p: print("FAIL",url); continue
        t=txt(p)
        open(p+".txt","w").write(t)
        print("### %s -> %s (%d chars text)"%(url,name,len(t)))
