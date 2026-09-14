import re,sys,html
p=sys.argv[1]
s=open(p,encoding='utf-8',errors='replace').read()
# Issue rows: href="/vllm-project/vllm/issues/NNN" ... title text
seen={}
for m in re.finditer(r'href="/([^/"]+)/([^/"]+)/(issues|pull)/(\d+)"[^>]*>(.*?)</a>', s, re.S):
    org,repo,kind,num,txt=m.groups()
    txt=re.sub(r'<[^>]+>',' ',txt)
    txt=html.unescape(re.sub(r'\s+',' ',txt)).strip()
    if not txt or len(txt)<8: continue
    seen.setdefault((kind,num),txt)
for (kind,num),t in sorted(seen.items(), key=lambda x:int(x[0][1])):
    print(f"{kind}/{num}: {t[:180]}")
