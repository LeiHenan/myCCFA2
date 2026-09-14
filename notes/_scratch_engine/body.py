import re,html,sys
f=sys.argv[1]
s=open(f,encoding='utf-8',errors='replace').read()
m=re.search(r'<title>(.*?)</title>',s,re.S); print("TITLE:",html.unescape(m.group(1))[:300])
# state
for pat in [r'State:\s*<[^>]*>([^<]{2,30})<', r'>(Merged|Closed|Open|Draft)<', r'"state":"(open|closed|merged|MERGED|OPEN|CLOSED)"', r'>\s*(Merged|Closed|Open)\s*<']:
    for mm in re.finditer(pat,s):
        print("STATE-HIT:",mm.group(1)); break
bodies=re.findall(r'"body":"((?:[^"\\]|\\.){80,12000})"',s)
for b in bodies[:1]:
    b=b.replace('\\n','\n').replace('\\"','"').replace('\\u003c','<').replace('\\u003e','>').replace('\\u0026','&')
    print("---BODY---"); print(b[:5000])
print("=== comment bodies:")
n=0
for cm in re.finditer(r'<div class="comment-body[^"]*"[^>]*>(.*?)</div>',s,re.S):
    t=re.sub(r'<[^>]+>',' ',cm.group(1)); t=html.unescape(re.sub(r'\s+',' ',t)).strip()
    if len(t)>60:
        n+=1; print(f"  [{n}]",t[:900])
    if n>=6: break
