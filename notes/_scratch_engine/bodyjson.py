import re,sys,json,html
f=sys.argv[1]
s=open(f,encoding='utf-8',errors='replace').read()
m=re.search(r'<title>(.*?)</title>',s,re.S); print("TITLE:",html.unescape(m.group(1))[:250])
# Try robust brace-matching extraction of a JSON object containing "body"
out=[]
i=0
while True:
    i=s.find('"body":"',i+1)
    if i<0: break
    j=i+8; depth=0; buf=[]
    while j<len(s):
        c=s[j]
        if c=='\\': buf.append(s[j:j+2]); j+=2; continue
        if c=='"': break
        buf.append(c); j+=1
    raw=''.join(buf)
    try: txt=json.loads('"'+raw+'"')
    except Exception: txt=raw
    out.append(txt)
for k,t in enumerate(out):
    if len(t)>60:
        print(f"---BODY[{k}] len={len(t)}---"); print(t[:6000]); print()
