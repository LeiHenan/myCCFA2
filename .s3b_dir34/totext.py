import re,sys,html
p=sys.argv[1]; out=sys.argv[2]
t=open(p,encoding='utf-8',errors='ignore').read()
t=re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>',' ',t)
t=re.sub(r'(?i)</(p|div|li|h1|h2|h3|h4|tr|section)>','\n',t)
t=re.sub(r'<[^>]+>',' ',t)
t=html.unescape(t)
t=re.sub(r'[ \t\xa0]+',' ',t)
t=re.sub(r'\n\s*\n+','\n',t)
open(out,'w',encoding='utf-8').write(t)
print(out, len(t))
