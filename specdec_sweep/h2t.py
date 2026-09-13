import sys,re,html
p=sys.argv[1]
s=open(p,encoding='utf-8',errors='ignore').read()
s=re.sub(r'(?is)<script.*?</script>','',s)
s=re.sub(r'(?is)<style.*?</style>','',s)
s=re.sub(r'(?is)<(br|/p|/div|/li|/h[1-6]|/tr)>','\n',s)
s=re.sub(r'(?s)<[^>]+>',' ',s)
s=html.unescape(s)
s=re.sub(r'[ \t]+',' ',s)
s=re.sub(r'\n\s*\n+','\n',s)
print(s.strip())
