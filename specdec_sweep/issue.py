import sys,re,html
p=sys.argv[1]
s=open(p,encoding='utf-8',errors='ignore').read()
s=re.sub(r'(?is)<script.*?</script>','',s)
s=re.sub(r'(?is)<style.*?</style>','',s)
s=re.sub(r'(?is)<svg.*?</svg>','',s)
s=re.sub(r'(?is)<(br|/p|/div|/li|/h[1-6]|/tr|/pre|/td)>','\n',s)
s=re.sub(r'(?s)<[^>]+>',' ',s)
s=html.unescape(s)
s=re.sub(r'[ \t]+',' ',s)
s=re.sub(r'\n\s*\n+','\n',s)
lines=[l.strip() for l in s.split('\n') if l.strip()]
idx=[i for i,l in enumerate(lines) if l=='Copy link']
st=max(0,idx[0]-4) if idx else 0
en=len(lines)
for i in range(st,len(lines)):
    if lines[i].startswith('Footer') or 'Reactions are currently unavailable' in lines[i] or lines[i]=='Related Issues':
        en=i;break
print('\n'.join(lines[st:en]))
