import sys,re,html,base64
h=open(sys.argv[1],encoding='utf-8',errors='replace').read()
parts=h.split('class="b_algo"')[1:]
seen=set()
n=0
for b in parts:
    # real target URL
    urls=re.findall(r'u=a1([A-Za-z0-9_\-]+)',b)
    url=None
    for u in urls:
        s=u.replace('-','+').replace('_','/')
        s+='='*(-len(s)%4)
        try:
            d=base64.b64decode(s).decode('utf-8',errors='replace')
        except Exception: continue
        if d.startswith('http') and 'bing.com' not in d:
            url=d; break
    if not url:
        m=re.search(r'<h2[^>]*>\s*<a[^>]*href="(http[^"]+)"',b)
        url=html.unescape(m.group(1)) if m else None
    if not url: continue
    if url in seen: continue
    seen.add(url)
    m=re.search(r'<h2[^>]*>(.*?)</h2>',b,flags=re.S)
    title=re.sub('<[^>]+>','',html.unescape(m.group(1))).strip() if m else '?'
    txt=re.sub('<[^>]+>',' ',b)
    txt=re.sub(r'\s+',' ',html.unescape(txt)).strip()
    n+=1
    print(f'[{n}] {title}')
    print(f'    {url}')
    print(f'    {txt[:400]}')
    print()
