import sys,re,html
for p in sys.argv[1:]:
    s=open(p,encoding='utf-8',errors='ignore').read()
    s=re.sub(r'(?is)<script.*?</script>','',s)
    s=re.sub(r'(?s)<[^>]+>',' ',s); s=html.unescape(s); s=re.sub(r'[ \t]+',' ',s)
    lines=[l.strip() for l in s.split('\n') if l.strip()]
    st='?'
    for i,l in enumerate(lines):
        if l in ('Open','Closed','Merged','Draft') and i>5:
            st=l; break
    for i,l in enumerate(lines):
        if l.startswith('Closed as'):
            st=l; break
    m=re.search(r'merged \d+ commits? into',s)
    mm=re.search(r'wants to merge \d+ commits? into',s)
    extra = 'MERGED' if m else ('unmerged' if mm else '')
    print(f"{p}: state={st} {extra}")
