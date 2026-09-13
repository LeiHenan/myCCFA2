import re,sys,html
for f in sys.argv[1:]:
    t=open(f,encoding='utf-8',errors='ignore').read()
    # search result links look like href="/owner/repo/issues/123"
    seen=set()
    print("=== ",f)
    for m in re.finditer(r'href="(/[^/"]+/[^/"]+/(?:issues|pull)/\d+)"[^>]*>(.*?)</a>', t, re.S):
        u,txt=m.group(1),re.sub(r'<[^>]+>','',m.group(2))
        txt=html.unescape(txt).strip()
        if u in seen or not txt: continue
        seen.add(u)
        print(f"https://github.com{u}\t{txt[:200]}")
