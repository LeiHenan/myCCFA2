import sys, re, html, urllib.request, os, subprocess

def fetch(url, out=None):
    r = subprocess.run(["curl","-sL","--max-time","45","-A","Mozilla/5.0",url],capture_output=True,text=True)
    return r.stdout

def totext(h):
    h = re.sub(r'(?is)<script.*?</script>','',h)
    h = re.sub(r'(?is)<style.*?</style>','',h)
    h = re.sub(r'(?is)<svg.*?</svg>','',h)
    h = re.sub(r'(?is)<(br|/p|/div|/li|/h[1-6]|/tr)>','\n',h)
    h = re.sub(r'(?s)<[^>]+>',' ',h)
    h = html.unescape(h)
    h = re.sub(r'[ \t]+',' ',h)
    h = re.sub(r'\n\s*\n+','\n',h)
    return h.strip()

if __name__ == "__main__":
    url = sys.argv[1]
    h = fetch(url)
    open('/tmp/last.html','w').write(h)
    t = totext(h)
    if len(sys.argv)>2:
        print(t[:int(sys.argv[2])])
    else:
        print(t[:20000])
