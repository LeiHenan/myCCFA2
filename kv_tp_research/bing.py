#!/usr/bin/env python3
import sys, re, html, subprocess, urllib.parse, time
def search(q, count=20):
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(q) + f"&count={count}&setlang=en&cc=US"
    raw = subprocess.run(["curl","-sL","--retry","3","--retry-delay","2","--retry-all-errors","--max-time","45",
        "-A","Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
        url], capture_output=True, text=True).stdout
    blocks = raw.split('class="b_algo"')[1:]
    out=[]
    for b in blocks:
        m = re.search(r'<h2>\s*<a[^>]*href="([^"]+)"', b)
        if not m:
            m = re.search(r'href="(http[^"]+)"', b)
        if not m: continue
        u = html.unescape(m.group(1))
        t = re.search(r'<h2>(.*?)</h2>', b, re.S)
        title = html.unescape(re.sub('<[^>]+>','',t.group(1))).strip() if t else ''
        txt = html.unescape(re.sub('<[^>]+>',' ', b))
        txt = re.sub(r'\s+',' ',txt).strip()
        out.append((title,u,txt[:350]))
    return out
if __name__ == "__main__":
    for q in sys.argv[1:]:
        print("### QUERY:", q)
        r = search(q)
        if not r: print("  [0 results]")
        for t,u,s in r:
            print(f"* {t}\n  {u}\n  {s}\n")
        print()
