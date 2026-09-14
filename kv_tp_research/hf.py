#!/usr/bin/env python3
"""HuggingFace papers search -> print arXiv-id + title pairs."""
import sys, re, html, subprocess, urllib.parse
def search(q):
    url="https://huggingface.co/papers?q="+urllib.parse.quote(q)
    raw=subprocess.run(["curl","-sL","--retry","3","--retry-delay","2","--retry-all-errors","--max-time","60",
      "-A","Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",url],
      capture_output=True,text=True).stdout
    seen=[]
    for m in re.finditer(r'href="/papers/(\d{4}\.\d{4,5})"[^>]*>(.*?)</a>', raw, re.S):
        pid=m.group(1); title=html.unescape(re.sub('<[^>]+>','',m.group(2))).strip()
        title=re.sub(r'\s+',' ',title)
        if pid not in [s[0] for s in seen]: seen.append((pid,title))
    if not seen:
        # fallback: ids then titles separately
        ids=[];  
        for m in re.finditer(r'/papers/(\d{4}\.\d{4,5})',raw):
            if m.group(1) not in ids: ids.append(m.group(1))
        for t in re.findall(r'<h3[^>]*>(.*?)</h3>',raw,re.S):
            seen.append(('?', re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>','',t))).strip()))
        seen=[(i,'') for i in ids]+seen
    return seen
for q in sys.argv[1:]:
    print("### HF:",q)
    for pid,t in search(q)[:25]:
        print(f"  {pid}  {t[:130]}")
    print()
