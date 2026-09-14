#!/usr/bin/env python3
"""Fetch arxiv abs pages, print VERIFIED title/date/abstract head. Never trust an ID without this."""
import sys,re,html,subprocess
for aid in sys.argv[1:]:
    url=f"https://arxiv.org/abs/{aid}"
    raw=subprocess.run(["curl","-sL","--retry","4","--retry-delay","2","--retry-all-errors","--max-time","60",
      "-A","Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",url],
      capture_output=True,text=True).stdout
    t=re.search(r'<title>(.*?)</title>',raw,re.S)
    title=html.unescape(re.sub(r'\s+',' ',re.sub('<[^>]+>','',t.group(1)))).strip() if t else 'NO TITLE'
    sub=re.search(r'\[Submitted on ([^\]<]+)\]',raw)
    abs_m=re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>',raw,re.S)
    abstract=html.unescape(re.sub(r'\s+',' ',re.sub('<[^>]+>','',abs_m.group(1)))).strip() if abs_m else ''
    ok='arxiv.org' in raw and title!='NO TITLE'
    print(f"--- {aid}  ||  {title}")
    print(f"    submitted: {sub.group(1) if sub else 'N/A'}")
    print(f"    abstract[:700]: {abstract[:700]}")
    print()
