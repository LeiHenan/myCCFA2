#!/usr/bin/env python3
"""Extract title + key metadata from every fetched page."""
import os, re, html, json, glob

P = '/Users/leihenan/Desktop/myProject/.verify13/pages'
log = {}
for line in open('/Users/leihenan/Desktop/myProject/.verify13/urllog.tsv'):
    parts = line.rstrip('\n').split('\t')
    if len(parts) >= 5:
        log[parts[4]] = {'url': parts[0], 'code': parts[1], 'eff': parts[2], 'bytes': int(parts[3])}

def meta(raw, name):
    m = re.search(r'<meta[^>]+(?:name|property)=["\']%s["\'][^>]*>' % re.escape(name), raw, re.I)
    if not m: return ''
    tag = m.group(0)
    c = re.search(r'content=["\'](.*?)["\']', tag, re.S|re.I)
    return html.unescape(c.group(1)).strip() if c else ''

rows = []
for h, info in sorted(log.items(), key=lambda kv: kv[1]['url']):
    raw = open(os.path.join(P, h + '.html'), 'rb').read().decode('utf-8', 'replace')
    t = re.search(r'<title>(.*?)</title>', raw, re.S|re.I)
    title = html.unescape(re.sub(r'\s+', ' ', t.group(1))).strip() if t else ''
    ct = meta(raw, 'citation_title') or meta(raw, 'og:title') or meta(raw, 'twitter:title')
    # detect error pages
    low = raw.lower()
    err = ''
    if 'page not found' in low or '404' in title: err = 'possible-404'
    if info['code'] != '200': err = 'http' + info['code']
    rows.append({'url': info['url'], 'code': info['code'], 'eff': info['eff'], 'bytes': info['bytes'],
                 'title': title[:220], 'citation_title': ct[:220], 'err': err, 'hash': h})

json.dump(rows, open('/Users/leihenan/Desktop/myProject/.verify13/meta.json', 'w'), indent=1)
for r in rows:
    print(f"{r['code']} | {r['url']}")
    print(f"     title: {r['title']}")
    if r['citation_title'] and r['citation_title'] != r['title']:
        print(f"     cit  : {r['citation_title']}")
    if r['eff'] != r['url']:
        print(f"     EFF  : {r['eff']}")
    if r['err']:
        print(f"     !!!! : {r['err']}")
