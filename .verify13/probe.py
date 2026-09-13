#!/usr/bin/env python3
"""Targeted quote probe with fuzzy diagnostics.

usage: probe.py "<quote>" <page-hash-or-htmlid> [...]
       probe.py --grep "<regex fragment>" <page>
"""
import os, re, sys, unicodedata, difflib, glob

BASE = '/Users/leihenan/Desktop/myProject/.verify13'
PAGES = os.path.join(BASE, 'pages')

def norm(s):
    s = unicodedata.normalize('NFKD', s)
    for a, b in [('\u2019', "'"), ('\u2018', "'"), ('\u201c', '"'), ('\u201d', '"'),
                 ('\u2014', '--'), ('\u2013', '-'), ('\u2212', '-'), ('\u00a0', ' ')]:
        s = s.replace(a, b)
    s = s.lower()
    s = re.sub(r'[^a-z0-9%$./+-]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def load(name):
    for cand in (name, 'html_' + name, 'html_' + name.replace('.', '_')):
        p = os.path.join(PAGES, cand + '.txt')
        if os.path.exists(p):
            return norm(open(p, 'rb').read().decode('utf-8', 'replace')), cand
    return None, None

def probe(quote, name, width=3):
    q = norm(quote)
    text, used = load(name)
    if text is None:
        print(f'  [{name}] PAGE MISSING'); return
    if q in text:
        print(f'  [{used}] FOUND EXACTLY'); return
    words = q.split()
    # try several anchors across the quote
    best = (0.0, '')
    for start in range(0, max(1, len(words) - 6), 3):
        anchor = ' '.join(words[start:start + 8])
        for m in list(re.finditer(re.escape(anchor), text))[:3]:
            wl = len(q)
            c = text[max(0, m.start() - wl // 4): m.start() - wl // 4 + int(wl * 1.6)]
            r = difflib.SequenceMatcher(None, q, c).ratio()
            if r > best[0]:
                best = (r, c)
    print(f'  [{used}] best-ratio={round(best[0],4)}')
    if best[0] > 0:
        sm = difflib.SequenceMatcher(None, q, best[1])
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != 'equal':
                print(f'      {tag}: claimed[{q[i1:i2]!r}] page[{best[1][j1:j2]!r}]')

if __name__ == '__main__':
    if sys.argv[1] == '--grep':
        frag, name = sys.argv[2], sys.argv[3]
        text, used = load(name)
        print(f'[{used}] matches for /{frag}/:')
        for m in list(re.finditer(frag, text, re.I))[:10]:
            print('   ...', text[max(0, m.start() - 160): m.end() + 220].replace('\n', ' '), '...')
    else:
        quote = sys.argv[1]
        print(f'QUOTE: {quote[:110]}')
        for name in sys.argv[2:]:
            probe(quote, name)
