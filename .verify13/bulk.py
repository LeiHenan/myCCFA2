#!/usr/bin/env python3
"""Bulk cross-page verbatim search.

For every quote extracted from S13.md, search ALL retrieved page texts
(abstract pages, fulltext HTML, PDF text) for the exact normalized stream.
Reports which page (if any) contains it. This establishes ground truth
independent of which page the agent claimed to read.
"""
import os, re, sys, json, unicodedata, glob

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

def norm_loose(s):
    """Extra-aggressive: also drop spaces around hyphens and strip all hyphens,
    to survive PDF extraction artifacts ('Photonic - CXL')."""
    s = norm(s)
    s = s.replace(' - ', '-').replace('- ', '-').replace(' -', '-')
    s = s.replace(' ', '')
    return s

# load all corpus texts
corpus = {}
for p in sorted(glob.glob(os.path.join(PAGES, '*.txt'))):
    name = os.path.basename(p)[:-4]
    corpus[name] = norm(open(p, 'rb').read().decode('utf-8', 'replace'))
loose = {k: norm_loose(v) for k, v in corpus.items()}
print(f'# corpus pages: {len(corpus)}', file=sys.stderr)

out = []
for line in open(os.path.join(BASE, 'quotes_extracted.txt')):
    parts = line.rstrip('\n').split('\t')
    if len(parts) < 4:
        continue
    sec, ln, kind, quote = parts[0], parts[1], parts[2], '\t'.join(parts[3:])
    q = norm(quote)
    ql = norm_loose(quote)
    hits = []
    for k, v in corpus.items():
        if len(q) > 20 and q in v:
            hits.append(('exact', k))
    if not hits:
        for k, v in loose.items():
            if len(ql) > 20 and ql in v:
                hits.append(('exact-loose', k))
    if not hits:
        # try fragment split for compound quotes
        frs = [f for f in re.split(r'\.\s|\band\b', q) if len(f) > 30]
        if len(frs) > 1:
            for k, v in corpus.items():
                if all(f in v for f in frs):
                    hits.append(('all-fragments', k))
    out.append({'sec': sec, 'line': ln, 'quote': quote, 'hits': hits})

json.dump(out, open(os.path.join(BASE, 'bulk_quotes.json'), 'w'), indent=1)
nf = sum(1 for r in out if not r['hits'])
print(f'total quotes: {len(out)}  verbatim-found: {len(out)-nf}  NOT-found-anywhere: {nf}', file=sys.stderr)
for r in out:
    tag = ','.join(f'{a}:{b}' for a, b in r['hits']) or 'NOT-FOUND'
    print(f"{r['sec']}\t{r['line']}\t{tag[:70]}\t{r['quote'][:95]}")
