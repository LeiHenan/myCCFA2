#!/usr/bin/env python3
"""Given fragments, find the best matching page + surrounding context (normalized search)."""
import sys, os, json, re
sys.path.insert(0, '/Users/leihenan/Desktop/myProject/.verify13')
from vcheck import CORPUS, norm, norm_code

def find(frag, prefer=None):
    q = norm(frag)
    hits = []
    for name, text in CORPUS.items():
        if q and q in text:
            i = text.find(q)
            hits.append((name, text[max(0, i - 200): i + len(q) + 240]))
    if hits and prefer:
        hits.sort(key=lambda h: 0 if h[0].startswith(prefer) or prefer in h[0] else 1)
    return hits

if __name__ == '__main__':
    for spec in json.load(open(sys.argv[1])):
        print('=' * 100)
        print(f"{spec['row']}  fragment: {spec['frag']!r}   prefer={spec.get('prefer')}")
        h = find(spec['frag'], spec.get('prefer'))
        if not h:
            print('   >>> NOT FOUND anywhere in corpus')
        for name, ctx in h[:3]:
            print(f'   [{name}] ...{ctx}...')
