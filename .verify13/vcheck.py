#!/usr/bin/env python3
"""Comprehensive verification of S13.md claims against live-fetched evidence.

Outputs a machine-readable verdict table for every extracted quote, matched
against the page the row cites, plus cross-corpus fallback.
"""
import os, re, sys, json, glob, unicodedata, difflib, html as H

BASE = '/Users/leihenan/Desktop/myProject/.verify13'
PAGES = os.path.join(BASE, 'pages')

def norm(s):
    s = unicodedata.normalize('NFKD', s)
    for a, b in [('\u2019', "'"), ('\u2018', "'"), ('\u201c', '"'), ('\u201d', '"'),
                 ('\u2014', '--'), ('\u2013', '-'), ('\u2212', '-'), ('\u00a0', ' '),
                 ('\u2265', '>='), ('\u2264', '<='), ('\u00d7', 'x')]:
        s = s.replace(a, b)
    s = s.lower()
    s = re.sub(r'\\u([0-9a-f]{4})', lambda m: chr(int(m.group(1), 16)), s)
    s = s.replace('\\n', ' ').replace('\\t', ' ').replace('\\"', '"').replace("\\'", "'")
    s = re.sub(r'[^a-z0-9%$./+->=<]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def norm_code(s):
    """Code-tolerant: drop underscores and hyphen spacing so rendered
    markdown code spans / PDF hyphen artifacts still match."""
    s = norm(s)
    s = s.replace('_', '').replace('-', '')
    s = re.sub(r"\s*'\s*", '', s)          # "pr s" -> "prs"
    s = re.sub(r'\s*=\s*', '=', s)         # "utilization =0.9"
    s = re.sub(r'\s*([,.;:!?])\s*', r'\1', s)   # "truncation ." -> "truncation."
    s = re.sub(r'\s+', ' ', s)
    return s

def load_all():
    """Return {name: normalized_text} using the richest available source per page."""
    c = {}
    for p in sorted(glob.glob(os.path.join(PAGES, '*.txt'))):
        name = os.path.basename(p)[:-4]
        variants = [p]
        gh = os.path.join(PAGES, name + '.ghjson.txt')
        if os.path.exists(gh):
            variants.append(gh)
        txt = '\n'.join(open(v, 'rb').read().decode('utf-8', 'replace') for v in variants)
        c[name] = norm(txt)
    return c

CORPUS = load_all()

def best_window(q, text):
    words = q.split()
    best = (0.0, '', '')
    step = max(3, len(words) // 8) if len(words) > 8 else 1
    anchors = [' '.join(words[i:i + 8]) for i in range(0, max(1, len(words) - 6), step)]
    if not anchors:
        anchors = [q]
    for a in anchors:
        for m in list(re.finditer(re.escape(a), text))[:2]:
            wl = len(q)
            cand = text[max(0, m.start() - wl // 4): m.start() - wl // 4 + int(wl * 1.7)]
            r = difflib.SequenceMatcher(None, q, cand).ratio()
            if r > best[0]:
                best = (r, cand, a)
    if best[0] == 0.0 and len(text) > 200:
        wl = min(len(q) * 2, len(text))
        for i in range(0, max(1, len(text) - wl), max(1, len(text) // 300)):
            cand = text[i:i + wl]
            r = difflib.SequenceMatcher(None, q, cand).ratio()
            if r > best[0]:
                best = (r, cand, 'scan')
    return best

def check(quote, names):
    q = norm(quote)
    if len(q) < 12:
        return {'status': 'TOO SHORT TO CHECK', 'where': '', 'ratio': None, 'detail': ''}
    # exact in cited pages first
    qc = norm_code(quote)
    for n in names:
        t = CORPUS.get(n) or CORPUS.get('html_' + n.replace('.', '_'))
        if t and q in t:
            return {'status': 'FOUND EXACTLY', 'where': n, 'ratio': 1.0, 'detail': ''}
    for n in names:
        t = CORPUS.get(n) or CORPUS.get('html_' + n.replace('.', '_'))
        if t and len(qc) > 20 and norm_code(t).find(qc) >= 0:
            return {'status': 'FOUND EXACTLY (code-normalized)', 'where': n, 'ratio': 1.0,
                    'detail': 'differs only in code-span/underscore/hyphen rendering'}
    # exact anywhere (helps identify source)
    for n, t in CORPUS.items():
        if q in t:
            return {'status': 'FOUND EXACTLY (other page)', 'where': n, 'ratio': 1.0, 'detail': ''}
    for n, t in CORPUS.items():
        if len(qc) > 20 and norm_code(t).find(qc) >= 0:
            return {'status': 'FOUND EXACTLY (other page, code-normalized)', 'where': n,
                    'ratio': 1.0, 'detail': 'differs only in code-span rendering'}
    # approximate in cited pages
    best = (0.0, '', '', '')
    for n in names:
        t = CORPUS.get(n) or CORPUS.get('html_' + n.replace('.', '_'))
        if not t:
            continue
        r, cand, a = best_window(q, t)
        if r > best[0]:
            best = (r, cand, n, a)
    if best[0] >= 0.80:
        return {'status': 'FOUND APPROXIMATELY', 'where': best[2], 'ratio': round(best[0], 4), 'detail': best[1][:400]}
    return {'status': 'NOT FOUND', 'where': best[2], 'ratio': round(best[0], 4), 'detail': best[1][:400]}

if __name__ == '__main__':
    spec = json.load(open(sys.argv[1]))
    res = []
    for row in spec:
        r = check(row['quote'], row.get('pages', []))
        r.update(row_id=row['id'], quote=row['quote'][:160], pages=row.get('pages', []))
        res.append(r)
        print(f"{row['id']:<16} {r['status']:<28} ratio={r['ratio']} where={r['where']}")
    json.dump(res, open(sys.argv[2], 'w'), indent=1)
