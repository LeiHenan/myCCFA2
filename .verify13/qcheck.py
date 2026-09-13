#!/usr/bin/env python3
"""Normalized verbatim quote checker.

For each claimed quote, search the retrieved page text (and optionally the
arXiv full-text HTML) for the exact character stream after normalizing
whitespace/punctuation/unicode. Report exact / approximate / not found.
"""
import os, re, sys, json, html, difflib, unicodedata

BASE = '/Users/leihenan/Desktop/myProject/.verify13'
PAGES = os.path.join(BASE, 'pages')

def norm(s):
    s = unicodedata.normalize('NFKD', s)
    s = s.replace('\u2019', "'").replace('\u2018', "'")
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2014', '--').replace('\u2013', '-')
    s = s.replace('\u2212', '-').replace('\u00a0', ' ')
    s = s.lower()
    s = re.sub(r'[^a-z0-9%$./+-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def load_text(path):
    if not os.path.exists(path):
        return ''
    raw = open(path, 'rb').read().decode('utf-8', 'replace')
    return norm(raw)

def find_quote(quote, sources):
    """sources: list of (label, text). Returns (status, label, extra)."""
    q = norm(quote)
    if not q:
        return ('EMPTY', '', '')
    # 1. exact substring
    for label, text in sources:
        if text and q in text:
            return ('FOUND EXACTLY', label, '')
    # 2. try splitting on ellipsis / bracketed elision
    parts = [p for p in re.split(r'\.\.\.|\u2026', q) if len(p.strip()) > 8]
    if len(parts) > 1:
        for label, text in sources:
            if text and all(p.strip() in text for p in parts):
                return ('FOUND EXACTLY (with elision)', label, f'{len(parts)} fragments')
    # 3. approximate: sliding window ratio on best source
    best = ('NOT FOUND', '', 0.0)
    for label, text in sources:
        if not text:
            continue
        # locate the most distinctive 8-gram of the quote
        words = q.split()
        if len(words) < 4:
            continue
        anchor = ' '.join(words[:8])
        positions = [m.start() for m in re.finditer(re.escape(anchor), text)]
        if not positions:
            # fall back to scanning windows of the same length
            wl = min(len(q), 4000)
            step = max(1, (len(text) - wl) // 400 or 1)
            cands = [text[i:i+wl] for i in range(0, max(1, len(text) - wl), step)]
            cands.append(text[:wl])
        else:
            wl = len(q)
            cands = [text[max(0, p - wl // 3): p - wl // 3 + wl * 2] for p in positions[:8]]
        for c in cands:
            r = difflib.SequenceMatcher(None, q, c).ratio()
            if r > best[2]:
                best = ('FOUND APPROXIMATELY', label, round(r, 4))
    if best[2] >= 0.62:
        return best
    if best[2] > 0:
        return ('NOT FOUND', best[1], f'best-ratio={best[2]}')
    return ('NOT FOUND', '', '')

def sources_for(arxiv_id, page_hash, use_html=True):
    srcs = []
    for suffix in ('.txt',):
        p = os.path.join(PAGES, page_hash + suffix)
        if os.path.exists(p):
            srcs.append(('abs-page', load_text(p)))
    href = None
    if arxiv_id:
        hp = os.path.join(PAGES, 'html_' + arxiv_id.replace('.', '_') + '.txt')
        if os.path.exists(hp):
            href = load_text(hp)
            srcs.append(('fulltext-html', href))
    return srcs, href

if __name__ == '__main__':
    spec = json.load(open(sys.argv[1]))
    out = []
    for row in spec:
        srcs, _ = sources_for(row.get('arxiv'), row['hash'])
        status, label, extra = find_quote(row['quote'], srcs)
        rec = dict(row_id=row['id'], status=status, where=label, extra=extra)
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
