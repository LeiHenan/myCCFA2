# -*- coding: utf-8 -*-
"""Generate _out_C.md (Section C: attempted-and-abandoned cells).

Quotes and URLs are pulled character-for-character out of _in_C.tsv unless a
verification report supplies corrected wording (literal overrides below).
"""
import csv, re, io, sys

SRC = '_in_C.tsv'
OUT = '_out_C.md'

ROWS = {}
ORDER = []
for r in csv.DictReader(open(SRC, encoding='utf-8'), delimiter='\t'):
    k = r['seg'] + '-' + r['id']
    ROWS[k] = r
    ORDER.append(k)

def R(k):
    return ('r', k)

def T(s):
    return ('t', s)

def urls_of(keys):
    out = []
    for k in keys:
        raw = ROWS[k]['url']
        found = re.findall(r'https?://[^\s\)\]]+', raw)
        if not found:
            found = [raw]
        for u in found:
            u = u.rstrip('.,;')
            if u not in out:
                out.append(u)
    return out

def ev_of(k):
    e = ROWS[k]['evidence'].upper()
    return 'TITLE ONLY' if 'TITLE ONLY' in e else 'READ BODY'

def quote_part(spec, fixes):
    kind, val = spec
    if kind == 't':
        return val
    q = ROWS[val]['quote']
    for a, b in fixes.get(val, []):
        q = q.replace(a, b)
    return q

def render(items):
    fixes = {
        'S6-ABANDONED-12': [('should still be open', 'should remain open')],
        'S11-ABANDONED-21': [('That gate was', 'that gate was')],
    }
    out = []
    n = 0
    cur = None
    for it in items:
        if it['sec'] != cur:
            cur = it['sec']
            out.append('### ' + cur + '\n')
        n += 1
        keys = it['u']
        prim = keys[0]
        urls = ' | '.join(urls_of(keys))
        who = it.get('w') or ROWS[prim]['who']
        date = it.get('d') or ROWS[prim]['date']
        ev = it.get('e') or ev_of(prim)
        parts = []
        for j, spec in enumerate(it['q']):
            txt = quote_part(spec, fixes)
            if j == 0:
                parts.append('"%s"' % txt)
            else:
                lab = it.get('qlab', [])[j - 1] if it.get('qlab') else 'also'
                parts.append('%s: "%s"' % (lab, txt))
        out.append('#### C%d. %s' % (n, it['h']))
        out.append('- **WHO:** %s' % who)
        out.append('- **URL:** %s' % urls)
        out.append('- **Closure status:** %s' % it['c'])
        out.append('- **STATED REASON, VERBATIM:** ' + ' — and, '.join(parts))
        out.append('- **Date:** %s' % date)
        out.append('- **Evidence:** %s' % ev)
        out.append('')
        it['_n'] = n
    return '\n'.join(out), n
