# -*- coding: utf-8 -*-
"""Generate _out_C.md — Section C (attempted-and-abandoned cells).

Quotes and URLs are pulled character-for-character from _in_C.tsv unless a
verification report supplies corrected wording (literal overrides)."""
import csv, re, sys

SRC = '_in_C.tsv'
OUT = '_out_C.md'

SEC_TITLES = {
    'C.1': 'C.1 Closed-unmerged pull requests',
    'C.2': 'C.2 wontfix / not-planned / by-design / out-of-scope closures',
    'C.3': 'C.3 Stale-closed (automated) — no human reason on record',
    'C.4': 'C.4 Reverted, withdrawn, or retracted by the author',
    'C.5': 'C.5 Negative results: measured, and it did not pay off',
    'C.6': 'C.6 Papers whose own limitations undercut their technique',
}

ROWS = {}
ORDER = []
for r in csv.DictReader(open(SRC, encoding='utf-8'), delimiter='\t'):
    k = r['seg'] + '-' + r['id']
    ROWS[k] = r
    ORDER.append(k)

from _gen_C_util import R, T

FIXES = {
    'S6-ABANDONED-12': [('should still be open', 'should remain open')],
    'S11-ABANDONED-21': [('That gate was', 'that gate was')],
    'S5-ABANDONED-8': [('treated a setup specific', 'treated as a setup specific')],
    'S2-ABANDONED-9': [('treated a setup specific', 'treated as a setup specific')],
    'S11-ABANDONED-44': [('treated a setup specific', 'treated as a setup specific')],
    'S13-ABANDONED-18': [('treated a setup specific', 'treated as a setup specific')],
}

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

def quote_part(spec):
    kind, val = spec
    if kind == 't':
        return val
    q = ROWS[val]['quote']
    for a, b in FIXES.get(val, []):
        q = q.replace(a, b)
    return q


META = ['"', 'VERBATIM', 'verbatim', 'the file', 'no maintainer', 'No maintainer',
        'also present', 'Also present', 'preceded by', 'followed by', 'Companion',
        'Related', 'Corroborating', 'No human', 'No stated', 'No closure',
        'not retrievable', 'TITLE ONLY', 'the row', '(sic', 'per the paper',
        'explicitly', 'no closing comment', 'the RFC', 'the page', 'the paper']

def is_pure_quote(txt):
    """True only when the evidence field is a bare quote with no agent framing,
    so that framing text is never placed inside quotation marks."""
    return not any(m in txt for m in META)

def render(items):
    out = ['## C. ATTEMPTED-AND-ABANDONED cells (most valuable!)', '']
    n = 0
    cur = None
    used = set()
    for it in items:
        if it['sec'] != cur:
            cur = it['sec']
            out.append('### ' + SEC_TITLES[cur])
            out.append('')
        n += 1
        keys = it['u']
        used.update(keys)
        prim = keys[0]
        urls = ' | '.join(urls_of(keys))
        who = it.get('w') or ROWS[prim]['who']
        date = it.get('d') or ROWS[prim]['date']
        ev = it.get('e') or ev_of(prim)
        parts = []
        specs = it.get('q')
        if not specs:
            specs = [R(prim)]
        labels = it.get('qlab') or []
        for j, spec in enumerate(specs):
            txt = quote_part(spec)
            body = '"%s"' % txt if is_pure_quote(txt) else txt
            if j == 0:
                parts.append(body)
            else:
                lab = labels[j - 1] if j - 1 < len(labels) and labels[j - 1] else 'also'
                parts.append('%s: %s' % (lab, body))
        reason = ' — and, '.join(parts)
        if it.get('approx'):
            reason += ' (approximate — see verification note)'
        out.append('#### C%d. %s' % (n, it['h']))
        out.append('- **WHO:** %s' % who)
        out.append('- **URL:** %s' % urls)
        out.append('- **Closure status:** %s' % it['c'])
        out.append('- **STATED REASON, VERBATIM:** ' + reason)
        out.append('- **Date:** %s' % date)
        out.append('- **Evidence:** %s' % ev)
        out.append('')
    return '\n'.join(out), n, used

from _gen_C_items1 import ITEMS1
from _gen_C_items2 import ITEMS2
from _gen_C_items3 import ITEMS3
from _gen_C_items4 import ITEMS4
from _gen_C_items5 import ITEMS5
from _gen_C_items5b import ITEMS5B
from _gen_C_items6 import ITEMS6

ITEMS = ITEMS1 + ITEMS2 + ITEMS3 + ITEMS4 + ITEMS5 + ITEMS5B + ITEMS6

md, n, used = render(ITEMS)
open(OUT, 'w', encoding='utf-8').write(md + '\n')

missing = [k for k in ORDER if k not in used]
print('items:', n)
print('rows covered:', len(used), 'of', len(ORDER))
print('rows not covered (%d): %s' % (len(missing), ', '.join(missing)))
print('output bytes:', len(md))
for s in sorted(SEC_TITLES):
    print(s, sum(1 for it in ITEMS if it['sec'] == s))
