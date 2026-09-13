#!/usr/bin/env python3
"""Parse arXiv abs HTML pages and XML API responses into readable text records."""
import glob, html, os, re, sys

def clean(s):
    s = re.sub(r'<[^>]+>', ' ', s)
    s = html.unescape(s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_abs(path):
    raw = open(path, encoding='utf-8', errors='replace').read()
    out = {}
    m = re.search(r'<meta name="citation_title" content="([^"]*)"', raw)
    if m: out['title'] = html.unescape(m.group(1))
    m = re.search(r'<meta name="citation_date" content="([^"]*)"', raw)
    if m: out['date'] = m.group(1)
    m = re.search(r'<meta name="citation_online_date" content="([^"]*)"', raw)
    if m: out['online'] = m.group(1)
    m = re.search(r'<blockquote class="abstract[^>]*>(.*?)</blockquote>', raw, re.S)
    if m:
        out['abstract'] = clean(m.group(1)).replace('Abstract: ', '', 1)
    # submission history
    m = re.search(r'<div class="submission-history">(.*?)</div>', raw, re.S)
    if m:
        out['history'] = clean(m.group(1))[:220]
    m = re.search(r'<meta name="citation_arxiv_id" content="([^"]*)"', raw)
    out['id'] = m.group(1) if m else os.path.basename(path).replace('.html','')
    m = re.search(r'<meta name="citation_subject" content="([^"]*)"', raw)
    if m: out['subject'] = html.unescape(m.group(1))
    return out

NEG = re.compile(r"\b(negative result|does not (?:help|improve|yield|pay|translate|materialize|hold|generalize|outperform|lead|provide|reduce|exist)|fail(?:s|ed)? to|no (?:significant |real |measurable |consistent |clear )?(?:speedup|speed-up|gain|improvement|benefit|advantage|reduction|better)|not worth|underperform|worse than|degrades?|degradation|hurt(?:s)? accuracy|no free lunch|challenge|pitfall|misleading|overestimat|revisit|rethink|re-examin|myth|contrary to|surpris|unexpected|limitation|not necessary|unnecessary|ineffective|inefficien|hurts|costs outweigh|question|failed to|collapse|drop in accuracy|accuracy loss|loss of accuracy|not (?:always|necessarily)|only marginal|marginal(?:ly)? (?:improve|gain|benefit)|no better)\b", re.I)

def main():
    files = sorted(glob.glob('abs/*.html'))
    for f in files:
        d = parse_abs(f)
        ab = d.get('abstract','')
        hits = sorted(set(x.group(0).lower() for x in NEG.finditer(ab)))
        print('='*100)
        print('ID:', d.get('id'), '| DATE:', d.get('date'), '| online:', d.get('online',''))
        print('TITLE:', d.get('title'))
        print('HISTORY:', d.get('history','')[:160])
        print('NEGHITS:', ', '.join(hits))
        print('ABSTRACT:', ab)
        print()

if __name__ == '__main__':
    main()
