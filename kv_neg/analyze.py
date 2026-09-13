#!/usr/bin/env python3
"""Scan all fetched abs pages for negative-result language; rank and dump."""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import parse_abs

PATTERNS = [
    (r"negative result", 6),
    (r"does not (?:help|improve|yield|pay|translate|materialize|hold|generalize|outperform|lead|provide|reduce|exist|benefit|lead to)", 5),
    (r"do not (?:help|improve|yield|pay|translate|materialize|hold|generalize|outperform|lead|provide|reduce|exist|benefit)", 5),
    (r"fail(?:s|ed)? to (?:improve|help|generalize|translate|materialize|deliver|outperform|reduce|hold|pay)", 5),
    (r"no (?:significant |real |measurable |consistent |clear |meaningful )?(?:speedup|speed-up|gain|improvement|benefit|advantage|better|reduction|savings)", 5),
    (r"not worth", 5),
    (r"underperform", 5),
    (r"worse than", 5),
    (r"no free lunch", 5),
    (r"pitfall", 4),
    (r"misleading", 4),
    (r"overestimat", 4),
    (r"contrary to (?:popular|common|the )?(?:belief|wisdom|assumption|claims?)", 5),
    (r"(?:popular|common|prevailing|widely[- ]held) (?:belief|assumption|wisdom|claim)", 4),
    (r"challeng(?:e|es|ing) the (?:premise|assumption|belief|claim)", 5),
    (r"we (?:find|show|demonstrate|observe) that (?:attention|eviction|compression|quantization)[^.]{0,80}(?:not|fail|unreliable|weak|poor)", 4),
    (r"(?:weak|no|zero|negligible) correlation", 4),
    (r"unreliable", 4),
    (r"collapse", 3),
    (r"catastrophic", 4),
    (r"degrad(?:e|es|ation)", 2),
    (r"accuracy (?:loss|drop|degradation)", 3),
    (r"not (?:always|necessarily)", 3),
    (r"only marginal", 4),
    (r"marginal(?:ly)? (?:improve|gain|benefit|better)", 4),
    (r"rethink", 3),
    (r"revisit", 3),
    (r"re-?examin", 3),
    (r"myth", 5),
    (r"limitation", 2),
    (r"ineffective", 4),
    (r"unnecessary", 4),
    (r"not necessary", 5),
    (r"hurts", 4),
    (r"hidden cost", 4),
    (r"overhead", 1),
    (r"break-?even", 3),
    (r"not (?:a )?(?:free|silver bullet)", 4),
    (r"we find (?:that )?(?:no|little|limited)", 4),
    (r"question(?:s|ing)? the", 2),
    (r"suboptimal", 2),
    (r"surpris", 2),
    (r"unexpected", 2),
    (r"re-?evaluat", 3),
    (r"critique|criticiz", 4),
    (r"invalidat", 5),
    (r"does not (?:scale|deliver|justify)", 5),
    (r"insufficient", 3),
    (r"fragil", 2),
]

def analyze(path):
    d = parse_abs(path)
    ab = d.get('abstract', '') or ''
    score = 0
    hits = []
    for pat, w in PATTERNS:
        m = re.search(pat, ab, re.I)
        if m:
            score += w
            hits.append(m.group(0).lower())
    d['score'] = score
    d['hits'] = sorted(set(hits))
    return d

def main():
    recs = []
    for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'abs', '*.html')):
        d = analyze(f)
        if not d.get('abstract'): continue
        recs.append(d)
    recs.sort(key=lambda d: (-d['score'], d.get('date','')))
    print(f"# scanned {len(recs)} abstracts\n")
    for d in recs:
        dt = d.get('date','')
        print(f"[{d['score']:3d}] {d.get('id')} | {dt} | hits={','.join(d['hits'][:8])}")
        print(f"      {d.get('title')}")
    print()
    print("#"*100)
    for d in recs:
        if d['score'] < 4: continue
        print('='*100)
        print(f"ID: {d.get('id')} | DATE: {d.get('date')} | SCORE {d['score']} | HITS: {', '.join(d['hits'])}")
        print('TITLE:', d.get('title'))
        print('ABSTRACT:', d.get('abstract'))
        print()

if __name__ == '__main__':
    main()
