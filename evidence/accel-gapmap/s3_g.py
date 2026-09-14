#!/usr/bin/env python3
"""S3: regex-grep a stripped-HTML text file with context. Usage: s3_g.py <file> <pattern> [n]"""
import re, sys
f, pat = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
t = open(f).read()
t = re.sub(r"\s+", " ", t)
t = re.sub(r"\[\d+(,\s*\d+)*\]", "", t)
hits = list(re.finditer(pat, t, re.I))
print(f"# {f} :: /{pat}/ -> {len(hits)}")
for m in hits[:n]:
    a = max(0, m.start()-260); b = min(len(t), m.end()+320)
    print("...", t[a:b].strip(), "\n")
