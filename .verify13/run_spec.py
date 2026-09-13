#!/usr/bin/env python3
"""Run the full claim spec: every substantive quote in S13 mapped to its cited page."""
import json, os, sys, re, collections
sys.path.insert(0, '/Users/leihenan/Desktop/myProject/.verify13')
from vcheck import check, CORPUS

BASE = '/Users/leihenan/Desktop/myProject/.verify13'
hashmap = {}
for line in open(os.path.join(BASE, 'urllog.tsv')):
    p = line.rstrip('\n').split('\t')
    hashmap[p[0]] = p[4]

def pages_for(urls):
    out = []
    for u in urls:
        out.append(hashmap[u])
        if 'arxiv.org' in u:
            aid = u.rsplit('/', 1)[1].replace('.', '_')
            if os.path.exists(os.path.join(BASE, 'pages', 'html_' + aid + '.txt')):
                out.append('html_' + aid)
    return out

URLS = {
 'A1': 'https://github.com/vllm-project/vllm/pull/42645',
 'A2': 'https://github.com/NVIDIA/TensorRT-LLM/pull/18711',
 'A3': 'https://arxiv.org/abs/2510.18672',
 'A4': 'https://arxiv.org/abs/2608.23658',
 'A5': 'https://arxiv.org/abs/2608.00902',
 'A6': 'https://github.com/sgl-project/sglang/pull/21064',
 'A7': 'https://github.com/vllm-project/vllm/pull/51098',
 'A8': 'https://github.com/vllm-project/vllm/issues/37823',
 'A9': 'https://github.com/vllm-project/vllm/issues/29286',
 'A10': 'https://github.com/vllm-project/vllm/issues/42185',
 'A11': 'https://github.com/vllm-project/vllm/issues/40533',
 'A12': 'https://github.com/NVIDIA/TensorRT-LLM/pull/15828',
 'A13': 'https://github.com/sgl-project/sglang/pull/38720',
 'A14': 'https://arxiv.org/abs/2601.14279',
 'A15': 'https://arxiv.org/abs/2608.20397',
 'A16': 'https://arxiv.org/abs/2606.04557',
 'A17': 'https://arxiv.org/abs/2605.09490',
 'A18': 'https://arxiv.org/abs/2609.11744',
 'A19': 'https://github.com/sgl-project/sglang/pull/20088',
 'C1': 'https://github.com/sgl-project/sglang/pull/29173',
 'C2': 'https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py',
 'C3': 'https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md',
 'C4': 'https://github.com/vllm-project/vllm/pull/52041',
 'C5': 'https://arxiv.org/abs/2607.21604',
 'C6': 'https://arxiv.org/abs/2602.22603',
 'C7': 'https://arxiv.org/abs/2505.15347',
 'C8': 'https://arxiv.org/abs/2509.17396',
 'C9': 'https://arxiv.org/abs/2609.03494',
 'C10': 'https://arxiv.org/abs/2605.18825',
 'C11': 'https://arxiv.org/abs/2506.02634',
 'C12': 'https://arxiv.org/abs/2603.08743',
 'C13': 'https://arxiv.org/abs/2601.21927',
 'O1': 'https://github.com/vllm-project/vllm/pull/52629',
 'O2': 'https://github.com/vllm-project/vllm/pull/54625',
 'O3': 'https://github.com/vllm-project/vllm/issues/53749',
 'O4': 'https://github.com/vllm-project/vllm/pull/53395',
 'O4b': 'https://github.com/vllm-project/vllm/issues/54607',
 'O5': 'https://github.com/sgl-project/sglang/pull/37359',
 'O6': 'https://github.com/sgl-project/sglang/pull/38000',
 'O7': 'https://arxiv.org/abs/2607.08032',
 'O8': 'https://github.com/NVIDIA/TensorRT-LLM/pull/16252',
 'O9': 'https://github.com/vllm-project/vllm/issues/53670',
 'H1': 'https://arxiv.org/abs/2608.14624',
 'H2': 'https://arxiv.org/abs/2604.25899',
 'H3': 'https://arxiv.org/abs/2509.17396',
 'H4': 'https://arxiv.org/abs/2609.11744',
 'H5': 'https://arxiv.org/abs/2609.02027',
 'H6': 'https://arxiv.org/abs/2608.25683',
 'H7': 'https://arxiv.org/abs/2607.27187',
 'H8': 'https://arxiv.org/abs/2608.00902',
}

spec = json.load(open(os.path.join(BASE, os.environ.get('SPEC','spec.json'))))
res = []
for row in spec:
    pages = pages_for([URLS[k] for k in row['src']])
    r = check(row['quote'], pages)
    r['row'] = row['row']
    r['kind'] = row['kind']
    res.append(r)
    print(f"{row['row']:<28} {r['status']:<28} ratio={r['ratio']} where={r['where']}")
json.dump(res, open(os.path.join(BASE, os.environ.get('OUT','claim_results.json')), 'w'), indent=1)
