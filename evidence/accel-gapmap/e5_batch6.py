#!/usr/bin/env python3
import sys, concurrent.futures as cf
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from e5_srch import search
QUERIES = [
    ("sgl-project/sglang", "multiple stop strings detokenizer trim"),
    ("sgl-project/sglang", "trim_matched_stop"),
    ("sgl-project/sglang", "sampling mask speculative decoding unsupported"),
    ("sgl-project/sglang", "non-logits fields LogitsProcessorOutput"),
    ("sgl-project/sglang", "alloc_extend only for paged allocator"),
    ("sgl-project/sglang", "reasoning structural tag decode again xgrammar"),
    ("sgl-project/sglang", "prometheus timer decorator overhead async"),
    ("sgl-project/sglang", "mixed mode overlap record batch"),
]
def run(item):
    repo, q = item
    try:
        url, rows = search(repo, q)
    except Exception as e:
        return f"\n### [{repo}] {q}  -> ERROR {e}\n"
    out = [f"\n### [{repo}] {q}  -> {len(rows)} rows"]
    for num, state, title, u in rows:
        out.append(f"{num}\t{state}\t{title}\t{u}")
    return "\n".join(out) + "\n"
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    for f in cf.as_completed([ex.submit(run, q) for q in QUERIES]):
        print(f.result(), flush=True)
