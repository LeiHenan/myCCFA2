#!/usr/bin/env python3
import sys, concurrent.futures as cf
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from e5_srch import search
QUERIES = [
    ("sgl-project/sglang", "return_sampling_mask"),
    ("sgl-project/sglang", "ForwardBatch ScheduleBatch aliasing"),
    ("sgl-project/sglang", "seq_lens family removal"),
    ("sgl-project/sglang", "kv-committed lengths"),
    ("sgl-project/sglang", "blocksparse FA4 cute kernel"),
    ("sgl-project/sglang", "delayed sampling hide sampling latency"),
    ("sgl-project/sglang", "mixed chunked prefill input logprob"),
    ("sgl-project/sglang", "priority scheduling lpm policy"),
    ("vllm-project/vllm", "priority scheduling lpm lof"),
    ("pytorch/pytorch", "numpy array indexing performance regression block_table"),
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
