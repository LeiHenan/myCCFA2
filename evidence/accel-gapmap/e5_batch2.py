#!/usr/bin/env python3
"""E5 round-2 targeted searches."""
import sys, concurrent.futures as cf
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from e5_srch import search

QUERIES = [
    ("sgl-project/sglang", "logprob mixed decode prefill batch"),
    ("sgl-project/sglang", "enable_mixed_chunk"),
    ("sgl-project/sglang", "priority scheduling lpm routing key"),
    ("sgl-project/sglang", "needs_cpu_seq_lens"),
    ("sgl-project/sglang", "delayed sampling overlap scheduler default"),
    ("sgl-project/sglang", "parallel_sample_num"),
    ("sgl-project/sglang", "batch and parallel sampling not optimized"),
    ("sgl-project/sglang", "mcp tool server streaming responses"),
    ("sgl-project/sglang", "logprob_start_len session"),
    ("sgl-project/sglang", "utilization metric stuck"),
    ("sgl-project/sglang", "time_func_latency prometheus async"),
    ("sgl-project/sglang", "FA4 rotary sm120"),
    ("sgl-project/sglang", "block sparse attention SM120"),
    ("sgl-project/sglang", "sliding window triton attention backend"),
    ("sgl-project/sglang", "get_cos_sin_with_position mrope"),
    ("sgl-project/sglang", "prefill-aware SWA page size 1"),
    ("sgl-project/sglang", "fast_prefill_plan"),
    ("sgl-project/sglang", "page size 1 allocator extend"),
    ("sgl-project/sglang", "SWA allocator page size"),
    ("sgl-project/sglang", "API key multi tokenizer"),
    ("sgl-project/sglang", "grammar backend none json schema"),
    ("sgl-project/sglang", "xgrammar unsupported tokenizer"),
    ("sgl-project/sglang", "sampling observer"),
    ("sgl-project/sglang", "sampling mask speculative decoding overlap"),
    ("sgl-project/sglang", "streaming sessions logprob KV"),
    ("sgl-project/sglang", "memcpy cpu dp attention"),
    ("sgl-project/sglang", "elastic ep active_ranks host sync"),
    ("sgl-project/sglang", "jump forward decoding disabled"),
    ("sgl-project/sglang", "host overhead per step scheduler"),
    ("sgl-project/sglang", "sm120 rtpro 6000 attention backend support"),
    ("sgl-project/sglang", "detokenizer worker batch decode"),
    ("sgl-project/sglang", "Qwen3-4B sm120"),
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


if __name__ == "__main__":
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run, q) for q in QUERIES]
        for f in cf.as_completed(futs):
            print(f.result(), flush=True)
