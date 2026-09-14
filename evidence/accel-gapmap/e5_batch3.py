#!/usr/bin/env python3
"""E5 round-3: remaining round-1 queries + precision probes."""
import sys, concurrent.futures as cf
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from e5_srch import search

QUERIES = [
    ("sgl-project/sglang", "tokenizer not supported by XGrammar"),
    ("sgl-project/sglang", "grammar backend disabled"),
    ("sgl-project/sglang", "return_sampling_mask speculative decoding overlap scheduling"),
    ("sgl-project/sglang", "detokenize top logprobs batch"),
    ("sgl-project/sglang", "detokenization optimization positions top-k"),
    ("sgl-project/sglang", "parallel sampling n greater than 1 batch throughput"),
    ("sgl-project/sglang", "length validation unify scheduler tokenizer"),
    ("sgl-project/sglang", "item sync allocation request"),
    ("sgl-project/sglang", "numpy block_table indexing performance regression"),
    ("sgl-project/sglang", "return_token_ids streaming chat completions"),
    ("sgl-project/sglang", "responses api streaming logprobs include"),
    ("sgl-project/sglang", "max_running_requests_under_SLO"),
    ("sgl-project/sglang", "prometheus_client async timer decorator"),
    ("sgl-project/sglang", "flash attention 4 rotary embedding"),
    ("sgl-project/sglang", "blocksparse SM120"),
    ("sgl-project/sglang", "kv_indptr sliding window"),
    ("sgl-project/sglang", "2D positions RoPE base"),
    ("sgl-project/sglang", "prefill aware swa unified memory"),
    ("sgl-project/sglang", "custom_mask prefill plan"),
    ("sgl-project/sglang", "alloc_extend paged allocator"),
    ("sgl-project/sglang", "MHATokenToKOnlyPool"),
    ("sgl-project/sglang", "multi-tokenizer API key"),
    ("sgl-project/sglang", "LogitsProcessorOutput workaround fields"),
    ("sgl-project/sglang", "sampling observer not supported"),
    ("sgl-project/sglang", "chunked prefill disabled multimodal"),
    ("sgl-project/sglang", "return_logprob mixed chunked prefill not supported"),
    ("sgl-project/sglang", "TBO seq_lens_cpu requirement"),
    ("sgl-project/sglang", "delayed sample"),
    ("sgl-project/sglang", "rtx pro 6000 sm120 attention"),
    ("vllm-project/vllm", "return logprobs mixed batch chunked prefill"),
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
