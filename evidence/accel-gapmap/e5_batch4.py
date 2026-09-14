#!/usr/bin/env python3
"""E5 round-4: rig-specific precision probes."""
import sys, concurrent.futures as cf
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from e5_srch import search

QUERIES = [
    ("sgl-project/sglang", "trtllm_mha sm120"),
    ("sgl-project/sglang", "sm120 flashinfer fallback attention backend"),
    ("sgl-project/sglang", "skip softmax trtllm prefill"),
    ("sgl-project/sglang", "return_meta_info streaming"),
    ("sgl-project/sglang", "logprob_start_len streaming"),
    ("sgl-project/sglang", "API key multi tokenizer mode"),
    ("sgl-project/sglang", "MCP tool server background mode"),
    ("sgl-project/sglang", "sampling observer unsupported path"),
    ("sgl-project/sglang", "delayed sample scheduling overlap"),
    ("sgl-project/sglang", "xgrammar construct object from string"),
    ("sgl-project/sglang", "zero_allocator torch.zeros"),
    ("sgl-project/sglang", "logits processor output passthrough fields"),
    ("sgl-project/sglang", "priority scheduling lof policy"),
    ("sgl-project/sglang", "chunked prefill budget too conservative reservation"),
    ("sgl-project/sglang", "num_reserved_tokens context length validation"),
    ("pytorch/pytorch", "numpy array indexing torch tensor performance regression"),
    ("sgl-project/sglang", "RTX PRO 6000 Blackwell sm120 backend matrix"),
    ("sgl-project/sglang", "Qwen3 dense long context yarn rope scaling"),
    ("sgl-project/sglang", "grammar backend fallback none tokenizer unsupported"),
    ("sgl-project/sglang", "scheduler cpu overhead per token"),
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
