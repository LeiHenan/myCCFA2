#!/usr/bin/env python3
"""S10 batch driver: run many repo list-page queries, print compact candidate rows."""
import sys, json
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from s10_srch import rows

# (repo, kind, query)
QUERIES = [
    # ---- vLLM ----
    ("vllm-project/vllm", "issues", "is:issue benchmark prefix caching throughput inflated"),
    ("vllm-project/vllm", "issues", "is:issue benchmark wrong methodology"),
    ("vllm-project/vllm", "issues", "is:issue misleading benchmark"),
    ("vllm-project/vllm", "issues", "is:issue benchmark warmup"),
    ("vllm-project/vllm", "issues", "is:issue benchmark variance run to run"),
    ("vllm-project/vllm", "issues", "is:issue benchmark not reproducible"),
    ("vllm-project/vllm", "issues", "is:issue ignore_eos benchmark"),
    ("vllm-project/vllm", "issues", "is:issue enforce-eager benchmark"),
    ("vllm-project/vllm", "issues", "is:issue benchmark_serving"),
    ("vllm-project/vllm", "issues", "is:issue benchmark noisy speedup"),
    ("vllm-project/vllm", "pulls", "is:pr is:closed is:unmerged benchmark"),
    ("vllm-project/vllm", "pulls", "is:pr is:closed is:unmerged benchmark_serving"),
    ("vllm-project/vllm", "pulls", "is:pr is:closed benchmark warmup num-warmups"),
    ("vllm-project/vllm", "pulls", "is:pr is:closed benchmark noise variance"),
    # ---- SGLang ----
    ("sgl-project/sglang", "issues", "is:issue benchmark wrong prefix cache"),
    ("sgl-project/sglang", "issues", "is:issue benchmark warmup variance"),
    ("sgl-project/sglang", "issues", "is:issue ignore_eos benchmark"),
    ("sgl-project/sglang", "issues", "is:issue bench_serving"),
    ("sgl-project/sglang", "issues", "is:issue not reproducible performance benchmark"),
    ("sgl-project/sglang", "issues", "is:issue misleading benchmark"),
    ("sgl-project/sglang", "pulls", "is:pr is:closed benchmark fix warmup"),
    ("sgl-project/sglang", "pulls", "is:pr is:closed is:unmerged benchmark"),
    ("sgl-project/sglang", "pulls", "is:pr is:closed flush-cache benchmark"),
    # ---- TensorRT-LLM ----
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue benchmark wrong latency"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue benchmark not reproducible"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue misleading benchmark"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue benchmark unfair comparison"),
    ("NVIDIA/TensorRT-LLM", "pulls", "is:pr is:closed benchmark fix warmup"),
    ("NVIDIA/TensorRT-LLM", "pulls", "is:pr is:closed is:unmerged benchmark"),
]

ALL = len(QUERIES)
for i, (repo, kind, q) in enumerate(QUERIES, 1):
    try:
        url, r = rows(repo, q, kind)
    except Exception as e:
        print(f"\n### FAIL {repo} {kind} {q}: {e}")
        continue
    print(f"\n### [{i}/{ALL}] {repo} {kind} :: {q} -> {len(r)}")
    print(f"### URL: {url}")
    for num, k, t in r:
        print(f"{k}/{num}\t{t}")
    sys.stdout.flush()
