#!/usr/bin/env python3
"""S10 batch driver wave 2."""
import sys
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from s10_srch import rows

QUERIES = [
    # ---- vLLM deeper ----
    ("vllm-project/vllm", "issues", "is:issue prefix caching inflates benchmark"),
    ("vllm-project/vllm", "issues", "is:issue benchmark throughput number wrong"),
    ("vllm-project/vllm", "issues", "is:issue benchmark error bars confidence interval"),
    ("vllm-project/vllm", "issues", "is:issue cuda graph capture first request latency benchmark"),
    ("vllm-project/vllm", "issues", "is:issue benchmark random dataset prefix"),
    ("vllm-project/vllm", "issues", "is:issue measured incorrectly"),
    ("vllm-project/vllm", "issues", "is:issue number is inflated"),
    ("vllm-project/vllm", "issues", "is:issue 2x speedup not reproducible"),
    ("vllm-project/vllm", "pulls", "is:pr bench serve"),
    ("vllm-project/vllm", "pulls", "is:pr is:closed benchmark prefix caching"),
    ("vllm-project/vllm", "pulls", "is:pr is:closed num-warmups"),
    ("vllm-project/vllm", "pulls", "is:pr is:closed ignore-eos"),
    # ---- SGLang deeper ----
    ("sgl-project/sglang", "issues", "is:issue bench_serving wrong"),
    ("sgl-project/sglang", "issues", "is:issue benchmark prefix cache hit"),
    ("sgl-project/sglang", "issues", "is:issue benchmark number inflated"),
    ("sgl-project/sglang", "issues", "is:issue speedup not reproducible"),
    ("sgl-project/sglang", "issues", "is:issue benchmark run to run variation"),
    ("sgl-project/sglang", "pulls", "is:pr bench_serving"),
    ("sgl-project/sglang", "pulls", "is:pr is:closed disable-ignore-eos"),
    ("sgl-project/sglang", "pulls", "is:pr is:closed warmup-requests"),
    # ---- TRT-LLM deeper ----
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue benchmark script bug"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue perf number wrong benchmark"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue compare vllm benchmark unfair"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue speedup not reproducible"),
    ("NVIDIA/TensorRT-LLM", "pulls", "is:pr is:closed benchmark latency fix"),
    ("NVIDIA/TensorRT-LLM", "pulls", "is:pr is:closed benchmark warmup"),
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
