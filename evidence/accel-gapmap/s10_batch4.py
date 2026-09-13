#!/usr/bin/env python3
"""S10 batch driver wave 4 — TRT-LLM heavy + follow-ups."""
import sys
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from s10_srch import rows

QUERIES = [
    # ---- TensorRT-LLM ----
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue benchmark unfair"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue trtllm-bench"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue benchmark measurement"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue comparison with vllm"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue speedup claim"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue benchmark script"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue perf regression variance"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue unable to reproduce performance"),
    ("NVIDIA/TensorRT-LLM", "pulls", "is:pr is:closed is:unmerged benchmark"),
    ("NVIDIA/TensorRT-LLM", "pulls", "is:pr is:closed is:unmerged perf"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue benchmark wrong number"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue latency measurement wrong"),
    # ---- SGLang follow-ups ----
    ("sgl-project/sglang", "issues", "is:issue benchmark misleading"),
    ("sgl-project/sglang", "issues", "is:issue bench_serving metrics wrong"),
    ("sgl-project/sglang", "issues", "is:issue benchmark inaccurate"),
    ("sgl-project/sglang", "issues", "is:issue warmup request benchmark latency"),
    ("sgl-project/sglang", "pulls", "is:pr is:closed benchmark metric fix"),
    ("sgl-project/sglang", "pulls", "is:pr is:closed is:unmerged benchmark"),
    # ---- vLLM follow-ups ----
    ("vllm-project/vllm", "issues", "is:issue benchmark misleading number"),
    ("vllm-project/vllm", "issues", "is:issue warmup request first request latency"),
    ("vllm-project/vllm", "issues", "is:issue throughput measurement wrong"),
    ("vllm-project/vllm", "issues", "is:issue prefix cache benchmark docs pitfall"),
    ("vllm-project/vllm", "pulls", "is:pr is:closed prefix cache benchmark warn"),
    ("vllm-project/vllm", "pulls", "is:pr is:closed bench serve docs warmup"),
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
