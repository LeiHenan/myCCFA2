#!/usr/bin/env python3
"""S10 batch driver wave 3 — in:title precision queries."""
import sys
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from s10_srch import rows

QUERIES = [
    # ---- vLLM in:title ----
    ("vllm-project/vllm", "issues", "is:issue in:title benchmark sort:updated-desc"),
    ("vllm-project/vllm", "issues", "is:issue in:title \"bench serve\""),
    ("vllm-project/vllm", "issues", "is:issue in:title benchmark noise"),
    ("vllm-project/vllm", "issues", "is:issue in:title variance"),
    ("vllm-project/vllm", "issues", "is:issue in:title warmup"),
    ("vllm-project/vllm", "issues", "is:issue in:title \"prefix caching\" throughput"),
    ("vllm-project/vllm", "issues", "is:issue in:title retract"),
    ("vllm-project/vllm", "issues", "is:issue in:title \"not reproducible\""),
    ("vllm-project/vllm", "issues", "is:issue in:title throughput measurement"),
    ("vllm-project/vllm", "issues", "is:issue in:title speedup"),
    # ---- SGLang ----
    ("sgl-project/sglang", "issues", "is:issue in:title benchmark"),
    ("sgl-project/sglang", "issues", "is:issue in:title warmup"),
    ("sgl-project/sglang", "issues", "is:issue in:title variance"),
    ("sgl-project/sglang", "issues", "is:issue in:title \"prefix cache\""),
    ("sgl-project/sglang", "issues", "is:issue in:title reproducible"),
    ("sgl-project/sglang", "issues", "is:issue in:title speedup"),
    ("sgl-project/sglang", "issues", "is:issue in:title flush-cache"),
    ("sgl-project/sglang", "issues", "is:issue in:title ignore_eos"),
    ("sgl-project/sglang", "pulls", "is:pr in:title benchmark"),
    ("sgl-project/sglang", "pulls", "is:pr in:title warmup"),
    # ---- TRT-LLM ----
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue in:title benchmark"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue in:title benchmark wrong"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue in:title perf comparison"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue in:title throughput"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue in:title variance"),
    ("NVIDIA/TensorRT-LLM", "issues", "is:issue in:title reproducible"),
    ("NVIDIA/TensorRT-LLM", "pulls", "is:pr in:title benchmark"),
    ("NVIDIA/TensorRT-LLM", "pulls", "is:pr in:title warmup"),
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
