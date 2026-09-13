#!/usr/bin/env python3
"""S12: generate + run a large wave of GitHub global-search queries, dump parsed rows."""
import subprocess, sys, os, json
from urllib.parse import quote

HERE = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap"

REPOS = {
 "trt": "NVIDIA/TensorRT-LLM",
 "llc": "ggml-org/llama.cpp",
 "fa": "Dao-AILab/flash-attention",
 "fi": "flashinfer-ai/flashinfer",
 "tgi": "huggingface/text-generation-inference",
 "tf": "huggingface/transformers",
 "opt": "huggingface/optimum",
 "cut": "NVIDIA/CUTLASS",
 "pt": "pytorch/pytorch",
 "ds": "microsoft/DeepSpeed",
 "mii": "microsoft/DeepSpeed-MII",
 "vfa": "vllm-project/flash-attention",
 "te": "NVIDIA/TransformerEngine",
 "awq": "casper-hansen/AutoAWQ",
 "ex2": "turboderp-org/exllamav2",
 "flex": "FMInference/FlexGen",
 "pi": "SJTU-IPADS/PowerInfer",
 "slm": "mit-han-lab/streaming-llm",
 "tri": "openai/triton",
 "ft": "NVIDIA/FasterTransformer",
 "fgen": "microsoft/DeepSpeed-FastGen",
 "lmd": "InternLM/lmdeploy",
 "mlc": "mlc-ai/mlc-llm",
}

# (repo-key-or-None, extra query text)
QUERIES = []
def q(repo, text):
    QUERIES.append((repo, text))

# --- TensorRT-LLM: removed / deprecated / unmerged perf work
for t in ['"we are removing"', '"has been deprecated"', '"is deprecated"', '"we are deprecating"',
          '"not planned"', '"won\'t fix"', '"no plans to support"', '"we do not plan to support"',
          'is:pr is:closed is:unmerged "revert"', 'is:pr is:closed is:unmerged attention plugin',
          'is:pr is:closed is:unmerged "remove" plugin', 'is:pr is:closed is:unmerged cuda graph',
          'is:pr is:closed is:unmerged "flash attention"', 'is:issue is:closed wontfix',
          'is:pr is:closed is:unmerged "in-flight batching"', 'deprecated attention',
          '"XQA"', '"remove XQA"', '"we will not"', '"out of scope"', '"not supported"']:
    q("trt", t)

# --- llama.cpp CUDA/backend perf (NOT kv cache / NOT spec decode)
for t in ['is:pr is:closed is:unmerged CUDA graph', 'is:pr is:closed is:unmerged MMQ',
          'is:pr is:closed is:unmerged cuBLAS', 'is:pr is:closed is:unmerged tensor core',
          'is:pr is:closed is:unmerged flash attention', 'is:pr is:closed is:unmerged "-fa"',
          'is:pr is:closed is:unmerged ubatch', 'is:pr is:closed is:unmerged "n-gpu-layers"',
          'is:pr is:closed is:unmerged scheduler overhead', 'is:pr is:closed is:unmerged backend scheduler',
          'is:issue is:closed wontfix CUDA', 'is:issue is:closed wontfix performance',
          '"GGML_CUDA_FORCE_MMQ"', '"won\'t fix"', '"not planned"', '"we will not"',
          'is:pr is:closed is:unmerged "graph" cuda', 'is:issue is:open CUDA performance',
          'is:pr is:closed is:unmerged "remove" backend', 'is:pr is:closed is:unmerged "revert"']:
    q("llc", t)

# --- flash-attention: sm120 / Blackwell / FA3 / FA4 gating
for t in ['sm120', 'Blackwell', '"we do not plan to support"', '"no plans"', 'FA3',
          'is:pr is:closed is:unmerged', 'is:issue is:closed "not supported"',
          'is:issue is:open sm120', 'is:issue is:open "consumer"', '"sm_120"', '"RTX 50"',
          '"5090"', 'is:issue is:open FA4', '"Hopper only"', '"does not support"']:
    q("fa", t)

# --- flashinfer: sm120 / sm90 gating
for t in ['sm120', 'is:pr is:closed is:unmerged', '"not currently supported"', '"no plans"',
          'is:issue is:open sm120', 'sm90 only', '"we do not"', 'Blackwell', 'is:issue is:open unsupported',
          '"not supported"']:
    q("fi", t)

# --- TGI
for t in ['is:pr is:closed is:unmerged performance', 'is:pr is:closed is:unmerged router',
          '"maintenance mode"', '"maintenance"', 'is:issue is:closed "not planned"',
          '"no longer"', 'deprecat', 'is:pr is:closed is:unmerged "revert"',
          'is:pr is:closed is:unmerged kernel', 'is:issue is:open performance']:
    q("tgi", t)

# --- transformers / optimum
for t in ['BetterTransformer', '"torch.fx"', 'is:pr is:closed is:unmerged inference performance',
          '"optimum-nvidia"', 'deprecat', '"removed"', 'is:issue is:closed wontfix performance',
          'is:pr is:closed is:unmerged "flash attention"', '"no longer supported"']:
    q("tf", t)
for t in ['BetterTransformer', '"optimum-nvidia"', 'deprecat', 'is:pr is:closed is:unmerged',
          'is:issue is:open "not supported"']:
    q("opt", t)

# --- CUTLASS / pytorch / DeepSpeed / TE
for t in ['is:issue is:closed sm120', 'sm120', 'is:issue is:open sm120', '"not supported"',
          'is:pr is:closed is:unmerged sm120', '"we do not plan"']:
    q("cut", t)
for t in ['FlexAttention limitation', 'is:issue is:closed wontfix "cuda graph"',
          'is:issue is:closed wontfix inductor', '"by design" cuda graph',
          'is:issue is:open FlexAttention', '"inference mode" torch.compile',
          'is:issue is:closed "not planned" cuda graph']:
    q("pt", t)
for t in ['is:issue is:open "FastGen"', '"MII"', 'deprecat', '"no longer"',
          'is:pr is:closed is:unmerged inference', '"not maintained"']:
    q("ds", t)
for t in ['"not supported"', 'sm120', 'is:issue is:open sm120', 'deprecat', '"no plans"']:
    q("te", t)
for t in ['is:pr is:closed is:unmerged', 'sm120', '"not supported"', 'is:issue is:open']:
    q("vfa", t)

# --- abandoned projects: verbatim deprecation
for t in ['deprecat', '"no longer maintained"', '"not maintained"', 'archiv']:
    q("awq", t)
    q("ex2", t)
    q("pi", t)
    q("ft", t)
    q("mii", t)
for t in ['"no longer maintained"', 'archiv', 'deprecat']:
    q("flex", t)
    q("slm", t)
    q("fgen", t)
for t in ['is:issue is:closed wontfix attention', 'attention tutorial', 'deprecat']:
    q("tri", t)
for t in ['deprecat', '"not supported"', 'is:issue is:open', '"no plans"']:
    q("lmd", t)
    q("mlc", t)

lines = []
for rk, text in QUERIES:
    full = (f"repo:{REPOS[rk]} " if rk else "") + text
    url = "https://github.com/search?q=" + quote(full) + "&type=issues"
    lines.append(url)

with open(sys.argv[1], "w") as f:
    f.write("\n".join(lines) + "\n")
print(len(lines), "queries")
