#!/bin/bash
# GitHub search API sweep for abandoned/negative-result KV cache work.
OUT=/Users/leihenan/Desktop/myProject/kv_discovery
mkdir -p "$OUT/raw"

run() {
  local name="$1"
  local q="$2"
  local page="${3:-1}"
  curl -s -G "https://api.github.com/search/issues" \
    --data-urlencode "q=$q" \
    --data-urlencode "per_page=100" \
    --data-urlencode "page=$page" \
    --data-urlencode "sort=created" \
    --data-urlencode "order=desc" \
    -o "$OUT/raw/${name}.json"
  local n
  n=$(jq -r '.total_count // "ERR"' "$OUT/raw/${name}.json" 2>/dev/null)
  echo "[$name] total=$n  q=$q"
  sleep 8
}

### ---------- vllm-project/vllm ----------
R=vllm-project/vllm
run vllm_pr_kvcache_title  "repo:$R is:pr is:unmerged \"kv cache\" in:title created:>=2025-06-01"
run vllm_pr_prefixcache     "repo:$R is:pr is:unmerged \"prefix cache\" in:title created:>=2025-06-01"
run vllm_pr_offload         "repo:$R is:pr is:unmerged offload in:title created:>=2025-06-01"
run vllm_pr_eviction        "repo:$R is:pr is:unmerged eviction in:title created:>=2025-06-01"
run vllm_pr_cache_broad     "repo:$R is:pr is:unmerged cache in:title created:>=2025-06-01"
run vllm_iss_notplanned_kv  "repo:$R is:issue is:closed reason:\"not planned\" \"kv cache\" in:title created:>=2025-06-01"
run vllm_iss_notplanned_cache "repo:$R is:issue is:closed reason:\"not planned\" cache in:title created:>=2025-06-01"
run vllm_iss_wontfix        "repo:$R is:issue label:wontfix"
run vllm_iss_stale_kv       "repo:$R is:issue is:closed label:stale kv"
run vllm_pr_hicache         "repo:$R is:pr is:unmerged hicache in:title"
run vllm_pr_disagg          "repo:$R is:pr is:unmerged disaggregated in:title created:>=2025-06-01"
run vllm_pr_radix           "repo:$R is:pr is:unmerged radix in:title created:>=2025-06-01"

### ---------- sgl-project/sglang ----------
R=sgl-project/sglang
run sglang_pr_kvcache_title "repo:$R is:pr is:unmerged \"kv cache\" in:title created:>=2025-06-01"
run sglang_pr_prefixcache   "repo:$R is:pr is:unmerged \"prefix cache\" in:title created:>=2025-06-01"
run sglang_pr_offload       "repo:$R is:pr is:unmerged offload in:title created:>=2025-06-01"
run sglang_pr_eviction      "repo:$R is:pr is:unmerged eviction in:title created:>=2025-06-01"
run sglang_pr_cache_broad   "repo:$R is:pr is:unmerged cache in:title created:>=2025-06-01"
run sglang_iss_notplanned_kv "repo:$R is:issue is:closed reason:\"not planned\" \"kv cache\" in:title created:>=2025-06-01"
run sglang_iss_notplanned_cache "repo:$R is:issue is:closed reason:\"not planned\" cache in:title created:>=2025-06-01"
run sglang_iss_wontfix      "repo:$R is:issue label:wontfix"
run sglang_iss_stale_kv     "repo:$R is:issue is:closed label:stale kv"
run sglang_pr_hicache       "repo:$R is:pr is:unmerged hicache in:title"
run sglang_pr_disagg        "repo:$R is:pr is:unmerged disaggregated in:title created:>=2025-06-01"
run sglang_pr_radix         "repo:$R is:pr is:unmerged radix in:title created:>=2025-06-01"

### ---------- NVIDIA/TensorRT-LLM ----------
R=NVIDIA/TensorRT-LLM
run trtllm_pr_kvcache_title "repo:$R is:pr is:unmerged \"kv cache\" in:title created:>=2025-06-01"
run trtllm_pr_cache_broad   "repo:$R is:pr is:unmerged cache in:title created:>=2025-06-01"
run trtllm_iss_notplanned_kv "repo:$R is:issue is:closed reason:\"not planned\" \"kv cache\" in:title created:>=2025-06-01"
run trtllm_iss_wontfix      "repo:$R is:issue label:wontfix"

### ---------- flashinfer-ai/flashinfer ----------
R=flashinfer-ai/flashinfer
run flashinfer_pr_kvcache_title "repo:$R is:pr is:unmerged \"kv cache\" in:title created:>=2025-06-01"
run flashinfer_pr_cache_broad   "repo:$R is:pr is:unmerged cache in:title created:>=2025-06-01"
run flashinfer_iss_notplanned_kv "repo:$R is:issue is:closed reason:\"not planned\" \"kv cache\" in:title created:>=2025-06-01"
run flashinfer_iss_wontfix      "repo:$R is:issue label:wontfix"

### ---------- LMCache/LMCache ----------
R=LMCache/LMCache
run lmcache_pr_kvcache_title "repo:$R is:pr is:unmerged \"kv cache\" in:title created:>=2025-06-01"
run lmcache_pr_cache_broad   "repo:$R is:pr is:unmerged cache in:title created:>=2025-06-01"
run lmcache_iss_notplanned_kv "repo:$R is:issue is:closed reason:\"not planned\" \"kv cache\" in:title created:>=2025-06-01"
run lmcache_iss_wontfix      "repo:$R is:issue label:wontfix"

### ---------- ai-dynamo/dynamo ----------
R=ai-dynamo/dynamo
run dynamo_pr_kvcache_title "repo:$R is:pr is:unmerged \"kv cache\" in:title created:>=2025-06-01"
run dynamo_pr_cache_broad   "repo:$R is:pr is:unmerged cache in:title created:>=2025-06-01"
run dynamo_iss_notplanned_kv "repo:$R is:issue is:closed reason:\"not planned\" \"kv cache\" in:title created:>=2025-06-01"
run dynamo_iss_wontfix      "repo:$R is:issue label:wontfix"

### ---------- deepspeedai/DeepSpeed ----------
R=deepspeedai/DeepSpeed
run ds_pr_kvcache_title "repo:$R is:pr is:unmerged \"kv cache\" in:title created:>=2025-06-01"
run ds_iss_notplanned_kv "repo:$R is:issue is:closed reason:\"not planned\" \"kv cache\" in:title created:>=2025-06-01"
run ds_iss_wontfix      "repo:$R is:issue label:wontfix"

### ---------- volcengine/verl (volcengine/veRL) ----------
R=volcengine/verl
run verl_pr_kvcache_title "repo:$R is:pr is:unmerged \"kv cache\" in:title created:>=2025-06-01"
run verl_iss_notplanned_kv "repo:$R is:issue is:closed reason:\"not planned\" \"kv cache\" in:title created:>=2025-06-01"
run verl_iss_wontfix      "repo:$R is:issue label:wontfix"

### ---------- huggingface/transformers ----------
R=huggingface/transformers
run hf_pr_kvcache_title "repo:$R is:pr is:unmerged \"kv cache\" in:title created:>=2025-06-01"
run hf_pr_cache_broad   "repo:$R is:pr is:unmerged cache in:title created:>=2025-06-01"
run hf_iss_notplanned_kv "repo:$R is:issue is:closed reason:\"not planned\" \"kv cache\" in:title created:>=2025-06-01"
run hf_iss_wontfix      "repo:$R is:issue label:wontfix"

echo "=== SWEEP DONE ==="
