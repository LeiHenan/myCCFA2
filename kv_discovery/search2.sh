#!/bin/bash
# Phrase sweep: GitHub issue search covers titles, bodies AND comments.
# Each hit = an issue/PR whose text or comments contain the phrase.
OUT=/Users/leihenan/Desktop/myProject/kv_discovery
mkdir -p "$OUT/raw2"

run() {
  local name="$1"; local q="$2"
  curl -s -G "https://api.github.com/search/issues" \
    --data-urlencode "q=$q" --data-urlencode "per_page=100" \
    --data-urlencode "sort=created" --data-urlencode "order=desc" \
    -o "$OUT/raw2/${name}.json"
  local n; n=$(jq -r '.total_count // "ERR"' "$OUT/raw2/${name}.json" 2>/dev/null)
  echo "[$name] total=$n"
  sleep 8
}

for R in vllm-project/vllm sgl-project/sglang; do
  T=$(echo "$R" | tr '/' '_')
  run "${T}_p_wontfix_kv"    "repo:$R \"won't fix\" \"kv cache\""
  run "${T}_p_notplanned_kv" "repo:$R \"not planned\" \"kv cache\""
  run "${T}_p_bydesign_kv"   "repo:$R \"by design\" \"kv cache\""
  run "${T}_p_outofscope"    "repo:$R \"out of scope\" cache"
  run "${T}_p_decidedagainst" "repo:$R \"we decided against\" cache"
  run "${T}_p_nolongerpursuing" "repo:$R \"no longer pursuing\" cache"
  run "${T}_p_noplansto"     "repo:$R \"no plans to\" cache"
  run "${T}_p_notworth"      "repo:$R \"not worth\" \"kv cache\""
  run "${T}_p_nobenefit"     "repo:$R \"no benefit\" \"kv cache\""
  run "${T}_p_doesnothelp"   "repo:$R \"does not help\" \"kv cache\""
  run "${T}_p_reverted"      "repo:$R reverted cache in:title"
  run "${T}_p_regression"    "repo:$R regression \"kv cache\" in:title"
done

for R in NVIDIA/TensorRT-LLM flashinfer-ai/flashinfer LMCache/LMCache ai-dynamo/dynamo deepspeedai/DeepSpeed volcengine/verl huggingface/transformers; do
  T=$(echo "$R" | tr '/' '_')
  run "${T}_p_kv_neg"   "repo:$R \"kv cache\" (\"won't fix\" OR \"by design\" OR \"not planned\" OR \"out of scope\" OR \"no longer pursuing\")"
  run "${T}_p_cache_neg" "repo:$R cache (\"we decided against\" OR \"not worth\" OR \"no benefit\" OR \"does not help\")"
done

echo "=== PHRASE SWEEP DONE ==="
