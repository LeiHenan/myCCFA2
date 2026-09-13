#!/usr/bin/env bash
# p11 小半径线索：SGLang NGRAM 的**草案语料**能否把会话内接受率增益提前到第一次请求？
#
# 由来（本会话 p09 实测）：SGLang + NGRAM 在同一 serve 会话内，accept 从 1.97（前两次）漂到 2.65（第三次），
# 疑似 ngram trie/语料在会话内累积 ⇒ 若预置语料（`--speculative-ngram-external-corpus-path`）生效，
# 该增益应当从**第一次请求**就出现。判据（一轮内可判）：预置组的首次 accept 是否 ≥ 基线稳态 accept 的 92%，
# 且吞吐提升 ≥8%。
#
# 用法（tmux 内）：CORPUS=/path/to/corpus.txt bash probes/p11-ngram-corpus/run_corpus.sh
set -uo pipefail
. /root/ccfa_env.sh 2>/dev/null || true

MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3-4B}
SGLPY=${SGLPY:-/root/autodl-tmp/venvs/sglang/bin/python}
OUT=${OUT:-/root/ccfa_results/$(date +%F)/p11_corpus}
PORT=${PORT:-8000}
CTX=${CTX:-4096}
REPS=${REPS:-5}
CONC=${CONC:-32}
REQS=${REQS:-32}
OUTLEN=${OUTLEN:-128}
MAXLEN=${MAXLEN:-40960}
GAMMA=${GAMMA:-7}
UTIL=${UTIL:-0.75}
PROMPTS=${PROMPTS:-/root/autodl-tmp/prompts/prompts_${CTX}.jsonl}
CORPUS=${CORPUS:-}              # 空 = 基线（无预置语料）

mkdir -p "$OUT/bs${CONC}"
tag="corpus_$( [ -n "$CORPUS" ] && echo on || echo off )"
LOG="$OUT/${tag}.serve.log"
extra=()
[ -n "$CORPUS" ] && extra=(--speculative-ngram-external-corpus-path "$CORPUS")

echo "== $tag  $(date '+%F %T')  corpus=${CORPUS:-（无）} =="
LD_PRELOAD=${SGL_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6} \
  "$SGLPY" -m sglang.launch_server --model-path "$MODEL" --port "$PORT" \
  --context-length "$MAXLEN" --mem-fraction-static "$UTIL" --max-running-requests "$CONC" \
  --enable-metrics --kv-cache-dtype bfloat16 \
  --speculative-algorithm NGRAM --speculative-num-steps "$GAMMA" \
  --speculative-num-draft-tokens "$((GAMMA + 1))" "${extra[@]}" > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 150); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  sleep 2
done
echo "  服务就绪 $(date '+%T')"

for r in $(seq 1 "$REPS"); do
  f="${tag}_r${r}"
  curl -s --max-time 5 "http://localhost:${PORT}/metrics" > "$OUT/bs${CONC}/${f}.pre.metrics" || true
  vllm bench serve --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name custom --dataset-path "$PROMPTS" --custom-output-len "$OUTLEN" \
    --num-prompts "$REQS" --max-concurrency "$CONC" --disable-shuffle --ignore-eos --seed 0 \
    --save-result --result-dir "$OUT/bs${CONC}" --result-filename "${f}.bench.json" \
    > "$OUT/bs${CONC}/${f}.bench.log" 2>&1 || echo "  !! bench 失败 $f"
  curl -s --max-time 5 "http://localhost:${PORT}/metrics" > "$OUT/bs${CONC}/${f}.post.metrics" || true
done
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo "== $tag 完成 $(date '+%T') =="
