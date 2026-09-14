#!/usr/bin/env bash
# C-1c: get ACCEPTANCE-RATE evidence via the OpenAI server's /metrics.
#
# Why: three offline attempts to read acceptance from the in-process `LLM` API all
# failed (engine core is a separate process; llm.py:228 forces disable_log_stats).
# The server exposes the counters directly:
#     vllm:spec_decode_num_accepted_tokens_total
#     vllm:spec_decode_num_draft_tokens_total
#     vllm:spec_decode_num_drafts
# so acceptance rate and mean acceptance length are computable exactly.
#
# Runs the SAME real-text multi-turn workload against dense and hybrid targets and
# records, per target: throughput AND acceptance. That separates two hypotheses for
# the hybrid slowdown:
#     H-A  low acceptance  (engine drafts but they are rejected)
#     H-B  expensive draft/verify path (only K-scaling and step-time decomposition
#          can distinguish this)
set -uo pipefail

PY=/root/miniconda3/bin/python
OUT=/root/autodl-tmp/c1c_out
mkdir -p "$OUT"
MODELS="/root/autodl-tmp/models/Qwen3-4B /root/autodl-tmp/models/Qwen3.5-4B"
K="${K:-3}"
PORT=8600

for MODEL in $MODELS; do
  NAME=$(basename "$MODEL")
  for PC in on off; do
    CELL="${NAME}_pc${PC}_K${K}"
    LOG="$OUT/${CELL}.log"
    echo "=== $CELL ==="
    pkill -9 -f "vllm.entrypoints.openai" 2>/dev/null; sleep 4

    args=(--model "$MODEL" --served-model-name m --port "$PORT"
          --gpu-memory-utilization 0.42 --max-model-len 4096
          --enforce-eager --language-model-only
          --speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":$K,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}")
    [ "$PC" = "on" ] && args+=(--enable-prefix-caching) || args+=(--no-enable-prefix-caching)

    setsid nohup env VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
      "$PY" -m vllm.entrypoints.openai.api_server "${args[@]}" \
      > "$LOG" 2>&1 < /dev/null &
    SPID=$!

    ok=0
    for _ in $(seq 1 60); do
      if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null; then ok=1; break; fi
      kill -0 "$SPID" 2>/dev/null || break
      sleep 4
    done
    if [ "$ok" != "1" ]; then
      echo "  SERVER FAILED; tail:"; tail -n 6 "$LOG"; continue
    fi

    "$PY" /root/autodl-tmp/c1c_client.py --url "http://127.0.0.1:${PORT}" \
        --model m --out "$OUT/${CELL}.json" 2>&1 | tail -12

    kill -TERM -"$SPID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -"$SPID" 2>/dev/null || true
    sleep 3
  done
done
echo "=== artifacts in $OUT ==="
ls -la "$OUT"
