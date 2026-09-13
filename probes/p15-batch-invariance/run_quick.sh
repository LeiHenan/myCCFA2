#!/usr/bin/env bash
# p15 快速格（≤0.1 GPU·h）：sm120 上 VLLM_BATCH_INVARIANT 是否真的成立？
# 只跑 BI∈{0,1} × n∈{8} × 3 次重复，用 openai 协议（/v1/completions —— vLLM 没有 /generate）。
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
P=/root/myCCFA/probes/p15-batch-invariance
OUT=/root/ccfa_results/$(date +%F)/p15_quick; mkdir -p "$OUT"
PORT=32011
PROMPT="The capital of the state containing New York City is"

echo "########## p15 快速格 $(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
for bi in 0 1; do
  VLLM_BATCH_INVARIANT=$bi "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name q3 --port $PORT \
    --max-model-len 2048 --max-num-seqs 16 --gpu-memory-utilization 0.5 \
    --no-enable-prefix-caching --enforce-eager > "$OUT/bi$bi.serve.log" 2>&1 &
  pid=$!
  ok=0
  for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 2
  done
  if [ "$ok" != 1 ]; then echo "!! [bi=$bi] 未就绪"; tail -15 "$OUT/bi$bi.serve.log"; kill $pid 2>/dev/null; continue; fi
  echo "[bi=$bi] ready pid=$pid $(date +%T)"
  grep -oE "Using [A-Z_]+ attention backend" "$OUT/bi$bi.serve.log" | head -1 | tee "$OUT/bi$bi.backend.txt"
  "$PY" "$P/probe_determinism.py" --tag "quick_bi${bi}_n8" --protocol openai --model q3 \
    --base "http://127.0.0.1:$PORT" --prompt "$PROMPT" --n 8 --max-tokens 48 --repeats 3 \
    --out "$OUT/bi${bi}_n8.jsonl" 2>&1 | tee "$OUT/bi${bi}_n8.probe.log"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
done
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "vllm[.]entrypoints" || echo "无残留 vLLM 进程"
echo "done $(date -Is)"
