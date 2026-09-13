#!/usr/bin/env bash
# p15：vLLM 0.29.0 的 VLLM_BATCH_INVARIANT 在 sm120 上是否真的成立？
#
# 设计（唯一自变量 = BI 开关；批内放 n 份**完全相同**的请求）：
#   eval_batch_size(n) = 同一 batch 内 n 份相同请求，贪心 + 固定 seed
#   ⇒ 不变性成立时 n 份输出必须逐字相同（divergence=0）
# 臂：BI=0 × n∈{1,8,32,64}  ／ BI=1 × n∈{1,8,32,64}，每格 5 次重复（共 320 请求）
# 成本估计：8 个 serve ×(起+跑) ≈ 12–15 min ⇒ ≤0.35 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
P=/root/myCCFA/probes/p15-batch-invariance
OUT=/root/ccfa_results/$(date +%F)/p15_vllm; mkdir -p "$OUT"
PORT=32010; MAXTOK=64; REPS=5
PROMPT="The capital of the state containing New York City is"

serve () { # $1=tag $2=BI(0/1) $3=max_num_seqs
  local tag=$1 bi=$2 mns=$3
  VLLM_BATCH_INVARIANT=$bi VLLM_ATTENTION_BACKEND=${ATTN:-FLASHINFER} \
  "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name q3 --port $PORT \
    --max-model-len 2048 --max-num-seqs "$mns" --gpu-memory-utilization 0.55 \
    --no-enable-prefix-caching --enforce-eager \
    > "$OUT/$tag.serve.log" 2>&1 &
  SERVE_PID=$!
  local ok=0
  for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2
  done
  [ "$ok" = 1 ] || { echo "!! [$tag] 未就绪"; tail -25 "$OUT/$tag.serve.log"; return 1; }
  echo "[$tag] ready pid=$SERVE_PID bi=$bi max_num_seqs=$mns $(date +%T)"
  grep -oiE "Using [A-Za-z_]+ backend|attention backend[^,]*" "$OUT/$tag.serve.log" | head -3
}
stop () { kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null; sleep 4; }

for bi in 0 1; do
  mns=$([ "$bi" = 1 ] && echo 64 || echo 64)
  tag="vllm_bi${bi}"
  serve "$tag" "$bi" "$mns" || continue
  "$PY" "$P/probe_determinism.py" --tag "$tag" --base "http://127.0.0.1:$PORT" \
    --prompt "$PROMPT" --n 64 --max-tokens $MAXTOK --repeats $REPS \
    --out "$OUT/$tag.jsonl" 2>&1 | tee "$OUT/$tag.probe.log"
  stop
done

echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "vllm[.]entrypoints" || echo "无残留 vLLM 进程"
echo "done $(date -Is)"
