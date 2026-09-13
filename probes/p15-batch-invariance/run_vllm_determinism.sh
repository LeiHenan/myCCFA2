#!/usr/bin/env bash
# p15：vLLM 0.29.0 的 VLLM_BATCH_INVARIANT 在 sm120 上是否真的成立？
#
# 设计（唯一自变量 = BI 开关；**批组成**用 eval_batch_size 的多个取值覆盖）：
#   同一 batch 内放 n 份**完全相同**的请求，贪心 + 固定 seed ⇒ 不变性成立时 n 份输出必须逐字相同。
# 网格：BI∈{0,1} × n∈{1,8,32,64}，每格 5 次重复（共 1050 个请求）
# 成本：8 个 serve ×(起 ~40s + 跑) ≈ 14–18 min ⇒ ≤0.4 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; PY=$VENV/bin/python
# ⚠️ vLLM 侧同样踩到 ninja/PATH：flashinfer 的 **sampling** JIT（flashinfer/sampling.py:68 → run_ninja）
#    按**名字**找 ninja，而它在 $VENV/bin/ 里。报错是 FileNotFoundError: 'ninja'，会伪装成引擎起不来。
#    （同类修复见 p12 的 SGLang 侧；这是第 2 次，故写进脚本头。）
export PATH="$VENV/bin:$PATH"
MODEL=/root/autodl-tmp/models/Qwen3-4B
P=/root/myCCFA/probes/p15-batch-invariance
OUT=/root/ccfa_results/$(date +%F)/p15_vllm; mkdir -p "$OUT"
PORT=32010; MAXTOK=64; REPS=5
PROMPT="The capital of the state containing New York City is"

serve () { # $1=bi $2=max_num_seqs
  VLLM_BATCH_INVARIANT=$1 \
  "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name q3 --port $PORT \
    --max-model-len 2048 --max-num-seqs "$2" --gpu-memory-utilization 0.55 \
    --no-enable-prefix-caching --enforce-eager \
    > "$OUT/bi$1_mns$2.serve.log" 2>&1 &
  SERVE_PID=$!
  local ok=0
  for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2
  done
  [ "$ok" = 1 ] || { echo "!! [bi=$1 mns=$2] 未就绪"; tail -25 "$OUT/bi$1_mns$2.serve.log"; return 1; }
  echo "[bi=$1 mns=$2] ready pid=$SERVE_PID $(date +%T)"
}
stop () { kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null; sleep 4; }

echo "########## vLLM 确定性网格 $(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
for bi in 0 1; do
  serve "$bi" 64 || continue
  # 记录本 serve 实际用的 attention backend（从日志抓，作为口径证据）
  grep -iE "attention backend|Using .* backend" "$OUT/bi$1_mns$2.serve.log" 2>/dev/null | head -2 \
    | tee "$OUT/bi$bi.backend.txt"
  grep -iE "attention backend|Using .* backend" "$OUT/bi${bi}_mns64.serve.log" 2>/dev/null | head -2 \
    | tee "$OUT/bi$bi.backend.txt"
  for n in 1 8 32 64; do
    "$PY" "$P/probe_determinism.py" --tag "vllm_bi${bi}_n${n}" --base "http://127.0.0.1:$PORT" \
      --prompt "$PROMPT" --n "$n" --max-tokens $MAXTOK --repeats $REPS \
      --out "$OUT/bi${bi}_n${n}.jsonl" 2>&1 | tee "$OUT/bi${bi}_n${n}.probe.log"
  done
  stop
done
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "vllm[.]entrypoints" || echo "无残留 vLLM 进程"
echo "done $(date -Is)"
