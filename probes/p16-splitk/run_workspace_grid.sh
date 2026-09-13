#!/usr/bin/env bash
# p16：把「不变性 / 代价」与 **cuBLAS 工作区配置** 联系起来（纯环境变量，不改引擎代码）。
#
# 依据（零卡源码）：BI 在 SM90/SM100/SM120 上**不替换 matmul 算子**，而是靠
#   `CUBLAS_WORKSPACE_CONFIG=":16:8"` + `CUBLASLT_WORKSPACE_SIZE="1"` 禁掉 split-k
#   （`determinism/batch_invariant.py:918-940` 的 else 分支）。⇒ 假设：**工作区配置是决定不变性的因素之一**。
#
# 网格：BI=0 固定；CUBLAS_WORKSPACE_CONFIG ∈ {默认(不设), :4096:8, :16:8}；每个配置测 n=1 与 n=8（并发相同请求）。
# 判据：
#   · 若某个 config 在 n=8 下 divergence=0 ⇒ 找到**比全开 BI 更省的等价手段**（真优化）
#   · 若都不为 0 ⇒ 不变性另有来源（softmax/mean/rms_norm 的替换，或 BI 的其它设置）
# 成本：3 个 serve ×(起~40s + 2 格) ≈ 6–8 min ⇒ ≤0.15 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
P=/root/myCCFA/probes/p15-batch-invariance
OUT=/root/ccfa_results/$(date +%F)/p16_workspace; mkdir -p "$OUT"
PORT=32050; MAXTOK=64; REPS=2
PROMPT=$("$PY" -c "
import json
print(json.loads(open('/root/autodl-tmp/prompts/prompts_4096.jsonl',encoding='utf-8').readline())['prompt'])")

echo "########## p16 工作区配置网格 $(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
i=0
for WS in "unset" ":4096:8" ":16:8"; do
  i=$((i+1)); tag="ws${i}_$(echo "$WS" | tr -d ':' | tr -d ' ')"
  if [ "$WS" = "unset" ]; then
    env -u CUBLAS_WORKSPACE_CONFIG VLLM_BATCH_INVARIANT=0 "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name q3 --port $PORT --max-model-len 8192 \
      --max-num-seqs 8 --gpu-memory-utilization 0.6 --no-enable-prefix-caching --enforce-eager \
      > "$OUT/$tag.serve.log" 2>&1 &
  else
    CUBLAS_WORKSPACE_CONFIG="$WS" VLLM_BATCH_INVARIANT=0 "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name q3 --port $PORT --max-model-len 8192 \
      --max-num-seqs 8 --gpu-memory-utilization 0.6 --no-enable-prefix-caching --enforce-eager \
      > "$OUT/$tag.serve.log" 2>&1 &
  fi
  pid=$!; ok=0
  for k in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 2
  done
  if [ "$ok" != 1 ]; then echo "!! [$tag WS=$WS] 未就绪"; tail -12 "$OUT/$tag.serve.log"; kill $pid 2>/dev/null; continue; fi
  echo "[$tag] WS=$WS ready pid=$pid $(date +%T)"
  for n in 1 8; do
    "$PY" "$P/probe_determinism.py" --tag "${tag}_n${n}" --protocol openai --model q3 \
      --mode concurrent --base "http://127.0.0.1:$PORT" --prompt "$PROMPT" \
      --n "$n" --max-tokens $MAXTOK --repeats $REPS --out "$OUT/${tag}_n${n}.jsonl" \
      2>&1 | tee "$OUT/${tag}_n${n}.probe.log"
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
done
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "vllm[.]entrypoints" || echo "无残留进程"
echo "done $(date -Is)"
