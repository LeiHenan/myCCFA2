#!/usr/bin/env bash
# p15：vLLM 在 sm120 上，**并发请求 + 变化的批规模上限**是否产生批组成发散？BI 关/开对照。
# 设计说明（诚实边界）：vLLM 客户端**无法构造**"一批同 prompt"（/v1/completions 不支持、贪心下 n>1 被禁），
#   故自变量用 **--max-num-seqs（批规模上限）+ 并发请求数**（= issue #51187 用的自变量）。
#   与 SGLang 的构造性合批相比，这里**合批仍未被直接证明** ⇒ 结论口径弱一档，必须写明。
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
P=/root/myCCFA/probes/p15-batch-invariance
OUT=/root/ccfa_results/$(date +%F)/p15_vllm_conc; mkdir -p "$OUT"
PORT=32041; MAXTOK=64; REPS=2
PROMPT=$("$PY" -c "
import json
line=open('/root/autodl-tmp/prompts/prompts_4096.jsonl',encoding='utf-8').readline()
print(json.loads(line)['prompt'])")

echo "########## vLLM 并发/批上限网格 $(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
for bi in 0 1; do
 for mns in 8 32; do
  VLLM_BATCH_INVARIANT=$bi "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name q3 --port $PORT \
    --max-model-len 8192 --max-num-seqs $mns --gpu-memory-utilization 0.6 \
    --no-enable-prefix-caching --enforce-eager > "$OUT/bi${bi}_mns${mns}.serve.log" 2>&1 &
  pid=$!; ok=0
  for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 2
  done
  if [ "$ok" != 1 ]; then echo "!! [bi=$bi mns=$mns] 未就绪"; tail -12 "$OUT/bi${bi}_mns${mns}.serve.log"; kill $pid 2>/dev/null; continue; fi
  echo "[bi=$bi mns=$mns] ready pid=$pid $(date +%T)"
  for n in 1 8; do
    "$PY" "$P/probe_determinism.py" --tag "vllm_bi${bi}_mns${mns}_n${n}" --protocol openai --model q3 \
      --mode concurrent --base "http://127.0.0.1:$PORT" --prompt "$PROMPT" \
      --n "$n" --max-tokens $MAXTOK --repeats $REPS --out "$OUT/bi${bi}_mns${mns}_n${n}.jsonl" \
      2>&1 | tee "$OUT/bi${bi}_mns${mns}_n${n}.probe.log"
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
 done
done
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "vllm[.]entrypoints" || echo "无残留进程"
echo "done $(date -Is)"
