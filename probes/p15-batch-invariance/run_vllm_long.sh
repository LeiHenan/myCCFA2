#!/usr/bin/env bash
# p15：vLLM 的 VLLM_BATCH_INVARIANT 在 **sm120 + 长序列** 下是否有效？
# 动机：issue 报告的发散都在 sm_90/FA3 或高共驻（#51187 ~44 seqs），**没人报 sm120 长序列**；
#       而 vLLM 的 envs.py 注释声称需要 sm≥9.0（docs 说 8.0+）⇒ sm120 属"文档声称支持"的算力。
# 用按构造合批（openai /v1/completions，n=1 但 prompt 为 4096-token），批规模靠并发进程数？
# 不 —— 这里改用 **SGLang 已验证的构造**在 vLLM 不可用（无 text=List），故用**同一 prompt 多请求**：
#   vLLM 的 /v1/completions 不支持"一批同 prompt"，因此本轮**只测单请求在不同 BI 下的输出是否一致**，
#   并用 SGLang 的结果作为"批组成效应存在"的已知对照。
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
P=/root/myCCFA/probes/p15-batch-invariance
OUT=/root/ccfa_results/$(date +%F)/p15_vllm_long; mkdir -p "$OUT"
PORT=32040; MAXTOK=64; REPS=3
PROMPT=$("$PY" -c "
import json
line=open('/root/autodl-tmp/prompts/prompts_4096.jsonl',encoding='utf-8').readline()
print(json.loads(line)['prompt'])
")

echo "########## vLLM 长序列（BI 关/开）$(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
for bi in 0 1; do
  VLLM_BATCH_INVARIANT=$bi "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name q3 --port $PORT \
    --max-model-len 8192 --max-num-seqs 8 --gpu-memory-utilization 0.6 \
    --no-enable-prefix-caching --enforce-eager > "$OUT/bi$bi.serve.log" 2>&1 &
  pid=$!; ok=0
  for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 2
  done
  if [ "$ok" != 1 ]; then echo "!! [bi=$bi] 未就绪"; tail -20 "$OUT/bi$bi.serve.log"; kill $pid 2>/dev/null; continue; fi
  echo "[bi=$bi] ready pid=$pid $(date +%T)"; grep -oE "Using [A-Z_]+ attention backend" "$OUT/bi$bi.serve.log" | head -1 | tee "$OUT/bi$bi.backend.txt"
  # 同一 prompt 连发 3 次（每次 n=1），比较**跨请求**是否一致（单请求内无批组成可言）
  "$PY" "$P/probe_repeat_once.py" --tag "vllm_bi${bi}" --base "http://127.0.0.1:$PORT" \
    --model q3 --prompt "$PROMPT" --repeats $REPS --max-tokens $MAXTOK \
    --out "$OUT/bi$bi.jsonl" 2>&1 | tee "$OUT/bi$bi.probe.log"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
done
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "vllm[.]entrypoints" || echo "无残留 vLLM 进程"
echo "done $(date -Is)"
