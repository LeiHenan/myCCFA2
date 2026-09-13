#!/usr/bin/env bash
# p12 冒烟：一个 SGLang NGRAM serve + 探针 5 reps。
# 目的（PIPELINE §3.1 先验仪器再花 GPU）：
#   ① 服务能起；② 三条采集路径都能采到字段；③ 单位/口径核对；④ 看会话内累积是否存在。
# 成本估计：serve 启动 ~60 s + 5×32 请求，输出 128 tok、串行 ⇒ 约 3–6 min（≤0.1 GPU·h）。
set -uo pipefail
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
# SGLang NGRAM 的 JIT 内核需要 GLIBCXX_3.4.30（conda 只有 3.4.29）⇒ 必须 preload 系统 libstdc++
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export SGLANG_DISABLE_CUDA_GRAPH=0

VENV=/root/autodl-tmp/venvs/sglang
OUTDIR=/root/ccfa_results/$(date +%F)/p12_smoke
mkdir -p "$OUTDIR"
PORT=31001
MODEL=/root/autodl-tmp/models/Qwen3-4B
PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl

echo "=== outdir=$OUTDIR  port=$PORT  date=$(date -Is) ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# ---- 启动 serve（后台，日志留档）----
LOG="$OUTDIR/serve.log"
"$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port $PORT \
  --speculative-algorithm NGRAM \
  --speculative-num-draft-tokens 8 \
  --mem-fraction-static 0.7 \
  --context-length 8192 \
  --disable-radix-cache \
  > "$LOG" 2>&1 &
SERVE_PID=$!
echo "serve pid=$SERVE_PID"

# ---- 等就绪（探 /health，最多 300 s）----
ready=0
for i in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; echo "ready after ${i}x2s"; break; fi
  if ! kill -0 $SERVE_PID 2>/dev/null; then echo "!! serve 进程已退出"; break; fi
  sleep 2
done
if [ "$ready" != "1" ]; then
  echo "!! serve 未就绪；日志尾部："; tail -40 "$LOG"; kill $SERVE_PID 2>/dev/null; exit 1
fi

echo "=== server_info 摘要 ==="
curl -sf "http://127.0.0.1:$PORT/get_server_info" | head -c 1200; echo
echo "=== /generate 单请求字段探测 ==="
curl -sf -X POST "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
  -d '{"text":"The quick brown fox jumps over the lazy dog.","sampling_params":{"temperature":0.0,"max_new_tokens":16,"ignore_eos":true}}' \
  | head -c 1500; echo

echo "=== 探针：5 reps × 32 prompts × 128 out-tokens（串行）==="
"$VENV/bin/python" /root/myCCFA/probes/p12-draft-corpus/probe.py \
  --tag smoke --base "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
  --n 32 --out-len 128 --reps 5 --out "$OUTDIR/smoke.jsonl" 2>&1 | tee "$OUTDIR/probe.log"

echo "=== get_internal_state 尾部 ==="
curl -sf "http://127.0.0.1:$PORT/get_internal_state" | head -c 800; echo

# ---- 收工 ----
kill $SERVE_PID 2>/dev/null
wait $SERVE_PID 2>/dev/null
sleep 3
echo "=== 收工自检 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "done $(date -Is)"
