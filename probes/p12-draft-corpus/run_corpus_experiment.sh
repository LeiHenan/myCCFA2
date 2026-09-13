#!/usr/bin/env bash
# p12 T0/T1：冷窗口能否被「预置语料」闭合？（+ 顺带留一条 combineResults 的观察）
#
# 设计（同一 prompt 集、固定顺序、串行，每个臂 = 一个全新 serve 进程）：
#   A_control ×3 : 全新 serve，空语料，跑 1 遍 32 条      → 冷启动对照
#   B_treat   ×3 : 全新 serve，**启动后立刻经 /add_external_corpus 装入预置语料**，再跑 1 遍 → 干预
#   C_steady  ×1 : 全新 serve，空语料，跑 4 遍            → 稳态（校准用）
#
# 判据（预登记见 candidates/sglang-draft-corpus/50_prereg.md）：
#   recovery = (accept_treat_first − accept_control_first) / (accept_steady − accept_control_first) ≥ 80%
#
# 成本：3×(36s 装载+~25s) + 3×(36s+~26s) + 1×(36s+~35s) ≈ 5 min ⇒ ≤0.15 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

VENV=/root/autodl-tmp/venvs/sglang
export PATH="$VENV/bin:$PATH"          # flashinfer JIT 需要 ninja（在 venv/bin）
PY="$VENV/bin/python"
MODEL=/root/autodl-tmp/models/Qwen3-4B
PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl
PROBE=/root/myCCFA/probes/p12-draft-corpus/probe.py
CORPUS_CLIENT=/root/myCCFA/probes/p12-draft-corpus/corpus_client.py
OUTDIR=/root/ccfa_results/$(date +%F)/p12_corpus
mkdir -p "$OUTDIR"
NPROMPT=32
OUTLEN=128
PORT=31002

run_serve () {   # $1=tag  其余=额外 serve 参数
  local tag="$1"; shift
  local log="$OUTDIR/${tag}.serve.log"
  "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --speculative-algorithm NGRAM \
    --speculative-num-steps 7 --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.7 --context-length 8192 \
    --disable-radix-cache --max-running-requests 32 \
    "$@" > "$log" 2>&1 &
  SERVE_PID=$!
  local ok=0
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 $SERVE_PID 2>/dev/null; then break; fi
    sleep 2
  done
  if [ "$ok" != "1" ]; then echo "!! [$tag] serve 未就绪"; tail -25 "$log"; return 1; fi
  echo "[$tag] serve ready (pid=$SERVE_PID) at $(date +%T)"
  return 0
}

stop_serve () {
  kill $SERVE_PID 2>/dev/null
  wait $SERVE_PID 2>/dev/null
  sleep 5
}

echo "############ p12 corpus experiment  $(date -Is) ############"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# ---------- A_control ×3 ----------
for r in 1 2 3; do
  tag="A_control_r$r"
  run_serve "$tag" || continue
  "$PY" "$PROBE" --tag "$tag" --base "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
      --n $NPROMPT --out-len $OUTLEN --reps 1 --out "$OUTDIR/$tag.jsonl" \
      2>&1 | tee "$OUTDIR/$tag.probe.log"
  stop_serve
done

# ---------- B_treat ×3（预置语料）----------
for r in 1 2 3; do
  tag="B_treat_r$r"
  run_serve "$tag" || continue
  echo "[$tag] 装入预置语料（documents = 同 32 条 prompt，模拟『预置/恢复的语料』）"
  t0=$(date +%s.%N)
  "$PY" "$CORPUS_CLIENT" --base "http://127.0.0.1:$PORT" add \
      --docs-file "$PROMPTS" --n $NPROMPT --id "prewarm_r$r" --timeout 900 \
      2>&1 | tee "$OUTDIR/$tag.corpus.log"
  t1=$(date +%s.%N)
  echo "[$tag] preload_s=$(echo "$t1 - $t0" | bc)"
  "$PY" "$CORPUS_CLIENT" --base "http://127.0.0.1:$PORT" list 2>&1 | tee -a "$OUTDIR/$tag.corpus.log"
  "$PY" "$PROBE" --tag "$tag" --base "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
      --n $NPROMPT --out-len $OUTLEN --reps 1 --out "$OUTDIR/$tag.jsonl" \
      2>&1 | tee "$OUTDIR/$tag.probe.log"
  stop_serve
done

# ---------- C_steady ×1（4 遍，取后两遍为稳态）----------
tag="C_steady"
run_serve "$tag"
"$PY" "$PROBE" --tag "$tag" --base "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
    --n $NPROMPT --out-len $OUTLEN --reps 4 --out "$OUTDIR/$tag.jsonl" \
    2>&1 | tee "$OUTDIR/$tag.probe.log"
stop_serve

echo "############ 分析 ############"
"$PY" /root/myCCFA/probes/p12-draft-corpus/analyze.py --in \
  "$OUTDIR"/A_control_r*.jsonl "$OUTDIR"/B_treat_r*.jsonl "$OUTDIR"/C_steady.jsonl \
  2>&1 | tee "$OUTDIR/analysis.txt"

echo "############ 收工自检 ############"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "sglang[.]launch_server" || echo "无残留 serve 进程"
echo "done $(date -Is)"
