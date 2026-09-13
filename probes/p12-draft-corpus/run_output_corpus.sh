#!/usr/bin/env bash
# p12 补测：**output-as-corpus**（模型自己的续写）能否闭合冷窗口？
#
# 为什么必须做这一臂：上游 PR #22569 的基准用的是 *output-as-corpus*，而我上一轮用的是
# **prompt 原文**当语料 —— 两者语义不同（续写在模型分布内、prompt 只是半截上下文）。
# budget 对照已证明 prompt-语料在 budget=7 下是**有害**的（1.5337 < 1.8360）；本臂判定
# "有害"是**语料语义**造成的还是**机制本身**造成的。这是判死前的最后一次补测。
#
# 臂（各自独立 serve 进程，同 prompt 集、同顺序、串行、无 radix cache）：
#   D_gen   : budget=7、不装语料、跑 1 遍并**保存 output**   → 生成 output-as-corpus 语料
#   D_treat : budget=7、装入 D_gen 的 output 语料、跑 1 遍   → 干预
#   D_ctrl  : budget=7、不装语料、跑 1 遍                    → 同 budget 的对照（隔离 budget 本身的影响）
# 成本：3×(36s + ~25s) ≈ 4 min ⇒ ≤0.1 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
VENV=/root/autodl-tmp/venvs/sglang; export PATH="$VENV/bin:$PATH"; PY="$VENV/bin/python"
MODEL=/root/autodl-tmp/models/Qwen3-4B
PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl
P=/root/myCCFA/probes/p12-draft-corpus
OUTDIR=/root/ccfa_results/$(date +%F)/p12_outcorpus; mkdir -p "$OUTDIR"
PORT=31004; NPROMPT=32; OUTLEN=128; BUDGET=7

serve () { # $1=tag
  "$PY" -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --speculative-algorithm NGRAM --speculative-num-steps 7 --speculative-num-draft-tokens 8 \
    --speculative-ngram-external-sam-budget $BUDGET \
    --mem-fraction-static 0.7 --context-length 8192 \
    --disable-radix-cache --max-running-requests 32 > "$OUTDIR/$1.serve.log" 2>&1 &
  SERVE_PID=$!; local ok=0
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2
  done
  [ "$ok" = 1 ] || { echo "!! [$1] serve 未就绪"; tail -20 "$OUTDIR/$1.serve.log"; return 1; }
  echo "[$1] ready budget=$BUDGET pid=$SERVE_PID $(date +%T)"; return 0
}
stop () { kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null; sleep 5; }

echo "########## output-as-corpus 补测 $(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

serve D_gen && "$PY" "$P/probe.py" --tag D_gen --base "http://127.0.0.1:$PORT" \
  --prompts "$PROMPTS" --n $NPROMPT --out-len $OUTLEN --reps 1 --out "$OUTDIR/D_gen.jsonl" \
  2>&1 | tee "$OUTDIR/D_gen.probe.log"; stop

"$PY" "$P/build_corpus.py" --in "$OUTDIR/D_gen.jsonl" --out "$OUTDIR/corpus_output.jsonl"
wc -l "$OUTDIR/corpus_output.jsonl"

serve D_treat && "$PY" "$P/corpus_client.py" --base "http://127.0.0.1:$PORT" add \
  --docs-file "$OUTDIR/corpus_output.jsonl" --n $NPROMPT --id out_corpus --timeout 900 \
  2>&1 | tee "$OUTDIR/D_treat.corpus.log" \
  && "$PY" "$P/probe.py" --tag D_treat --base "http://127.0.0.1:$PORT" \
  --prompts "$PROMPTS" --n $NPROMPT --out-len $OUTLEN --reps 1 --out "$OUTDIR/D_treat.jsonl" \
  2>&1 | tee "$OUTDIR/D_treat.probe.log"; stop

serve D_ctrl && "$PY" "$P/probe.py" --tag D_ctrl --base "http://127.0.0.1:$PORT" \
  --prompts "$PROMPTS" --n $NPROMPT --out-len $OUTLEN --reps 1 --out "$OUTDIR/D_ctrl.jsonl" \
  2>&1 | tee "$OUTDIR/D_ctrl.probe.log"; stop

echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "sglang[.]launch_server" || echo "无残留 serve 进程"
echo "done $(date -Is)"
