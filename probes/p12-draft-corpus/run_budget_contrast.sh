#!/usr/bin/env bash
# p12 T2 补充对照：预置语料的收益走「外置 SAM 路径」还是「trie 路径」？
#
# 依据（源码，decision #101）：`ngram.cpp:173` = num_sams>0 ? min(external_sam_budget, total_draft_token_num) : 0
#   ⇒ 关闭 SAM 路径的**唯一**方式是让 external_sam_budget = 0（其合法下界），语料仍照常装载。
#   对照臂与主实验的 B 臂**唯一差异**就是这一个参数（预算 7 → 0）⇒ 单变量对照。
#
# 臂：
#   B0（新）：serve 带 --speculative-ngram-external-sam-budget 0，装入同一份语料，跑 1 遍
#   B7（新）：serve 带 --speculative-ngram-external-sam-budget 7，装入同一份语料，跑 1 遍
# 判据（判别式，不是性能主张）：
#   若 B0 的 accept ≈ 1.84（控制水平）而 B7 ≈ 7.65 ⇒ 收益来自**外置 SAM 路径**（预置语料确实在起作用）
#   若 B0 ≈ B7 ⇒ 预算参数在预置场景下无效，收益来自 trie（会把机制解释改写）
#
# 成本：2 × (36s 装载 + 5s preload + ~25s) ≈ 3 min ⇒ ≤0.1 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

VENV=/root/autodl-tmp/venvs/sglang
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python"
MODEL=/root/autodl-tmp/models/Qwen3-4B
PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl
PROBE=/root/myCCFA/probes/p12-draft-corpus/probe.py
CORPUS_CLIENT=/root/myCCFA/probes/p12-draft-corpus/corpus_client.py
OUTDIR=/root/ccfa_results/$(date +%F)/p12_budget
mkdir -p "$OUTDIR"
PORT=31003
NPROMPT=32
OUTLEN=128

run_arm () {  # $1=tag  $2=sam_budget
  local tag="$1" budget="$2"
  local log="$OUTDIR/${tag}.serve.log"
  "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --speculative-algorithm NGRAM \
    --speculative-num-steps 7 --speculative-num-draft-tokens 8 \
    --speculative-ngram-external-sam-budget "$budget" \
    --mem-fraction-static 0.7 --context-length 8192 \
    --disable-radix-cache --max-running-requests 32 \
    > "$log" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 $pid 2>/dev/null; then break; fi
    sleep 2
  done
  if [ "$ok" != "1" ]; then echo "!! [$tag] serve 未就绪"; tail -25 "$log"; kill $pid 2>/dev/null; return 1; fi
  echo "[$tag] serve ready (budget=$budget, pid=$pid) at $(date +%T)"

  "$PY" "$CORPUS_CLIENT" --base "http://127.0.0.1:$PORT" add \
      --docs-file "$PROMPTS" --n $NPROMPT --id "budget_$budget" --timeout 900 \
      2>&1 | tee "$OUTDIR/$tag.corpus.log"
  "$PY" "$CORPUS_CLIENT" --base "http://127.0.0.1:$PORT" list 2>&1 | tee -a "$OUTDIR/$tag.corpus.log"

  "$PY" "$PROBE" --tag "$tag" --base "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
      --n $NPROMPT --out-len $OUTLEN --reps 1 --out "$OUTDIR/$tag.jsonl" \
      2>&1 | tee "$OUTDIR/$tag.probe.log"

  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
}

echo "############ p12 budget contrast  $(date -Is) ############"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
run_arm "B0_budget0" 0
run_arm "B7_budget7" 7
echo "############ 收工自检 ############"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "sglang[.]launch_server" || echo "无残留 serve 进程"
echo "done $(date -Is)"
