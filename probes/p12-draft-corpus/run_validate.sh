#!/usr/bin/env bash
# p12 验证轮：把 D_treat 的 σ 做实（重复 2 次）+ 预算极值核对（budget 3 是否也已够用）
#   V_treat_r2/r3 : budget=7 + output-as-corpus（与 D_treat 同配置，重复两次）
#   V_budget3     : budget=3 + output-as-corpus（检验收益是否随配额单调；非 Go 判据，只作机制证据）
# 成本：3×(36s + 5s + ~25s) ≈ 4 min ⇒ ≤0.1 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
VENV=/root/autodl-tmp/venvs/sglang; export PATH="$VENV/bin:$PATH"; PY="$VENV/bin/python"
MODEL=/root/autodl-tmp/models/Qwen3-4B; PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl
P=/root/myCCFA/probes/p12-draft-corpus
OUTDIR=/root/ccfa_results/$(date +%F)/p12_validate; mkdir -p "$OUTDIR"
CORPUS=/root/ccfa_results/2026-09-13/p12_outcorpus/corpus_output.jsonl
PORT=31005; NPROMPT=32; OUTLEN=128

arm () { # $1=tag $2=budget
  local tag=$1 b=$2
  "$PY" -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --speculative-algorithm NGRAM --speculative-num-steps 7 --speculative-num-draft-tokens 8 \
    --speculative-ngram-external-sam-budget "$b" \
    --mem-fraction-static 0.7 --context-length 8192 \
    --disable-radix-cache --max-running-requests 32 > "$OUTDIR/$tag.serve.log" 2>&1 &
  SERVE_PID=$!; local ok=0
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2
  done
  [ "$ok" = 1 ] || { echo "!! [$tag] 未就绪"; tail -20 "$OUTDIR/$tag.serve.log"; return 1; }
  echo "[$tag] ready budget=$b pid=$SERVE_PID $(date +%T)"
  "$PY" "$P/corpus_client.py" --base "http://127.0.0.1:$PORT" add \
    --docs-file "$CORPUS" --n $NPROMPT --id oc_$b --timeout 900 2>&1 | tee "$OUTDIR/$tag.corpus.log"
  "$PY" "$P/probe.py" --tag "$tag" --base "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
    --n $NPROMPT --out-len $OUTLEN --reps 1 --out "$OUTDIR/$tag.jsonl" \
    2>&1 | tee "$OUTDIR/$tag.probe.log"
  kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null; sleep 5
}
echo "########## 验证轮 $(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
arm V_treat_r2 7
arm V_treat_r3 7
arm V_budget3 3
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "sglang[.]launch_server" || echo "无残留 serve 进程"
echo "done $(date -Is)"
