#!/usr/bin/env bash
# p15 按构造合批：长序列 4096-token prompt，批规模 n∈{1,8,32}，SGLang 的确定性开关关/开。
# 一次 /generate 调用 = 一个批（text 传 List）⇒ 合批按构造成立，无需证明。
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/autodl-tmp/venvs/sglang; export PATH="$VENV/bin:$PATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
P=/root/myCCFA/probes/p15-batch-invariance
OUT=/root/ccfa_results/$(date +%F)/p15_batch; mkdir -p "$OUT"
PORT=32030; MAXTOK=64; REPS=2

echo "########## p15 按构造合批（长序列）$(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
for det in 0 1; do
  extra=(); [ "$det" = 1 ] && extra=(--enable-deterministic-inference)
  "$PY" -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.6 --context-length 8192 --disable-radix-cache \
    --max-running-requests 64 --attention-backend flashinfer "${extra[@]}" \
    > "$OUT/det$det.serve.log" 2>&1 &
  pid=$!; ok=0
  for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 2
  done
  if [ "$ok" != 1 ]; then echo "!! [det=$det] 未就绪"; tail -20 "$OUT/det$det.serve.log"; kill $pid 2>/dev/null; continue; fi
  echo "[det=$det] ready pid=$pid $(date +%T)"; grep -iE "deterministic|attention backend" "$OUT/det$det.serve.log" | head -3 | tee "$OUT/det$det.notes.txt"
  for n in 1 8 32; do
    "$PY" "$P/probe_batch.py" --tag "sgl_det${det}_n${n}" --base "http://127.0.0.1:$PORT" \
      --prompt-file /root/autodl-tmp/prompts/prompts_4096.jsonl \
      --n "$n" --max-tokens $MAXTOK --repeats $REPS --out "$OUT/det${det}_n${n}.jsonl" \
      2>&1 | tee "$OUT/det${det}_n${n}.probe.log"
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
done
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "sglang[.]launch_server" || echo "无残留进程"
echo "done $(date -Is)"
