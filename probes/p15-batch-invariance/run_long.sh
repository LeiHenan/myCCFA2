#!/usr/bin/env bash
# p15 长序列/高并发格：issue 报告的发散条件（长序列 + 高共驻并发），我们此前只测了短序列/低并发。
# 用 SGLang 原生 /generate 的 input_ids 路径送 4096-token prompt（避免重新 tokenize）。
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/autodl-tmp/venvs/sglang; export PATH="$VENV/bin:$PATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
P=/root/myCCFA/probes/p15-batch-invariance
OUT=/root/ccfa_results/$(date +%F)/p15_long; mkdir -p "$OUT"
PORT=32020; MAXTOK=128; REPS=3

# 位置参数：并发度列表（默认 1 32）
NS="${*:-1 32}"

echo "########## p15 长序列格 $(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 先把第一条 4096-token prompt 编码成 token id 列表，供 SGLang input_ids 用
"$PY" - "$OUT" <<'PYEOF'
import json, sys, os
from transformers import AutoTokenizer
out = sys.argv[1]
tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen3-4B")
line = open("/root/autodl-tmp/prompts/prompts_4096.jsonl", encoding="utf-8").readline()
obj = json.loads(line)
ids = tok.encode(obj["prompt"], add_special_tokens=False)
json.dump({"input_ids": ids}, open(os.path.join(out, "prompt_ids.json"), "w"))
print("prompt tokens =", len(ids))
PYEOF

for n in $NS; do
  "$PY" -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.6 --context-length 8192 --disable-radix-cache \
    --max-running-requests 64 --attention-backend flashinfer \
    > "$OUT/n$n.serve.log" 2>&1 &
  pid=$!; ok=0
  for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 2
  done
  [ "$ok" = 1 ] || { echo "!! [n=$n] SGLang 未就绪"; tail -15 "$OUT/n$n.serve.log"; kill $pid 2>/dev/null; continue; }
  echo "[n=$n] SGLang ready pid=$pid $(date +%T)"
  "$PY" "$P/probe_ids.py" --tag "sgl_n$n" --base "http://127.0.0.1:$PORT" \
    --ids-file "$OUT/prompt_ids.json" --n "$n" --max-tokens $MAXTOK --repeats $REPS \
    --out "$OUT/sgl_n$n.jsonl" 2>&1 | tee "$OUT/sgl_n$n.probe.log"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
done
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "sglang[.]launch_server" || echo "无残留进程"
echo "done $(date -Is)"
