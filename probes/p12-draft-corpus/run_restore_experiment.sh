#!/usr/bin/env bash
# p12 R 阶段：语料恢复的成本与等价性（预登记 notes/prereg/sglang-draft-corpus-r.md，采数前冻结）
#
# 六臂，每臂一个**全新 serve 进程**，串行，同 prompt 集同序：
#   R_gen           无 -> 跑一遍并落盘续写（测 cost_gen）
#   R_regen         用 build_corpus 从 R_gen 导出的续写装载 -> 复测 accept_regen
#   R_cold          无（冷启动对照）
#   R_restore_docs  从 R_gen 落盘的 documents 恢复（测 cost_restore + accept_restored）
#   R_restore_combo 从「prompt 原文 + 续写」合并 documents 恢复（H-R3）
#   R_restore_xproc 用 R_restore_docs 的**同一落盘文件**在**另一个新进程**再恢复一次（跨进程可移植性）
# 成本：6×(36s + ~25s) ≈ 7 min ⇒ ≤0.15 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
VENV=/root/autodl-tmp/venvs/sglang; export PATH="$VENV/bin:$PATH"; PY="$VENV/bin/python"
MODEL=/root/autodl-tmp/models/Qwen3-4B
PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl
P=/root/myCCFA/probes/p12-draft-corpus
OUT=/root/ccfa_results/$(date +%F)/p12_restore; mkdir -p "$OUT"
PREV=/root/ccfa_results/2026-09-13/p12_outcorpus
PORT=31010; N=32; OUTLEN=128; BUDGET=7

serve () { # $1=tag
  "$PY" -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --speculative-algorithm NGRAM --speculative-num-steps 7 --speculative-num-draft-tokens 8 \
    --speculative-ngram-external-sam-budget $BUDGET \
    --mem-fraction-static 0.7 --context-length 8192 \
    --disable-radix-cache --max-running-requests $N > "$OUT/$1.serve.log" 2>&1 &
  SERVE_PID=$!; local ok=0
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2
  done
  [ "$ok" = 1 ] || { echo "!! [$1] serve 未就绪"; tail -20 "$OUT/$1.serve.log"; return 1; }
  echo "[$1] ready pid=$SERVE_PID $(date +%T)"; return 0
}
stop () { kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null; sleep 5; }
probe () { "$PY" "$P/probe.py" --tag "$1" --base "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
  --n $N --out-len $OUTLEN --reps 1 --out "$OUT/$1.jsonl" 2>&1 | tee "$OUT/$1.probe.log"; }

echo "########## R 阶段 $(date -Is) ##########"; nvidia-smi --query-gpu=memory.used --format=csv,noheader

# ---- R_gen：产出续写语料 + cost_gen ----
serve R_gen && probe R_gen; stop
"$PY" "$P/build_corpus.py" --in "$OUT/R_gen.jsonl" --out "$OUT/corpus_out.jsonl"
# 合并语料 = 32 条 prompt 原文 + 32 条续写（H-R3 用的"累积后 SAM 全部内容"）
cat "$PROMPTS" | head -$N > "$OUT/corpus_combo.jsonl"
cat "$OUT/corpus_out.jsonl" >> "$OUT/corpus_combo.jsonl"
wc -l "$OUT/corpus_out.jsonl" "$OUT/corpus_combo.jsonl"

# ---- R_cold ----
serve R_cold && probe R_cold; stop

# ---- R_regen：重新生成路径（对照组）----
serve R_regen && "$PY" "$P/corpus_client.py" --base "http://127.0.0.1:$PORT" add \
  --docs-file "$OUT/corpus_out.jsonl" --n $N --id regen --timeout 900 2>&1 | tee "$OUT/R_regen.corpus.log" \
  && probe R_regen; stop

# ---- R_restore_docs：从落盘 documents 恢复（成本 + 等价性）----
serve R_restore_docs && "$PY" "$P/restore_client.py" --base "http://127.0.0.1:$PORT" \
  --docs-file "$OUT/corpus_out.jsonl" --n $N --id r_docs --out "$OUT/restore_cost.jsonl" \
  2>&1 | tee "$OUT/R_restore_docs.restore.log" && probe R_restore_docs; stop

# ---- R_restore_combo：prompt 原文 + 续写 合并恢复（H-R3）----
serve R_restore_combo && "$PY" "$P/restore_client.py" --base "http://127.0.0.1:$PORT" \
  --docs-file "$OUT/corpus_combo.jsonl" --n $((N*2)) --id r_combo --out "$OUT/restore_cost.jsonl" \
  2>&1 | tee "$OUT/R_restore_combo.restore.log" && probe R_restore_combo; stop

# ---- R_restore_xproc：同一落盘文件，另一个全新进程再恢复（跨进程可移植性）----
serve R_restore_xproc && "$PY" "$P/restore_client.py" --base "http://127.0.0.1:$PORT" \
  --docs-file "$OUT/corpus_out.jsonl" --n $N --id r_xproc --out "$OUT/restore_cost.jsonl" \
  2>&1 | tee "$OUT/R_restore_xproc.restore.log" && probe R_restore_xproc; stop

echo "########## 汇总 ##########"
"$PY" "$P/compare_restore.py" --dir "$OUT" 2>&1 | tee "$OUT/compare.txt"
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "sglang[.]launch_server" || echo "无残留 serve 进程"
echo "done $(date -Is)"
