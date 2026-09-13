#!/usr/bin/env bash
# p12 P 阶段：新副本的语料获取策略（预登记 notes/prereg/sglang-draft-corpus-p.md，采数前冻结）
#
# 目标 workload：tenantB_prefix 的 32 条（512-token 共享前缀 + 4096-token 原文）
# 五臂，每臂一个全新 serve 进程（sam_budget=7）：
#   S_cold  无语料                       S_self  目标分布自己的续写
#   S_peer  异租户续写（无前缀 prompt）   S_text  目标 prompt 原文（零生成）
#   S_bare  目标无前缀版本的续写
# 成本：5 臂 ×(36 s 起 + ≤30 s + 测 32 条) ≈ 8–10 min ⇒ ≤0.25 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
VENV=/root/autodl-tmp/venvs/sglang; export PATH="$VENV/bin:$PATH"; PY="$VENV/bin/python"
MODEL=/root/autodl-tmp/models/Qwen3-4B
OUT=/root/ccfa_results/$(date +%F)/p12_policy; mkdir -p "$OUT"
P=/root/myCCFA/probes/p12-draft-corpus
PORT=31020; N=32; OUTLEN=128; BUDGET=7

echo "########## 构造租户 prompt ##########"
"$PY" "$P/build_tenant_prompts.py" --src /root/autodl-tmp/prompts/prompts_4096.jsonl \
  --corpus /root/autodl-tmp/prompts/corpus.txt --model "$MODEL" --n $N --prefix-tokens 512 \
  --outdir "$OUT" 2>&1 | tee "$OUT/build_tenants.log"
TA="$OUT/tenantA_prefix.jsonl"; TB="$OUT/tenantB_prefix.jsonl"; TN="$OUT/tenantB_noprefix.jsonl"

serve () {
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
probe () { "$PY" "$P/probe.py" --tag "$1" --base "http://127.0.0.1:$PORT" --prompts "$2" \
  --n $N --out-len $OUTLEN --reps 1 --out "$OUT/$1.jsonl" 2>&1 | tee "$OUT/$1.probe.log"; }
restore () { "$PY" "$P/restore_client.py" --base "http://127.0.0.1:$PORT" --docs-file "$2" \
  --n "$3" --id "$1" --out "$OUT/cost.jsonl" 2>&1 | tee "$OUT/$1.restore.log"; }

# ---- 1) 生成三种源语料（每源一遍，各自独立 serve；生成用无前缀 prompt）----
echo "########## 生成源语料 ##########"
serve S_gen_self && probe S_gen_self "$TB"; stop                       # 目标分布自己的续写
"$PY" "$P/build_corpus.py" --in "$OUT/S_gen_self.jsonl" --out "$OUT/corpus_self.jsonl"
serve S_gen_peer && probe S_gen_peer "$TN"; stop                       # 异租户续写（共享前缀但尾巴不同）
"$PY" "$P/build_corpus.py" --in "$OUT/S_gen_peer.jsonl" --out "$OUT/corpus_peer.jsonl"
serve S_gen_bare && probe S_gen_bare "$TN"; stop                       # 无前缀目标自身的续写
"$PY" "$P/build_corpus.py" --in "$OUT/S_gen_bare.jsonl" --out "$OUT/corpus_bare.jsonl"
wc -l "$OUT"/corpus_*.jsonl "$TB" "$TN"

# ---- 2) 五个策略臂：都服务目标 workload = TB ----
echo "########## 策略臂 ##########"
serve S_cold && probe S_cold "$TB"; stop
serve S_self && restore p_self "$OUT/corpus_self.jsonl" $N && probe S_self "$TB"; stop
serve S_peer && restore p_peer "$OUT/corpus_peer.jsonl" $N && probe S_peer "$TB"; stop
serve S_text && restore p_text "$TB" $N && probe S_text "$TB"; stop
serve S_bare && restore p_bare "$OUT/corpus_bare.jsonl" $N && probe S_bare "$TB"; stop

echo "########## 汇总 ##########"
"$PY" "$P/compare_policy.py" --dir "$OUT" 2>&1 | tee "$OUT/compare.txt"
echo "########## 收工自检 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af "sglang[.]launch_server" || echo "无残留 serve 进程"
echo "done $(date -Is)"
