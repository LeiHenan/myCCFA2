#!/usr/bin/env bash
# p23 —— **带 seed vs 不带 seed** 的匹配对照：逐请求采样循环的每步代价有多大？CUDA graph 会吸收它吗？
#
# 唯一变量 = 请求里有没有 `seed`（⇒ SamplingType.RANDOM_SEED ⇒ generators 非空 ⇒ 逐请求 kernel 循环）。
# 三个配置：eager@conc8（看随批大小缩放）、eager@conc32、**默认(CUDA graph)@conc32**（关键：图会不会吸收这些 launch）。
# 每臂：1 趟预热丢弃 + 3 次测量；两臂**交错**执行（unseeded→seeded 每轮交替）以抵消漂移。
#
# 用法： MODE=smoke bash run_seeded_ab.sh  |  MODE=full bash run_seeded_ab.sh
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl
P=/root/myCCFA/probes/p23-seeded-sampling
MODE=${MODE:-full}
OUT=/root/ccfa_results/$(date +%F)/p23_seeded
mkdir -p "$OUT"; TSV="$OUT/summary.tsv"
printf 'cfg\tgraphs\tconcurrency\tarm\trep\twall_s\tgen_tokens\tout_tput\treq_tput\tmean_lat_s\n' > "$TSV"
if [ "$MODE" = smoke ]; then CFGS="eager_c8 eager 8"; else CFGS="eager_c8 eager 8
eager_c32 eager 32
graph_c32 graph 32"; fi
REPS=3; NPROMPT=24; MAXTOK=128; PORT=32110
gpu_used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
reap () {
  kill "$SERVE_PID" 2>/dev/null
  pkill -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null
  pkill -x "VLLM::EngineCore" 2>/dev/null
  local i; for i in $(seq 1 20); do sleep 3; [ "$(gpu_used)" -lt 300 ] && { sleep 2; return 0; }; done
  pkill -9 -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null; pkill -9 -x "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 10); do sleep 3; [ "$(gpu_used)" -lt 300 ] && return 0; done
  echo "  ERROR 显存未回落：$(gpu_used) MiB"; return 1; }
serve () { # $1=graphs
  local g=$1 log="$OUT/serve_$1.log"; local args=(--model "$MODEL" --served-model-name q3 --port $PORT
    --max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization 0.55)
  [ "$g" = eager ] && args+=(--enforce-eager)
  "$PY" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$log" 2>&1 &
  SERVE_PID=$!; local ok=0 i
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2; done
  [ "$ok" != 1 ] && { echo "[$g] serve 未就绪"; grep -nE "Error|error" "$log" | tail -3; return 1; }
  echo "[$g] ready $(grep -oE 'cudagraph_mode=[A-Za-z_.]+' "$log" | head -1) $(date +%T)"; return 0; }
one () { # $1=cfg $2=graphs $3=conc $4=arm $5=rep $6=seeded-flag
  local cfg=$1 g=$2 c=$3 arm=$4 r=$5 flag=$6 j="$OUT/$cfg.$arm.r$r.json"
  local rc=0
  "$PY" "$P/seeded_ab.py" --base-url "http://127.0.0.1:$PORT" --model q3 --prompts "$PROMPTS" \
     --n-prompts "$NPROMPT" --max-tokens "$MAXTOK" --concurrency "$c" $flag --out "$j" > "$OUT/$cfg.$arm.r$r.log" 2>&1 || rc=1
  if [ $rc -ne 0 ] || [ ! -s "$j" ]; then echo "  [$cfg $arm r$r] 零产出 rc=$rc"; tail -4 "$OUT/$cfg.$arm.r$r.log"; return 1; fi
  "$PY" - "$cfg" "$g" "$c" "$arm" "$r" "$j" "$TSV" <<'PYEOF'
import json,sys
cfg,g,c,arm,r,j,tsv=sys.argv[1:8]
d=json.load(open(j))
row=[cfg,g,c,arm,r,"%.3f"%d["wall_s"],str(d["generation_tokens"]),
     "%.3f"%(d["output_throughput"] or 0),"%.4f"%(d["request_throughput"] or 0),
     "%.3f"%(d["mean_latency_s"] or 0)]
open(tsv,"a").write("\t".join(row)+"\n")
print("  [%s %-8s r%s] wall=%.2fs gen=%s out_tput=%.2f err=%s"%(
    cfg,arm,r,d["wall_s"],d["generation_tokens"],d["output_throughput"] or 0,d["errors"]))
PYEOF
  return 0; }
run_cfg () { # $1=cfg $2=graphs $3=conc
  local cfg=$1 g=$2 c=$3 fail=0 r
  echo "---- $cfg (graphs=$g, concurrency=$c) ----"
  serve "$g" || { reap; return 1; }
  echo "  [预热]（丢弃）"; one "$cfg" "$g" "$c" warmup 0 "--seeded" >/dev/null 2>&1 || true
  one "$cfg" "$g" "$c" warmup 0 "" >/dev/null 2>&1 || true
  for r in $(seq 1 $REPS); do
    one "$cfg" "$g" "$c" unseeded "$r" "" || fail=1
    one "$cfg" "$g" "$c" seeded   "$r" "--seeded" || fail=1
  done
  reap || fail=1
  return $fail; }
echo "########## p23 seed A/B $(date -Is) ##########"; echo "起始显存：$(gpu_used) MiB"
rc=0
while read -r cfg g c; do [ -z "${cfg:-}" ] && continue
  run_cfg "$cfg" "$g" "$c" || { echo "[$cfg] 有失败"; rc=1; }; done <<< "$CFGS"
echo "########## 收工 ##########"; echo "末显存：$(gpu_used) MiB"
pgrep -af "vllm[.]entrypoints" || echo "无残留 API server"
pgrep -x "VLLM::EngineCore" >/dev/null && echo "警告 仍有 EngineCore" || echo "无残留 EngineCore"
echo "===== summary.tsv ====="; column -t -s$'\t' "$TSV" 2>/dev/null || cat "$TSV"
[ $rc -eq 0 ] && echo "ALL_CFGS_OK" || echo "SOME_CFGS_FAILED"
