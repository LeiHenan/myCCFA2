#!/usr/bin/env bash
# p20 DSD 表存在性开销（并发 + 前缀缓存开启）—— 遵循 notes/prereg/p20-dsd-concurrency.md（含"采数前修订 #1"）
#
# v2 关键修正（两次零产出网格的根因）：
#   1. bench 入口必须是 `python -m vllm.entrypoints.cli.main bench serve`
#      —— `python -m vllm.benchmarks.serve` 在本环境 0.29.0 里**没有 __main__ 块**，
#         import 后静默退出且退出码 = 0 ⇒ 日志 0 字节、无结果 JSON，`|| echo 失败` 永不触发。
#   2. 每次 bench **断言**结果 JSON 存在、非空、能解析出 output_throughput；失败即如实上报。
#   3. `--result-filename` 确实存在（此前"不存在"的结论是被 #1 误导得出的，已更正）。
#   4. serve 按 PID/pgrep 回收，回收后**轮询显存**到 ~0 才进下一臂（防孤儿 VLLM::EngineCore 污染下一臂）。
#
# 用法： MODE=smoke bash run_dsd_grid.sh    # 2 臂 × 1 重复 × 8 prompt，验证管线
#        MODE=full  bash run_dsd_grid.sh    # 6 臂 × 3 重复 × 24 prompt
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8

VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
DF=/root/autodl-tmp/models/dflash2
PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl
P=/root/myCCFA/probes/p20-dsd
MODE=${MODE:-full}

case "$MODE" in
  smoke) OUT=/root/ccfa_results/$(date +%F)/p20_smoke_v3; ARMS="A2_static_k3 A4_table_allk0"; REPS=1; NPROMPT=8;  CONC=8 ;;
  full)  OUT=/root/ccfa_results/$(date +%F)/p20_dsd_v3;   ARMS="A1_nospec A2_static_k3 A3_static_k1 A4_table_allk0 A5_table_switch A6_table_const3"; REPS=3; NPROMPT=24; CONC=8 ;;
  *) echo "MODE 必须是 smoke|full"; exit 2 ;;
esac
# WARMUP=1 ⇒ 测量前先跑一遍同样请求集并**丢弃**结果（预热前缀缓存）。
# 事出有因（p21 实测）：**dflash 系投机**下前缀缓存第 2 遍仍 0 命中、第 3 遍才命中
# （prefill_kv_computed 第 2 遍仍 4096/4096，第 3 遍降到 32/4096）。不预热就会把"缓存升温过程"
# 误读成臂间差异：v2 网格里 rep1/rep2 对 A1(无投机) 已暖、对 A2–A6(有投机) 未暖，
# **冷热状态不匹配** ⇒ 热 p50 不可比。预热后所有被测量的重复都处于稳态。
WARMUP=${WARMUP:-0}
OUTLEN=128; PORT=32080
mkdir -p "$OUT"; TSV="$OUT/summary.tsv"
[ -s "$TSV" ] || printf 'tag\trep\tout_tput\treq_tput\tmed_ttft_ms\tmed_tpot_ms\tmed_itl_ms\tmed_e2el_ms\tcompleted\tnum_prompts\n' > "$TSV"

spec_of () { case "$1" in
  A1_nospec)        echo "" ;;
  A2_static_k3)     echo "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":3}" ;;
  A3_static_k1)     echo "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":1}" ;;
  A4_table_allk0)   echo "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":1,\"num_speculative_tokens_per_batch_size\":[[1,8192,0]]}" ;;
  A5_table_switch)  echo "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":3,\"num_speculative_tokens_per_batch_size\":[[1,3,3],[4,8192,0]]}" ;;
  A6_table_const3)  echo "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":3,\"num_speculative_tokens_per_batch_size\":[[1,8192,3]]}" ;;
  *) return 1 ;; esac; }

gpu_used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }

reap () { # 回收本臂的 serve + 可能残留的 EngineCore，然后等显存回落
  kill "$SERVE_PID" 2>/dev/null
  pkill -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null
  pkill -x "VLLM::EngineCore" 2>/dev/null
  local i
  for i in $(seq 1 20); do sleep 3; [ "$(gpu_used)" -lt 300 ] && { sleep 2; return 0; }; done
  echo "  ⚠️ 显存未回落（$(gpu_used) MiB），强杀"
  pkill -9 -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null; pkill -9 -x "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 10); do sleep 3; [ "$(gpu_used)" -lt 300 ] && return 0; done
  echo "  ❌ 显存仍被占用：$(gpu_used) MiB"; return 1; }

serve () { # $1=tag
  local tag=$1 spec; spec=$(spec_of "$tag") || { echo "[$tag] 未知臂"; return 1; }
  local log="$OUT/$tag.serve.log"
  local args=(--model "$MODEL" --served-model-name q3 --port $PORT --max-model-len 8192
              --max-num-seqs 32 --gpu-memory-utilization 0.55 --enforce-eager)
  # 前缀缓存**保持默认开启**（本格的条件 C；0.29.0 的 config/cache.py:138 默认 True；与上游基准相反）
  [ -n "$spec" ] && args+=(--speculative-config "$spec" --per-request-spec-decode-metrics detailed)
  "$PY" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$log" 2>&1 &
  SERVE_PID=$!; local ok=0 i
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2
  done
  if [ "$ok" != 1 ]; then echo "[$tag] ❌ serve 未就绪"; grep -nE "Value error|Error|not supported" "$log" | tail -3; return 1; fi
  echo "[$tag] ✅ ready $(grep -oE 'Using V[12] Model Runner' "$log" | head -1) 前缀缓存=$(grep -oE 'enable_prefix_caching=[A-Za-z]+' "$log" | head -1) $(date +%T)"
  return 0; }

run_rep () { # $1=tag $2=rep
  local tag=$1 r=$2 rj="$OUT/$tag.r$r.json" bl="$OUT/$tag.r$r.bench.log"
  rm -f "$rj"
  "$PY" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
    --model q3 --tokenizer "$MODEL" --dataset-name custom --dataset-path "$PROMPTS" \
    --num-prompts "$NPROMPT" --max-concurrency "$CONC" --ignore-eos --disable-shuffle \
    --output-len "$OUTLEN" --save-result --result-dir "$OUT" --result-filename "$tag.r$r.json" \
    --label "$tag.r$r" --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    > "$bl" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$rj" ]; then
    echo "[$tag r$r] ❌ 零产出 rc=$rc json=$([ -s "$rj" ] && echo 有 || echo 无) 日志字节=$(stat -c%s "$bl" 2>/dev/null || echo 0)"
    tail -6 "$bl"; return 1; fi
  local row
  row=$("$PY" - "$rj" "$tag" "$r" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
def g(k,f="%.4f"):
    v=d.get(k)
    return (f % v) if isinstance(v,(int,float)) else "nan"
print("\t".join([sys.argv[2],sys.argv[3],g("output_throughput","%.3f"),g("request_throughput"),
                 g("median_ttft_ms","%.2f"),g("median_tpot_ms","%.3f"),g("median_itl_ms","%.3f"),
                 g("median_e2el_ms","%.2f"),str(d.get("completed")),str(d.get("num_prompts"))]))
PYEOF
)
  [ -z "$row" ] && { echo "[$tag r$r] ❌ 结果 JSON 解析失败"; return 1; }
  echo "$row" >> "$TSV"
  echo "[$tag r$r] ✅ out_tput=$(echo "$row"|cut -f3) req_tput=$(echo "$row"|cut -f4) ttft=$(echo "$row"|cut -f5) tpot=$(echo "$row"|cut -f6) completed=$(echo "$row"|cut -f9)"
  return 0; }

run_warmup () { # $1=tag —— 预热一遍同样请求集并**丢弃**结果（不进 TSV）
  local tag=$1 bl="$OUT/$tag.warmup.bench.log"
  "$PY" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
    --model q3 --tokenizer "$MODEL" --dataset-name custom --dataset-path "$PROMPTS" \
    --num-prompts "$NPROMPT" --max-concurrency "$CONC" --ignore-eos --disable-shuffle \
    --output-len "$OUTLEN" --label "$tag.warmup" \
    > "$bl" 2>&1
  local rc=$?
  echo "  [预热] $tag rc=$rc 日志字节=$(stat -c%s "$bl" 2>/dev/null || echo 0)（结果已丢弃）"
  [ $rc -ne 0 ] && { tail -5 "$bl"; return 1; }
  return 0; }

run_arm () { # $1=tag
  local tag=$1 spec; spec=$(spec_of "$tag") || return 1
  local fail=0 r
  serve "$tag" || { reap; return 1; }
  local pre="$OUT/$tag.metrics.pre.json" post="$OUT/$tag.metrics.post.json"
  if [ -n "$spec" ]; then "$PY" "$P/collect_spec_metrics.py" --base "http://127.0.0.1:$PORT" fetch --out "$pre" >/dev/null 2>&1 || true; fi
  if [ "${WARMUP}" = "1" ]; then run_warmup "$tag" || fail=1; fi
  for r in $(seq 1 $REPS); do run_rep "$tag" "$r" || fail=1; done
  if [ -n "$spec" ]; then
    "$PY" "$P/collect_spec_metrics.py" --base "http://127.0.0.1:$PORT" fetch --out "$post" >/dev/null 2>&1 || true
    "$PY" "$P/collect_spec_metrics.py" --before "$pre" --after "$post" --out "$OUT/$tag.specdiff.json" diff > "$OUT/$tag.specdiff.txt" 2>&1 || true
    echo "  [投机计数] $(tr '\n' ' ' < "$OUT/$tag.specdiff.txt" 2>/dev/null | cut -c1-220)"
  fi
  reap || fail=1
  return $fail; }

echo "########## p20 DSD 网格 v3 ($MODE) WARMUP=$WARMUP $(date -Is) ｜ 臂：$ARMS ｜ 每臂 $REPS 重复 ｜ 并发 $CONC ｜ prompt $NPROMPT ##########"
echo "起始显存：$(gpu_used) MiB"
rc_all=0
for tag in $ARMS; do
  echo "---- $tag ----"
  run_arm "$tag" || { echo "[$tag] ⚠️ 本臂有失败（已记录）"; rc_all=1; }
done
echo "########## 收工 ##########"
echo "末显存：$(gpu_used) MiB"; pgrep -af "vllm[.]entrypoints" || echo "无残留 API server"
pgrep -x "VLLM::EngineCore" >/dev/null && echo "⚠️ 仍有 EngineCore" || echo "无残留 EngineCore"
echo "===== 汇总（summary.tsv）====="; column -t -s$'\t' "$TSV" 2>/dev/null || cat "$TSV"
[ $rc_all -eq 0 ] && echo "ALL_ARMS_OK" || echo "SOME_ARMS_FAILED"
