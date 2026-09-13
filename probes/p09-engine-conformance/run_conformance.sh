#!/usr/bin/env bash
# p09 跨引擎一致性实测：vLLM 0.29.0 vs SGLang 0.5.19，同模型/同负载/同客户端
#
# 判据（**采数前**预登记于 notes/prereg/engine-metric-conformance.md）：C1–C4 四条可执行断言。
# 设计要点：
#   1. **同一个客户端**（`vllm bench serve`）打两家的 OpenAI 端点 ⇒ 消除客户端差异；
#   2. **NGRAM** 投机（两家都支持且**不需要额外 drafter 权重**）⇒ 消除 drafter 差异；
#   3. **服务端指标只从各自 /metrics 取**，且**用后台轮询抓取**（SGLang 的 spec_accept_length
#      是 gauge，跑完会归零 —— decision #66 的同源陷阱）；
#   4. 同一 serve 会话内跨 rep 的**计数器必须用增量**（decision #76 的仪器教训）。
#
# 用法（tmux 内）：bash probes/p09-engine-conformance/run_conformance.sh
set -uo pipefail
. /root/ccfa_env.sh 2>/dev/null || true

MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3-4B}
SGLPY=${SGLPY:-/root/autodl-tmp/venvs/sglang/bin/python}
VLLMPY=${VLLMPY:-/root/ccfa_venv/bin/python}
OUT=${OUT:-/root/ccfa_results/$(date +%F)/p09_conformance}
PORT=${PORT:-8000}
NAME=${NAME:-qwen3-4b}
CTX=${CTX:-4096}
MAXLEN=${MAXLEN:-40960}
OUTLEN=${OUTLEN:-128}
MAXSEQS=${MAXSEQS:-32}
REQS=${REQS:-32}
REPS=${REPS:-3}
CONCS=${CONCS:-"1 32"}
SPECS=${SPECS:-"off ngram"}
ENGINES=${ENGINES:-"vllm sglang"}
GAMMA=${GAMMA:-7}
UTIL=${UTIL:-0.75}
PROMPTS=${PROMPTS:-/root/autodl-tmp/prompts/prompts_${CTX}.jsonl}
SCRAPE_EVERY=${SCRAPE_EVERY:-3}

mkdir -p "$OUT"
echo "=== p09 跨引擎一致性实测 $(date '+%F %T') ==="
echo "vLLM: $($VLLMPY -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
echo "SGLang: $($SGLPY -c 'import sglang;print(sglang.__version__)' 2>/dev/null)"
echo "负载：$PROMPTS（$REQS 条 × $OUTLEN tokens），并发 $CONCS，投机 $SPECS，γ=$GAMMA，reps=$REPS"

scrape_loop() {   # $1=输出文件  $2=说明
  while :; do
    { echo "### $(date '+%s.%N')"; curl -s --max-time 5 "http://localhost:${PORT}/metrics" || true; } >> "$1"
    sleep "$SCRAPE_EVERY"
  done
}

serve_vllm() {    # $1=spec(off|ngram)
  local extra=()
  if [ "$1" = "ngram" ]; then
    extra=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${GAMMA},\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}")
  fi
  "$VLLMPY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$NAME" \
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAXSEQS" \
    --kv-cache-dtype bfloat16 --enforce-eager --port "$PORT" "${extra[@]}" > "$LOG" 2>&1 &
  echo $!
}

serve_sglang() {  # $1=spec(off|ngram)
  local extra=()
  if [ "$1" = "ngram" ]; then
    extra=(--speculative-algorithm NGRAM --speculative-num-steps "$GAMMA" --speculative-num-draft-tokens "$((GAMMA + 1))")
  fi
  "$SGLPY" -m sglang.launch_server --model-path "$MODEL" --served-model-name "$NAME" \
    --port "$PORT" --context-length "$MAXLEN" --mem-fraction-static "$UTIL" \
    --max-running-requests "$MAXSEQS" --enable-metrics --kv-cache-dtype bfloat16 "${extra[@]}" > "$LOG" 2>&1 &
  echo $!
}

for eng in $ENGINES; do
  for spec in $SPECS; do
    tag="${eng}_${spec}"
    echo "== $tag  $(date '+%T') =="
    LOG="$OUT/${tag}.serve.log"
    case "$eng" in
      vllm)   SRV=$(serve_vllm "$spec") ;;
      sglang) SRV=$(serve_sglang "$spec") ;;
      *) echo "  未知引擎 $eng"; continue ;;
    esac
    ready=0
    for _ in $(seq 1 120); do
      if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then ready=1; break; fi
      if ! kill -0 "$SRV" 2>/dev/null; then break; fi
      sleep 2
    done
    if [ "$ready" != "1" ]; then
      echo "  !! 服务未起来（见 $log）"; tail -5 "$log" | sed 's/^/     /'
      kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 5; continue
    fi
    echo "  服务就绪；启动 metrics 轮询（每 ${SCRAPE_EVERY}s）"
    scrape_loop "$OUT/${tag}.scrapes" &
    SC=$!
    for bc in $CONCS; do
      mkdir -p "$OUT/bs${bc}"
      for r in $(seq 1 "$REPS"); do
        f="${tag}_c${bc}_r${r}"
        echo "$(date '+%s.%N') START $f" >> "$OUT/${tag}.windows"
        vllm bench serve --model "$NAME" --base-url "http://localhost:${PORT}" \
          --dataset-name custom --dataset-path "$PROMPTS" --custom-output-len "$OUTLEN" \
          --num-prompts "$REQS" --max-concurrency "$bc" --disable-shuffle --ignore-eos --seed 0 \
          --save-result --result-dir "$OUT/bs${bc}" --result-filename "${f}.bench.json" \
          > "$OUT/bs${bc}/${f}.bench.log" 2>&1 || echo "  !! bench 失败 $f"
        echo "$(date '+%s.%N') END $f" >> "$OUT/${tag}.windows"
        curl -s --max-time 5 "http://localhost:${PORT}/metrics" > "$OUT/bs${bc}/${f}.metrics" || true
      done
    done
    sleep 2; kill "$SC" 2>/dev/null; wait "$SC" 2>/dev/null
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    sleep 6
  done
done

echo "=== 完成 $(date '+%F %T') ==="
ls "$OUT"/bs*/*.bench.json 2>/dev/null | wc -l | sed 's/^/bench 文件数：/'
echo "下一步：python probes/p09-engine-conformance/analyze_conformance.py --dir $OUT"
