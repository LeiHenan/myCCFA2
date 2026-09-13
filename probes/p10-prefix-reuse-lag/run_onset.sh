#!/usr/bin/env bash
# p10 前缀复用"生效时延"（onset）实验
#
# 判据（**采数前**登记：notes/prereg/prefix-reuse-lag.md）：
#   C1 复现：与 A3 同条件（util 0.95 / bs32 / 同 prompt）下，第 2 次命中增量 = 0、第 3 次 > 0
#   C2 交叉验证：命中出现的那次，mean_ttft_ms 相对前一次下降 ≥30%
#   C3 依赖关系：onset 随 并发{1,32} 与 池大小{0.30,0.95} 如何变化
#   C4 可回收性（Go 判据）：存在不改变负载的干预使 onset 从 3 提前到 2，且该 rep 吞吐 ≥+8%
#
# 设计要点：
#   * **每个 rep 前/后各抓一次 /metrics**，用**增量**算该 rep 的命中/查询（累计计数器必须做差，decision #76）
#   * **TTFT 作独立第二路径**（命中 ⇒ TTFT 阶梯下降）
#   * A3 的证据是**带投机**的（d5/γ=3），故第一阶段先**按原条件复现**，再在 spec=off 下扫依赖关系
#
# 用法（tmux 内）：bash probes/p10-prefix-reuse-lag/run_onset.sh
set -uo pipefail
. /root/ccfa_env.sh 2>/dev/null || true

MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3-4B}
DRAFTER=${DRAFTER:-/root/autodl-tmp/models/dflash2}
VLLMPY=${VLLMPY:-/root/ccfa_venv/bin/python}
OUT=${OUT:-/root/ccfa_results/$(date +%F)/p10_onset}
PORT=${PORT:-8000}
CTX=${CTX:-4096}
MAXLEN=${MAXLEN:-40960}
OUTLEN=${OUTLEN:-128}
REQS=${REQS:-32}
MAXSEQS=${MAXSEQS:-32}
REPS=${REPS:-5}
PROMPTS=${PROMPTS:-/root/autodl-tmp/prompts/prompts_${CTX}.jsonl}

mkdir -p "$OUT"
echo "=== p10 复用 onset 实验 $(date '+%F %T') ==="
echo "prompt: $PROMPTS  每条约 4104 tokens；reps=$REPS；OUTLEN=$OUTLEN"

scrape() { curl -s --max-time 5 "http://localhost:${PORT}/metrics" > "$1" || true; }

run_serve() {   # $1=util  $2=spec(off|d5g3)  $3=并发列表
  local util=$1 spec=$2 concs=$3
  local tag="u${util}_${spec}"
  local LOG="$OUT/${tag}.serve.log"
  local extra=()
  if [ "$spec" = "d5g3" ]; then
    extra=(--speculative-config "{\"model\":\"$DRAFTER\",\"num_speculative_tokens\":3}")
  fi
  echo "== $tag  $(date '+%T') =="
  "$VLLMPY" -m vllm.entrypoints.openai.api_server --model "$MODEL" \
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$util" --max-num-seqs "$MAXSEQS" \
    --kv-cache-dtype bfloat16 --enforce-eager --port "$PORT" "${extra[@]}" > "$LOG" 2>&1 &
  local SRV=$!
  local ready=0
  for _ in $(seq 1 120); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$SRV" 2>/dev/null || break
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    echo "  !! 未起来"; tail -4 "$LOG" | sed 's/^/     /'
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 5; return
  fi
  grep -m1 -o "GPU KV cache size: [0-9,]* tokens" "$LOG" | sed 's/^/  /'
  for bc in $concs; do
    mkdir -p "$OUT/bs${bc}"
    for r in $(seq 1 "$REPS"); do
      local f="${tag}_c${bc}_r${r}"
      scrape "$OUT/bs${bc}/${f}.pre.metrics"
      vllm bench serve --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name custom --dataset-path "$PROMPTS" --custom-output-len "$OUTLEN" \
        --num-prompts "$REQS" --max-concurrency "$bc" --disable-shuffle --ignore-eos --seed 0 \
        --save-result --result-dir "$OUT/bs${bc}" --result-filename "${f}.bench.json" \
        > "$OUT/bs${bc}/${f}.bench.log" 2>&1 || echo "  !! bench 失败 $f"
      scrape "$OUT/bs${bc}/${f}.post.metrics"
    done
  done
  kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 5
}

# 阶段 1：按 A3 原条件复现（带投机 d5/γ=3，util 0.95，bs32）
run_serve 0.95 d5g3 "32"
# 阶段 2：spec=off 下的依赖关系（池大小 × 并发两端）
run_serve 0.95 off "1 32"
run_serve 0.30 off "1 32"

echo "=== 完成 $(date '+%F %T') ==="
ls "$OUT"/bs*/*.bench.json 2>/dev/null | wc -l | sed 's/^/bench 文件数：/'
echo "下一步：python probes/p10-prefix-reuse-lag/analyze_onset.py --dir $OUT"
