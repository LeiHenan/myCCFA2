#!/usr/bin/env bash
# p08 ground truth：数据集 × 温度 × 并发 —— 验证自检器的判别力
#
# 目的（预登记见 notes/prereg/specbench-methodology.md，**采数前**写下）：
#   判据① 自检器能区分"协议有效/无效"：`random` 数据集应触发 I2/I4 报警，`custom`（真实文本）应全部自洽。
#   判据② I1（`tpot ≈ itl/accept_len`）在所有格成立。
#   判据③ 温度是**接受率**的混淆变量（2605.24217 只测了温度对吞吐的 21% 影响，没测接受率）。
#
# 网格：数据集{custom, random} × 温度{0.7, 0} × 并发{1,32} × 3 reps = 4 serve / 24 bench
# 用法（tmux 内）：bash probes/p08-specbench/run_groundtruth.sh
set -uo pipefail
. /root/ccfa_env.sh 2>/dev/null || true

TARGET=${TARGET:-/root/autodl-tmp/models/Qwen3-4B}
DRAFTER=${DRAFTER:-/root/autodl-tmp/models/dflash2}
OUT=${OUT:-/root/ccfa_results/$(date +%F)/p08_groundtruth}
PORT=${PORT:-8000}
GAMMA=${GAMMA:-7}
CTX=${CTX:-4096}
MAXLEN=${MAXLEN:-40960}
OUTLEN=${OUTLEN:-128}
MAXSEQS=${MAXSEQS:-32}
REPS=${REPS:-3}
CONCS=${CONCS:-"1 32"}
TEMPS=${TEMPS:-"0.7 0"}
DATASETS=${DATASETS:-"custom random"}
REQS=${REQS:-32}
GPU_UTIL=${GPU_UTIL:-0.75}
PROMPTS=${PROMPTS:-/root/autodl-tmp/prompts}

mkdir -p "$OUT"
echo "=== p08 ground truth  $(date '+%F %T') ==="
echo "模型 $TARGET ｜ drafter $DRAFTER ｜ γ=$GAMMA ｜ 数据 $DATASETS ｜ 温度 $TEMPS ｜ 并发 $CONCS"

for ds in $DATASETS; do
  if [ "$ds" = "custom" ]; then
    DARGS=(--dataset-name custom --dataset-path "$PROMPTS/prompts_${CTX}.jsonl" --custom-output-len "$OUTLEN")
  else
    DARGS=(--dataset-name random --random-input-len "$CTX" --random-output-len "$OUTLEN")
  fi
  for temp in $TEMPS; do
    tag="${ds}_t${temp}"
    echo "== $tag  $(date '+%T') =="
    vllm serve "$TARGET" \
      --speculative-config "{\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$GAMMA}" \
      --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs "$MAXSEQS" \
      --kv-cache-dtype bfloat16 --port "$PORT" --enforce-eager \
      > "$OUT/${tag}.serve.log" 2>&1 & SRV=$!
    ready=0
    for _ in $(seq 1 90); do
      if curl -sf "http://localhost:${PORT}/health" >/dev/null; then ready=1; break; fi
      if ! kill -0 "$SRV" 2>/dev/null; then break; fi
      sleep 2
    done
    if [ "$ready" != "1" ]; then
      echo "  !! 服务未起来"; tail -4 "$OUT/${tag}.serve.log" | sed 's/^/     /'
      kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 4; continue
    fi
    grep -m1 -o 'GPU KV cache size: [0-9,]* tokens' "$OUT/${tag}.serve.log" | sed 's/^/  /'
    for bc in $CONCS; do
      mkdir -p "$OUT/bs${bc}"
      for r in $(seq 1 "$REPS"); do
        f="${tag}_c${bc}_r${r}"
        vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
          "${DARGS[@]}" --num-prompts "$REQS" --max-concurrency "$bc" --disable-shuffle \
          --temperature "$temp" --ignore-eos --seed 0 \
          --save-result --result-dir "$OUT/bs${bc}" --result-filename "${f}.bench.json" \
          > "$OUT/bs${bc}/${f}.bench.log" 2>&1 || echo "  !! bench 失败 $f"
        curl -s "http://localhost:${PORT}/metrics" > "$OUT/bs${bc}/${f}.metrics" || true
      done
    done
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 4
  done
done

echo "=== 完成 $(date '+%F %T') ==="
ls "$OUT"/bs*/*.bench.json 2>/dev/null | wc -l | sed 's/^/bench 文件数：/'
echo "下一步：python probes/p08-specbench/check_consistency.py --dir $OUT --gamma $GAMMA --out $OUT"
