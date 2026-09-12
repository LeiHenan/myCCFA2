#!/usr/bin/env bash
# T0 深度旋钮验证（runbook: T0-runbook.md；判据见其 §4）
#
# 固定 bs=1 / ctx=4096 / γ=7，对每个 d ∈ DEPTHS 起一次服务，测：
#   - tok/s（vllm bench serve）
#   - 接受长度（/metrics 的 spec_decode 计数）
#   - lossless（同一 prompt 的 greedy 输出，与 d=max 对比）
#
# 用法：
#   DRY=1 bash run_t0.sh                          # 只打印命令
#   TARGET=/path/to/Qwen3-4B DRAFTER_ROOT=/path/drafters OUT=/path/out bash run_t0.sh
#   KV_DTYPE=float16 bash run_t0.sh               # 96 GB 卡用 FP16 KV（合法值只有 auto/float16/bfloat16/fp8*）
#
# 前置：DRAFTER_ROOT 下须有 d1..d5 变体（由 make_depth_variants.py 生成）
set -uo pipefail

TARGET=${TARGET:-Qwen/Qwen3-4B}
DRAFTER_ROOT=${DRAFTER_ROOT:-upstream/drafters}
OUT=${OUT:-results/p06-frontier/$(date +%F)/T0}
PORT=${PORT:-8000}
GAMMA=${GAMMA:-7}
CTX=${CTX:-4096}
KV_DTYPE=${KV_DTYPE:-auto}
MAXLEN=${MAXLEN:-8192}
GPU_UTIL=${GPU_UTIL:-0.85}
DEPTHS=${DEPTHS:-"1 2 3 4 5"}
DRY=${DRY:-0}
PROMPT=${PROMPT:-"The capital of France is"}

mkdir -p "$OUT"
show() { printf '  [dry]'; printf ' %q' "$@"; printf '\n'; }

for d in $DEPTHS; do
  DRAFT="$DRAFTER_ROOT/d${d}"
  if [ "$DRY" != "1" ] && [ ! -d "$DRAFT" ]; then
    echo "跳过 depth=${d}：缺 $DRAFT"; continue
  fi
  echo "== depth=${d} (gamma=${GAMMA}, ctx=${CTX}) =="
  SERVE=(vllm serve "$TARGET"
         --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$GAMMA}"
         --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL"
         --kv-cache-dtype "$KV_DTYPE" --port "$PORT")
  if [ "$DRY" = "1" ]; then
    show "${SERVE[@]}"
    show vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
      --dataset-name random --random-input-len "$CTX" --random-output-len 128 \
      --num-prompts 8 --max-concurrency 1 --save-result --result-dir "$OUT" \
      --result-filename "d${d}.bench.json"
    show curl -s "http://localhost:${PORT}/metrics" -o "$OUT/d${d}.metrics"
    show curl -s "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" \
      -d "{\"model\":\"$TARGET\",\"prompt\":\"$PROMPT\",\"max_tokens\":32,\"temperature\":0}" -o "$OUT/d${d}.prompt.json"
    continue
  fi

  "${SERVE[@]}" > "$OUT/d${d}.serve.log" 2>&1 & SRV=$!
  ready=0
  for _ in $(seq 1 150); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null; then ready=1; break; fi
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    echo "  !! depth=${d} 服务未起来（见 $OUT/d${d}.serve.log 尾部）"
    tail -5 "$OUT/d${d}.serve.log" | sed 's/^/     /'
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 5; continue
  fi

  vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len "$CTX" --random-output-len 128 \
    --num-prompts 8 --max-concurrency 1 \
    --save-result --result-dir "$OUT" --result-filename "d${d}.bench.json" \
    > "$OUT/d${d}.bench.log" 2>&1 || echo "  !! bench 失败（见 d${d}.bench.log）"

  curl -s "http://localhost:${PORT}/metrics" > "$OUT/d${d}.metrics" || true

  curl -s "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$TARGET\",\"prompt\":\"$PROMPT\",\"max_tokens\":32,\"temperature\":0}" \
    > "$OUT/d${d}.prompt.json" 2>/dev/null || true

  kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
  sleep 5
done

echo "完成 → $OUT"
echo "下一步：python probes/p06-frontier/analyze_t0.py --dir $OUT"
