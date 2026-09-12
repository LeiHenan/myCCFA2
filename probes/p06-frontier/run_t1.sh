#!/usr/bin/env bash
# T1 反转探针 sweep —— 冻结网格（pre-reg）：depth{1,3,5} × γ{1,3,7} × ctx{4k,128k} × bs{1}
# 18 格 × REPS 次重复。**未在真机验证过**：flag 以 `vllm serve --help` / `vllm bench serve --help` 为准。
#
#   DRY=1 bash run_t1.sh            # 只打印将执行的命令（瞬时，不启服务）
#   OUT=results/p06-frontier/$(date +%F) bash run_t1.sh
#   KV_DTYPE=fp8 bash run_t1.sh     # 24 GB 卡跑 128k 格必须开 fp8
set -uo pipefail

TARGET=${TARGET:-Qwen/Qwen3-4B}
DRAFTER_ROOT=${DRAFTER_ROOT:-upstream/drafters}   # 含 d1/ d3/ d5/（make_depth_variants.py 生成）
OUT=${OUT:-results/p06-frontier/$(date +%F)}
PORT=${PORT:-8000}
REPS=${REPS:-3}
KV_DTYPE=${KV_DTYPE:-auto}
MAXLEN=${MAXLEN:-140000}
GPU_UTIL=${GPU_UTIL:-0.85}
DRY=${DRY:-0}
DEPTHS=${DEPTHS:-"1 3 5"}; GAMMAS=${GAMMAS:-"1 3 7"}; CTXS=${CTXS:-"4096 131072"}

mkdir -p "$OUT"
show() { printf '  [dry]'; printf ' %q' "$@"; printf '\n'; }

for d in $DEPTHS; do for g in $GAMMAS; do
  DRAFT="$DRAFTER_ROOT/d${d}"
  if [ "$DRY" != "1" ] && [ ! -d "$DRAFT" ]; then
    echo "跳过 depth=${d}：缺 $DRAFT"; continue
  fi
  echo "== depth=${d}  gamma=${g} =="
  SERVE=(vllm serve "$TARGET"
         --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$g}"
         --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL"
         --kv-cache-dtype "$KV_DTYPE" --port "$PORT")
  if [ "$DRY" = "1" ]; then
    show "${SERVE[@]}"
    for ctx in $CTXS; do for r in $(seq 1 "$REPS"); do
      show vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len "$ctx" --random-output-len 128 \
        --num-prompts 8 --max-concurrency 1 --save-result --result-dir "$OUT" \
        --result-filename "d${d}_g${g}_ctx${ctx}_r${r}.bench.json"
      show curl -s "http://localhost:${PORT}/metrics" -o "$OUT/d${d}_g${g}_ctx${ctx}_r${r}.metrics"
    done; done
    continue
  fi
  "${SERVE[@]}" & SRV=$!
  for _ in $(seq 1 150); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null && break || sleep 2
  done
  for ctx in $CTXS; do for r in $(seq 1 "$REPS"); do
    TAG="d${d}_g${g}_ctx${ctx}_r${r}"
    vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
      --dataset-name random --random-input-len "$ctx" --random-output-len 128 \
      --num-prompts 8 --max-concurrency 1 \
      --save-result --result-dir "$OUT" --result-filename "${TAG}.bench.json"
    curl -s "http://localhost:${PORT}/metrics" > "$OUT/${TAG}.metrics" || true
  done; done
  kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
  sleep 5
done; done

echo "完成 → $OUT"
echo "下一步：python probes/p06-frontier/analyze_t1.py --dir $OUT --out $OUT"
