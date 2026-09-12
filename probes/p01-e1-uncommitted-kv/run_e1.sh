#!/usr/bin/env bash
# E1 压测（#1 · 未提交投机 KV 的 R_byte / R_reserve / H）—— 与 notes/prereg/p01-e1-uncommitted-kv.md 一致
#
# 判据（预登记）：全部 HBM 可行点上 p95 < 5% ⇒ #1 定死；存在点 p95 ≥ 10% ⇒ 存活；
#                D3 支线：R_reserve / R_byte > 2 ⇒ 立项「消除保守预留」
#
# 用法（在实例上，**必须在 T1 等别的实验结束之后**，因为插桩会改写 venv 里的 vllm 源码）：
#   DRY=1 bash run_e1.sh
#   OUT=$OUT_BASE/E1 DATASET=custom DATASET_DIR=/root/autodl-tmp/prompts bash run_e1.sh
#
# 前置：先 `python .../e1_patch.py --status` 确认补丁状态（锚点已在 vLLM 0.29.0 上核对通过）
set -uo pipefail

TARGET=${TARGET:-Qwen/Qwen3-4B}
DRAFTER=${DRAFTER:-}            # 留空则用 ngram（预登记允许：E1 只为 KV 记账，不需要高质量 drafter）
OUT=${OUT:-results/p01-e1-uncommitted-kv/$(date +%F)}
PORT=${PORT:-8000}
KV_DTYPE=${KV_DTYPE:-bfloat16}
MAXLEN=${MAXLEN:-40960}
GPU_UTIL=${GPU_UTIL:-0.5}
EAGER=${EAGER:-1}
DRY=${DRY:-0}
E1_LOG=${E1_LOG:-/tmp/e1_metrics.jsonl}
GAMMAS=${GAMMAS:-"3 5 7"}
CTXS=${CTXS:-"4096 32768"}
CONCS=${CONCS:-"1 8"}
REPS=${REPS:-3}
REQS=${REQS:-64}
OUTLEN=${OUTLEN:-256}
DATASET=${DATASET:-custom}
DATASET_DIR=${DATASET_DIR:-}
BLOCK_SIZE=${BLOCK_SIZE:-16}

mkdir -p "$OUT"
show() { printf '  [dry]'; printf ' %q' "$@"; printf '\n'; }

show python probes/p01-e1-uncommitted-kv/instrument/e1_patch.py --status
if [ "$DRY" = "1" ]; then
  for g in $GAMMAS; do
    SPEC="{\"method\":\"ngram\",\"num_speculative_tokens\":$g}"
    [ -n "$DRAFTER" ] && SPEC="{\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$g}"
    show vllm serve "$TARGET" --speculative-config "$SPEC" --max-model-len "$MAXLEN" \
      --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype "$KV_DTYPE" --port "$PORT" --enforce-eager
    for ctx in $CTXS; do for bc in $CONCS; do for r in $(seq 1 "$REPS"); do
      show vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
        --dataset-name "$DATASET" --dataset-path "$DATASET_DIR/prompts_$ctx.jsonl" \
        --custom-output-len "$OUTLEN" --num-prompts "$REQS" --max-concurrency "$bc" \
        --result-filename "g${g}_ctx${ctx}_bs${bc}_r${r}.bench.json" --result-dir "$OUT"
    done; done; done
  done
  exit 0
fi

echo "== 1) 打补丁（幂等）=="
python probes/p01-e1-uncommitted-kv/instrument/e1_patch.py --apply || exit 1
python probes/p01-e1-uncommitted-kv/instrument/e1_patch.py --status

for g in $GAMMAS; do
  SPEC="{\"method\":\"ngram\",\"num_speculative_tokens\":$g}"
  [ -n "$DRAFTER" ] && SPEC="{\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$g}"
  echo "== gamma=${g} =="
  : > "$E1_LOG"
  E1_LOG="$E1_LOG" vllm serve "$TARGET" --speculative-config "$SPEC" \
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" --port "$PORT" --enforce-eager > "$OUT/g${g}.serve.log" 2>&1 &
  SRV=$!
  ready=0
  for _ in $(seq 1 150); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null && { ready=1; break; }
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    echo "  !! gamma=${g} 服务未起来（见 $OUT/g${g}.serve.log 尾部）"
    tail -5 "$OUT/g${g}.serve.log" | sed 's/^/     /'
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 5; continue
  fi
  for ctx in $CTXS; do for bc in $CONCS; do for r in $(seq 1 "$REPS"); do
    TAG="g${g}_ctx${ctx}_bs${bc}_r${r}"
    vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
      --dataset-name "$DATASET" --dataset-path "$DATASET_DIR/prompts_${ctx}.jsonl" \
      --custom-output-len "$OUTLEN" --num-prompts "$REQS" --max-concurrency "$bc" \
      --result-filename "${TAG}.bench.json" --result-dir "$OUT" \
      > "$OUT/${TAG}.bench.log" 2>&1 || echo "  !! bench 失败：${TAG}"
    cp "$E1_LOG" "$OUT/${TAG}.e1.jsonl" 2>/dev/null || true
    python probes/p01-e1-uncommitted-kv/instrument/analyze.py \
      --log "$OUT/${TAG}.e1.jsonl" --block-size "$BLOCK_SIZE" --out "$OUT" --tag "$TAG" \
      > "$OUT/${TAG}.analyze.log" 2>&1 || echo "  !! analyze 失败：${TAG}"
  done; done; done
  kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
  sleep 5
done

echo "== 收工：撤销补丁（保持环境干净）=="
python probes/p01-e1-uncommitted-kv/instrument/e1_patch.py --revert
echo "完成 → $OUT"
