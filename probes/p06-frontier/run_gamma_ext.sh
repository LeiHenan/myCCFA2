#!/usr/bin/env bash
# T1 扩展探针（**探索性**，非预登记确证）：把 γ 推到网格上边界之外，回答两个问题：
#   ① depth≥3 时 γ* 落在 γ=7（T1 网格上边界）——真最优是否在 γ>7？
#   ② 在更大的 γ 上，浅 drafter 是否可能反超深 drafter（即"深度↔长度"权衡是否真的存在）？
#
# 这一步直接决定 T1 的 No-Go 结论是否稳健：
#   - 若 depth=5 在 γ∈{7,15,31} 上持续最优、且最优仍在上边界 ⇒ 不存在内部最优 ⇒ 分配机制无立足点
#   - 若出现内部最优/浅层反超 ⇒ 需要更大的网格重判
#
# 用法：DRY=1 bash run_gamma_ext.sh ｜ OUT=... bash run_gamma_ext.sh
set -uo pipefail

TARGET=${TARGET:-Qwen/Qwen3-4B}
DRAFTER_ROOT=${DRAFTER_ROOT:-/root/autodl-tmp/dflash-variants}
OUT=${OUT:-results/p06-frontier/$(date +%F)/T1ext}
PORT=${PORT:-8000}
KV_DTYPE=${KV_DTYPE:-bfloat16}
MAXLEN=${MAXLEN:-40960}
GPU_UTIL=${GPU_UTIL:-0.5}
EAGER=${EAGER:-1}
DRY=${DRY:-0}
DATASET=${DATASET:-custom}
DATASET_DIR=${DATASET_DIR:-}
OUTLEN=${OUTLEN:-128}
REQS=${REQS:-8}
REPS=${REPS:-3}
CTX=${CTX:-4096}
DEPTHS=${DEPTHS:-"1 5"}
GAMMAS=${GAMMAS:-"7 15 31"}

mkdir -p "$OUT"
show() { printf '  [dry]'; printf ' %q' "$@"; printf '\n'; }

for d in $DEPTHS; do for g in $GAMMAS; do
  DRAFT="$DRAFTER_ROOT/d${d}"
  case "$(printf '%s' "$DRAFT" | tr 'A-Z' 'a-z')" in *dflash*) ;; *)
    echo "  !! 跳过：drafter 路径不含 dflash（${DRAFT}）"; continue ;; esac
  if [ "$DRY" != "1" ] && [ ! -d "$DRAFT" ]; then echo "跳过 d=${d}：缺 $DRAFT"; continue; fi
  echo "== ext depth=${d} gamma=${g} =="
  SERVE=(vllm serve "$TARGET"
         --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$g}"
         --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL"
         --kv-cache-dtype "$KV_DTYPE" --port "$PORT")
  [ "$EAGER" = "1" ] && SERVE+=(--enforce-eager)
  if [ "$DRY" = "1" ]; then
    show "${SERVE[@]}"
    show vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
      --dataset-name "$DATASET" --dataset-path "$DATASET_DIR/prompts_${CTX}.jsonl" \
      --custom-output-len "$OUTLEN" --num-prompts "$REQS" --max-concurrency 1 \
      --save-result --result-dir "$OUT" --result-filename "d${d}_g${g}_r1.bench.json"
    continue
  fi
  "${SERVE[@]}" > "$OUT/d${d}_g${g}.serve.log" 2>&1 & SRV=$!
  ready=0
  for _ in $(seq 1 150); do curl -sf "http://localhost:${PORT}/health" >/dev/null && { ready=1; break; }; sleep 2; done
  if [ "$ready" != "1" ]; then
    echo "  !! d=${d} g=${g} 服务未起来"; tail -4 "$OUT/d${d}_g${g}.serve.log" | sed "s/^/     /"
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 5; continue
  fi
  for r in $(seq 1 "$REPS"); do
    TAG="d${d}_g${g}_r${r}"
    vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
      --dataset-name "$DATASET" --dataset-path "$DATASET_DIR/prompts_${CTX}.jsonl" \
      --custom-output-len "$OUTLEN" --num-prompts "$REQS" --max-concurrency 1 \
      --save-result --result-dir "$OUT" --result-filename "${TAG}.bench.json" \
      > "$OUT/${TAG}.bench.log" 2>&1 || echo "  !! bench 失败：${TAG}"
    curl -s "http://localhost:${PORT}/metrics" > "$OUT/${TAG}.metrics" || true
  done
  kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
  sleep 5
done; done
echo "完成 → $OUT"
