#!/usr/bin/env bash
# T1 反转探针 sweep —— 冻结网格（pre-reg）：depth{1,3,5} × γ{1,3,7} × ctx{4k,128k} × bs{1}
# 18 格 × REPS 次重复。
#
#   DRY=1 bash run_t1.sh            # 只打印将执行的命令（瞬时，不启服务）
#   OUT=results/p06-frontier/$(date +%F) bash run_t1.sh
#   DATASET=custom DATASET_DIR=/root/autodl-tmp/prompts bash run_t1.sh     # 真实文本（**必须**）
#
# ⚠️ 数据集（2026-09-12 实测教训）：**不要用 `--dataset-name random`** —— 随机 token id 让 drafter
#    无从预测，每一档的 Mean acceptance length 都是 1.00（零接受），脊线被压平。
#    必须 `DATASET=custom DATASET_DIR=<含 prompts_<ctx>.jsonl 的目录>`，每格按 ctx 自动选文件。
#
# ⚠️ KV dtype：必须与模型 dtype 一致（本模型 bfloat16）；合法值 auto/float16/bfloat16/fp8*（无 fp16）。
# ⚠️ 变体：DRAFTER_ROOT 下 d1/d3/d5 必须由 make_depth_variants.py **带 --prune-weights** 生成
#    （vLLM 严格加载，权重多出的层会报 "no module or parameter named 'layers.N'"）。
set -uo pipefail

TARGET=${TARGET:-Qwen/Qwen3-4B}
DRAFTER_ROOT=${DRAFTER_ROOT:-upstream/drafters}   # 含 d1/ d3/ d5/
OUT=${OUT:-results/p06-frontier/$(date +%F)}
PORT=${PORT:-8000}
REPS=${REPS:-3}
KV_DTYPE=${KV_DTYPE:-auto}
MAXLEN=${MAXLEN:-140000}
GPU_UTIL=${GPU_UTIL:-0.5}
DRY=${DRY:-0}
DATASET=${DATASET:-random}
DATASET_DIR=${DATASET_DIR:-}
OUTLEN=${OUTLEN:-128}
REQS=${REQS:-8}
DEPTHS=${DEPTHS:-"1 3 5"}; GAMMAS=${GAMMAS:-"1 3 7"}; CTXS=${CTXS:-"4096 131072"}

mkdir -p "$OUT"
show() { printf '  [dry]'; printf ' %q' "$@"; printf '\n'; }

bench_args() {   # $1 = ctx
  if [ "$DATASET" = "custom" ]; then
    printf '%s\n' --dataset-name custom --dataset-path "$DATASET_DIR/prompts_$1.jsonl" --custom-output-len "$OUTLEN"
  else
    printf '%s\n' --dataset-name "$DATASET" --random-input-len "$1" --random-output-len "$OUTLEN"
  fi
}

for d in $DEPTHS; do for g in $GAMMAS; do
  DRAFT="$DRAFTER_ROOT/d${d}"
  if [ "$DRY" != "1" ] && [ ! -d "$DRAFT" ]; then
    echo "跳过 depth=${d}：缺 $DRAFT"; continue
  fi
  # 护栏：vLLM 0.29 的 method 推断只看路径里有没有 "dflash"（DFlash2DraftModel 不在架构白名单）
  #   ⇒ 路径不含 dflash 会退化成通用 draft_model，接受率恒 0（2026-09-12 实测踩坑）
  case "$(printf '%s' "$DRAFT" | tr 'A-Z' 'a-z')" in
    *dflash*) ;;
    *) echo "  !! 致命：drafter 路径不含 'dflash'（${DRAFT}）⇒ 接受率会恒为 0；请用 DRAFTER_ROOT=/root/autodl-tmp/dflash-variants"
       continue ;;
  esac
  echo "== depth=${d}  gamma=${g}  dataset=${DATASET} =="
  SERVE=(vllm serve "$TARGET"
         --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$g}"
         --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL"
         --kv-cache-dtype "$KV_DTYPE" --port "$PORT")
  if [ "$DRY" = "1" ]; then
    show "${SERVE[@]}"
    for ctx in $CTXS; do for r in $(seq 1 "$REPS"); do
      DS=(); while IFS= read -r _l; do DS+=("$_l"); done < <(bench_args "$ctx")
      show vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
        "${DS[@]}" --num-prompts "$REQS" --max-concurrency 1 --save-result --result-dir "$OUT" \
        --result-filename "d${d}_g${g}_ctx${ctx}_r${r}.bench.json"
      show curl -s "http://localhost:${PORT}/metrics" -o "$OUT/d${d}_g${g}_ctx${ctx}_r${r}.metrics"
    done; done
    continue
  fi
  "${SERVE[@]}" > "$OUT/d${d}_g${g}.serve.log" 2>&1 & SRV=$!
  ready=0
  for _ in $(seq 1 150); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null; then ready=1; break; fi
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    echo "  !! depth=${d} gamma=${g} 服务未起来（见 $OUT/d${d}_g${g}.serve.log 尾部）"
    tail -5 "$OUT/d${d}_g${g}.serve.log" | sed 's/^/     /'
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 5; continue
  fi
  for ctx in $CTXS; do for r in $(seq 1 "$REPS"); do
    TAG="d${d}_g${g}_ctx${ctx}_r${r}"
    mapfile -t DS < <(bench_args "$ctx")
    vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
      "${DS[@]}" --num-prompts "$REQS" --max-concurrency 1 \
      --save-result --result-dir "$OUT" --result-filename "${TAG}.bench.json" \
      > "$OUT/${TAG}.bench.log" 2>&1 || echo "  !! bench 失败：${TAG}"
    curl -s "http://localhost:${PORT}/metrics" > "$OUT/${TAG}.metrics" || true
  done; done
  kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
  sleep 5
done; done

echo "完成 → $OUT"
echo "下一步：python probes/p06-frontier/analyze_t1.py --dir $OUT --out $OUT"
