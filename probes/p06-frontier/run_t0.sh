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
#   DATASET=custom DATASET_DIR=/root/autodl-tmp/prompts bash run_t0.sh    # 真实文本（推荐）
#
# ⚠️ 数据集选择（2026-09-12 实测教训）：**不要用 `--dataset-name random`**。
#    它喂随机 token id，drafter 无从预测 ⇒ 每一档的 Mean acceptance length 都是 1.00（零接受），
#    脊线被整体压平、失去信号。必须用真实文本：`DATASET=custom DATASET_DIR=<含 prompts_<ctx>.jsonl 的目录>`。
#
# ⚠️ KV dtype 必须与模型 dtype 一致（bf16 模型给 float16 会让 FA 报 query/key dtype 不匹配）。
#
# 前置：DRAFTER_ROOT 下须有 d1..d5 变体（make_depth_variants.py，**必须带 --prune-weights**：
#       vLLM 严格加载，权重里多出的层会直接报 "no module or parameter named 'layers.1'"）。
set -uo pipefail

TARGET=${TARGET:-Qwen/Qwen3-4B}
DRAFTER_ROOT=${DRAFTER_ROOT:-upstream/drafters}
OUT=${OUT:-results/p06-frontier/$(date +%F)/T0}
PORT=${PORT:-8000}
GAMMA=${GAMMA:-7}
CTX=${CTX:-4096}
KV_DTYPE=${KV_DTYPE:-auto}
MAXLEN=${MAXLEN:-8192}
GPU_UTIL=${GPU_UTIL:-0.5}
DEPTHS=${DEPTHS:-"1 2 3 4 5"}
DRY=${DRY:-0}
EAGER=${EAGER:-0}   # =1 追加 --enforce-eager（绕过 torch.compile/fake-kernel 的 flakiness；全网格必须一致）
PROMPT=${PROMPT:-"The capital of France is"}
DATASET=${DATASET:-random}
DATASET_DIR=${DATASET_DIR:-}
OUTLEN=${OUTLEN:-128}
REQS=${REQS:-8}

mkdir -p "$OUT"
show() { printf '  [dry]'; printf ' %q' "$@"; printf '\n'; }

bench_args() {   # $1 = ctx
  if [ "$DATASET" = "custom" ]; then
    printf '%s\n' --dataset-name custom --dataset-path "$DATASET_DIR/prompts_$1.jsonl" --custom-output-len "$OUTLEN"
  else
    printf '%s\n' --dataset-name "$DATASET" --random-input-len "$1" --random-output-len "$OUTLEN"
  fi
}

for d in $DEPTHS; do
  DRAFT="$DRAFTER_ROOT/d${d}"
  if [ "$DRY" != "1" ] && [ ! -d "$DRAFT" ]; then
    echo "跳过 depth=${d}：缺 $DRAFT"; continue
  fi
  # 护栏（2026-09-12 实测踩坑）：vLLM 0.29 的 method 推断只看**路径字符串**里有没有 "dflash"
  #   （`"dflash" in draft_model_config.model.lower()`；`DFlash2DraftModel` 不在架构白名单里）。
  #   路径不含 dflash ⇒ 退化成通用 `draft_model` ⇒ DFlash 专属接线失效 ⇒ **接受率恒为 0**。
  case "$(printf '%s' "$DRAFT" | tr 'A-Z' 'a-z')" in
    *dflash*) ;;
    *) echo "  !! 致命：drafter 路径不含 'dflash'（${DRAFT}）⇒ vLLM 会当通用 draft_model、接受率恒 0。"
       echo "     请把变体放在含 'dflash' 的目录下，例如 DRAFTER_ROOT=/root/autodl-tmp/dflash-variants"
       continue ;;
  esac
  echo "== depth=${d} (gamma=${GAMMA}, ctx=${CTX}, dataset=${DATASET}) =="
  SERVE=(vllm serve "$TARGET"
         --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$GAMMA}"
         --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL"
         --kv-cache-dtype "$KV_DTYPE" --port "$PORT")
  [ "$EAGER" = "1" ] && SERVE+=(--enforce-eager)
  if [ "$DRY" = "1" ]; then
    show "${SERVE[@]}"
    DS=(); while IFS= read -r _l; do DS+=("$_l"); done < <(bench_args "$CTX")
    show vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
      "${DS[@]}" --num-prompts "$REQS" --max-concurrency 1 \
      --save-result --result-dir "$OUT" --result-filename "d${d}.bench.json"
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

  mapfile -t DS < <(bench_args "$CTX")
  vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
    "${DS[@]}" --num-prompts "$REQS" --max-concurrency 1 \
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
echo "下一步：python probes/p06-frontier/analyze_t0.py --dir $OUT --drafter-root $DRAFTER_ROOT"
