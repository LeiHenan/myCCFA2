#!/usr/bin/env bash
# T1 / S1 sweep —— 深度 × γ × ctx × **bs** 网格
#   预登记原网格：depth{1,3,5} × γ{1,3,7} × ctx{4k,32k} × bs{1}
#   2026-09-12 复核后按 prereg 的 **bs 扩展条款**加入 bs 维度（decision #53）
#
#   DRY=1 bash run_t1.sh                                  # 只打印命令
#   CONCS="1 8 32" CTXS="4096" REPS=5 bash run_t1.sh      # 96 GB 卡：4k 可到 bs=32
#   CONCS="1 2 4"  CTXS="32768" REPS=3 bash run_t1.sh     # 32 GB 卡：32k 只能到 bs=4（KV 4.5 GiB/请求）
#
# 设计要点（均为 2026-09-12 审计后的修正，见 decision #53/#54）：
#   1. **必须** `DATASET=custom`（真实文本）：`--dataset-name random` 会让接受率恒为 1.00（零接受）。
#   2. **必须** `--disable-shuffle`：否则每次 rep 的 prompt 顺序不同 + vLLM 默认开 prefix caching
#      ⇒ 同格重复之间的 tok/s 极差可达 10%，会把 3–5% 的真实差异淹掉（实测）。
#   3. 变体路径**必须含 `dflash`**（vLLM 0.29 仅凭路径串推断投机方法），且必须 `--prune-weights` 生成。
#   4. 结果按 bs 分目录（`$OUT/bs<N>/`），文件名保持 `d<d>_g<g>_ctx<c>_r<r>` ⇒ analyze_t1.py 可直接吃。
#   5. **逐次原始 JSON 必须留存**（`--save-result`）并入库 —— 否则算不出 p50/p95（曾因此永久丢数据）。
set -uo pipefail

TARGET=${TARGET:-Qwen/Qwen3-4B}
DRAFTER_ROOT=${DRAFTER_ROOT:-/root/autodl-tmp/dflash-variants}
OUT=${OUT:-results/p06-frontier/$(date +%F)/T1}
PORT=${PORT:-8000}
REPS=${REPS:-3}
KV_DTYPE=${KV_DTYPE:-auto}
MAXLEN=${MAXLEN:-40960}
GPU_UTIL=${GPU_UTIL:-0.9}
DRY=${DRY:-0}
EAGER=${EAGER:-1}
DATASET=${DATASET:-custom}
DATASET_DIR=${DATASET_DIR:-}
OUTLEN=${OUTLEN:-128}
REQS=${REQS:-32}
DEPTHS=${DEPTHS:-"1 3 5"}
GAMMAS=${GAMMAS:-"3 7"}
CTXS=${CTXS:-"4096 32768"}
CONCS=${CONCS:-"1"}

mkdir -p "$OUT"   # ⚠️ 必须：serve.log 与 bs 子目录都写在 $OUT 下（重写脚本时漏过这一行，导致首次启动即失败）

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
  case "$(printf '%s' "$DRAFT" | tr 'A-Z' 'a-z')" in *dflash*) ;; *)
    echo "  !! 致命：drafter 路径不含 'dflash'（${DRAFT}）⇒ 接受率会恒为 0"; continue ;; esac
  if [ "$DRY" != "1" ] && [ ! -d "$DRAFT" ]; then echo "跳过 depth=${d}：缺 $DRAFT"; continue; fi
  echo "== depth=${d} gamma=${g} =="
  SERVE=(vllm serve "$TARGET"
         --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$g}"
         --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL"
         --kv-cache-dtype "$KV_DTYPE" --port "$PORT")
  [ "$EAGER" = "1" ] && SERVE+=(--enforce-eager)

  if [ "$DRY" = "1" ]; then
    show "${SERVE[@]}"
    for bc in $CONCS; do for ctx in $CTXS; do for r in $(seq 1 "$REPS"); do
      DS=(); while IFS= read -r _l; do DS+=("$_l"); done < <(bench_args "$ctx")
      show vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
        "${DS[@]}" --num-prompts "$REQS" --max-concurrency "$bc" --disable-shuffle \
        --save-result --result-dir "$OUT/bs${bc}" \
        --result-filename "d${d}_g${g}_ctx${ctx}_r${r}.bench.json"
    done; done; done
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
  for bc in $CONCS; do for ctx in $CTXS; do for r in $(seq 1 "$REPS"); do
    TAG="d${d}_g${g}_ctx${ctx}_r${r}"
    D="$OUT/bs${bc}"; mkdir -p "$D"
    DS=(); while IFS= read -r _l; do DS+=("$_l"); done < <(bench_args "$ctx")
    vllm bench serve --model "$TARGET" --base-url "http://localhost:${PORT}" \
      "${DS[@]}" --num-prompts "$REQS" --max-concurrency "$bc" --disable-shuffle \
      --save-result --result-dir "$D" --result-filename "${TAG}.bench.json" \
      > "$D/${TAG}.bench.log" 2>&1 || echo "  !! bench 失败：bs${bc}/${TAG}"
    curl -s "http://localhost:${PORT}/metrics" > "$D/${TAG}.metrics" || true
  done; done; done
  kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
  sleep 5
done; done

echo "完成 → $OUT"
echo "下一步（逐 bs）：python probes/p06-frontier/analyze_t1.py --dir $OUT/bs1 --out $OUT/bs1"
