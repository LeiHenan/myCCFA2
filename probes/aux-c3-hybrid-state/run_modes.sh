#!/usr/bin/env bash
# aux-c3 取证第 2 步：混合模型的 mamba 缓存模式 × 显存预算 → 吞吐 / 容量 / 复用
#
# 要回答的问题（预登记见 notes/prereg/aux-c3-hybrid-state.md）：
#   **状态内存占比 27.5–43.6%（aux-c3 第 1 步算出）到底值不值 ≥8% 的吞吐？**
#   判据：三种模式（none / align / all）× 两个显存预算下，
#         ① serve 日志的 "Maximum concurrency" 差多少（容量代价，直接读数）
#         ② 端到端 tok/s 差多少（吞吐代价）
#         ③ 重复前缀的 TTFT / prefix-cache 命中差多少（复用收益）
#   三档吞吐差 <8% ⇒ P1/P3 不构成可写课题，直接杀。
#
# 为什么必须调 util：96 GB 卡跑 8B 模型时状态内存（bs32×4k ≈ 1.5 GiB）根本不构成约束。
# 生产里混合模型多跑在 24–48 GB 卡上 ⇒ **用 --gpu-memory-utilization 复现那个"内存稀缺"区间**，
# 否则测出来的"没差别"是假阴性。
#
# 用法（tmux 内）：bash probes/aux-c3-hybrid-state/run_modes.sh
set -uo pipefail
. /root/ccfa_env.sh 2>/dev/null || true

MODEL=${MODEL:-/root/autodl-tmp/models/Nemotron-H-8B}
OUT=${OUT:-/root/ccfa_results/$(date +%F)/A6_hybrid_modes}
PORT=${PORT:-8000}
CTX=${CTX:-4096}
MAXLEN=${MAXLEN:-8192}
OUTLEN=${OUTLEN:-128}
MAXSEQS=${MAXSEQS:-32}
REPS=${REPS:-3}
CONCS=${CONCS:-"8 32"}
UTILS=${UTILS:-"0.30 0.22"}
BASEMODES=${BASEMODES:-"none align all"}
BLOCKS=${BLOCKS:-"default 1024 128"}      # 只对 all/align 生效
PROMPTS=${PROMPTS:-/root/autodl-tmp/prompts}
DUP=${DUP:-$OUT/prompts_${CTX}_dup.jsonl}

mkdir -p "$OUT"
# 重复前缀数据集：把同一份 prompt 文件拼两遍 ⇒ 后半段与前半段**前缀完全相同**，
# 用来触发 prefix caching / 状态检查点复用（--disable-shuffle 保证顺序固定）。
if [ ! -s "$DUP" ]; then cat "$PROMPTS/prompts_${CTX}.jsonl" "$PROMPTS/prompts_${CTX}.jsonl" > "$DUP"; fi
echo "数据集：$DUP（$(wc -l < "$DUP") 条，前一半与后一半前缀相同）"

run_one() {           # $1=util  $2=mode  $3=block(可空)
  local util=$1 mode=$2 block=$3
  local tag="u${util}_${mode}${block:+_b${block}}"
  local log="$OUT/${tag}.serve.log"
  local sargs=(--max-model-len "$MAXLEN" --gpu-memory-utilization "$util"
               --max-num-seqs "$MAXSEQS" --port "$PORT" --enforce-eager)
  case "$mode" in
    none)  sargs+=(--no-enable-prefix-caching) ;;
    align) : ;;                                        # PC 开 ⇒ 默认就是 align
    all)   sargs+=(--mamba-cache-mode all) ;;
  esac
  [ -n "$block" ] && sargs+=(--mamba-block-size "$block")

  echo "== $tag =="
  vllm serve "$MODEL" "${sargs[@]}" > "$log" 2>&1 & local srv=$!
  local ready=0
  for _ in $(seq 1 90); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null; then ready=1; break; fi
    if ! kill -0 "$srv" 2>/dev/null; then break; fi
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    echo "  !! 服务未起来（见 $log 尾部）"; tail -4 "$log" | sed 's/^/     /'
    kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; sleep 4
    printf '%s,START_FAIL,,,,,\n' "$tag" >> "$OUT/summary.csv"; return
  fi
  # 容量读数（P3 的直接证据）：serve 日志自己算的"最大并发"
  local kv conc mblk
  kv=$(grep -o 'GPU KV cache size: [0-9,]* tokens' "$log" | tail -1 | grep -o '[0-9,]*' | tr -d ,)
  conc=$(grep -o 'Maximum concurrency for [0-9,]* tokens per request: [0-9.]*x' "$log" | tail -1 | grep -o '[0-9.]*x$' | tr -d x)
  mblk=$(grep -oE 'mamba_block_size[^,}]*' "$log" | tail -1)
  echo "  KV tokens=$kv  max_conc=${conc}x  $mblk"
  for bc in $CONCS; do for r in $(seq 1 "$REPS"); do
    local f="d_${mode}${block:+_b${block}}_c${bc}_r${r}"
    vllm bench serve --model "$MODEL" --base-url "http://localhost:${PORT}" \
      --dataset-name custom --dataset-path "$DUP" --custom-output-len "$OUTLEN" \
      --num-prompts "$((bc * 2))" --max-concurrency "$bc" --disable-shuffle \
      --save-result --result-dir "$OUT/bs${bc}" --result-filename "${tag}_${f}.bench.json" \
      > "$OUT/bs${bc}/${tag}_${f}.bench.log" 2>&1 || echo "  !! bench 失败 $tag bs=$bc r=$r"
    curl -s "http://localhost:${PORT}/metrics" > "$OUT/bs${bc}/${tag}_${f}.metrics" || true
  done; done
  # 汇总一行（取最后一个 rep 的吞吐与 TTFT）
  local last="$OUT/bs${CONCS##* }/${tag}_d_${mode}${block:+_b${block}}_c${CONCS##* }_r${REPS}.bench.json"
  if [ -s "$last" ]; then
    /root/ccfa_venv/bin/python - "$last" "$tag" "$util" "$mode" "$block" "$kv" "$conc" "$OUT/summary.csv" <<'PY'
import json, sys
f, tag, util, mode, block, kv, conc, out = sys.argv[1:9]
d = json.load(open(f))
open(out, "a").write(f"{tag},{util},{mode},{block or 'default'},{kv},{conc},"
                     f"{d.get('output_throughput',0):.1f},{d.get('mean_ttft_ms',0):.0f},"
                     f"{d.get('spec_decode_acceptance_length') or ''}\n")
PY
  fi
  kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; sleep 4
}

[ -s "$OUT/summary.csv" ] || echo "tag,util,mode,block,kv_tokens,max_conc,tok_s,mean_ttft_ms,accept" > "$OUT/summary.csv"

for util in $UTILS; do
  for mode in $BASEMODES; do
    if [ "$mode" = "none" ] || [ "$mode" = "align" ]; then run_one "$util" "$mode" ""
    else for b in $BLOCKS; do
           if [ "$b" = "default" ]; then run_one "$util" "$mode" ""; else run_one "$util" "$mode" "$b"; fi
         done
    fi
  done
done

echo "=== 完成 $(date '+%F %T') ==="
cat "$OUT/summary.csv"
echo "下一步：python probes/aux-c3-hybrid-state/analyze_modes.py --dir $OUT"
