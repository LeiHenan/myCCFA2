#!/usr/bin/env bash
# 三层 smoke test（对应 docs/SERVER_BOOTSTRAP.md §4）
#   ① 无投机      —— 引擎在本卡上能起、能出 token
#   ② ngram 投机  —— 投机解码路径通
#   ③ 真 drafter  —— **#6 的家族前提**：DFlash2 权重能否被加载并真正参与投机
#
# 用法：
#   bash smoke_test.sh
#   DRY=1 bash smoke_test.sh
#   TARGET=/root/autodl-tmp/models/Qwen3-4B DRAFTER=/root/autodl-tmp/models/dflash2 \
#     KV_DTYPE=bfloat16 OUT=/root/myCCFA/results/p06-frontier/$(date +%F)/smoke bash smoke_test.sh
#
# 判定：③ 的 spec_decode 计数必须 >0（否则"起了服务"不等于"投机真的接上了"）。
#
# ⚠️ KV dtype 规则（2026-09-12 在 96 GB Blackwell 上实测踩到）：**KV cache dtype 必须与模型 dtype 一致**。
#    本模型是 bfloat16；若传 `--kv-cache-dtype float16`，FlashAttention 会直接报
#    `mha_varlen_fwd ... query and key must have the same dtype` 并让 EngineCore 初始化失败。
#    合法取值（vLLM v0.29 `CacheDType`）：auto / float16 / bfloat16 / fp8* ⇒ 用 `auto` 或 `bfloat16`。
set -uo pipefail

TARGET=${TARGET:-Qwen/Qwen3-4B}
DRAFTER=${DRAFTER:-upstream/drafters/src}
OUT=${OUT:-results/p06-frontier/$(date +%F)/smoke}
PORT=${PORT:-8000}
KV_DTYPE=${KV_DTYPE:-auto}
MAXLEN=${MAXLEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.85}
PROMPT=${PROMPT:-"The capital of France is"}
DRY=${DRY:-0}

mkdir -p "$OUT"
show() { printf '  [dry]'; printf ' %q' "$@"; printf '\n'; }

run_layer() {
  local name="$1"; shift
  local serve=(vllm serve "$TARGET" "$@"
               --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL"
               --kv-cache-dtype "$KV_DTYPE" --port "$PORT")
  echo "== 层 ${name} =="
  if [ "$DRY" = "1" ]; then
    show "${serve[@]}"
    show curl -s "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" \
      -d "{\"model\":\"$TARGET\",\"prompt\":\"$PROMPT\",\"max_tokens\":32,\"temperature\":0}"
    show curl -s "http://localhost:${PORT}/metrics"
    return 0
  fi

  "${serve[@]}" > "$OUT/${name}.serve.log" 2>&1 & local srv=$!
  local ready=0
  for _ in $(seq 1 150); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null && { ready=1; break; }
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    echo "  ✘ 服务未起来；日志尾部："
    tail -12 "$OUT/${name}.serve.log" | sed 's/^/     /'
    kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; sleep 5
    return 1
  fi

  curl -s "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$TARGET\",\"prompt\":\"$PROMPT\",\"max_tokens\":32,\"temperature\":0}" \
    > "$OUT/${name}.completion.json" 2>/dev/null || true
  curl -s "http://localhost:${PORT}/metrics" > "$OUT/${name}.metrics" || true

  local txt acc
  txt=$(python3 -c "import json,sys;print((json.load(open('$OUT/${name}.completion.json')).get('choices') or [{}])[0].get('text','').strip()[:60])" 2>/dev/null || echo "<?>")
  acc=$(grep -oE 'vllm:spec_decode_num_accepted_tokens(_total)?\{[^}]*\} [0-9.eE+-]+' "$OUT/${name}.metrics" 2>/dev/null | awk '{s+=$NF} END{print s+0}')
  echo "  输出: ${txt}"
  echo "  spec_decode 接受 token 数: ${acc}"
  kill "$srv" 2>/dev/null || true; wait "$srv" 2>/dev/null || true
  sleep 5
  return 0
}

run_layer "01-nospec"
run_layer "02-ngram" --speculative-config '{"method":"ngram","num_speculative_tokens":5}'
run_layer "03-dflash2" --speculative-config "{\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":7}"

echo
echo "完成 → $OUT"
echo "判读：① 出文本 ⇒ 引擎可用；② 同上 ⇒ 投机路径通；③ 的接受 token 数 >0 且服务日志无加载报错 ⇒ #6 家族前提在真机成立"
