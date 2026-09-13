#!/usr/bin/env bash
# p21 —— 前缀缓存命中的"仍要重算"量 + 缓存状态是否改变确定性输出（KV 地图 B107 / B116 / B109）
#
# 为什么是这一格：
#   B107（OPEN，untested）："EAGLE/MTP prefix-cache last-block drop causes a 1,648-token recompute per hit"，
#        姊妹 issue #51771 把 "EAGLE/MTP block drop + prefix caching" 明确记为 **untested**。
#   B116（OPEN，2026-08-31 新开）："The minimal accepted prefix-cache configuration produces different text for
#        two identical long-prompt requests, while the no-prefix baseline passes."
#   B109（OPEN，bug）：T=0 同一 prompt 10/10 输出完全不同，记者称"基础调度器 non-APC 路径的 block 生命周期 bug"。
#
# 判定量（**一次冷/热/再热三段对照同时回答三个问题**）：
#   · B107 ⇒ 热阶段 `vllm:request_prefill_kv_computed_tokens`（官方定义 "excluding cached tokens"）的**每请求均值**。
#             完整命中应 ≈1（只有最后那个 token）；若投机下有量级更大的值 ⇒ 命中仍在白算 prefill = **可回收量**。
#   · B116 ⇒ 冷阶段 vs 热阶段的**输出是否相同**。
#   · B109 ⇒ 热 vs 再热（同状态）的**输出是否相同**。
#
# 对照设计（唯一变量 = 有没有投机）：同一 prompt 集、同一顺序、同一 max_tokens、前缀缓存**全臂默认开启**。
#
# 用法： MODE=smoke bash run_hit_grid.sh   # 2 臂（nospec / dflash2-k3）
#        MODE=full  bash run_hit_grid.sh   # 4 臂（+ eagle3-k3 / ngram-7）
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8

VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl
P=/root/myCCFA/probes/p21-prefix-hit-recompute
MODE=${MODE:-full}

case "$MODE" in
  smoke) OUT=/root/ccfa_results/$(date +%F)/p21_hit_smoke; ARMS="S_nospec S_dflash2_k3"; NPROMPT=8 ;;
  full)  OUT=/root/ccfa_results/$(date +%F)/p21_hit;       ARMS="S_nospec S_dflash2_k3 S_eagle3_k3 S_ngram7"; NPROMPT=8 ;;
  *) echo "MODE 必须是 smoke|full"; exit 2 ;;
esac
MAXTOK=32; PORT=32090
mkdir -p "$OUT"

spec_of () { case "$1" in
  S_nospec)      echo "" ;;
  S_dflash2_k3)  echo "{\"method\":\"dflash\",\"model\":\"/root/autodl-tmp/models/dflash2\",\"num_speculative_tokens\":3}" ;;
  S_eagle3_k3)   echo "{\"method\":\"eagle3\",\"model\":\"/root/autodl-tmp/models/eagle3-qwen3-4b\",\"num_speculative_tokens\":3}" ;;
  S_ngram7)      echo "{\"method\":\"ngram\",\"num_speculative_tokens\":7}" ;;
  *) return 1 ;; esac; }

gpu_used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }

reap () {
  kill "$SERVE_PID" 2>/dev/null
  pkill -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null
  pkill -x "VLLM::EngineCore" 2>/dev/null
  local i
  for i in $(seq 1 20); do sleep 3; [ "$(gpu_used)" -lt 300 ] && { sleep 2; return 0; }; done
  echo "  ⚠️ 显存未回落（$(gpu_used) MiB），强杀"
  pkill -9 -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null; pkill -9 -x "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 10); do sleep 3; [ "$(gpu_used)" -lt 300 ] && return 0; done
  echo "  ❌ 显存仍被占用：$(gpu_used) MiB"; return 1; }

serve () { # $1=tag
  local tag=$1 spec; spec=$(spec_of "$tag") || { echo "[$tag] 未知臂"; return 1; }
  local log="$OUT/$tag.serve.log"
  local args=(--model "$MODEL" --served-model-name q3 --port $PORT --max-model-len 8192
              --max-num-seqs 32 --gpu-memory-utilization 0.55 --enforce-eager)
  # 前缀缓存**保持默认开启** —— 这正是本格的条件 C（B108/B116 都只在"缓存开启"下才成立）
  [ -n "$spec" ] && args+=(--speculative-config "$spec")
  "$PY" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$log" 2>&1 &
  SERVE_PID=$!; local ok=0 i
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2
  done
  if [ "$ok" != 1 ]; then echo "[$tag] ❌ serve 未就绪"; grep -nE "Value error|Error|not supported" "$log" | tail -3; return 1; fi
  echo "[$tag] ✅ ready $(grep -oE 'Using V[12] Model Runner' "$log" | head -1) $(grep -oE 'enable_prefix_caching=[A-Za-z]+' "$log" | head -1) $(date +%T)"
  return 0; }

run_arm () { # $1=tag
  local tag=$1
  serve "$tag" || { reap; return 1; }
  "$PY" "$P/probe_prefix_hit.py" --base-url "http://127.0.0.1:$PORT" --model q3 \
      --prompts "$PROMPTS" --n-prompts "$NPROMPT" --max-tokens "$MAXTOK" --logprobs 1 \
      --out "$OUT/$tag" 2>&1 | tee "$OUT/$tag.probe.log" | grep -E "^  ==|^【|^prompt 长度|^A_cold|^B_warm"
  local rc=${PIPESTATUS[0]}
  reap
  return $rc; }

echo "########## p21 前缀命中重算 ($MODE) $(date -Is) ｜ 臂：$ARMS ｜ 每臂 $NPROMPT prompt × 3 段 ##########"
echo "起始显存：$(gpu_used) MiB"
rc_all=0
for tag in $ARMS; do
  echo "---- $tag ----"
  run_arm "$tag" || { echo "[$tag] ⚠️ 有失败"; rc_all=1; }
done
echo "########## 收工 ##########"
echo "末显存：$(gpu_used) MiB"; pgrep -af "vllm[.]entrypoints" || echo "无残留 API server"
pgrep -x "VLLM::EngineCore" >/dev/null && echo "⚠️ 仍有 EngineCore" || echo "无残留 EngineCore"
[ $rc_all -eq 0 ] && echo "ALL_ARMS_OK" || echo "SOME_ARMS_FAILED"
