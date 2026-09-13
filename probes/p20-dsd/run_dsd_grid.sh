#!/usr/bin/env bash
# p20：DSD 表存在性开销（并发 8 + 前缀缓存开启）—— 遵循 notes/prereg/p20-dsd-concurrency.md
# 六臂 A1-A6；每臂独立 serve；并发固定 8；每格 3 次重复请求集；/metrics 计数器做差。
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
DF=/root/autodl-tmp/models/dflash2
P=/root/myCCFA/probes/p20-dsd
OUT=/root/ccfa_results/$(date +%F)/p20_dsd; mkdir -p "$OUT"
PORT=32080; CONC=8; NPROMPT=24; OUTLEN=128; REPS=3

serve () { # $1=tag $2=spec-json(可为空)
  local tag=$1 spec="$2"; local log="$OUT/$tag.serve.log"
  local args=(--model "$MODEL" --served-model-name q3 --port $PORT --max-model-len 8192
              --max-num-seqs 32 --gpu-memory-utilization 0.55 --enforce-eager)
  # 前缀缓存**保持开启**（本格的条件 C；与上游基准相反）
  [ -n "$spec" ] && args+=(--speculative-config "$spec" --per-request-spec-decode-metrics detailed)
  "$PY" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$log" 2>&1 &
  SERVE_PID=$!; local ok=0
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; kill -0 $SERVE_PID 2>/dev/null || break; sleep 2; done
  if [ "$ok" != 1 ]; then echo "[$tag] ❌ 未就绪"; grep -nE "Value error|Error|not supported" "$log" | tail -3; return 1; fi
  echo "[$tag] ✅ ready $(grep -oE 'Using V[12] Model Runner' "$log" | head -1) $(date +%T)"
}

run_arm () { # $1=tag $2=spec
  local tag=$1 spec="$2"
  serve "$tag" "$spec" || { echo "[$tag] SKIP"; return 1; }
  local pre="$OUT/$tag.metrics.pre.json" post="$OUT/$tag.metrics.post.json"
  if [ -n "$spec" ]; then "$PY" "$P/collect_spec_metrics.py" --base "http://127.0.0.1:$PORT" fetch --out "$pre" >/dev/null 2>&1 || true; fi
  for r in 1 2 3; do
    "$PY" -m vllm.benchmarks.serve --backend openai --base-url "http://127.0.0.1:$PORT" \
      --endpoint /v1/completions --model q3 --dataset-name custom --dataset-path /root/autodl-tmp/prompts/prompts_4096.jsonl \
      --num-prompts $NPROMPT --max-concurrency $CONC --ignore-eos --disable-shuffle \
      --output-len $OUTLEN --save-result --result-dir "$OUT" --result-filename "$tag.r$r.json" \
      > "$OUT/$tag.r$r.bench.log" 2>&1 || echo "[$tag r$r] bench 失败"
  done
  if [ -n "$spec" ]; then "$PY" "$P/collect_spec_metrics.py" --base "http://127.0.0.1:$PORT" fetch --out "$post" >/dev/null 2>&1 || true
    "$PY" "$P/collect_spec_metrics.py" --before "$pre" --after "$post" --out "$OUT/$tag.specdiff.json" diff > "$OUT/$tag.specdiff.txt" 2>&1 || true; fi
  kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null; sleep 4
}

echo "########## p20 DSD 网格 $(date -Is) 共 6 臂 × 3 重复 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
run_arm A1_nospec ""
run_arm A2_static_k3        "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":3}"
run_arm A3_static_k0        "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":0}"
run_arm A4_table_allk0      "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":0,\"num_speculative_tokens_per_batch_size\":[[1,8192,0]]}"
run_arm A5_table_switch     "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":3,\"num_speculative_tokens_per_batch_size\":[[1,3,3],[4,8192,0]]}"
run_arm A6_table_const3     "{\"method\":\"dflash\",\"model\":\"$DF\",\"num_speculative_tokens\":3,\"num_speculative_tokens_per_batch_size\":[[1,8192,3]]}"
echo "########## 收工 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -af "vllm[.]entrypoints" || echo 无残留
