#!/usr/bin/env bash
# p10 干净版 onset 实验：**每个配置独立起一个 serve**（避免同一会话内先跑别的并发把缓存预热）
#
# 为什么要重做：上一版把 bs=1 排在 bs=32 之前放在**同一个 serve** 里 ⇒ bs=32 的第一 rep 其实
# 继承了 bs=1 那几轮已填充的前缀缓存，"onset=1" 是会话残留，不是"投机关就立刻命中"。
# 判据仍是 notes/prereg/prefix-reuse-lag.md 的 C1/C2/C3；本次只做干净对照。
#
# 用法（tmux 内）：SPEC=off CONC=32 bash probes/p10-prefix-reuse-lag/run_onset_clean.sh
set -uo pipefail
. /root/ccfa_env.sh 2>/dev/null || true

MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3-4B}
DRAFTER=${DRAFTER:-/root/autodl-tmp/models/dflash2}
VLLMPY=${VLLMPY:-/root/ccfa_venv/bin/python}
OUT=${OUT:-/root/ccfa_results/$(date +%F)/p10_onset_clean}
PORT=${PORT:-8000}
CTX=${CTX:-4096}
MAXLEN=${MAXLEN:-40960}
OUTLEN=${OUTLEN:-128}
REQS=${REQS:-32}
MAXSEQS=${MAXSEQS:-32}
REPS=${REPS:-5}
SPEC=${SPEC:-off}          # off | d5g3
CONC=${CONC:-32}
UTIL=${UTIL:-0.95}
PROMPTS=${PROMPTS:-/root/autodl-tmp/prompts/prompts_${CTX}.jsonl}

mkdir -p "$OUT/bs${CONC}"
tag="u${UTIL}_${SPEC}_c${CONC}"
LOG="$OUT/${tag}.serve.log"
echo "== $tag 独立 serve  $(date '+%F %T') =="

extra=()
[ "$SPEC" = "d5g3" ] && extra=(--speculative-config "{\"model\":\"$DRAFTER\",\"num_speculative_tokens\":3}")

"$VLLMPY" -m vllm.entrypoints.openai.api_server --model "$MODEL" \
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAXSEQS" \
  --kv-cache-dtype bfloat16 --enforce-eager --port "$PORT" "${extra[@]}" > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  sleep 2
done
grep -m1 -o "GPU KV cache size: [0-9,]* tokens" "$LOG" | sed 's/^/  /'

for r in $(seq 1 "$REPS"); do
  f="${tag}_r${r}"
  curl -s --max-time 5 "http://localhost:${PORT}/metrics" > "$OUT/bs${CONC}/${f}.pre.metrics" || true
  vllm bench serve --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name custom --dataset-path "$PROMPTS" --custom-output-len "$OUTLEN" \
    --num-prompts "$REQS" --max-concurrency "$CONC" --disable-shuffle --ignore-eos --seed 0 \
    --save-result --result-dir "$OUT/bs${CONC}" --result-filename "${f}.bench.json" \
    > "$OUT/bs${CONC}/${f}.bench.log" 2>&1 || echo "  !! bench 失败 $f"
  curl -s --max-time 5 "http://localhost:${PORT}/metrics" > "$OUT/bs${CONC}/${f}.post.metrics" || true
done

kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo "== $tag 完成 $(date '+%T') =="
