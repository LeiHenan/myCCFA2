#!/usr/bin/env bash
# p26 —— cascade attention 到底值多少？（同后端内的干净 A/B）
#
# 背景（全部**我已在安装树亲验**）：
#  · `config/model.py:280  disable_cascade_attn: bool = True` ⇒ **默认关闭**，文档明写 "users must opt in ... by setting this to False"
#  · `flashinfer.py` 的 `use_cascade_attention()` **无条件 return False**（`# TODO: Cascade attention doesn't work, disable it for now`），
#    而 `flash_attn.py:1665` 有真正的启发式实现
#  · 把**我们负载**的参数喂给那条启发式 ⇒ **返回 True**（24 请求共享 4096 前缀；对照 <256 前缀 / <8 请求 / SWA / alibi / DCP>1 全部 False）
#  · S3b：禁用 PR **#26130 MERGED**；其原因 **#25679（"correctness issue with cascade attention on the FlashInfer backend"）CLOSED / NOT_PLANNED**；
#    更早给 FlashInfer 加 cascade 的 **#8132 CLOSED 未 merge** ⇒ **没人正在修**
#  · 机制：`common_prefix_len = min(common_prefix_len, num_computed_tokens.min())`（gpu_model_runner.py:2764）
#    ⇒ 必须有**已缓存**的公共前缀才算数 ⇒ 本实验**保持前缀缓存开启**（默认），并先跑一趟预热把前缀灌进缓存
#
# **唯一变量** = `--hf-overrides '{"disable_cascade_attn": false}'`（开）vs 默认（关），二者**同一后端 FLASH_ATTN**、同一负载、同一并发。
# 口径：共享前缀（17655 字符 ≈4096 token）+ 每请求唯一后缀（≈6000 字符）⇒ 每条请求都要**在共享 KV 上做一次不短的 prefill**，
#       cascade 正是把这部分**跨请求只算一次**。输出长度压到 8 ⇒ 指标由 prefill/TTFT 主导。
#
# 用法： MODE=smoke|full bash run_cascade_ab.sh
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
PROMPTS=/root/autodl-tmp/prompts/shared_prefix_4096_suffix2048_x24.jsonl
MODE=${MODE:-full}
OUT=/root/ccfa_results/$(date +%F)/p26_cascade
mkdir -p "$OUT"; TSV="$OUT/summary.tsv"
printf 'arm\tcascade\tbackend\trep\tout_tput\tmed_ttft_ms\tmed_e2el_ms\tcompleted\n' > "$TSV"
ARMS=${ARMS:-"off on"}; REPS=3; NPROMPT=24; CONC=24; OUTLEN=8; PORT=32120

gpu_used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
reap () {
  kill "$SERVE_PID" 2>/dev/null
  pkill -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null
  pkill -x "VLLM::EngineCore" 2>/dev/null
  local i; for i in $(seq 1 20); do sleep 3; [ "$(gpu_used)" -lt 300 ] && { sleep 2; return 0; }; done
  pkill -9 -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null; pkill -9 -x "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 10); do sleep 3; [ "$(gpu_used)" -lt 300 ] && return 0; done
  echo "  ERROR 显存未回落：$(gpu_used) MiB"; return 1; }
serve () { # $1=arm(on|off)
  local arm=$1 log="$OUT/serve_$1.log"
  local args=(--model "$MODEL" --served-model-name q3 --port $PORT --max-model-len 8192
              --max-num-seqs 32 --gpu-memory-utilization 0.55 --enforce-eager
              --attention-backend FLASH_ATTN)
  # ⚠️ 前缀缓存**保持默认开启** —— cascade 的 common_prefix_len 需要已缓存的公共前缀
  # ⚠️ 开关必须用 **`--no-disable-cascade-attn`**（BooleanOptionalAction）。
  #    我先前试的 `--hf-overrides '{"disable_cascade_attn": false}'` **无效** ——
  #    `hf_overrides` 只喂给 **HF 的 config**（`config/model.py:561-575`），不是 vLLM 的 `ModelConfig` 字段；
  #    实测 `ModelConfig(hf_overrides={"disable_cascade_attn": False}).disable_cascade_attn` 仍是 `True`。
  #    而 argparse 实测 `--no-disable-cascade-attn` ⇒ `False`，`EngineArgs:1840` 会把它传进 ModelConfig。
  #    （血的教训：上一次改这里时，我的两步替换**把这一行整个删掉了**，于是 on 臂其实没带任何 flag、
  #      与 off 臂完全相同 —— 是运行器里的断言把这次无效 A/B 拦了下来，没有产出假结论。）
  [ "$arm" = on ] && args+=(--no-disable-cascade-attn)
  "$PY" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$log" 2>&1 &
  SERVE_PID=$!; local ok=0 i
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2; done
  if [ "$ok" != 1 ]; then echo "[$arm] serve 未就绪"; grep -nE "Error|error" "$log" | tail -3; return 1; fi
  # **断言开关真的生效**（不假设）：从 EngineCore 的配置转储里读回实际取值
  # 旧正则 `disable_cascade_attn[=: ]+` 漏掉了名字后面的**引号**（日志写的是 `'disable_cascade_attn': False`）
  local got; got=$(grep -oE "disable_cascade_attn'?[=: ]+(True|False|true|false)" "$log" | head -1)
  echo "[$arm] ready  $(grep -oE 'Using V[12] Model Runner' "$log" | head -1)  配置回读: ${got:-未找到}"
  if [ "$arm" = on ] && ! echo "$got" | grep -qiE "false"; then
    echo "  ⚠️ 期望 disable_cascade_attn=False 但回读到 '${got:-空}' ⇒ 该臂**开关未生效**，A/B 无效"; return 1; fi
  if [ "$arm" = off ] && echo "$got" | grep -qiE "false"; then
    echo "  ⚠️ 期望默认(True) 但回读到 False ⇒ A/B 无效"; return 1; fi
  return 0; }
one () { # $1=arm $2=rep $3=label
  local arm=$1 r=$2 label=$3 j="$OUT/$1.$3.json"
  "$PY" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
    --model q3 --tokenizer "$MODEL" --dataset-name custom --dataset-path "$PROMPTS" \
    --num-prompts "$NPROMPT" --max-concurrency "$CONC" --ignore-eos --disable-shuffle \
    --output-len "$OUTLEN" --save-result --result-dir "$OUT" --result-filename "$1.$3.json" \
    --label "$1.$3" --percentile-metrics ttft,e2el --metric-percentiles 50 \
    > "$OUT/$1.$3.bench.log" 2>&1
  local rc=$? sz; sz=$(stat -c%s "$OUT/$1.$3.bench.log" 2>/dev/null || echo 0)
  if [ $rc -ne 0 ] || [ ! -s "$j" ]; then
    echo "  [$arm $label] 零产出 rc=$rc log=${sz}B"; tail -4 "$OUT/$1.$3.bench.log"; return 1; fi
  "$PY" - "$arm" "$r" "$j" "$TSV" <<'PYEOF'
import json,sys
arm,r,j,tsv=sys.argv[1:5]
d=json.load(open(j))
def g(k,f="%.3f"):
    v=d.get(k); return (f%v) if isinstance(v,(int,float)) else "nan"
open(tsv,"a").write("\t".join([arm,"on" if arm=="on" else "off","FLASH_ATTN",str(r),
    g("output_throughput"),g("median_ttft_ms","%.2f"),g("median_e2el_ms","%.2f"),str(d.get("completed"))])+"\n")
print("  [%s %s] out_tput=%s med_ttft=%s ms"%(arm,r,g("output_throughput"),g("median_ttft_ms","%.2f")))
PYEOF
  return 0; }
run_arm () { # $1=arm
  local arm=$1 fail=0 r
  echo "---- arm=$arm ----"
  serve "$arm" || { reap; return 1; }
  echo "  [预热]（灌前缀进缓存，丢弃结果）"; one "$arm" 0 warmup || echo "    (预热失败，原因见上)"
  for r in $(seq 1 $REPS); do one "$arm" "$r" "r$r" || fail=1; done
  reap || fail=1
  return $fail; }
echo "########## p26 cascade A/B $(date -Is) ｜ arms=$ARMS ｜ conc=$CONC ｜ 共享前缀 workload ##########"
echo "起始显存：$(gpu_used) MiB"
rc=0
for arm in $ARMS; do run_arm "$arm" || { echo "[$arm] 有失败"; rc=1; }; done
echo "########## 收工 ##########"; echo "末显存：$(gpu_used) MiB"
pgrep -af "vllm[.]entrypoints" || echo "无残留 API server"
pgrep -x "VLLM::EngineCore" >/dev/null && echo "警告 仍有 EngineCore" || echo "无残留 EngineCore"
echo "===== summary.tsv ====="; column -t -s$'\t' "$TSV" 2>/dev/null || cat "$TSV"
[ $rc -eq 0 ] && echo "ALL_ARMS_OK" || echo "SOME_ARMS_FAILED"
