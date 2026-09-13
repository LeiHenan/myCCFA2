#!/usr/bin/env bash
# p22 —— 对上游那条**因果断言**做**匹配对照**：前缀缓存命中率在并发下崩塌，真的是"异构 block size 的 lcm 溢出"造成的吗？
#
# 被检验的上游断言（vLLM C109 关闭理由，引 #42948 线程上 @stecasta 的 block-pool 插桩，原文）：
#   "the real bug is **131% pool overflow from the homogeneous `lcm=256` physical block layout
#    vs DSv4-Flash's `{256, 64, 4, 8}`-block-size KV groups**. Under that overflow, some eviction
#    every alloc cycle is mandatory"
# 配套现象（地图 B112 / vLLM #42948）：命中率 **94.3% @bs1-2 → 1.0% @bs4 → 0.3% @bs8**。
#
# **匹配对照的逻辑**：该断言把原因归给"**逐层 block size 不同** ⇒ 取 lcm ⇒ 物理池溢出 131%"。
# 把那个变量**单独拿掉**——dense Qwen3-4B 逐层 page size 相同 ⇒ lcm 就等于 page size ⇒ 不溢出——预测应**无崩塌**；
# 同时把另一个独立变量**单独变动**：池压力（--gpu-memory-utilization 缩池）。于是 2x2：{宽松池/紧张池} x 并发 {1,8}。
#   · 紧张池 + 并发 8 仍无崩塌 ⇒ 上游归因**获支持** ⇒ B112 对本机从"关键词判死"升级为**机制上不可达**；
#   · 出现崩塌 ⇒ 上游归因**被证伪**，且"自由块与已缓存条目混淆"在 **dense + 池压力**下即成立 ——
#     这个组合**所有已知报告都没覆盖**（既有全是 hybrid/SWA：#42948、#48435、#43447、vllm-ascend 的 DSv4 移植）。
#
# **对照纪律**：全臂 nospec —— p21 已实测 dflash 系前缀缓存 0 命中（#47930），带投机测命中率会被那个已知缺陷污染。
#
# 用法： MODE=smoke bash run_hit_concurrency.sh   |   MODE=full bash run_hit_concurrency.sh
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8

VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B
PROMPTS=/root/autodl-tmp/prompts/prompts_4096.jsonl
H=/root/myCCFA/probes/p22-dense-hit-collapse
MODE=${MODE:-full}
OUT=/root/ccfa_results/$(date +%F)/p22_hit_conc
mkdir -p "$OUT"; TSV="$OUT/summary.tsv"
printf 'cfg\tgpu_mem_util\tconcurrency\tphase\tqueries\thits\thit_rate\tprefill_kv_per_req\n' > "$TSV"

if [ "$MODE" = smoke ]; then
  CFGS="loose_c1 0.55 1"
else
  CFGS="loose_c1 0.55 1
loose_c8 0.55 8
tight_c1 0.30 1
tight_c8 0.30 8"
fi
NPROMPT=24; OUTLEN=128; PORT=32100

gpu_used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }

reap () {
  kill "$SERVE_PID" 2>/dev/null
  pkill -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null
  pkill -x "VLLM::EngineCore" 2>/dev/null
  local i
  for i in $(seq 1 20); do sleep 3; [ "$(gpu_used)" -lt 300 ] && { sleep 2; return 0; }; done
  pkill -9 -f "vllm[.]entrypoints.openai.api_server" 2>/dev/null; pkill -9 -x "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 10); do sleep 3; [ "$(gpu_used)" -lt 300 ] && return 0; done
  echo "  ERROR 显存未回落：$(gpu_used) MiB"; return 1; }

serve () { # $1=util
  local util=$1 log="$OUT/serve_util$1.log"
  "$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name q3 \
    --port $PORT --max-model-len 8192 --max-num-seqs 32 \
    --gpu-memory-utilization "$util" --enforce-eager > "$log" 2>&1 &
  SERVE_PID=$!; local ok=0 i
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SERVE_PID 2>/dev/null || break; sleep 2
  done
  if [ "$ok" != 1 ]; then echo "[util=$util] serve 未就绪"; grep -nE "Error|error" "$log" | tail -3; return 1; fi
  echo "[util=$util] ready $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$log" | head -1) $(date +%T)"
  return 0; }

bench_once () { # $1=conc $2=label —— 断言产物（决策 #123：入口写错时会静默退出 0、日志 0 字节）
  "$PY" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
    --model q3 --tokenizer "$MODEL" --dataset-name custom --dataset-path "$PROMPTS" \
    --num-prompts "$NPROMPT" --max-concurrency "$1" --ignore-eos --disable-shuffle \
    --output-len "$OUTLEN" --label "$2" > "$OUT/$2.bench.log" 2>&1
  local rc=$? sz; sz=$(stat -c%s "$OUT/$2.bench.log" 2>/dev/null || echo 0)
  echo "     bench($2) rc=$rc log=${sz}B"
  if [ "$rc" -ne 0 ] || [ "$sz" -lt 200 ]; then tail -5 "$OUT/$2.bench.log"; return 1; fi
  return 0; }

run_cfg () { # $1=tag $2=util $3=conc
  local tag=$1 util=$2 conc=$3 fail=0
  echo "---- $tag (gpu_mem_util=$util, concurrency=$conc) ----"
  serve "$util" || { reap; return 1; }
  "$PY" "$H/hit_metrics.py" --base "http://127.0.0.1:$PORT" fetch --out "$OUT/$tag.m0.json" || fail=1
  bench_once "$conc" "$tag.A_cold" || fail=1
  "$PY" "$H/hit_metrics.py" --base "http://127.0.0.1:$PORT" fetch --out "$OUT/$tag.m1.json" || fail=1
  bench_once "$conc" "$tag.B_warm" || fail=1
  "$PY" "$H/hit_metrics.py" --base "http://127.0.0.1:$PORT" fetch --out "$OUT/$tag.m2.json" || fail=1
  local pair ph a b
  for pair in "A_cold m0 m1" "B_warm m1 m2"; do
    set -- $pair; ph=$1; a=$2; b=$3
    "$PY" "$H/hit_metrics.py" --before "$OUT/$tag.$a.json" --after "$OUT/$tag.$b.json" \
        --out "$OUT/$tag.$ph.diff.json" diff || fail=1
    "$PY" - "$tag" "$util" "$conc" "$ph" "$OUT/$tag.$ph.diff.json" "$TSV" <<'PYEOF'
import json,sys
tag,util,conc,ph,dp,tsv=sys.argv[1:7]
d=json.load(open(dp))
hr=d.get("hit_rate"); kv=d.get("prefill_kv_computed_per_request")
with open(tsv,"a") as f:
    f.write("\t".join([tag,util,conc,ph,str(d.get("vllm:prefix_cache_queries_total")),
                       str(d.get("vllm:prefix_cache_hits_total")),
                       ("%.4f"%hr) if hr is not None else "NA",
                       ("%.1f"%kv) if kv is not None else "NA"])+"\n")
print("  [%s] hit_rate=%s prefill_kv_per_req=%s"%(ph, ("%.4f"%hr) if hr is not None else "NA", kv))
PYEOF
  done
  reap || fail=1
  return $fail; }

echo "########## p22 匹配对照：dense 模型命中率 vs 并发/池压力 $(date -Is) ##########"
echo "起始显存：$(gpu_used) MiB"
rc=0
while read -r tag util conc; do
  [ -z "${tag:-}" ] && continue
  run_cfg "$tag" "$util" "$conc" || { echo "[$tag] 有失败"; rc=1; }
done <<< "$CFGS"
echo "########## 收工 ##########"
echo "末显存：$(gpu_used) MiB"; pgrep -af "vllm[.]entrypoints" || echo "无残留 API server"
pgrep -x "VLLM::EngineCore" >/dev/null && echo "警告 仍有 EngineCore" || echo "无残留 EngineCore"
echo "===== summary.tsv ====="; column -t -s$'\t' "$TSV" 2>/dev/null || cat "$TSV"
[ $rc -eq 0 ] && echo "ALL_CFGS_OK" || echo "SOME_CFGS_FAILED"
