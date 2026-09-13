#!/usr/bin/env bash
# p20 冒烟（先验仪器，再花 GPU）：vLLM 0.29 + 三种投机来源 + 新的 per-request 投机指标能否采到。
# 目的（PIPELINE §3.1）：① 服务能起 ② --per-request-spec-decode-metrics 真能返回字段 ③ EAGLE3 能否加载
# ④ 各方法给出可分辨的接受长度。成本：4 个 serve ×(起~40s + 1 次请求) ≈ 4 min ⇒ ≤0.1 GPU·h
set -uo pipefail
export LC_ALL=C.UTF-8; export LANG=C.UTF-8
VENV=/root/ccfa_venv; export PATH="$VENV/bin:$PATH"; PY=$VENV/bin/python
OUT=/root/ccfa_results/$(date +%F)/p20_smoke; mkdir -p "$OUT"
PORT=32070
PROMPT=$("$PY" -c "import json;print(json.loads(open('/root/autodl-tmp/prompts/prompts_4096.jsonl',encoding='utf-8').readline())['prompt'])")

run_arm () { # $1=tag  $2=spec-config-json（空=不投机）
  local tag=$1 spec="$2"; local log="$OUT/$tag.serve.log"
  local args=(--model /root/autodl-tmp/models/Qwen3-4B --served-model-name q3 --port $PORT
              --max-model-len 8192 --gpu-memory-utilization 0.55 --enforce-eager
              --per-request-spec-decode-metrics detailed)
  [ -n "$spec" ] && args+=(--speculative-config "$spec")
  "$PY" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$log" 2>&1 &
  local pid=$! ok=0
  for i in $(seq 1 100); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 2; done
  if [ "$ok" != 1 ]; then echo "[$tag] ❌ 未就绪"; grep -nE "Error|error|does not support|not supported" "$log" | tail -4; kill $pid 2>/dev/null; return 1; fi
  echo "[$tag] ✅ ready pid=$pid $(date +%T)  runner=$(grep -oE 'Using V[12] Model Runner' "$log" | head -1)"
  "$PY" - "$PORT" "$tag" "$PROMPT" <<'PYEOF'
import json,sys,urllib.request
port,tag,prompt = sys.argv[1], sys.argv[2], sys.argv[3]
body={"model":"q3","prompt":prompt,"max_tokens":64,"temperature":0.0,"ignore_eos":True}
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",data=json.dumps(body).encode(),
                           headers={"Content-Type":"application/json"})
d=json.loads(urllib.request.urlopen(req,timeout=600).read().decode())
# 投机指标可能出现在顶层或 choices[0]；两处都找
found={}
for scope,obj in (("top",d),("choice",(d.get("choices") or [{}])[0])):
    for k,v in (obj or {}).items():
        if "spec" in k.lower() or "accept" in k.lower():
            found[f"{scope}.{k}"]=v if not isinstance(v,(list,dict)) else type(v).__name__
print(f"[{tag}] 投机相关字段: {json.dumps(found, ensure_ascii=False)[:400]}")
print(f"[{tag}] 全部顶层键: {sorted(d.keys())}")
PYEOF
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
}

echo "########## p20 冒烟 $(date -Is) ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
run_arm "nospec" ""
run_arm "ngram"  '{"method":"ngram","num_speculative_tokens":7}'
run_arm "dflash2" '{"method":"dflash","model":"/root/autodl-tmp/models/dflash2","num_speculative_tokens":3}'
run_arm "eagle3" '{"method":"eagle3","model":"/root/autodl-tmp/models/eagle3-qwen3-4b","num_speculative_tokens":3}'
echo "########## 收工 ##########"
nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -af "vllm[.]entrypoints" || echo 无残留
