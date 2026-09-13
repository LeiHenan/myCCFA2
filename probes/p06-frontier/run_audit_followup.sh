#!/usr/bin/env bash
# 审计后续两项（decision #68，预登记见 notes/prereg/p06-frontier.md 2026-09-13 行）
#   A5      = γ 补格：4k，bs{1,8,32} × depth{1,3,5} × **γ{1,2,5}** × 5 reps（135 格）
#             其余参数与 A1 **逐字相同**（util 0.45 / MAXSEQS 32 / eager / bf16 / REQS 32 / OUTLEN 128）
#             ⇒ 与 A1 的 γ{3,7} 行可直接合并分析。检验结构预测 P1/P2/P3。
#   A1-32k  = 收窄版：**仅 bs=8**，γ{3,7} × depth{1,3,5} × 5 reps（30 格）
#             util 0.72 / MAXSEQS 8 / REQS 8（抬高 util 使 KV 不成为约束）
#             **可采纳前置条件**：该格 `num_preemptions_total == 0` 且
#             serve log 的 `GPU KV cache size ≥ 8×32768 = 262,144` tokens；
#             不满足者只作现象记录、不计入机制判定（判据⑦）。
#
# 用法（tmux 内）：bash probes/p06-frontier/run_audit_followup.sh
set -uo pipefail

[ -f /root/ccfa_env.sh ] && . /root/ccfa_env.sh       # 必须：否则 TARGET 退回 HF id 会触发下载
ROOT=${ROOT:-/root/myCCFA}
PROMPTS=${PROMPTS:-/root/autodl-tmp/prompts}
OUTD=${OUTD:-/root/ccfa_results/$(date +%F)}
cd "$ROOT" || { echo "缺 $ROOT"; exit 1; }

echo "=== A5 开始 $(date '+%F %T') ==="
CTXS=4096 CONCS="1 8 32" DEPTHS="1 3 5" GAMMAS="1 2 5" REPS=5 REQS=32 OUTLEN=128 \
  GPU_UTIL=0.45 MAXSEQS=32 MAXLEN=40960 KV_DTYPE=bfloat16 EAGER=1 \
  DATASET=custom DATASET_DIR="$PROMPTS" OUT="$OUTD/A5_4k" \
  bash probes/p06-frontier/run_t1.sh

echo "=== A1-32k(bs8) 开始 $(date '+%F %T') ==="
CTXS=32768 CONCS="8" DEPTHS="1 3 5" GAMMAS="3 7" REPS=5 REQS=8 OUTLEN=128 \
  GPU_UTIL=0.72 MAXSEQS=8 MAXLEN=40960 KV_DTYPE=bfloat16 EAGER=1 \
  DATASET=custom DATASET_DIR="$PROMPTS" OUT="$OUTD/A1_32k_bs8" \
  bash probes/p06-frontier/run_t1.sh

echo "=== 全部完成 $(date '+%F %T') ==="
echo "分析：python probes/p06-frontier/analyze_bs_grid.py --dir $OUTD/A5_4k --out $OUTD/A5_4k"
echo "      python probes/p06-frontier/analyze_mechanism.py --dir $OUTD/A5_4k"
echo "      python probes/p06-frontier/analyze_bs_grid.py --dir $OUTD/A1_32k_bs8"
echo "      grep -h 'GPU KV cache size' $OUTD/A1_32k_bs8/*.serve.log   # ⑦ 可采纳性"
