#!/usr/bin/env bash
# 新机器验收脚本 —— 规格与理由见 docs/MACHINE_REQUIREMENTS.md
#
# 用法：
#   bash docs/acceptance_check.sh              # 只检测
#   bash docs/acceptance_check.sh --install    # 先建 venv 并装 vllm==0.29.0，再检测
#   bash docs/acceptance_check.sh --tier-b     # 按降级方案（vllm==0.26.0）检测
#   bash docs/acceptance_check.sh --smoke      # 追加端到端测试（0.6B + ngram；Blackwell/sm_120 必做）
#
# 两条硬门槛：驱动 >= 580（Tier A）、DFlash2DraftModel 被引擎注册。
set -uo pipefail

WANT_TIER="a"
DO_INSTALL=0
DO_SMOKE=0
for arg in "$@"; do
  case "${arg}" in
    --install) DO_INSTALL=1 ;;
    --smoke)   DO_SMOKE=1 ;;
    --tier-b)  WANT_TIER="b" ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "未知参数: ${arg}"; exit 2 ;;
  esac
done

if [ "${WANT_TIER}" = "a" ]; then
  VLLM_PIN="vllm==0.29.0"; DRAFT_ARCH="DFlash2DraftModel"; ARCH_ALT="DFlashDraftModel"
  MIN_DRIVER=580
else
  VLLM_PIN="vllm==0.26.0"; DRAFT_ARCH="DFlashDraftModel"; ARCH_ALT="DFlash2DraftModel"
  MIN_DRIVER=525
fi

PASS=0; FAIL=0; WARN=0
ok()   { printf '  [PASS] %s\n' "$1"; PASS=$((PASS+1)); }
no()   { printf '  [FAIL] %s\n' "$1"; FAIL=$((FAIL+1)); }
warn() { printf '  [WARN] %s\n' "$1"; WARN=$((WARN+1)); }
hdr()  { printf '\n=== %s ===\n' "$1"; }

PY="${PY:-python3}"
VENV="${VENV:-$HOME/ccfa_venv}"

hdr "1. 驱动（硬门槛）"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv
  DMAX=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
         | awk -F. '{print $1}' | sort -n | tail -1)
  FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk '$1 < 2000' | wc -l | tr -d ' ')
  if [ -z "${DMAX}" ]; then
    no "取不到驱动版本"
  elif [ "${DMAX}" -ge "${MIN_DRIVER}" ]; then
    ok "驱动主版本 ${DMAX} >= ${MIN_DRIVER}；空闲卡 ${FREE} 张"
  else
    no "驱动主版本 ${DMAX} < ${MIN_DRIVER}（CUDA 12.x 区间无法运行 CUDA 13 构建的 vLLM >= 0.28）"
  fi
  [ "${FREE}" = "0" ] && warn "当前没有显存占用 < 2 GB 的空闲卡（可与他人共用，但需注意 OOM）"
else
  no "找不到 nvidia-smi（不是 GPU 机器，或未装驱动）"
fi

hdr "2. Python（3.10-3.12，3.11 最佳）"
if command -v "${PY}" >/dev/null 2>&1; then
  PYV=$("${PY}" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
  case "${PYV}" in
    3.10|3.11|3.12) ok "Python ${PYV}" ;;
    *) warn "Python ${PYV} 不在 3.10-3.12 区间" ;;
  esac
else
  no "找不到 ${PY}"
fi

if [ "${DO_INSTALL}" = "1" ]; then
  hdr "2b. 建 venv 并安装 ${VLLM_PIN}（可能耗时 5-15 分钟）"
  if [ ! -x "${VENV}/bin/python" ]; then
    "${PY}" -m venv "${VENV}" && ok "已建 venv: ${VENV}" || no "建 venv 失败"
  else
    ok "复用已有 venv: ${VENV}"
  fi
  # shellcheck disable=SC1091
  . "${VENV}/bin/activate" 2>/dev/null && PY="${VENV}/bin/python"
  "${PY}" -m pip install -U pip >/dev/null 2>&1 && ok "pip 已更新" || warn "pip 更新失败"
  if "${PY}" -m pip install "${VLLM_PIN}"; then ok "已安装 ${VLLM_PIN}"; else no "安装 ${VLLM_PIN} 失败"; fi
fi

hdr "3. torch 与 CUDA 是否真的可用"
if "${PY}" -c 'import torch' >/dev/null 2>&1; then
  TINFO=$("${PY}" - <<'EOF' 2>&1 | tail -1
import torch
try:
    print("%s cuda=%s dev=%s" % (torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)))
except Exception as e:
    print("ERR " + type(e).__name__ + ": " + str(e)[:120])
EOF
)
  case "${TINFO}" in
    ERR*) no "torch 可用但 GPU 不可用：${TINFO}" ;;
    *)    ok "torch: ${TINFO}" ;;
  esac
  if "${PY}" - <<'EOF' >/dev/null 2>&1
import torch
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
_ = (a @ a).sum().item()
EOF
  then ok "真算子通过（4096x4096 bf16 matmul）"; else no "真算子失败（只能 import 不能算）"; fi
else
  warn "当前解释器没有 torch（若已 --install，请确认激活了 venv）"
fi

hdr "4. 决定性检查：drafter 架构是否被引擎注册"
if "${PY}" -c 'import vllm' >/dev/null 2>&1; then
  VV=$("${PY}" -c 'import vllm;print(vllm.__version__)' 2>/dev/null)
  ok "vllm 可 import，版本 ${VV}"
  REG=$("${PY}" - <<EOF 2>&1 | tail -1
from vllm.model_executor.models.registry import ModelRegistry as M
a = M.get_supported_archs()
print("%s %s" % ("${DRAFT_ARCH}" in a, "${ARCH_ALT}" in a))
EOF
)
  case "${REG}" in
    "True "*) ok "${DRAFT_ARCH} 已注册（本机原 550.67 就是在这一步失败的）" ;;
    "False True") no "${DRAFT_ARCH} 未注册（只支持 ${ARCH_ALT}）⇒ 该机器只能走降级方案 Tier B" ;;
    *) no "架构查询失败：${REG}" ;;
  esac
else
  no "vllm 不可 import（装引擎用: bash $0 --install）"
fi

hdr "5. 磁盘"
DU=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2{printf "%.0f", $4/1048576}')
if [ -n "${DU}" ]; then
  if   [ "${DU}" -ge 200 ]; then ok "可用 ${DU} GB（推荐线 200 GB）"
  elif [ "${DU}" -ge 100 ]; then warn "可用 ${DU} GB（过了 100 GB 底线，低于 200 GB 建议线）"
  else no "可用 ${DU} GB < 100 GB 底线"; fi
else
  warn "无法读取磁盘可用空间"
fi

hdr "6. 网络"
if "${PY}" -c 'import urllib.request as u;u.urlopen("https://pypi.org/simple/",timeout=10)' >/dev/null 2>&1; then
  ok "PyPI 可达"
else
  warn "PyPI 直连不可达（考虑 TUNA 镜像）"
fi
if HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 "${PY}" \
   -c 'import urllib.request as u;u.urlopen("https://hf-mirror.com",timeout=10)' >/dev/null 2>&1; then
  ok "hf-mirror.com 可达（记得 HF_HUB_DISABLE_XET=1）"
else
  warn "hf-mirror.com 不可达（若直连 HF 可用则忽略）"
fi

if [ "${DO_SMOKE}" = "1" ]; then
  hdr "7. 端到端 smoke（Qwen3-0.6B + ngram 投机解码；Blackwell/sm_120 必做）"
  if "${PY}" -c 'import vllm' >/dev/null 2>&1; then
    SMOKE_RC=0
    "${PY}" - <<'EOF' || SMOKE_RC=1
import time
from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(model="Qwen/Qwen3-0.6B",
          speculative_config={"method": "ngram", "num_speculative_tokens": 3},
          max_model_len=2048, gpu_memory_utilization=0.30, enforce_eager=True)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=32, temperature=0))
print("SMOKE_TEXT:", out[0].outputs[0].text.strip()[:60], "| elapsed=%.1fs" % (time.time() - t0))
EOF
    if [ "${SMOKE_RC}" -eq 0 ]; then
      ok "端到端生成通过（引擎 + 注意力内核 + 投机路径在该卡上可用）"
    else
      no "端到端生成失败 ⇒ 按 MACHINE_REQUIREMENTS §4.10 换后端重试（FLASH_ATTN → FLASHINFER → TRITON_ATTN），仍失败则换卡"
    fi
  else
    no "vllm 不可 import，无法 smoke（先跑 --install）"
  fi
fi

hdr "汇总"
printf '  PASS=%d  FAIL=%d  WARN=%d  （目标 Tier %s，驱动下限 %s）\n' \
  "${PASS}" "${FAIL}" "${WARN}" "${WANT_TIER}" "${MIN_DRIVER}"
if [ "${FAIL}" -eq 0 ]; then
  echo "  ==> 验收通过，可以开跑。下一步：docs/SERVER_BOOTSTRAP.md §1 起"
else
  echo "  ==> 验收未通过；对照 docs/MACHINE_REQUIREMENTS.md §1（驱动事实链）与 §7（降级方案）"
fi
exit $([ "${FAIL}" -eq 0 ] && echo 0 || echo 1)
