#!/usr/bin/env bash
# 新机器环境安装（AutoDL / 恒源云 容器）—— 与 docs/MACHINE_REQUIREMENTS.md §3「Tier A」一致
#
# 用法：
#   scp -P <port> docs/server_setup.sh root@<host>:/root/setup.sh
#   tmux new-session -d -s setup 'bash /root/setup.sh > /root/setup.log 2>&1'
#
# 设计原则（decision #36）：venv 优先、不污染 conda base；一个 env 只用一种包管理器。
# 磁盘约定：venv 在 /root，缓存与模型在数据盘（AutoDL 上 /root/autodl-tmp），给系统盘留空间。
set -uo pipefail

# ⚠️ 必须在任何 python 进程之前设好 locale（2026-09-12 实测踩坑）：
#   容器常把 LC_ALL=en_US.UTF-8 写进环境但**并未生成该 locale**（`locale -a` 只有 C / C.utf8）。
#     ⇒ `setlocale()` 失败 ⇒ python `import readline` 段错误（rl_initialize → _rl_init_locale）
#     ⇒ vLLM EngineCore 静默死亡，只报 "Engine core initialization failed"，极难定位。
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

PY="${PY:-/root/miniconda3/bin/python3}"
VENV="${VENV:-/root/ccfa_venv}"
VLLM_PIN="${VLLM_PIN:-vllm==0.29.0}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pipcache}"
export HF_HUB_DISABLE_XET=1
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}"

say() { printf '\n=== %s ===\n' "$1"; }

say "0. 硬门槛：驱动 >= 580"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv

say "1. 建 venv（借用 conda 的 python，但不使用 conda 的环境）"
if [ ! -x "${VENV}/bin/python" ]; then
  "${PY}" -m venv "${VENV}" || { echo "!! venv 创建失败"; exit 1; }
fi
# shellcheck disable=SC1091
. "${VENV}/bin/activate"
python -V
echo "python: $(command -v python)"

say "2. 安装引擎 ${VLLM_PIN}"
python -m pip install -U pip
if ! python -m pip install "${VLLM_PIN}"; then
  echo "!! 镜像可能尚未同步该版本，回退官方 PyPI"
  python -m pip install -i https://pypi.org/simple "${VLLM_PIN}" || { echo "!! 安装失败"; exit 1; }
fi

say "3. 自检（版本 / CUDA / DFlash2 注册 / 真算子）"
python - <<'PY'
import torch
print("TORCH :", torch.__version__, "| cuda:", torch.version.cuda, "| dev:", torch.cuda.get_device_name(0))
import vllm
print("VLLM  :", vllm.__version__)
from vllm.model_executor.models.registry import ModelRegistry as M
a = M.get_supported_archs()
print("DFLASH2:", "DFlash2DraftModel" in a,
      "| DFLASH1:", "DFlashDraftModel" in a,
      "| DSPARK:", "DSparkDraftModel" in a)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("MATMUL:", round((x @ x).sum().item(), 2))
PY

say "磁盘占用"
df -h / /root/autodl-tmp 2>/dev/null | tail -3

date
echo SETUP_DONE
