#!/usr/bin/env bash
# 装第二个推理引擎（SGLang）到**独立 venv**，放在数据盘（不吃系统盘）
#
# 为什么独立 venv：SGLang 与 vLLM 对 torch/transformers 的钉版本不同，装进同一个 env 会把
# 已验证可用的 vLLM 0.29 环境搞坏（本工作区第一条纪律：一个 env 只用一种包管理器 + 不混装）。
# 为什么用 TUNA 源：实测同一台机器 TUNA ≈51 MB/s，阿里云 PyPI 只有 ~113 KB/s（差 450×）。
#
# 用法（tmux 内）：bash probes/p09-engine-conformance/install_sglang.sh
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

V=${V:-/root/autodl-tmp/venvs/sglang}
PY=${PY:-/root/miniconda3/bin/python3}
IDX=${IDX:-https://pypi.tuna.tsinghua.edu.cn/simple}

echo "=== 创建独立 venv：$V  ($(date '+%F %T')) ==="
mkdir -p "$(dirname "$V")"
[ -x "$V/bin/python" ] || "$PY" -m venv "$V" || { echo "!! venv 创建失败"; exit 1; }
# shellcheck disable=SC1091
. "$V/bin/activate" || exit 1

echo "=== 升级打包工具 ==="
python -m pip install -q -U pip setuptools wheel -i "$IDX" 2>&1 | tail -2

echo "=== 安装 sglang ==="
python -m pip install -U sglang -i "$IDX" 2>&1 | tail -8

echo "=== 自检（必须能 import 且 CUDA 可用）==="
python - <<'PY'
import sys
try:
    import sglang, torch
except Exception as e:                      # noqa: BLE001
    print("!! 导入失败:", type(e).__name__, e)
    sys.exit(1)
print("sglang", sglang.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
try:
    import sglang.srt.entrypoints.http_server  # noqa: F401
    print("http_server 入口：OK")
except Exception as e:                      # noqa: BLE001
    print("!! http_server 导入失败:", type(e).__name__, e)
    sys.exit(1)
PY
rc=$?
echo "自检退出码：$rc"
[ "$rc" = "0" ] && echo INSTALL_DONE || echo INSTALL_FAILED
