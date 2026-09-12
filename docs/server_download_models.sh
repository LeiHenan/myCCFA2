#!/usr/bin/env bash
# 在实例上下载实验所需权重 —— 走国内快源（实测：ModelScope 6.9 MB/s vs hf-mirror 1.2 MB/s）
#
# 用法（在实例上）：tmux new-session -d -s dl 'bash /root/dl_models.sh > /root/dl.log 2>&1'
#
# 目标模型 Qwen/Qwen3-4B 走 ModelScope（阿里官方源，国内快）；
# drafter mgoin/Qwen3-4B-speculator.dflash2 是 HF-only 仓库 ⇒ 先试 AutoDL 学术加速代理，失败回退 hf-mirror。
set -uo pipefail

MODELS=/root/autodl-tmp/models
CONDA=/root/miniconda3/bin
mkdir -p "${MODELS}"

say() { printf '\n=== %s ===\n' "$1"; }

say "0. 测速（12 秒采样）"
curl -sL -o /dev/null -w "  ModelScope : HTTP %{http_code}  %{speed_download} B/s\n" --max-time 12 \
  "https://modelscope.cn/models/Qwen/Qwen3-4B/resolve/master/model-00001-of-00003.safetensors"

say "1. 目标模型 Qwen/Qwen3-4B  ← ModelScope"
"${CONDA}/pip" install -q -U modelscope 2>&1 | tail -2
"${CONDA}/modelscope" download --model Qwen/Qwen3-4B --local_dir "${MODELS}/Qwen3-4B" \
  || echo "!! ModelScope 下载失败，稍后回退 HF"

say "2. drafter Qwen3-4B-speculator.dflash2  ← 学术加速 → hf-mirror 回退"
export HF_HOME=/root/autodl-tmp/hf HF_HUB_DISABLE_XET=1
# shellcheck disable=SC1091
source /etc/network_turbo >/dev/null 2>&1 || true
if ! "${CONDA}/hf" download mgoin/Qwen3-4B-speculator.dflash2 --local-dir "${MODELS}/dflash2"; then
  echo "!! 学术加速失败，改用 hf-mirror"
  unset http_proxy https_proxy
  export HF_ENDPOINT=https://hf-mirror.com
  "${CONDA}/hf" download mgoin/Qwen3-4B-speculator.dflash2 --local-dir "${MODELS}/dflash2"
fi

say "3. 结果"
du -sh "${MODELS}"/* 2>/dev/null
ls "${MODELS}/dflash2" 2>/dev/null
df -h / /root/autodl-tmp 2>/dev/null | tail -2
echo DL_DONE
