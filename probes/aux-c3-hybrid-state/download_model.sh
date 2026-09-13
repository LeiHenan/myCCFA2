#!/usr/bin/env bash
# 下载混合模型权重（Nemotron-H-8B）—— 走 hf-mirror + aria2c 多连接
#
# 为什么不用 `hf download`：2026-09-12 实测在本容器上会卡死重试；aria2c -x16 实测 126 MB/s。
# 用法（实例上，tmux 内）：bash probes/aux-c3-hybrid-state/download_model.sh
set -uo pipefail

REPO_ID=${REPO_ID:-nvidia/Nemotron-H-8B-Base-8K}
DIR=${DIR:-/root/autodl-tmp/models/Nemotron-H-8B}
BASE=${BASE:-https://hf-mirror.com/${REPO_ID}/resolve/main}
FILES_DEFAULT="config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer_config.json"
SHARDS_DEFAULT="model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors"
FILES=${FILES:-$FILES_DEFAULT}
SHARDS=${SHARDS:-$SHARDS_DEFAULT}

mkdir -p "$DIR"; cd "$DIR" || exit 1
echo "=== 目标：$REPO_ID → $DIR  ($(date '+%F %T')) ==="

# 小文件：串行，失败即报
for f in $FILES; do
  if [ -s "$f" ]; then echo "  跳过已存在：$f"; continue; fi
  aria2c -q -c -x8 -s8 --file-allocation=none -o "$f" "$BASE/$f" || echo "  !! 失败：$f"
done

# 大分片：并行 4 路，每路 16 连接
LIST=$(mktemp)
for f in $SHARDS; do
  printf '%s\n  out=%s\n  dir=%s\n' "$BASE/$f" "$f" "$DIR" >> "$LIST"
done
aria2c -c -x16 -s16 -j4 --file-allocation=none --summary-interval=20 -i "$LIST"
rm -f "$LIST"

echo "=== 校验 ==="
du -sh "$DIR"
ls -la "$DIR" | head -14
/root/ccfa_venv/bin/python - <<'PY'
import json, os, glob
d = "/root/autodl-tmp/models/Nemotron-H-8B"
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    w = json.load(open(idx))["weight_map"]
    need = sorted(set(w.values()))
    missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
    print(f"权重分片：需要 {len(need)}，缺失 {len(missing)} {missing}")
    print("总字节：", sum(os.path.getsize(os.path.join(d, f)) for f in need if os.path.exists(os.path.join(d, f))) / 2**30, "GiB")
else:
    print("!! 缺 model.safetensors.index.json")
PY
df -h /root/autodl-tmp | tail -1
echo DL_DONE
