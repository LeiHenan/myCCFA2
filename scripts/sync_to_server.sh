#!/usr/bin/env bash
# 把本仓库的指定子路径以 `git archive` 方式同步到服务器（服务器上**从不** git pull）。
# 纪律来源：decision log —— 服务器侧禁止 git pull，统一用 git archive 同步。
#
# 用法： scripts/sync_to_server.sh probes/p20-dsd notes/prereg
#        scripts/sync_to_server.sh            # 默认同步 probes/ pipeline/ notes/
# 前提：工作树对将被同步的路径是干净的（git archive 取的是 HEAD）。
set -euo pipefail
SSH="ssh -p 11640 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
HOST=root@connect.westd.seetacloud.com
DEST=/root/myCCFA
PATHS=("$@")
[ ${#PATHS[@]} -eq 0 ] && PATHS=(probes pipeline notes)

cd "$(dirname "$0")/.."
# 检查将被同步的路径是否干净
dirty=$(git status --porcelain -- "${PATHS[@]}" || true)
if [ -n "$dirty" ]; then
  echo "❌ 以下路径有未提交改动，git archive 会漏掉它们：" >&2
  echo "$dirty" >&2
  echo "   请先 git add/commit，或改为只同步已提交路径。" >&2
  exit 1
fi

echo "== 同步到 $DEST ：${PATHS[*]} =="
git archive --format=tar HEAD -- "${PATHS[@]}" | $SSH "$HOST" "mkdir -p $DEST && tar -x -C $DEST && echo '解包完成'"
$SSH "$HOST" "cd $DEST && ls -la ${PATHS[0]} | head -5"
echo "== 完成（服务器侧为 HEAD=$(git rev-parse --short HEAD) 的内容）=="
