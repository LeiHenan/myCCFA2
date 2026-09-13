#!/bin/bash
# Fetch abs pages for every ID in ids/harvested.tsv not already downloaded, using N parallel workers.
cd "$(dirname "$0")" || exit 1
mkdir -p abs
LIST=ids/harvested.tsv
WORKERS=4
worker() {
  local shard=$1
  while read -r id rest; do
    [ -z "$id" ] && continue
    if [ -s "abs/$id.html" ] && [ "$(wc -c < "abs/$id.html")" -gt 20000 ]; then continue; fi
    curl -s -m 45 -A "Mozilla/5.0 (lit-review)" "https://arxiv.org/abs/$id" -o "abs/$id.html"
    sz=$(wc -c < "abs/$id.html" 2>/dev/null || echo 0)
    if [ "$sz" -lt 20000 ]; then
      echo "RETRY $id ($sz)"
      sleep 8
      curl -s -m 45 -A "Mozilla/5.0 (lit-review)" "https://arxiv.org/abs/$id" -o "abs/$id.html"
    fi
    sleep 2
  done < "$LIST.$shard"
  echo "worker $shard done"
}
split -n l/$WORKERS -d "$LIST" "$LIST." 2>/dev/null || split -n r/$WORKERS -d "$LIST" "$LIST."
for s in $(seq -f "%02g" 0 $((WORKERS-1))); do worker $s & done
wait
echo "ABS FETCH DONE: $(ls abs | wc -l) files"
