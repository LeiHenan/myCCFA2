#!/bin/bash
# Fetch abs pages for all shards in parallel.
cd "$(dirname "$0")" || exit 1
mkdir -p abs
worker() {
  local f=$1
  while read -r id; do
    [ -z "$id" ] && continue
    if [ -s "abs/$id.html" ] && [ "$(wc -c < "abs/$id.html")" -gt 20000 ]; then continue; fi
    curl -s -m 45 -A "Mozilla/5.0 (lit-review)" "https://arxiv.org/abs/$id" -o "abs/$id.html"
    sz=$(wc -c < "abs/$id.html" 2>/dev/null || echo 0)
    if [ "$sz" -lt 20000 ]; then
      sleep 10
      curl -s -m 45 -A "Mozilla/5.0 (lit-review)" "https://arxiv.org/abs/$id" -o "abs/$id.html"
      sz=$(wc -c < "abs/$id.html" 2>/dev/null || echo 0)
      [ "$sz" -lt 20000 ] && echo "FAIL $id ($sz)"
    fi
    sleep 2
  done < "$f"
  echo "$f done"
}
for f in "$@"; do worker "$f" & done
wait
echo "DONE: $(ls abs | wc -l) files"
