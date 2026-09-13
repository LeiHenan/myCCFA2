#!/bin/bash
# usage: batch.sh <outdir> <url> [url...]
OUT="$1"; shift
mkdir -p "$OUT"
for u in "$@"; do
  name=$(echo "$u" | sed -e 's|https\?://||' -e 's|[/?&=:]|_|g' | cut -c1-160)
  if [ ! -s "$OUT/$name" ]; then
    python3 fetch.py raw "$u" > "$OUT/$name" 2>"$OUT/$name.err"
    echo "$(wc -c < "$OUT/$name") $u -> $OUT/$name"
  else
    echo "CACHED $u -> $OUT/$name"
  fi
done
