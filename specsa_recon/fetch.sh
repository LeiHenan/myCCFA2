#!/bin/bash
# usage: fetch.sh <cachekey> <url>
KEY="$1"; URL="$2"
OUT="/Users/leihenan/Desktop/myProject/specsa_recon/raw/$KEY"
if [ -s "$OUT" ] && ! grep -q "Rate exceeded\|Too Many Requests\|429" "$OUT" 2>/dev/null; then echo "CACHED $OUT"; exit 0; fi
for i in 1 2 3 4 5 6; do
  curl -sL --max-time 60 --retry 2 --retry-delay 3 --retry-all-errors \
    -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" \
    "$URL" -o "$OUT"
  if [ -s "$OUT" ] && ! grep -q "Rate exceeded\|Too Many Requests" "$OUT" 2>/dev/null; then echo "OK $OUT ($(wc -c < "$OUT") bytes)"; exit 0; fi
  sleep $((i*6))
done
echo "FAIL $OUT"; exit 1
