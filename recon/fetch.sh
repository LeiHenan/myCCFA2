#!/bin/bash
# usage: fetch.sh <url> <outbase>
u="$1"; o="$2"
curl -sL --max-time 50 --retry 3 --retry-delay 2 --retry-all-errors \
  -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" \
  "$u" -o "$o.html"
sed -e 's/<[^>]*>/ /g' "$o.html" | tr -s ' \n' ' \n' | grep -v '^\s*$' > "$o.txt"
echo "$o: html=$(wc -c < $o.html) txt=$(wc -c < $o.txt) lines=$(wc -l < $o.txt)"
