#!/usr/bin/env bash
# fetch.sh <url> <outfile>  -- reliable fetch through the local proxy
mkdir -p /Users/leihenan/Desktop/myProject/evidence/kv-gapmap
curl -sL --retry 4 --retry-delay 2 --retry-all-errors -m 40 \
  -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36' \
  -o "$2" -w 'HTTP=%{http_code} bytes=%{size_download} url=%{url_effective}\n' "$1"
