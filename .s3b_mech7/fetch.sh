#!/bin/bash
# usage: fetch.sh <slug> <url>
slug="$1"; url="$2"
out="/Users/leihenan/Desktop/myProject/.s3b_mech7/raw/${slug}.html"
curl -sL --retry 4 --retry-delay 2 --retry-all-errors \
  -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" \
  "$url" -o "$out" -w "HTTP %{http_code} size=%{size_download} -> $out\n"
