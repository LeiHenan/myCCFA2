#!/bin/bash
# usage: axs.sh "<url-encoded query>" <tag> [size]
q="$1"; tag="$2"; sz="${3:-50}"
curl -sL --max-time 50 --retry 3 --retry-delay 2 --retry-all-errors \
 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" \
 "https://arxiv.org/search/?searchtype=all&query=${q}&start=0&size=${sz}" -o "s_$tag.html"
echo "=== $tag :: $q :: size=$(wc -c < s_$tag.html) ==="
sed -e 's/<[^>]*>/ /g' "s_$tag.html" | tr -s ' \n' ' \n' | grep -v '^\s*$' \
 | grep -E "^arXiv:" | head -60
