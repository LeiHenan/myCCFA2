#!/bin/bash
# usage: f.sh <url> <outfile>
curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" "$1" -o "$2" -w "HTTP %{http_code} size=%{size_download} url=%{url_effective}\n"
