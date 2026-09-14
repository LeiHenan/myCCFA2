#!/bin/bash
# usage: fetch.sh <url> <outfile>
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
curl -sL --max-time 60 --retry 3 --retry-delay 2 --retry-all-errors -A "$UA" "$1" -o "$2"
echo "exit=$? bytes=$(wc -c < "$2")"
