#!/bin/bash
# usage: gh.sh <url> <outname>
url="$1"; f="$2"
curl -sL --max-time 45 "$url" -o "$f.raw.html" -w "%{http_code} "
python3 strip.py "$f.raw.html" > "$f.txt"
wc -l < "$f.txt"
