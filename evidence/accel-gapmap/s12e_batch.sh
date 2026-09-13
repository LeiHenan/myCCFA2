#!/bin/bash
# S12e: batch-fetch GitHub issue/PR pages via fetch.py gh into /tmp/accelsrc/s12e/
# usage: cat urls.txt | xargs -P 8 -n 1 /path/s12e_batch.sh
D=/Users/leihenan/Desktop/myProject/evidence/accel-gapmap
OUT=/tmp/accelsrc/s12e
mkdir -p "$OUT"
url="$1"
key=$(echo "$url" | sed -E 's#https://github.com/##; s#/#_#g')
f="$OUT/$key.gh.txt"
if [ -s "$f" ]; then echo "CACHED $f"; exit 0; fi
python3 "$D/fetch.py" gh "$url" > "$f" 2>&1
echo "DONE $f $(wc -l < "$f") lines"
