#!/bin/bash
# S12 batch fetch of GitHub issue/PR pages into .s12scratch/items/
cd /Users/leihenan/Desktop/myProject/evidence/accel-gapmap
mkdir -p .s12scratch/items
fetch_one() {
  u="$1"
  num=$(echo "$u" | sed -E 's#.*/(issues|pull)/([0-9]+)#\2#')
  kind=$(echo "$u" | sed -E 's#.*/(issues|pull)/[0-9]+#\1#')
  out=".s12scratch/items/${kind}_${num}.txt"
  if [ -s "$out" ]; then echo "skip $u"; return; fi
  python3 s12_gh.py "$u" > "$out" 2>&1
  echo "done $u -> $out ($(wc -c < "$out") bytes)"
}
export -f fetch_one
cat "$1" | xargs -P 8 -I{} bash -c 'fetch_one "$@"' _ {}
echo ALLDONE
