#!/bin/bash
# Fetch many arXiv API queries with rate limiting + backoff.
cd "$(dirname "$0")" || exit 1
mkdir -p xml

QUERIES=(
'abs:%22KV+cache%22+AND+abs:%22eviction%22'
'abs:%22KV+cache%22+AND+abs:%22quantization%22'
'abs:%22key-value+cache%22+AND+abs:%22compression%22'
'abs:%22KV+cache%22+AND+abs:%22offloading%22'
'abs:%22token+eviction%22+AND+abs:%22accuracy%22'
'abs:%22KV+cache%22+AND+abs:%22negative+result%22'
'abs:%22KV+cache%22+AND+abs:%22does+not+help%22'
'abs:%22KV+cache%22+AND+abs:%22myth%22'
'abs:%22KV+cache%22+AND+abs:%22re-evaluation%22'
'abs:%22KV+cache%22+AND+abs:%22rethinking%22'
'abs:%22KV+cache%22+AND+abs:%22no+free+lunch%22'
'abs:%22KV+cache%22+AND+abs:%22misleading%22'
'abs:%22KV+cache%22+AND+abs:%22benchmark%22+AND+abs:%22overhead%22'
'abs:%22attention+sink%22+AND+abs:%22not+necessary%22'
'abs:%22sparse+attention%22+AND+abs:%22KV+cache%22+AND+abs:%22fails%22'
'abs:%22KV+cache%22+AND+abs:%22revisit%22'
'abs:%22KV+cache%22+AND+abs:%22empirical+study%22'
'abs:%22KV+cache%22+AND+abs:%22limitations%22'
'abs:%22KV+cache+compression%22+AND+abs:%22accuracy%22'
'abs:%22eviction%22+AND+abs:%22long-context%22+AND+abs:%22degradation%22'
'abs:%22KV+cache%22+AND+abs:%22overestimated%22'
'abs:%22attention+sinks%22+AND+abs:%22revisit%22'
'abs:%22H2O%22+AND+abs:%22KV+cache%22'
'abs:%22SnapKV%22'
'abs:%22StreamingLLM%22'
'abs:%22KV+cache%22+AND+abs:%22latency%22+AND+abs:%22no+speedup%22'
'abs:%22prefix+caching%22+AND+abs:%22KV+cache%22'
'abs:%22PagedAttention%22'
'abs:%22KV+cache%22+AND+abs:%22accuracy+degradation%22'
'abs:%22quantization%22+AND+abs:%22key-value+cache%22+AND+abs:%22outlier%22'
'abs:%22KV+cache%22+AND+abs:%22fails+to%22'
'abs:%22cache+eviction%22+AND+abs:%22does+not+improve%22'
'abs:%22KV+cache%22+AND+abs:%22efficiency%22+AND+abs:%22trade-off%22+AND+abs:%22wall-clock%22'
'abs:%22sparse+attention%22+AND+abs:%22long+context%22+AND+abs:%22not+better%22'
'abs:%22KV+cache%22+AND+abs:%22survey%22+AND+abs:%22open+problems%22'
)

i=0
for q in "${QUERIES[@]}"; do
  i=$((i+1))
  out="xml/q$(printf '%02d' $i).xml"
  if [ -s "$out" ] && [ "$(wc -c < "$out")" -gt 2000 ]; then
    echo "SKIP $i $out"; continue
  fi
  url="https://export.arxiv.org/api/query?search_query=${q}&start=0&max_results=40&sortBy=submittedDate&sortOrder=descending"
  echo "$q" > "xml/q$(printf '%02d' $i).query"
  ok=0
  for attempt in 1 2 3 4; do
    curl -s -m 90 "$url" -o "$out"
    sz=$(wc -c < "$out" 2>/dev/null || echo 0)
    if [ "$sz" -gt 2000 ]; then ok=1; break; fi
    echo "  attempt $attempt failed (size=$sz) for q$i: $(head -c 80 "$out")"
    sleep 25
  done
  echo "Q$i ok=$ok size=$(wc -c < "$out") :: $q"
  sleep 6
done
echo "ALL DONE"
