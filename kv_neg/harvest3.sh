#!/bin/bash
# Third harvest round: sharp negative-phrasing queries + technique-specific critiques.
cd "$(dirname "$0")" || exit 1
mkdir -p ui3 ids

QUERIES=(
"KV cache compression illusion"
"KV cache compression counterproductive"
"attention sink artifact unnecessary"
"attention sink redundant analysis"
"KV cache eviction no benefit production serving"
"KV cache compression does not pay off"
"KV cache quantization accuracy loss long context study"
"is KV cache compression worth it"
"do we need KV cache compression"
"KV cache eviction evaluation bias"
"KV cache eviction comparison fair baseline"
"KV cache compression speedup not realized"
"KV cache eviction overhead exceeds savings"
"KV cache compression decode latency worse"
"KV cache quantization dequantization overhead"
"H2O eviction critique"
"SnapKV analysis weakness"
"PyramidKV evaluation"
"KV cache eviction attention weights unreliable"
"KV cache compression theoretical impossibility"
"KV cache eviction no silver bullet"
"KV cache compression hidden costs"
"KV cache eviction reasoning models catastrophic"
"KV cache quantization catastrophic forgetting"
"KV cache eviction long context benchmark gaming"
"KV cache compression averaged metrics hide failures"
"KV cache eviction recovery full cache oracle"
"KV cache compression adversarial robustness degradation"
"KV cache eviction safety degradation"
"prefix caching KV reuse attention distribution shift"
"KV cache eviction layer head budget allocation ineffective"
"KV cache compression vision language model failure"
"KV cache eviction attention sink myth debunked"
"KV cache eviction exact top-k optimality bound"
"greedy KV eviction suboptimal analysis"
"KV cache compression energy cost efficiency"
"KV cache compression serving throughput vs latency tradeoff study"
)

n=0
for q in "${QUERIES[@]}"; do
  n=$((n+1))
  enc=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote_plus(sys.argv[1]))" "$q")
  out="ui3/w$(printf '%02d' $n).html"
  if [ ! -s "$out" ]; then
    curl -s -m 60 -A "Mozilla/5.0 (lit-review)" "https://arxiv.org/search/?searchtype=all&query=${enc}&start=0" -o "$out"
    sleep 5
  fi
  echo "W$n size=$(wc -c < "$out") :: $q"
done

python3 - <<'PY'
import glob,re,html
seen={}
for f in sorted(glob.glob('ui3/w*.html')):
    raw=open(f,encoding='utf-8',errors='replace').read()
    for m in re.finditer(r'<a href="https://arxiv\.org/abs/([\d]{4}\.[\d]{4,5})"[^>]*>(.*?)</a>', raw, re.S):
        i=m.group(1); t=re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>','',m.group(2)))).strip()
        if i not in seen: seen[i]=t
with open('ids/harvested3.tsv','w') as fh:
    for i,t in sorted(seen.items()):
        fh.write(f"{i}\t{t}\n")
print("total unique ids round3:",len(seen))
PY
echo "HARVEST3 DONE"
