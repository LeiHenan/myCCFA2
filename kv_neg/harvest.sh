#!/bin/bash
# Harvest arXiv IDs via the arXiv search UI for many query strings.
cd "$(dirname "$0")" || exit 1
mkdir -p ui ids

QUERIES=(
"KV cache eviction does not improve"
"KV cache compression negative result"
"KV cache quantization accuracy degradation"
"token eviction hurts accuracy"
"KV cache eviction fails long context"
"KV cache compression pitfalls"
"attention sink not necessary"
"StreamingLLM limitation attention sink"
"H2O heavy hitter oracle eviction"
"SnapKV criticism"
"KV cache quantization no speedup"
"KV cache offloading overhead not worth"
"prefix caching does not help"
"PagedAttention limitations"
"sparse attention long context worse than dense"
"KV cache eviction benchmark reevaluation"
"KV cache eviction rethinking"
"KV cache compression misleading benchmark"
"KV cache eviction query agnostic fails"
"KV cache budget equal comparison unfair"
"KV cache compression reasoning degradation"
"KV cache eviction theoretical analysis limitation"
"eviction attention dilution full cache better"
"KV cache compression not worth the tradeoff"
"KV cache quantization outlier per channel"
"KV cache eviction layer-wise budget ineffective"
"KV cache compression survey open problems"
"KV cache eviction instruction following degrade"
"KV cache eviction vs full cache accuracy drop"
"KV cache compression production scale no benefit"
"KV cache eviction negative results"
"KV cache quantization perplexity misleading"
"KV cache eviction semantic integrity"
"KV cache eviction information loss"
"KV cache compression does not generalize"
"KV cache eviction robustness attack"
"KV cache compression latency wall clock overhead"
"KV cache eviction no free lunch"
)

n=0
for q in "${QUERIES[@]}"; do
  n=$((n+1))
  enc=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote_plus(sys.argv[1]))" "$q")
  out="ui/u$(printf '%02d' $n).html"
  if [ ! -s "$out" ]; then
    curl -s -m 60 -A "Mozilla/5.0 (lit-review)" "https://arxiv.org/search/?searchtype=all&query=${enc}&start=0" -o "$out"
    sleep 3
  fi
  echo "U$n size=$(wc -c < "$out") :: $q"
done

# Extract all IDs + titles into a TSV
python3 - <<'PY'
import glob,re,html,os
seen={}
for f in sorted(glob.glob('ui/u*.html')):
    raw=open(f,encoding='utf-8',errors='replace').read()
    # each result: <p class="list-title ..."><a href="https://arxiv.org/abs/ID">Title</a>
    for m in re.finditer(r'<a href="https://arxiv\.org/abs/([\d]{4}\.[\d]{4,5})"[^>]*>(.*?)</a>', raw, re.S):
        i=m.group(1); t=re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>','',m.group(2)))).strip()
        if i not in seen: seen[i]=t
with open('ids/harvested.tsv','w') as fh:
    for i,t in sorted(seen.items()):
        fh.write(f"{i}\t{t}\n")
print("total unique ids:",len(seen))
PY
echo "HARVEST DONE"
