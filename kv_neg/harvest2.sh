#!/bin/bash
# Second harvest round: negative-result-oriented queries.
cd "$(dirname "$0")" || exit 1
mkdir -p ui2 ids

QUERIES=(
"KV cache compression vision language model degradation"
"KV cache quantization breaks long context retrieval"
"KV cache eviction safety alignment jailbreak"
"prefix caching security privacy attack LLM"
"KV cache reuse positional bias"
"attention sink cause not necessary"
"KV cache compression reasoning model collapse"
"KV cache eviction oracle gap analysis"
"quantization KV cache massive activations outlier"
"KV cache eviction random baseline competitive"
"does KV cache compression save memory bytes"
"KV cache eviction latency not improved"
"long context efficiency myths"
"KV cache offloading break even bandwidth"
"sparse attention not needed long context"
"KV cache compression throughput decode bottleneck"
"KV cache eviction multi-turn dialogue degradation"
"KV cache compression agentic workflow"
"prefix cache hit rate production traces"
"KV cache eviction distribution shift robustness"
"attention score correlation weak token importance"
"KV cache compression memory savings overstated"
"KV cache quantization per-channel overhead latency"
"KV cache compression evaluation flawed"
"KV cache eviction information theoretic limit"
"SubQuadratic sparse attention quality gap"
"KV cache pruning no benefit small batch"
"KV cache eviction long context RULER drop"
"KV cache compression accuracy variance seed"
"token dropping attention dilution evidence"
"KV cache quantization kernel speedup limited"
"KV cache reuse prompt duplication overhead"
"KV cache management industrial deployment lessons"
"KV cache compression not free quality cost"
"rethinking efficient attention long context tradeoffs"
"KV cache eviction hierarchical budget critique"
)

n=0
for q in "${QUERIES[@]}"; do
  n=$((n+1))
  enc=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote_plus(sys.argv[1]))" "$q")
  out="ui2/v$(printf '%02d' $n).html"
  if [ ! -s "$out" ]; then
    curl -s -m 60 -A "Mozilla/5.0 (lit-review)" "https://arxiv.org/search/?searchtype=all&query=${enc}&start=0" -o "$out"
    sleep 5
  fi
  echo "V$n size=$(wc -c < "$out") :: $q"
done

python3 - <<'PY'
import glob,re,html
seen={}
for f in sorted(glob.glob('ui2/v*.html')):
    raw=open(f,encoding='utf-8',errors='replace').read()
    for m in re.finditer(r'<a href="https://arxiv\.org/abs/([\d]{4}\.[\d]{4,5})"[^>]*>(.*?)</a>', raw, re.S):
        i=m.group(1); t=re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>','',m.group(2)))).strip()
        if i not in seen: seen[i]=t
with open('ids/harvested2.tsv','w') as fh:
    for i,t in sorted(seen.items()):
        fh.write(f"{i}\t{t}\n")
print("total unique ids round2:",len(seen))
PY
echo "HARVEST2 DONE"
