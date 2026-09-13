#!/bin/bash
cd /Users/leihenan/Desktop/myProject/evidence/accel-gapmap
run(){ echo "##### QUERY: $1"; python3 s12_gs.py "$1" 2>&1 | head -40; echo; }
run 'https://github.com/search?q=repo%3Adeepspeedai%2FDeepSpeed-MII+unmaintained&type=issues'
run 'https://github.com/search?q=repo%3Adeepspeedai%2FDeepSpeed-MII+maintenance&type=issues'
run 'https://github.com/search?q=repo%3Adeepspeedai%2FDeepSpeed-MII+FastGen&type=issues'
run 'https://github.com/search?q=repo%3Amit-han-lab%2Fstreaming-llm+maintain&type=issues'
run 'https://github.com/search?q=repo%3Aturboderp-org%2Fexllamav2+maintain&type=issues'
run 'https://github.com/search?q=repo%3Aturboderp-org%2Fexllamav2+%22stepping+back%22&type=issues'
run 'https://github.com/search?q=repo%3ANVIDIA%2FFasterTransformer+TensorRT-LLM&type=issues'
