#!/bin/bash
cd /Users/leihenan/Desktop/myProject/evidence/accel-gapmap
run(){ echo "##### QUERY: $1"; python3 s12_gs.py "$1" 2>&1 | head -40; echo; }
run 'https://github.com/search?q=repo%3AFMInference%2FFlexGen+maintain&type=issues'
run 'https://github.com/search?q=repo%3AFMInference%2FFlexGen+%22no+longer%22&type=issues'
run 'https://github.com/search?q=repo%3AFMInference%2FFlexGen+archived&type=issues'
run 'https://github.com/search?q=repo%3ASJTU-IPADS%2FPowerInfer+maintain&type=issues'
run 'https://github.com/search?q=repo%3ASJTU-IPADS%2FPowerInfer+%22no+longer%22&type=issues'
run 'https://github.com/search?q=repo%3Abytedance%2Flightseq+maintain&type=issues'
run 'https://github.com/search?q=repo%3Abytedance%2Flightseq+archived&type=issues'
