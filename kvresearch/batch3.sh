cd /Users/leihenan/Desktop/myProject/kvresearch
python3 runq.py --tag C3 --kind issue \
 'repo:sgl-project/sglang "hierarchical cache" in:title' \
 'repo:sgl-project/sglang "KV offload" in:title' \
 'repo:sgl-project/sglang "context length" in:title' \
 'repo:sgl-project/sglang "needle" in:title' \
 'repo:vllm-project/vllm "KV cache offload" in:title' \
 'repo:vllm-project/vllm "hierarchical cache" in:title' \
 'repo:vllm-project/vllm "context length" in:title' \
 'repo:vllm-project/vllm "128k" in:title' > /tmp/c3.txt 2>&1
