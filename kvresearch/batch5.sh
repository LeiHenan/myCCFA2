cd /Users/leihenan/Desktop/myProject/kvresearch
python3 runq.py --tag C5 --kind issue \
 'repo:vllm-project/vllm "KV cache" "long context" in:title' \
 'repo:vllm-project/vllm "1M token" in:title' \
 'repo:vllm-project/vllm "sparse attention" in:title' \
 'repo:vllm-project/vllm "prefix caching" in:title' \
 'repo:sgl-project/sglang "1M" in:title' \
 'repo:sgl-project/sglang "OOM" "long" in:title' \
 'repo:sgl-project/sglang "attention sink" in:body' \
 'repo:NVIDIA/TensorRT-LLM "attention sink" in:title' > /tmp/c5.txt 2>&1
