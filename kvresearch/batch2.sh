cd /Users/leihenan/Desktop/myProject/kvresearch
python3 runq.py --tag C2 --kind issue \
 'repo:flashinfer-ai/flashinfer "long context" in:title' \
 'repo:flashinfer-ai/flashinfer "attention sink" in:title' \
 'repo:deepseekai/DeepSpeed "long context" in:title' \
 'repo:deepseekai/DeepSpeed "KV cache" in:title' \
 'repo:ai-dynamo/dynamo "context parallel" in:title' \
 'repo:ai-dynamo/dynamo "KV cache" in:title' \
 'repo:vllm-project/vllm "needle in a haystack" in:title' \
 'repo:vllm-project/vllm "system prompt" in:title' > /tmp/c2.txt 2>&1
