cd /Users/leihenan/Desktop/myProject/kvresearch
python3 runq.py --tag C1 --kind issue \
 'repo:NVIDIA/TensorRT-LLM "long context" in:title' \
 'repo:NVIDIA/TensorRT-LLM "context parallel" in:title' \
 'repo:NVIDIA/TensorRT-LLM "chunked prefill" in:title' \
 'repo:NVIDIA/TensorRT-LLM "KV cache" in:title' \
 'repo:LMCache/LMCache "long context" in:title' \
 'repo:LMCache/LMCache "context parallel" in:title' \
 'repo:LMCache/LMCache "attention sink" in:title' \
 'repo:ai-dynamo/dynamo "long context" in:title' > /tmp/c1.txt 2>&1
