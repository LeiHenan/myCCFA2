cd /Users/leihenan/Desktop/myProject/kvresearch
python3 runq.py --tag E2 --kind pr --state closed \
 'repo:LMCache/LMCache "MLA" in:title' \
 'repo:LMCache/LMCache "chunked prefill" in:title' \
 'repo:NVIDIA/TensorRT-LLM "chunked prefill" in:title' \
 'repo:flashinfer-ai/flashinfer "attention sink" in:title' > /tmp/e2.txt 2>&1
