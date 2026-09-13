cd /Users/leihenan/Desktop/myProject/kvresearch
python3 runq.py --tag C4 --kind issue \
 'repo:NVIDIA/TensorRT-LLM "offload" in:title' \
 'repo:LMCache/LMCache "chunked prefill" in:title' \
 'repo:LMCache/LMCache "hierarchical" in:title' \
 'repo:LMCache/LMCache "MLA" in:title' \
 'repo:flashinfer-ai/flashinfer "attention sink" in:body' \
 'repo:ai-dynamo/dynamo "chunked prefill" in:title' \
 'repo:ai-dynamo/dynamo "offload" in:title' \
 'repo:deepseekai/DeepSpeed "context parallel" in:title' > /tmp/c4.txt 2>&1
