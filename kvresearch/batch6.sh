cd /Users/leihenan/Desktop/myProject/kvresearch
python3 runq.py --tag C6 --kind issue \
 'repo:vllm-project/vllm "4 GPU" "long context" in:title' \
 'repo:vllm-project/vllm "multi-GPU" in:title' \
 'repo:vllm-project/vllm "1M" in:title' \
 'repo:sgl-project/sglang "ring" in:title' \
 'repo:flashinfer-ai/flashinfer "sm120" in:title' \
 'repo:flashinfer-ai/flashinfer "prefill" in:title' \
 'repo:ai-dynamo/dynamo "attention sink" in:title' \
 'repo:deepseekai/DeepSpeed "inference" "long" in:title' > /tmp/c6.txt 2>&1
