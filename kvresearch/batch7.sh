cd /Users/leihenan/Desktop/myProject/kvresearch
python3 runq.py --tag E1 --kind pr --state closed \
 'repo:sgl-project/sglang "context parallel" in:title' \
 'repo:sgl-project/sglang "ring attention" in:title' \
 'repo:sgl-project/sglang "chunked prefill" in:title' \
 'repo:sgl-project/sglang "attention sink" in:title' > /tmp/e1.txt 2>&1
