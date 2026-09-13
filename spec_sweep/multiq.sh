#!/bin/bash
# pairs of name|url
run(){ name="$1"; url="$2"; curl -sL --max-time 45 "$url" -o "s_$name.html" -w "$name=%{http_code} "; echo ""; }
run vllm_seed "https://github.com/vllm-project/vllm/issues?q=speculative+decoding+seed+reproducib&state=all"
run vllm_diff "https://github.com/vllm-project/vllm/issues?q=%22spec+decode%22+different+output&state=all"
run vllm_rej "https://github.com/vllm-project/vllm/issues?q=rejection+sampling+speculative&state=all"
run sgl_lossless "https://github.com/sgl-project/sglang/issues?q=speculative+lossless&state=all"
run sgl_mask "https://github.com/sgl-project/sglang/issues?q=speculative+tree+mask&state=all"
run sgl_incorrect "https://github.com/sgl-project/sglang/issues?q=speculative+incorrect+output&state=all"
run vllm_unmerged "https://github.com/vllm-project/vllm/issues?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative+verifier"
run sgl_unmerged "https://github.com/sgl-project/sglang/issues?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative"
