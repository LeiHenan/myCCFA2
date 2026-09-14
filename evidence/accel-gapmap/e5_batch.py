#!/usr/bin/env python3
"""E5 batch search driver: repo-scoped GitHub issue/PR search for candidate gap claims."""
import sys, json
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from e5_srch import search

QUERIES = [
    # --- Q1 return_logprob + mixed chunked prefill ---
    ("sgl-project/sglang", "return_logprob mixed chunked prefill"),
    ("sgl-project/sglang", "is_mixed_chunk logprob"),
    ("sgl-project/sglang", "mixed chunk logprob unsupported"),
    ("vllm-project/vllm", "mixed chunked prefill logprobs not supported"),
    # --- Q2 priority scheduling schedule policy ---
    ("sgl-project/sglang", "priority scheduling schedule policy fcfs lof"),
    ("sgl-project/sglang", "priority scheduling not supported schedule_policy"),
    # --- Q3 conservative prefill budget ---
    ("sgl-project/sglang", "prefill budget conservative out of memory chunked"),
    ("sgl-project/sglang", "chunked prefill OOM budget too conservative"),
    # --- Q4 TBO without seq_lens_cpu ---
    ("sgl-project/sglang", "two batch overlap seq_lens_cpu"),
    ("sgl-project/sglang", "two-batch-overlap host overhead"),
    # --- Q5 min_p + sampling seed ---
    ("sgl-project/sglang", "min_p sampling seed multinomial"),
    ("sgl-project/sglang", "sampling seed min_p wrong results"),
    ("vllm-project/vllm", "min_p sampling seed"),
    # --- Q6 delayed sample default ---
    ("sgl-project/sglang", "delayed sample default behavior"),
    ("sgl-project/sglang", "delay_sample_func"),
    # --- Q7 custom logit processor + sampling mask ---
    ("sgl-project/sglang", "return_sampling_mask custom logit processor"),
    ("sgl-project/sglang", "DisallowedTokensLogitsProcessor sampling mask"),
    # --- Q8 xgrammar tokenizer unsupported ---
    ("sgl-project/sglang", "tokenizer not supported by XGrammar grammar backend disabled"),
    ("sgl-project/sglang", "grammar backend disabled tokenizer"),
    # --- Q9 return_sampling_mask speculative ---
    ("sgl-project/sglang", "return_sampling_mask speculative decoding"),
    # --- Q10 detokenize top logprobs batching ---
    ("sgl-project/sglang", "detokenize top logprobs batch performance"),
    ("sgl-project/sglang", "detokenization performance optimization top-k"),
    # --- Q11 batch + parallel_sample_num perf ---
    ("sgl-project/sglang", "parallel_sample_num batch performance"),
    ("sgl-project/sglang", "parallel sampling n>1 large batch performance"),
    # --- Q12 duplicated length validation ---
    ("sgl-project/sglang", "length validation tokenizer manager scheduler unify"),
    # --- Q13 .item() syncs in allocation ---
    ("sgl-project/sglang", "item sync allocation per request"),
    ("sgl-project/sglang", "host device sync mem cache allocation"),
    # --- Q14 numpy index perf regression ---
    ("sgl-project/sglang", "numpy arrays block_table performance regression"),
    ("pytorch/pytorch", "numpy array indexing tensor performance regression"),
    # --- Q15 streaming return_token_ids ---
    ("sgl-project/sglang", "return_token_ids not supported with streaming"),
    ("sgl-project/sglang", "return_prompt_token_ids streaming"),
    # --- Q16 responses api logprobs streaming ---
    ("sgl-project/sglang", "logprobs streaming responses api"),
    ("sgl-project/sglang", "include output logprobs streaming"),
    # --- Q17 MCP tool server background streaming ---
    ("sgl-project/sglang", "MCP tool server background streaming"),
    # --- Q18 logprob_start_len streaming sessions ---
    ("sgl-project/sglang", "logprob_start_len streaming session"),
    # --- Q19 metrics utilization stuck 0 ---
    ("sgl-project/sglang", "utilization stuck at 0"),
    ("sgl-project/sglang", "max_running_requests_under_SLO"),
    # --- Q20 prometheus async timer ---
    ("sgl-project/sglang", "prometheus async context manager timer decorator"),
    # --- Q21 FA4 rotary embedding ---
    ("sgl-project/sglang", "FA4 rotary embedding not supported"),
    ("sgl-project/sglang", "flash attention 4 rotary"),
    # --- Q22 block sparsity SM120 ---
    ("sgl-project/sglang", "block sparsity SM120"),
    ("sgl-project/sglang", "blocksparse Blackwell"),
    # --- Q23 triton sliding window kv_indptr ---
    ("sgl-project/sglang", "sliding window kv_indptr_buf"),
    ("sgl-project/sglang", "triton attention sliding window"),
    # --- Q24 base RoPE 2D positions ---
    ("sgl-project/sglang", "2D positions multimodal RoPE RotaryEmbedding"),
    ("sgl-project/sglang", "get_cos_sin_with_position"),
    # --- Q25 prefill-aware SWA ---
    ("sgl-project/sglang", "prefill-aware SWA page_size"),
    ("sgl-project/sglang", "prefill_aware_swa unified memory"),
    # --- Q26 fast_prefill_plan custom mask ---
    ("sgl-project/sglang", "fast_prefill_plan custom mask"),
    # --- Q27 alloc_extend only paged allocator ---
    ("sgl-project/sglang", "alloc_extend paged allocator"),
    ("sgl-project/sglang", "non-paged allocator alloc_extend alloc_decode"),
    # --- Q28 pure SWA page size > 1 ---
    ("sgl-project/sglang", "PureSWA page size"),
    ("sgl-project/sglang", "pure sliding window allocator page_size"),
    # --- Q29 MHATokenToKOnlyPool ---
    ("sgl-project/sglang", "MHATokenToKOnlyPool"),
    # --- Q30 API key multi-tokenizer ---
    ("sgl-project/sglang", "API key multi-tokenizer mode"),
    # --- Q31 logits processor fields ---
    ("sgl-project/sglang", "LogitsProcessorOutput mm_input_embeds"),
    # --- Q32 overlap record mixed mode ---
    ("sgl-project/sglang", "record_batch_in_overlap mixed mode"),
    # --- extra: sampling observers ---
    ("sgl-project/sglang", "sampling observer not supported"),
    # --- extra: chunked prefill auto-off ---
    ("sgl-project/sglang", "chunked prefill not supported multimodal automatically turn off"),
]

if __name__ == "__main__":
    for repo, q in QUERIES:
        try:
            url, rows = search(repo, q)
        except Exception as e:
            print(f"\n### [{repo}] {q}  -> ERROR {e}", flush=True)
            continue
        print(f"\n### [{repo}] {q}  -> {len(rows)} rows", flush=True)
        for num, state, title, u in rows:
            print(f"{num}\t{state}\t{title}\t{u}", flush=True)
