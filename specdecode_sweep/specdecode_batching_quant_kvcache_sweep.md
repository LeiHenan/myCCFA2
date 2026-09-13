# Prior-Art / Gap-Mapping Sweep: Speculative Decoding under Batching, Quantization, and KV Caching + Engine Integration Status

**Date of sweep:** 2026-09-13
**Window:** 2025-06 → 2026-09 (older only where canonical)
**Engines audited:** vLLM 0.29.0, SGLang 0.5.19 (plus TensorRT-LLM where it closes a question)

**Headline:** Every combination in scope is either documented-incompatible, still open, or closed-unmerged. There is no shipped engine configuration in which spec decode + continuous batching + quantization + prefix caching all compose cleanly.

---

## A. CLOSED (someone shipped or published a working answer)

### A.1 Engine support status — vLLM 0.29.0

v0.29.0 released 09 Sep 2026 (`ckluu`), 594 commits / 277 contributors. Latest release as of sweep date (no 0.30 exists on the releases page).

| # | Question it answers | Who closed it | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A1 | Which spec-decode methods does vLLM 0.29.0 support? | vLLM maintainers | Speculative Decoding docs (0.29.0 + latest) | https://docs.vllm.ai/en/v0.29.0/features/speculative_decoding/ ; https://docs.vllm.ai/en/latest/features/speculative_decoding/ | Shipped | READ BODY |
| A2 | What is the full documented incompatibility list for vLLM 0.29.0? | vLLM maintainers | "Known Feature Incompatibility" section | https://docs.vllm.ai/en/v0.29.0/features/speculative_decoding/ | Shipped | READ BODY |
| A3 | vLLM 0.29.0 release contents | khluu | Release notes | https://github.com/vllm-project/vllm/releases/tag/v0.29.0 | Released 09 Sep 2026 | READ BODY |
| A4 | Dynamic SD + DP: what happens? | vLLM maintainers | Dynamic Speculative Decoding doc | https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ | Shipped | READ BODY |
| A5 | DSD + DP disable path in source | vLLM maintainers | `vllm/config/vllm.py:991` `_maybe_disable_dynamic_sd_for_data_parallel` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/vllm.py | Merged | READ BODY |
| A6 | DSD + full CUDA graphs under MRv2 | ekagra-ranjan / LucasWilkinson | PR #45953 "[MRV2][SD] Make Dynamic SD comatible with Full Cuda Graphs" | https://github.com/vllm-project/vllm/pull/45953 | Merged 4 Jul 2026 | READ BODY |
| A7 | Prefix-cache retention restored for hybrid EAGLE/MTP | vLLM maintainers | PR #55760, #55861 (in 0.29.0 notes) | https://github.com/vllm-project/vllm/releases/tag/v0.29.0 | Merged/shipped | READ BODY |
| A8 | Chunked prefill + spec decode mask bug | seven-mile / benchislett | PR #26231 | https://github.com/vllm-project/vllm/pull/26231 | Merged 6 Oct 2025 | READ BODY |
| A9 | Trailing prefix-cache block drop made opt-out-able | ZeldaHuang / ZJY0516 | PR #53388 | https://github.com/vllm-project/vllm/pull/53388 | Merged 1 Sep 2026 | READ BODY |
| A10 | Spec-decode + quantization: draft-model quant support | shreyas269 | PR #28435 | https://github.com/vllm-project/vllm/pull/28435 | Merged 17 Nov 2025 | READ BODY |
| A11 | FP8 target + Eagle3 drafter KV dtype mismatch | vllmellm | PR #24505 | https://github.com/vllm-project/vllm/pull/24505 | Merged | READ BODY |
| A12 | DFlash + KV quantization ruled fundamentally incompatible | seantechco (filed), closed COMPLETED | Issue #41559 | https://github.com/vllm-project/vllm/issues/41559 | Closed COMPLETED | READ BODY |
| A13 | Per-request acceptance metrics shipped | vLLM maintainers | `--per-request-spec-decode-metrics` (#48915) in 0.29.0 | https://github.com/vllm-project/vllm/releases/tag/v0.29.0 | Shipped | READ BODY |

**vLLM 0.29.0 supported methods** (from the docs page): EAGLE, Multi-Token Prediction (MTP), Draft Model, Parallel Draft Model (PARD), Multi-Layer Perceptron, N-Gram, Suffix Decoding, Hidden State Extraction, **Custom Proposer Backend (Experimental)**, Dynamic Speculative Decoding, Adaptive Verification, Per-Request Acceptance Metrics.

**vLLM 0.29.0 documented incompatibilities — the COMPLETE "Known Feature Incompatibility" section**, verbatim:

> "Pipeline parallelism is not composable with speculative decoding as of vllm<=0.15.0"
> "Speculative decoding with draft models is not supported in vllm<=0.10.0"

Note this section contains only two legacy version-scoped items; every *current* incompatibility is scattered elsewhere (the DSD page, release notes, source comments, individual issues) rather than in this matrix.

**Verbatim, A4 (Dynamic SD doc, "Limitations"):**
> "Tested with Eagle, Eagle-3, and DFlash. Other SD methods may or may not work out of the box"
> "Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature"
> "Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. When DP is enabled, vLLM automatically disables `num_speculative_tokens_per_batch_size` and falls back to the static `num_speculative_tokens` value."

Also verbatim from the same page, the framing of why DSD exists:
> "As BS increases, the effective BS becomes BS*K which increases the compute requirement during verification. When this BS*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT)."

**Verbatim, A5 (source-level confirmation of the DP disable), `_maybe_disable_dynamic_sd_for_data_parallel`:**
> "Dynamic speculative decoding is not supported with data parallelism because data-parallel ranks can select different speculative-token counts, causing DP divergence and deadlocks. Disabling num_speculative_tokens_per_batch_size and falling back to static num_speculative_tokens=%d."

**Verbatim, A3 (0.29.0 release notes, MRV2 gap):**
> "Some features are not yet supported in MRV2 but we are planning for these gaps to be closed within the next 2-3 weeks. These include sequence parallelism, dual-batch overlap, elastic expert parallellism, custom logits processors and certain speculative decoding methods. For now, vLLM will still fall back to use MRV1 if any of these features are configured."
> "we are considering Model Runner V1 deprecated and are targeting v0.32 for its removal. We do not intend to accept any more MRV1-specific improvements or optimizations."

**Verbatim, A3 (0.29.0 spec-decode + Mamba prefix caching):**
> "Mamba prefix caching: internal prefill checkpoints deliver a 9%-25% TTFT improvement (#52789); prefix_cache_retention_interval is now a CLI argument defaulting to 0 (#52216), with dense retention automatically restored for hybrid models using EAGLE/MTP (#55760, #55861)."

**Verbatim, A6 (PR #45953 body):**
> "Followup to DSD: #32374 which ran only with MRv1 and Piecewise graph. This PR mainly enables DSD with MRv2 + FCG. DSD + MRv1 is still on PW. Currently, we capture FCG for decode only for given K say K=3 which meant only multiple of K+1=4 size of graph are captured. However, in DSD the chosen optimal draft k can be <K during scheduling which would lead to missing the graph."

Its benchmark is **BS1 only** and every command passes `--no-enable-prefix-caching`:
> `TPOT (ms) base EAGLE3 - 2.91 / DSD EAGLE3 PW - 2.97 / DSD EAGLE3 FCG - 2.91`

**Verbatim, A8 (PR #26231, chunked prefill):**
> "When a prefill request is chunked due to exceeding the token budget 8192, it contributes to discard_request_indices (it's surely not supposed to be sampled before all prompt tokens are consumed). But the fast path still yields a valid_mask full of 1, which is incorrect and leads to sentinel values leaking and further OOB."

**Verbatim, A9 (PR #53388):**
> "Add an opt-in disable_eagle_block_drop speculative-decoding option for EAGLE-family methods, including dSpark. When enabled, vLLM keeps the trailing prefix-cache block instead of conservatively dropping it after speculative-model prefill. This does not bypass target-model verification."

**Verbatim, A10 (PR #28435):**
> "This PR adds comprehensive quantization support for Eagle and Eagle3 draft models in speculative decoding, including full KV cache quantization support. Previously, Eagle draft models could not use quantized weights in fully connected layers, or quantized KV caches."

**Verbatim, A11 (PR #24505):**
> "The drafter model was not inheriting the kv_cache_dtype from the target model, causing its KV cache to default to uint8 while the target model used fp8, leading to a crash in the Triton kernel."

**Verbatim, A12 (issue #41559 — a CLOSED negative result):**
> "This means DFlash spec decode `cannot` compose with any of `fp8_e5m2`, `fp8_e4m3`, or `turboquant_4bit_nc` — it is locked to `bfloat16` KV cache."
> "The spec decode throughput gains from DFlash do not justify halving the KV pool for long-context workloads."
> Crash signature: `ValueError: KV cache dtype fp8_e5m2 is not supported for non-causal attention`

### A.2 Engine support status — SGLang 0.5.19

v0.5.19 released 05 Sep 2026 (Qiaolin-Yu), 786 PRs / 214 contributors.

| # | Question it answers | Who closed it | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A14 | SGLang supported spec-decode methods + constraints | SGLang maintainers | Speculative Decoding docs | https://docs.sglang.io/docs/advanced_features/speculative_decoding | Shipped | READ BODY |
| A15 | SGLang 0.5.19 release contents | Qiaolin-Yu | Release notes | https://github.com/sgl-project/sglang/releases/tag/v0.5.19 | Released 05 Sep 2026 | READ BODY |
| A16 | Mixed chunk prefill with spec enabled | Oasis-Git | PR #36933 "[2/N][Mixed] Mixed chunk prefill with spec enabled" | https://github.com/sgl-project/sglang/pull/36933 | Merged (in 0.5.19 list) | READ BODY |
| A17 | FP8 MTP acceptance 0% → 60% | ayush1399 / kpham-sgl | PR #32440 | https://github.com/sgl-project/sglang/pull/32440 | Merged 26 Jul 2026 | READ BODY |
| A18 | Spec-v2 constrained-decoding incompatibility | hnyls2002 | PR #12615 "[spec-v2] Fix incompatibility with constrained decoding" | https://github.com/sgl-project/sglang/pull/12615 | Merged 4 Nov 2025 | READ BODY |
| A19 | DFLASH grammar-constrained decoding (removed HTTP 400) | hsthe29 / hnyls2002 | PR #30096 | https://github.com/sgl-project/sglang/pull/30096 | Merged 4 Jul 2026 | READ BODY |
| A20 | DFlash2 local conv + candidate selector | SGLang maintainers | PR #35371 (in 0.5.19 list) | https://github.com/sgl-project/sglang/releases/tag/v0.5.19 | Shipped | READ BODY |
| A21 | Gemma-4 FP8 MTP bridge quant (in 0.5.19 list) | SGLang maintainers | PR #32440 | https://github.com/sgl-project/sglang/releases/tag/v0.5.19 | Shipped | READ BODY |
| A22 | Online MXFP4 quantization of bf16 MTP draft experts (AMD) | zijiecode / HaiShaw | PR #38748 | https://github.com/sgl-project/sglang/pull/38748 | Merged 10 Sep 2026 | READ BODY |

**SGLang 0.5.19 supported methods:** EAGLE-2/EAGLE-3, MTP, UNO, DFLASH, STANDALONE (classic draft model), NGRAM.

**SGLang documented incompatibilities**, verbatim:

| Method | Verbatim constraint |
|---|---|
| UNO | "CUDA and FA3; currently requires TP=PP=1" |
| DFLASH | "No `--enable-dp-attention`; pp_size == 1; disables overlap scheduler & mixed chunked prefill" |
| NGRAM | "CUDA-only; no `--enable-dp-attention`; disables overlap scheduler & mixed chunked prefill" |
| STANDALONE | "Does not support `--enable-dp-attention`" |
| Overlap scheduler (spec V2) | "The overlap scheduler currently only supports `--speculative-eagle-topk 1`; set `--speculative-eagle-topk 1` explicitly. If you explicitly set `--speculative-eagle-topk > 1`, the server will error." |
| NGRAM + page_size>1 | "If `--speculative-ngram-max-bfs-breadth > 1` (thus `speculative_eagle_topk > 1`) and `page_size > 1`, use `--attention-backend flashinfer`; otherwise the server will error." |
| Beam search (new in 0.5.19) | "it does not yet mix with speculative decoding, disaggregation, DP attention, or HiCache (#31626)" |
| Quantization | `--speculative-draft-model-quantization`: "Quantization for the draft model. Use 'unquant' to disable quantization on the draft even when the target is quantized." |

Additional verbatim from SGLang docs (auto-tuning hazard):
> "If you omit `--speculative-eagle-topk`, auto-tuning may pick topk > 1 for some models (e.g. Llama). This is incompatible with the overlap scheduler and may not always trigger an immediate config error, so set `--speculative-eagle-topk 1` explicitly."

**Verbatim, A17 (PR #32440 — quantization silently zeroing MTP acceptance):**
> "The block-FP8 gemma-4-31B-it-assistant checkpoint gets a 0% draft-token acceptance rate, while the BF16 assistant gets approximately 65%."
> "The checkpoint stores pre_projection and post_projection as block-FP8 weights with corresponding weight_scale tensors. However, these layers were constructed with `quant_config=None`, causing the FP8 weights to load without their block scales and corrupting the MTP activations."
> "Before the fix: BF16 MTP acceptance rate: approximately 65% / FP8 MTP acceptance rate: 0%. End-to-end acceptance results after the fix: 60%"

**Verbatim, A18 (PR #12615 — the grammar × spec-decode incompatibility):**
> "This pull request resolves an incompatibility issue that arises when using constrained decoding (specifically with grammars) in conjunction with speculative decoding. It introduces a new internal flag to identify the presence of a speculative algorithm and modifies the batch generation process to ensure that overlap scheduling for constrained decoding is disabled under these specific conditions, thereby preventing potential conflicts and ensuring correct operation."

**Verbatim, A19 (PR #30096 — the engine's own error string):**
> "Under `--speculative-algorithm DFLASH`, any grammar-constrained request is rejected with HTTP 400: `DFLASH speculative decoding does not support grammar-constrained decoding yet.`"
> "This makes DFLASH unusable for structured / forced tool calling and JSON output — a very common serving path (agents, function calling, document extraction)."
> "Simply removing the guard would be unsafe — grammar requests would then run through an unconstrained verify and emit malformed output silently."

**Verbatim, A20 (0.5.19 release notes — batch-size decay quantified by the vendor):**
> "[Spec] DFlash2: local convolution + candidate selector (3.43x over no-spec at batch 1 and about 24% over DFlash at concurrency 64): #35371"

### A.3 Papers / artifacts that close part of the question

| # | Question it answers | Who closed it | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A23 | Does spec decode actually pay off in a production engine at realistic batch sizes? | Liu, Yu, Park, Stoica, Cheung (UC Berkeley) | arXiv 2601.11580, MLSys 2026 **oral** | https://arxiv.org/abs/2601.11580 | Published + MLSys 2026 oral (Thu 21 May 2026, 9:00–9:15 AM PDT) | READ BODY |
| A24 | Full text of A23 | same | ar5iv HTML | https://ar5iv.labs.arxiv.org/html/2601.11580 | Published | READ BODY |
| A25 | MLSys session metadata + author list | MLSys | Oral page | https://mlsys.org/virtual/2026/oral/3782 | Published | READ BODY |
| A26 | Slides for A23 | same | MLSys slide deck | https://mlsys.org/media/mlsys-2026/Slides/3782.pdf | Published | TITLE ONLY (fonts are subset-encoded; text not extractable) |
| A27 | Simulator released with A23 | same | GitHub | https://github.com/SpecDecode-Bench/simulator | Published | TITLE ONLY |
| A28 | Does acceptance collapse as generation length grows? | (Test-Time Speculation authors) | arXiv 2605.09329 | https://arxiv.org/abs/2605.09329 | Published | READ BODY |
| A29 | Can batch-parallel SD hide drafting behind verification? | (MineDraft authors) | arXiv 2603.18016, vLLM plugin | https://arxiv.org/abs/2603.18016 | Published | READ BODY |
| A30 | Can an engine decide when to stop speculating? | (Nightjar authors) | arXiv 2512.22420 | https://arxiv.org/abs/2512.22420 | Published | READ BODY |
| A31 | Does SD degrade under large batch for agent workloads? | (AgentSpec authors, Microsoft Research) | arXiv 2608.24004 | https://arxiv.org/abs/2608.24004 ; https://www.microsoft.com/en-us/research/publication/agentspec-speculative-decoding-for-batch-inference-of-llm-agents/ | Published | READ BODY |
| A32 | Can the target KV cache be reused by the drafter? | (KVShot authors) | arXiv 2604.26412 | https://arxiv.org/abs/2604.26412 | Published | READ BODY |
| A33 | Quantized drafts plug-and-play (MXFP4) | Georganas, Kalamkar, Kozlov, Heinecke (Intel) | arXiv 2503.13565 "ML-SpecQD" | https://arxiv.org/abs/2503.13565 | Published (arXiv v1 17 Mar 2025) | READ BODY |
| A34 | Is SD *compatible* with 4-bit weight-quantized targets? (negative result) | Yudi Zhang et al. | arXiv 2505.22179 | https://arxiv.org/abs/2505.22179 | Published (v2 29 May 2025) | READ BODY |
| A35 | Self-speculative decoding with hierarchical quantized KV | Tiwari et al. | arXiv 2502.10424 "QuantSpec" | https://arxiv.org/abs/2502.10424 | Published | READ BODY |
| A36 | Quantized self-speculative verification | Huang & Wen | arXiv 2603.01399 "Quasar" | https://arxiv.org/abs/2603.01399 | Published | READ BODY |
| A37 | Quantized target + quantized drafter end-to-end | Kim et al. | arXiv 2607.04244 | https://arxiv.org/abs/2607.04244 | Published (v2 7 Jul 2026), ICML 2026 Workshop | READ BODY |
| A38 | Empirical anatomy of SD on consumer hardware (incl. batch/quant costs) | Chordiya | arXiv 2607.17283 "Lossless but Not Free" | https://arxiv.org/abs/2607.17283 | Published | READ BODY |
| A39 | SD + tensor parallelism (async/disaggregated) | (SwiftSpec authors) | arXiv 2506.11309 | https://arxiv.org/abs/2506.11309 | Published | READ BODY |

**Verbatim, A23/A24 (the batch-size finding — the single most load-bearing quote for this sweep):**
> "We present, to our knowledge, the first systematic study of SD on a production-grade and widely deployed inference engine (vLLM), covering multiple SD variants ($n$-gram, EAGLE/EAGLE-3, Draft-Model, Multi-Token Prediction) across diverse workloads, model scales, and batch sizes."
> "**Batch Size.** Increasing batch size improves absolute throughput but systematically reduces the relative speedup of speculative decoding, an effect that is amplified for larger models."
> "for the Llama3.1-8B model on the GSM8K dataset, the speedup of EAGLE over the non-SD baseline decreases from 1.73× to 1.21× as the batch size increases from 1 to 128."
> "with EAGLE, on the ShareGPT dataset, increasing the batch size from 1 to 32 reduces the speedup by 4.3% (1.68× to 1.61×) for Llama3.1-8B, whereas for Llama3-70B, the speedup drops by 14.0% (1.96× to 1.72×)."
> "As batch size grows, verification takes up a larger share of the decoding time, so fixed-$k$ methods would incur larger overhead for verifying draft tokens that are ultimately rejected."

**Verbatim, A23/A24 — the drafting-vs-verification cost at different batch sizes (directly answers the "cost of drafting vs verification" question):**
> "For Llama3-70B, drafting takes from 21% at batch size of 1 to 3% at batch size of 512. For Qwen3-8B, drafting is about 47% at batch size of 1 and decreases to 16% at batch size of 512. As a result, the verification fraction is correspondingly lower (roughly 72% for Llama3-70B and 42% for Qwen3-8B at batch size of 1)."
> "time spent on the proposal stage **increases** (roughly linearly) with the number of proposed tokens, ranging from 42% to 95% in all execution methods, which increases with the model size and batch size." *(note: this is the proposal-share-of-SD-cost figure; read in context at the cited URL)*
> "For $n$-gram, drafting time accounts for less than 2% across all executions, adding negligible overhead to the baseline execution. Both EAGLE and EAGLE3 show similar drafting overhead characteristics, accounting for a 12% to 20% of the execution time at batch size 1, and 3% to 7% at batch size of 512."
> "the sampling stage takes a negligible amount of time, accounting for less than 1.7% of total time across all executions. Lastly, vLLM overhead accounts for 3–12% of total execution time."
> "**Implication.** Verification cost dominates the end-to-end execution of speculative decoding... However, verifying tokens that are later rejected can incur substantial wasted computation."

**Verbatim, A23/A24 (conclusion):**
> "This work presents the first systematic evaluation of speculative decoding (SD) in a real, optimized inference engine. By systematically dissecting end-to-end performance across workloads and SD variants, we identify verification as the dominant bottleneck and reveal strong variability in acceptance behavior across positions, requests, and datasets. Leveraging these insights, we quantify the theoretical upper bound of SD speedup and expose the gap between observed and ideal efficiency."

**Verbatim, A23/A24 (stated future work):**
> "Taken together, these results point to a promising direction for future work: developing an accurate yet lightweight predictor capable of adapting to varying levels of acceptance behavior across workloads, requests and token positions."

**Verbatim, A23/A24 (oracle gap, quantifying the headroom):**
> "on the Instructcoder dataset with a batch size of one and $n$-gram as the SD method, the oracle setup achieves a speedup of approximately 2.75×, whereas the fixed proposed length configuration attains about 2.1× for the best proposed length of 5. Moreover, the gap generally widens as batch size increases: the fixed-$k$ curves drop much faster, while the oracle speedup degrades more gently."

**Verbatim, A28 (Test-Time Speculation — acceptance decay with generation length):**
> "Our studies show that the acceptance length of even state-of-the-art speculators, like DFlash, EAGLE-3 and PARD degrade with generation length, reaching values close to 1 (i.e. no speedup) within just a few thousand output tokens, making speculators ineffective for long-response tasks."

**Verbatim, A29 (MineDraft):**
> "the performance of standard SD is often limited by the strictly sequential execution of these drafting and verification stages... MineDraft realizes the PSD through a novel batch-parallel design that maintains two batches of requests, overlapping drafting for one batch with verification for the other."
> "we have implemented MineDraft as a plugin for vLLM, demonstrating its practicality for production-ready inference systems."

**Verbatim, A30 (Nightjar — explicitly frames the high-load tradeoff):**
> "this method presents a critical trade-off: it improves throughput in low-load, memory-bound systems but degrades performance in high-load, compute-bound environments due to verification overhead. Existing speculative decoding methods use fixed lengths and cannot adapt to workload changes or decide when to stop speculation... Under high load, the benefit of speculation diminishes, while retaining the draft model reduces KV cache capacity, limiting batch size and degrading throughput."

**Verbatim, A31 (AgentSpec):**
> "state-of-the-art speculative decoding algorithms exhibit substantial speed degradation under large batch sizes, limiting their effectiveness to deploy in real-world agent applications. In this work, we first present a systematic analysis of speculative decoding for LLM agents and identify two dominant factors of speedup degradation: high rejection rate of speculative tokens, and under-utilization of dynamic token budgets."

**Verbatim, A32 (KVShot — negative/marginal result on KV reuse for drafters):**
> "Extensive evaluations on Qwen3-8B show that KV-Reuse improves long-range acceptance, although end-to-end speedups remain marginal under current training pipelines."
> "long-range decay persists even in TTT-trained drafters."

**Verbatim, A33 (ML-SpecQD):**
> "we propose using MXFP4 models as drafts in a plug-and-play fashion since the MXFP4 Weight-Only-Quantization (WOQ) merely direct-casts the BF16 target model weights to MXFP4. In practice, our plug-and-play solution gives speedups up to 2x over the BF16 baseline... Combining Multi-Level Speculative Decoding with MXFP4 Quantized Drafts we outperform state-of-the-art speculative decoding, yielding speedups up to 2.72x over the BF16 baseline."

**Verbatim, A34 (the canonical negative compatibility result — 4-bit target + EAGLE-2):**
> "Surprisingly, experiments applying the advanced speculative decoding method EAGLE-2 to various quantized models reveal that the memory benefits from 4-bit weight quantization are diminished by the computational load from speculative decoding. Specifically, verifying a tree-style draft incurs significantly more time overhead than a single-token forward pass on 4-bit weight quantized models."

**Verbatim, A35 (QuantSpec):**
> "existing methods often struggle to achieve significant speedups due to inefficient KV cache optimization strategies and result in low acceptance rates." Claims ">90%" acceptance.

**Verbatim, A36 (Quasar):**
> "Our empirical analysis reveals that while aggressive structural pruning significantly degrades verification accuracy, quantization-based verification preserves the logit distribution with high fidelity while effectively halving memory traffic."

**Verbatim, A37 (quantized target + quantized drafter):**
> "Because the drafter is invoked at every speculative decoding step, we further reduce its overhead with quantization and sliding-window attention, preserving draft-token acceptance while improving long-context decoding latency."

**Verbatim, A38 (consumer-hardware anatomy — a negative result):**
> "three of five configurations decelerate, either because the draft fails to out-speed a small target or because the quantized Metal backend executes "parallel" verification serially, an effect we isolate and quantify."

**Verbatim, A39 (SwiftSpec — why SD + TP is hard):**
> "conventional approaches fail to apply both simultaneously due to imbalanced compute requirements (between draft and target models), KV-cache inconsistencies, and communication overheads under small-batch tensor-parallelism."
> "serves Llama3-70B at 348 tokens/s on 8 Nvidia Hopper GPUs"

### A.4 Specific lead: vLLM PR #54748

**Important correction to the brief.** The brief asked for *"[Spec Decode] Count scheduler steps by the K dynamic SD selected"* as vLLM PR #54748. The actual artifact at #54748 is titled **"[Spec Decode] Make the dynamic SD schedule observable"** by seongyun1104, **OPEN, wants-to-merge, 2 commits**.

| # | Question it answers | Who | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A40 | How do you observe which K the dynamic SD schedule actually chose? | seongyun1104 | PR #54748 | https://github.com/vllm-project/vllm/pull/54748 | **OPEN** (not merged), opened 1 Sep 2026, marked ready for review | READ BODY |
| A41 | Companion RFC on the K-decision interface | seongyun1104 | Issue #54749 (an ISSUE, not a PR) | https://github.com/vllm-project/vllm/issues/54749 | OPEN, labelled `kimi` | READ BODY |

The counter the PR *adds* is `vllm:spec_decode_scheduled_steps`, verbatim:
> "Add `vllm:spec_decode_scheduled_steps`, a counter over scheduler steps labelled by the number of speculative tokens the schedule selected, including 0."

The K=0 blind spot it fixes, verbatim:
> "Steps the schedule sent to K=0 increment no counter, so "the schedule stopped speculating on purpose" and "speculation is not running" look identical in metrics."
> "K=0 is also the tier a batch-keyed schedule reaches first under load, so the blind spot covers the regime an operator most wants to inspect."

The second blind spot, verbatim:
> "The resolved schedule is never printed, so you cannot tell from the logs whether the table you passed survived config resolution."
> "`SpeculativeConfig.__repr__` returns only method, model and num_spec_tokens, so the startup banner shows num_speculative_tokens=6 and nothing about the schedule."

A structural hazard it surfaces — the schedule can be **silently dropped after parsing**, verbatim:
> "`_maybe_disable_dynamic_sd_for_data_parallel` sets `num_speculative_tokens_per_batch_size = None` under DP, so a config that was accepted at the command line may not be the config that is running."

Author's own stated limitation, verbatim:
> "I could not run the full pytest suite on this machine — my local torch is too old for current main, which fails at import in `vllm/utils/torch_utils.py::direct_register_custom_op` — so CI is the first full run."

Author's framing of why the counter is needed, verbatim:
> "I have been measuring dynamic-SD schedules (#48627, #48944) and trying to calibrate a cost model that picks K. The coefficients were guesses and the model picked the wrong tier; there is no counter that would have told me so from inside vLLM."

---

## B. OPEN (explicitly unsolved, with evidence someone is still asking)

### B.1 Documented incompatibility matrices ("speculative decoding is not supported with X")

| # | Question | WHO is still asking | URL | What's missing | VERBATIM quote | Read |
|---|---|---|---|---|---|---|
| B1 | Is spec decode compatible with data parallelism? | vLLM maintainers (documented refusal) | https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ | DSD is silently disabled under DP; no DP-compatible dynamic-K policy exists | "Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks." | READ BODY |
| B2 | Is spec decode compatible with pipeline parallelism? | vLLM maintainers | https://docs.vllm.ai/en/v0.29.0/features/speculative_decoding/ | Only a legacy version-scoped note; actual PP+spec status is tracked in an open RFC | "Pipeline parallelism is not composable with speculative decoding as of vllm<=0.15.0" | READ BODY |
| B3 | Is spec decode compatible with prefix caching? | vLLM maintainers + many operators | https://github.com/vllm-project/vllm/pull/51769 | Fix is a **warning**, not a fix; real mitigation is an unmerged PR | "EAGLE-style speculation (including MTP) cannot reuse the KV of the last matched block: the drafter runs one token ahead of the verified prefix, so KVCacheCoordinator drops one drop-unit off every affected group's prefix-cache hit. **That is correct and intentional**" … "The problem is that nothing tells the operator." | READ BODY |
| B4 | Is `--enable-dp-attention` compatible with spec decode (SGLang)? | SGLang maintainers | https://docs.sglang.io/docs/advanced_features/speculative_decoding | UNO requires TP=PP=1; STANDALONE/NGRAM/DFLASH all refuse dp-attention | "Does not support `--enable-dp-attention`" / "No `--enable-dp-attention`; pp_size == 1" | READ BODY |
| B5 | Is `--enable-mixed-chunk` (chunked prefill) compatible with spec decode (SGLang)? | SGLang maintainers | https://docs.sglang.io/docs/advanced_features/speculative_decoding | DFLASH and NGRAM still *disable* it; only merged in 0.5.19 for other paths | "disables overlap scheduler & mixed chunked prefill" | READ BODY |
| B6 | Is the overlap scheduler compatible with tree speculation? | SGLang maintainers | https://docs.sglang.io/docs/advanced_features/speculative_decoding | Hard constraint topk=1 | "The overlap scheduler currently only supports `--speculative-eagle-topk 1`… If you explicitly set `--speculative-eagle-topk > 1`, the server will error." | READ BODY |
| B7 | Is spec decode compatible with KV-cache quantization (DFlash family)? | closed COMPLETED, but the constraint stands | https://github.com/vllm-project/vllm/issues/41559 | Fix is an unmerged PR (#53979) guarded by `NotImplementedError` | "DFlash spec decode `cannot` compose with any of `fp8_e5m2`, `fp8_e4m3`, or `turboquant_4bit_nc` — it is locked to `bfloat16` KV cache." | READ BODY |

### B.2 Open issues asking for spec decode + prefix caching / + FP8 KV / + chunked prefill / + CUDA graph

| # | Question | WHO is asking | URL | What's missing | VERBATIM quote | Read |
|---|---|---|---|---|---|---|
| B8 | Why does enabling MTP *reduce* prefix cache hit rate? | uOnePiece (H20-3e) | https://github.com/vllm-project/vllm/issues/38182 | Confirmed expected behaviour, but no user-facing mitigation beyond a proposed warning | "Based on Qwen3.5-35B-A3B, why does enabling MTP speculative decoding actually reduce the prefix cache hit rate? … Prefix cache hit rate: around 92% … with `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` Prefix cache hit rate: around 71%" | READ BODY |
| B9 | Can the EAGLE last-block drop be eliminated safely? | skajre (RFC), ZJY0516 (impl) | https://github.com/vllm-project/vllm/issues/50438 ; https://github.com/vllm-project/vllm/pull/50897 | PR #50897 still OPEN (17 commits); contract unvalidated per method | "The one-token contract must be validated for each current implementation—`eagle`, `eagle3`, `mtp`, `dflash`, and `dspark`—before its pop is disabled." | READ BODY |
| B10 | Prefix caching + MTP corrupts tool-call emission | gr4ig (RTX 5090) | https://github.com/vllm-project/vllm/issues/50188 | No root cause; guard behaviour unclear | "3 of 6 repeat runs: response in 1.4–3.5 s with tiny output; message.content contains raw tool-syntax the parser never transformed, no tool_calls" ; "Removing `--enable-prefix-caching` eliminates the failure completely." | READ BODY |
| B11 | DFlash/DSpark acceptance collapses with automatic prefix caching | giorgiopiatti-caffeinated (4×B200) | https://github.com/vllm-project/vllm/issues/47930 | Open, unassigned | Title: "DFlash/DSpark draft acceptance collapses with automatic prefix caching enabled" | READ BODY |
| B12 | SGLang: EAGLE destroys radix prefix reuse for multi-turn traffic | divyvasal (8×B200) | https://github.com/sgl-project/sglang/issues/32459 | Open, assigned to hzh0425; no root cause published | "Enabling EAGLE speculative decoding collapses radix prefix reuse for multi-turn agentic traffic — at any draft length… no crash, silent 97%→40-53% reuse collapse" ; "The residual ~50% "hits" are consistent with within-request/partial matching only." | READ BODY |
| B13 | Can MTP be combined with disaggregation + radix cache? | freeliuzc | https://github.com/sgl-project/sglang/issues/7694 | Closed as **inactive**, never answered | "I think radix cache is enabled in normal speculative decoding，such as MTP… Howerve, it seems that i cannot open both at the same time in MTP mode？" | READ BODY |
| B14 | Does spec decode + P/D disaggregation corrupt KV at block boundaries? | NickLucche (vLLM maintainer) | https://github.com/vllm-project/vllm/issues/43996 | Open, labelled **stale** (>90 days inactivity) | "When P/D disaggregation is used with speculative decoding (EAGLE), the `effective_lookahead_tokens` workaround… only zeroes lookahead on D… P keeps its full `num_lookahead_tokens`, so at block boundaries P allocates one more block than D. The connector's prefix-cache trimming then drops the first data block instead of the extra lookahead block at the end." ; "This is a correctness bug at block boundaries (prompt length is an exact multiple of block_size)." | READ BODY |
| B15 | Is speculation worth it at bs>1? (dynamic-K interface) | seongyun1104 | https://github.com/vllm-project/vllm/issues/54749 | Six competing signals; no agreed decision site | "Six different signals are being proposed for this one decision right now." ; "nothing in the engine currently measures the quantity these policies need. Acceptance is exported in aggregate, but per-(batch bucket, ctx bucket, K) draft and verify latency, first-rejection position, committed tokens per verification, and K-keyed graph hit/miss are not." | READ BODY |
| B16 | Dynamic SD schedules have no seq-length hook → DFlash is a net loss at long context | wannagoxuf (4×RTX 4090) | https://github.com/vllm-project/vllm/issues/54691 | No per-sequence-length disable hook exists | "single-stream 185k-context decode drops from ~71 tok/s (spec OFF) to ~16 tok/s (DT=4). At short context the same setup is a large win (218 tok/s at DT=8), so this is specifically a long-context failure mode - and there is currently no per-sequence-length hook to disable/attenuate speculation the way batch-size-based hooks exist." | READ BODY |
| B17 | (the proposed fix for B16) | ArjunPakhan | https://github.com/vllm-project/vllm/pull/54801 | OPEN, unmerged | "In long-context workloads, draft verification latency scales with sequence length $L$. Allowing dynamic downscaling of $K$… eliminates draft-overhead bottlenecks on long inputs" | READ BODY |
| B18 | Dynamic SD causes catastrophic aggregate-throughput collapse at the batch threshold | tobby168 (GB10/DGX Spark) | https://github.com/vllm-project/vllm/issues/49548 | Open; four code-reading hypotheses, no maintainer diagnosis | "Enabling dynamic speculative decoding via `num_speculative_tokens_per_batch_size` produces two effects… a catastrophic aggregate-throughput collapse under concurrency that looks like a pathological stall, not just "no speculation at high batch."" ; 8 concurrent × 180 tok: `~232 tok/s` static vs `24–157 tok/s` dynamic; wall-clock `~6.3 s` vs `40–60 s` | READ BODY |
| B19 | Does MTP pay off at all, even with good acceptance? | JumpingRain | https://github.com/vllm-project/vllm/issues/47277 | Open, assigned; no fix | "MTP1 reaches around 82%-88% acceptance, and MTP2 can reach mean accepted length above 2.3. However, end-to-end throughput in graph mode is still below the best no-MTP vLLM baseline." ; "MTP1 is about 0.86x of the no-MTP graph baseline, not a speedup." | READ BODY |
| B20 | DSD + MTP crashes full CUDA graph capture | seongyun1104 | https://github.com/vllm-project/vllm/issues/48494 | Open; workaround is static K | "The presence of the batch-size table is the trigger; K=0 tiers are not required." ; "Perf context (v0.24.0…): at client concurrency 256 we see DSD mode incur a 12–25% throughput penalty vs no-spec (oscillating run-to-run) with TTFT p50 inflating ~6×, **even with an all-K=0 table producing near-zero draft tokens**." | READ BODY |
| B21 | MTP + pipeline parallelism | atassis | https://github.com/vllm-project/vllm/issues/44697 | Open RFC; two prior PRs abandoned | "MTP speculative decoding doesn't work under pipeline parallelism (PP > 1). In some configs it crashes; in others it silently diverges from the no-spec greedy baseline." ; "There are two earlier PP+MTP attempts (#39704, #38104), but both are untested and currently conflicting." | READ BODY |
| B22 | Is speculation under batching fundamentally worth it? | TRT-LLM (NVIDIA) | https://github.com/NVIDIA/TensorRT-LLM/issues/16767 | Open; root cause identified but upstream plumbing unfixed | "DSpark speculative decoding… produces the expected accept length only at generation batch size 1. As soon as the disaggregated generation server batches more than one request, the measured accept length collapses toward ~1.0 (drafts are almost never accepted), so DSpark provides essentially no speedup under real serving load." ; "Accept length is a per-request property and should be batch-invariant. MTP on the identical setup is ~2.85 at both batch 1 and batch 16." | READ BODY |
| B23 | FP8 draft-KV dtype crashes startup on hybrid GDN | wannagoxuf | https://github.com/vllm-project/vllm/issues/54690 | Open; fix PR unmerged | "Setting a draft-only KV cache dtype via `--speculative-config '{"kv_cache_dtype": "fp8", ...}'` on a hybrid GDN model… crashes engine startup in the FlashInfer attention metadata builder: `ValueError: Unrecognized dtype: auto`" ; "The FlashInfer fp8-KV path appears to be non-functional on Ada for this (draft, non-causal) configuration, and it takes the whole engine down instead of failing backend selection with a readable error." | READ BODY |
| B24 | Quantized DFlash2 draft silently yields ~0% acceptance | noonghunna (SGLang) | https://github.com/sgl-project/sglang/issues/39087 | Open; reporter explicitly did not root-cause | "A `quantized` DFlash2 draft checkpoint loads into `DFlash2DraftModel`, serves, and produces drafts that are rejected ~100% of the time. There is `no error and no warning` — acceptance `simply collapses from ~3.7 to ~1.0`" ; "Warn (or refuse) when a DFLASH draft resolves to a quantized checkpoint… A one-line boot warning would have saved a full benchmark cycle here." | READ BODY |
| B25 | `deepseek_nextn` hardcodes `quant_config=None` for modelopt_fp4 | (SGLang) | https://github.com/sgl-project/sglang/issues/36599 | Open | "it took `--speculative-draft-model-quantization modelopt_fp4` just to get a quant config to the draft at all… and then this override silently discarded it, which makes the flag a no-op for this case." | READ BODY |
| B26 | GPTQ/INT4 DFlash drafts crash on load | vitoryeso | https://github.com/vllm-project/vllm/pull/55262 | Open; currently `NotImplementedError` | `AttributeError: 'QKVParallelLinear' object has no attribute 'weight'. Did you mean: 'qweight'?` | READ BODY |
| B27 | Refusal: non-causal NVFP4-KV path blocked pending kernel parity | TechPrototyper | https://github.com/vllm-project/vllm/pull/53979 | Open; guard added by author | "I'd rather block it than ship a mode I haven't put paired greedy numbers behind." ; "block-diffusion speculative decoding and 4-bit KV are currently mutually exclusive on exactly the memory-bound cards that want both." | READ BODY |
| B28 | MRv2 spec-decode method coverage gaps | yjyang62 | https://github.com/vllm-project/vllm/issues/49562 | Closed via #49811, but the gap existed through 0.29.0 dev | "Model Runner V2: `vllm/v1/worker/gpu/spec_decode/__init__.py` init_speculator() currently registers only dflash / dspark / Gemma4 MTP / mtp / eagle-family (`use_eagle()`). Other methods, including `extract_hidden_states`, fall through to `NotImplementedError`." | READ BODY |

### B.3 Explicit "still open" statements about the batching question

| # | Question | WHO | URL | VERBATIM | Read |
|---|---|---|---|---|---|
| B29 | Is speculation worth it at high batch — is the engine even measuring the right thing? | seongyun1104 (RFC) | https://github.com/vllm-project/vllm/issues/54749 | "the one component here that was built as a policy rather than a table still cannot express a context-dependent K, and it has been dormant since 2026-05-07. Whether that is the intended destination for this decision is exactly what I do not know." | READ BODY |
| B30 | Cost models for K are uncalibrated | seongyun1104 | https://github.com/vllm-project/vllm/issues/54749 | "A cost model over the same interface gets that cell wrong… Run with the coefficients it ships, sweeping only the acceptance prior across 0.35–0.75… it selects K=0 at the high-batch tier at every context… I am reporting it because it is the honest state of that producer: a stated model, uncalibrated, currently disagreeing with my own measurement." | READ BODY |
| B31 | Marginal-utility criterion (paper-side statement of the same open question) | SparseSpec-L, cited in RFC #54749 | https://github.com/vllm-project/vllm/issues/54749 (citing arXiv 2607.27735) | "extending the speculation horizon can reduce rather than improve speedup when the marginal acceptance probability falls below the relative drafting cost." | READ BODY |

---

## C. ATTEMPTED-AND-ABANDONED (most valuable)

### C.1 dontfix / not-planned / stale / by-design

| # | What was tried | Who closed it | URL | Status | VERBATIM maintainer/author words | Read |
|---|---|---|---|---|---|---|
| C1 | Spec decode TTFT regression (EAGLE3/DFlash) | vLLM triage | https://github.com/vllm-project/vllm/issues/39790 | **Closed as not planned** + labelled `stale` ("Over 90 days of inactivity") | Reporter's numbers, never rebutted: "Mean TPOT: ~17.11 ms (a 39.3% improvement, which is great) / Mean TTFT: ~158.44 ms (a 114% increase) / P99 TTFT: ~1078.03 ms (a 331% increase)." | READ BODY |
| C2 | EAGLE last-block prefix-cache drop | vLLM maintainers (declared by design) | https://github.com/vllm-project/vllm/pull/51769 | By design; PR only adds a warning | "That is correct and intentional — see the discussion in #38182 and the design doc attached to #17137." ; "The drop itself is required for correctness… until that lands, the behaviour is here to stay and the only gap is that it is invisible. This is deliberately a UX-only change: no scheduling, allocation, or cache behaviour is touched." | READ BODY |
| C3 | Spec-decode + P/D disagg KV corruption | vLLM triage | https://github.com/vllm-project/vllm/issues/43996 | Open but **stale**-labelled; maintainer-authored | "This is a correctness bug at block boundaries" (NickLucche, vLLM maintainer) | READ BODY |
| C4 | SGLang jump-forward decoding (a structured-output acceleration) | merrymercy (Lianmin Zheng), merged | https://github.com/sgl-project/sglang/pull/4032 | **Removed**, merged 3 Mar 2025 | "Remove jump forward to simplify the code maintenance" | READ BODY |
| C5 | The promised replacement for C4 | zhyncs (SGLang maintainer) | https://github.com/sgl-project/sglang/pull/4032 | Never implemented; still asked about 16 months later | "Maybe Jump forward will be implemented using speculative decoding later on by @hnyls2002" | READ BODY |
| C6 | Still asking about C4/C5 | tshreyas-aws | https://github.com/sgl-project/sglang/discussions/32352 | Unanswered discussion, 24 Jul 2026 | "Hello, is there a specific reason jump forward decoding was disabled, other than maintenance issues (as described here: PR #4032 ( #4032 ))?" | READ BODY |
| C7 | Grammar-constrained decoding + EAGLE (NPU) | ChefWu551 (author, abandoned) | https://github.com/sgl-project/sglang/pull/20168 | Never merged; author withdrew | "This PR has been re-implemented in #20989. Please refer to that PR instead." ; original failure: "on Ascend NPU, apply_token_bitmask_inplace_triton can fail to enforce grammar masks for XGrammar-constrained decoding: tokens that are masked out (mask_allows=False) may still keep high logits and be selected, causing downstream Tokens not accepted errors in grammar.accept_token" | READ BODY |
| C8 | Grammar + EAGLE json_schema intermittent crash | (SGLang) | https://github.com/sgl-project/sglang/issues/20148 | **Closed as inactive** | "Turn on the eagle/eagle3 spec-decoding and send a request with json_schema, the service may crash intermittently after multiple consecutive requests. This problem also exists in other types of speculative inference on NPU, especially when the number of draft tokens is large." | READ BODY |
| C9 | MTP + disaggregation + radix cache | (SGLang) | https://github.com/sgl-project/sglang/issues/7694 | **Closed as inactive**, never answered | "Howerve, it seems that i cannot open both at the same time in MTP mode？" | READ BODY |
| C10 | SGLang radix + EAGLE NaN crash fix | ashtonchew | https://github.com/sgl-project/sglang/pull/19897 | **Closed** (unmerged) | "The server crashes with `ValueError: Detected errors during sampling! NaN in the logits` when a request hits a radix cache prefix (`cached_tokens > 0`) in Eagle V2 speculative decoding. The crash is 100% reproducible." ; "req_to_token_pool slot mappings are shared between target and draft models, but the KV tensors themselves are separate pools… the corresponding draft KV slots remain uninitialized (contain garbage data)." ; "Originally reported on 8x RTX PRO 6000 Blackwell (SM120)… but the bug is architecture-independent since it stems from the draft/target KV pool separation logic." | READ BODY |
| C11 | Quantized EAGLE draft (first attempt) | shreyas269 (author, self-closed) | https://github.com/vllm-project/vllm/pull/27434 | **Closed-unmerged** 11 Nov 2025 | "Closing this in favor of it's duplicate #28435 which has a DCO sign-off and smoke tests." (superseded, not technically rejected) | READ BODY |

### C.2 Closed-unmerged PRs that added spec-decode + X

| # | What it tried to add | URL | Status | VERBATIM | Read |
|---|---|---|---|---|---|
| C12 | DP-Draft: run draft model with TP=1 (SGLang) | https://github.com/sgl-project/sglang/pull/17418 | **OPEN since 20 Jan 2026, never merged** | "Add DP‑Draft so draft models run locally with TP=1 on each GPU while the target keeps full distributed parallelism, reducing unnecessary communication." Author's own caveats: "Dense models are a better fit: For MoE drafts, EP=1 concentrates all experts on a single GPU and can lead to more fragmented kernel shapes, so throughput may not improve." | READ BODY |
| C13 | Draft-DP mode, take 2 (SGLang) | https://github.com/sgl-project/sglang/pull/23407 | **OPEN, labelled WIP since 21 Apr 2026** | "In speculative decoding with MoE models (e.g., DeepSeek-R1 with MTP), the draft model's multi-step decode phase incurs significant collective communication overhead (allreduce, all-to-all) across TP/EP ranks. Since the draft model is much smaller than the target model, running it with full TP/EP parallelism is wasteful — the communication cost dominates the compute savings." Net gain claimed: only +4.7% throughput. | READ BODY |
| C14 | MTP under pipeline parallelism | https://github.com/vllm-project/vllm/issues/44697 | RFC OPEN; **two prior PRs (#39704, #38104) untested and conflicting** | "There are two earlier PP+MTP attempts (#39704, #38104), but both are untested and currently conflicting." | READ BODY |
| C15 | Lookahead-aware prefix-cache hashing (the real prefix-cache fix) | https://github.com/vllm-project/vllm/pull/50897 | **OPEN, 17 commits, unmerged** | "Skip the legacy EAGLE-style last-block drop only when successor-aware hashing is active. Preserve the existing conservative drop as the fallback for all unsupported configurations." | READ BODY |
| C16 | Disable trailing prefix-cache block drop | https://github.com/vllm-project/vllm/pull/53388 | Merged, but only as an **opt-in** flag | "Add an opt-in disable_eagle_block_drop speculative-decoding option" | READ BODY |
| C17 | DFlash/DSpark stop dropping the last prefix-cache block | https://github.com/vllm-project/vllm/pull/54163 | **OPEN** | "The spurious back-off made every prompt shorter than two mamba blocks skip the final block-aligned chunk, so the mamba recurrent state never materialized on a block boundary and the next turn's fixed-point prefix-cache lookup converged to 0 → the whole context was recomputed on every reply." | READ BODY |
| C18 | Cascade: utility-driven adaptive k for MoE spec decode | https://github.com/vllm-project/vllm/issues/44506 | OPEN feature request | (TITLE ONLY — page body did not render on fetch) | TITLE ONLY |
| C19 | FP8 draft KV isolation for DFlash | https://github.com/vllm-project/vllm/pull/54731 | OPEN (1 Sep 2026) | "When DFlash uses an explicit FP8 draft KV-cache dtype while the target keeps kv_cache_dtype=auto, the shared cache configuration can expose the unresolved target value to FlashInfer metadata construction. This causes startup to fail with `ValueError: Unrecognized dtype: auto`." | READ BODY |
| C20 | NVFP4 KV for DFlash-family drafters (sm12x) | https://github.com/vllm-project/vllm/pull/53979 | OPEN; author added an explicit `NotImplementedError` guard rather than ship unvalidated | "I'd rather block it than ship a mode I haven't put paired greedy numbers behind." | READ BODY |
| C21 | GPTQ DFlash draft support | https://github.com/vllm-project/vllm/pull/55262 | OPEN; raises `NotImplementedError` for other quant methods | "raises `NotImplementedError` for any other quantized method (their parameters do not follow the GPTQ layout this path reads) instead of silently reading wrong attributes." | READ BODY |
| C22 | TrtLlmFp8Experts on SM_12x (consumer Blackwell / DGX Spark) | https://github.com/vllm-project/vllm/pull/43911 | **OPEN since 28 May 2026, unmerged** | "TrtLlmFp8ExpertsBase._supports_current_device() gated on is_device_capability_family(100) (SM_10x — B100/B200 datacenter Blackwell only). This caused MXFP8 MoE to always fall back to MARLIN W8A16 on SM_120/SM_121 hardware… Full enablement requires FlashInfer to ship flashinfer_trtllm_moe compiled for SM_12x targets (tracked in #43906)." | READ BODY |
| C23 | Pack-quantized `fc` weights for DFlash2 drafts (the fix for C24) | https://github.com/sgl-project/sglang/pull/39254 | OPEN (13 Sep 2026) | "the packed `fc` checkpoint tensors could not be resolved by `load_weights()` and were silently skipped. The `fc` layer therefore remained randomly initialized. This does not prevent the model or server from starting, but causes the draft model to produce poor proposals." | READ BODY |
| C24 | Qwen3.5 MTP partial MXFP4 quantization | https://github.com/sgl-project/sglang/pull/38870 | OPEN | (body read; status per subagent) | READ BODY |
| C25 | Quantized NextN drafts for modelopt_fp4 | https://github.com/sgl-project/sglang/issues/36599 | OPEN | "it took `--speculative-draft-model-quantization modelopt_fp4` just to get a quant config to the draft at all… and then this override silently discarded it, which makes the flag a no-op for this case." | READ BODY |
| C26 | Opt-in FP8 proposal head for Qwen4Exp MTP | https://github.com/vllm-project/vllm/pull/56577 | Open (12 Sep 2026) | (READ BODY) | READ BODY |
| C27 | Missing KV scale param crash in Gemma4 MTP | https://github.com/vllm-project/vllm/pull/56539 | Open (11 Sep 2026) | "Fixes an engine initialization crash when serving Gemma 4 speculative decoding (MTP) with an official unquantized BF16 assistant drafter… in front of a quantized or calibrated-KV target model (e.g., ModelOpt NVFP4 or FP8 KV cache)." | READ BODY |

### C.3 Negative results from the literature (an author's own negative conclusion)

| # | Finding | URL | VERBATIM | Read |
|---|---|---|---|---|
| C28 | EAGLE-2 tree verification is *worse* than a single forward on 4-bit targets | https://arxiv.org/abs/2505.22179 | "the memory benefits from 4-bit weight quantization are diminished by the computational load from speculative decoding. Specifically, verifying a tree-style draft incurs significantly more time overhead than a single-token forward pass on 4-bit weight quantized models." | READ BODY |
| C29 | KV reuse for drafters: acceptance improves, speedup does not | https://arxiv.org/abs/2604.26412 | "KV-Reuse improves long-range acceptance, although end-to-end speedups remain marginal under current training pipelines." | READ BODY |
| C30 | 3 of 5 consumer configurations decelerate | https://arxiv.org/abs/2607.17283 | "three of five configurations decelerate, either because the draft fails to out-speed a small target or because the quantized Metal backend executes "parallel" verification serially, an effect we isolate and quantify." | READ BODY |
| C31 | Acceptance decays to ~1 within a few thousand tokens | https://arxiv.org/abs/2605.09329 | "the acceptance length of even state-of-the-art speculators, like DFlash, EAGLE-3 and PARD degrade with generation length, reaching values close to 1 (i.e. no speedup) within just a few thousand output tokens, making speculators ineffective for long-response tasks." | READ BODY |
| C32 | Vendor's own batch-1-vs-64 spec-decode decay | https://github.com/sgl-project/sglang/releases/tag/v0.5.19 | "DFlash2: local convolution + candidate selector (3.43x over no-spec at batch 1 and about 24% over DFlash at concurrency 64)" | READ BODY |
| C33 | Vendor's own bs-scaling decay AND a net loss | https://github.com/sgl-project/sglang/pull/17418 | DP-Draft speedup by batch size: 1.87x @ bs1 → 1.80x @ bs2 → 1.83x @ bs4 → 1.78x @ bs8 → 1.58x @ bs16 → **1.20x @ bs32**. Separately, Kimi-K2: 3173 tok/s with draft vs **3466 tok/s without draft** (draft is a net loss); DeepSeek-R1 NEXTN: 3441 vs 3195 (net win). | READ BODY |
| C34 | Aggressive local fast paths produced non-clean output | https://github.com/vllm-project/vllm/issues/47277 | "We also tried more aggressive local fast paths: a greedy rejection fast path for temperature=0, packed/allfast variants, faster attention metadata experiments, GPU token cache / active request gating variants. Some of these either regressed performance or produced non-clean outputs such as length:1 i…" | READ BODY |

---

## D. Hardware-ruled-out for a single-GPU researcher (1× RTX PRO 6000 Blackwell 96 GB, sm120, single node)

**Critical framing:** sm120 is **not** sm100. vLLM gates many FP4/FP8 datacenter kernel paths on `is_device_capability_family(100)`, which evaluates **False** on sm120 (`120 // 10 = 12 ≠ 100 // 10 = 10`). vLLM's `platforms/cuda.py` special-cases family(120) in several places, confirming this is a distinct code path.

### D.1 Ruled out by GPU count / interconnect (needs >1 GPU or multi-node)

| # | Feature | Why ruled out | URL | Evidence | Read |
|---|---|---|---|---|---|
| D1 | DSD (`num_speculative_tokens_per_batch_size`) + data parallelism | Hard-disabled by engine, deadlock rationale | https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ | "Not compatible with data parallelism (`--data-parallel-size > 1`)… causing DP collective divergence and deadlocks. When DP is enabled, vLLM automatically disables `num_speculative_tokens_per_batch_size`" | READ BODY |
| D2 | MTP spec decode + pipeline parallelism | Crashes or silently diverges | https://github.com/vllm-project/vllm/issues/44697 | "MTP speculative decoding doesn't work under pipeline parallelism (PP > 1). In some configs it crashes; in others it silently diverges from the no-spec greedy baseline." | READ BODY |
| D3 | Spec decode + `--enable-dp-attention` (SGLang) | Refused by config for most methods | https://docs.sglang.io/docs/advanced_features/speculative_decoding | "Does not support `--enable-dp-attention`" (STANDALONE); "No `--enable-dp-attention`" (DFLASH, NGRAM) | READ BODY |
| D4 | UNO decoding | Requires TP=PP=1 **but** requires CUDA+FA3; only viable at exactly 1 GPU with FA3 | https://docs.sglang.io/docs/advanced_features/speculative_decoding | "CUDA and FA3; currently requires TP=PP=1" | READ BODY |
| D5 | Draft-DP / draft-TP-1 schemes | All unmerged; benchmarks are 8×B200 / 16-rank | https://github.com/sgl-project/sglang/pull/17418 ; https://github.com/sgl-project/sglang/pull/23407 | Benchmark setup "8×B200, DeepSeek-R1 MTP, FP8" | READ BODY |
| D6 | Spec decode + P/D disaggregation | Multi-node by construction; and the KV-trim bug is unfixed+stale | https://github.com/vllm-project/vllm/issues/43996 | "This is a correctness bug at block boundaries" — labelled `stale` | READ BODY |
| D7 | Spec decode + expert parallelism for large MoE targets | Requires multi-GPU for the target at all; draft EP explicitly counterproductive | https://github.com/sgl-project/sglang/pull/17418 | "For MoE drafts, EP=1 concentrates all experts on a single GPU and can lead to more fragmented kernel shapes, so throughput may not improve." | READ BODY |
| D8 | Disaggregated serving + DSpark/MTP | Single-node disagg still hits the accept-length collapse | https://github.com/NVIDIA/TensorRT-LLM/issues/16767 | "As soon as the disaggregated generation server batches more than one request, the measured accept length collapses toward ~1.0" | READ BODY |
| D9 | SwiftSpec-style async/disaggregated SD | 8×Hopper | https://arxiv.org/abs/2506.11309 | "serves Llama3-70B at 348 tokens/s on 8 Nvidia Hopper GPUs" | READ BODY |

### D.2 Ruled out / degraded by sm120 arch (kernel-level)

| # | Feature | Status on sm120 | URL | VERBATIM | Read |
|---|---|---|---|---|---|
| D10 | NVFP4 MoE (`FLASHINFER_CUTLASS`, `FLASHINFER_CUTEDSL`, `TRTLLM_MOE`, CUTLASS MoE) | **Reported broken on RTX PRO 6000 Blackwell (SM12.0)** | https://github.com/vllm-project/vllm/issues/33416 | "vLLM v0.15.0 fails to run NVFP4 quantized MoE models on RTX Blackwell GPUs (compute capability 12.0, e.g., RTX PRO 6000 Blackwell Workstation Edition). The NVFP4 MoE backend selection code only checks for SM9.0 (Hopper) and SM10.x family (data center Blackwell B100/B200), but not SM12.0 (RTX Blackwell)." ; error: `ValueError: NvFp4 MoE backend 'FLASHINFER_CUTLASS' does not support the deployment configuration since kernel does not support current device.` ; "The device capability checks in the NVFP4 MoE backend selection code use is_device_capability_family(100) which only matches SM10.x" | READ BODY |
| D11 | MXFP8 MoE native path | **Falls back to MARLIN W8A16** (dequant to BF16) | https://github.com/vllm-project/vllm/pull/43911 | "This caused MXFP8 MoE to always fall back to MARLIN W8A16 on SM_120/SM_121 hardware (RTX 5000-series, DGX Spark / GB10), even though SM_12x implements the same tcgen05.mma MX tensor core instructions as SM_10x." ; "Full enablement requires FlashInfer to ship flashinfer_trtllm_moe compiled for SM_12x targets (tracked in #43906)." — **PR is OPEN, unmerged since 28 May 2026** | READ BODY |
| D12 | FP8 KV cache in a spec-decode config | Native crash on pre-sm100 | https://github.com/vllm-project/vllm/issues/54690 | "the engine then dies with a native crash (no Python traceback) during attention init on SM89" ; related SM89-fp8-KV work is still open (#49077, #52202) | READ BODY |
| D13 | DFlash/DSpark + any KV quantization (fp8/fp4/turboquant) | **Mutually exclusive** | https://github.com/vllm-project/vllm/issues/41559 ; https://github.com/vllm-project/vllm/pull/53979 | "DFlash spec decode `cannot` compose with any of `fp8_e5m2`, `fp8_e4m3`, or `turboquant_4bit_nc` — it is locked to `bfloat16` KV cache." ; "block-diffusion speculative decoding and 4-bit KV are currently mutually exclusive on exactly the memory-bound cards that want both." | READ BODY |
| D14 | CUDA Graph capture for spec decode | Full CG only via MRv2; MRv1 piecewise-only; and DSD+MTP full-CG capture **crashes** | https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ ; https://github.com/vllm-project/vllm/issues/48494 | "Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature" ; "The presence of the batch-size table is the trigger; K=0 tiers are not required." | READ BODY |
| D15 | Draft exceeding captured graph range | Runtime assert | https://github.com/vllm-project/vllm/issues/32591 | "The scheduler schedules up to max-num-batched-tokens tokens (e.g. 2048), but during spec decoding with draft models we pass one extra token per sequence." → `AssertionError: Shape: 2065 out of considered ranges: [(1, 2048)]` | READ BODY |

### D.3 What remains viable on this exact hardware

Not advice — reported scope of what the sources above leave unblocked. Every item below still carries at least one of the constraints in B.1/B.2:

- **EAGLE / EAGLE-3 / MTP on a dense or small-MoE target at single-GPU**, with `--enforce-eager` or piecewise graphs (no full-CG unless on MRv2 and without a DSD table), **no DP/PP**, `--no-enable-prefix-caching` or accepting the documented last-block drop, and **no fp8/fp4 KV** if the drafter is DFlash-family.
- **vLLM's own DSD reference benchmark was run at BS1 only with `--no-enable-prefix-caching`** (PR #45953), so the DSD path on this hardware is essentially uncharacterized at concurrency.
- Note SGLang PR #19897's radix-cache + EAGLE NaN bug was **originally reported on 8× RTX PRO 6000 Blackwell (SM120)** — i.e. this hardware class is where that defect was found, though the author states it is architecture-independent.

---

## Search log

### web_search queries (verbatim, in order)

1. `Speculative Decoding Performance or Illusion MLSys 2026 Xiaoxuan Liu`
2. `speculative decoding batch size acceptance collapse high batch`
3. `vLLM PR 54748 Count scheduler steps K dynamic SD selected`
4. `arXiv "Speculative Decoding: Performance or Illusion"`
5. `"Speculative Decoding: Performance or Illusion" batch size conclusion`
6. `vLLM issue speculative decoding prefix caching incompatible not supported`
7. `vLLM "Speculative Decoding not supported with prefix caching" mamba_attn.py`
8. `SGLang speculative decoding docs supported methods EAGLE draft model`
9. `"speculative decoding" "chunked prefill" incompatible vLLM issue`
10. `"speculative decoding" tensor parallelism worth it communication overhead negative result`
11. `"speculative decoding" "not planned" OR "wontfix" vllm combination unsupported`
12. `"Speculative decoding" "structured output" grammar xgrammar incompatible issue`
13. `vLLM EAGLE spec decode prefix caching radix cache issue acceptance`
14. `SGLang RadixAttention speculative decoding prefix cache incompatible`
15. `speculative decoding "future work" batch size limitation paper 2026`
16. `RTX PRO 6000 Blackwell sm120 FP4 NVFP4 kernel not supported datacenter sm100`
17. `vLLM sm120 Blackwell consumer GPU supported quantization FP8 Marlin`
18. `vLLM tensor parallel speculative decoding draft_tensor_parallel_size multi-GPU`
19. `vLLM "Speculative decoding" "tensor_parallel_size" draft model TP support`
20. `speculative decoding acceptance rate degrades with tensor parallelism communication`
21. `vLLM docs attention backend feature support speculative decoding`
22. `vLLM structured output speculative decoding incompatible grammar bitmask rejection sampling`
23. `SGLang grammar constrained decoding speculative decoding incompatible issue`
24. `"speculative decoding" "not planned" github issue maintainer closed wontfix`
25. `SGLang "jump forward decoding" speculative disabled discussion 32352`
26. `speculative decoding quantized draft model EAGLE acceptance degradation 2026 paper`
27. `"ML-SpecQD" quantified draft speculative decoding`
28. `speculative decoding continuous batching high batch size speculation not worth it paper 2026`
29. `speculative decoding disaggregated serving P/D accept length collapses batch`
30. `speculative decoding expert parallelism MoE draft EP communication 2026`
31. `"speculative decoding" continuous batching batch size 2026 arXiv throughput analysis`
32. `AgentSpec speculative decoding batch inference LLM agents Microsoft`
33. `vLLM spec decode + FP8 KV cache crash issue 44184 48406`
34. `vLLM pull request spec decode support closed unmerged "not planned" 2026`
35. `speculative decoding "we decided not to" OR "closing this" maintainer vllm sglang`
36. `vLLM "lookahead scheduling" spec decode batch expansion explanation`
37. `vLLM spec decode not supported data parallel DP attention EAGLE limitation docs`
38. `"speculative decoding" "expert parallelism" draft model EP vLLM SGLang issue`
39. `speculative decoding paged attention backend incompatibility TRITON_ATTN FLASH_ATTN`
40. `speculative decoding CUDA graph capture fails dynamic num_speculative_tokens vLLM issue`
41. `vLLM spec decode MRV2 not supported speculative decoding methods fall back MRV1`
42. `"lookahead decoding" batch size scaling negative result`
43. `speculative decoding ML-SpecQD MXFP4 draft plug-and-play limitations batch size`
44. `speculative decoding "prefix caching" vLLM issue EAGLE lookahead block drop`
45. `SGLang 0.5.19 speculative decoding known issues beam search not mix`
46. `vLLM issue 38182 MTP prefix cache hit rate lower expected behaviour maintainer`
47. `vLLM 50188 prefix caching MTP spec decode corrupts tool call`
48. `EAGLE spec decode prefix caching "expected behaviour" not a bug vllm`
49. `vLLM 55760 55861 dense retention hybrid models EAGLE MTP prefix cache`
50. `speculative decoding prefix cache retention interval hybrid MTP vLLM`

### arXiv listing / abs fetches (verbatim IDs)

`2601.11580` (via arxiv.org/abs and ar5iv.labs.arxiv.org/html), `2506.11309`, `2503.13565`, `2505.22179`, `2605.09329`, `2603.18016`, `2512.22420`, `2608.24004`, `2604.26412`, `2603.01399`, `2607.04244`, `2607.17283`, `2502.10424`. (Also probed `2505.22148` — that is an unrelated paper; the correct ML-SpecQD ID is `2503.13565`.)

### GitHub HTML fetches (verbatim URLs)

**vLLM PRs:** /pull/54748, /pull/54749 (actually an issue), /pull/45953, /pull/26231, /pull/53388, /pull/54163, /pull/50897, /pull/51769, /pull/28435, /pull/24505, /pull/27434, /pull/55262, /pull/56577, /pull/56539, /pull/53979, /pull/54731, /pull/43911, /pull/44086
**vLLM issues:** /issues/49548, /issues/54749, /issues/48494, /issues/54690, /issues/54691, /issues/54801, /issues/47277, /issues/39809, /issues/43996, /issues/44697, /issues/33416, /issues/32591, /issues/49562, /issues/38182, /issues/50188, /issues/39790, /issues/50438, /issues/47930, /issues/41559, /issues/5543, /issues/48406, /issues/44506
**vLLM releases/docs:** /releases/tag/v0.29.0, /releases, docs.vllm.ai/en/latest/features/speculative_decoding/, docs.vllm.ai/en/v0.29.0/features/speculative_decoding/, docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/, docs.vllm.ai/en/latest/features/spec_decode/ (redirects), discuss.vllm.ai/t/standalone-draft-model-spec-decode-support-in-v0-x-and-v1/2241
**SGLang:** /pull/4032, /pull/12615, /pull/19897, /pull/20168, /pull/20148 (issue), /pull/17418, /pull/23407, /pull/30096, /pull/32440, /pull/36933, /pull/38870, /pull/39064, /pull/39254, /issues/32459, /issues/19796, /issues/7694, /issues/36599, /issues/39087, /discussions/32352, /releases, /releases/tag/v0.5.19, /tags, docs.sglang.io/docs/advanced_features/speculative_decoding
**TensorRT-LLM:** /issues/16767, /pull/14369, /pull/17408
**raw.githubusercontent.com:** vllm/v1/attention/backends/mamba_attn.py, vllm/config/vllm.py, vllm/config/speculative.py, vllm/v1/core/sched/scheduler.py, vllm/v1/spec_decode/dynamic/utils.py, vllm/platforms/cuda.py

### Other
- `mlsys.org/virtual/2026/oral/3782` (READ BODY)
- `mlsys.org/media/mlsys-2026/Slides/3782.pdf` (downloaded 1.9 MB; TITLE ONLY — PDF uses subset-encoded fonts with no `ToUnicode` CMap, so text extraction yields glyph garbage; findings sourced from the peer-reviewed HTML instead)
- `github.com/noonghunna/club-3090` (TITLE ONLY; referenced for NVFP4-KV sm100/103 gating on consumer Blackwell)

### Failures / limitations encountered
- `api.github.com` avoided per instructions (rate limit); all GitHub access via HTML or raw.githubusercontent.com.
- Transient connection failures (`HTTP:000`) on some github.com fetches during the sweep; retried with a 4-attempt backoff helper, all target pages eventually retrieved.
- `pdftotext` unavailable in the environment; the MLSys slide PDF was attempted with a pure-Python zlib/stream extractor, which failed due to subset font encoding (no `ToUnicode` CMap). Slides are therefore **TITLE ONLY**; all paper claims come from the arXiv HTML/abs pages.
- `export.arxiv.org` API returned HTTP 429 ("Rate exceeded") on all queries; arXiv discovery was done through `arxiv.org/abs` and the ar5iv HTML endpoint instead.
- Negative result worth recording: searches for literal `is:issue is:closed reason:"not planned"` and `label:wontfix` on spec-decode topics returned **zero** hits in both vLLM and SGLang. The only `not planned` closure found is issue #39790 (via its page banner, not a maintainer comment). **No maintainer statement using the words "not planned" or "wontfix" about a spec-decode combination was verified.**

---

## E. Verbatim closing reasons for not-planned / stale-closed items

**Method note (important):** GitHub's issue HTML pages do **not** include comment bodies, and `api.github.com` returned `API rate limit exceeded for 202.120.234.156` on every attempt during this sweep. I therefore could not retrieve a maintainer *comment* for any of these. What I **can** verify is (a) the state banner text and (b) the visible timeline/comments state. **No closing comment exists to quote — the closure is a bare state change with no accompanying words.** I am reporting that fact rather than paraphrasing, per the no-invention rule.

| # | Issue | URL | Closing state (verbatim banner) | Closing comment / event text | Verified how |
|---|---|---|---|---|---|
| E1 | vLLM #53215 — "MTP acceptance metrics fluctuate within a benchmark run for GLM-5.2-FP8 on 8×H200 with vLLM v0.24.0" | https://github.com/vllm-project/vllm/issues/53215 | **"Closed as not planned"** | **NONE.** Zero comments. Page shows only the boilerplate "Reactions are currently unavailable … Sign in to comment" and Metadata "Assignees: No one assigned / Labels: glm, performance, quantization". No maintainer, bot, or contributor prose. | READ BODY (HTML) |
| E2 | vLLM #46088 — "MTP speculative decoding with `--kv-cache-dtype auto` produces cross-sequence garbage under batching (Gemma-4 W4A16; fp8 KV unaffected)" | https://github.com/vllm-project/vllm/issues/46088 | **"Closed as not planned"** | **NONE.** Zero comments. Metadata block present; no Labels shown; no timeline prose. | READ BODY (HTML) |
| E3 | vLLM #39790 — "Significant TTFT Regression with Speculative Decoding (EAGLE3)" | https://github.com/vllm-project/vllm/issues/39790 | **"Closed as not planned"** + labels `performance`, `stale` ("Over 90 days of inactivity") | **NONE.** Zero comments. The `stale` label is the only signal of *why*; it is a label, not prose. Author's un-rebutted numbers: "Mean TPOT: ~17.11 ms (a 39.3% improvement, which is great) / Mean TTFT: ~158.44 ms (a 114% increase) / P99 TTFT: ~1078.03 ms (a 331% increase)." | READ BODY (HTML) |
| E4 | SGLang #36001 — "Speculative decoding (DFLASH) crashes with `--kv-cache-dtype nvfp4`: `extend_prefix_lens_cpu` is None in dequant workspace" | https://github.com/sgl-project/sglang/issues/36001 | **"Closed as not planned"** | **NONE.** Zero comments. Metadata: "Assignees: No one assigned / Labels: No labels / Milestone: No milestone / Development: No branches or pull requests". No maintainer prose. | READ BODY (HTML) |

**Why E4 matters most for this researcher — it is the exact hardware + model combination.** Verbatim from the issue's Environment block:
> "Image: `lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2` (commit `5f55db3`)"
> "GPU: **RTX PRO 6000 Blackwell Workstation Edition, 1x, 96GB**"
> "FlashInfer 0.6.17"
> "Target checkpoint: `Qwen/Qwen3.8-27B-FP8`"
> "Draft checkpoint: `incoai/Qwen3.8-27B-DFlash2`"

And the verbatim conclusion that **no working configuration exists** on this hardware:
> "So there's currently no working `--speculative-draft-attention-backend` value for NVFP4 KV cache at all."
> "`trtllm_mha` is decode-only for native FP4 KV (`trtllm_mha_backend.py`: "TRTLLM MHA with native FP4 KV cache supports decode only; use a separate prefill backend such as flashinfer or triton."), and flashinfer is the only one that supports NVFP4 prefill/extend — which is exactly the path that hits the bug above."
> "Dropping only `--kv-cache-dtype nvfp4` (everything else identical) boots and serves correctly, confirming the NVFP4 dequant path is the trigger."
> "any speculative-decoding draft-extend/verify step that ends up on the FlashInfer prefill/extend path with an NVFP4 KV cache will hit this deterministically."

### E.5 SGLang 0.5.19 release notes (requested in full)

- **URL:** https://github.com/sgl-project/sglang/releases/tag/v0.5.19
- **Tag:** `v0.5.19`, commit `0bcd822`, signed (GPG key ID `B5690EEEBB952194`)
- **Released:** 05 Sep 02:27, by **Qiaolin-Yu**. "Latest" release. 629 commits to main since; **786 PRs from 214 contributors.**

**Verbatim spec-decode-relevant bullets:**

Highlights section:
> "**Beam search.** SGLang can now do beam search. Pass `beam_width` in your request and you get back the n best sequences instead of a single sample. It works out of the box next to regular requests, though it does not yet mix with speculative decoding, disaggregation, DP attention, or HiCache (#31626)."
> "**Faster speculative kernels.** DSA prefill top-k moves to the v2 kernel, 1.3 to 1.8 times faster on B200 (#35175). KDA models get an opt-in fused accept path, `SGLANG_OPT_KDA_FUSED_ACCEPT_STATE=1`, that cuts MTP verify-and-commit time by 45% to 63% on Kimi-Linear shapes with bit-identical output (#33722)."
> "**Unified radix tree by default.** The unified tree is now the cache for every model, not just hybrid ones (#35081, see Breaking Changes). It also picked up three things this cycle: PD decode workers can reuse cached prefixes for SWA hybrid models like gpt-oss (#27770), you can attach or detach L3 storage on a running server (#35269), and pipeline parallelism with HiCache L3 stays consistent across ranks (#27010)."

Dedicated **"Speculative Decoding"** section, verbatim in listed order:
> "[KDA] Fused-accept state advance for FlashInfer KDA MTP verify: #33722 ⭐"
> "[Spec] DFlash2: local convolution + candidate selector (**3.43x over no-spec at batch 1 and about 24% over DFlash at concurrency 64**): #35371"
> "[Spec] Support quantized target lm_head in the DFlash2 selector: #35496"
> "[Spec] Add LFM2 and LFM2-MoE DSpark speculative decoding support (1.05 to 2.42x faster decoding on LFM2.5 targets, 1xH100): #31041"
> "[Model] Support Nemotron 3.5 Lightning speculative decoding (GSM8K 94.6% to 95.8% across MTP, DFlash, and DSpark on GB300): #36186"
> "[Spec][LoRA] Support multi-adapter LoRA with EAGLE/NEXTN/DFLASH/DSPARK speculative decoding: #34337"
> "[Spec][DSA] Add `--speculative-dsa-topk-backend`: #36313"
> "**[2/N][Mixed] Mixed chunk prefill with spec enabled: #36933**"
> "Support custom draft worker classes in DSpark: #35397"
> "Make draft attention backends extensible: #35932"
> "[Memory] Borrow CUDA graph pool storage for EAGLE sampling: #35375"
> "[Spec] Fix Dspark and Dflash state divergence across TP rank: #33614"
> "[Fix][Spec] fix startup crash and reduce CUDA graph memory usage for speculative adaptive: #35275"
> "Fix DSV4 DSpark sample-from-anchor initialization: #36419"
> "[Fix] Drop the duplicated DSpark draft `sample_block` call: #36934"
> "**fix(gemma4): quantize MTP bridge projections (Gemma-4 FP8 MTP acceptance from 0% to 60%): #32440**"

CUDA-graph bullet touching spec decode:
> "[Fix] Fix full prefill CUDA graph padding and EAGLE capture: #35588"

Note the tension between the docs and the release notes: the docs page still states DFLASH and NGRAM "disable … mixed chunked prefill", while 0.5.19 lists "[2/N][Mixed] Mixed chunk prefill with spec enabled: #36933". The docs statement is scoped to those two methods specifically; #36933 generalizes mixed chunk prefill for the spec-enabled path.

---

## Provenance notes / corrections to the brief

1. **Correction — vLLM PR #54748.** The brief's title *"[Spec Decode] Count scheduler steps by the K dynamic SD selected"* does not match the artifact. PR #54748 is titled **"[Spec Decode] Make the dynamic SD schedule observable"** and is **OPEN, unmerged**. The counter it adds is `vllm:spec_decode_scheduled_steps`, which does count scheduler steps by selected K (including K=0) — so the brief's description of the *content* is accurate; only the title and merge status differ. Companion **#54749 is an ISSUE (RFC), not a PR**.
2. **"Speculative Decoding: Performance or Illusion?" is arXiv 2601.11580**, an MLSys 2026 **oral** (Thu 21 May 2026, 9:00–9:15 AM PDT), by Xiaoxuan Liu, Jiaxiang Yu, Jongseok Park, Ion Stoica, Alvin Cheung (UC Berkeley). Note the arXiv listing is dated 2025.12 in third-party notes but the MLSys venue is 2026. OpenReview and slides are linked from the MLSys oral page; the slide PDF's text is not machine-extractable.
3. **ML-SpecQD is arXiv 2503.13565** (Georganas, Kalamkar, Kozlov, Heinecke, Intel). Its arXiv page lists **no journal reference and no comments field** — venue is **NOT VERIFIED**; treat as arXiv-only.
4. **SGLang 0.5.19 is the latest SGLang release** (05 Sep 2026); SGLang tags stop at v0.5.19. **vLLM 0.29.0 is the latest vLLM release** (09 Sep 2026); no 0.30 exists.
5. **The vLLM docs URL in the brief redirects.** `https://docs.vllm.ai/en/latest/features/spec_decode/` 302s to `.../features/speculative_decoding/`. The versioned `https://docs.vllm.ai/en/v0.29.0/` index exists; the correct page is `https://docs.vllm.ai/en/v0.29.0/features/speculative_decoding/`. The versioned page content is identical to `latest` for the incompatibility section.

---

# APPENDIX A — Structured output × spec decode (grammar/xgrammar)

**⚠️ Provenance caveat for this appendix:** retrieved by a sub-sweep operating under a sandbox where github.com HTML was network-blocked and `api.github.com` core quota was exhausted (0/60). Issue/PR **bodies** were obtained via the GitHub **search** API (which returns body text); **comment threads could not be retrieved**, so **no maintainer comment is quoted below** and closure reasons marked NOT VERIFIED are exactly that. Source-file and docs quotes are the strongest evidence here. Items I verified myself are marked ✓.

## A-a. The core result: rejection sampling is **not** lossless under grammar masks

| # | Question | Artifact | URL | Status | Read |
|---|---|---|---|---|---|
| G1 | Is spec decode lossless under a grammar mask? **No.** | arXiv 2605.07698, "Future Validity is the Missing Statistic: From Impossibility to Φ-Estimation for Grammar-Faithful Speculative Decoding" (Nie et al., 8 May 2026) | https://arxiv.org/abs/2605.07698 | Published | READ BODY |

Verbatim:
> "any speculative decoder with local mask access, Leviathan rejection, and rollback soundness samples from the locally projected distribution mu^proj rather than the grammar-conditional distribution mu^star. This extends the GAD impossibility result to speculative decoding; on Dyck grammars with Qwen3-8B, the total-variation gap can reach 0.996."

Scope caveat, verbatim:
> "All fidelity claims are scoped to enumerable grammars and token tries."

This directly contradicts the impression given by vLLM's own "Lossless guarantees of Speculative Decoding" section (read directly at https://docs.vllm.ai/en/v0.29.0/features/speculative_decoding/), which states the implementation is "algorithmically validated to be lossless". That claim concerns the **unconstrained** rejection sampler; it does not extend to the grammar-conditional distribution.

## A-b. vLLM: documented incompatible → fixed → still leaking

| # | Question | Artifact | URL | Status | Read |
|---|---|---|---|---|---|
| G2 | Were they documented as incompatible? | PR #12373 | https://github.com/vllm-project/vllm/pull/12373 | Merged 2025-01-24 | READ BODY |
| G3 | How were they made to work? | PR #14702 | https://github.com/vllm-project/vllm/pull/14702 | Merged 2025-04-30 | READ BODY |
| G4 | A backend that **still refuses** spec tokens | `backend_lm_format_enforcer.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/structured_output/backend_lm_format_enforcer.py | In main | READ BODY |
| G5 | Mask **fails open** at real sampling positions | Issue #54437 | https://github.com/vllm-project/vllm/issues/54437 | **OPEN** | READ BODY |
| G6 | Grammar-aware draft sampling | PR #47885 | https://github.com/vllm-project/vllm/pull/47885 | OPEN | READ BODY |
| G7 | ~20 FSM/bitmask bugs; several still open | #27210, #27969, #43388, #44006 (closed-completed); #44297, #44993, #53046, #52436, #29821, #32102, #33374 (merged); **#44292, #48516, #44927, #49738 (closed-unmerged)**; #49694, #52620, #49210, #47025, #40875, #52452, #52477, #43424, #40962 (OPEN) | `https://github.com/vllm-project/vllm/issues/<n>` | mixed | TITLE ONLY (index) |

Verbatim, G2: *"Update the feature compatibility matrix to reflect that speculative decoding and structured output do not currently work together."*

Verbatim, G3: *"The generated speculative draft tokens are validated according to the grammar to ensure they match the grammar constraints. This operation should not modify the matcher state."*

Verbatim, G4 (a live hard incompatibility in source): **"LM Format Enforcer backend does not support speculative tokens"**

Verbatim, G5 (the fail-open defect): *"which then writes `_full_mask` — every token in the vocabulary allowed — at rows that are real sampling positions."* and *"The failure is silent by construction: the mask fails open."*

Verbatim, G6: *"These tokens are guaranteed to be rejected by the verifier regardless of probability alignment, wasting entire speculative rounds."*

## A-c. SGLang: grammar × spec decode

| # | Question | Artifact | URL | Status | Read |
|---|---|---|---|---|---|
| G8 | DFLASH rejected all grammar requests (HTTP 400) | PR #30096 ✓ | https://github.com/sgl-project/sglang/pull/30096 | Merged 4 Jul 2026 | READ BODY |
| G9 | Outlines backend incompatible with **any** speculative algorithm | PR #24499 | https://github.com/sgl-project/sglang/pull/24499 | **closed-unmerged** | READ BODY |
| G10 | The alternative attempt, also abandoned | PR #24889 | https://github.com/sgl-project/sglang/pull/24889 | **closed-unmerged** (2026-09-13) | READ BODY |
| G11 | xgrammar rollback is O(N^2) under spec decode | Issue #38863 | https://github.com/sgl-project/sglang/issues/38863 | OPEN | TITLE ONLY |

Verbatim, G8 (engine's own error string ✓): *"DFLASH speculative decoding does not support grammar-constrained decoding yet."*

Verbatim, G9: *"OutlinesGrammar does not implement rollback(), which the speculative DFS verifier requires"*; *"Outlines uses a dense torch.bool mask, but spec_utils.traverse_tree() expects packed int32 bitmasks"*

## A-d. Chunked prefill × spec decode — SGLang hard-disables it

| # | Question | Artifact | URL | Status | Read |
|---|---|---|---|---|---|
| G12 | SGLang disables mixed chunked prefill for several algorithms | main source warning strings | `python/sglang/srt/...` (via sub-sweep) | In main | READ BODY |
| G13 | Same in released versions | v0.4.9 → v0.5.7 strings | — | Shipped | READ BODY |
| G14 | vLLM took the opposite path | PR #9291 / issue #5016 | https://github.com/vllm-project/vllm/pull/9291 | Merged 2024-11-07 | TITLE ONLY |
| G15 | MTP + chunked prefill corrupts prompt_logprobs | Issue #53488 | https://github.com/vllm-project/vllm/issues/53488 | OPEN | READ BODY |
| G16 | Mixed spec/plain decode batches read stale recurrent state | Issue #56531 | https://github.com/vllm-project/vllm/issues/56531 | OPEN | READ BODY |

Verbatim, G12: *"Mixed chunked prefill is disabled: %s speculative decoding does not support it."* / *"Mixed chunked prefill is disabled because of using Frozen-KV MTP speculative decoding."* / *"Mixed chunked prefill is disabled for UNO speculative decoding."* / `supports_mixed_chunk()` docstring: *"ngram cannot join as is: its overlap relay skips output_tokens_buf, which the mixed input resolve reads."*

Verbatim, G13: **"Pipeline parallelism is not compatible with overlap schedule, speculative decoding, mixed chunked prefill."**

Verbatim, G15: `prompt_logprobs` is *"wildly wrong for a subset of requests"* with MTP + chunked prefill.

Verbatim, G16: *"generation continues from a state that silently lost a few tokens of history"*

## A-e. Paged-attention backend incompatibilities

| # | Question | Artifact | URL | Status | Read |
|---|---|---|---|---|---|
| G17 | FlashInfer + spec decode silently downgrades CUDA-graph mode and costs 16% | Issue #49547 | https://github.com/vllm-project/vllm/issues/49547 | **OPEN** | READ BODY |
| G18 | Cutlass-MLA produces **incorrect output** with spec decode | PR #25984 | https://github.com/vllm-project/vllm/pull/25984 | Merged (known issue) | READ BODY |
| G19 | AITER MLA can't be used with Eagle3 | PR #39616 | https://github.com/vllm-project/vllm/pull/39616 | Merged | READ BODY |
| G20 | MLA is decode-only for full CUDA-graph capture | Issue #21505 | https://github.com/vllm-project/vllm/issues/21505 | READ BODY |
| G21 | SGLang: tree spec + page_size>1 restricted to 3 backends | source | SGLang main | In main | READ BODY |
| G22 | Flex Attention refuses spec decode | source | SGLang main | In main | READ BODY |
| G23 | Open backend/spec-decode bugs | #55581, #47899, #47847, #48495, #52499, #50885, #47979 | `https://github.com/vllm-project/vllm/issues/<n>` | OPEN | TITLE ONLY (index) |

Verbatim, G17 (measured): FlashInfer + spec-decode *"silently downgrades cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE"*; *"measured 47.5 → 55.2 tok/s (+16%) switching only `--attention-backend FLASHINFER` → `FLASH_ATTN`"*

Verbatim, G18: *"The Cutlass-MLA backend produces incorrect output when using speculative decoding."*

Verbatim, G19: AITER MLA *"currently cannot be used with speculative decoding methods like Eagle3 due to a kernel_block_size conflict"*

Verbatim, G20: *"MLA only supports decode-only full CUDAGraph capture."*

Verbatim, G21: *"topk > 1 + page_size > 1 needs the two-pass cascade draft-decode ... flashmla / trtllm_mla can't express the per-branch tree, so reject."* and *"speculative_eagle_topk > 1 with page_size > 1 is only supported on ('flashinfer', 'fa3', 'triton')"*

Verbatim, G22: *"Speculative decoding is currently not supported with Flex Attention backend"* (MiniCPM / Mamba-1 / hpc_ops / MiniMax-M3 backends also refuse spec decode)

Verbatim, from arXiv 2607.20475 (SonicSampler): existing samplers *"assume homogeneous sampling behavior across a batch, limiting support for dynamic serving workloads and preventing efficient CUDA Graph execution."*

## A-f. More TP/EP detail

| # | Question | Artifact | URL | Status | Read |
|---|---|---|---|---|---|
| G24 | Elastic EP refuses draft models | `vllm/v1/worker/gpu/eplb_utils.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu/eplb_utils.py | In main | READ BODY |
| G25 | TP comm cost of greedy drafting | `use_local_argmax_reduction` | vLLM source | Shipped | READ BODY |
| G26 | SGLang source-level dp-attention refusals | `arg_groups/speculative_hook.py` | SGLang main | In main | READ BODY |
| G27 | spec × EP × DP-attention only validated at 8-GPU EP8×DP2 | SGLang Kimi-K3 card | SGLang docs | Shipped | READ BODY |
| G28 | DSpark incompatible with DP Attention | SGLang DeepSeek-V4 card | SGLang docs | Shipped | READ BODY |
| G29 | EAGLE3 TP=2 + draft TP=2 NCCL **deadlock** | Issue #44063 | https://github.com/vllm-project/vllm/issues/44063 | **OPEN** | READ BODY |
| G30 | DP>1 + EP + DSpark AssertionError | Issue #56281 | https://github.com/vllm-project/vllm/issues/56281 | OPEN | READ BODY |
| G31 | Draft/target backends can have **disjoint KV cache layouts** | Issue #55312 | https://github.com/vllm-project/vllm/issues/55312 | OPEN | READ BODY |
| G32 | SGLang PRs adding DP-attention for spec decode (all open) | #35898, #35321, #29506, #28616, #30916, #30642 | `https://github.com/sgl-project/sglang/pull/<n>` | OPEN | TITLE ONLY (index) |

Verbatim, G24: **"Elastic EP is not supported with draft model."**

Verbatim, G25: *"Reduces communication from O(vocab_size) to O(2 * tp_size) per token."*

Verbatim, G26: *"Currently DFLASH speculative decoding does not support dp attention on non-NPU devices."* / *"Currently standalone speculative decoding does not support dp attention."* / *"Currently ngram speculative decoding does not support dp attention."*

Verbatim, G27: *"spec × EP × DP-attention is validated only at 8-GPU EP8 × DP2 (full GSM8K) — experimental at these scales."*

Verbatim, G28: *"they use DP Attention, which DSpark is incompatible with on current releases."*

## A-g. CUDA graphs are explicitly keyed on K

| # | Question | Artifact | URL | Status | Read |
|---|---|---|---|---|---|
| G33 | Graphs keyed on K; backend support table | `docs/design/cuda_graphs.md` | https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md | In main | READ BODY |
| G34 | Capture sizes rounded to multiples of K+1 | `vllm/config/compilation.py` | raw.githubusercontent.com | In main | READ BODY |
| G35 | Eagle is piecewise-only, called a hack in-source | `vllm/v1/worker/gpu_model_runner.py` | raw.githubusercontent.com | In main | READ BODY |
| G36 | Closed-unmerged graph work | #34880, #26937, #47513, #7090 | `https://github.com/vllm-project/vllm/pull/<n>` | closed-unmerged | TITLE ONLY (index) |
| G37 | Not-planned graph RFCs (reasons NOT VERIFIED) | #49488, #27325 | `https://github.com/vllm-project/vllm/issues/<n>` | CLOSED-NOT-PLANNED | TITLE ONLY (index) |

Verbatim, G33: *"speculative decode (max_query_len =1+num_spec_tokens) as uniform decode batches"*; `AttentionCGSupport.UNIFORM_BATCH` docstring *"this can be used for spec-decode i.e. 'decodes' are 1 + num_speculative_tokens"*; *"only FlashAttention 3 supports it currently) or only support CUDA Graphs for pure decode batches (e.g., Flashinfer, FlashMLA, and Mamba, etc.)"*; backend table assigns FlashInfer / AITER MLA / CUTLASS MLA / Mamba to `UNIFORM_SINGLE_TOKEN_DECODE`, and **"Unlisted backends are all declared as NEVER."**

Verbatim, G34: `adjust_cudagraph_sizes_for_spec_decode` rounds every capture size to a multiple of *"num_speculative_tokens + 1"*; guard: *"CUDAGraphMode.{name} is not supported with spec-decode for attention backend {min_cg_attn_backend}"* → falls back to PIECEWISE or NONE.

Verbatim, G35: **"Eagle currently only supports PIECEWISE cudagraphs. ... NOTE(lucas): this is a hack, need to clean up."**

Verbatim, G37 (#27325): *"the adaptation of cuda_graph_sizes causes the decoding process to fall back to eager"*

## A-h. Additional prefix-cache quantification

| # | Question | Artifact | URL | Status | Read |
|---|---|---|---|---|---|
| G38 | Measured cost of the EAGLE last-block drop | Issue #53670 | https://github.com/vllm-project/vllm/issues/53670 | **OPEN** | READ BODY |
| G39 | MTP scoring **zero** prefix-cache hits | PR #55519 | https://github.com/vllm-project/vllm/pull/55519 | OPEN | READ BODY |
| G40 | Upstream authors on the unsolved general case | Issue #51771 (quoting PR #33524) | https://github.com/vllm-project/vllm/issues/51771 | OPEN | READ BODY |
| G41 | Spec decode disabled in a PR due to prefix-caching bugs | Issue #41884 (quoting PR #30877, #33726) | https://github.com/vllm-project/vllm/issues/41884 | **CLOSED-NOT-PLANNED** | READ BODY |
| G42 | Hybrid KV cache unsupported for eagle + chunked local attention | vLLM source | raw.githubusercontent.com | In main | READ BODY |
| G43 | SGLang PD-disagg radix cache incompatible with spec decode | SGLang source | SGLang main | In main | READ BODY |

Verbatim, G38: *"EAGLE/MTP prefix-cache last-block drop causes a 1,648-token recompute per hit on one hybrid Qwen3.8 GDN layout — ~30-40% batch throughput loss on prefix-reusing workloads with speculative decoding enabled"*

Verbatim, G39: with MTP *"the run scores 0 prefix-cache hits"*; *"Adding `\"disable_eagle_block_drop\": true` restores 22,176 hits out of 72,992 queried"*

Verbatim, G40: *"for more complicated models with multiple attention groups, this PR does not fully address the EAGLE spiral block drop issue either. A general fix ... cannot directly cache the hit_blocks list returned by each attention type, because SWA attn and Mamba-style attn do not follow the downward-closed property"*; *"Fortunately, we don't have such complex models yet, so this is not a huge issue for now."*

Verbatim, G41 (quoting PR #30877): *"Speculative decoding is temporarily disabled in this PR as there are still corner-case bugs when using with prefix-caching in align mode."* and (PR #33726): *"Prefix Caching (old style, have not tested 'align' mode)"*

Verbatim, G42: *"Hybrid KV cache is not supported for eagle + chunked local attention."* ; MTP configs in `qwen3_next_mtp.py` / `qwen3_5_mtp.py` / `qwen4_exp`: *"currently does not support 'all' prefix caching"*

Verbatim, G43: *"Incompatible with --enable-hisparse, speculative decoding, and --disaggregation-transfer-backend fake."* ; *"HiCache does not support Inkling MTP draft state yet."*

I-1. Verbatim vLLM source confirming the structural conflict (`single_type_kv_cache_manager.py`):
> "If eagle is enabled, drop the last matched block to force recompute the last block to get the required hidden states for eagle drafting head."
> "Eagle needs the tokens right before the generation point recomputed: drop one hash unit when fine-grained ..., else one cache block."

This independently corroborates PR #53388 and PR #51769 (both verified directly in Section B/C).

## A-i. 🔴 The documentation gap (material to the "documented incompatibility matrices" question)

Two grep-verified negative documentation facts:

1. **vLLM's "Known Feature Incompatibility" section contains EXACTLY two items** and mentions **none** of structured output, TP/EP, paged attention, CUDA graphs, chunked prefill, or prefix caching. Verbatim from `docs/features/speculative_decoding/README.md`:
   > "## Known Feature Incompatibility
   > 1. Pipeline parallelism is not composable with speculative decoding as of `vllm<=0.15.0`
   > 2. Speculative decoding with draft models is not supported in `vllm<=0.10.0`"
   (Matches my own independent read of the hosted docs at v0.29.0 and latest.)

2. **The vLLM APC design doc (`docs/design/prefix_caching.md`) contains ZERO occurrences of "EAGLE" / "speculative" / "spec"** — verified by grep. The spec-decode × prefix-caching conflict is documented only in source code and the *user-facing* APC doc, never in the design doc.

**Consequence:** every incompatibility catalogued in this appendix is absent from the official matrices. They are discoverable only by reading source or issue trackers.

## A-j. NOT VERIFIED in this appendix
- No maintainer **comment** text anywhere (GitHub rate limit + network block). Closure reasons for not-planned #22230, #49488, #27325, #41884, #20531, #7569 → **NOT VERIFIED**.
- TechRxiv "Transactional KV Caching for Speculative Decoding under Paged KV Memory" (DOI 10.36227/techrxiv.177101038.80960856) — **TITLE ONLY**; HTTP 403 Cloudflare on both `curl` and `web_fetch`.
- arXiv API returned HTTP 429 for the whole session in that sub-sweep.
- TensorRT-LLM #17095, #16767, #16448, #16377 — TITLE ONLY in that sub-sweep. (I verified #16767 myself in full; its accept-length-collapse table stands in Section B.)
