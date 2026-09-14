# Negative-evidence web search: speculative-decoding engine gaps (outside GitHub issues/PRs)

Date context: September 2026. Searcher: delegated subagent (general-web lane; GitHub issues/PRs and arXiv excluded as another agent's lane).

## Method / surfaces actually searched

- `web_search` (42 queries, listed verbatim per item below).
- HackerNews via Algolia API (`hn.algolia.com/api/v1/search`, tags=story,comment): 22 queries.
- Reddit: `search.json` and `old.reddit.com/search.json` are **blocked** (return an HTML block page). `www.reddit.com/r/<sub>/search.rss?q=...&restrict_sr=1` **works** — used for r/LocalLLaMA and r/MachineLearning, 16 queries each. Per-thread `.rss` fetches were **HTTP 429 (rate-limited)** for the 3 threads I tried.
- Docs sites: docs.vllm.ai, docs.sglang.io, vllm.ai/blog, lmsys.org/blog, z-lab.ai, inco.ai/blog, HuggingFace model cards, fergusfinn.com.
- Developer forums: forums.developer.nvidia.com (Discourse JSON API — the HTML pages defeat naive text extraction; use `https://forums.developer.nvidia.com/t/<id>.json`).
- **ErrLookup / errors.standardbeagle.com** — a third-party public error-message knowledge base (`datasetVersion 2026-09-13`, "403,603 errors across 1,821 repos"), with per-repo datasets at `https://errors.standardbeagle.com/data/repos/<owner>/<repo>.json`. **Caveat: it is auto-extracted from source code, not human discussion.** It is a public docs surface outside GitHub issues/PRs, but it mirrors source, so treat it as a source mirror, not as "someone discussed this".
- **Blocked / inconclusive surfaces:** Zhihu (HTTP 403 on article body); Reddit per-thread comments (HTTP 429). Reddit post-level covered via RSS.

Not-found results are stated as "I searched X, Y, Z and found no matching discussion" — never "nobody has discussed this".

## Summary table

| # | Item | Verdict |
|---|---|---|
| 1 | Sampling-mask incompatible with spec decode | **FOUND** (ErrLookup mirror) |
| 2 | Folding sampling params / rejection into verify kernel, kill FP32 logits buffer | NOT-FOUND |
| 3 | Draft-extend CUDA graph limited to subset of attention backends | NOT-FOUND |
| 4 | n-gram capped to 1 CPU thread (TP matching unimplemented) | NOT-FOUND |
| 5 | Adaptive spec length unsupported w/ DP attention, TBO, PDMux, multi-layer EAGLE, frozen-KV MTP | **FOUND** (partially: DP attention, TBO, PDMux, topk>1 confirmed; multi-layer EAGLE + frozen-KV MTP NOT-FOUND) |
| 6 | EAGLE3 hot/reduced draft vocab + d2t scatter | NOT-FOUND |
| 7 | Hardcoded tree-traversal time threshold (=1) picking CUDA vs Torch fallback | NOT-FOUND |
| 8 | Drafter dies, verifier keeps running silently | NOT-FOUND |
| 9 | DFlash quantized target lm_head only at TP=1 | NOT-FOUND |
| 10 | Fused context KV failing on non-neox RoPE / k_scale / v_scale / quantized qkv_proj / qkv bias | **FOUND** for non-neox RoPE only; other three sub-conditions NOT-FOUND |
| 11 | Hidden states unsupported for DFlash-class drafters | NOT-FOUND (adjacent UNO-scoped statement found) |
| 12 | Multi-modal / external-embedding inputs unsupported by draft model | NOT-FOUND |
| 13 | Spec decode mutually exclusive with Mamba/hybrid prefill cache checkpoint blocks | NOT-FOUND |
| 14 | Overlap scheduling unsupported for certain spec algorithms | **FOUND** |
| 15 | Verification kernels only GREEDY draft sampling | **FOUND** |
| 16 | Multi-module MTP silent CUDA-graph downgrade (piecewise → FULL_DECODE_ONLY/NONE) | NOT-FOUND |
| 17 | EAGLE draft/draft-extend graph runners lacking `num_token_non_padded` (DSpark has it) | NOT-FOUND |
| 18 | Adaptive spec gated to EAGLE/EAGLE3 + topk 1 (unavailable in tree mode) | **FOUND** |
| 19 | `QLEN_ONLY_BITPACKING` unsupported in Triton impl | NOT-FOUND |
| 20 | Hardcoded tree-traversal threshold constant 1 (same as #7) | NOT-FOUND |
| 21 | n-gram capped to 1 CPU numba thread per TP rank (cap 1→8) | NOT-FOUND |

---

## ITEM 1 — Sampling mask / top-p support mask incompatible with speculative decoding

**VERDICT: FOUND**

- https://errors.standardbeagle.com/vllm-project/vllm/sampling-distribution-replay-does-not-support-spec/ — title: *"sampling distribution replay does not support speculative decoding — vllm"*. Quote: *"When return_sampling_mask is enabled, VllmConfig also rejects speculative decoding configurations. Sampling-distribution replay assumes a single forward's softmax/nucleus distribution; speculative decoding samples from multiple draft/target positions with accept-reject correction, so the returned mask would not correspond to a replayable per-token distribution."* Thrown at `vllm/config/vllm.py:1033`, raised when `return_sampling_mask` is truthy and `speculative_config is not None`.
- Same page lists solutions: drop `--speculative-config`, or run two deployments (speculative for serving, non-speculative V2 for mask collection).
- Related verified pages: https://errors.standardbeagle.com/vllm-project/vllm/sampling-distribution-replay-requires-model-runner/ , https://errors.standardbeagle.com/vllm-project/vllm/sampling-distribution-replay-requires-logprobs-mod/
- **Important negative:** vLLM's own docs page https://docs.vllm.ai/en/latest/training/sampling_mask/ (fetched in full) has a "Limitations" section that does **NOT** mention speculative decoding — it only says `--return-sampling-mask` globally disables the FlashInfer fused sampler and that there is no streaming support. So the official doc is not a source for this item.
- SGLang's own `return_sampling_mask` error entries are about disaggregation capacity, not spec decode (`.../return-sampling-mask-with-disaggregation-requires/`).

**SEARCHED:** `speculative decoding sampling mask incompatible top-p support mask vLLM` · `vLLM logitsprocs sampling mask speculative decoding not supported` · `vLLM "sampling mask" unsupported speculative decoding limitation` · `投机解码 采样掩码 不支持 top-p support mask` · `"sampling mask" OR "top-p support mask" speculative decoding incompatible engine returns mask` · `site:reddit.com vLLM sampling mask speculative decoding incompatible` · HN: `sampling mask speculative` (3 hits, all irrelevant) · Reddit r/LocalLLaMA: `speculative decoding sampling mask` (0) · Reddit r/MachineLearning: `top-p support mask spec decode` (0).

**NOTES:** Verified live (HTTP 200, title checked). Source is a source-mirror site, not discussion.

## ITEM 2 — Folding sampling-parameter application / rejection sampling into the verification kernel to remove a ~1GB FP32 target-logits scratch buffer

**VERDICT: NOT-FOUND**

**SEARCHED:** `speculative decoding rejection sampling fused verification kernel logits buffer` · `speculative decoding target logits FP32 buffer 1GB memory verification kernel blog` · `logits buffer rejection sampling fusing` (Reddit r/LocalLLaMA: 10 fuzzy hits, none on-topic) · HN: `rejection sampling fused kernel logits` (0 hits).

**NOTES:** I searched vLLM + SGLang error datasets (all 553 vLLM / 3,398 SGLang entries) for `logits_buffer|target_logits|rejection.*fuse|fuse.*rejection|chunked|scratch` — no entry about a large FP32 target-logits buffer in the verification path. The closest public artifact is a **GitHub PR** (your lane): *"[Perf][Spec Decode] Fuse target temperature in rejection sampler"* PR #53090, seen in search results. No blog, forum, HN or Reddit discussion found.

## ITEM 3 — CUDA-graph support for DRAFT-EXTEND limited to a subset of attention backends

**VERDICT: NOT-FOUND**

**SEARCHED:** `SGLang speculative decoding "draft extend" CUDA graph attention backend eager fallback` · `draft extend cuda graph` (HN: 0) · `site:reddit.com speculative decoding draft extend cuda graph attention backend` · `draft extend cuda graph attention backend` (Reddit r/LocalLLaMA: 10 fuzzy hits, none on-topic) · `draft extend cuda graph` (Reddit r/MachineLearning: 0) · `speculative decoding tree fallback torch` (HN: 0).

**NOTES:** GitHub-only pointers (your lane), seen with real titles in search results: vLLM PR #47460 *"[Bugfix] Initialize draft CUDA-graph keys for the native draft_model proposer"*; SGLang PR #28782 *"[Spec] Support FlashInfer CUDA graph for EAGLE draft-extend"*. The closest **non-GitHub** statement I found is a forum post, and it is about attention backends generally, **not** about CUDA-graph draft-extend: NVIDIA Developer Forums thread *"DFlash LLM for DGX Spark - too good to be true?"* (https://forums.developer.nvidia.com/t/dflash-llm-for-dgx-spark-too-good-to-be-true/366445 ), user *jwarner*: *"It works but currently only on flash attention backend. That holds it back."* Do not cite that as evidence for item 3's specific claim.

## ITEM 4 — n-gram / prompt-lookup speculative decoding capped to 1 CPU thread (TP n-gram matching unimplemented)

**VERDICT: NOT-FOUND**

**SEARCHED:** `speculative decoding n-gram prompt lookup tensor parallel CPU thread 1` · `sglang ngram speculative decoding single thread tensor parallel not implemented discussion` · `prompt lookup decoding tensor parallel ngram matching single thread SGLang blog reddit` · `ngram speculative decoding tensor parallel` (HN: 0; Reddit r/LocalLLaMA: 0; r/MachineLearning: 0) · `prompt lookup decoding thread` (HN: 1 irrelevant hit).

**NOTES:** GitHub-only pointer (your lane), seen with real title: vLLM PR #26056 *"[Spec Decode] Add Tensor Parallel Ngram decoding"*. No blog/forum/HN/Reddit discussion of a 1-thread cap. vLLM's n-gram docs page (docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/, fetched) does not mention threads or TP at all.

## ITEM 5 — Adaptive/dynamic speculation-length control unsupported with DP attention, two-batch overlap, PD-multiplexing, multi-layer EAGLE, or frozen-KV MTP

**VERDICT: FOUND (partial — see NOTES for which sub-clauses are unsupported by evidence)**

Public docs (fetched, HTTP 200, titles checked):

- https://docs.sglang.io/docs/advanced_features/speculative_decoding — UNO section, *"Key requirements and limitations"*: *"UNO requires CUDA with FA3 for prefill and decode, tensor and pipeline parallel sizes of 1, and no DP attention or context parallelism."* And: *"Ordinary overlap scheduling is supported. Tree mode does not yet support PDMux or the separate --enable-two-batch-overlap feature."* → covers **DP attention**, **PD-multiplexing (PDMux)** and **two-batch overlap**.
- https://errors.standardbeagle.com/sgl-project/sglang/currently-dflash-speculative-decoding-does-not-sup/ — *"Currently DFLASH speculative decoding does not support dp attention."*
- https://errors.standardbeagle.com/sgl-project/sglang/currently-standalone-speculative-decoding-does-not/ — *"Currently standalone speculative decoding does not support dp attention."*
- https://errors.standardbeagle.com/vllm-project/vllm/adaptive-verification-only-supported-with-dspark/ — *"Adaptive verification only supported with DSpark"*.
- https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding — *"Current support: Only --speculative-algorithm EAGLE or EAGLE3; Only --speculative-eagle-topk 1; If either condition is not met, SGLang falls back to static speculative settings."*

**NOT FOUND:** any public statement tying adaptive-length control specifically to **multi-layer EAGLE** or **frozen-KV MTP**. The nearest hits are about *rejection sampling*, not adaptive length: https://errors.standardbeagle.com/sgl-project/sglang/speculative-use-rejection-sampling-with-multi-la/ (*"--speculative-use-rejection-sampling with multi-layer EAGLE (--enable-multi-layer-eagle) requires --speculative-eagle-topk 1"*), and https://errors.standardbeagle.com/sgl-project/sglang/hicache-does-not-support-inkling-mtp-draft-state-y/ (*"HiCache does not support Inkling MTP draft state yet."*).

**SEARCHED:** `SGLang adaptive speculative decoding dynamic num draft tokens data parallel attention unsupported` · `two batch overlap PD multiplexing speculative decoding sglang blog` · `adaptive speculation length unsupported multi-layer EAGLE frozen KV MTP` · `adaptive speculative decoding` (HN: 9 hits incl. *"Adaptive speculative decoding: picking draft lengths at runtime"* https://fergusfinn.com/blog/adaptive-speculation/ — fetched, 200; it does **not** discuss these engine-specific incompatibilities) · `adaptive speculation length EAGLE` (Reddit r/LocalLLaMA: 0; r/MachineLearning: 0).

## ITEM 6 — Reduced/"hot" draft vocabulary in EAGLE3, d2t scatter before sampling for lossless rejection sampling

**VERDICT: NOT-FOUND** (outside GitHub issues/PRs and arXiv)

**SEARCHED:** `EAGLE3 hot draft vocabulary d2t scatter draft probabilities full vocab rejection sampling` · `sglang "hot" draft vocabulary d2t scatter EAGLE3 blog` · `reddit LocalLLaMA EAGLE3 draft vocab d2t reduced vocabulary` · `EAGLE3 "hot" draft vocabulary reduced vocab sampling scatter full vocab blog` · `heterogeneous vocab speculative decoding draft vocabulary mapping vLLM docs` · `EAGLE3 draft vocabulary d2t` (HN: 0; Reddit r/LocalLLaMA: 0; r/MachineLearning: 0).

**NOTES:** Only GitHub/arXiv pointers surfaced (your lane), seen with real titles: GitHub issue *"Clarification on d2t and t2d Mapping Logic in EAGLE-3 Draft Model"* (SafeAILab/EAGLE#222) and arXiv *"Speculative Decoding with a Speculative Vocabulary"*. Closest public *docs* statement is about a different mechanism (full-vocab requirement, not pruned-vocab scatter): https://errors.standardbeagle.com/sgl-project/sglang/standalone-speculative-decoding-requires-the-draft/ — *"STANDALONE speculative decoding requires the draft model to share the same vocabulary as the target model…"*. No blog/forum/HN/Reddit discussion found.

## ITEM 7 — Hardcoded/arbitrary tree-traversal time threshold (value 1) deciding CUDA tree kernel vs Python/Torch fallback

**VERDICT: NOT-FOUND**

**SEARCHED:** `tree speculative decoding CUDA kernel python fallback threshold traversal time` · `tree speculative decoding traversal time threshold kernel fallback python` · `SGLang tree speculative decoding kernel threshold python fallback time` · `tree speculative decoding kernel` (HN: 4 hits, none on this) · `tree speculative decoding kernel` (Reddit r/LocalLLaMA: 0; r/MachineLearning: 0) · `speculative decoding tree fallback torch` (HN: 0).

**NOTES:** No public discussion of a magic time-threshold constant. GitHub-only pointer (your lane), seen with real title: SGLang PR #36201 *"[Kernel] Bound the EAGLE tree ancestor walk and stop when the ancestor is missing"*. Error-mirror datasets contain no such entry (expected — a constant/comment does not raise).

## ITEM 8 — Drafter thread/process dies, verifier silently stops applying that request's controls but keeps running

**VERDICT: NOT-FOUND**

**SEARCHED:** `verifier continues after draft model worker dies speculative decoding` · `verifier drafter crash speculative` (HN: 0; Reddit r/MachineLearning: 2 fuzzy hits, both papers — Orthrus, HAMburger; r/LocalLLaMA: 0).

**NOTES:** No blog, forum, HN, Reddit or error-mirror entry describing this failure mode. Error-mirror search for `draft.*(die|dead|crash|kill)|verifier.*(die|dead|stop)` in SGLang and `draft.*(unavailable|lost|gone|stop|fail)|worker.*(die|dead)` returned no matching entries.

## ITEM 9 — DFLASH/DFlash-class drafting with a quantized target lm_head only working at tensor-parallel size 1

**VERDICT: NOT-FOUND** (outside GitHub)

**SEARCHED:** `DFlash draft model "lm_head" quantized tp_size 1` · `site:reddit.com DFlash lm_head quantized tensor parallel` · `DFlash lm_head quantized tp` (HN: 0; Reddit r/LocalLLaMA: 0; r/MachineLearning: 0).

**NOTES:** GitHub pointers only (your lane), seen with real titles: SGLang PR #35462 *"[Spec] Support quantized target lm_head in DFlash2 selector"*; commit `hanwlax/sglang@6c70fbe` *"[Spec] Support quantized target lm_head in the DFlash2 selector (#35496)"*. A **repo markdown file** (not an issue/PR, but GitHub-hosted — flagging, not counting) reads: *"SPEC-DFLASH2 | The DFlash2 candidate selector could not consume a QUANTIZED target lm_head, so the arm was refused on the…"* (`mudler/vllm.cpp`, `.agents/completed/issue-index.md`). Non-GitHub: I fetched the NVIDIA DGX Spark DFlash thread (38 posts) and the DSpark thread (47 posts) — neither states a TP=1 restriction; the community HF card https://huggingface.co/KingsonHO/Qwen3.6-27B-DFlash does use `--tp-size 1` but states no such limitation (the z-lab card uses `--tp-size 8`).

## ITEM 10 — "Fused context KV" for the draft model failing on non-unit k_scale/v_scale, non-neox RoPE, quantized qkv_proj, or qkv bias

**VERDICT: FOUND for the non-neox RoPE half only. k_scale/v_scale, quantized qkv_proj and qkv bias: NOT-FOUND.**

- https://errors.standardbeagle.com/sgl-project/sglang/only-neox-style-rope-is-supported/ — title *"Only neox-style RoPE is supported. — sglang"*. It is a `NotImplementedError` at `python/sglang/kernels/ops/speculative/fused_kv_materialize.py:270`: `if not self.is_neox_style: raise NotImplementedError("Only neox-style RoPE is supported.")`. This is the **speculative** kernel dir (`kernels/ops/speculative/`), i.e. the draft/target fused KV path.
- https://errors.standardbeagle.com/sgl-project/sglang/rope-config-mismatch-across-layers-for-fused-kv-pa/ — *"RoPE config mismatch across layers for fused KV path: expected (rotary_dim=…, neox=…), got (…) at layer {layer_id}."* (line 316).
- Also in the same file: `num_kv_heads`/`head_dim` cross-layer mismatch, cos/sin-cache-too-short, and shape/dtype validation errors (all verified live).

**NOT FOUND:** no public error message or discussion anywhere on these surfaces for **k_scale/v_scale**, **quantized qkv_proj**, or **qkv bias** in the fused-KV path. The mirrored source excerpt does show the fused path slicing `qkv_w = attn.qkv_proj.weight` and `kv_weight = qkv_w[attn.q_size : attn.q_size + 2*attn.kv_size]` (a dense weight slice), which is consistent with the claim, but **no explicit message states it**, so I am not asserting it. GitHub pointer for the quantized half (your lane): vLLM PR #51620 *"[Bugfix][Spec Decode] Route DFlash fused context-KV projection through quant_method for quantized drafters"*.

**SEARCHED:** `"fused context KV" draft model` · `SGLang "fused context KV" DFlash quantized draft k_scale v_scale` · `fused context kv` dataset grep across all vLLM+SGLang entries for `fused.context|context.kv|k_scale|v_scale|neox`.

**NOTES:** Error mirror = source mirror, not discussion. This is the strongest non-GitHub confirmation in the set.

## ITEM 11 — Returning hidden states unsupported for DFlash-class drafters

**VERDICT: NOT-FOUND** (for DFlash specifically)

**SEARCHED:** `return hidden states unsupported DFlash drafter speculative decoding` · `SGLang DFlash hidden states not supported draft model limitation` · `speculative decoding hidden states` (HN: 1 irrelevant hit) · `DFlash hidden states` (Reddit r/LocalLLaMA: 0; r/MachineLearning: 0). Dataset grep for `hidden_state|return_hidden|aux.hidden` across both repos.

**NOTES:** Adjacent but **not** DFlash-scoped, so it does not establish the item: SGLang spec-decode docs UNO limitations bullet (https://docs.sglang.io/docs/advanced_features/speculative_decoding) — *"Grammar decoding, returned logprobs or hidden states, sampling penalties, min_p, logit bias, custom logit processors, strict thinking, and deterministic inference are not yet supported."* (this list sits inside the **UNO** section). Also https://errors.standardbeagle.com/sgl-project/sglang/draft-sampler-set-but-the-draft-forward-has-no-hid/ — *"draft sampler set but the draft forward has no hidden_states to capture into the graph."* vLLM's *"Hidden State Extraction"* docs page (https://docs.vllm.ai/en/latest/features/speculative_decoding/extract_hidden_states/, fetched in full) describes the feature and says nothing about DFlash.

## ITEM 12 — Multi-modal / external-embedding inputs unsupported by the draft model

**VERDICT: NOT-FOUND** (outside GitHub)

**SEARCHED:** `speculative decoding draft model multimodal input not supported blog` · `vLLM draft model multimodal input embeddings speculative decoding limitation docs` · `draft model multimodal speculative decoding` (HN: 1 irrelevant hit; Reddit r/LocalLLaMA: 0; r/MachineLearning: 0).

**NOTES:** GitHub-only pointers (your lane), seen with real titles: vLLM PR #35714 *"Support multimodal speculative decoding in draft_model mode"*, vLLM PR #36097 *"[Model Runner V2] Support multi-modal embeddings for spec decode model"*, vLLM issue #43820. Dataset grep for `multimodal.*(spec|draft)` across both repos and for `embedding.*(spec|draft)` found nothing relevant (only diffusion `multimodal_gen` noise and *"Nemotron 3.5 DFLASH draft requires its checkpoint embedding"*, which is unrelated).

## ITEM 13 — Speculative decoding mutually exclusive with Mamba/hybrid-model prefill cache checkpoint blocks

**VERDICT: NOT-FOUND**

**SEARCHED:** `speculative decoding Mamba hybrid model prefill cache checkpoint incompatible` · `mamba speculative decoding prefill cache checkpoint` (Reddit r/LocalLLaMA: 0; r/MachineLearning: 0) · `Mamba speculative decoding` (HN: 1 irrelevant hit). Dataset grep for `mamba.*spec|spec.*mamba|hybrid.*spec|checkpoint.*spec` across both repos.

**NOTES:** Nothing states this exclusivity. Closest entries are about *different* constraints: https://errors.standardbeagle.com/sgl-project/sglang/speculative-decoding-with-enable-unified-memory/ (*"Speculative decoding with --enable-unified-memory is only supported for hybrid-Mamba targets; the unified hybrid-SWA pool's draft sizing (virtual-id space) is not wired yet."* — the inverse direction) and https://errors.standardbeagle.com/sgl-project/sglang/enable-int8-mamba-checkpoint-is-not-supported-to/ (*"--enable-int8-mamba-checkpoint is not supported together with --enable-hierarchical-cache"* — no spec decode involved).

## ITEM 14 — Overlap scheduling unsupported for certain speculative algorithms

**VERDICT: FOUND**

- https://errors.standardbeagle.com/sgl-project/sglang/speculative-algorithm-self-name-does-not-support/ — title *"Speculative algorithm {self.name} does not support overlap scheduling. — sglang"*, from `python/sglang/srt/speculative/spec_registry.py`.
- https://docs.sglang.io/docs/advanced_features/speculative_decoding (fetched) — *"The overlap scheduler currently only supports --speculative-eagle-topk 1; set --speculative-eagle-topk 1 explicitly. If you explicitly set --speculative-eagle-topk > 1, the server will error."*
- https://errors.standardbeagle.com/vllm-project/vllm/currently-async-scheduling-is-only-supported-with/ — *"Currently, async scheduling is only supported with EAGLE/MTP/Draft Model/NGram GPU/DSpark kind of speculative decoding"*.
- Related context: LMSYS blog *"The next generation of speculative decoding: DFlash and Spec V2"* (https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/, fetched) documents the overlap scheduler as the V2 engine's headline feature for DFlash — i.e. public material exists on overlap scheduling, but the *restriction* is documented in the SGLang docs page above.

**SEARCHED:** `overlap scheduling speculative decoding` (HN: 0; Reddit r/LocalLLaMA: 0; r/MachineLearning: 0) plus the two batches above.

## ITEM 15 — Verification kernels supporting only GREEDY draft sampling

**VERDICT: FOUND**

- https://errors.standardbeagle.com/vllm-project/vllm/use-heterogeneous-vocab-currently-only-supports-gr/ — verbatim title: *"use_heterogeneous_vocab currently only supports greedy draft sampling. Set draft_sample_method='greedy' (the default) or omit it. — vllm"*, from `vllm/config/speculative.py`. Found independently by search engine and confirmed live (HTTP 200).
- Companion entry: https://errors.standardbeagle.com/vllm-project/vllm/use-heterogeneous-vocab-only-works-with-method-dr/ — *"use_heterogeneous_vocab only works with method='draft_model'"*.

**SEARCHED:** `speculative decoding verification kernel supports only greedy sampling draft` · `greedy draft sampling verification kernel` (HN: 0; Reddit r/LocalLLaMA: 0; r/MachineLearning: 0) · `"use_heterogeneous_vocab currently only supports greedy draft sampling"`.

## ITEM 16 — Multi-module/multi-layer MTP silently downgrading its CUDA-graph mode

**VERDICT: NOT-FOUND** (outside GitHub)

**SEARCHED:** `multi-module MTP CUDA graph FULL_DECODE_ONLY piecewise vLLM downgrade` · `multi-module MTP` (HN: 1 irrelevant hit) · `FULL_DECODE_ONLY cuda graph` (HN: 2 hits, both irrelevant) · dataset grep `multi_module|FULL_DECODE_ONLY|piecewise.*(mtp|spec|draft)` across vLLM+SGLang (none).

**NOTES:** GitHub-only pointers (your lane), seen with real titles: vLLM issue #49547 *"FlashInfer + spec-decode silently downgrades to PIECEWISE cudagraphs (−16% measured); warning understates the cost"*; vLLM PR #50885 *"perf: Capture FULL decode cudagraphs for spec-decode on FlashInfer native path"*. No blog/forum/HN/Reddit discussion found.

## ITEM 17 — SGLang EAGLE draft / draft-extend CUDA-graph runners not supporting `num_token_non_padded` (DSpark path does)

**VERDICT: NOT-FOUND** (outside GitHub)

**SEARCHED:** `num_token_non_padded EAGLE draft extend cuda graph sglang` · `num_token_non_padded` (HN: 0) · dataset grep `num_token_non_padded` across SGLang (only `kernels/ops/moe/fill_padded_rows.py` argument-validation entries, nothing about draft-extend graph runners).

**NOTES:** The only public `num_token_non_padded` material I retrieved is MoE-kernel validation messages (e.g. https://errors.standardbeagle.com/sgl-project/sglang/num-token-non-padded-must-be-a-torch-tensor/ ) — unrelated to the draft-extend graph claim. A raw file mirror (https://raw.githubusercontent.com/sgl-project/sglang/fdebc938.../python/sglang/srt/model_executor/forward_batch_info.py , seen in search results) shows `capture_hidden_mode: CaptureHiddenMode = None` but I did not verify it supports or refutes this item, so I make no claim.

## ITEM 18 — Adaptive speculative decoding gated to EAGLE/EAGLE3 and `speculative_eagle_topk 1` (unavailable in tree mode)

**VERDICT: FOUND**

- https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding (fetched, HTTP 200, title *"Adaptive Speculative Decoding - SGLang Documentation"*) — *"Current support: Only --speculative-algorithm EAGLE or EAGLE3; Only --speculative-eagle-topk 1; If either condition is not met, SGLang falls back to static speculative settings."* The same page documents that each candidate tier owns its own captured CUDA graphs and that "Tier switch happens after the current round completes."

**SEARCHED:** `SGLang adaptive speculative decoding dynamic num draft tokens data parallel attention unsupported` · `adaptive speculation length EAGLE` (Reddit r/LocalLLaMA: 0; r/MachineLearning: 0) · `adaptive speculative decoding` (HN: 9 hits, none stating this restriction).

## ITEM 19 — EAGLE tree-mask bitpacking: `QLEN_ONLY_BITPACKING` unsupported in the Triton implementation

**VERDICT: NOT-FOUND**

**SEARCHED:** `QLEN_ONLY_BITPACKING SGLang Triton NotImplementedError tree mask bitpacking` · `QLEN_ONLY_BITPACKING` (HN: 1 irrelevant hit) · `eagle tree mask bitpacking` (HN: 0) · dataset grep `bitpack|QLEN_ONLY` across vLLM+SGLang (no entry — consistent with a TODO/comment rather than a raise).

**NOTES:** GitHub-only pointers (your lane), seen with real titles: SGLang commit `5589b75` *"Add treemask mode to build_eagle_tree & release sgl-kernel 0.2.3 (#7756)"*; SGLang PR #19591 *"[AMD] Port tree speculative sampling kernel to HIP"*. I found no blog, forum, HN or Reddit mention.

## ITEM 20 — Hardcoded tree-traversal time threshold (constant 1) choosing CUDA tree kernel vs Torch fallback

**VERDICT: NOT-FOUND** — identical finding to ITEM 7; same searches, no new evidence. No public discussion of the constant or of the fallback heuristic.

## ITEM 21 — n-gram/prompt-lookup drafting capped to ONE CPU numba thread per TP rank (cap 1→8 comment)

**VERDICT: NOT-FOUND** — identical finding to ITEM 4. Additional query in this round: `numba thread cap tensor parallel ngram speculative decoding 1 to 8` · `ngram numba thread` (HN: 0). No public mention of the cap or the "bump 1 → 8" comment. (The only related source artifact seen was a raw mirror of `vllm/v1/spec_decode/ngram_proposer.py` in search results; that is source, not discussion.)

---

## Blocked / inconclusive surfaces (explicit)

- **Zhihu:** `https://zhuanlan.zhihu.com/p/2054617534528237672` returned **HTTP 403**; title seen only in search results: *"把draft训得又小又准:NeMo-Automodel投机实战 (EAGLE-3/DFlash /Dspark)"*. Its body could not be read → **INCONCLUSIVE** for Zhihu content. Query issued: `知乎 投机解码 EAGLE3 d2t 词表 剪枝`.
- **Reddit comments:** `www.reddit.com/search.json` and `old.reddit.com/search.json` are blocked (HTML block page). `r/<sub>/search.rss` works and was used for post-level coverage. Per-thread `.rss` fetches for the three DFlash threads (`1urxfa1`, `1sx8uok`, `1vsy4l2`) returned **HTTP 429** twice → their comment bodies are **INCONCLUSIVE**. Post titles were retrieved and are real: *"Any ideas how to tune up DFlash Qwen3.6 27B on DGX Spark?"*, *"Luce DFlash: Qwen3.6-27B at up to 2x throughput on a single RTX 3090"*, *"I pushed Qwen3.8-27B limits again... Dflash2 - 134 tps on a RTX 3090"*.
- **Medium, PyTorch forums:** not fetched directly; covered only through `web_search` — no hits attributable to them for any of the 21 items.
- **NVIDIA Developer Forums:** HTML pages defeat text extraction; success came via the Discourse JSON API (`/t/<id>.json`). Two DFlash/DSpark threads were read in full (38 and 47 posts); they discuss acceptance rates, prefill throughput and container builds, not the items above.

## Honesty caveats

1. ErrLookup entries are **auto-generated source mirrors** (the site states: "AI-assisted analysis of vllm-project/vllm@c794754062 (2026-08-14)"). They prove the limitation string exists in public code, and that a third party publishes it outside GitHub issues/PRs — they are **not** evidence that a human discussed it.
2. All ErrLookup and docs URLs cited above were individually fetched (HTTP 200) with titles checked; GitHub PR/issue titles were seen in search results with their real titles.
3. For items marked NOT-FOUND, the correct statement is: *I searched these specific surfaces with these specific queries and found no matching public discussion* — not "nobody has discussed this".
