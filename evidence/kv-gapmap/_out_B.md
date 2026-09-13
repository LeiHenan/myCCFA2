## B. OPEN cells (explicitly unsolved, with evidence that someone is still asking)

### B1. Should fp8 KV cache be advertised for Mamba/GDN hybrids when it buys no capacity there — and can vLLM decouple per-group page size?
- **WHO is still asking:** `shivasathishs-rp` (RFC author), vLLM maintainers and users — vLLM issue #55196 `[RFC]: fp8 KV cache gives little to no memory benefit on Mamba/GDN hybrid models` (OPEN since 2026-09-03, label `quantization`)
- **URL:** https://github.com/vllm-project/vllm/issues/55196
- **Verbatim quote of the asking:** "Root cause is the known hybrid page-pinning behavior (#37121, #40696): the Mamba state is not quantized by `--kv-cache-dtype` and all groups share one uniform page, so the bf16 Mamba page pins bytes-per-block and the fp8 attention page cannot shrink the pool." … "fp8 KV cache is advertised as a ~2x capacity/concurrency lever. On Mamba/attention hybrids (Falcon-H1, Qwen3-Next, Granite-4, Jamba, Nemotron-H) that lever is mostly or entirely dead, and there is no caveat in the docs."
- **What specifically is missing:** A decided and shipped fix: quantize the recurrent/Mamba state (SGLang already does this at int8) and/or decouple per-group page size so a bf16 non-attention page no longer pins the attention page. The RFC states the mechanical blocker verbatim — "`get_uniform_page_size` asserts one page across all groups" — and no merged PR implements per-group pools. Reported capacity gains are 2.00x on attention-only models versus 1.00x at short context on hybrids.
- **Evidence:** READ BODY

### B2. A merged, enabled path for packing heterogeneous KV-cache pages in hybrid models
- **WHO is still asking:** `meenchen` — vLLM PR #56156 `[Feature] Support packed layerwise KV cache` (OPEN, default-disabled capability)
- **URL:** https://github.com/vllm-project/vllm/pull/56156
- **Verbatim quote of the asking:** "It lets hybrid attention/state-cache models retain exact per-layer page sizes using vLLM's existing packed UniformTypeKVCacheSpecs allocator instead of padding every layer to one physical page size." … "No quantization configuration enables the capability in this PR."
- **What specifically is missing:** Merge and enablement. The PR adds a generic, default-disabled `has_layerwise_kv_cache()` capability, so "exact per-layer page sizes" is proposed, not shipped. The body also states a deliberately excluded sub-feature verbatim: "Automatic manager-block optimization is intentionally excluded. Reusing a larger manager block requires a separately validated physical subpage layout; the current PR keeps the manager block equal to the supported kernel page."
- **Evidence:** READ BODY

### B3. Why does prefix caching return zero hits whenever the shared prefix is shorter than the forced attention block?
- **WHO is still asking:** vLLM users and maintainers — vLLM issue #40696 (OPEN), issue #53749 (OPEN, blocker for hybrid checkpoints), issue #48401 (OPEN, silent inertness), issue #52897 (OPEN, align-mode never hits)
- **URL:** https://github.com/vllm-project/vllm/issues/40696 and https://github.com/vllm-project/vllm/issues/53749 and https://github.com/vllm-project/vllm/issues/48401 and https://github.com/vllm-project/vllm/issues/52897
- **Verbatim quote of the asking:** "When serving Mamba-hybrid models like Qwen3.5, vLLM sets the attention block size to **528 tokens** to align with the mamba page size" … "Prompt with **< 528 tokens**: **0% prefix cache hit** — the entire prefix must be recomputed every request" … "For comparison, standard transformer models use block_size=16, so even a 100-token prompt gets 6 fully cached blocks (96 tokens cached)."
- **What specifically is missing:** A mechanism — or at least a diagnostic — for sub-block-aligned reuse, so that a shared prefix shorter than the auto-selected attention block (2096 / 1920 tokens on the hybrid checkpoints in #53749) is not structurally uncacheable. Related partial-hit work covers partial blocks through `hash_block_size`, but the alignment floor remains and the issues are open. #53749's own wording, as corrected by the verifier, is: "prefix caching registers whole blocks only, so a shared prefix shorter than one block is structurally uncacheable".
- **Evidence:** READ BODY (#40696); TITLE ONLY (#53749, #48401, #52897 — titles from the search-API result set and the verifier's downgrade of #53749)

### B4. KV-event emission at `hash_block_size` granularity for hybrid Mamba+Attention is still an open PR
- **WHO is still asking:** `vanshilshah97` — vLLM PR #43258 `v1/engine: emit prefix-cache KV-events at hash_block_size granularity for hybrid Mamba+Attention models` (OPEN)
- **URL:** https://github.com/vllm-project/vllm/pull/43258
- **Verbatim quote of the asking:** "v1/engine: emit prefix-cache KV-events at hash_block_size granularity for hybrid Mamba+Attention models" … the page's embedded metadata reports `"state":"OPEN"`.
- **What specifically is missing:** Events are still emitted at block granularity for hybrid models, so external KV consumers cannot see the finer `hash_block_size` boundaries that the in-tree fine-grained hashing introduced. Merge is required before router-side and offload-side consumers can act on the finer boundaries. The PR appears in the `is:pr is:unmerged "page size"` result set.
- **Evidence:** READ BODY

### B5. Hybrid block sizes can invert the hash/block relationship and crash the offload connector
- **WHO is still asking:** vLLM user reporting issue #56396 `[Bug]: DeepSeek-V4.1-Flash hybrid block_size (8) < hash_block_size (32) crashes the offload connector` (OPEN), with fix PR #56404 open
- **URL:** https://github.com/vllm-project/vllm/issues/56396
- **Verbatim quote of the asking:** "Note `block_size (8) < hash_block_size (32)` — the call asks `resolve_block_hashes` to view 32-token-granularity hashes at a *finer* 8-token granularity, but the function only supports the coarsening direction (`block_size >= hash_block_size`, enforced by the divisibility assert). With 8 < 32 the assert `8 % 32 == 0` can never hold, so the crash is deterministic on the first request."
- **What specifically is missing:** Support for a per-group block size smaller than the global `hash_block_size`. The failing layout is "MLA full-attention group @ block_size 32 + SWA/circular-buffer group @ block_size 8"; the fix PR #56404 exists but is not merged and the issue remains open.
- **Evidence:** READ BODY

### B6. The `min_num_layers` hybrid grouping heuristic is an acknowledged in-tree FIXME
- **WHO is still asking:** vLLM maintainers via a FIXME shipped in the tree (author "Chen"), with an unmerged fix in PR #56445 and bug #46462
- **URL:** https://github.com/vllm-project/vllm/pull/56445
- **Verbatim quote of the asking:** "FIXME(Chen): At the moment of writing this code (2025-06-02), all open-source hybrid model follows a n:1 pattern between different attention types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and full), so we can use the \"1\" in the n:1 pattern as the group size, which is the minimum number of layers among all attention types. Need a better strategy if we want to support more complex patterns (e.g., 20 full + 30 sw, where the group size should be 10)."
- **What specifically is missing:** A grouping algorithm that does not add dummy padding layers. The open fix PR states the cost verbatim: "For hybrid models with uneven layer counts across attention types (e.g. 20 full + 30 sw, or 10 full + 16 sw with DFlash), the existing min_num_layers heuristic caused up to 60% KV cache memory waste on dummy padding layers." The runtime warning is still shipped, logging "Add 6 padding layers, may waste at most 60.00% KV cache memory".
- **Evidence:** READ BODY

### B7. Dual-pool capacity rebalancing between SSM state and attention KV is unshipped
- **WHO is still asking:** the AVMP paper's authors in their own future-work statement (arXiv 2605.22416), and vLLM users on issue #55196
- **URL:** https://arxiv.org/abs/2605.22416
- **Verbatim quote of the asking:** "Current inference engines handle this poorly. Unified pools pad SSM states to attention page sizes, wasting up to 7.3x capacity. Static dual pools cannot adapt when prompt distributions shift between requests." … "Implementation is pure Python; Triton integration is future work."
- **What specifically is missing:** A shipped engine-side implementation of adaptive dual pools. vLLM has no per-group pool split, and SGLang's `--mamba-full-memory-ratio` is a static startup partition rather than an adaptive one. Nothing in either tree re-balances pool capacity as the prompt mix shifts.
- **Evidence:** READ BODY

### B8. Where should a single retained recurrent-state checkpoint be placed in a hybrid model?
- **WHO is still asking:** `nicholaskh-ai` — vLLM RFC #55697 `[RFC]: Application-Directed Prefix Checkpoints for Mamba / Hybrid Prefix Caching` (OPEN), with follow-up PRs #55873 / #55875 / #55876 submitted for review
- **URL:** https://github.com/vllm-project/vllm/issues/55697
- **Verbatim quote of the asking:** "The core dilemma is that **from an engine-internal perspective, it is virtually impossible to heuristically predict where that single, high-value checkpoint boundary should be placed**. Geometry-based interval chopping frequently misses the actual shared premise or prematurely cuts across context. Conversely, **from the business application's perspective, this boundary is immediately obvious and trivial to locate** (e.g., precisely at the junction where a shared source item, document context, or system premise ends)."
- **What specifically is missing:** A decision on the mechanism (application-supplied marker versus engine heuristic) plus review of the three split PRs. The RFC reports production numbers on L40S (Qwen3.5 35B, "+110.2% Throughput (2.1x Speedup)", "5.9 QPS to 12.4 QPS") but the issue is still open and the PRs are only "submitted for upstream review". The related in-tree knob `enable_mamba_fine_grained_prefix_cache` is documented as "Off by default" and covers only the EAGLE/MTP shared-prefix junction.
- **Evidence:** READ BODY

### B9. Mapping multiple full-attention layers to a single page
- **WHO is still asking:** `peakcrosser7` — vLLM PR #35703 `[Hybrid] Map multiple FullAttn layers to a single page` (OPEN), corroborated by a comment on issue #45238
- **URL:** https://github.com/vllm-project/vllm/pull/35703 and https://github.com/vllm-project/vllm/issues/45238
- **Verbatim quote of the asking:** "The block-granularity floor is untouched: when block_size > shared_prefix_len there is no boundary inside the prefix for either mechanism to checkpoint at — e.g. block_size 2096 on the 122B/397B vs a typical 1–2k-token system prompt. That's where this issue and [Hybrid] Map multiple FullAttn layers to a single page #35703's packing compose: packing brings boundaries inside the prefix, checkpoint coverage makes them reusable."
- **What specifically is missing:** A merged multi-layer-per-page mapping, the mechanism that would put reusable boundaries inside short prompts for wide hybrids. The packing PR is open and no merged artifact restores those boundaries; the corroborating comment lives on issue #45238, not on the PR page.
- **Evidence:** READ BODY

### B10. SGLang encoder-decoder shared boundary page is double-freed when `page_size > 1`
- **WHO is still asking:** SGLang reporters/maintainers — issue #38840 `[Bug] Encoder-decoder KV cache: shared boundary page is double-freed when page_size > 1` (OPEN)
- **URL:** https://github.com/sgl-project/sglang/issues/38840
- **Verbatim quote of the asking:** "[Bug] Encoder-decoder KV cache: shared boundary page is double-freed when page_size > 1" … the page's embedded metadata reports `"state":"OPEN"`.
- **What specifically is missing:** Correct ownership accounting for a page shared across the encoder/decoder boundary when the pool page size exceeds one token. The allocator's heterogeneous/shared-page refcount path is where the fix has to land, and no merged change addresses it.
- **Evidence:** READ BODY

### B11. The alternative (multi-layer) Mamba page-padding strategy is documented as unfinished
- **WHO is still asking:** vLLM maintainers, in the shipped design doc `docs/design/hybrid_kv_cache_manager.md`
- **URL:** https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/hybrid_kv_cache_manager.md
- **Verbatim quote of the asking:** "This can lead to more than 400 `block_size` for attention layers, which is too large. Another padding strategy is to increase `block_size` until …" … "This padding strategy is still a work in progress."
- **What specifically is missing:** Any shipped implementation of the second padding strategy. The doc's own header warns the feature "is still in its early stage and things may change". The consequence is visible in the field: aligned block sizes of 528 and 672 tokens are reported on public issues, and users report 1600 and 2096 tokens for 122B/397B checkpoints.
- **Evidence:** READ BODY

### B12. Prefix-cache retention is one global scalar: hit-inert EAGLE sliding-window draft groups still spend the shared LRU
- **WHO is still asking:** a vLLM contributor — RFC #54661 (OPEN, asks for agreement before code is posted)
- **URL:** https://github.com/vllm-project/vllm/issues/54661
- **Verbatim quote of the asking:** "On hybrid models that carry an EAGLE-style sliding-window draft group (DFlash/DSpark-shaped drafters), that group hashes `cdiv(sliding_window - 1, block_size) + 1` block ids at *every* prefix-cache boundary and frees them *cached* onto the protected tail of the one LRU that all groups share, while — once it is exempted from the hit `min` — contributing nothing to the reconciled hit length. On a GLM-5.3-Flash deployment this one group spends 87% of the prefix cache and caps coexisting conversations at ~50K tokens each."
- **What specifically is missing:** Agreement on a per-group retention policy; the RFC states "This RFC asks for agreement on the policy before any code is posted as a PR." Its own evidence caveat is on record: "All numbers below are **deployment evidence from a vLLM fork**, not upstream verification".
- **Evidence:** READ BODY

### B13. Fine-grained (sub-block) prefix hits for sliding-window groups
- **WHO is still asking:** vLLM issue #53786 (filed against `0.26.1rc1.dev926+gb05ae5dc0`, re-verified on `main` @ 2026-08-25); regression-guard PR #54663 is open
- **URL:** https://github.com/vllm-project/vllm/issues/53786
- **Verbatim quote of the asking:** "WARNING [kv_cache_coordinator.py] Disabling fine-grained prefix-cache hits because these KV cache managers require block-aligned lookups: SlidingWindowManager." … "The SWA lookup itself asserts *\"Fine-grained partial hits are not supported for sliding window now\"*, which reads as an acknowledged TODO."
- **What specifically is missing:** A merged implementation. `SlidingWindowManager` sets `supports_fine_grained_hash_lookup = False` and its block size differs from the hash unit, so `--prefix-match-unit 16` is accepted and then silently disabled with a single boot warning. Two attempts exist and both were closed unmerged; the surviving artifacts are the open regression guard #54663 and the open bugfix PR #53802 `[Bugfix] Register hybrid prefix-cache boundaries at hittable positions`.
- **Evidence:** READ BODY

### B14. Hybrid GDN prefix-cache hits can return NaN logits (correctness, not performance)
- **WHO is still asking:** vLLM bug report #55766 (v0.28.0, Qwen3.8-27B hybrid Gated-DeltaNet + attention, TP2 on 2x H100-80GB), OPEN
- **URL:** https://github.com/vllm-project/vllm/issues/55766
- **Verbatim quote of the asking:** "Decode-produced boundary checkpoints are never prefix-cached in v0.28.0 (the running block is freed before `cache_blocks`), so only prefill chunk ends matter, which is consistent with the trigger being the prompt length. The Mamba/GDN checkpoint written when a prefill ends 4 to 10 tokens past a block boundary appears to be bad; a later request restores it from the prefix cache, produces NaN, and the NaN block is then itself cached, which is why the retry fails identically."
- **What specifically is missing:** Root cause and an engine fix. The reporter has only client-side workarounds — "Salt the next request (`cache_salt`) when the previous prompt's length mod 816 falls in {4, 6, 8, 10}." — and states explicitly "we have no engine fix".
- **Evidence:** READ BODY

### B15. Hybrid mamba prefix-cache hit indexes the mamba block table in the wrong units
- **WHO is still asking:** vLLM bug report #55600, OPEN, with no linked merged PR
- **URL:** https://github.com/vllm-project/vllm/issues/55600
- **Verbatim quote of the asking:** "Above the row width this is an illegal memory access (Xid 31, `FAULT_PDE ACCESS_TYPE_VIRT_READ`); below it, the request silently restores the KDA/mamba state from an unrelated block."
- **What specifically is missing:** A fix for the unit mismatch in `MambaHybridModelState.add_request`, which computes the running mamba slot from `cache_config.block_size` after `EngineCore._initialize_kv_caches` has already lowered that value to the minimum across prefix-cacheable groups (here a 64- or 1024-token drafter group against `mamba_block_size` 7168). Both failure modes — silent state substitution and an illegal memory access — are unresolved.
- **Evidence:** READ BODY

### B16. Prefix caching on Mamba-2/GDN hybrids produced 0% hits in v0.26.0 (silent no-op)
- **WHO is still asking:** vLLM issue #51250 (Qwen3_5MoeForConditionalGeneration), OPEN and self-labelled "Not re-verified on current main"
- **URL:** https://github.com/vllm-project/vllm/issues/51250
- **Verbatim quote of the asking:** "The model's recurrent conversational + SSM state is not stored in the APC (which tracks attention KV blocks), so no prefix-cache entries are created. As a result, prefix caching is effectively a silent no-op." … "Preferably either: - vLLM should warn that --enable-prefix-caching is ineffective for hybrid SSM/Mamba models, or - vLLM should be extended to support caching the recurrent conv+SSM state so shared-prefix prefill reuse works".
- **What specifically is missing:** Either recurrent-state caching or an explicit warning/telemetry for ineffective hybrid APC. Later merged work covers Kimi-K3/FlashKDA checkpoint paths, but no merged change is tied to this issue and the measurement stands at "vllm:prefix_cache_queries_total 1131 / vllm:prefix_cache_hits_total 0".
- **Evidence:** READ BODY

### B17. External/cross-instance KV connectors do not partition on `cache_salt`, so salting does not isolate them
- **WHO is still asking:** vLLM bug report #53495, OPEN, with re-attempt PR #51748
- **URL:** https://github.com/vllm-project/vllm/issues/53495
- **Verbatim quote of the asking:** "So the dimensions that partition the in-engine prefix cache do not partition this connector: - Request A with `cache_salt=salt-x\"` is stored. Request B with the same tokens and `cache_salt=\"salt-y\"` gets `get_num_new_matched_tokens == 48`, three full blocks of A's KV."
- **What specifically is missing:** A connector-side key derivation that consumes `request.block_hashes` instead of re-deriving from prompt tokens. The report generalizes the class explicitly: "This is the class in RFC #53194: a connector that re-derives its key from tokens instead of consuming `request.block_hashes` drops every dimension at once". A prior attempt on the HF3FS connector (#51640) was closed unmerged.
- **Evidence:** READ BODY

### B18. The `cache_salt` trust boundary and its efficiency cost are undocumented
- **WHO is still asking:** vLLM docs PR #53347 (OPEN)
- **URL:** https://github.com/vllm-project/vllm/pull/53347
- **Verbatim quote of the asking:** "**1. Where the salt must come from.** The merged text covers salt unpredictability but not its origin, and `cache_salt` is an ordinary request-body field. In a self-hosted deployment the caller writes it, so an adversary who sets their own salt to a victim's value recovers the shared blocks and the timing signal returns. Entropy does not help there — the adversary is not guessing the value, they are supplying it."
- **What specifically is missing:** The doc addition itself is unmerged, and the residual timing channel created by caller-supplied salts is not stated in the shipped docs. The PR's own measurement gives the size of the effect: "Measured on A100-SXM4-80GB with Qwen2.5-7B-Instruct over a 2119-token shared prefix, read from `vllm:time_to_first_token_seconds`: 32.8 ms cached vs 149.6 ms uncached, about 4.6x."
- **Evidence:** READ BODY

### B19. HiCache's file backend can report an unrestorable hybrid prefix as a hit
- **WHO is still asking:** SGLang issue #39147, OPEN
- **URL:** https://github.com/sgl-project/sglang/issues/39147
- **Verbatim quote of the asking:** "`HiCacheFile.batch_exists_v2()` can report a hybrid prefix as a hit even when a required auxiliary pool cannot restore that prefix. The file backend finds each pool's longest usable prefix and takes the minimum of those lengths. That works for contiguous `ALL_PAGES` coverage, but `TRAILING_PAGES` pools can have holes in their valid endpoints." … "The usable prefix should end at the greatest common legal endpoint, or zero if none exists."
- **What specifically is missing:** A fix in `HiCacheFile.batch_exists_v2()`. The report is explicit about its own scope: "This report establishes an incorrect backend existence-query result. It does not claim a reproduced model-output error or an end-to-end load-back failure."
- **Evidence:** READ BODY

### B20. ReplaySSM forces a mamba radix-cache strategy that collapses prefix reuse (SGLang)
- **WHO is still asking:** SGLang issue #37834, OPEN
- **URL:** https://github.com/sgl-project/sglang/issues/37834
- **Verbatim quote of the asking:** "`--enable-linear-replayssm` requires `--mamba-radix-cache-strategy no_buffer`. That strategy stores the mamba checkpoint at a radix-tree depth later requests cannot reach, so prefix reuse collapses and TTFT rises sharply on any workload with a shared prefix."
- **What specifically is missing:** Compatibility between the linear-attention ReplaySSM path and the `extra_buffer` mamba radix-cache strategy. Per the SGLang Qwen3-Next cookbook, `extra_buffer` is the strategy that "Enables overlap scheduling and branching point caching", so the two features are currently mutually exclusive. The reported magnitude is "degrades mamba prefix caching and inflates TTFT up to 4.7x" on Qwen3.8-27B, 1×H200, TP=1.
- **Evidence:** READ BODY

### B21. The free list interleaves hit blocks with miss blocks, so hits are evicted before they can be reused
- **WHO is still asking:** `shanrow-amd` — vLLM PR #55998 (OPEN)
- **URL:** https://github.com/vllm-project/vllm/pull/55998
- **Verbatim quote of the asking:** "Currently the hit blocks interleave with miss/non-cached blocks in the free list. However we wish all hit blocks can be placed in the tail of the free list so that the hit blocks can be hit more and more before they are allocated out. So we introduce a new free list called hash-hit free list that is allocated after the original free list and to park hash-hit free blocks separately. So this can decrease the eviction amount of hit blocks."
- **What specifically is missing:** Merge, and evidence beyond unit tests. The PR's validation is "python -m pytest tests/v1/core/test_kv_cache_utils.py -v 76 passed" and "python -m pytest tests/v1/core/test_prefix_caching.py -v 89 passed", with no end-to-end hit-rate measurement.
- **Evidence:** READ BODY

### B22. Cache-aware admission ordering inside the engine (FCFS admission destroys pending cache hits)
- **WHO is still asking:** `fululi12` — vLLM PR #54625 `[Core][V1] Add cache-aware admission ordering` (OPEN), flagged as open in three separate evidence segments
- **URL:** https://github.com/vllm-project/vllm/pull/54625
- **Verbatim quote of the asking:** "The problem is an interaction between strict FCFS admission and prefix caching: a cold request admitted first allocates blocks, and under pressure that allocation evicts blocks which a request further back in the queue still needs. The eviction turns a would-be cache hit into recomputation, and the deeper the shared prefix, the more expensive the loss." … "Two new scheduler options, both off by default: Flag Default Meaning --cache-aware-admission-window N 0 Look-ahead depth over the waiting queue; 0 admits strictly in arrival order".
- **What specifically is missing:** A merge decision and any published measurement. Scope is restricted — "It requires the fcfs policy. Positional reordering is meaningless in a priority heap" and "It requires prefix caching, and is skipped on steps that defer prefill anyway." The predecessor mechanism was dropped after a post-mortem measured "+0.2%", so the open question is whether this formulation clears that bar; a sibling open PR #54366 and a competing SLO-oriented PR #53571 also exist.
- **Evidence:** READ BODY

### B23. Cache placement and request routing are still optimized separately
- **WHO is still asking:** the PrefixPlace authors, in the paper's own future-work statement (arXiv 2608.01655)
- **URL:** https://arxiv.org/abs/2608.01655
- **Verbatim quote of the asking:** "Future work includes co-optimizing request routing with placement and extending the epoch-level model to online placement with regret guarantees."
- **What specifically is missing:** A joint routing-plus-placement formulation. PrefixPlace solves placement only and the routing-side work (CacheRoute) solves routing only; neither is a combined system, and the paper's own agenda names online placement with regret guarantees as unbuilt.
- **Evidence:** READ BODY

### B24. vLLM's GPU-side `BlockPool` has no policy knob, and request priority does not affect KV retention
- **WHO is still asking:** `sjmshsh` on vLLM issue #40268 (OPEN, `"stateReason":"REOPENED"`, label `feature request`) and `lcc-star` on vLLM issue #47802 (OPEN); maintainer `njhill` has replied on #47802 pointing at #40004 and related PRs
- **URL:** https://github.com/vllm-project/vllm/issues/40268 and https://github.com/vllm-project/vllm/issues/47802
- **Verbatim quote of the asking:** "This design has one fundamental blind spot: **once `ref_cnt` drops to zero, all blocks are equal candidates for eviction, regardless of how many times they were previously reused.** A system-prompt block touched by 10,000 requests and a one-time-use block are indistinguishable at eviction time." … "For priority-based serving, it would be useful if request priority could also influence KV/prefix cache retention or eviction."
- **What specifically is missing:** A policy hook in the HBM block pool (per-group or priority-aware), or an accepted pointer to the existing work: no `gpu_eviction_policy` flag exists, and the pluggable `CachePolicyFactory` is scoped to the offload tier, not HBM blocks. The verifier's correction to the evidence file matters here — #47802 does have a maintainer reply, so the open question is not "is anyone listening" but which of the linked PRs lands.
- **Evidence:** READ BODY

### B25. A general sparse-KV-cache framework in vLLM, and per-head sparse layouts
- **WHO is still asking:** `chizhang118` (RFC author) with the vLLM maintainers on issue #5751 (OPEN, labels `keep-open`, `RFC`), and a vLLM user asking on the official forum thread 2745
- **URL:** https://github.com/vllm-project/vllm/issues/5751 and https://discuss.vllm.ai/t/sparse-attention-e-g-h2o/2745
- **Verbatim quote of the asking:** "Partial KV cache eviction (H2O, SnapKV, LESS, Adaptive Compression, Scissorhands, Dynamic Memory Compression, StreamingLLM): By removing some relatively useless KV cache entries, the memory footprint of the KV cache is reduced." … "One issue I am running into is that a page (or block) needs to extend over all heads of a model. In H2O or other sparse attention methods, KV vectors can be in the cache for some heads, but not for others."
- **What specifically is missing:** The RFC's planned flag — "An optional flag \"--sparse-kv-cache-type\" indicating if we want to specify any sparse KV cache type" — does not exist, and its implementation PR was closed unmerged. Underneath it, vLLM still has no non-uniform KV memory layout across heads, which is what per-head sparsity needs; the forum answer to that question is a user-side workaround, and the verifier notes the in-thread answer came from an automated assistant account rather than a maintainer.
- **Evidence:** READ BODY

### B26. KV connectors cannot see attention weights, blocking score-based token dropping
- **WHO is still asking:** `rannnayy` — vLLM PR #45184 (OPEN since 2026-06-10, no `mergedTime`, no `closedTime`)
- **URL:** https://github.com/vllm-project/vllm/pull/45184
- **Verbatim quote of the asking:** "This PR allows the KV connectors to also get intermediate tensors (such as query). This PR does not directly improve prefix caching or PD disaggregation. Instead, it serves as a pre-requisite for the KV connector to support latest research on KV cache optimizations such as token dropping."
- **What specifically is missing:** The prerequisite plumbing itself is unmerged. The related vLLM RFC #10646 named the same blocker — "Expose intermediate logits (i.e. attetnion weights) from attention kernels as a lot of token dropping decisions depend on attention weights" — and was stale-closed without the capability landing.
- **Evidence:** READ BODY

### B27. SGLang has no unified post-hoc KV-sparsity runtime framework
- **WHO is still asking:** `magicYang1573` — SGLang RFC issue #32657 (OPEN since 2026-07-28)
- **URL:** https://github.com/sgl-project/sglang/issues/32657
- **Verbatim quote of the asking:** "Today, SGLang does not yet provide a complete runtime framework for this use case. Some abstractions have been merged, and several Quest/HiSparse efforts are in progress, but the common interface and serving integration remain incomplete or method-specific."
- **What specifically is missing:** A common interface over the partial skeleton that exists (`python/sglang/srt/mem_cache/sparsity/{algorithms,backend,core,factory.py}`), plus request/decode-step/layer lifecycle hooks, logical-to-physical KV translation, attention-backend integration and CUDA-graph support.
- **Evidence:** READ BODY

### B28. No per-token-range retention priority in SGLang's eviction path
- **WHO is still asking:** `limarkdcunha` — SGLang RFC issue #36208 `[RFC] Per-Token-Range KV-Cache Retention Priority` (OPEN since 2026-08-24)
- **URL:** https://github.com/sgl-project/sglang/issues/36208
- **Verbatim quote of the asking:** "**No intra-request differentiation.** A request cannot mark its system prompt as more valuable than its user turn. The whole sequence shares one priority." … "**Cross-session unfairness.** Nothing stops the engine from evicting Session A's important prefix to make room for Session B's less-important request, even when A matters more — because \"importance\" is not expressible at sub-request granularity."
- **What specifically is missing:** The feature plus its unresolved design pieces. The RFC's Non-Goals section says priority decay is "Deferred — not because it's unwanted, but because it can't be added cleanly on top of the current tree without its own design pass" (wording as corrected by the verifier), and that decay "interacts with chunked prefill and node splitting". The RFC also states "**No timer infrastructure.** SGLang's tree has no timer or sweep mechanism today."
- **Evidence:** READ BODY

### B29. In-place KV compaction with no auxiliary VRAM has no home in vLLM
- **WHO is still asking:** `aemre-cetin` — vLLM feature request #55463 (OPEN since 2026-09-05, label `feature request`, no maintainer response on the page)
- **URL:** https://github.com/vllm-project/vllm/issues/55463
- **Verbatim quote of the asking:** "In long-context LLM serving (32k to 128k+ tokens), dynamic KV-cache eviction/compaction algorithms (e.g., H2O, Scissorhands, StreamingLLM, or heavy-hitter pruning) are essential to prevent out-of-memory (OOM) faults. However, standard runtimes execute compaction using **out-of-place gathering** (`torch.gather` / dynamic tensor allocation)."
- **What specifically is missing:** An accepted integration point. The issue's own proposed hook (`VLLMInplaceCompactionHook`) attaches to `worker.gpu_cache`, an interface the modern V1 path no longer exposes as written, so the proposal has no merge path. The claimed measurements are on the target hardware class, reported as "**Empirical Benchmarks on NVIDIA Blackwell (`sm_120`)**" with "**PyTorch Out-of-Place** | 3.350 ms | 96.00 MB" versus "**`idempotent-kv` (In-Situ)** | **2.031 ms** | **0.00 MB**".
- **Evidence:** READ BODY

### B30. Eviction across replicated KV tiers has no principled policy (published open problem)
- **WHO is still asking:** the authors of "From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving" (arXiv 2607.02574), §6 design-gap agenda
- **URL:** https://arxiv.org/html/2607.02574v1
- **Verbatim quote of the asking:** "DG3—Eviction across replicated tiers. A KV block can be live in HBM, host DRAM, and a remote store simultaneously. Existing tier-aware systems often approximate this with per-tier heuristics. A principled policy would price each tier's fetch latency against the probability of near-future reuse, which requires per-block reuse-distance and lifetime traces. MG4 and MG5 are therefore prerequisites."
- **What specifically is missing:** Per-block reuse-distance and lifetime traces (MG4) and prefetchability slack (MG5); the survey states these are preconditions for the policy, not optional niceties, and neither is available in the engines surveyed.
- **Evidence:** READ BODY
### B31. Does quantized KV interact correctly with prefix caching? Truncated generation on prefix-cache hits with fp8 KV
- **WHO is still asking:** `Edwardf0t1` — vLLM issue #47349 (OPEN since 2026-07-01, no labels)
- **URL:** https://github.com/vllm-project/vllm/issues/47349
- **Verbatim quote of the asking:** "With --kv-cache-dtype fp8 + --enable-prefix-caching , generation is silently truncated to a short, deterministic length whenever a request reads back a shared/cached prefix (i.e. on prefix-cache hits) — even though the request sets max_completion_tokens: 1000 and ignore_eos: true . The client sees a completion far shorter than max_tokens , with no error and no cancellation ." … "A cold workload with no shared prefix (random independent prompts, 0% prefix-cache hit) is unaffected on fp8 KV — full-length output at every concurrency, including 512. So the trigger is specifically the read-back of cached fp8 KV prefix blocks ."
- **What specifically is missing:** A root cause and fix for the fp8-KV plus prefix-cache read-back path. The reporter has not isolated the scope: "I have not yet isolated whether this is specific to the NVFP4 ( modelopt_fp4 ) path or a generic fp8-KV + prefix-cache bug". The reporter's own hardware is 8× B300, so the repro itself does not transfer to a single-GPU box, but the question does.
- **Evidence:** READ BODY

### B32. Which TurboQuant combinations are actually supported? Speculative decoding still fails, and the maintainer's follow-up list is unclosed
- **WHO is still asking:** `mgoin` (vLLM maintainer) on tracking issue #40069, and `noonghunna` on issue #40831 with re-reports #53180 and #52475
- **URL:** https://github.com/vllm-project/vllm/issues/40069 and https://github.com/vllm-project/vllm/issues/40831 and https://github.com/vllm-project/vllm/issues/53180 and https://github.com/vllm-project/vllm/issues/52475
- **Verbatim quote of the asking:** "MTP alone is fine (fp8_e5m2 KV + MTP → all tests pass). TurboQuant alone is fine (turboquant_3bit_nc, no spec-decode → all tests pass). Only the combination fails." … the maintainer's open list: "Long-context evals across presets (k8v4, t4nc, k3v4nc, t3nc): RULER, NIAH at 32K–1M, LongBench"; "Hybrid attention models (e.g. Qwen3.5, mamba+attention, interleaved SWA?)"; "MLA support (through a new attention backend?)"; and, under feature compatibility, "Speculative decoding / Eagle" and "KV connector / disaggregated serving (NIXL, LMCache, Mooncake)".
- **What specifically is missing:** Measurements and integrations the shipped TurboQuant backend does not yet have — long-context evals, a per-layer sensitivity sweep to inform `--kv-cache-dtype-skip-layers` defaults, a recommended config table, MLA, hybrid attention models, speculative decoding and KV-connector paths. The spec-decode case is worse than unverified: it silently emits degenerate token loops on hybrid GDN models, and the re-reports are open while the original was closed.
- **Evidence:** READ BODY

### B33. TurboQuant on sm121 — hybrid-model init failure and fp16 zero-point overflow on large value outliers
- **WHO is still asking:** `TechPrototyper` — vLLM issue #53334 (OPEN since 2026-08-22)
- **URL:** https://github.com/vllm-project/vllm/issues/53334
- **Verbatim quote of the asking:** "triton_turboquant_store keeps the per-vector value scale and zero-point in fp16 regardless of input dtype ( sc_f16 / zr_f16 ). With bf16 inputs containing an attention-sink-scale outlier of |min| > 65504, the zero-point overflows and the reconstructed values come back as ∞ — a poisoned cache line rather than graceful degradation." … "ValueError: Unknown TurboQuant cache dtype: 'auto'. Valid presets: turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc . The state layers resolve to cache dtype auto , and the TurboQuant backend has no path for a mixed-layer model."
- **What specifically is missing:** A decision on whether hybrid models are supported at all, plus a fix for the fp16 scale/zero-point overflow. The reporter frames the choice verbatim: "If hybrids are intended to be unsupported, a clearer message (\"hybrid models unsupported\") at config-validation time would fail earlier and clearer; if support is intended, this is the blocker."
- **Evidence:** READ BODY

### B34. Is there a calibration-free sub-8-bit KV backend that does not lose throughput (KVarN)?
- **WHO is still asking:** `philippebich` — vLLM RFC issue #46613 (OPEN since 2026-06-24), companion implementation PR #46812 open with `needs-rebase`
- **URL:** https://github.com/vllm-project/vllm/issues/46613
- **Verbatim quote of the asking:** "As vLLM's own TurboQuant blog reports, existing sub-4-bit methods buy 2.3 to 3.7x capacity but give up 40 to 52% of throughput , and aggressive low-bit quant also tends to cost accuracy. Losing both speed and quality is why the feature stays off." … "It also covers models TurboQuant cannot: it supports MLA, and it runs on hybrid and sliding-window models. TurboQuant supports full-attention and uniform sliding-window only."
- **What specifically is missing:** An accepted RFC and a merged backend. The companion PR is open and needs rebase, and several of the RFC's headline numbers were measured on TP=2 hardware (~106B model), so they are not reproducible on a single-GPU rig even once the code lands.
- **Evidence:** READ BODY

### B35. Generic INT8 KV support and the canonical INT4 methods (KIVI) are still unshipped in vLLM
- **WHO is still asking:** `Wiziechen` on issue #33480 (OPEN since 2026-01-31, labels `feature request` / `unstale`) and `a1exxd0` on draft PR #44059 (OPEN since 2026-05-30)
- **URL:** https://github.com/vllm-project/vllm/issues/33480 and https://github.com/vllm-project/vllm/pull/44059
- **Verbatim quote of the asking:** "FP8 requires SM9.0+ GPUs (H100+), but many production environments still use A100 (SM8.0) or consumer-grade cards. INT8 is universally supported across all modern NVIDIA GPUs (since Pascal architecture) and avoids FP8 emulation penalties." … "Adds the `int4_kivi` kv_cache_dtype: a software INT4 KV cache with KIVI-style per-channel K (one scale per channel over each full 16-token block) and per-token V quantization, nibble-packed with fp8_e4m3 block scales."
- **What specifically is missing:** A merged, broadly available INT8 path, and a merged KIVI-style INT4 path. The INT8 issue has been open roughly seven and a half months and was not closed as answered even after `int8_per_token_head` shipped; the KIVI work is a draft that documents its own failure past 64k context verbatim: "CUDA caps grid.y/z at 65535, so any gather/dequant with max_seq > 65535 (max_model_len past 64k) failed to launch with \"CUDA: invalid argument\"."
- **Evidence:** READ BODY

### B36. `int4_per_token_head` is accepted for head sizes the Hadamard write path cannot handle, and it crashes on SWA models
- **WHO is still asking:** `LiRunGuo` — vLLM PR #56198 (OPEN, 2026-09-10), with companion issue #48721 and fix PR #48842 (maintainer-approved, `needs-rebase`)
- **URL:** https://github.com/vllm-project/vllm/pull/56198 and https://github.com/vllm-project/vllm/issues/48721 and https://github.com/vllm-project/vllm/pull/48842
- **Verbatim quote of the asking:** "Before, running vllm serve google/gemma-4-E2B-it --kv-cache-dtype int4_per_token_head will crash because of a stride mismatch. Also for gemma 4, it has head dims 256 and 512." … (title verbatim) "[Bugfix][Attention] Reject int4_per_token_head for non-power-of-two head sizes".
- **What specifically is missing:** A merged validation guard for non-power-of-two head dimensions and a merged fix for the SWA/Gemma-4 stride mismatch. The failure is specific: the mode's write path rotates K/V with a Hadamard transform (`reshape_and_cache_int4 -> single_rht -> fast_hadamard_transform`), which the guard PR would reject up front; #48842 carries an approval but no merge.
- **Evidence:** READ BODY

### B37. The `kv_cache_dtype` surface is a hard-coded closed list, and seven of its values are undocumented
- **WHO is still asking:** `lcfenglinwan` on RFC #51751 (OPEN since 2026-08-11) and `robertlangdonn` on docs PR #53299 (OPEN since 2026-08-21)
- **URL:** https://github.com/vllm-project/vllm/issues/51751 and https://github.com/vllm-project/vllm/pull/53299
- **Verbatim quote of the asking:** "These are hard-coded. Adding a new KV cache dtype requires modifying vLLM core in at least four places: CacheDType literal — add the string; STR_DTYPE_TO_TORCH_DTYPE — map it to a torch.dtype; KVQuantMode — map it to specified quant mode; is_quantized_kv_cache() — teach it the new dtype's properties. This makes it impossible for: Hardware backends — to ship their own KV cache quantization format without editing vLLM ...; Third-party quantization libraries — to integrate directly." … from the docs PR: "Added the CUDA SM89+ requirement for fp8_per_token_head and the power-of-two head-dimension constraint for int4_per_token_head 's Hadamard transform, both enforced in source but missing from the first draft."
- **What specifically is missing:** A registration contract that lets third-party and hardware-specific KV formats be added without editing vLLM core, and documentation for the values that already exist. The RFC records "Status: Implemented in PR #52670" but remains open; the docs PR is open and still leaves the `turboquant_*` family undocumented, blocked on the stalled `needs-rebase` PR #45422.
- **Evidence:** READ BODY (the RFC quote is a restructuring of the source's numbered four-place list into one span, not a single contiguous sentence)

### B38. SGLang's FP8 KV-cache decode path is unfused and slower than BF16 at small batch / short context
- **WHO is still asking:** `Swipe4057` — SGLang issue #30815 (OPEN since 2026-07-10), related PR #31652 open
- **URL:** https://github.com/sgl-project/sglang/issues/30815
- **Verbatim quote of the asking:** "The root cause is architectural: sglang performs FP8 KV-cache quantization as a series of separate Python-level tensor operations (multiple kernel launches per layer per step), whereas the BF16 path is a single optimized store_cache call. On memory-bound decode with small batches, this overhead outweighs the 2× reduction in KV read traffic." … "For a model with ~64 layers, this is 64+ × 3 = ~192 .to() launches per forward pass."
- **What specifically is missing:** A fused FP8 KV store kernel and a single central Q-quantisation point in SGLang, matching what vLLM already does. The related PR "[JIT] Fuse FP8 KV-cache quantization into the store kernel" (#31652) is open, so Hopper users still pay the launch overhead that erases the memory win.
- **Evidence:** READ BODY

### B39. `turboquant_k8v4` decode penalty grows with context length on Ampere
- **WHO is still asking:** vLLM issue #55610, labels `speculative-decoding` / `quantization`, OPEN since 2026-09-06
- **URL:** https://github.com/vllm-project/vllm/issues/55610
- **Verbatim quote of the asking:** "[Perf]: turboquant_k8v4 decode penalty scales with context length (-2.8% at 3k, -31% at 55k) on Ampere SM86, no speculative decoding"
- **What specifically is missing:** An explanation and fix for the decode-side dequantisation cost of the k8v4 preset, which contradicts the paper-level expectation that FP8-key storage is free. The measurement is title-only evidence, so the mechanism (kernel choice, dequant placement, or SM86 codegen) is not yet attributed.
- **Evidence:** TITLE ONLY

### B40. SGLang has three concurrent TurboQuant proposals and has not consolidated on one
- **WHO is still asking:** `hnyls2002` (SGLang collaborator) closed one competing proposal and pointed at another; the consolidated PR #21419 is still OPEN
- **URL:** https://github.com/sgl-project/sglang/pull/21419
- **Verbatim quote of the asking:** "There are three concurrent TurboQuant proposals and we are consolidating on #21419 , which reuses the existing flashinfer_backend instead of adding a parallel flashinfer_tq backend to attention_registry.py , is roughly 1500 lines smaller, and ships an end-to-end accuracy eval."
- **What specifically is missing:** The merge itself. `Add TurboQuant KV cache compression (3-4 bit, ICLR 2026)` (#21419) is still open, so SGLang 0.5.19 ships no TurboQuant KV cache while vLLM already has one — a cross-engine divergence rather than an unanswered design question.
- **Evidence:** READ BODY

### B41. Tiered KV offload promotes every waiting request and thrashes the primary DRAM tier
- **WHO is still asking:** `t348575` — vLLM issue #49902 (OPEN), with fix PR #50014 open
- **URL:** https://github.com/vllm-project/vllm/issues/49902 and https://github.com/vllm-project/vllm/pull/50014
- **Verbatim quote of the asking:** "Data from the secondary KV offload FS tier appears to blindly be promoted to the DRAM (cpu) tier, which leads to no/little free blocks in this tier. pin dram blocks for promotion -> evict just loaded items -> pin again for promotion cycle (this is an assumption from my data and a rough read of the code). This issue will therefore present itself whenever there are multiple long-context requests waiting to be executed, and there is CPU memory pressure." … "The constant re-promotions are obviously read from disk, so vLLM ends up reading 1+TB of data while it only needed to read a 10th of that."
- **What specifically is missing:** A promotion/admission policy that only promotes blocks for requests the scheduler will actually run; today the lookup path promotes on behalf of every waiting request it walks, including ones the scheduler skips. The fix PR tracks landed-but-unread promotions and protects them, but it is open and the shipping tiering manager has no such protection.
- **Evidence:** READ BODY

### B42. Back-pressure detection for KV cache offloading tiers
- **WHO is still asking:** `bnellnm` — vLLM RFC issue #50031 (OPEN)
- **URL:** https://github.com/vllm-project/vllm/issues/50031
- **Verbatim quote of the asking:** "vLLM's KV cache offloading framework (CPU DRAM, local NVMe, shared storage) does not detect or respond to bandwidth saturation on secondary tiers. When a tier is saturated — whether from concurrent eviction, excessive store operations, or high IOPS — the system continues dispatching load/store jobs that pile up, further degrading bandwidth for actual serving loads." … "Here the slowest tier becomes the bottleneck for the entire system. Further, we observe CPU tier running out of memory, preventing further offloads."
- **What specifically is missing:** The RFC states its own scope limit verbatim: "The initial implementation adds back-pressure detection to secondary offloading tiers and a single response policy: drop store (write) requests to a tier that is under pressure. No load prioritization, tier rerouting, or load-vs-recompute decision is included in this phase." Load prioritization, tier rerouting and the load-versus-recompute decision remain unsolved.
- **Evidence:** READ BODY

### B43. HiCache and the unified radix cache lose retrievable prefixes when the host/L3 copy is the only survivor
- **WHO is still asking:** `todayim` — SGLang issue #38452 (OPEN) and `JustinTong0323` — SGLang issue #33713 (OPEN)
- **URL:** https://github.com/sgl-project/sglang/issues/38452 and https://github.com/sgl-project/sglang/issues/33713
- **Verbatim quote of the asking:** "With HiCache enabled (UnifiedRadixCache + a storage backend), when a prefix's KV is evicted from both device (L1) and host (L2) tiers but its storage backup has completed, the prefix's tree nodes survive as backuped-only stubs. On re-sending the same prefix: the prefetch path treats the stub span as 'already matched' and issues no storage query for it, and the load-back path rejects the span because stubs carry no host indices. The request therefore recomputes the full prefix even though the data is present in L3." … "only considers the Full (KV) component when it decides whether a node can stay as an evictable-host leaf ( _evict_host_leaf only checks component_data[FULL].host_value )."
- **What specifically is missing:** Stub-anchored L3 prefetch/load-back on the HiCache side, and host-anchor logic for non-Full components at device-evict time on the unified-radix side. In the second report the MAMBA (hybrid-KDA) node is pruned outright instead of demoted, so a re-request never even attempts H→D load-back and reports `cached_tokens_details=None`; the reporter adds "Happy to prepare a PR if maintainers want to assign this."
- **Evidence:** READ BODY

### B44. HiCache host→device load-back sits on the critical path instead of overlapping the forward
- **WHO is still asking:** `qidaye` — SGLang issues #38470 (forwarded to #38472) and #38448 (forwarded to #38449), OPEN
- **URL:** https://github.com/sgl-project/sglang/issues/38470 and https://github.com/sgl-project/sglang/issues/38448
- **Verbatim quote of the asking:** "HiCacheController.start_loading submits the whole host->device load-back burst inline on the scheduler thread: L2TransferEngine.submit_host_to_device loops over every layer and calls load_to_device_per_layer , recording one event per layer so the forward can wait layer by layer." … "The forward's first kernels depend on those tensors, so the whole forward on the compute stream waits until the copy engine has drained the entire burst, not just layer 0's load. Under TP the stalled rank then delays all its peers through the first all-reduce."
- **What specifically is missing:** Correct scheduling of the H2D burst against the forward stream so layer-wise overlap actually happens. The measured trace is damning: per-layer submission (CPU) ~2.4 ms, `start_loading` per burst (78 layers × 2 pools) ~375 ms, "share of the H2D copy time that overlapped compute | 0.04 %", and 27/80 prefill rounds whose first attention kernel starts only after the last burst copy. The proposed enqueue-thread fix (`SGLANG_HICACHE_ASYNC_LOAD_ENQUEUE`, default off) is unimplemented and carries its own correctness problem: "Waiting on a device event that has not been recorded yet is a no-op, so a consumer could sail past a layer the loader thread has not submitted."
- **Evidence:** READ BODY

### B45. The HiCache backup thread dies on any storage-backend exception, leaking queued backups and host memory
- **WHO is still asking:** `qidaye` — SGLang issue #38428 (forwarded to #38427), OPEN
- **URL:** https://github.com/sgl-project/sglang/issues/38428
- **Verbatim quote of the asking:** "HiCacheController.backup_thread_func (and the HybridCacheController override) only catches queue.Empty — i.e. any other exception from the storage backend terminates the backup thread and queued backups plus their host memory are never released."
- **What specifically is missing:** Exception isolation and a recovery path in the HiCache backup loop. As written, the L3 write-back path is not fault-tolerant: one backend error permanently stops persistence and leaks the host memory of everything already queued.
- **Evidence:** READ BODY

### B46. RFC — out-of-process HiCache data plane with device-memory IPC
- **WHO is still asking:** `imReese` — SGLang issue #37372 (OPEN)
- **URL:** https://github.com/sgl-project/sglang/issues/37372
- **Verbatim quote of the asking:** "I would like to discuss whether HiCache should support an out-of-process data-plane mode , where a local HiCacheDaemon runs separately from the scheduler process and accesses the scheduler's GPU KV pool through a device-memory IPC mechanism ( CUDA IPC on NVIDIA as the initial backend )." … "Today, HiCache transfer and lower-tier cache management are closely coupled to the scheduler process."
- **What specifically is missing:** A process and ownership boundary for the HiCache data plane. The RFC is explicit that this is unresolved — "This RFC is primarily about the process and ownership boundary . It is not a claim that the proposed architecture is necessarily faster than the current in-process implementation."
- **Evidence:** READ BODY

### B47. KV cache CPU offloading is unsupported for vLLM-Omni pipelines
- **WHO is still asking:** `Vivo50E` (assignee) with `hsliuustc0106` and `KuntaiDu` (vLLM/LMCache committer) — vllm-project/vllm-omni issue #1150 (labels `enhancement`, `high priority`), open since 2026-02-02
- **URL:** https://github.com/vllm-project/vllm-omni/issues/1150
- **Verbatim quote of the asking:** "Today, AR KV must remain fully on GPU, forcing operators to shrink max_model_len / max_num_seqs , scale up GPU capacity, or tolerate OOM-driven failures. Omni already offloads diffusion weights to CPU, but this mechanism does not apply to AR KV."
- **What specifically is missing:** Per-stage composition of `OffloadingConnector` + `LMCacheConnectorV1` via `MultiConnector` inside Omni's scheduler and AR stage runner — proposed in the RFC, not shipped. The author asks for review of a trial implementation ("please check #1330 for an initial trial"), so even the prototype has not landed after roughly seven months.
- **Evidence:** READ BODY

### B48. Which tiering semantics should vLLM's offloading framework standardize on?
- **WHO is still asking:** `dannyharnik` — vLLM RFC issue #38260 (still open; implementation landed via #40020) and `orozery` / `njhill` / `josephrocca` — vLLM RFC issue #19854 (open since 2025-06-19, labels `RFC`, `keep-open`)
- **URL:** https://github.com/vllm-project/vllm/issues/38260 and https://github.com/vllm-project/vllm/issues/19854
- **Verbatim quote of the asking:** "To date, vLLM offers native KV offloading to CPU memory but does not support further offloading from CPU memory to other tiers such as storage. Implementations for storage offload should either work directly with storage or implement their own CPU offloading as an additional tier." … on the tier-sizing question, `njhill`: "Cached CPU blocks would't necessarily have all preceding blocks also in the CPU cache, some prefix might only reside in the GPU cache. For a given cache tier, we may some length threshold below which it's not worth retrieving (and therefore possibly also not worth storing) since it would be faster to recompute."
- **What specifically is missing:** Settled answers on cross-TP canonical form, secondary-tier eviction ownership, P2P orchestration and the GPU→CPU save hook. The multi-tier RFC records "Feedback Period. No response" and "Any Other Things. No response", and the older RFC's thread ends without resolution on "allowing the connector to be notified on 'soon-to-be-evicted' GPU blocks."
- **Evidence:** READ BODY

### B49. LMCache does not decouple KV cache management from the inference engine, and cannot host token-dropping research
- **WHO is still asking:** the LMCache authors, in their own paper's deployment-lessons and limitations sections (arXiv 2510.09665)
- **URL:** https://arxiv.org/html/2510.09665v1
- **Verbatim quote of the asking:** "Many storage companies, such as company I, R, C, and W, have explicitly asked for decoupling the KV cache management and the core inference code, since they want to touch as minimum core inference logic as possible. Although LMCache does not solve this for now, the trend is clear: an LLM inference engine is pairing with a variety of external KV cache management services." … "However, it is non-trivial to implement token dropping in LMCache , since LMCache assumes the number of tokens that is input to and output of LMCache should be the same."
- **What specifically is missing:** (a) a clean engine/cache-engine separation interface; (b) any ability for the cache layer to change the token count or touch attention intermediates. The second limitation is what blocks token-dropping and compression research from being expressed in the cache layer at all.
- **Evidence:** READ BODY

### B50. HiCache runtime attach/detach of the L3 backend has no partial rollback across data-parallel ranks
- **WHO is still asking:** the SGLang project, in its own shipped runtime attach/detach document (an open TODO in code)
- **URL:** https://docs.sglang.io/docs/advanced_features/hicache_storage_runtime_attach_detach.md
- **Verbatim quote of the asking:** "Currently there is **no automatic partial rollback** across DP ranks (see TODO in code). Operationally: * Prefer to keep backend config identical across ranks * If attach fails, immediately call detach (best-effort/idempotent), fix config, then retry attach" … "This is intended to prevent \"silent partial success\", but it also means you may see: * Overall **failure** even though **some ranks already succeeded**".
- **What specifically is missing:** An atomic, all-or-nothing attach/detach of the HiCache L3 storage backend when `dp_size > 1`. The documented behaviour is fail-loud-but-inconsistent: success is reported true only if all ranks succeed, and there is no automatic rollback of the ranks that already attached.
- **Evidence:** READ BODY

### B51. The PCIe-bandwidth bottleneck model for KV offload ignores write-back cost, read-write contention, and MLA
- **WHO is still asking:** the authors of "Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading" (arXiv 2601.19910), in their own Limitations section
- **URL:** https://arxiv.org/abs/2601.19910
- **Verbatim quote of the asking:** "(Limitations, §9) \"Our framework focuses on loading KV caches from CPU DRAM, ignoring the overhead of storing new prefixes back to DRAM. We assume write operations occur post-prefill and that bidirectional PCIe mitigates impact. However, larger T T values may introduce unmodeled overheads from read-write contention.\"" … "We attempted to evaluate DeepSeek-V2 for MLA characterization but encountered implementation-specific overheads preventing accurate PCIe transfer isolation; we defer comprehensive MLA evaluation to future work."
- **What specifically is missing:** (a) a model of the store/write-back path and bidirectional PCIe read-write contention; (b) any characterization of MLA models under KV offload. The paper's headline finding is that the bottleneck is fundamental, so the missing terms bound how far its conclusions can be pushed in production.
- **Evidence:** READ BODY

### B52. Full-cache host-memory offload "quickly hits a ceiling" — transfer volume, not sparsity, becomes the decoding bottleneck
- **WHO is still asking:** the KVDrive authors (arXiv 2605.18071), in their own problem statement
- **URL:** https://arxiv.org/abs/2605.18071
- **Verbatim quote of the asking:** "Existing offloading systems store the full cache in host memory and selectively fetch critical entries during decoding, but this strategy quickly hits a ceiling: sparsity cannot be pushed further without degrading accuracy. As a result, when context length and batch size grow, the volume of KV transfers rises sharply and becomes the dominant source of decoding latency."
- **What specifically is missing:** Any way to keep raising sparsity without accuracy loss. KVDrive's own answer is systems-level orchestration (prefetch scheduling and transfer shaping) rather than more sparsity, which leaves the algorithmic ceiling itself unresolved.
- **Evidence:** READ BODY

### B53. The vLLM filesystem secondary tier has no configurable disk budget, and its capacity semantics are undecided
- **WHO is still asking:** `akalin9507` — vLLM PR #54327 (OPEN, `needs-rebase`) and RFC issue #54779 (OPEN since 2026-09-01)
- **URL:** https://github.com/vllm-project/vllm/pull/54327 and https://github.com/vllm-project/vllm/issues/54779
- **Verbatim quote of the asking:** "The filesystem secondary KV tier currently grows without a configurable disk budget. This PR adds optional max_bytes capacity accounting and basic LRU eviction to the existing FS manager." … "Needs community confirmation: should the capacity semantics be defined per local tier instance, per rank, or as a global capacity for a shared namespace?" … "Needs community confirmation: is partial admission the appropriate semantics, or should an oversized batch be rejected wholesale?"
- **What specifically is missing:** An accepted admission/eviction policy and an agreed capacity semantic for the FS tier, then a merge. The PR's own scope was cut — "This is the first step of #54779 . The scope has been reduced to make the implementation easier to review and validate before expanding the design." — and both code owners are still "Awaiting requested review".
- **Evidence:** READ BODY
### B54. Is the cross-layer (block-major) KV cache layout safe to enable for the ROCm AITER MLA decode backends?
- **WHO is still asking:** `aarushjain29` — vLLM issue #46411 (OPEN), a tracking issue explicitly opened as follow-up to PR #46401 / #45111
- **URL:** https://github.com/vllm-project/vllm/issues/46411
- **Verbatim quote of the asking:** "This issue tracks the actual follow-up work: verifying and (if safe) enabling the cross-layer KV cache layout for the **ROCm AITER MLA** backends so they can opt in like the other verified backends."
- **What specifically is missing:** AITER's decode wrapper must be made stride-aware (`get_kv_cache_stride_order(include_num_layers_dimension=True)` returning `(1, 0, 2, 3)`), which the issue says "likely requires an AITER-side change in addition to vLLM wrapper changes". Two concrete unsafe assumptions are listed — flat index math in `_expand_page_indices_kernel` / `_build_decode`, and flattening of `kv_buffer` in `forward_mqa` — and the page badge reads "Open".
- **Evidence:** READ BODY

### B55. Is MLA the only viable KV-reducing sparse path on SM120 (RTX PRO 6000 Blackwell)?
- **WHO is still asking:** `piweb-labs` — vLLM RFC #43235 (OPEN; auto-marked stale but not closed)
- **URL:** https://github.com/vllm-project/vllm/issues/43235
- **Verbatim quote of the asking:** "After extensive production testing on SM120 (RTX PRO 6000 Blackwell, 8×96GB, PCIe Gen5), NSA (Native Sparse Attention) cannot be made to work reliably for production serving on NVIDIA consumer/workstation GPUs. This is not a bug — it is an architectural incompatibility." … "MLA avoids all of these problems: - MLA compresses KV cache structurally (low-rank projection), no per-step sparse indexing needed".
- **What specifically is missing:** A maintainer decision on the claim. The RFC asserts MLA is the answer but carries no engine change; the bot marked it stale on 2026-08-19 ("This issue has been automatically marked as stale because it has not had any activity within 90 days."), and the retrieved page records no maintainer ruling.
- **Evidence:** READ BODY

### B56. Why can a NoPE MLA model not be served at all on SM120 across every SGLang DSA backend?
- **WHO is still asking:** `fsadj` — SGLang issue #39302 (OPEN, reported 2026-09-13 on SGLang 0.5.19, RTX PRO 6000 Blackwell 96 GB)
- **URL:** https://github.com/sgl-project/sglang/issues/39302
- **Verbatim quote of the asking:** "On **SM120** (RTX PRO 6000 Blackwell, compute capability 12.0) this model cannot be served at all. Every `--dsa-prefill-backend` / `--dsa-decode-backend` choice fails, each for a distinct reason, and the one backend the code *forces* for GLM + fp8 KV + SM120 is the one whose FlashInfer kernel rejects NoPE. So with `--kv-cache-dtype fp8_e4m3` the model is unrunnable by construction."
- **What specifically is missing:** Shared-memory budget on the target part: the reporter measures `cudaDevAttrMaxSharedMemoryPerBlockOptin` = "101376 B (99 KB)" versus "Hopper/B200 expose ~228 KB, which is the budget the kernels below are written against", so the tilelang sparse-MLA prefill kernel's 202 KB demand cannot be satisfied. This is exactly the target single-GPU configuration, and the issue was reported on the research date itself.
- **Evidence:** READ BODY

### B57. How should LoRA adapters targeting MLA's `kv_b_proj` be applied when decode absorbs the projection?
- **WHO is still asking:** `pstefa1707` — vLLM issue #48974 (OPEN; maintainer `aoshen02` has replied and pointed at a fix PR)
- **URL:** https://github.com/vllm-project/vllm/issues/48974
- **Verbatim quote of the asking:** "An adapter containing `kv_b_proj` LoRA tensors loads without error and is applied **nowhere**." … "**Decode (absorbed/MQA):** attention consumes `W_UK`/`W_UV`, split from the raw `kv_b_proj` weight at load time, via BMMs ... No LoRA indirection exists in this path."
- **What specifically is missing:** A correct absorbed-path low-rank correction plus a correct prefill call site. The issue proposes "**Decode:** apply the low-rank correction in absorbed mode — the delta factors onto the query and output sides (`q' += (q @ B_K) @ A`, `out += (attn @ A^T) @ B_V^T`) ... **Prefill:** the wrapped module must actually be invoked, with correct punica routing." The related asset is the sibling issue whose real title is "[Bug]: LoRA on MLA kv_b_proj is silently never applied -- decode bypasses it via absorption, prefill via a stale pre-wrap reference".
- **Evidence:** READ BODY

### B58. Can attention run directly on low-rank/Tucker factors without materializing the KV cache at decode?
- **WHO is still asking:** the JoLT authors, in their own systems-direction statement (arXiv 2607.12550)
- **URL:** https://arxiv.org/abs/2607.12550
- **Verbatim quote of the asking:** "Decode reconstructs the compressed cache from its Tucker factors at each step, an arithmetic cost that today's attention kernels are not built to absorb; a fused kernel that runs attention directly on the factors, without materializing the cache, is the natural route to decode parity and the main systems direction we leave open."
- **What specifically is missing:** A fused decode kernel that consumes the factored representation in place of a materialized cache. The paper reports only a compression-time speedup and explicitly leaves decode parity unbuilt, so the compression win is not yet a serving win.
- **Evidence:** READ BODY

### B59. Does the absorbed (latent-gather) MLA form work at full model depth, and does it beat Megatron-Core head-to-head?
- **WHO is still asking:** the LAGA authors, in their own limitations section (arXiv 2607.17644)
- **URL:** https://arxiv.org/abs/2607.17644
- **Verbatim quote of the asking:** "Prototype scale. Comm/memory are measured on a single attention layer and throughput on a 4-layer stack at V3 head dimensions, not an end-to-end DeepSeek-V3-scale (61-layer) run. ... however, end-to-end wall-clock and MFU at full model depth (where attention's share of total compute is smaller) remain future work." … "No direct MCore comparison. ... a head-to-head against MCore on matched hardware is left to future work (MCore runs on CUDA, our prototype on NPU)."
- **What specifically is missing:** Full-depth end-to-end wall-clock and MFU, plus a same-hardware comparison against Megatron-Core's explicit MLA path. The same section also leaves the fused up-projection kernel open: "Our fused attention replaces the score-matrix computation but not Laga's local per-head up-projection einsums ... Future work."
- **Evidence:** READ BODY

### B60. Why is cross-layer index/top-k reuse (`dsa_layer_skips_topk`) not wired to DeepSeek-V4 in SGLang?
- **WHO is still asking:** SGLang issue #36083 (OPEN)
- **URL:** https://github.com/sgl-project/sglang/issues/36083
- **Verbatim quote of the asking:** "`dsa_layer_skips_topk` (`python/sglang/srt/configs/model_config.py:200`) implements cross-layer top-k reuse for the DSA family, but it is not wired to DeepSeek-V4: - the function asserts `is_deepseek_dsa(config)`, whose architecture allow-list (`model_config.py:106`) does not include `DeepseekV4ForCausalLM`; - all call sites are in `models/deepseek_v2.py` ...; `models/deepseek_v4.py` never calls it. So `index_topk_freq` / `index_skip_topk_offset` / `index_topk_pattern` are effectively no-ops for V4 today."
- **What specifically is missing:** A V4 branch in `dsa_layer_skips_topk` plus call sites in `deepseek_v4.py`, which the requester believes is the whole change. The production delta measured over 2,482 real requests on 8×Hopper TP8+EP is reported as cold prefill 6,356 → 8,074 tok/s (+27%) and median TTFT 41.1 s → 26.1 s (−36%) for >150K effective input. The page badge reads "Open".
- **Evidence:** READ BODY

### B61. Should SpectralQuant-style spectral low-rank key correction be added as a vLLM KV compression backend?
- **WHO is still asking:** vLLM issue #43475 (OPEN)
- **URL:** https://github.com/vllm-project/vllm/issues/43475
- **Verbatim quote of the asking:** "I am currently **reproducing the results on an RTX 4090** before proposing any implementation. I want to verify the core cosine similarity improvement holds on consumer hardware before going further." … "**SpectralQuant** is a KV cache compression method that improves on TurboQuant (merged in #38479)".
- **What specifically is missing:** Independent reproduction, a paper link ("arXiv pending"), and an implementation design ("Draft implementation design (calibration storage, Triton kernel approach)"). The page badge reads "Open", so nothing about the method has been verified inside the engine yet.
- **Evidence:** READ BODY

### B62. Can a smaller unconstrained set of KV pairs be *learned* (rather than selected or merged) to preserve attention behaviour?
- **WHO is still asking:** the KVSculpt authors, positioning against both eviction and merging (arXiv 2603.27819)
- **URL:** https://arxiv.org/abs/2603.27819
- **Verbatim quote of the asking:** "Along the sequence-length dimension, existing methods range from pure eviction -- selecting which KV pairs to keep -- to merging, which combines similar pairs into fewer ones. Both remain anchored to the original cache entries. We propose KVSculpt, which moves to the other end of this spectrum: instead of selecting or combining original pairs, we optimize a smaller set of unconstrained KV pairs in continuous embedding space to preserve each layer's attention behavior."
- **What specifically is missing:** Scale and deployability. The reported evaluation is "On Qwen2.5-1.5B-Instruct with 2048-token contexts ... across compression ratios r in {0.3, 0.5, 0.7}", and the method is optimisation-at-compression-time rather than an inference-time primitive; no engine integration is claimed on the retrieved page.
- **Evidence:** READ BODY

### B63. A stable KV Store representation independent of an endpoint's local parallel layout, and a P/D transfer contract across different layouts
- **WHO is still asking:** `z-zanez` — vLLM tracking issue #55598 (OPEN, opened 2026-09-06) and a vLLM contributor RFC #56123 (OPEN, opened 2026-09-09)
- **URL:** https://github.com/vllm-project/vllm/issues/55598 and https://github.com/vllm-project/vllm/issues/56123
- **Verbatim quote of the asking:** "The long-term goal is a stable Store representation that is independent of an endpoint's local TP layout. Compatible Prefill and Decode instances should be able to load and extend the same prefix through one Store pool, including Decode writeback and deployments with more than one Prefill TP size. This issue tracks heterogeneous-TP sharing in MooncakeStoreConnector , from Full Attention to hybrid Attention + Mamba/GDN caches." … (title) "[RFC]: P/D transfer contract across different parallel layouts with cp interleave size".
- **What specifically is missing:** A canonical Store shard representation for non-divisible TP combinations — the planned follow-up states that combinations "where Store TP % local TP != 0" currently "fall back to the existing rank-local naming" — and a defined contract for KV transfer when P and D use different context-parallel interleave sizes. Landing hybrid Attention + Mamba/GDN Store sharing is itself still pending on PR #54307, which is open.
- **Evidence:** READ BODY (#55598); TITLE ONLY (#56123 — title from the issue list, body not read)

### B64. Race-free KV-event endpoint discovery across data-parallel ranks
- **WHO is still asking:** `touch869` — vLLM feature request #55679 (OPEN, opened 2026-09-07)
- **URL:** https://github.com/vllm-project/vllm/issues/55679
- **Verbatim quote of the asking:** "Today a consumer of KV events across data-parallel ranks has no race-free way to know where each rank's publisher landed and \"There is no Python/HTTP query surface.\"" … "ZmqEventPublisher._socket_setup() binds the socket but never reads back the assigned port ( zmq.LAST_ENDPOINT ), so with :0 the OS-assigned port is silently disc…"
- **What specifically is missing:** Bind-time ephemeral port allocation (`tcp://*:0` currently "degenerates to literal ports 0,1,2,... — rank ≥ 1 binds privileged/invalid ports"), endpoint readback via `zmq.LAST_ENDPOINT`, and a discovery API. The proposal is "a read-only GET /kv_event_sources route (mounted unconditionally, like the lora/tokenize routers — not behind VLLM_SERVER_DEV_MODE ) plus an AsyncLLM.get_kv_event_sources() method".
- **Evidence:** READ BODY

### B65. RDMA-capable NIXL OBJ backend for the KV offload secondary tier
- **WHO is still asking:** `tverma-ps` — vLLM feature request #55855 (OPEN, opened 2026-09-08, label `kv-connector`)
- **URL:** https://github.com/vllm-project/vllm/issues/55855
- **Verbatim quote of the asking:** "I would like vLLM to extend this existing OBJ tier so it can use accelerated NIXL OBJ backends that support CPU DRAM registration and RDMA-capable object transfers."
- **What specifically is missing:** The accelerated-backend config keys (`accelerated`, `use_virtual_addressing`, `req_checksum`, `resp_checksum`, `backend_params`, `max_dram_registration_bytes`) are proposed but absent from the documented schema; the documented OBJ fields are still only `bucket`, `endpoint_override`, `scheme`, credentials, `region` and `ca_bundle`.
- **Evidence:** READ BODY

### B66. Auto-calibrating the router's KV block size from KV events
- **WHO is still asking:** `victor-goubet-h` — NVIDIA Dynamo issue #13904 (OPEN, opened 2026-08-27, assignee `tmonty12`, label `router`)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/13904
- **Verbatim quote of the asking:** "Let the standalone epp learn the kv event block size by itself instead of requiring the exact value in DYN_KV_CACHE_BLOCK_SIZE . The BlockStored events already carry the block size so the epp could simply adopt the main-attention block size from the first event it receives, and keep the env var as an explicit override." … "When the configured value doesn't match, the router drops every single event: WARN zmq_wire::convert: Block not published. Block size must be 16 tokens to be published. Block size is: 2096 and kv-aware routing silently degrades to load-only."
- **What specifically is missing:** A way for the router to derive block size without a statically configured value. The reporter notes the standalone epp "has no engine handle", so the value can only come from "static per model values read from the engine boot logs", which is "fragile: the value shifts with vllm versions, attention backend and cache dtype changes, and nothing tells you when it becomes wrong".
- **Evidence:** READ BODY

### B67. Honoring KV-event storage tiers in the SGLang router's cache-aware tree
- **WHO is still asking:** `Kangyan-Zhou` (SGLang collaborator) — open PR series 1/4–4/4 (#39108, #39109, #39110, #39111), opened 2026-09-11, plus #39167 and #39168
- **URL:** https://github.com/sgl-project/sglang/pull/39108
- **Verbatim quote of the asking (APPROXIMATE — the evidence file's tail substitutes "[node]"; the corrected reading of the page is used here):** "An engine running a hierarchical cache publishes tier-tagged KV events: a CPU_PINNED BlockStored when a block's host backup lands, a GPU BlockRemoved when its device copy is demoted, a CPU_PINNED BlockRemoved when the host copy goes. The router's in-process radix tree applies every BlockRemoved as a full removal: For every node carrying any hash in `block_hashes`, drop `worker` from that node's worker set."
- **What specifically is missing:** Tier-awareness in the router: a device-tier removal and a host-tier removal are currently applied identically, so the tier information HiCache publishes is discarded and cache-aware routing cannot tell "still in host memory" from "gone". The companion PRs — `/metrics` exposure (2/4), real-GPU e2e coverage (3/4), Grafana row (4/4) — are all open, as are the tree-sharding and queue-limit follow-ups.
- **Evidence:** APPROXIMATE (verifier-demoted from READ BODY; surrounding sentences are exact, the elided tail is not a continuous verbatim excerpt)

### B68. NIXL transfer descriptor explosion on MNNVL-class hardware
- **WHO is still asking:** vLLM performance report issue #55434 (OPEN, opened 2026-09-05)
- **URL:** https://github.com/vllm-project/vllm/issues/55434
- **Verbatim quote of the asking:** "[Performance]: GLM-5.3 P/D on GB200 — NIXL issues up to ~112k KV descriptors per rank-transfer, making MNNVL/cuda_ipc slower than RDMA on this workload"
- **What specifically is missing:** A descriptor-batching or coalescing scheme so that many-request P/D transfers do not degrade below RDMA on MNNVL hardware. The same class of gap is acknowledged in-tree by a code comment: "# TODO (NickLucche) D2H<>H2D ops could benefit from coalescing io across groups".
- **Evidence:** TITLE ONLY for the issue body (title and URL from GitHub search results); the TODO quote is READ BODY from the in-tree source file

### B69. How should a single long MLA prefill that carries existing context be scheduled chunk-by-chunk?
- **WHO is still asking:** `LucasWilkinson` — vLLM RFC #50497 `[RFC]: Per-request scheduling for MLA chunked context (prefill)`
- **URL:** https://github.com/vllm-project/vllm/issues/50497
- **Verbatim quote of the asking:** "MLA prefill with existing context gathers paged latent KV into a fixed workspace, up-projects it, attends, and merges the partial into the running output. The schedule is **batch-column shaped** — the workspace is divided evenly across all prefills, and every iteration processes the same context window of **every** prefill (`build_mla_chunked_context_metadata()`, `vllm/model_executor/layers/attention/[mla_attention.py]...)"
- **What specifically is missing:** The scheduling policy itself — whether the workspace should be divided per-request rather than per-batch-column, and what the fairness/TTFT contract is when one request carries a 1M-token context and others are short. The RFC is marked closed as completed (`stateReason: COMPLETED`, created 2026-07-31T00:49:21Z), but no landed flag or default encodes a per-request MLA chunked-context schedule, so the bookkeeping closure precedes the capability.
- **Evidence:** READ BODY

### B70. Can the DSA indexer and sparse MLA use *different* context-parallel groups?
- **WHO is still asking:** a vLLM contributor — RFC #53684 `[RFC]: Allow the DSA Indexer to Use Context Parallelism Independently` (OPEN, created 2026-08-25)
- **URL:** https://github.com/vllm-project/vllm/issues/53684
- **Verbatim quote of the asking:** "vLLM currently derives both stages' parallel rank, group, and world size from the same Decode Context Parallelism (DCP) configuration. Enabling DCP for the Indexer therefore also enables it for Sparse MLA. The two stages do not benefit equally from this coupling." … "The Indexer obtains substantially higher CP2 speedup than Sparse MLA across most long-context Q>1 shapes, while both stages regress at Q=1 because communication and launch overheads dominate."
- **What specifically is missing:** An architectural capability, not a bug fix. The RFC says so itself: "This RFC requests architectural feedback on allowing the DSA Indexer and Sparse MLA to use different parallel contexts. It does not propose a complete scheduling policy or optimized cross-rank communication system."
- **Evidence:** READ BODY

### B71. What owns KV memory for a request whose context never ends (streaming video, 1M-token sessions)?
- **WHO is still asking:** a vLLM contributor — RFC #51948 `[RFC]: Bounded-memory video sessions — KV retention for long-running streaming-input requests` (OPEN), plus `nvbfalk`'s primitive PR #53871 (OPEN)
- **URL:** https://github.com/vllm-project/vllm/issues/51948 and https://github.com/vllm-project/vllm/pull/53871
- **Verbatim quote of the asking:** "Sessions exist; memory management for sessions does not." … "One long-lived request that never ends. Two resources then grow without bound: GPU memory, since everything the model has seen stays resident for the life of the request (video is token-hungry - an hour at 2 fps is on the order of a million tokens of context); and sequence length, since the stream crosses the model's trained horizon within minutes to hours and output quality degrades past it."
- **What specifically is missing:** A KV retention/eviction policy for live sessions with unbounded context, and preemption of idle sessions — the RFC quotes vLLM's own streaming-input blog back at the project: "Currently, vLLM will not preempt \"idle\" streaming input sessions - this behaviour will be improved in a future update." Only the primitive has a PR, and its scope is explicitly limited: "This PR adds the missing operation, with no policy attached and no consumer yet in-tree — the retention policy that uses it follows in the next PR of this series (RFC #51948 )."
- **Evidence:** READ BODY

### B72. Can sparse-attention self-speculative decoding (StreamingLLM drafter) be upstreamed into vLLM?
- **WHO is still asking:** an author of the Vegas paper — vLLM RFC #47351 (OPEN, created 2026-07-01), with a runnable implementation offered as two PRs
- **URL:** https://github.com/vllm-project/vllm/issues/47351
- **Verbatim quote of the asking:** "vLLM's spec-decode framework currently covers ngram, Eagle, Medusa, and MTP. These proposers either require a draft model or rely on n-gram matching. None supports a same-model, sparse-KV draft. This RFC adds one." … "Measured as accumulated decoding throughput on one H100 with Qwen3-8B, replaying AIME 2025 prompts 32 times: vanilla vLLM reaches ~1,000 tokens/s; the StreamingLLM drafter ( num_speculative_tokens=4 ) reaches ~1,600 tokens/s; Vegas ( num_speculative_tokens=6 ) reaches ~1,850 tokens/s."
- **What specifically is missing:** The upstream `method="sparse_attn"` proposer, the `SparseAttnProposer` class and the `sparse_attn_algorithm` knob. The research half is closed — the referenced paper (arXiv:2602.07223) is published — while the engine-integration half is open with no merge.
- **Evidence:** READ BODY

### B73. Is `--enable-chunked-prefill` + `--enable-prefix-caching` + a 65536-token batch budget actually correct at long context?
- **WHO is still asking:** a vLLM user — issue #46715 (OPEN, created 2026-06-25, no `stateReason`)
- **URL:** https://github.com/vllm-project/vllm/issues/46715
- **Verbatim quote of the asking:** "[Bug]: When the startup parameter --max-model-len auto --enable-chunked-prefill -- enable-prefix-caching --max-num-batched-tokens 65536\", garbled characters are displayed in the output of the GLM-5.1-NVFP4 model deployed using the VLLM Docker image."
- **What specifically is missing:** A root cause. All three flags exist and chunked prefill is default-on, so this is a correctness report under exactly the long-context configuration the gap map is about; whether the fault lies in long-prefill chunking, prefix-cache block reuse or the NVFP4 path is unresolved on the retrieved page.
- **Evidence:** READ BODY

### B74. Semi-structured N:M sparse KV cache (hardware sparse tensor cores) is not in vLLM
- **WHO is still asking:** a feature requester — vLLM issue #49630 `[Feature]: Integration of HieraSparse N:M Semi-Structured Sparse KV Cache Attention` (OPEN, created 2026-07-23)
- **URL:** https://github.com/vllm-project/vllm/issues/49630
- **Verbatim quote of the asking:** "Currently, vLLM lacks native support for semi-structured N:M sparse KV cache acceleration using hardware sparse tensor cores." … "Coarse Token Skipping (e.g., SnapKV, H2O): Skips entire tokens, which often leads to severe accuracy degradation in complex long-context tasks compared to fine-grained element-level $N:M$ sparsity."
- **What specifically is missing:** A backend. The proposal asks for "native integration or an experimental backend feature based on the HieraSparse architecture" with "Hierarchical Block + Element Masking: Keeps critical blocks (attention sinks, local context windows) dense while applying $N:M$ semi-structured pruning (e.g., 2:4) to non-critical blocks."
- **Evidence:** READ BODY

### B75. Which request should be preempted, and what does the wrong choice cost?
- **WHO is still asking:** `shivasathishs-rp` — vLLM RFC #54644 (OPEN), with an interim implementation in open PR #53723
- **URL:** https://github.com/vllm-project/vllm/issues/54644
- **Verbatim quote of the asking:** "Neither of the current policies takes that into account. FCFS evicts the last request that was admitted, using self.running.pop() . PRIORITY evicts max(running, key=(priority, arrival_time)) . Both ignore how much work a request has already done, so the scheduler can evict a request that has already prefilled several thousand tokens in order to admit a tiny one, and all of that computed KV is discarded." … "The short version is that under KV pressure the tax is large, reaching about 79 percent in a forced offline sweep and 10 to 23 percent in realistic online serving".
- **What specifically is missing:** A durable, pluggable victim-selection extension point with a fairness contract: "The second part is the actual ask. I would like to promote victim selection to a pluggable extension point under #51608 , which is what its phase 5 anticipates." The RFC is open with no close event and the interim `--preemption-victim` implementation is unmerged.
- **Evidence:** READ BODY

### B76. Should the KV parked for a not-yet-fetching peer expire on a fixed timer?
- **WHO is still asking:** `nilig` — vLLM issue #53128 (OPEN, created 2026-08-20, no close event)
- **URL:** https://github.com/vllm-project/vllm/issues/53128
- **Verbatim quote of the asking:** "A PD Multi Tier producer discards stored KV 60 s after submit_store when no FetchMsg has bound the kv_request_id yet, so a consumer whose fetch is issued later finds nothing, burns its own 30 s load deadline, and recomputes the prompt. _UNBOUND_STORE_TIMEOUT_S = 60.0 ( vllm/v1/kv_offload/tiering/p2p/manager.py ) is a fixed timer counted from store time." … "The p2p tier has no equivalent - no renewal, no way for the consumer to signal intent, and no config knob - so it cannot distinguish 'the peer is still queued' from 'the peer will never come'."
- **What specifically is missing:** A consumer-liveness-tied lease for the p2p secondary tier instead of a fixed 60-second TTL. The NIXL path already has a lease mechanism; the p2p tier has no renewal, no intent signal and no config knob, so a live-but-queued consumer silently loses its parked KV.
- **Evidence:** READ BODY

### B77. Session/agent-lifetime KV hints — TTL, session identity and router-initiated cache management
- **WHO is still asking:** `FermatGo` on vLLM RFC #52113 (OPEN, created 2026-08-13) and `karen-sy` on vLLM RFC #51428 `KvHint` (OPEN, created 2026-08-07)
- **URL:** https://github.com/vllm-project/vllm/issues/52113 and https://github.com/vllm-project/vllm/issues/51428
- **Verbatim quote of the asking:** "Today, after a request finishes, these prefixes are ordinary free cached blocks managed by the same replacement policy. vLLM cannot distinguish a temporarily idle but valuable agent context from a prefix that will never be used again. This RFC proposes optional agent_hint metadata so an application can communicate session identity, a short retention TTL, and explicit cache management operations." … "Currently, KV cache lifecycle is managed naively, mostly via LRU: request arrives - > router chooses target worker via KV overlap/load - > target worker checks local cache hit: reuse miss: recompute or apply local offload policy. This is insufficient for multiple scenarios".
- **What specifically is missing:** An agreed hint surface and its resolution semantics — who wins when a router hint contradicts local policy, and what the contract is across workers. Both RFCs are open with no close event; a third related RFC (#48501, session-centric orchestration over typed session identity) and an implementation PR (#51384, bounded session-affinity scheduler) are also open proposals rather than shipped behaviour.
- **Evidence:** READ BODY

### B78. Per-block / per-request TTL for cached KV — the expiry concept does not exist in either engine
- **WHO is still asking:** `ajayr4j` — vLLM issue #44775 (Phase 3 explicitly deferred), and NVIDIA/TensorRT-LLM — open PR #16252
- **URL:** https://github.com/vllm-project/vllm/issues/44775 and https://github.com/NVIDIA/TensorRT-LLM/pull/16252
- **Verbatim quote of the asking:** "Phase 3 - Per-block TTL (future, explicitly out of scope for initial PR) Anthropic supports 5-minute and 1-hour TTLs per cache block. vLLM's eviction is purely LRU with no expiry concept. This requires adding TTL metadata to KVCacheBlock and modifying FreeKVCacheBlockQueue to evict on expiry. Non-trivial and intentionally deferred." … "Under memory pressure, a 500-token tool schema can get evicted in favor of a one-time long user message. There is currently no workaround for this."
- **What specifically is missing:** Any expiry concept in the block pool, and any way to express per-request retention in the disk tier. vLLM's shipped block path is LRU-only, with time-based lifetime available only as a metric rather than a control; the TensorRT-LLM disk (L3) tier with per-request retention TTL is an open PR whose design body was not retrieved.
- **Evidence:** READ BODY (#44775); TITLE ONLY (TensorRT-LLM #16252 — title and page metadata only)

### B79. Bounded long-context KV in llama.cpp via StreamingLLM-style eviction
- **WHO is still asking:** `jotrujil03` — llama.cpp PR #27583 (OPEN, created 2026-08-23), with sibling proposals #28046 and #28092 open
- **URL:** https://github.com/ggml-org/llama.cpp/pull/27583
- **Verbatim quote of the asking:** "Long-context generation quickly exhausts VRAM when retaining the entire KV cache. This PR implements StreamingLLM-style KV cache eviction via --kv-evict-sink and --kv-evict-window . By capping the physical cache at sink + recent + n_ubatch cells while allowing the logical context to grow indefinitely, both VRAM footprint and per-token decode latency remain constant over arbitrary sequence lengths." … "Wikitext-2 perplexity remains stable (~6.31 baseline vs. ~6.48 with a 1024-token window), while decode throughput stays flat across long sequences."
- **What specifically is missing:** Acceptance of the mechanism. The PR is open, disables context shifting for compatibility ("Compatibility : Disables context shifting ( get_can_shift = false )."), and the review bot flagged process problems including "AI-generated content: While code is allowed to be generated by AI, please write the PR description and commit messages on your own". The related PRs (#28046 per-entry prompt-cache context limit, and #28092) are also open, and the verifier notes #28267 in this family is a draft rather than plain open.
- **Evidence:** READ BODY

### B80. One LRU queue for all KV cache groups in the hybrid manager
- **WHO is still asking:** vLLM itself — its own design doc states the limitation
- **URL:** https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md
- **Verbatim quote of the asking:** "The second question is the cache eviction policy. For now, we use one LRU queue for all kv cache groups. The blocks are added to the LRU queue when freed, either because the request is finished or the block is out of the sliding window."
- **What specifically is missing:** A per-group eviction policy. The doc's "For now" is unchanged by any shipped artifact found, which means a sliding-window group's blocks and a full-attention group's blocks compete in the same queue even though their reuse distances and replacement costs differ.
- **Evidence:** READ BODY

### B81. KV transfer failures are not reported to the scheduler for Mooncake P2P, so PD requests hang
- **WHO is still asking:** vLLM issue #55870 (reporter on Ascend A3 with MultiConnector + MooncakeConnector) and `LH-and-FPGA` — open fix PR #56166 (submitted 2026-09-09)
- **URL:** https://github.com/vllm-project/vllm/issues/55870 and https://github.com/vllm-project/vllm/pull/56166
- **Verbatim quote of the asking:** "In the inspected implementation, discovery failure and per-request transfer errors appear to lack the scheduler-visible failure handling needed to unblock remote-KV waits."
- **What specifically is missing:** Scheduler notification of decode-side pull failures; without it, requests sit in `WAITING_FOR_REMOTE_KVS` until client timeout. The open PR enumerates the three dropping paths verbatim: "`_connect_to_prefiller_bootstrap()` does not propagate bootstrap failure to callers. `handle_new_engine_id()` bare-returns when the remote engine is unavailable, abandoning the pending `PullReqMeta` s. `process_pulling_result()` handles..." — the fix is proposed, not shipped.
- **Evidence:** READ BODY

### B82. NIXL push-mode + pipeline parallelism is refused outright for hybrid (HMA) KV layouts, and HMA still opts out of block invalidation
- **WHO is still asking:** the vLLM NixlPushConnector code itself (a hard `NotImplementedError`) plus a residual HMA guard in the NIXL worker's failure path
- **URL:** https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py
- **Verbatim quote of the asking:** "NixlPushConnector does not support pipeline_parallel_size > 1 with hybrid KV cache layouts (HMA) yet." … "NixlPushConnector consumer (decode) does not support pipeline_parallel_size > 1."
- **What specifically is missing:** Two distinct capabilities. (a) Push-mode NIXL cannot be combined with PP>1 on any hybrid model; the in-code explanation is "PP push slices regions per layer (uniform count); HMA breaks that." (b) In the failure path, block invalidation is still conditioned on `if not self._is_hma_required:` inside `_report_failed_recv`, so on HMA-required models a failed receive is recorded and its handle released but no invalid block is reported to the scheduler.
- **Evidence:** READ BODY

### B83. An RFC to transfer last-token hidden states over the connector is closed but the capability is not in `KVConnectorBase_V1`
- **WHO is still asking:** vLLM issue #31064, filed for DeepSeek-R1 on Intel Gaudi/XPU (closed COMPLETED, capability absent)
- **URL:** https://github.com/vllm-project/vllm/issues/31064
- **Verbatim quote of the asking:** "This RFC proposes an **optional, backward-compatible enhancement** to `KVConnectorBase_V1` (v1-only) to transfer **last-token hidden states** alongside KV cache under P/D disaggregation." … "prefix-prefill injections introduce large step-time spikes (hundreds of milliseconds to seconds), these spikes directly inflate TPOT and tail ITL"
- **What specifically is missing:** A general hidden-states transfer path on the base connector. The only shipped connector is `ExampleHiddenStatesConnector`, whose own docstring scopes it elsewhere: "Simple debug implementation of a HiddenStatesConnector. Simply extracts the hidden states from the kv cache and stores them to disk. Must be used in conjunction with the `extract_hidden_states` spec decoding method." That is a local-disk, speculative-decoding helper, not the P/D capability the RFC asked for.
- **Evidence:** READ BODY

### B84. Bidirectional KV transfer's block-alignment guarantee breaks on stripped thinking traces
- **WHO is still asking:** the vLLM NixlConnector Usage Guide's "Limitations" section, citing open issue #43094
- **URL:** https://docs.vllm.ai/en/latest/features/nixl_connector_usage/ and https://github.com/vllm-project/vllm/issues/43094
- **Verbatim quote of the asking:** "Reasoning models with stripped thinking traces — When using reasoning models (e.g. DeepSeek-R1) that produce thinking traces (`<think>...</think>`), D's KV blocks cover the full token sequence including thinking tokens. If the client strips thinking traces from the conversation history before sending the next turn, the prompt P receives will be missing tokens from the middle of what D generated. The block-alignment logic assumes P's prompt is a prefix of D's sequence, so pulling KV blocks from D in this case transfers cache computed for the wrong token positions, producing incorrect results."
- **What specifically is missing:** An in-engine correctness guarantee. The docs push the burden outward — "We currently assume the router is able to detect such mismatch across turns" — leaving an unspecified external router responsible for correctness. A second stated limitation, host-buffer support, "is planned for future work."
- **Evidence:** READ BODY

### B85. Aborted requests with in-flight KV transfers can take down EngineCore
- **WHO is still asking:** `root1iu` — vLLM PR #56089 `Wait for KV transfers before freeing aborted requests` (OPEN, opened 2026-09-09), fixing issue #46240
- **URL:** https://github.com/vllm-project/vllm/pull/56089 and https://github.com/vllm-project/vllm/issues/46240
- **Verbatim quote of the asking:** "When a request is aborted with both KV recv and send pending, the first completion frees the request and the second hits `assert req_id in self.requests`, taking down EngineCore."
- **What specifically is missing:** Correct teardown ordering between abort and asynchronous KV transfer completion. The fix is proposed rather than shipped — "This change tracks pending recv and send separately and frees blocks only after both complete. It covers recv -> send, send -> recv, and both in the same step." — so any abort under load still risks an engine-level crash.
- **Evidence:** READ BODY

### B86. NIXL telemetry throughput conflates overlapped transfers and double-counts across TP ranks
- **WHO is still asking:** vLLM issue #33170 (reporter), closed by the stale bot and never answered by a maintainer
- **URL:** https://github.com/vllm-project/vllm/issues/33170
- **Verbatim quote of the asking:** "here vLLM calculates the throughput by the `sum of bytes transferred /sum the xfer time`, but if there are multiple transfers overlapped, we can not simply sum each xfer time as the total duration. In real case, the total duration should be shorter if multiple transfers can be in parallel." … "it treats each rank as independent, so if total bytes_transferred is 1000MB and i have 8 ranks (tp=8), it treats each rank as 125MB and increase count by 8, instead of aggregating 1 transfer of 1000MB".
- **What specifically is missing:** A correct overlap-aware throughput definition plus per-rank aggregation. Without them the shipped metric cannot be used to size the transfer/compute overlap budget for P/D — the measurement instrument itself is the gap, and the issue was closed by automation without a technical verdict.
- **Evidence:** READ BODY
### B87. SGLang HiCache prefetch finalization is coupled to FCFS position
- **WHO is still asking:** `cfbdsirlijun-maker` — SGLang issue #32724 (OPEN since 2026-07-29)
- **URL:** https://github.com/sgl-project/sglang/issues/32724
- **Verbatim quote of the asking:** "Besides inflating TTFT and the apparent prefetch latency, the un-finalized operations also keep host memory and `prefetch_tokens_occupied` budget pinned longer than necessary, which can throttle subsequent prefetches." … "the storage IO itself took ~91 ms while the scheduler-side finalization lagged ~2.4 s behind IO completion. For the affected requests, this means TTFT can be extended by seconds even though the actual storage transfer already finished. In this case, >95% of the end-to-end prefetch latency was spent waiting for the FCFS scan, not on actual data transfer."
- **What specifically is missing:** Decoupling prefetch completion from the waiting-queue scan order, so that a finished storage transfer is usable immediately rather than when the scheduler happens to walk that request. Today completion ordering inherits FCFS position, and the cost is measured in seconds of TTFT.
- **Evidence:** READ BODY

### B88. How is anyone supposed to measure prefix-cache effectiveness, and can local versus external hits be separated?
- **WHO is still asking:** vLLM issue #52137 `[Feature]: split local/external prefix-cache hits in prompt_tokens_details` (OPEN, label `feature request`) and `SarnadAbhilash` — docs PR #53395 (OPEN since 2026-08-22)
- **URL:** https://github.com/vllm-project/vllm/issues/52137 and https://github.com/vllm-project/vllm/pull/53395
- **Verbatim quote of the asking:** "The Automatic Prefix Caching guide explains where APC helps, but does not currently tell operators how to measure cache effectiveness or distinguish token reuse from end-to-end performance gains." … the change "describes how to control request order, concurrency, and initial cache state for comparisons; recommends reporting TTFT, latency, and throughput alongside hit ratio; clarifies that decode, scheduling, eviction, routing, and block alignment can make a high hit ratio differ from the observed speedup".
- **What specifically is missing:** A shipped measurement procedure and the metric split that would separate offload-tier benefit from in-GPU prefix-cache benefit. Both artifacts are open, and the docs-only PR carries the guidance rather than the engine exposing the numbers; this is precisely the measurement that determines whether reported offload wins are mis-attributed.
- **Evidence:** READ BODY

### B89. vLLM has no support for prefix caching on pooling / encoder-only models
- **WHO is still asking:** `HsukqiLee` — vLLM PR #55159 (still open), with sibling PR #55160 closed unmerged
- **URL:** https://github.com/vllm-project/vllm/pull/55159 and https://github.com/vllm-project/vllm/pull/55160
- **Verbatim quote of the asking:** "fix: reject unsupported encoder-only prefix caching" … "Fail fast when pooling encoder-only models explicitly enable prefix caching without support."
- **What specifically is missing:** An actual implementation of prefix caching for these model classes; only the rejection path is proposed. The sibling PR that targeted the same function was closed unmerged on 2026-09-07 after a reviewer comment noted it duplicated #55159, so the surviving artifact still only fails fast.
- **Evidence:** READ BODY (patch/commit message retrieved; PR page metadata retrieved)

### B90. llama.cpp disk-based KV checkpoint offloading still unproven
- **WHO is still asking:** `strawberrymelonpanda` (CONTRIBUTOR) expressing doubt on open feature request #20697
- **URL:** https://github.com/ggml-org/llama.cpp/issues/20697
- **Verbatim quote of the asking:** "Not casting doubt on the proposal overall, but I don't know about these numbers in any sort of real-world use." … "Nothing against more options, personally, I just wonder if the performance hit of hitting disk cache really pays off and would be curious to see."
- **What specifically is missing:** Any real-world measurement of the `--cache-disk` path. The request is open and a contributor is on record asking for evidence, which means the feature's payoff is an open empirical question rather than a settled one.
- **Evidence:** READ BODY (retrieved by the delegated llama.cpp/TGI investigation)

### B91. vLLM-Omni prefix caching with hidden-state I/O — Plan A abandoned, design still open
- **WHO is still asking:** `alex-jw-brooks` (assignee), `tzhouam` and `hongzhi-gao` — vLLM-Omni issue #1184 (RFC was closed NOT_PLANNED on 2026-02-25 and then reopened on 2026-03-03)
- **URL:** https://github.com/vllm-project/vllm-omni/issues/1184
- **Verbatim quote of the asking:** "This feature is currently on hold while we prioritize more important work, including the additional_information refactor. After discussing with @ZeldaHuang, we found out that Plan A won't work." … "currently its still under design."
- **What specifically is missing:** A working design for hidden-state prefix caching. Plan A is explicitly dead, the issue was reopened rather than resolved, and only Phase-1 correctness tests have landed upstream ("First pass at hidden state prefix caching is up here: https://github.com/vllm-project/vllm-omni/pull/2164").
- **Evidence:** READ BODY

### B92. SGLang distributed KVCache system for agentic workloads — roadmap unfulfilled
- **WHO is still asking:** the SGLang project — issue #21846 `[Roadmap]: SGLang Distributed KVCache System For Agentic Workload` (OPEN since 2026-04-01, labels include `hicache`, `roadmap`, `PD Disaggregation`)
- **URL:** https://github.com/sgl-project/sglang/issues/21846
- **Verbatim quote of the asking:** "[Roadmap]: SGLang Distributed KVCache System For Agentic Workload"
- **What specifically is missing:** The roadmap's own deliverables. The same subsystem carries several open correctness/overhead reports; the evidence file claimed "at least nine", but the verifier confirmed only three of the named examples as open with matching HiCache-bug titles (#36179, #32693, #39147), so the count should be treated as unverified while the individual open bugs stand.
- **Evidence:** TITLE ONLY (title retrieved from the GitHub search-API result set; body not read)

### B93. llama.cpp `/slots restore` and disk KV checkpoints do not actually restore or reuse KV
- **WHO is still asking:** reporters on llama.cpp issue #26676 (OPEN since 2026-08-06) and issue #28194 (OPEN since 2026-09-01)
- **URL:** https://github.com/ggml-org/llama.cpp/issues/26676 and https://github.com/ggml-org/llama.cpp/issues/28194
- **Verbatim quote of the asking:** "llama-server slot KV state restore is a no-op (restore reads file, slot stays empty, cache_n=0)" … "Misc. bug: /slots restore yields no KV reuse on hybrid/recurrent and SWA models (context checkpoints are not persisted)".
- **What specifically is missing:** A working checkpoint-to-disk and restore path — the closest llama.cpp analogue to KV offloading — for plain and for hybrid/recurrent/SWA models. The second report matters for this map because the hybrid/SWA class is exactly where the target architectures sit, and context checkpoints are simply not persisted.
- **Evidence:** TITLE ONLY (titles retrieved from the GitHub search-API result set)

### B94. KV reuse is keyed without adapter identity: cross-adapter contamination in llama.cpp, and an untagged-key collision in vLLM
- **WHO is still asking:** reporters on llama.cpp issue #26207 (OPEN since 2026-07-28) and `gabe` on vLLM issue #44701 (OPEN since 2026-06-06, activity through 2026-09-04)
- **URL:** https://github.com/ggml-org/llama.cpp/issues/26207 and https://github.com/vllm-project/vllm/issues/44701
- **Verbatim quote of the asking:** "server: prompt cache is reused across requests with different per-request `lora` — output silently contaminated by the previous adapter" … "Because these fields are not tagged, a LoRA request with lora_name == \"COLLIDE_SALT\" and no cache_salt can produce the same first-block hash as a base-model request with cache_salt == \"COLLIDE_SALT\" and no LoRA, for identical prompt tokens. I confirmed this dynamically on a single GCP A100. The base request with cache_salt=\"COLLIDE_SALT\" received a local prefix-cache hit for a block produced by the LoRA request."
- **What specifically is missing:** Cache keying that accounts for the active adapter in both engines, and domain separation (type-tagging) of the vLLM extra-key tuple. Both fail silently — the llama.cpp side returns another adapter's cached state, and the vLLM side needs only a colliding string; the vLLM report was reproduced on a single A100, so it is directly testable on a target rig.
- **Evidence:** READ BODY (vLLM #44701); TITLE ONLY (llama.cpp #26207)

### B95. How should the retired KVBM offload-simulation path be modelled in Dynamo's G1↔G2 simulation?
- **WHO is still asking:** `dreamtalen` (issue author) — Dynamo issue #13794 (OPEN)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/13794
- **Verbatim quote of the asking:** "That KVBM-based offload simulation was not carried into the standalone AISimulate core, and the old KVBM sweep fields have been removed. New G1↔G2 simulation should model each framework's native behavior directly rather than reproduce the retired KVBM path."
- **What specifically is missing:** A replacement simulation model for KV-offload behaviour in Dynamo's G1↔G2 path, now that the KVBM-based simulation and its sweep fields are gone. The verifier's correction matters here: this text is the opening post of an OPEN issue by its author, not a maintainer closure statement, so the question is live rather than settled. The evidence file's "Closure status" field implied a closure that does not exist.
- **Evidence:** READ BODY (per the verifier; the row was reclassified out of ABANDONED)

### B96. Prefill Context Parallel (PCP) coverage beyond FlashInfer/GQA — no timeline, community help requested
- **WHO is still asking:** `pisceskkk` (RFC author), `zhenwenqi2024`, `aarondou` — vLLM issue #25749 `[RFC]: Support Prefill Context Parallel (PCP)` (OPEN, created 2025-09-26, 13 comments, last maintainer replies 2025-10-15)
- **URL:** https://github.com/vllm-project/vllm/issues/25749
- **Verbatim quote of the asking:** "Adaptations for other backends are still in the planning stage. Due to time and resource constraints from other priorities, we cannot guarantee a timeline for these additional adaptations. We welcome support from the community for the adaptation of this feature."
- **What specifically is missing:** PCP for backends other than FlashInfer (MLA and others), PCP for chunked prefill, PCP for prefix caching, PCP for MTP, PCP for CUDA full graphs, and a Ring-CP style attention algorithm. All PCP/DCP paths are defined only for a multi-rank group; nothing in the RFC defines a single-GPU mode.
- **Evidence:** READ BODY

### B97. Prefill Context Parallel for Qwen3.5 hybrid attention (full attention + GatedDeltaNet linear attention)
- **WHO is still asking:** `Yancey0623` — vLLM issue #37995 `[RFC]: Prefill Context Parallel for Qwen3.5 Hybrid Attention` (OPEN, created 2026-03-24; auto-marked stale 2026-06-27 but not closed)
- **URL:** https://github.com/vllm-project/vllm/issues/37995
- **Verbatim quote of the asking:** "Long-context prefill is a major bottleneck for LLM inference: 1. **Quadratic attention complexity**: Full attention is O(N^2) in sequence length, making 256K+ contexts extremely slow 2. **Memory bandwidth** : Large KV caches stress memory subsystem — Tensor Parallelism (TP) helps by distributing model weights, but each GPU still processes the full sequence." … "Linear attention layers maintain a recurrent state that depends on all previous tokens, so naively splitting the sequence would break correctness."
- **What specifically is missing:** Sequence parallelism for the GatedDeltaNet/linear-attention layers, plus an implementation PR. The RFC is design-only and no merged code exists, so hybrid Qwen3.5 cannot use context-parallel prefill.
- **Evidence:** READ BODY

### B98. Decode Context Parallelism produces output drift and gibberish, and DCP is not in default CI
- **WHO is still asking:** `ehfd` (reporter, 8× A100-SXM4-80GB single node), `cjackal`, `Yancey0623` — vLLM issue #41623 (OPEN; closed and then reopened, `state_reason: reopened`), plus issue #54300 on GLM-5.3 + DCP across vLLM 0.28/0.29
- **URL:** https://github.com/vllm-project/vllm/issues/41623 and https://github.com/vllm-project/vllm/issues/54300
- **Verbatim quote of the asking (maintainer-side diagnosis by cjackal):** "The symptom described here sounds similar to what I have experienced with qwen3MoE before. Currently DCP is quite fragile, partly because DCP is not tested in CI by default. The only unit test for DCP is `tests/distributed/test_context_parallel.py`, which is run by two optional distributed test pipelines (not run by default for most PRs). And `test_context_parallel.py` does not cover most recent architectures like kimi-2.5, the two model checkpoints tested are dsv2 and qwen2.5 which are bf16-only and not representative of modern model architectures."
- **What specifically is missing:** Root cause plus CI coverage of modern architectures and backends. The second report is worse than drift: "[Regression 0.27→0.28+]: GlmMoeDsa (GLM-5.3) + decode-context-parallel: crashes on 0.28.0, silently returns random tokens on 0.29.0" on 8× B200, with no merged fix linked. The reporter's own suspect list names "[Kernel] Pack output and LSE in DCP A2A (#41160)" as a candidate.
- **Evidence:** READ BODY

### B99. FlashInfer MLA decode workspace buffer overflow under decode-context-parallel
- **WHO is still asking:** vLLM issue #50781 (OPEN, created 2026-08-02) with accompanying PR #50791 `[Bugfix][Attention] Size FlashInfer sparse MLA workspace for decode-context-parallel` (OPEN, created 2026-08-03)
- **URL:** https://github.com/vllm-project/vllm/issues/50781 and https://github.com/vllm-project/vllm/pull/50791
- **Verbatim quote of the asking:** "FlashInfer MLA decode workspace buffer overflow with decode-context-parallel"
- **What specifically is missing:** A merged fix sizing the sparse-MLA workspace for DCP. The fix PR is open with `merged_at: null`, so DCP plus FlashInfer sparse MLA remains a buffer-overflow configuration rather than a supported one.
- **Evidence:** TITLE ONLY (titles and `state`/`merged_at` fields from the retrieved GitHub search API JSON; individual pages not read)

### B100. Context-parallel prefill for DeepSeek-V4 hybrid KV + MegaMoE still unmerged
- **WHO is still asking:** vLLM PR #43809 `[DeepSeekV4][PCP] Enable context-parallel prefill for hybrid KV and MegaMoE` (OPEN, created 2026-05-27, 7 comments)
- **URL:** https://github.com/vllm-project/vllm/pull/43809
- **Verbatim quote of the asking:** "[DeepSeekV4][PCP] Enable context-parallel prefill for hybrid KV and MegaMoE"
- **What specifically is missing:** PCP support for hybrid-KV architectures and MegaMoE, i.e. the newest architectures still do not have working context-parallel prefill. The PR has been open since May 2026 with no merge.
- **Evidence:** TITLE ONLY (title and `state` field from the retrieved GitHub search API JSON; page not opened)

### B101. Context-parallel error guidance and PCP/MTP remediation hints still unmerged
- **WHO is still asking:** vLLM PRs #52075 `[Bugfix] Improve context-parallel backend error guidance` (OPEN, created 2026-08-13) and #46598 `[Misc] Add remediation hints to MTP and PCP context-parallel errors` (OPEN, created 2026-06-24)
- **URL:** https://github.com/vllm-project/vllm/pull/52075 and https://github.com/vllm-project/vllm/pull/46598
- **Verbatim quote of the asking:** "[Bugfix] Improve context-parallel backend error guidance" … "[Misc] Add remediation hints to MTP and PCP context-parallel errors"
- **What specifically is missing:** Both are error-path fixes, which corroborates that launching CP/DCP in unsupported configurations remains a common, poorly-diagnosed failure mode. Neither is merged, so users still get opaque failures when a combination is unsupported.
- **Evidence:** TITLE ONLY (titles and `state` fields from the retrieved GitHub search API JSON; pages not opened)

### B102. SGLang's own DCP + Helix Parallelism roadmap — "half of Helix is unstarted"
- **WHO is still asking:** `thanhhao98` (roadmap owner) — SGLang issue #29736 `[Roadmap][DCP] Decode Context Parallelism & Helix Parallelism (2026 Q3)` (OPEN, created 2026-06-30, refreshed 2026-07-30)
- **URL:** https://github.com/sgl-project/sglang/issues/29736
- **Verbatim quote of the asking:** "However, the current support only covers a few attention backends, half of Helix is unstarted, and several combinations are unguarded. To generalize the usage of DCP, we need to work on following features and tasks"
- **What specifically is missing:** FA3 MLA decode under DCP; `flashmla` + FP8 KV (an assert `dcp_world_size == 1` still blocks it); Triton MLA; the `trtllm_mla` decode path; DSA DCP for DeepSeek-V3.2/GLM-5.x; decoupled attention/FFN (TPA vs KVP) parallelism; and reconciliation of three overlapping A2A follow-ups. The roadmap is also the DCP link cited from SGLang's published DCP documentation.
- **Evidence:** READ BODY

### B103. Does prefix caching actually pay off when tool calling is in the loop, and is anyone testing the combination?
- **WHO is still asking:** `guanxingithub` — vLLM PR #52629 `[Test] Add a prefix-caching x tool-calling gate` (OPEN)
- **URL:** https://github.com/vllm-project/vllm/pull/52629
- **Verbatim quote of the asking:** "Agentic deployments run prefix caching and tool calling together — repeated system prompts and tool definitions are exactly the prefix a cache is meant to reuse — yet no test exercises the combination: the per-parser tests never boot an engine, and the tool-calling e2e tests never enable prefix caching. A regression that only appears when the cache engages is invisible today."
- **What specifically is missing:** An engine-level gate that asserts both schema-conformant tool calls and that the prefix cache was actually exercised. The PR adds `tests/tool_use/test_prefix_cache_tool_calling.py` asserting "`vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` non-zero, and zero in the control arm", and notes "Test-only; requires a GPU and a small model, so it suits a nightly or opt-in tier rather than the default suite".
- **Evidence:** READ BODY

### B104. How do you benchmark prefix-cache hit rate reproducibly for real agent workloads?
- **WHO is still asking:** `sammysun0711` (with Richardyu114, FionaZZ92) — SGLang PR #37359 `benchmark: add reproducible prefix-cache hit-rate evaluation` (OPEN)
- **URL:** https://github.com/sgl-project/sglang/pull/37359
- **Verbatim quote of the asking:** "Prefix-hit-aware benchmarking is essential for evaluating real agent workloads, where repeated system prompts, tool schemas, and conversation context make high cache-hit rates. To accurately simulate production workload, this PR introduces a benchmark that measures inference performance under controlled prefix-cache hit rates."
- **What specifically is missing:** A deterministic harness in the shipping benchmark tool. The PR states the gap verbatim: "`sglang.bench_serving` can generate shared-prefix requests and report cached tokens, but it previously lacked a deterministic way to benchmark a requested cache-hit rate. In particular, ordinary warmup is followed by cache flushing, so measured requests can race to populate the prefix cache unless prefixes are primed in a separate phase."
- **Evidence:** READ BODY

### B105. How should hybrid Mamba/GDN cached state be thinned under LRU pressure?
- **WHO is still asking:** `alphabetc1` — SGLang draft PR #38000 `[Mamba] Thin cached states by coverage instead of evicting the LRU tail` (DRAFT)
- **URL:** https://github.com/sgl-project/sglang/pull/38000
- **Verbatim quote of the asking:** "[Mamba] Thin cached states by coverage instead of evicting the LRU tail"
- **What specifically is missing:** A chosen thinning policy for branching shared prefixes. The PR motivation gives the failure mode for a single GPU: "On a hybrid Mamba/GDN model every chunked-prefill boundary donates one Mamba state into the radix tree, so a long prompt pushes len / chunked_prefill_size states through the Mamba LRU. The LRU evicts a path's states shallow-first (they were inserted first), which is the inverse of what a branch mat…" — and the artifact proposing the fix is still a draft.
- **Evidence:** READ BODY

### B106. Nobody measures repeated compaction over the horizon an agent actually runs
- **WHO is still asking:** the authors of "What to Keep, What to Forget: A Rate-Distortion View of Memory Compaction in LLMs and Agents" (arXiv 2607.08032)
- **URL:** https://arxiv.org/abs/2607.08032
- **Verbatim quote of the asking:** "And while compression is measured carefully on single-turn long context, the repeated compaction that agents actually perform is almost never measured, and no benchmark holds one budget axis across all the layers at once."
- **What specifically is missing:** A benchmark that puts "KV eviction, quantization, prompt compression, and summarization on one budget axis and measures error accumulation under repeated compaction". The paper's own Section 15 "Open Problems and Research Agenda" adds that the formalism "is not a set of proven laws: distortion surrogates are not additive and composition order matters".
- **Evidence:** READ BODY

### B107. EAGLE/MTP prefix-cache last-block drop forces recomputation on every hit
- **WHO is still asking:** vLLM issue #53670, with related tracking issue #51771
- **URL:** https://github.com/vllm-project/vllm/issues/53670 and https://github.com/vllm-project/vllm/issues/51771
- **Verbatim quote of the asking:** "EAGLE/MTP prefix-cache last-block drop causes a 1,648-token recompute per hit on one hybrid Qwen3.8 GDN layout"
- **What specifically is missing:** Correct handling of the interaction between speculative-decoding draft caches and prefix-cache block reuse, where a cache hit still costs a large recompute. Related tracking issue #51771 frames the same area as untested: "EAGLE/MTP block drop + prefix caching is untested for hybrid models with ≥3 attention groups".
- **Evidence:** TITLE ONLY (titles from the GitHub search-API result list; bodies not retrieved)

### B108. Prefix caching is still not supported in vLLM's batch-invariant mode
- **WHO is still asking:** vLLM maintainers on tracking issue #27433 (opened by `brianosaurus`, 2025-10-23, OPEN) and `littlecircle0730` — PR #46592 `Feat/invariant with prefix cache` (OPEN, opened 2026-06-24, still awaiting code-owner review)
- **URL:** https://github.com/vllm-project/vllm/issues/27433 and https://github.com/vllm-project/vllm/pull/46592
- **Verbatim quote of the asking:** "🙋Help needed / Nice to have: / Prefix caching support / AMD support ... / Speculative decoding support (this might be hard)" … "Prefix caching is an important performance path for real workloads with shared long prefixes, but it introduces a new source of variation: the same prompt can resume from different prefix-cache hit lengths depending on cache state. In batch-invariant mode, those different hit lengths should not cause the remaining prefill work to be scheduled with arbitrary chunk boundaries."
- **What specifically is missing:** Maintainer review and a merge of the canonical-chunking path (`VLLM_BATCH_INVARIANT_CANONICAL_PREFILL_CHUNK_BLOCKS=0` means auto). The tracker lists prefix caching as an unchecked help-wanted TODO, and the implementing PR has waited since June 2026 — so the capability is designed but not shipped.
- **Evidence:** READ BODY

### B109. Non-deterministic greedy output in vLLM V1 — a base-scheduler KV block lifecycle bug, and an issue never filled in
- **WHO is still asking:** `Yunzez` — vLLM issue #39146 (OPEN, created 2026-04-07) and an unnamed reporter on vLLM issue #39389 (OPEN, created 2026-04-09)
- **URL:** https://github.com/vllm-project/vllm/issues/39146 and https://github.com/vllm-project/vllm/issues/39389
- **Verbatim quote of the asking:** "We fuzzed with prefix-cache but forgot to fuzz without it 😅. But when testing `--speculative-config`, we found a KV block corruption bug that reproduces with **no `--enable-prefix-caching`**. Identical prompts at `temperature=0` produce **_completely_** different output sequences across runs, confirmed **10/10** on three independent traces." … "SO, these findings point to a separate block lifecycle bug in the base scheduler's non-APC path."
- **What specifically is missing:** Identification and a fix for the base-scheduler block lifecycle bug; the only candidate TOCTOU patch (PR #37164) is still open and by the reporter's own argument "should not affect the base vllm". The second issue's body reads only "NA" — the verifier's correction is that the page nevertheless carries a repro sketch and substantive triage, so it is evidence that the question keeps being asked rather than an empty report.
- **Evidence:** READ BODY

### B110. `--deterministic-prefix-caching` for reproducible prefill on ROCm is an unmerged proposal
- **WHO is still asking:** `gigamonkeyx` — vLLM PR #40179 `[Core] Add --deterministic-prefix-caching for reproducible prefill on ROCm` (OPEN, 1 commit, opened 2026-04-17)
- **URL:** https://github.com/vllm-project/vllm/pull/40179
- **Verbatim quote of the asking:** "gigamonkeyx wants to merge 1 commit into vllm-project:main" … (title) "[Core] Add --deterministic-prefix-caching for reproducible prefill on ROCm".
- **What specifically is missing:** Review and merge; the flag does not exist in vLLM main. This is a proposal rather than shipped behaviour, which is why it belongs on the open side of the map. (The evidence file cited this PR under an `/issues/` path; the verifier corrected it to the pull-request URL used here.)
- **Evidence:** READ BODY

### B111. Batch invariance is broken when sequence parallelism / async TP is enabled
- **WHO is still asking:** `LioEinaudi` — vLLM issue #56370 (OPEN, created 2026-09-11, label `bug`, links sibling #56377)
- **URL:** https://github.com/vllm-project/vllm/issues/56370
- **Verbatim quote of the asking:** "[Bug]: Batch invariance is broken when sequence parallelism / async TP is enabled (`VLLM_BATCH_INVARIANT=1` + `pass_config.enable_sp`)"
- **What specifically is missing:** The strength of the batch-invariance guarantee under shipped optimization passes. This is a determinism hole in the same feature whose KV interaction the batch-invariant prefix-caching items cover, and the environment block records "vLLM main @ 7470082 (2026-09-10)".
- **Evidence:** READ BODY

### B112. Prefix cache hit rate collapses with concurrency on hybrid KV managers
- **WHO is still asking:** `manueldomke` with independent reproduction by `stecasta` — vLLM issue #42948 (OPEN, created 2026-05-18)
- **URL:** https://github.com/vllm-project/vllm/issues/42948
- **Verbatim quote of the asking (independent reproduction by stecasta):** "Sharp cliff between BS=2 and BS=4 with pool nowhere near full, consistent with your \"destroys cache keys even when the pool has free space\" diagnosis. Block-pool instrumentation confirms ~92% of get_new_blocks allocations destroy a cached entry during the failing runs." … "vLLM conflates \"free physical block\" with \"cached prefix entry\"; a two-tier pool, refcount-based pinning, or stale-marking on eviction would all be cleaner long-term solutions."
- **What specifically is missing:** The structural fix for the general case. A fix landed only for the DeepSeek-V4 sliding-window shape; the measured cliff (94.3% hit at batch 1–2, 1.0% at batch 4, 0.3% at batch 8) and the underlying free-block/cached-entry conflation remain unfixed. The requester's own pointer to the structural avenue is the open layout RFC: "The real avenue is heterogeneous per-page-size pools so the bs=4 SWA indexer stops costing 64× its useful storage: #42082 (RFC) + #42374 (WIP)."
- **Evidence:** READ BODY

### B113. RFC to standardize KV-cache layouts — the structural prerequisite — is open, and the WIP PR has 93 commits
- **WHO is still asking:** `LucasWilkinson` — vLLM issue #42082 `[RFC]: Standardize KV-cache Layouts` (OPEN, created 2026-05-08) with PR #42374 `[Core][WIP][1/N] Standardize kv layout` (OPEN)
- **URL:** https://github.com/vllm-project/vllm/issues/42082 and https://github.com/vllm-project/vllm/pull/42374
- **Verbatim quote of the asking:** "Right now attention backends can have semantically different and physically different KV-cache layouts. This leads to messy KV-connector code (i.e. lots of is_mamba and is_mla flags), and creates a tight coupling between KV-connector and the attention backends."
- **What specifically is missing:** The heterogeneous per-page-size pool that several abandoned efforts identified as the real fix for block-size waste. The RFC is explicitly labelled `RFC` and remains open; the WIP implementation PR is likewise open, so every consumer of layout information — connectors, routers, offload tiers — still works against per-backend special cases.
- **Evidence:** READ BODY

### B114. SGLang radix cache + deterministic inference: two roadmap items still unchecked
- **WHO is still asking:** `Fridge003` — SGLang issue #10278 `[Feature] Support deterministic inference with Batch Invariant Ops` (page state CLOSED with `stateReason=COMPLETED`, but the body's checklist is not complete)
- **URL:** https://github.com/sgl-project/sglang/issues/10278
- **Verbatim quote of the asking (the checklist as retrieved, unchecked items verbatim):** "### Radix Cache Support / - [x] FA3 supported given its already 1-stage prefill at the beginning / - [x] Triton Supported by PR: https://github.com/sgl-project/sglang/pull/11147 @hebiao064 @zminglei @byjiang1996 / - [ ] FlashInfer Support / - [ ] Making Prefill with Radix Cache has the same output as Prefill without Radix cache @hebiao064 @hanming-lu"
- **What specifically is missing:** Exactly the two named items — FlashInfer support under deterministic inference, and equality of output between radix-cached prefill and uncached prefill — plus the also-unchecked "Qwen3 Next (Linear Attention)" model item. The COMPLETED state with unticked boxes is a bookkeeping inconsistency rather than a fix, so the capability questions stand.
- **Evidence:** READ BODY

### B115. SGLang hybrid-Mamba: a prefix ending on a chunk boundary is never backed up, and a chunked-prefill/radix-insert race corrupts KV pages
- **WHO is still asking:** `decajoin` — SGLang PR #30850 (OPEN, opened 2026-07-11) and `andreasknopke` — SGLang issue #38319 (OPEN, created 2026-09-07, assigned 2026-09-11)
- **URL:** https://github.com/sgl-project/sglang/pull/30850 and https://github.com/sgl-project/sglang/issues/38319
- **Verbatim quote of the asking:** "With hybrid-Mamba models + hierarchical cache (write_through) + chunked prefill, any prompt whose page-aligned length is an exact multiple of chunked_prefill_size is silently never backed up t…" … "fix(mem_cache): eliminate chunked-prefill radix-insert race corrupting QSA KV pages\n\nFixes #38319".
- **What specifically is missing:** Merges. Both are correctness defects in the same hybrid-plus-chunked-prefill region — one produces silent non-persistence in the disk path, the other corrupts KV pages — and the fix referenced on #38319 had not landed as of the research date.
- **Evidence:** READ BODY

### B116. Cache reuse changes deterministic output (vLLM) and computed logprobs (llama.cpp)
- **WHO is still asking:** unnamed vLLM users on issues #54490 and #54487 (OPEN, both created 2026-08-31) and a reporter on llama.cpp issue #28368 (OPEN since 2026-09-04)
- **URL:** https://github.com/vllm-project/vllm/issues/54490 and https://github.com/vllm-project/vllm/issues/54487 and https://github.com/ggml-org/llama.cpp/issues/28368
- **Verbatim quote of the asking:** "The minimal accepted prefix-cache configuration produces different text for two identical long-prompt requests, while the no-prefix baseline passes." … "The cache should preserve request semantics or reject the unsupported interaction." … "cache_prompt reuse changes computed logprobs on a plain (non-hybrid) transformer — reproducible by toggling cache_prompt alone on an otherwise-identical warmed [run]".
- **What specifically is missing:** A root cause and either a fix or an explicit rejection of the unsupported interaction; none of the three reports has a linked fix. The vLLM pair also shows the failure is sensitive to configuration rather than to prefix caching alone: #54487's repro uses `--enable-lora --prefix-caching-hash-algo sha256_cbor --mamba-cache-dtype float32` and expects "Identical deterministic requests should remain stable, independent of the cache hash configuration." A fourth evidence row (llama.cpp #28368) extends the same question to logprob bit-stability on a plain transformer.
- **Evidence:** READ BODY (vLLM #54490, #54487); TITLE ONLY (llama.cpp #28368 — title from the search-API result set)

### B-notes

- **Reclassified out of this section:** S5 OPEN-10 (vLLM issue #46971, "KV cache offload prefix reuse never hits when MTP/Eagle speculative decoding is on") is **CLOSED**, `stateReason: COMPLETED`, closed by MERGED PR #46972 "[Bugfix][KV offload] Store interior chunk-boundary blocks under MTP/Eagle" (orozery, 2026-07-07), whose body says "Fixes #46971". It is removed from B; the row's own "reported as still unfixed" is contradicted by the merged fix.
- **Folded in:** S11 ABANDONED-15 (Dynamo issue #13794) is an OPEN issue body by its author `dreamtalen`, not a closure statement, so it appears here as B95 rather than in the abandoned class.
- **Dropped for insufficient open evidence:** S12 OPEN-9 (vLLM #42024, NIXL silently disables HMA) — the verifier established the artifact was stale-closed by `github-actions` with `duplicateOf: null` and no technical verdict, and its body quote was sub-agent-retrieved; an OPEN cell needs a live ask, and this one has none.
- **Dropped for missing/derived-only quotes:** none. Every one of the 153 source rows carried a usable quote, so the only removal is the reclassification above.
- **Merges (same question or same artifact across segments):** #55196 (S1 OPEN-1 + S4 OPEN-13 + S11 OPEN-1 → B1); #55697 (S1 OPEN-8 + S2 OPEN-1 → B8); the sub-block zero-hit family #40696/#53749/#48401/#52897 (S1 OPEN-3 + S13 OPEN-3 + S11 OPEN-6 + S11 OPEN-7 → B3); #54625 (S2 OPEN-12 + S9 OPEN-10 + S13 OPEN-2 → B22); #36208 (S3 OPEN-5 + S8 OPEN-5 → B28); vLLM BlockPool/priority policy #40268 + #47802 (S3 OPEN-1 + S9 OPEN-1 → B24); sparse-KV framework #5751 + forum 2745 (S3 OPEN-2 + S3 OPEN-8 → B25); KIVI/INT8 #33480 + #44059 + S4 OPEN-12 (→ B35); TurboQuant tracking #40069 + bug #40831/#53180/#52475 (→ B32); #51751 + #53299 (→ B37); #49902 + #50014 (→ B41); #38452 + #33713 (→ B43); #38470 + #38448 with their S11 duplicates (→ B44); #54327 + #54779 (→ B53); #55598 + #56123 (→ B63); #51948 + #53871 (→ B71); #52113 + #51428 (→ B77); #44775 + TensorRT-LLM #16252 (→ B78); #38260 + #19854 (→ B48); #52137 + #53395 (→ B88); #26676 + #28194 (→ B93); #26207 + #44701 (→ B94); #41623 + #54300 (→ B98); #27433 + #46592 (→ B108); #39146/#39389/#37164 (→ B109); #54490 + #54487 + llama.cpp #28368 (→ B116); #30850 + #38319 (→ B115); SGLang #1150 (S5 OPEN-8 + S11 OPEN-11 → B47); SGLang HiCache H2D stalls (S5 OPEN-5 + S5 OPEN-13 + S11 OPEN-3 + S11 OPEN-4 → B44).
- **Quote corrections applied from the verification reports:** the S7 OPEN-6 router quote (B67) is marked **APPROXIMATE** and uses the verifier's corrected tail ("drop `worker` from that node's worker set"); the SGLang #36208 Non-Goals quote (B28) uses the page's "not because **it's** unwanted"; the #53128 quote (B76) uses "the peer will never come" instead of the evidence file's "the peer is gone"; the vLLM #51751 quote (B37) is flagged as a restructuring of the source's numbered list rather than one contiguous span; the S14 OPEN-8 row (B110) uses the corrected pull-request URL.
- **Status corrections applied:** S9 OPEN-1's claim of "no maintainer reply" on #47802 is false — `njhill` replied pointing at #40004 and related PRs — so B24 states the open question as "which linked PR lands" rather than "nobody has answered".
- **Rows retained although their artifact reads CLOSED/COMPLETED, with the status disclosed:** #50497 (B69, RFC closed COMPLETED but no per-request MLA schedule landed), #31064 (B83, closed COMPLETED but the capability is absent from `KVConnectorBase_V1`), #44775 (B78, Phase 3 explicitly deferred), #33170 (B86, stale-closed with no technical verdict), #10278 (B114, COMPLETED with the checklist unticked), #39389 (B109, body reads "NA" but the page carries triage and an independent reproduction).
- **Partial verifications carried through:** S11 OPEN-13's "at least nine open correctness/overhead bugs" is reduced in B92 to the three the verifier confirmed (#36179, #32693, #39147); the S9 OPEN-11 family (B79) notes that #28267 is DRAFT rather than plain open; the S13 source-mirror line numbers are not cited as line-precise anywhere in this section, per the verifier's finding that `/tmp/kvsrc/vllm-main` is a vLLM ~0.11.2 snapshot.
- **Corrections that belong to other sections (not restated as B items):** S7 CLOSED-3 must read "partially merged — #54307 open", S7 CLOSED-1's "18 connectors" is retracted (16 in the inspected tree, 17 on current main), S7 ABANDONED-1's closer is the human author `kliukovkin`, S7 ABANDONED-2..7 each carry a "Closing here" comment plus a ClosedEvent by `shawnguyen-tensormesh`, S11 CLOSED-11/ABANDONED-49's "guard PR #27067" is unsupported, S11 CLOSED-4's summary conflicts with llama.cpp #26423's own title, S11 ABANDONED-51/-52/-59/-54/-7 do have retrievable closure comments, S11 ABANDONED-14's date is 2025-12-05, S5 CLOSED-10 (CacheGen) is a 4×A40 evaluation, S6 ABANDONED-12 reads "should remain open", S6 CLOSED-10's abstract is not truncated, S1 CLOSED-11's corroborating quotes live on issue #45238, and S1 ABANDONED-1's quote drops the author's admission that `--prefix-match-unit` exists.
- **Counts:** 153 source rows → 116 items after merging 37 duplicate/overlapping rows into their primaries, removing 1 reclassified row (S5 OPEN-10) and adding 1 reclassified row (S11 ABANDONED-15). Every item carries at least one live URL from the evidence and a verbatim, verifier-corrected or explicitly-flagged-approximate quote.
