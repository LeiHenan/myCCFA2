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
