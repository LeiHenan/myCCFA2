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
