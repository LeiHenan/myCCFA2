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
