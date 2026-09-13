# KV Cache Optimization for LLM Inference — Gap Map (CLOSED / OPEN / ABANDONED / HARDWARE-RULED-OUT)

**Compiled:** 2026-09-13 · **Window:** prioritised 2025-06 → 2026-09; older work only where it is the canonical reference for a still-open question.
**Target setup:** ONE GPU (RTX PRO 6000 Blackwell 96 GB, sm120, driver 580), 208 CPU cores, ample disk, single node, no NVLink/InfiniBand/cluster; vLLM 0.29.0, SGLang 0.5.19, Qwen3-4B.

## How this map was built, and how much to trust each row

* 14 parallel subsection researchers each swept one KV subtopic (paged/block management, prefix caching, eviction policy, quantization, offloading/tiering, compression/low-rank, cross-request & cross-replica sharing, long context, KV lifetime/TTL economics, disaggregated P/D, negative results, hardware-ruled-out, single-GPU agentic, and correctness/observability). They harvested **697 evidence blocks**: 197 CLOSED, 153 OPEN, 233 ABANDONED, 114 HARDWARE-RULED-OUT.
* Every block was then re-checked by a **separate adversarial verifier** which re-fetched each URL over HTTP, re-located each claimed verbatim quote in the retrieved page, and tried to falsify each status claim. The 14 verification reports are on disk at `evidence/kv-gapmap/S*.verify.md`.
* **Row counts here: 131 CLOSED, 116 OPEN, 185 ABANDONED, 94 HARDWARE-RULED-OUT** after de-duplication across subsections and after applying the verifiers' classification corrections.
* **Classification corrections forced by verification are already applied.** The most consequential ones: a row claiming to be OPEN but fixed by a merged PR (vLLM #46971 ← #46972) was moved out of Section B; a row filed as ABANDONED whose artifact is still open (Dynamo #13794) was moved to Section B; a "merged" claim covering an actually-open PR (vLLM #54307) is marked partially merged; a CacheGen hardware quote that had been silently edited from "four GPUs" to a single GPU was corrected, which moves that paper into Section D.
* **Six ABANDONED-class rows were dropped entirely** because their only closure text was an automated stale-bot notice with no human statement anywhere on the page: vLLM #42510, Dynamo #7802, SGLang PR #22841, vLLM #10611, llama.cpp PR #24004, vLLM PR #42985. Per the "no verbatim quote ⇒ drop the row" rule.
* **Evidence labels are per item.** `READ BODY` means the page/PDF/API content was actually retrieved and read. `TITLE ONLY`, `TITLE + STATE ONLY`, and `APPROXIMATE` are used where the verifier established that only the title, or only an altered/paraphrased quote, was available. Rows the verifier downgraded carry the downgraded label.
* Verifier-reported residual defects that a reader should know about, because they could not be fully repaired without re-doing the research: one quote in the S1 group is truncated mid-sentence without an ellipsis; two "verbatim" quotes in the S6 group differ from the source only by arXiv HTML rendering artefacts; a small number of discussion quotes were attributed to the wrong participant in the S4 group (the quote text itself is genuine); several "quoted" question labels originating in the S13 group are the researcher's paraphrase rather than a quotation, and are marked as such. Line numbers cited from the local `/tmp/kvsrc/vllm-main` mirror are a vLLM ~0.11.2 snapshot and have drifted on live `main`; do not treat them as line-precise.
* **Measured fidelity of Section C's quotation fields.** A blind 40-item random sample of Section C's reason fields was re-fetched at source and matched mechanically: **28/40 (70%) reproduce verbatim at the cited URL**, 6/40 contain a genuine quotation embedded in retrieved framing text, and 6/40 are paraphrase or composite rather than one continuous quotation. Accordingly, **74 of the 185 Section C items carry a field explicitly labelled `STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation)`** rather than `STATED REASON, VERBATIM`; the other 111 carry a single continuous quotation. Read the label before treating any reason as an exact quotation.
* URLs use the proxy/retry fetch path that worked here (`curl -sL --retry 4`); the `web_fetch` tool was unavailable in this environment.

---

## A. CLOSED cells (someone shipped or published a working answer)

| # | Question it answers | Who closed it | Artifact (paper/PR/flag) | URL | Status (merged/shipped/published) | READ BODY or TITLE ONLY |
|---|---|---|---|---|---|---|
| A1 | block-paged KV management near-eliminates fragmentation and enables sharing within and across requests | PagedAttention/vLLM authors, arXiv 2309.06180 | PagedAttention + vLLM (paper) | https://arxiv.org/abs/2309.06180 | published | READ BODY |
| A2 | contiguous virtual-memory KV is a performant alternative to paging | vAttention authors, arXiv 2405.04437 | vAttention | https://arxiv.org/abs/2405.04437 | published | READ BODY |
| A3 | mixed-format paged attention (FP8 anchor pages + packed 3-bit pages) runs on one 96 GB Blackwell GPU | Minima-KV authors, arXiv 2608.23834 | Minima-KV mixed-format paged attention | https://arxiv.org/abs/2608.23834 | published | READ BODY |
| A4 | decode can execute directly over native page tables without repacking | PersistentKV authors, arXiv 2606.26666 | PersistentKV page-aware decode scheduling | https://arxiv.org/abs/2606.26666 | published | READ BODY |
| A5 | one shared KV memory pool can serve heterogeneous attention layer types | vLLM | Hybrid KV Cache Manager design doc | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/hybrid_kv_cache_manager.md ; https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ | shipped | READ BODY |
| A6 | kernel block size can be decoupled from KV page size in hybrid Mamba/attention models | vLLM (zhiyuan1i) | PR #24486 "[Hybrid]: Decouple Kernel Block Size from KV Page Size" + `--mamba-block-size` | https://github.com/vllm-project/vllm/pull/24486 | merged 2025-10-09 | READ BODY |
| A7 | the V1 block manager never allocates beyond max_model_len | vLLM (WoosukKwon) | PR #10730 "[V1] Do not allocate beyond the max_model_len" | https://github.com/vllm-project/vllm/pull/10730 | merged 2024-11-28 | TITLE + STATE ONLY |
| A8 | restoring LIFO block reuse order in the free pool when prefix caching is off | vLLM (theminghuang) | PR #51482 "free_blocks: restore prepend (LIFO) reuse order" | https://github.com/vllm-project/vllm/pull/51482 | merged 2026-08-11 | READ BODY |
| A9 | GPU-side KV events for the hybrid memory allocator | vLLM (hickeyma) | PR #37688 "[HMA] [KVEvent] Enable GPU-side KV events for HMA" | https://github.com/vllm-project/vllm/pull/37688 | merged 2026-04-12 | READ BODY |
| A10 | how page sizes are unified or padded across heterogeneous layers | vLLM | `unify_kv_cache_spec_page_size` in `vllm/v1/core/kv_cache_utils.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/kv_cache_utils.py | shipped | READ BODY |
| A11 | Marconi-style shared-prefix admission for hybrid cache align mode | vLLM (s3woz) | PR #37898 "[Hybrid] Marconi-style admission policy for hybrid cache" | https://github.com/vllm-project/vllm/pull/37898 | merged 2026-06-10 | READ BODY (verifier: this row's corroborating quotes are on issue #45238, not #54199) |
| A12 | llama.cpp's KV update/defrag mechanism | llama.cpp (ggerganov) | PR #13988 "kv-cache : refactor the update/defrag mechanism" | https://github.com/ggml-org/llama.cpp/pull/13988 | merged 2025-06-04 | TITLE + STATE ONLY |
| A13 | hash-based full-block prefix caching with a collision-resistant default hash | vLLM | Automatic Prefix Caching design doc (`sha256` default, `cache_salt`, extra hashes) | https://docs.vllm.ai/en/latest/design/prefix_caching/ | shipped | READ BODY |
| A14 | per-request isolation of prefix caches in multi-tenant serving | vLLM | PR #54353 "Bound cache_salt length to prevent DoS via scheduler CPU exhaustion" | https://github.com/vllm-project/vllm/pull/54353 | merged 2026-08-30 | READ BODY |
| A15 | prefix caching is on by default | vLLM | `CacheConfig.enable_prefix_caching = True` | https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py | shipped | READ BODY |
| A16 | hybrid Mamba models can get prefix-cache hits without a second full-model pass | vLLM | PR #52789 "internal prefill checkpoints for Mamba prefix caching (9%~25% TTFT)" | https://github.com/vllm-project/vllm/pull/52789 | merged 2026-08-22 | READ BODY |
| A17 | prefix-cache hits can land inside a physical block for hybrid models | vLLM | RFC #45702 "Partial Cache Hits for Hybrid Models" + PR #53614 + `--enable-mamba-fine-grained-prefix-cache` doc | https://github.com/vllm-project/vllm/issues/45702 ; https://github.com/vllm-project/vllm/pull/53614 ; https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md ; https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/ | shipped (RFC closed COMPLETED; PR merged 2026-09-06) | READ BODY |
| A18 | fine-grained hits fall back instead of asserting when a KV group cannot answer them | vLLM | PR #51843 (fine-grained hits disabled with logging; scheduler-block-aligned fallback) | https://github.com/vllm-project/vllm/pull/51843 | merged 2026-08-12 | READ BODY |
| A19 | offloaded blocks outside the sliding window are not stored or loaded | vLLM (orozery) | PR #51886 retention interval for `OffloadingConnector` | https://github.com/vllm-project/vllm/pull/51886 | merged 2026-09-04 | READ BODY |
| A20 | selectable prefix-cache eviction policies (lru / lfu / slru / priority) | SGLang | `--radix-eviction-policy` + `--radix-eviction-policy-config` doc | https://docs.sglang.io/docs/advanced_features/radix_eviction_policy ; https://docs.sglang.io/docs/advanced_features/radix_eviction_policy.md | shipped | READ BODY |
| A21 | session-referenced KV is evicted after unreferenced KV | SGLang (hzh0425) | PR #29173 "Session-reference-aware Unified Radix Cache" + session radix cache doc | https://github.com/sgl-project/sglang/pull/29173 ; https://docs.sglang.io/docs/advanced_features/session_radix_cache.md | merged 2026-08-02 | READ BODY |
| A22 | routing requests to the worker with the highest prefix match | SGLang (SMG); NVIDIA Dynamo | SMG cache-aware policy + Dynamo KV-aware router PR #14498 | https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide.md ; https://github.com/ai-dynamo/dynamo/pull/14498 | shipped; PR merged | READ BODY |
| A23 | multimodal tensors covered by a prefix-cache hit are not re-broadcast to TP workers | vLLM (sseanliu, merged by njhill) | PR #52041 | https://github.com/vllm-project/vllm/pull/52041 | merged 2026-08-19 | READ BODY |
| A24 | cross-instance prefix reuse through a three-tier hierarchy (L1 GPU / L2 host / L3 storage) | SGLang | HiCache design + best practices docs (L2 is node-local and instance-private; host tier must exceed device tier) | https://docs.sglang.io/docs/advanced_features/hicache_design.md ; https://docs.sglang.io/docs/advanced_features/hicache_best_practices.md | shipped | READ BODY |
| A25 | the L3 cache backend can be attached or detached without restarting | SGLang | HiCache storage runtime attach/detach doc | https://docs.sglang.io/docs/advanced_features/hicache_storage_runtime_attach_detach.md | shipped | READ BODY |
| A26 | radix-tree prefix reuse with a shared memory pool | SGLang authors, arXiv 2312.07104 | RadixAttention (SGLang paper) | https://arxiv.org/abs/2312.07104 | published | READ BODY |
| A27 | eviction policies for the CPU offload tier can be selected or plugged in out-of-tree | vLLM (orozery) | PR #37874 (pluggable `CachePolicy`, `cpu/` package) + PR #49114 (`CachePolicyFactory`) | https://github.com/vllm-project/vllm/pull/37874 ; https://github.com/vllm-project/vllm/pull/49114 | merged 2026-03-24 and 2026-07-29 | READ BODY |
| A28 | training-free decode-time KV eviction inside a production engine | NVIDIA (Hudayday) | TriAttention, TensorRT-LLM PR #16957 | https://github.com/NVIDIA/TensorRT-LLM/pull/16957 | merged 2026-08-04 | READ BODY |
| A29 | a library of ready-made KV compression/eviction presses | NVIDIA | NVIDIA/kvpress (SnapKV, StreamingLLM, PyramidKV, AdaKV, CriticalKV, DecodingPress) | https://github.com/NVIDIA/kvpress | shipped | READ BODY (verifier: no H2O press and no Scissorhands press — scope caveat) |
| A30 | head-wise adaptive KV budget allocation | Ada-KV authors (NeurIPS 2025); Cloudflare kvcompress | Ada-KV repo + arXiv 2407.11550 | https://github.com/FFY0/AdaKV | published (NeurIPS 2025) | READ BODY |
| A31 | attention sinks reach the decode kernel and are provably necessary | vLLM (plliao; AndreasKaratzas); arXiv 2603.11487 authors | PR #26325 (cascade sink tensor) + PR #54404 (sparse-MLA sinks) + sink-necessity theory paper | https://github.com/vllm-project/vllm/pull/26325 ; https://github.com/vllm-project/vllm/pull/54404 ; https://arxiv.org/abs/2603.11487 | merged (#54404 on 2026-09-07); published (v5 2026-04-17) | READ BODY (PRs) / APPROXIMATE (theory quote drops the word "simple" — verifier) |
| A32 | recency/frequency adaptive replacement for KV blocks | vLLM (ApostaC); arXiv 2606.21238 authors | PR #27039 (ARC eviction for CPU offload) + ARC-style adaptive block caching paper | https://github.com/vllm-project/vllm/pull/27039 ; https://arxiv.org/abs/2606.21238 | merged 2025-11-12; published | READ BODY |
| A33 | reuse-aware eviction for coding-agent workloads | CacheWise authors, arXiv 2606.16824 | CacheWise (implemented in vLLM) | https://arxiv.org/abs/2606.16824 | published | READ BODY |
| A34 | preemption by recompute rather than CPU swap in V1 | vLLM | PR #36216 + `docs/configuration/optimization.md` | https://github.com/vllm-project/vllm/pull/36216 | merged | READ BODY |
| A35 | tiered lookups report a hit only once KV is materialized in the primary tier | vLLM (orozery) | PR #51840 (fixes RFC #51439 tiered lookup resolution) | https://github.com/vllm-project/vllm/pull/51840 ; https://github.com/vllm-project/vllm/issues/51439 | merged 2026-08-12 | READ BODY |
| A36 | session-based streaming input for the V1 engine | vLLM (njhill) | PR #28973 "[Feature] add session based streaming input support to v1" | https://github.com/vllm-project/vllm/pull/28973 | merged 2026-01-24 | READ BODY |
| A37 | per-request KV time-to-live from reload cost plus queueing delay | Continuum/CacheTTL authors, arXiv 2511.02230 | Continuum (per-request KV TTL) | https://arxiv.org/abs/2511.02230 | published (v7 2026-09-08) | READ BODY |
| A38 | logical token liveness decoupled from physical block placement | vToken authors, arXiv 2608.13263 | vToken (vLLM v0.18.0 prototype) | https://arxiv.org/abs/2608.13263 | published | READ BODY |
| A39 | which tenant group bears reclamation pressure in a shared prefix cache | PrefixShield authors, arXiv 2608.01657 | PrefixShield (implemented in vLLM) | https://arxiv.org/abs/2608.01657 | published | READ BODY |
| A40 | policy-directed removal or replacement of a cached span without re-prefill | Leyline authors, arXiv 2606.01065 | Leyline splice kernel | https://arxiv.org/abs/2606.01065 | published | READ BODY |
| A41 | semantic/role-aware eviction for multi-turn prefix caches | SAECache authors, arXiv 2605.18825 | "Not All Tokens Are Worth Caching" (in-paper system name SAECache) | https://arxiv.org/abs/2605.18825 | published 2026-05-12 | TITLE ONLY (verifier: the quoted "question" is the agent's paraphrase; the system name is not the title) |
| A42 | lossless eviction scored by position-aware recomputation cost | AsymCache authors, arXiv 2606.02964 | AsymCache / Multi-Segment Attention | https://arxiv.org/abs/2606.02964 | published | READ BODY |
| A43 | FP8 KV cache works without calibration, and its long-context accuracy collapse is fixed | vLLM | FP8 KV-cache blog (per-tensor scale = 1.0; two-level accumulation fix; `--kv-cache-dtype-skip-layers`, PR #33695, and the quantized-KV dtype doc) | https://vllm.ai/blog/2026-04-22-fp8-kvcache ; https://github.com/vllm-project/vllm/pull/33695 ; https://docs.vllm.ai/en/v0.28.0/features/quantization/quantized_kvcache/ | published 2026-04-22 | READ BODY |
| A44 | per-token-head dynamic INT8/FP8/INT4 KV scales are expressible without calibration | vLLM (tdoublep) | `KVQuantMode` enum in `kv_cache_interface.py` + PR #40835 (INT4 per-token-head) | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/kv_cache_interface.py ; https://github.com/vllm-project/vllm/pull/40835 | merged 2026-06-24 | READ BODY |
| A45 | sub-4-bit online KV compression via Hadamard rotation plus Lloyd-Max scalar quantization | vLLM (vibhavagarwal5); TurboQuant authors | PR #38479 `--kv-cache-dtype turboquant_k8v4` + arXiv 2504.19874 | https://github.com/vllm-project/vllm/pull/38479 ; https://arxiv.org/abs/2504.19874 | merged 2026-04-15; published | READ BODY (PR body read for #38479 in S4) / PR METADATA ONLY (same PR in S6) |
| A46 | fp8_e5m2 KV cache is reachable for weight-only quantized checkpoints (Ampere) | vLLM (yewentao256) | PR #45040 | https://github.com/vllm-project/vllm/pull/45040 | merged 2026-06-18 | READ BODY |
| A47 | quantization granularity, outlier handling and key/value asymmetry for low-bit KV | KIVI authors 2402.02750; KVQuant authors 2401.18079; QAQ authors 2403.04643 | KIVI + KVQuant + QAQ | https://arxiv.org/abs/2402.02750 ; https://arxiv.org/abs/2401.18079 ; https://arxiv.org/abs/2403.04643 | published | READ BODY |
| A48 | calibration-free variance-normalized quantization against autoregressive error accumulation | KVarN authors, arXiv 2606.03458 | KVarN (2-bit, MATH500/AIME24/HumanEval) | https://arxiv.org/abs/2606.03458 | published | READ BODY |
| A49 | the dequant-once Vulkan q8_0 kernel is only a win behind a measured benefit gate | llama.cpp | PR #25494 | https://github.com/ggml-org/llama.cpp/pull/25494 | merged 2026-08-19 | APPROXIMATE (merge state retrieved via search API; the quote is a search-API text fragment, not a full-page read) |
| A50 | the "quantized KV produces garbage on master" report is retracted by its reporter | llama.cpp | issue #26423 | https://github.com/ggml-org/llama.cpp/issues/26423 | shipped (issue closed; the surviving title asserts the opposite — conflict disclosed) | READ BODY |
| A51 | MTP with `--kv-cache-dtype auto` cross-sequence corruption was not reproduced | vLLM | issue #46088 | https://github.com/vllm-project/vllm/issues/46088 | shipped (issue closed — not reproduced) | READ BODY |
| A52 | native CPU KV offload with an asynchronous copy path | vLLM (NickLucche; orozery) | OffloadingConnector blog + PR #24498 (`cpu_bytes_to_use`) + KV offloading usage guide | https://vllm.ai/blog/2026-01-08-kv-offloading-connector ; https://github.com/vllm-project/vllm/pull/24498 ; https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/kv_offloading_usage.md | merged 2026-01-12; shipped in vLLM 0.11.0 | READ BODY |
| A53 | a tiered offload framework with a CPU primary tier and disk/object/P2P secondary tiers | vLLM (orozery; ronensc) | PR #40020 `TieringOffloadingSpec` + usage guide + 2026-09-10 tiered-offloading blog | https://github.com/vllm-project/vllm/pull/40020 ; https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/kv_offloading_usage.md ; https://vllm.ai/blog/2026-09-10-tiered-kv-offloading | merged 2026-05-13 | READ BODY |
| A54 | all KV swap copies for a batch are issued in one driver call | vLLM (orozery/Etelis) | PR #38460 (`cuMemcpyBatchAsync` on CUDA 12.8+) | https://github.com/vllm-project/vllm/pull/38460 | merged 2026-04-03 | READ BODY |
| A55 | eager-registered CPU blocks become discoverable and the final flush happens | vLLM (ivanium) | PR #53532 (`SimpleCPUOffloadConnector`, closes #53498) | https://github.com/vllm-project/vllm/pull/53532 | merged 2026-09-02 | READ BODY |
| A56 | offloading KV to an external distributed store (CPU or disk) | vLLM | `MooncakeStoreConnector` usage doc | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/mooncake_store_connector_usage.md | shipped | READ BODY |
| A57 | a CPU tier for the multimodal encoder cache | vLLM | `ECCPUConnector` doc | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/ec_cpu_connector.md | shipped | READ BODY |
| A58 | multi-level (CPU / SSD / remote) KV offload as a registered connector | vLLM | `FlexKVConnectorV1` in `kv_connector/factory.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/factory.py | shipped | READ BODY |
| A59 | KV blocks can be exchanged instance-to-instance over RDMA with no shared filesystem | vLLM (liranschour) | P2P tier of the offloading connector + PR #47636 | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/kv_offloading_usage.md ; https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md ; https://github.com/vllm-project/vllm/pull/47636 | shipped | READ BODY |
| A60 | identical token content yields identical block keys across processes | vLLM | KV offloading guide (`NONE_HASH`, `PYTHONHASHSEED`; xxhash exception) | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/kv_offloading_usage.md ; https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md | shipped | READ BODY |
| A61 | CPU-GPU cooperative KV retrieval for block-diffusion LLMs | HERALD authors, arXiv 2606.21633 | HERALD | https://arxiv.org/abs/2606.21633 | published 2026-06-19 | READ BODY |
| A62 | joint cache placement, pipeline scheduling and cross-tier coordination over GPU / host DRAM / SSD | KVDrive authors, arXiv 2605.18071 | KVDrive | https://arxiv.org/abs/2605.18071 | published 2026-05-18 | READ BODY |
| A63 | SSD-backed KV with the CPU removed from the data and I/O control paths | Tutti authors, arXiv 2605.03375 | Tutti GPU-centric KV cache object store | https://arxiv.org/abs/2605.03375 | published 2026-05-05 | READ BODY (verifier: one "verbatim" result quote is a paraphrase) |
| A64 | importance-informed multi-tier prefix KV storage | IMPRESS authors, USENIX FAST 2025 | IMPRESS | https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress | published (FAST 2025) | READ BODY |
| A65 | disk KV loading with asynchronous direct I/O and scheduler-aware preloading | py-kvcache authors, arXiv 2609.11744 | py-kvcache vLLM KV offload connector | https://arxiv.org/abs/2609.11744 | published 2026-09-10 | READ BODY |
| A66 | KV compression plus streaming under bandwidth limits, and GPU+CPU conversation-state caching | CacheGen authors 2310.07240; Pensieve authors 2312.05516 | CacheGen + Pensieve | https://arxiv.org/abs/2310.07240 ; https://arxiv.org/abs/2312.05516 | published | TITLE + ABSTRACT ONLY (CacheGen; hardware is a four-GPU A40 server, not one A40) / READ BODY (Pensieve) |
| A67 | latent-vector KV compression and the shipped backend family that serves it | DeepSeek, arXiv 2405.04434; vLLM | MLA (DeepSeek-V2) + vLLM MLA backend doc | https://arxiv.org/abs/2405.04434 ; https://github.com/vllm-project/vllm/blob/main/docs/design/attention_backends.md | published; shipped | ABSTRACT ONLY (paper) / READ BODY (backend doc) |
| A68 | sharing KV heads across query heads and uptraining checkpoints into grouped heads | MQA authors 1911.02150; GQA authors 2305.13245 | MQA + GQA | https://arxiv.org/abs/1911.02150 ; https://arxiv.org/abs/2305.13245 | published | ABSTRACT ONLY |
| A69 | learned online per-head and per-layer KV compression ratio | DMC authors, arXiv 2403.09636 | Dynamic Memory Compression | https://arxiv.org/abs/2403.09636 | published | ABSTRACT ONLY |
| A70 | merging KV across adjacent layers via magnitude/direction disentangling | MiniCache authors, arXiv 2405.14366 | MiniCache | https://arxiv.org/abs/2405.14366 | published | ABSTRACT ONLY |
| A71 | caching KV once and reusing it via cross-attention, plus the engine flag that exploits it | YOCO authors, arXiv 2405.05254; vLLM | YOCO + `kv_sharing_fast_prefill` (Gemma3n / Gemma4) | https://arxiv.org/abs/2405.05254 ; https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py | published; shipped | READ BODY (vLLM flag and source quotes) / ABSTRACT ONLY (YOCO paper — S6 downgrade) |
| A72 | sharing KV heads between adjacent layers | CLA authors, arXiv 2405.12981 | Cross-Layer Attention | https://arxiv.org/abs/2405.12981 | published | ABSTRACT ONLY |
| A73 | low-rank projection of the KV hidden dimension with fused kernels | Palu authors, arXiv 2407.21118 | Palu | https://arxiv.org/abs/2407.21118 | published | ABSTRACT ONLY |
| A74 | joint low-rank factorization of grouped-layer KV via aligned singular vectors | xKV authors, arXiv 2503.18893 | xKV | https://arxiv.org/abs/2503.18893 | published | ABSTRACT ONLY |
| A75 | converting a pretrained non-MLA checkpoint into MLA by post-training distillation | X-EcoMLA authors, arXiv 2503.11132 | X-EcoMLA | https://arxiv.org/abs/2503.11132 | published | ABSTRACT ONLY (verifier: the claimed mid-sentence truncation at the hardware name is false) |
| A76 | merging temporally adjacent KV vectors inside a latent design | MTLA authors, arXiv 2505.13544 | Multi-head Temporal Latent Attention | https://arxiv.org/abs/2505.13544 | published | ABSTRACT ONLY |
| A77 | training-free cross-layer KV compression via adjacent-parameter SVD sharing | CommonKV authors, arXiv 2508.16134 | CommonKV | https://arxiv.org/abs/2508.16134 | published | ABSTRACT ONLY |
| A78 | compressing KV below 2% by combining head/dimension reduction, layer sharing and quantization | CLLA authors, arXiv 2410.15252 | Cross-Layer Latent Attention | https://arxiv.org/abs/2410.15252 | published | ABSTRACT ONLY |
| A79 | a connector API with a documented suite of per-engine P/D connectors | vLLM; SGLang | disagg_prefill doc (9 documented connector types; registry count corrected to 16 in the inspected tree, 17 on current main) | https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md | shipped | READ BODY |
| A80 | push-mode cross-instance KV transfer with a dedicated writer thread | vLLM | `NixlPushConnector` design doc | https://github.com/vllm-project/vllm/blob/main/docs/design/nixl_kv_push_connector.md | shipped | READ BODY |
| A81 | persisted KV can be reused across instances with different TP sizes | vLLM (z-zanez) | PR #53129 + PR #52466; companion PR #54307 still open | https://github.com/vllm-project/vllm/pull/53129 ; https://github.com/vllm-project/vllm/pull/52466 ; https://github.com/vllm-project/vllm/pull/54307 | partially merged — #54307 open (verifier correction) | READ BODY |
| A82 | a renewable lease and heartbeat so a prefiller does not strand KV blocks | vLLM (NickLucche) | PR #41383 + NIXL KV lease design doc + KV role flags in the connector guide | https://github.com/vllm-project/vllm/pull/41383 ; https://github.com/vllm-project/vllm/blob/main/docs/design/nixl_kv_cache_lease.md ; https://docs.vllm.ai/en/latest/features/nixl_connector_usage/ | merged 2026-05-11 | READ BODY |
| A83 | extracting KV from an engine and sharing it across engines and queries | LMCache authors, arXiv 2510.09665 | LMCache | https://arxiv.org/abs/2510.09665 | published | READ BODY |
| A84 | multi-node P2P CPU-memory KV sharing over NVLink/RDMA/TCP | LMCache | LMCache README (multi-node P2P, MP architecture) | https://raw.githubusercontent.com/LMCache/LMCache/dev/README.md | shipped | READ BODY |
| A85 | skipping duplicate content-fingerprint registration | LMCache | PR #4757 `--enable-dedup-content` | https://github.com/LMCache/LMCache/pull/4757 | merged | READ BODY |
| A86 | datacenter-scale orchestration with KV-aware routing and multi-tier KVBM | NVIDIA | Dynamo README | https://raw.githubusercontent.com/ai-dynamo/dynamo/main/README.md | shipped | READ BODY |
| A87 | publishing KV cache events for external KV-aware routers | vLLM | `vllm/config/kv_events.py` (zmq publisher) | https://github.com/vllm-project/vllm/blob/main/vllm/config/kv_events.py | shipped | READ BODY |
| A88 | chunked prefill on by default, with a cap on how much of a long prompt one chunk takes | vLLM | optimization doc + `--long-prefill-token-threshold` in `vllm/config/scheduler.py` | https://docs.vllm.ai/en/latest/configuration/optimization/ ; https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py | shipped | READ BODY |
| A89 | sharding the KV cache along the sequence dimension for decode and prefill, with a written deployment contract | vLLM (youkaichao and contributors) | DCP PRs #23734 (MLA), #24864 (GQA/FlashAttention), #25438 (GQA/FlashInfer), #43729 (FlashInfer MLA); PCP PR #28718; doc PR #26877; deployment doc | https://github.com/vllm-project/vllm/pull/23734 ; https://github.com/vllm-project/vllm/pull/24864 ; https://github.com/vllm-project/vllm/pull/25438 ; https://github.com/vllm-project/vllm/pull/43729 ; https://github.com/vllm-project/vllm/pull/28718 ; https://github.com/vllm-project/vllm/pull/26877 ; https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md | merged 2025-09-06 to 2026-06-29 | READ BODY |
| A90 | prefill and decode context parallelism in SGLang with selectable sharding layouts | SGLang | DCP doc + server-arguments reference (`--enable-prefill-cp`, `--cp-strategy`, `--dcp-size`) | https://docs.sglang.ai/advanced_features/dcp.html ; https://docs.sglang.ai/advanced_features/server_arguments.html | shipped | READ BODY |
| A91 | a long-context benchmark beyond plain needle-in-a-haystack | NVIDIA, arXiv 2404.06654 | RULER | https://arxiv.org/abs/2404.06654 | published (v2 2024-08-06) | READ BODY |
| A92 | reusable pre-encoded KV caches for large document collections | Cartridges at Scale authors, arXiv 2606.04557 | Cartridges at Scale | https://arxiv.org/abs/2606.04557 | published 2026-06-03 | READ BODY |
| A93 | fixed chunk sizes with a virtual pipeline layout for chunked prefill | VPP authors, arXiv 2608.26523 | VPP (Virtual Pipeline Parallelism) | https://arxiv.org/abs/2608.26523 | published 2026-08-27 | READ BODY |
| A94 | a constant-size sink-plus-window KV budget for the MTP draft at 1M tokens on one GPU | Windowed-MTP authors, arXiv 2607.21535 | Windowed-MTP | https://arxiv.org/abs/2607.21535 | published 2026-07-23 | READ BODY |
| A95 | distributing attention and KV across devices or hosts | Ring Attention authors 2310.01889; Star Attention authors 2411.17116 | Ring Attention + Star Attention | https://arxiv.org/abs/2310.01889 ; https://arxiv.org/abs/2411.17116 | published | READ BODY |
| A96 | windowed streaming attention keeps working once sinks are kept | StreamingLLM authors, arXiv 2309.17453 | StreamingLLM (attention sinks) | https://arxiv.org/abs/2309.17453 | published (v4 2024-04-07) | READ BODY |
| A97 | HMA stays enabled by default for connectors that support it | vLLM (chfeng-cs, merged by NickLucche) | PR #41847 | https://github.com/vllm-project/vllm/pull/41847 | merged 2026-05-26 | READ BODY |
| A98 | what the decode instance does when a KV load fails | vLLM | `kv_load_failure_policy` (default `fail`), NIXL connector usage guide | https://docs.vllm.ai/en/latest/features/nixl_connector_usage/ | shipped | READ BODY |
| A99 | what must match and what may safely differ between prefill and decode instances | vLLM | NIXL connector compatibility doc (handshake compatibility hash) | https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/ | shipped | READ BODY |
| A100 | Mamba1 hybrid models in NIXL P/D disaggregation | vLLM | PR #45019 | https://github.com/vllm-project/vllm/pull/45019 | merged 2026-06-25 | READ BODY |
| A101 | how many prefill versus decode engines to run, and whether disaggregation raises throughput | NVIDIA Dynamo; vLLM | Dynamo tuning guide + vLLM disaggregated-prefill doc ("Disaggregated prefill DOES NOT improve throughput") | https://docs.dynamo.nvidia.com/dynamo/v-0-8-1/user-guides/tuning-disaggregated-performance ; https://docs.vllm.ai/en/latest/features/disagg_prefill/ | shipped | READ BODY |
| A102 | overlapping each layer's KV transfer with later-layer prefill | HMA-Serve authors, arXiv 2606.29986 | HMA-Serve (compute-transfer pipeline, deferred dequantization) | https://arxiv.org/abs/2606.29986 | published | READ BODY |
| A103 | reducing transferred KV bytes instead of overlapping the transfer | SCD authors, arXiv 2606.07684 | SCD | https://arxiv.org/abs/2606.07684 | published | READ BODY (verifier: one quote approximate — "5% F1") |
| A104 | splitting prefill and decode onto different machines or GPUs | DistServe authors 2401.09670; Splitwise authors 2311.18677 | DistServe + Splitwise | https://arxiv.org/abs/2401.09670 ; https://arxiv.org/abs/2311.18677 | published | READ BODY (Splitwise §IV-E quote corrected to "workload" — verifier) |
| A105 | a KVCache-centric disaggregated architecture with a disaggregated KV cache | Moonshot AI, arXiv 2407.00079 | Mooncake + shipped vLLM Mooncake connector | https://arxiv.org/abs/2407.00079 | published (v4 2025-09-03) | READ BODY |
| A106 | KV parallelism during attention on multi-GPU and NVL72 systems | NVIDIA, arXiv 2507.07120 | Helix Parallelism | https://arxiv.org/abs/2507.07120 | published 2025-07-07 | READ BODY |
| A107 | interior chunk-boundary blocks are stored so MTP/Eagle offload prefix reuse hits | vLLM (orozery) | PR #46972 "[Bugfix][KV offload] Store interior chunk-boundary blocks under MTP/Eagle" (fixes #46971) | https://github.com/vllm-project/vllm/pull/46972 ; https://github.com/vllm-project/vllm/issues/46971 | merged 2026-07-07 | READ BODY |
| A108 | KV reuse and management for agentic memory and long-horizon reasoning | AgentKVShift authors 2607.21604; SideQuest authors 2602.22603 | AgentKVShift + SideQuest | https://arxiv.org/abs/2607.21604 ; https://arxiv.org/abs/2602.22603 | published (2026-05-15 and 2026-02-26) | READ BODY |
| A109 | compressing multi-turn dialogue history without destroying earlier turns | FlowKV authors 2505.15347; EpiCache authors 2509.17396; SONIC authors 2601.21927 | FlowKV + EpiCache + SONIC | https://arxiv.org/abs/2505.15347 ; https://arxiv.org/abs/2509.17396 ; https://arxiv.org/abs/2601.21927 | published (2025-05-21, 2025-09-22, 2026-01-29) | READ BODY |
| A110 | on-demand or compressed paged KV budgeting that preserves prefix caching | GrowPage authors 2609.03494; Zipage authors 2603.08743 | GrowPage (nano-vLLM) + Zipage Compressed PagedAttention | https://arxiv.org/abs/2609.03494 ; https://arxiv.org/abs/2603.08743 | published (2026-09-03 and 2026-03-01) | READ BODY (GrowPage) / TITLE ONLY (Zipage — verifier: quoted "question" is a paraphrase and the result clause is truncated) |
| A111 | how much multi-turn KV reuse production traffic really shows | Aliyun authors, arXiv 2506.02634 | KVCache Cache in the Wild | https://arxiv.org/abs/2506.02634 | published 2025-06-03 | READ BODY |
| A112 | re-landing unified-KV pool sizing and sliding-window ring accounting | SGLang | PR #38192 (reland of #30315 after revert #38163) | https://github.com/sgl-project/sglang/pull/38192 | merged 2026-09-07 | READ BODY |
| A113 | whether HiCache host-tier hits are safe with MoE `routed_experts` sidecars | SGLang | issue #26975 (quotes verified); migration fix PR #27067 closed unmerged as superseded by #27326 | https://github.com/sgl-project/sglang/issues/26975 | closed not-planned 2026-07-10 (verifier: the "guard PR #27067" conclusion is unsupported — no guard PR shipped) | READ BODY |
| A114 | LMCache reverted its own multiprocess adapter shim | LMCache | PR #3111 | https://github.com/LMCache/LMCache/pull/3111 | merged 2026-04-23 | READ BODY |
| A115 | TensorRT-LLM prefix caching on VSWA models was silently disabled, then fixed | NVIDIA | issue #12563 + fix PR #13346 | https://github.com/NVIDIA/TensorRT-LLM/issues/12563 ; https://github.com/NVIDIA/TensorRT-LLM/pull/13346 | merged 2026-05-02 | READ BODY |
| A116 | structural boundary protection, not scoring, is what makes capped KV eviction work | arXiv 2605.18053 authors | retention-boundary study | https://arxiv.org/abs/2605.18053 | published | READ BODY |
| A117 | recomputing K/V from a residual-stream checkpoint instead of caching them | KV-Direct authors, arXiv 2603.19664 | KV-Direct | https://arxiv.org/abs/2603.19664 | published | READ BODY |
| A118 | what KV compression actually buys and costs in production serving | arXiv 2503.24000 authors | compression re-examination + measurement toolkit | https://arxiv.org/abs/2503.24000 | published | READ BODY |
| A119 | what TurboQuant reconstruction costs inside an engine | LMCache | PR #3193 (turboquant_4bit_nc vs fp8 correlation and encode/decode latency) | https://github.com/LMCache/LMCache/pull/3193 | merged | READ BODY |
| A120 | a reproducibility contract and batch-invariant execution for KV-dependent results | vLLM | reproducibility doc + batch-invariance doc (`VLLM_BATCH_INVARIANT`) | https://docs.vllm.ai/en/latest/usage/reproducibility/ | shipped | READ BODY |
| A121 | prefix-cache hit-rate counters plus KV block residency, idle and reuse-gap metrics | vLLM (NickLucche) | metrics design doc + PR #26245 (connector hit-rate stats) + `kv_cache_metrics.py` | https://docs.vllm.ai/en/latest/design/metrics/ ; https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md ; https://github.com/vllm-project/vllm/pull/26245 ; https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/kv_cache_metrics.py | shipped; PR merged | READ BODY |
| A122 | fixing "prefix cache hit rate == 0" on gpt-oss-style hybrid models | vLLM (heheda12345) | PR #33524 | https://github.com/vllm-project/vllm/pull/33524 | merged (Feb 2026) | READ BODY |
| A123 | selective prefix-cache retention for sliding-window KV (DeepSeek-V4) | vLLM (ywang96) | PR #43447 | https://github.com/vllm-project/vllm/pull/43447 | merged 2026-06-04 | READ BODY |
| A124 | deterministic inference and which attention backends keep radix cache | SGLang | deterministic inference doc (`--enable-deterministic-inference`) | https://docs.sglang.io/docs/advanced_features/deterministic_inference | shipped | READ BODY |
| A125 | metrics plus request dump/replay and crash dump for KV debugging | SGLang | observability doc | https://docs.sglang.io/docs/advanced_features/observability | shipped | READ BODY |
| A126 | HiCache consistency under pipeline parallelism | SGLang (whybeyoung) | issue #22607 "HiCache Consistency Fix Plan" (COMPLETED) | https://github.com/sgl-project/sglang/issues/22607 | shipped (issue closed COMPLETED) | READ BODY |
| A127 | prefix caching measurably changes agent trajectories | arXiv 2609.04748 authors | "Same Request, Different Answer" cache-induced divergence study | https://arxiv.org/abs/2609.04748 | published 2026-09-04 | READ BODY |
| A128 | per-step error certificates for KV eviction | arXiv 2607.21475 authors | Error Certificates for KV-Cache Eviction via Randomized Design | https://arxiv.org/abs/2607.21475 | published 2026-07-23 | READ BODY |
| A129 | query visibility decides KV compression rankings | arXiv 2607.11942 authors | matched-budget audit | https://arxiv.org/abs/2607.11942 | published 2026-07-11 | READ BODY |
| A130 | runtime observability that localizes silent KV corruption | arXiv 2608.05863 authors | Runtime Observability for Heterogeneous Attention Memory | https://arxiv.org/abs/2608.05863 | published 2026-08-06 | READ BODY |
| A131 | SSD-resident KV without the kernel filesystem | arXiv 2606.14779 authors | unified KV pooling + KV-passthrough (SPDK) | https://arxiv.org/abs/2606.14779 ; https://arxiv.org/html/2606.14779v1 | published 2026-06-10 | READ BODY (verifier: the quoted testbed sentences live on the HTML page, not the abstract URL) |

### A-notes

- Merged within paged/block management: RadixAttention (S1 CLOSED-2 + S2 CLOSED-3 → A26); hybrid KV cache manager doc (S1 CLOSED-3 + S2 CLOSED-12 → A5); PR #24486 with the `--mamba-block-size` flag row (S1 CLOSED-4 + S1 CLOSED-9 → A6).
- Merged within prefix caching: APC doc and hash-algorithm row (S2 CLOSED-1 + S14 CLOSED-1 → A13); partial/sub-block hits with the shared-prefix-junction checkpoint flag (S1 CLOSED-7 + S2 CLOSED-9 + S8 CLOSED-3 + S13 CLOSED-3 → A17); retention interval (S2 CLOSED-11 + S9 CLOSED-7 → A19); session radix cache with its implementing PR (S2 CLOSED-5 + S9 CLOSED-10 + S13 CLOSED-1 → A21); HiCache design/L3/tier rows plus the host-tier>device-tier constraint (S2 CLOSED-6, S5 CLOSED-6, S5 CLOSED-17, S7 CLOSED-7, S9 CLOSED-8, S14 CLOSED-10 → A24).
- Merged within eviction policy: pluggable offload eviction (S3 CLOSED-1 + S5 CLOSED-23 + S9 CLOSED-4 → A27); SGLang policy selection (S2 CLOSED-4 + S3 CLOSED-2 + S9 CLOSED-9 → A20); ARC policy with the ARC-style paper (S9 CLOSED-3 + S3 CLOSED-8 → A32); attention-sink shipping with the sink-necessity proof (S3 CLOSED-6 + S8 CLOSED-6 + S3 CLOSED-7 → A31).
- Merged within quantization: FP8 blog with its two-level-accumulation and skip-layers material (S4 CLOSED-1 + S4 CLOSED-5 + S4 CLOSED-12 + S11 CLOSED-1 → A43); TurboQuant PR with its theory paper (S4 CLOSED-4 + S4 CLOSED-9 + S6 CLOSED-15 → A45); three low-bit papers kept as one granularity/outlier row (S4 CLOSED-7 + S4 CLOSED-8 + S4 CLOSED-11 → A47).
- Merged within offloading/tiered storage: native CPU offload (S5 CLOSED-1 + S5 CLOSED-2 + S11 CLOSED-2 → A52); tiered framework (S5 CLOSED-3 + S5 CLOSED-4 + S9 CLOSED-14 + S14 CLOSED-3 → A53); P2P tier (S7 CLOSED-4 → A59) and cross-process key determinism (S7 CLOSED-5 → A60) kept separate; CacheGen with Pensieve (S5 CLOSED-10 + S5 CLOSED-11 → A66).
- Merged within sharing/disaggregation: connector suite with the per-engine connector listing (S7 CLOSED-1 + S10 CLOSED-7 → A79); NIXL lease with its PR and role flags (S7 CLOSED-6 + S9 CLOSED-2 + S10 CLOSED-3 → A82); Mooncake paper with its shipped connector (S5 CLOSED-8 + S7 CLOSED-11 + S12 CLOSED-10 → A105); LMCache paper (S5 CLOSED-9 + S7 CLOSED-10 → A83); Dynamo tuning guidance with the throughput caveat (S10 CLOSED-6 + S10 CLOSED-8 → A101); vLLM DCP+PCP PRs with the deployment doc (S8 CLOSED-4 + S12 CLOSED-1, -2, -3, -4, -5, -6 → A89); SGLang prefill+decode CP (S8 CLOSED-5 + S12 CLOSED-7, -8 → A90); Ring Attention with Star Attention (S8 CLOSED-11 + S12 CLOSED-13 → A95); DistServe with Splitwise (S12 CLOSED-11 + S12 CLOSED-12 → A104).
- Merged within agentic/multi-turn: AgentKVShift with SideQuest (S13 CLOSED-5 + S13 CLOSED-6 → A108); FlowKV with EpiCache and SONIC (S13 CLOSED-7, -8, -13 → A109); GrowPage with Zipage (S13 CLOSED-9 + S13 CLOSED-12 → A110).
- Merged within observability: hit-rate counters and connector stats with block residency/lifetime metrics (S9 CLOSED-5 + S14 CLOSED-4 + S14 CLOSED-5 + S14 CLOSED-6 → A121).
- Status corrections applied from the verification reports: A81 reads "partially merged — #54307 open" (S7 CLOSED-3); A79 does not repeat the retracted "18 registered connectors" count (S7 CLOSED-1: 16 in the inspected tree, 17 on current main); A66 corrects CacheGen's hardware to a four-GPU A40 server and is labelled TITLE + ABSTRACT ONLY (S5 CLOSED-10); A107 is added because S5 OPEN-10 (#46971) is CLOSED/COMPLETED by merged PR #46972 (2026-07-07); A113 records that no guard PR shipped and that the migration fix #27067 was closed as superseded by #27326 (S11 CLOSED-11); A50 discloses that llama.cpp #26423's surviving title contradicts the retraction summary (S11 CLOSED-4); A49's merge state and quote come from the GitHub search API rather than a page body (S11 CLOSED-5).
- Evidence-level corrections applied: A7 and A12 are TITLE + STATE ONLY (S1 CLOSED-5, -15); the S6 arXiv paper rows are ABSTRACT ONLY (13 rows) and S6 CLOSED-15 is PR METADATA ONLY, so A45's PR half is metadata-only (S6 verifier); A31's theory quote is APPROXIMATE (dropped "simple", S3 CLOSED-7); A110's Zipage half is TITLE ONLY and names the paper title rather than the in-paper system name (S13 CLOSED-10, CLOSED-12); A41 is TITLE ONLY (S13 CLOSED-10); A63's Tutti result quote is re-labelled a paraphrase (S5 CLOSED-21); A103's "5% F1" and A104's Splitwise §IV-E now use the verifier's wording ("workload"); A131 cites the HTML URL because its quotes are not on the abstract page (S14 CLOSED-17).
- Corrections that remove claims rather than rows: S1 CLOSED-11's corroborating quotes are attributed to issue #45238, not #54199 (A11); S6 CLOSED-10's "abstract ends mid-sentence at the hardware name" is false (A75); S5 CLOSED-17's quoted evidence did not answer its own heading, so it is folded into A24 rather than given a row; S13's line numbers come from a vLLM ~0.11.2 mirror and have drifted on live main, so no line numbers are presented as line-precise anywhere in this section; S1 CLOSED-10's cited line numbers are a snapshot of the remote copy, so only the symbol name is given (A10).
- Rows dropped for having no usable URL or no retrievable artifact: none — every source row in `_in_A.tsv` is represented above, except the duplicate rows merged into the groups listed in the first nine bullets.

---

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

---

## C. ATTEMPTED-AND-ABANDONED cells (most valuable!)

### C.1 Closed-unmerged pull requests

#### C1. vLLM `[Core] Cache policy framework` — the `CachePolicy` eviction abstraction for V0, closed unmerged
- **WHO:** `ShawnD200` (author) in vllm-project/vllm; closed by `hmellor`, a vLLM Member.
- **URL:** https://github.com/vllm-project/vllm/pull/11928
- **Closure status:** PR closed unmerged by a named human maintainer (`hmellor`, vLLM Member), not a bot.
- **STATED REASON, VERBATIM:** Hi @ShawnD200 , I'm going to close this PR for the following reasons: It is for V0, for which we are not accepting new features (bugfixes only) It's 4 months stale You have the RFC to continue the discussion and make a new PR for V1 when the time comes! — `hmellor`, Member, May 6, 2025.
- **Date:** created 2025-01-10T11:03:04Z; closed 2025-05-06T09:27:09Z.
- **Evidence:** READ BODY

#### C2. vLLM `[Core] Support sparse KV cache framework` — H2O/SnapKV/Scissorhands/StreamingLLM eviction framework
- **WHO:** `chizhang118` (author) in vllm-project/vllm; closed by `hmellor`, a vLLM Member. This is the implementation PR for the still-open RFC #5751.
- **URL:** https://github.com/vllm-project/vllm/pull/5752
- **Closure status:** PR closed unmerged by a named human maintainer (`hmellor`); the RFC it implements (#5751) stays open.
- **STATED REASON, VERBATIM:** Closing as stale. If you plan to continue this work, feel free to re-open or make a new PR. — `hmellor`, Member, Feb 28, 2025. - Additional in-thread blocker, VERBATIM (from a reviewer, before closure): "Hi @chizhang118 , I'm very interested in your design doc to see the detail of sparse kv cache framework. but it's not available now, may you share a valid link? Thanks!" — the RFC's linked design doc was dead.
- **Date:** created 2024-06-21T20:31:34Z; closed 2025-02-28T13:47:28Z.
- **Evidence:** READ BODY

#### C3. SGLang `[Feature] Semantic eviction for radix cache` — agent-workload reset/summarisation pruning
- **WHO:** `rexxy-sasori` (author) in sgl-project/sglang; closed by `hnyls2002`, a SGLang Collaborator (human, not a bot).
- **URL:** https://github.com/sgl-project/sglang/pull/20088
- **Closure status:** PR closed unmerged by a human Collaborator (`hnyls2002`); the semantic-event path was never added to main.
- **STATED REASON, VERBATIM:** "Thanks @rexxy-sasori ! This is still a draft with no commits since 2026-03-08, and it carries local dev artifacts that cannot merge - a .gitmodules adding external/sglang pointing at a personal fork, plus docker/dev.Dockerfile . Several touched files have also moved since ( srt/metrics/collector.py -> srt/observability/metrics_collector.py , scheduler_output_processor_mixin.py -> srt/managers/scheduler_components/ , Sphinx docs/ -> Mintlify docs/docs/ ). semantic_event is still absent from main, so a rebased, artifact-free PR would be welcome. Please reopen if I've missed something."
- **Date:** created 2026-03-07, closed 2026-08-10 (the S13 duplicate row for the same PR had no separable closure)
- **Evidence:** READ BODY

#### C4. SGLang TurboQuant KV-cache PR — closed unmerged as one of three concurrent proposals
- **WHO:** YanGaev2 (author); closed by hnyls2002 (SGLang collaborator)
- **URL:** https://github.com/sgl-project/sglang/pull/22048
- **Closure status:** closed unmerged by a human collaborator, consolidating on #21419 (itself still open).
- **STATED REASON, VERBATIM:** "Thanks @YanGaev2 ! There are three concurrent TurboQuant proposals and we are consolidating on #21419 , which reuses the existing flashinfer_backend instead of adding a parallel flashinfer_tq backend to attention_registry.py , is roughly 1500 lines smaller, and ships an end-to-end accuracy eval ( python/sglang/test/simple_eval_niah.py ) rather than unit tests plus a single-request throughput number. This PR also targets two paths that no longer exist - python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py and the retired test/srt/ suite dir. The fused Triton encode/decode kernels and cuda-graph handling here are the strongest part - please consider contributing those on top of #21419 . Please reopen if I've missed something."
- **Date:** opened Apr 3, 2026; closed Aug 11, 2026 by hnyls2002
- **Evidence:** READ BODY

#### C5. vLLM Triton INT4 / INT2 per-token-head KV quantisation — superseded by a narrower PR; the INT2 half dropped
- **WHO:** JartX (author); closed by the author after review by tdoublep
- **URL:** https://github.com/vllm-project/vllm/pull/40633 | https://github.com/vllm-project/vllm/pull/39074
- **Closure status:** PR closed unmerged by its own author in favour of #40835 (which merged). The companion PR #39074 was also closed unmerged with only bot text on the page.
- **STATED REASON, VERBATIM:** "@tdoublep closed in favor of: #40835" — and, on the companion PR #39074 the only stated text is the `mergify` bot notice: "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @JartX . (posted by the `mergify` bot, May 23, 2026)"
- **Date:** opened Apr 22, 2026; closed May 25, 2026. Note the preceding automated comment in the same thread is from the `mergify` bot ("This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @JartX ."), but the actual closure is a human author decision with the stated reason above.
- **Evidence:** READ BODY

#### C6. vLLM TurboQuant KV cache with PolarQuant + QJL — closed unmerged as a duplicate design
- **WHO:** allaspectsdev (author); asked to close by gaby (vLLM maintainer)
- **URL:** https://github.com/vllm-project/vllm/pull/38662
- **Closure status:** closed unmerged by its author on a maintainer request; the QJL half did not survive into the shipped backend.
- **STATED REASON, VERBATIM:** "@allaspectsdev Please close this. See #38479"
- **Date:** opened Mar 31, 2026; "allaspectsdev closed this Apr 7, 2026". The review on this PR also flagged, verbatim, "missing scaling factors in the QJL bias correction", a stack buffer overflow in the polar decode path, and hardcoded 256-element buffers that "assume head_size is a power of 2".
- **Evidence:** READ BODY

#### C7. vLLM’s first INT8 KV-cache-quantisation PR — closed without landing; the follow-up split PRs never landed either
- **WHO:** AniZpZ (author); closed by simon-mo (vLLM collaborator)
- **URL:** https://github.com/vllm-project/vllm/pull/1112
- **Closure status:** closed unmerged by a maintainer (`simon-mo`); the maintainer’s reason addresses W8A8/INT8 support generally, not KV quantisation specifically.
- **STATED REASON, VERBATIM:** "Closing as we do have W8A8 and Int8 support nowadays. 🙏"
- **Date:** opened Sep 20, 2023; closed Oct 1, 2024 by simon-mo. Immediately after the close, a user asked the still-unanswered question, verbatim: "Hi, is there any plan to support int4 KVCacheQuant?" This is the historical root of OPEN-6.
- **Evidence:** READ BODY

#### C8. vLLM CPU KV “swap space” — never allocated in V1, parameter removed; the abandonment PR was picked up by another author
- **WHO:** mcelrath (reporter), vLLM; removal merged by DarkLight1337
- **URL:** https://github.com/vllm-project/vllm/issues/27984 | https://github.com/vllm-project/vllm/pull/27988 | https://github.com/vllm-project/vllm/pull/36216 | https://github.com/vllm-project/vllm/pull/35652
- **Closure status:** issue closed with the feature deleted: `--swap-space` and the `swap_space` field were removed by merged PR #36216. The original removal PR #27988 was closed unmerged and picked up by another author; the later per-layer `SwapConnector` revival (#35652) was closed unmerged by its author the day it opened, with no stated reason on the page.
- **STATED REASON, VERBATIM:** Since the deprecation of best_of sampling in #13997 (see also #13361 ), the swap_space parameter is useless and unused in the codebase. To make matters worse, it seems best_of was never properly implemented because num_cpu_blocks in EngineCore::_initialize_kv_caches() is hard coded to zero at vllm/v1/engine/core.py:252 and always has been according to the git history. and "This is probably why best_of had no usage -- it never worked and has always been disabled." — and, the replacement PR #36216 records the abandonment: The replacement PR records the abandonment verbatim: "Picks up the abandoned PR #27988 as suggested by @DarkLight1337 in #27984 ." The replacement PR body also states the technical reason verbatim: "In the V1 engine, num_cpu_blocks is hardcoded to 0 ( v1/engine/core.py:287 ), confirming swap space was never allocated or used."
- **Date:** #27984 opened 2025-11-03, closed 2026-03-07; #27988 opened 2025-11-03, closed 2026-03-07; #36216 merged 2026-03-07; #35652 opened and closed 2026-03-01
- **Evidence:** READ BODY

#### C9. vLLM `Kvserve fusion` (KV compression + Mooncake connector) — 33 commits, never merged
- **WHO:** VerrPower, vLLM PR #51648
- **URL:** https://github.com/vllm-project/vllm/pull/51648
- **Closure status:** PR closed unmerged by maintainer `hmellor`; the only machine-recorded reason on the page is the `mergify` rebase request — no human maintainer rationale was posted.
- **STATED REASON, VERBATIM:** "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @VerrPower ."
- **Date:** opened 2026-08-10; closed 2026-08-10
- **Evidence:** READ BODY

#### C10. vLLM `add KV cache tiering residency and lifecycle management` — self-closed as targeting the wrong repository
- **WHO:** JieYang2001 (intellistream), vLLM PR #48837
- **URL:** https://github.com/vllm-project/vllm/pull/48837
- **Closure status:** PR closed unmerged by its own author, who renamed the title to record the reason.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** The author retitled the PR and closed it the same day; the recorded title is exactly: "[Closed: wrong repository]feat: add KV cache tiering residency and lifecycle management"
- **Date:** opened 2026-07-16; closed 2026-07-16
- **Evidence:** READ BODY

#### C11. vLLM `Fail fast when CPU offload region exceeds available space` — closed because main already carried the fix
- **WHO:** Thibaultjaigu, vLLM PR #47073
- **URL:** https://github.com/vllm-project/vllm/pull/47073
- **Closure status:** PR closed unmerged by its own author as superseded by already-merged work (#50358, #48414, #52596).
- **STATED REASON, VERBATIM:** Closing: main already carries this fix. #50358 (merged 2026-08-05) added check_shm_free_space(self.total_size_bytes) in the creator path with the file cleanup on failure, which is exactly the shape @orozery asked for here, and the region has since been reworked further in #48414 and #52596 . Nothing left for this branch to add. Thanks for the review.
- **Date:** opened 2026-06-29; closed 2026-09-12
- **Evidence:** READ BODY

#### C12. vLLM `Bypass power-of-2 rounding in KV offload CPU pinned allocation` — folded into a different PR
- **WHO:** vLLM PR #48436, author procr1337 (verifier-corrected; `woosebastian` only cross-referenced the PR), closed by orozery
- **URL:** https://github.com/vllm-project/vllm/pull/48436
- **Closure status:** PR closed unmerged “in #50094” by maintainer `orozery`; no prose rationale was posted. (The evidence file names the author as `woosebastian`; the verifier corrected this to `procr1337`.)
- **STATED REASON, VERBATIM:** "orozery closed this in #50094"
- **Date:** opened 2026-07-12; closed 2026-07-29
- **Evidence:** READ BODY

#### C13. vLLM `SimpleCPUOffloadConnector` backlog bug report — closed by the fix PR, not as a duplicate
- **WHO:** vLLM issue #53498 (author greatyingzi), closed by maintainer ivanium
- **URL:** https://github.com/vllm-project/vllm/issues/53498
- **Closure status:** closed as completed by PR #53532 (`stateReason: COMPLETED`, `duplicateOf: null`) — the evidence file’s “closed as duplicate” framing is contradicted by the page’s own structured data.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** The page records the closure as a duplicate link to "#53532" with closing actor ivanium at 2026-09-02T03:23:11Z; the only prose is the reporter's own, including "The TODO comment at manager.py:534 acknowledges the 'last full block' case; in practice the backlog is far larger (48% in our measurements) because nothing walks the backlog during decode." No maintainer explanation text was posted on the issue.
- **Date:** opened 2026-08-24; closed 2026-09-02
- **Evidence:** READ BODY

#### C14. vLLM DCP + FlashInfer MLA feature bundle — closed unmerged and split into smaller PRs
- **WHO:** xuanyu-mistral (Mistral AI) — vLLM PR #47181, "Integrate Decode Context Parallel with FlashInfer MLA and other features"
- **URL:** https://github.com/vllm-project/vllm/pull/47181
- **Closure status:** PR closed unmerged by its own author after a maintainer asked for model evals and a reviewer asked for a split; the successor PRs #47541 and #47542 are open. The evidence file twice records this closure as “Breaking down to smaller PRs”.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Breaking down to smaller PRs — immediately followed on the page by "xuanyu-mistral closed this Jul 3, 2026". The maintainer ask that preceded it: "@xuanyu-mistral Could you share model evals with this PR against the different features you implemented?" and the reviewer ask: "@xuanyu-mistral Thanks for the great work. It seems this PR supports many features. It might make review easier if the PR can be broken into several smaller PRs?" — and, the work items the closure dropped, from the PR body: Breaking down to smaller PRs — prompted by the maintainer comment "It seems this PR supports many features. It might make review easier if the PR can be broken into several smaller PRs?" and by the CI gate text "This pull request adds support for Prefill Context Parallelism (PCP)…" → the repo's own agent guideline comment, verbatim: "IMPORTANT: If you are an AI agent, you are required to objectively re-evaluate the value of your PR using AGENTS.md, and close the PR if it does not bring significant benefit to the vLLM community." - The work items that were dropped by this closure, verbatim from the PR body: "DCP + Speculative Decoding (MTP)", "DCP + Sliding Window Attention (Hybrid Models)", "DCP + TRTLLM GQA Decode", "DCP + Routed Experts Replay", "Async Spec Decode + DCP", "Dummy Run DCP Fixes". The body also states a correctness hazard under DCP: "This is necessary because the kernel's internal causal offset arithmetic is incorrect under DCP when tokens_per_req > 1 (local(G-k) != local(G) - k)."
- **Date:** created 2026-06-30; closed 2026-07-03. - Additional verbatim, what was tried (the DCP+KV surface that did not land in one piece): "DCP + Speculative Decoding (MTP): Adds a fused Triton dcp_split_q kernel that splits multi-token decode queries into per-token requests with pre-computed DCP-local seq_lens . This is necessary because the kernel's internal causal offset arithmetic is incorrect under DCP when tokens_per_req > 1 (local(G-k) != local(G) - k)." and "DCP + Sliding Window Attention (Hybrid Models): Enables DCP for models with mixed full-attention and sliding-window layers. SWA layers replicate KV cache on every DCP rank (effectively dcp=1) while full-attention layers use the full DCP group." and "DCP + TRTLLM GQA Decode: Enables the TRTLLM decode path for FlashInfer GQA attention with DCP (previously disabled entirely)." and "NaN × 0 guard in DCP output correction: when a rank has zero local KV tokens, the attention output is NaN and the LSE factor is 0. NaN * 0 = NaN, so we explicitly zero out."
- **Evidence:** READ BODY

#### C15. vLLM layerwise KV transfer in P/D disaggregation — measured worse than bulk transfer, closed unmerged
- **WHO:** chenqianfzh (ByteDance IAAS), vLLM PR #12523
- **URL:** https://github.com/vllm-project/vllm/pull/12523 | https://github.com/vllm-project/vllm/issues/10818
- **Closure status:** PR closed unmerged after its own author measured the layerwise method as slower; the roadmap item that tracked the same work was later closed `NOT_PLANNED`.
- **STATED REASON, VERBATIM:** "the layerwise method performs worse as it needs to retrieves layer_number tensors, so it requires more overhead. I will continue to work on the solution that the consumer starts to receive the KV cache once the producer completes a layer." — and, the roadmap entry that tracked it, verbatim: "- [ ] [Perf] layer-by-layer pipelining (#12523)"
- **Date:** PR created 2025-01-28, closed 2025-03-19; roadmap issue #10818 created 2024-12-02, closed not planned
- **Evidence:** READ BODY

#### C16. vLLM Extensible (growable) KV cache — 39-commit series closed unmerged
- **WHO:** `njhill` / `lwilkins` (Red Hat) — a 39-commit series
- **URL:** https://github.com/vllm-project/vllm/pull/50779
- **Closure status:** closed unmerged; the closure comment was retrieved by the verifier (“Replaced by …”) after the evidence file reported no justification text.
- **STATED REASON, VERBATIM:** "Replaced by https://github.com/vllm-project/vllm/pull/56492."
- **Date:** created 2026-08-02, closed 2026-09-11
- **Evidence:** READ BODY

#### C17. vLLM host-resident KV cache offloading for sparse MLA decode — closed unmerged in favour of a parallel implementation
- **WHO:** `LCAIZJ`
- **URL:** https://github.com/vllm-project/vllm/pull/51270
- **Closure status:** closed unmerged; the author’s own closure rationale was retrieved by the verifier after the evidence file reported no retrievable reason.
- **STATED REASON, VERBATIM:** "Since this PR and @MatthewBonanni's PR #51323 implement the same feature … I find PR #51323 to be a better fit for the first version to be merged. Therefore, I plan to close this PR and continue optimizing this feature based on PR #51323."
- **Date:** created 2026-08-06, closed 2026-08-11
- **Evidence:** READ BODY

#### C18. vLLM semantic checkpoints for recurrent-state prefix caching — `[Exp]` prototype closed unmerged
- **WHO:** `qianlihuang`
- **URL:** https://github.com/vllm-project/vllm/pull/49574
- **Closure status:** closed unmerged; no explicit closure comment exists on the page (the verifier confirmed this row’s “reason not retrieved” is defensible).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** No justification text retrievable. Commit subject of `[PATCH 1/5]`, verbatim: "\[Core\] Prototype semantic cache checkpoints" with body "Accept trusted frontend token boundaries and propagate them through the engine and scale-out request paths. Align recurrent prefill chunks so those boundaries can materialize under sparse retention." **No reason is asserted.**
- **Date:** created 2026-07-23, closed 2026-09-11
- **Evidence:** READ BODY

#### C19. vLLM rejecting prefix caching for pooling / encoder-only models — sibling PR closed unmerged
- **WHO:** `qsxustc` (closed) vs `HsukqiLee` (still open, #55159)
- **URL:** https://github.com/vllm-project/vllm/pull/55160
- **Closure status:** closed unmerged; no formal maintainer closure sentence exists, but a review comment on the page records the de-facto reason (the sibling PR #55159 targets the same function). The review comment is reproduced here exactly as the verification report renders it, including its ellipsis.
- **STATED REASON, VERBATIM:** "duplicates **#55159** … which targets the exact same function" — and, the commit subject of the closed PR itself: "\[Bugfix\] Reject unsupported prefix caching for pooling models"
- **Date:** created 2026-09-03, closed 2026-09-07
- **Evidence:** READ BODY

#### C20. TensorRT-LLM Blackwell-native FlashInfer context attention — closed as relocated out of the repository
- **WHO:** NVIDIA maintainers
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/issues/16860
- **Closure status:** closed as not planned — a relocation, not a technical rejection.
- **STATED REASON, VERBATIM:** "Moved to NVIDIA-dev/paragraf#140 so the work is tracked in the Paragraf repository."
- **Date:** created and closed 2026-07-25
- **Evidence:** READ BODY

#### C21. TensorRT-LLM closed-unmerged PR — a closure reason does exist on the page
- **WHO:** `trtllm-agent` (body author); closed by `yuxianq`
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/pull/18531
- **Closure status:** closed unmerged by `yuxianq`; the verifier retrieved the closure comment after the evidence file asserted no reason existed (#18544 is merged). Concerns FMHA routing rather than strictly KV cache.
- **STATED REASON, VERBATIM:** "Close since we already have a bugfix PR https://github.com/NVIDIA/TensorRT-LLM/pull/18544"
- **Date:** closed 2026-09-02
- **Evidence:** READ BODY

#### C22. vLLM unified `context_parallel_size` configuration/parallel-group design — superseded by the split PCP/DCP config
- **WHO:** qiruiyangmeta
- **URL:** https://github.com/vllm-project/vllm/pull/26057 | https://github.com/vllm-project/vllm/pull/26058
- **Closure status:** PRs closed unmerged by a human maintainer (`hmellor`) with a technical supersede reason — a named-maintainer verdict rather than bot text or a stale timer.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** This PR has been superseded by the Prefill Context Parallel (PCP) and Decode Context Parallel (DCP) implementations that have since been merged: [Feature] Prefill Context Parallel (PCP) basic support #28718 (PCP basic support) [Feature] Support Decode Context Parallel (DCP) for MLA #23734 (DCP for MLA) [DCP] Support Decode Context Parallel (DCP) for GQA with FlashAttention #24864 (DCP for GQA with FlashAttention) The codebase now uses separate prefill_context_parallel_size / decode_context_parallel_size config fields and get_pcp_group() / get_dcp_group() rather than the unified context_parallel_size / get_cp_group() approach proposed here. Thank you for the contribution! - The abandoned design's own sharding scheme, verbatim from the PR body: "Causal attention imposes a varying computational load for each token, as shown in the following figure. To ensure an even workload distribution, tokens should be partitioned across different context parallelism (CP) ranks. Specifically, the sequence is divided into 2 × cp_world_size chunks. Each CP rank i is assigned both the i-th chunk and the (2 × cp_world_size - i - 1)-th chunk."
- **Date:** #26057/#26058 created 2025-10-01, closed 2026-03-06
- **Evidence:** READ BODY

#### C23. vLLM MoRIIO multi-node TP16 prefill→decode KV dispatch — closed unmerged as superseded by three merged PRs
- **WHO:** chaeminlim-mb (+ edwinlim0919, MangoBoost)
- **URL:** https://github.com/vllm-project/vllm/pull/45228
- **Closure status:** PR closed unmerged; the closer comment is a human statement of supersession by #46116, #46115 and #47495.
- **STATED REASON, VERBATIM:** "The issue that this PR targets has been addressed by #46116, #46115, and #47495, all of which have been merged. This PR can be safely closed."
- **Date:** created 2026-06-11, final close ~2026-08-14 (#45228; companion #43063/#45230 quotes are sub-agent-retrieved)
- **Evidence:** READ BODY

#### C24. TensorRT-LLM `Reuse V2 decode KV cache across turns` — closed unmerged by a maintainer as a deferral
- **WHO:** erictsai-nv (PR author), closed by nvpohanh (Collaborator) on NVIDIA/TensorRT-LLM
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/pull/18711
- **Closure status:** PR closed unmerged by a human Collaborator (`nvpohanh`) — a direct, deliberate abandonment of cross-turn decode-KV reuse in the V2 cache manager, phrased as “for now”.
- **STATED REASON, VERBATIM:** "Closing this for now. I will ask @erictsai-nv to continue working on this when he is back."
- **Date:** last review activity 2026-09-04; closed 2026-09-08 (closedTime 2026-09-08T00:51:04Z)
- **Evidence:** READ BODY

#### C25. TensorRT-LLM inclusive host KV-cache tier (shadow reuse) — closed unmerged by its author after the reviewer never approved
- **WHO:** reasonsolo (PR author, closed it), reviewed by lowsfer (Member) on NVIDIA/TensorRT-LLM
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/pull/15828
- **Closure status:** PR closed unmerged by its author. NO CLOSURE COMMENT WAS POSTED — the quoted maintainer line is the last substantive comment on the design, not a closure statement.
- **STATED REASON, VERBATIM:** "Please hold on until bug reporter's response. It's a design choice. It uses more D2H bandwidth but D2H bandwidth is under-utilized anyway, and it improves cache hit rate because it avoids duplication and can hold more data."
- **Date:** opened 2026-07-02; closed 2026-07-31 (closedTime 2026-07-31T02:13:07Z)
- **Evidence:** READ BODY

#### C26. SGLang Router — honoring KV-event storage tiers in the cache-aware tree (closed as superseded, then re-split)
- **WHO:** Kangyan-Zhou (Collaborator) on SGLang
- **URL:** https://github.com/sgl-project/sglang/pull/38720
- **Closure status:** PR closed unmerged by its author as superseded; the same change was re-split into the stacked series #39108/#39109/#39110, all still open — the work moved rather than stopped.
- **STATED REASON, VERBATIM:** "Superseded — split into a stacked series and closed. The same change, byte-for-byte, now lands as #39108 (the tier-aware tree), #39109 (the sgl_router_kv_* /metrics series) and #39110 (the real-GPU e2e proof), all with their heads on sgl-project/sglang so they can chain base-to-head. Review there; this branch is gone."
- **Date:** opened 2026-09-09; closed 2026-09-11 (closedTime 2026-09-11T18:23:38Z)
- **Evidence:** READ BODY

#### C27. vLLM fine-grained prefix-cache hits for sliding-window groups — implementation closed unmerged twice by its own author
- **WHO:** vLLM contributor RichApple123 — PR #54319, then its revision PR #54397
- **URL:** https://github.com/vllm-project/vllm/pull/54397
- **Closure status:** both PRs (#54319 and its revision #54397) closed unmerged by their own author with no closure comment. The only retrievable statement of the blocking condition is an automated merge-conflict notice.
- **STATED REASON, VERBATIM:** "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @RichApple123 ." — and, the design intent that is now unmerged, from the PR body: This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @RichApple123 . — this is an automated GitHub notice, **not** a maintainer statement. The PR body itself records the design intent that is now unmerged: "Fixes #53786 by allowing `SlidingWindowManager` to participate safely in fine-grained hybrid prefix-cache lookup when its physical KV page is larger than the configured hash unit. Previously, one SWA group with a larger physical page disabled `enable_partial_hash_hits` for the entire hybrid coordinator." The predecessor #54319's body states: "This change only makes prefix endpoints hash-granular".
- **Date:** #54319 created 2026-08-29, closed 2026-08-29. #54397 created 2026-08-30, closed 2026-09-11.
- **Evidence:** READ BODY

### C.2 wontfix / not-planned / by-design / out-of-scope closures

#### C28. vLLM align-mode checkpoint retention visibility — author-closed as redundant, not a maintainer rejection
- **WHO:** vLLM issue #53595, author jschmied (a CONTRIBUTOR), self-closed
- **URL:** https://github.com/vllm-project/vllm/issues/53595
- **Closure status:** closed as not planned (`stateReason=NOT_PLANNED`) by the author himself; explicitly not a stale-bot closure and not a maintainer rejection. The report was folded into the fix PR #53596.
- **STATED REASON, VERBATIM:** "Closing: this was redundant with the PR that fixes it and with the PR that already describes the same invisibility." — and, the same closing comment, with the follow-up line: Closing: this was redundant with the PR that fixes it and with the PR that already describes the same invisibility. — followed by "- **#53596** carries the full report — cause, the asymmetry with the announced `mamba_cache_mode` default, and the measurement — and is where the fix is reviewed."
- **Date:** created 2026-08-24, closed 2026-08-24
- **Evidence:** READ BODY

#### C29. Sizing the SGLang hybrid Mamba state pool from explicit `max_running_requests` — marked not planned
- **WHO:** SGLang issue #33004 (feature request); maintainers marked it not planned; reporter is bash99 (Ben Rood)
- **URL:** https://github.com/sgl-project/sglang/issues/33004
- **Closure status:** closed as not planned by a maintainer (`huangzhilin-hzl`) on the day of filing. The maintainer’s own stated reason is NOT present in the retrieved payload; the only rationale text on the page is the reporter’s reaction.
- **STATED REASON, VERBATIM:** "I am disappointed that this issue was marked as 'not planned'."
- **Date:** created 2026-07-31, closed 2026-07-31
- **Evidence:** READ BODY

#### C30. Naive round-robin block allocation for sliding-window layers — deliberately designed out of the shipped hybrid KV manager
- **WHO:** vLLM maintainers, in the shipped hybrid KV cache design doc
- **URL:** https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/hybrid_kv_cache_manager.md
- **Closure status:** by design — considered and deliberately not adopted in the shipped vLLM hybrid KV cache design doc; a closed design decision, not a merged feature.
- **STATED REASON, VERBATIM:** "For sliding window attention layers, a naive implementation for memory allocation is to allocate `sliding_window_size` blocks and fill in the blocks in a round-robin way. But this naive implementation is not compatible with prefix caching so we didn't pick this design. In vLLM, we allocate different blocks for different tokens and free blocks that are outside the sliding window."
- **Date:** doc states it was written against commit `458e74` (the doc carries no date in the retrieved copy; the linked commit is the reference point)
- **Evidence:** READ BODY

#### C31. Cache-affinity-aware request ordering in the vLLM V1 scheduler (`CacheAffinityScheduler`) — dropped after the +7.4% result was shown to be a benchmark artefact
- **WHO:** `kliukovkin` (vLLM contributor), RFC #42185 / CacheAffinityScheduler plugin; also mirrored as S7-ABANDONED-1, S11-ABANDONED-1 and S13-ABANDONED-10
- **URL:** https://github.com/vllm-project/vllm/issues/42185
- **Closure status:** closed as not planned (`stateReason=NOT_PLANNED`). VERIFIER CORRECTION: the closer was the human RFC author `kliukovkin`, who posted a closing comment — NOT the stale bot and not a maintainer. The stale label/notice is genuinely present but is only a notice. The evidence file’s “the closure actor is the stale bot / no human reason exists” is false.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** **Decision:** Dropping PR — RAG improvement below threshold, no measurable benefit on correct async baseline. ... "RAG mean delta: **+0.2%** (noise). Decision threshold: ≥+5%." ... and the root cause of the earlier promising number: "### Why the original +7.4% RAG result was wrong — `CacheAffinityScheduler` initially extended synchronous `Scheduler` instead of `AsyncScheduler`." The author's own closing comment is also verbatim: "Closing this from my side: since filing, the ecosystem has moved in exactly this direction — prefix-cache-aware routing landed as a first-class concern in llm-d and AIBrix, which validates the premise better than further discussion here would." The automated message that also appears on the page is verbatim: "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!"
- **Date:** created 2026-05-09, closed 2026-08-18 (ClosedEvent 2026-08-18T00:44:01Z by kliukovkin)
- **Evidence:** READ BODY

#### C32. Sparse KV cache management framework RFC — closed “COMPLETED” while labelled stale, with only one of four tasks delivered
- **WHO:** `ShawnD200` (RFC author) in vllm-project/vllm; discussed by `heheda12345` (Collaborator).
- **URL:** https://github.com/vllm-project/vllm/issues/12254
- **Closure status:** `state=CLOSED`, `stateReason=COMPLETED`, labels include `RFC` and `stale`. No human ClosedEvent or explanatory comment was found; the only merged artifact (PR #12608) covers task (1) of four, and the CachePolicy interface for H2O/FastGen does not exist in the tree today. Treat “COMPLETED” as not substantiated for the eviction-policy content.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** the RFC's own goal statement now abandoned in practice: "CachePolicy is a general interface that manages KV cache allocations. It generalizes from basic cache behavior that always allocates new slots/blocks to append new tokens to the more sophisticated that dictates how cache space is used should there be new tokens, such as sliding window, H2O, FastGen, etc." - Thread positions showing the disagreement that stalled it, VERBATIM: `ShawnD200`: "For H2O you mentioned, I could be terribly wrong, but I don't see how it works without mapping, and you could have it either by explicit mapping or making block_size=1 so that block ids can be arranged to be a mapping as well. Prefix caching adds a layer of complexity, for \"full attention\" and \"sliding window\" it does not affect much, but for \"random\" selective eviction policy, either have block_size=1, or the block after token in it evicted can't be shared." `ShawnD200` again: "Then it comes down to: change block manager vs. attention module, this seems not a question, from single responsibility (and single place -- there is only one block manager but a few backends) point of view."
- **Date:** created 2025-01-21T08:46:39Z; issue last updated 2026-07-02T06:25:29Z.
- **Evidence:** READ BODY

#### C33. vLLM shipped TurboQuant backend drops QJL by design and needs a boundary-layer crutch for its aggressive presets
- **WHO:** vLLM (the TurboQuant backend authors, `vllm/model_executor/layers/quantization/turboquant/config.py`)
- **URL:** https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/quantization/turboquant/config.py
- **Closure status:** by design in shipped code — QJL is documented as deliberately omitted, and the aggressive presets are documented as requiring an empirical crutch (boundary-layer skipping).
- **STATED REASON, VERBATIM:** "QJL is intentionally omitted: community consensus (5+ independent groups) found it hurts attention quality by amplifying variance through softmax." — and, the second abandonment recorded in the same shipped file: "Empirically required for aggressive presets (k3v4_nc, 3bit_nc) — without it GSM8K drops ~30 points on Qwen3-4B."
- **Date:** code present in vLLM main as of retrieval (2026-09-13); the TurboQuant backend merged Apr 15, 2026 (PR #38479)
- **Evidence:** READ BODY

#### C34. QJL error correction (Algorithm 2) omitted from a second, independent TurboQuant implementation
- **WHO:** `Aaryan-Kapoor` and `veritatisquaesitoressumus`, in the ggml-org/llama.cpp discussion #20969 (verifier-corrected authorship; the evidence file attributed these words to TheTom)
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** omitted by design in an independent implementation — corroborates the vLLM TurboQuant omission. VERIFIER CORRECTION: these words were posted by `Aaryan-Kapoor` and `veritatisquaesitoressumus`, not by `TheTom`.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** QJL error correction (Algorithm 2) not implemented - Algorithm 1 alone is sufficient per the paper and, on the implementation plan, "The implementation follows Algorithm 1 (TurboQuant_mse) from the paper. Algorithm 2 (QJL error correction) is omitted as the paper shows MSE-optimal quantization alone is sufficient for KV cache compression without the extra bit cost."
- **Date:** posts dated Mar 2026 (thread started Mar 25, 2026)
- **Evidence:** READ BODY

#### C35. Direct GPU↔storage access (GPUDirect Storage / GPU-GPU) — explicitly excluded from vLLM’s multi-tier offloading design
- **WHO:** dannyharnik (vLLM), RFC #38260 "Multi-tier KV offloading via the vLLM offloading connector"
- **URL:** https://github.com/vllm-project/vllm/issues/38260
- **Closure status:** by design — an explicit “What we don’t intend to support” section in a maintainer-authored RFC; the shipped tiering implementation carries exactly this restriction.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** The RFC carries a section headed "What we don't intend to support", whose two bullets read verbatim: "Direct GPU access (neither GPU-storage or GPU-GPU communication)" and "Limited flexibility for variance in block size. While we allow vLLM block size to vary, CPU block size must be constant across all vLLM nodes (and a multiple of the underlying vLLM block size)." The shipped doc restates the first bullet verbatim: "Only the CPU primary tier has direct GPU access. Secondary tiers cannot read from or write to GPU memory; all GPU↔secondary transfers are staged through the CPU primary tier."
- **Date:** RFC opened 2026-03-26 (still Open as an RFC; the non-support decision is settled and codified in shipped code)
- **Evidence:** READ BODY

#### C36. MLA’s low-communication “absorb” form for training — hard-banned by Megatron-Core and measured to regress activation memory
- **WHO:** Megatron-Core (NVIDIA) as the abandoner; the LAGA authors as the measurers
- **URL:** https://arxiv.org/abs/2607.17644
- **Closure status:** by design — a hard assertion in shipped library code disables the technique with no documented reason; the accompanying paper explains why the ban is well-founded (20–34% activation-memory regression when ported to training).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Open Megatron-Core's MLA implementation NVIDIA (2025) and the forward begins with an assertion that bans its own low-communication path from training: assert not (self.training and self.cache_mla_latents). The absorb reformulation — folding kv_b_proj into the query so attention runs against the compressed latent and only the small latent crosses the collective — is fully implemented in the class, but gated to is_decode_only() inference and hard-asserted out of training. The library ships no low-communication MLA training path. Additional verbatim: "MCore's designers deliberately restrict the absorb trick to inference: the class's prepare_for_absorption even concedes that it is \"not doing true absorption. We will add this support at a later time.\" One contribution of this paper (§3.2/4.3) is to explain why that restriction is well-founded — the absorbed form, ported to training, regresses activation memory by 20–34% (up to 9.2 GB at V3 scale)". Also verbatim from the paper's own limitations: "Why this is both comm-cheap and memory-neutral. ... Measured: Laga peak memory matches B1 within ≤0.5% (<27 MB) while B2 balloons by up to 9.2 GB (Table 2)."
- **Date:** submitted 20 Jul 2026 (v1).
- **Evidence:** READ BODY

#### C37. LMCache CacheBlend: blend store cannot be disabled and has no size cap
- **WHO:** `shawnguyen-tensormesh` (LMCache), issues #4665 and #4664 (the two topics were filed together by the same author)
- **URL:** https://github.com/LMCache/LMCache/issues/4665 | https://github.com/LMCache/LMCache/issues/4664
- **Closure status:** both issues closed as not planned (`stateReason: NOT_PLANNED`). VERIFIER CORRECTION: each page carries a comment “Closing here” plus a ClosedEvent whose actor is the issue author `shawnguyen-tensormesh`; the evidence file’s repeated claim that comment text does not exist and the actor is unidentifiable is false for all six CacheBlend issues.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** that gate was **not carried over** (MP connector built as a parallel implementation ignoring `kv_role=kv_consumer` / `skip_save`). Companion #4664: "Unbounded blend store fills the L2 disk and evicts the serving pods via node DiskPressure - no size cap".
- **Date:** both created 2026-08-20, closed 2026-08-21 by the author (“Closing here”)
- **Evidence:** READ BODY

#### C38. LMCache CacheBlend: no per-model correctness gate for the blend re-RoPE approximation
- **WHO:** `shawnguyen-tensormesh` (LMCache), issue #4666
- **URL:** https://github.com/LMCache/LMCache/issues/4666
- **Closure status:** closed as not planned (`stateReason: NOT_PLANNED`) by the issue author, whose only closure comment is “Closing here” (verifier-corrected: the evidence file claimed no comment text and no identifiable actor).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** There is **no output-level, per-model acceptance test** … "correctness depends on ... **no automated check that the blended output actually matches a full-recompute reference** before it is served."
- **Date:** opened Aug 20, 2026; closed Aug 21, 2026
- **Evidence:** READ BODY

#### C39. LMCache CacheBlend: missing inter-document padding for non-prefix reuse (~11% measured seam loss)
- **WHO:** shawnguyen-tensormesh (LMCache), opened 2026-08-20, closed 2026-08-21
- **URL:** https://github.com/LMCache/LMCache/issues/4667
- **Closure status:** closed as not planned by the issue author (comment: “Closing here”) — a measured seam tax was reported and not acted on.
- **STATED REASON, VERBATIM:** "Closing here"
- **Date:** opened Aug 20, 2026; closed Aug 21, 2026
- **Evidence:** READ BODY

#### C40. LMCache cross-pod (cross-replica) sibling KV reuse — works, but is transient; closed as not planned
- **WHO:** shawnguyen-tensormesh (LMCache), opened 2026-08-20, closed 2026-08-21
- **URL:** https://github.com/LMCache/LMCache/issues/4668
- **Closure status:** closed as not planned by the issue author (comment: “Closing here”). The capability itself was reported to exist, but as a burst rather than steady state.
- **STATED REASON, VERBATIM:** "Closing here"
- **Date:** opened Aug 20, 2026; closed Aug 21, 2026
- **Evidence:** READ BODY

#### C41. KV reuse across LoRA adapters — deliberately excluded by vLLM’s prefix-cache block-hash key
- **WHO:** vLLM project (by design, enforced in code; reinforced by security PR #54342)
- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py
- **Closure status:** not a closure — “by design”. vLLM folds the LoRA adapter name into the block hash, so two adapters can never hit each other’s cached blocks; an adapter change invalidates the chain from that point. Reinforced by security PR #54342 (TITLE ONLY).
- **STATED REASON, VERBATIM:** The function docstring and body, retrieved character-for-character from the raw source file: "Return LoRA name of the request if it is a LoRA request. Return empty list otherwise." with the body "if not request.lora_request: return []" / "return [request.lora_request.lora_name]". The keys are combined unconditionally: "lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)" ... "extra_keys: list[Any] = ( lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys )".
- **Date:** current `main` as of retrieval (2026-09-13)
- **Evidence:** READ BODY

#### C42. vLLM `kv_role='kv_both'` for NixlConnector — deprecated and scheduled for removal
- **WHO:** vLLM project (deprecation shipped in `main`; docs PR to stop recommending it)
- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py
- **Closure status:** shipped deprecation warning in code (“will be removed in a future release”); the recommended usage was removed from the docs (PR #55693, CLOSED). Not yet deleted.
- **STATED REASON, VERBATIM:** "Using kv_role='kv_both' with NixlConnector is deprecated and will be removed in a future release. Please set kv_role='kv_producer' for prefill instances and kv_role='kv_consumer' for decode instances."
- **Date:** current `main` as of retrieval (2026-09-13)
- **Evidence:** READ BODY

#### C43. vLLM uneven decode context parallelism for heterogeneous TP ranks — explicit maintainer scope rejection
- **WHO:** efschu (contributor) — vLLM PR #50736, "[Feature] Uneven decode context parallelism for heterogeneous TP ranks"; closed by hmellor (vLLM member)
- **URL:** https://github.com/vllm-project/vllm/pull/50736
- **Closure status:** closed unmerged by a human maintainer (`hmellor`) as out of scope in-tree — a clean maintainer “no”, not a stale bot.
- **STATED REASON, VERBATIM:** "Uneven TP sharding is not something I think we are interested in supporting in-tree in vLLM as it is quite niche and appears to need changes to every supported model. In situations where you have different sized GPUs you likely don't have fast interconnect to run TP efficiently anyway. You should probably be using PP instead, which can already be unevenly split across GPUs using VLLM_PP_LAYER_PARTITION"
- **Date:** closed Aug 3, 2026 (close event timestamp 2026-08-03T14:03:55Z).
- **Evidence:** READ BODY

#### C44. TurboQuant KV cache + long-context chunked prefill — a workspace ratio that cannot be tuned around; reporter-closed as duplicate
- **WHO:** vLLM user (issue author), issue #46767; closed by the reporter as a duplicate
- **URL:** https://github.com/vllm-project/vllm/issues/46767
- **Closure status:** closed as not planned (`stateReason: NOT_PLANNED`) by the reporter themselves as a duplicate — not a maintainer rejection. The live siblings #41565 and #43357 remain open.
- **STATED REASON, VERBATIM:** "Closing as a duplicate of #41565 (TurboQuant `_continuation_prefill` workspace under-sizing at long context). I posted the quantified ~1.43x scaling analysis (util-independent; need and cap both scale with --max-num-batched-tokens) as a comment there. Also see #43357."
- **Date:** created 2026-06-25; closed by the reporter
- **Evidence:** READ BODY

#### C45. Avoiding recomputation of the last-token hidden states for spec decoding — declined in-thread as not worth the complexity
- **WHO:** vLLM maintainer NickLucche, replying to RFC #31064 (author xinyu-intel)
- **URL:** https://github.com/vllm-project/vllm/issues/31064
- **Closure status:** closed as completed, but the requested generalisation was declined in-thread by a maintainer; the shipped artifact is a different, narrower debug connector.
- **STATED REASON, VERBATIM:** "I have an aversion to adding this complexity in order to avoid running a single forward pass for 1 token --- so just want to make sure I understand why the current impl is not working well"
- **Date:** created 2025-12-20T05:56:32Z; closed 2025-12-20 (same day)
- **Evidence:** READ BODY

#### C46. Dynamo KVBM disk→device onboarding (and KVBM itself) — closed as no longer supported
- **WHO:** `jthomson04` (Dynamo maintainer), `sachalmalick`, `pjdurden`, `bennorris123`
- **URL:** https://github.com/ai-dynamo/dynamo/issues/12750 | https://github.com/ai-dynamo/dynamo/pull/13184
- **Closure status:** closed as not planned by a human maintainer (`jthomson04`). A scope/support decision, not confirmation that the reported disk-onboarding performance problem is fixed. The companion PR #13184 was closed without merging; the verifier retrieved a fuller direct closure comment on the PR itself.
- **STATED REASON, VERBATIM:** "Closing as not planned because KVBM is no longer supported. This is a scope/support decision, not confirmation that the reported disk-onboarding performance problem is fixed; the corresponding PR #13184 will be closed without merging." — and, the PR #13184 closure comment itself (verifier-retrieved, replacing the cross-reference): "Closing without merge because KVBM is no longer supported. The corresponding issue #12750 has been closed as not planned; this is a support/scope decision, not a claim that the layout or performance problem was fixed."
- **Date:** #12750 created 2026-08-06, closed 2026-09-10; PR #13184 created 2026-08-13, closed 2026-09-10
- **Evidence:** READ BODY

#### C47. Dynamo KVBM lifecycle bug — second independent confirmation of the KVBM support removal
- **WHO:** `jthomson04` (CONTRIBUTOR)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/13285
- **Closure status:** closed as not planned by a human maintainer (`jthomson04`); a distinct closure text from #12750’s (different PR #13297, different bug), produced from the same template.
- **STATED REASON, VERBATIM:** "Closing as not planned because KVBM is no longer supported. This is a scope/support decision, not confirmation that the reported lifecycle bug is fixed; the corresponding PR #13297 will be closed without merging."
- **Date:** maintained comment 2026-09-10
- **Evidence:** READ BODY

#### C48. Dynamo KVBM GPU HBM hit-rate exposure — closed with an explicit apology and rationale
- **WHO:** `dagil-nvidia` (NVIDIA/Dynamo maintainer)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/5716
- **Closure status:** closed as not planned by a human maintainer (`dagil-nvidia`) — a deliberate “we would rather close than leave you hanging”.
- **STATED REASON, VERBATIM:** "The Dynamo project has been moving fast and unfortunately this one slipped through the cracks. We don't currently have someone actively working on it, and to be honest with you we'd rather close out than leave you hanging indefinitely."
- **Date:** created 2026-01-28, closed 2026-05-26
- **Evidence:** READ BODY

#### C49. Dynamo KVBM documentation — “feature is being sunset”; docs re-add explicitly not planned
- **WHO:** Ben Hamm (Dynamo), commit author
- **URL:** https://github.com/ai-dynamo/dynamo/commit/62d0cf34a53cbef0d067f8a9d59d2031a2d30935
- **Closure status:** documentation removal merged; KVBM deprecation announced in the catalog entry. Not yet deleted from the tree.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Commit subject: "docs(recipes): drop KVBM references — feature is being sunset". Commit body: "Remove the agg-kvbm walkthrough pointer from the Qwen3-32B recipe and mark the catalog gap note accordingly. No docs re-add planned; source assets under recipes/qwen3-32b/vllm/agg-kvbm/ are untouched pending KVBM deprecation." The edited catalog entry reads, verbatim: "'agg-kvbm single-GPU walkthrough removed from the recipe entirely (2026-06-12): KVBM is being sunset, so no docs re-add is planned. Source assets at recipes/qwen3-32b/vllm/agg-kvbm/ remain untouched pending KVBM deprecation.'"
- **Date:** 2026-06-12
- **Evidence:** READ BODY

#### C50. Dynamo host-tier KV mirroring, offload and storage APIs — closed as superseded/deferred
- **WHO:** `hutm` (CONTRIBUTOR), GMS umbrella #10450 and children
- **URL:** https://github.com/ai-dynamo/dynamo/issues/10455
- **Closure status:** closed as not planned / superseded-deferred by a human contributor (`hutm`).
- **STATED REASON, VERBATIM:** "Closing this ticket as superseded/deferred, not completed."
- **Date:** closed 2026-07-13
- **Evidence:** READ BODY

#### C51. Dynamo GMS-managed KV umbrella retired — the SGLang, TRT-LLM, host/storage, router/P2P, TP=2 and Kubernetes scopes dropped from the MVP
- **WHO:** `hutm` (CONTRIBUTOR)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/10450
- **Closure status:** closed as not planned by a human contributor (`hutm`); all 18 still-open PRs from the umbrella were closed without deleting their branches.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** The SGLang, TensorRT-LLM, host/storage, router/P2P, TP=2, Kubernetes, and optimization scopes from this umbrella are not being carried into the MVP. The linked child tickets are being closed as superseded/deferred, not as completed. … "All 18 PRs that were still open from this umbrella have been closed without deleting their branches."
- **Date:** closed 2026-07-13
- **Evidence:** READ BODY

#### C52. llama.cpp DeepSeek-V4 prefix caching reprocesses the full prompt — closed not planned two days after the last comment
- **WHO:** `am17an` (CONTRIBUTOR); reporter `jaholmesuk`
- **URL:** https://github.com/ggml-org/llama.cpp/issues/25567
- **Closure status:** closed `not_planned` by a human contributor (`am17an`) — not a stale-bot closure (no stale label), though no explicit “won’t fix” sentence was retrieved.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** `am17an`: "The frozen anchor problem will still be there though. DSV4 can't do a partial seq_rm so it will restore to the nearest checkpoint instead and has to re-prefill everything after that. That's your +278 per request." Reporter's measurement: "So it reprocesses the full prompt on every request; no prefix reuse engaged over 20 passes up to ~7.7k tokens."
- **Date:** closed 2026-07-13
- **Evidence:** READ BODY

#### C53. llama.cpp canonical KV/long-context performance boundary — maintainers decline to invest in synthetic benchmarking
- **WHO:** `JohannesGaessler` (CONTRIBUTOR), `ngxson` (COLLABORATOR)
- **URL:** https://github.com/ggml-org/llama.cpp/issues/18722
- **Closure status:** closed `not_planned`. The retrieved page’s timeline was paginated and did not contain the ClosedEvent, so the closer cannot be named; the quoted maintainer statements are the reason on record.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** `JohannesGaessler`: "If at all possible we should measure performance using real models rather than synthetic benchmarks. I cannot speak for any of the other maintainers but I will not be investing any of my time into synthetic benchmarking." … "If you want to implement something other than what I'm describing you'll have to find someone else with review power." `ngxson`: "So, I don't see why we need to adding such test case in `llama-bench` for now."
- **Date:** created 2026-01-09, closed 2026-03-03
- **Evidence:** READ BODY

#### C54. SGLang HiCache host hits returned stale MoE sidecar data — closed not planned, with the fix pursued elsewhere
- **WHO:** `Cerdore`
- **URL:** https://github.com/sgl-project/sglang/issues/26975
- **Closure status:** closed `NOT_PLANNED` (2026-07-10) after a human root-cause comment. VERIFIER CORRECTION: the evidence file’s “guard PR #27067 shipped instead of the fix” conclusion is UNSUPPORTED — #27067 is itself the migration FIX (`fix(hicache): migrate captured indexer-topk on host-tier cache hits`), closed unmerged on 2026-06-05 as superseded by #27326. The verifier downgrades this row to READ BODY with the conclusion unsupported.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** the routed_experts capturer (state_capturer/) indexes its buffer by device KV-pool slot. HiCache load_back relocates tokens to new slots without migrating that sidecar data, so a host-tier hit returns stale rows from whatever previously occupied the new slot. MoE is skipped on cache hits, so there's no recomputation to fix it. Plain radix cache is unaffected (slots stay in place). Same structural hazard in enable_return_indexer_topk. … "Opened #27067 as a guard: hard error on the unsafe combo until the migration path lands. Auto-disabling either flag silently produces a different wrong result, so failing fast is the safer option."
- **Date:** created 2026-06-01, closed 2026-07-10
- **Evidence:** READ BODY (verifier downgrade: conclusion UNSUPPORTED)

#### C55. TensorRT-LLM reusing KV cache across different LoRAs — maintainer “not planned”
- **WHO:** filed by `ShuaiShao93`; closed by `karljang` (COLLABORATOR)
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/issues/11874
- **Closure status:** closed as not planned by a maintainer (`karljang`) — a direct maintainer “no” in that repository’s KV-reuse space. (The typo `apprach` and the double space are in the original.)
- **STATED REASON, VERBATIM:** Closing as not planned for now.  The aLoRA apprach is interesting and we'll keep it on the radar.\nThank you! (the typo `apprach` and the double space are in the original). Request verbatim: "Not sure if there are other options, but anything that supports reusing kv cache across different loras is useful for us."
- **Date:** created 2026-03-03, closed 2026-04-27. Labels: `KV-Cache Management`, `Lora/P-tuning`
- **Evidence:** READ BODY

#### C56. vLLM pipeline parallelism halves the usable KV block budget — answered by a maintainer as by-design
- **WHO:** reporter (issue body), answered by andoorve (vLLM maintainer)
- **URL:** https://github.com/vllm-project/vllm/issues/7039
- **Closure status:** closed as not planned by the stale bot AFTER a substantive maintainer answer; the maintainer’s “this is expected behaviour / the halving is fundamental” statement is the real human reason.
- **STATED REASON, VERBATIM:** This is kind of expected behavior based on what our implementation of PP aims to do. We report 1650 blocks because this is the total number of blocks available on your GPU. However, this gets divided into 2 KV cache sections to support multiple request streams at the same time. This is what allows us to have pipelining of request streams at the same time with load balancing of the two streams. This reporting could possibly be improved to reflect this. … "The situation you are talking about (very long prompt) is possible with some changes. You can submit a feature request for the same if there's a very clear use case for it. However, it wasn't a main focus until now since basically only 1 of those very long prompts could be resident in the cache at a time. This would mean essentially no pipelining, and you might want to see if tensor parallelism serves your use case better." The maintainer later restated it definitively: "So the halving is fundamental to the implementation here, whether you use 1 node or multiple the number of blocks per microbatch slot will be less compared to without pipeline parallelism."
- **Date:** created 2024-08-01, closed 2024-12-01
- **Evidence:** READ BODY

### C.3 Stale-closed (automated) — no human reason on record

#### C57. vLLM LoRA adapter reload can reuse stale prefix-cache blocks — stale-closed, never fixed
- **WHO:** vLLM issue #42125
- **URL:** https://github.com/vllm-project/vllm/issues/42125
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT — the closure reason is bot text, not a maintainer decision. A contributor offered to reproduce but no fix landed.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (preceded on the same page by "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!")
- **Date:** created 2026-05-09; closed in the 2026-09-12 stale wave (closure date from the GitHub search API)
- **Evidence:** READ BODY

#### C58. vLLM hybrid (DFlash speculative) + prefix caching IndexError in `get_temporal_copy_spec` — stale-closed
- **WHO:** vLLM issue #41884
- **URL:** https://github.com/vllm-project/vllm/issues/41884
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT — bot text only, no maintainer statement.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (also present: "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!")
- **Date:** created 2026-05-07; closed in the 2026-09-07 stale wave (closure date from the GitHub search API). Title verbatim: "[Bug][V1][Hybrid] IndexError in get_temporal_copy_spec during DFlash speculative decoding + prefix caching"
- **Evidence:** READ BODY

#### C59. vLLM `SimpleCPUOffloadConnector` + Hybrid KV Cache Manager block-pool exhaustion kills the engine — stale-closed
- **WHO:** vLLM issue #42085
- **URL:** https://github.com/vllm-project/vllm/issues/42085
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT. The bot’s closing comment was not present in the retrieved payload for this issue; the quoted label description is what the page carries. The failure is an engine-killing assertion.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Over 90 days of inactivity (the `stale` label description as embedded in the retrieved page). No human maintainer reason was retrievable.
- **Date:** created 2026-05-08; closed in the 2026-09-12 stale wave (closure date from the GitHub search API)
- **Evidence:** READ BODY

#### C60. vLLM per-request prefix-cache-token telemetry on OTel spans — stale-closed
- **WHO:** vLLM issue #41788
- **URL:** https://github.com/vllm-project/vllm/issues/41788
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT — bot text only.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2026-05-06; closed in the 2026-09-08 stale wave (closure date from the GitHub search API). Title verbatim: "[Feature]: Add request-level OTel span attribute for cached prefix-cache input tokens"
- **Evidence:** READ BODY

#### C61. vLLM RFC to support KV cache compaction / token dropping (FastGen, DuoAttention, H2O, attention sink) — stale-closed before it was revisited
- **WHO:** `YaoJiayi` (RFC author) in vllm-project/vllm; discussed by `KuntaiDu`, `simon-mo`, `heheda12345`.
- **URL:** https://github.com/vllm-project/vllm/issues/10646
- **Closure status:** closed as `NOT_PLANNED` BY THE `github-actions` STALE BOT, not by a maintainer (ClosedEvent actor `github-actions`, labels `RFC` + `stale`). The human comments in the thread are the closest thing to a human reason, and they are not refusals.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (comment by `github-actions`, 2025-03-27T02:04:38Z), preceded on 2025-02-25T02:00:45Z by: "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!" - Human maintainer context, VERBATIM (these are the closest thing to a human reason, and they are not refusals): `simon-mo` (2024-11-26T21:45:49Z): "On the research side, @lynnliu030 is experimenting with token drop's impact on memory allocation. I think we can revisit this around EOY and discuss the exact API change." `YaoJiayi` (2024-11-26T22:43:39Z): "The key challenge for supporting more methods is the memory management side. For both attention sink and H2O, they maintain the same number of tokens across different heads and layers. If we want to support more advanced methods, we need a more flexible memory layout."
- **Date:** created 2024-11-25T23:42:43Z; closed 2025-03-27T02:04:38Z; issue last updated 2026-07-02T13:48:30Z.
- **Evidence:** READ BODY

#### C62. SGLang WLFU (weighted LFU) radix-cache eviction policy PR — stale-closed
- **WHO:** `tejas-goyal` (author) in sgl-project/sglang.
- **URL:** https://github.com/sgl-project/sglang/pull/23781
- **Closure status:** PR closed unmerged BY THE `github-actions` STALE BOT, not a human maintainer.
- **STATED REASON, VERBATIM:** "Thanks @tejas-goyal . Closing this because it has had no updates in 125 days."
- **Date:** created 2026-04-27T00:35:53Z; closed 2026-08-30T01:22:06Z.
- **Evidence:** READ BODY

#### C63. SGLang T-LRU (Tail-Optimized LRU) cache eviction policy PR — stale-closed
- **WHO:** `wenxinzhang0` (author) in sgl-project/sglang.
- **URL:** https://github.com/sgl-project/sglang/pull/21708
- **Closure status:** PR closed unmerged BY THE `github-actions` STALE BOT, not a human maintainer.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Thanks @wenxinzhang0 . Closing this because it has had no updates in 136 days. - Additional detail from the PR description (its motivation, i.e. the workload-sensitive eviction criterion it wanted to add): "TEL-safe blocks : evicting them won't push the next turn's uncached token count above threshold ξ — these are evicted first."
- **Date:** created 2026-03-30T20:28:43Z; closed 2026-08-28T03:46:02Z.
- **Evidence:** READ BODY

#### C64. SGLang Marconi eviction PR — stale-closed while still a draft
- **WHO:** `qimcis` (author) in sgl-project/sglang.
- **URL:** https://github.com/sgl-project/sglang/pull/21862
- **Closure status:** PR closed unmerged BY THE `github-actions` STALE BOT, not a human maintainer; it was still a draft.
- **STATED REASON, VERBATIM:** "Thanks @qimcis . Closing this because it is still a draft and has not been updated in 146 days."
- **Date:** created 2026-04-01T17:40:52Z; closed 2026-08-26T01:32:50Z.
- **Evidence:** READ BODY

#### C65. SGLang fix for SLRU’s burst distortion and stale-hot-prefix stagnation (a shipped policy’s known failure mode) — stale-closed unfixed
- **WHO:** `HughLLiu` (author) in sgl-project/sglang.
- **URL:** https://github.com/sgl-project/sglang/pull/24075
- **Closure status:** PR closed unmerged BY THE `github-actions` STALE BOT, not a human maintainer. The shipped `slru` policy keeps the defect the PR described.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Thanks @HughLLiu . Closing this because it has had no updates in 117 days. - The PR's own statement of the shipped `slru` policy's defect, VERBATIM: "Zombie Protected entries : once a prefix enters Protected, historical heat never decays, so stale prefixes can stay protected long after the workload has shifted." and "naive SLRU has burst and stale-hot-prefix failure modes". The docs for the shipped policy corroborate the trade-off from the other direction: "they hurt when prefix popularity shifts over time, because a prefix that earned a high hit count keeps its advantage after it stops being useful."
- **Date:** created 2026-04-29T16:29:18Z; closed 2026-09-02T01:20:09Z.
- **Evidence:** READ BODY

#### C66. vLLM phase-aware (think-phase vs answer-phase) KV cache quantisation — stale-closed despite a measured 58% KL reduction
- **WHO:** myProjectsRavi (author), vLLM issue #39416
- **URL:** https://github.com/vllm-project/vllm/issues/39416
- **Closure status:** closed as not planned BY THE STALE BOT (GitHub Actions), NOT by a maintainer — no human maintainer reason exists on the page. The author’s own measured result is in the body.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (from the GitHub Actions stale bot; the earlier bot comment was "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!") — and, the author’s own measured claim, from the same issue body as relayed by the S11 row: "I have published a paper with a closed-form theorem and empirical validation showing this approach cuts attention KL divergence by **58%** compared to uniform 3-bit quantization on DeepSeek-R1-Distill-1.5B, with no additional inference-time compute."
- **Date:** opened Apr 9, 2026; closed Aug 10, 2026 (state_reason `not_planned`). The author's own result, verbatim: "Theory-aligned (4/3) 4 3 0.00126 -58%" (KL divergence vs uniform 3-bit, DeepSeek-R1-Distill-Qwen-1.5B, n=50, GSM8K) — i.e. a measured 58% distortion reduction that was never picked up. A separate but related proposal is also sitting open and stale: "https://github.com/vllm-project/vllm/issues/38930" ("Entropy-adaptive per-head KV cache quantization: +8% quality over uniform at same compression", OPEN, labeled `unstale`).
- **Evidence:** READ BODY

#### C67. Dynamic Memory Compression (DMC) for vLLM — maintainer pushback on the retrofit-data cost, then stale-closed (two duplicate requests)
- **WHO:** tchaton (requester) / vLLM project; pushback from condy0919 (vLLM maintainer)
- **URL:** https://github.com/vllm-project/vllm/issues/3549 | https://github.com/vllm-project/vllm/issues/4728
- **Closure status:** closed as `not_planned`; the visible close action was performed by the automation bot, not a human. A human maintainer objection precedes it on the first issue; the second (duplicate) issue has no maintainer reason on the page at all.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** According to the paper,\n\n> In our experiments, we equip pre-existing LLMs—such as Llama 2 (Touvron et al., 2023) 7B, 13B, and 70B—with DMC by retrofitting them on a negligible percentage of the original pre-training data (~2% for 2× compression, and ~8% for 8× compression) and without adding any extra parameters to the original LLM.\n\nThe required data is large IMHO. — then the bot: "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" — and, the duplicate request #4728, where only bot text exists: This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! and then "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** #3549 created 2024-03-21, bot-closed 2025-01-02; #4728 created 2024-05-10, bot-closed 2024-11-27
- **Evidence:** READ BODY

#### C68. vLLM LoRA adapters on DeepSeek-V2 have no effect (later diagnosed as the MLA `kv_b_proj` path) — stale-closed undiagnosed, re-filed ~15 months later
- **WHO:** chenchen0611 (reporter) / vLLM project
- **URL:** https://github.com/vllm-project/vllm/issues/17155
- **Closure status:** stale-closed BY THE AUTOMATION BOT ONLY — no maintainer reason on the page. The later issue #48974 characterises it as unresolved. (Verifier note: the page for #17155 never mentions `kv_b_proj`, MLA or absorption; that mechanism comes from the later diagnosis.)
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! and then "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" The later diagnosis (#48974) states verbatim: "closest is #17155 (stale-closed, undiagnosed — consistent with this root cause)."
- **Date:** created 2025-04-25; stale-marked 2025-07-25; bot-closed 2025-08-24.
- **Evidence:** READ BODY

#### C69. Attention-sink support in MLA — requested, never implemented, stale-closed
- **WHO:** lhtin (reporter) / vLLM project
- **URL:** https://github.com/vllm-project/vllm/issues/42319
- **Closure status:** closed as `not_planned`; the only closure text is the automation bot’s — no human maintainer reason appears on the retrieved page.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! — the request body itself is verbatim: "Like normal attention, hope support sink in MLA attention." The actual close comment on the page reads: "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2026-05-11; stale-marked 2026-08-10; bot-closed 2026-09-10.
- **Evidence:** READ BODY

#### C70. vLLM FA4 MLA decode backend — requested, a volunteer offered to do it, never landed, stale-closed
- **WHO:** LucasWilkinson (reporter) / vLLM project
- **URL:** https://github.com/vllm-project/vllm/issues/35919
- **Closure status:** closed as `not_planned`; the closure text is automation-bot only — no maintainer reason on the retrieved page. The community volunteer’s offer got no maintainer response before the bot closed it. VERIFIER CORRECTION: the page says “should remain open”, not “should still be open”.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! and "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" The unanswered offer, verbatim: "Hi @LucasWilkinson , I see the sm100_mla kernels are already in csrc/attention/mla. Is the goal of this issue to hook these new kernels into the v1 backend logic? I have access to a B200 environment and would like to help with the integration and benchmarking."
- **Date:** created 2026-03-03; volunteer comment 2026-05-04; stale-marked 2026-08-05; bot-closed 2026-09-06.
- **Evidence:** READ BODY

#### C71. BFLA — a training-free sparse prefill attention backend offered for upstreaming to vLLM — stale-closed
- **WHO:** Alicewithrabbit (reference implementation at `github.com/Alicewithrabbit/BFLA`), RFC against vLLM 0.19.1
- **URL:** https://github.com/vllm-project/vllm/issues/42419
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT, NOT by a maintainer — the only closure text is bot text. This must not be read as a maintainer rejection.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! — followed by "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" The issue also carries the label text "RFC stale Over 90 days of inactivity".
- **Date:** created 2026-05-12T12:27:26Z; closed ~2026-09 (stale-bot closure; page shows no human close event).
- **Evidence:** READ BODY

#### C72. vLLM frequency- and cost-aware prefix-cache eviction RFC and its implementation PR — both stale-closed
- **WHO:** luolun (issue author), vLLM; implementation by Aminsed (PR #27539)
- **URL:** https://github.com/vllm-project/vllm/issues/23641 | https://github.com/vllm-project/vllm/pull/27539
- **Closure status:** issue closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT (`github-actions`), NOT by a maintainer; the linked implementation PR #27539 was closed by the same stale bot.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (issue #23641, posted by github-actions Bot at 2026-01-10T02:13:16Z, immediately before "github-actions Bot closed this"); for PR #27539: "This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you!"
- **Date:** issue created 2025-08-26, bot-closed 2026-01-10; PR #27539 bot-closed 2026-03-03
- **Evidence:** READ BODY

#### C73. vLLM workload-aware KV cache eviction policy PR — stale-closed unmerged
- **WHO:** Chasingdreams6 (JinboHan), vLLM
- **URL:** https://github.com/vllm-project/vllm/pull/22236
- **Closure status:** PR closed unmerged BY THE AUTOMATED STALE BOT (`github-actions`), NOT by a maintainer; labels included `stale` and `needs-rebase`.
- **STATED REASON, VERBATIM:** This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you! (posted by github-actions Bot, Feb 9, 2026, immediately before the close; the earlier mark-stale comment on Jan 9, 2026 from the same bot reads "This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this pull request should remain open. Thank you!")
- **Date:** created 2025-08-05, bot-closed 2026-02-09
- **Evidence:** READ BODY

#### C74. vLLM changing `kv_load_failure_policy` default from “recompute” to “fail” — maintainers agreed, then the stale bot closed it
- **WHO:** vLLM issue #32200 (RFC author, CC @sdavidbd @wseaton @tlrmchlsmth @hasB4K)
- **URL:** https://github.com/vllm-project/vllm/issues/32200
- **Closure status:** closed as `NOT_PLANNED`, and the only closure text is automated stale-bot text, not a maintainer decision.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! followed by "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2026-01-12T18:28:21Z; stale-bot comments dated 2026-05-21 and 2026-06-21
- **Evidence:** READ BODY

#### C75. vLLM NIXL connector silently disabling HMA — stale-closed; the duplicate/closure framing in the evidence file is unsupported
- **WHO:** vLLM issue #42024 (reporter), maintainer response citing chfeng-cs
- **URL:** https://github.com/vllm-project/vllm/issues/42024
- **Closure status:** VERIFIER CORRECTED: the issue was stale-closed by `github-actions` on 2026-09-12 (`duplicateOf: null`), NOT closed as a duplicate on 2026-08-08 (that date is the stale warning). The quoted comment is by contributor `chfeng-cs`, the author of the merged PR it cites — not a maintainer and not the closure reason. Downgraded to TITLE ONLY.
- **STATED REASON, VERBATIM:** "This duplicates #41847 which addresses the same issue. Note that a class-level supports_hma() check is insufficient for MultiConnector — the wrapper class implements SupportsHMA but effective support depends on all configured children. #41847 handles this via KVConnectorFactory.supports_hma_config() which recurses into child connectors."
- **Date:** issue reports vLLM 0.18.x; closure trace dated 2026-08-08
- **Evidence:** TITLE ONLY (verifier downgrade)

#### C76. vLLM NIXL telemetry throughput not accounting for transfer overlapping — stale-closed with no answer from the pinged maintainer
- **WHO:** vLLM issue #33170 (reporter), maintainer pinged (@NickLucche) and never replied
- **URL:** https://github.com/vllm-project/vllm/issues/33170
- **Closure status:** closed as `NOT_PLANNED`; the closure text is automated stale-bot text only — no human closure reason exists on the page.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2026-01-27T18:07:23Z; stale-bot activity 2026-05-06 and 2026-06-05
- **Evidence:** READ BODY

#### C77. vLLM P2P NCCL XpYd prefill worker crash — stale-closed with the fix still unlocated
- **WHO:** vLLM issue #25043 (reporter), with a second reporter confirming on the same page
- **URL:** https://github.com/vllm-project/vllm/issues/25043
- **Closure status:** closed as `NOT_PLANNED`; the closure text is automated stale-bot text only. VERIFIER CORRECTION: the evidence file’s date line was internally impossible; the real timeline is created 2025-09-17, human comments 2025-09-21 → 2025-10-15, stale label 2026-01-15, stale-closed 2026-02-15.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2025-09-17, stale 2026-01-15, closed 2026-02-15 (verifier-corrected)
- **Evidence:** READ BODY

#### C78. NVIDIA Dynamo heterogeneous-TP P/D in the SGLang path crashed on the recommended configuration — stale-closed
- **WHO:** NVIDIA Dynamo issue #5870 (reporter on H200 SXM, Dynamo 0.8.0)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/5870
- **Closure status:** closed as `NOT_PLANNED`; the closure text is automated inactivity text only.
- **STATED REASON, VERBATIM:** "This issue has been closed due to inactivity. If you believe this issue is still relevant, please feel free to reopen it with additional context or information."
- **Date:** created 2026-02-01T23:38:19Z; stale warning 2026-03-04, inactive closure 2026-03-09
- **Evidence:** READ BODY

#### C79. Dynamo KVBM silently provided no benefit unless an undocumented flag was set — stale-closed, negative result left in the body
- **WHO:** reporter (issue #7566); closed by the **stale bot**
- **URL:** https://github.com/ai-dynamo/dynamo/issues/7566
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE, not a maintainer decision. The negative result is in the issue body, not the closure.
- **STATED REASON, VERBATIM:** "All KVBM integration tests use --num-gpu-blocks-override (defaulting to 10 blocks in tests/kvbm_integration/common.py:530), which masks this issue. The KVBM guide ... does not mention --num-gpu-blocks-override as a requirement or recommendation, so **users following the documentation will silently get no KVBM benefit**."
- **Date:** stale-bot closed 2026-04-30
- **Evidence:** READ BODY

#### C80. Dynamo KVBM disk reads issued as many small 32 KB IOs instead of one contiguous 2 MB transfer — closed by an automation
- **WHO:** `ekaynar` (reporter); closed by `linear[bot]`, not a human
- **URL:** https://github.com/ai-dynamo/dynamo/issues/6101
- **Closure status:** `state_reason=COMPLETED` but closed by an automation (`linear[bot]`), not a human; the last human comment asking “Did you resolve this? Thanks” went unanswered.
- **STATED REASON, VERBATIM:** "when KVBM reads a full block from G3 (disk) -> G1 (HBM), NIXL issues many small 32 KB IOs instead of a single contiguous 2 MB transfer."
- **Date:** closed 2026-04-22
- **Evidence:** READ BODY

#### C81. Dynamo KVBM TTFT degradation with MTP — stale-closed, no human closure reason
- **WHO:** reporter (issue #6353); closed by the **stale bot**
- **URL:** https://github.com/ai-dynamo/dynamo/issues/6353
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE (ClosedEvent actor `github-actions`); the quoted fragment is the reported symptom, not a closure statement.
- **STATED REASON, VERBATIM:** "median TTFT increases significantly"
- **Date:** stale-bot closed 2026-04-23
- **Evidence:** READ BODY

#### C82. LMCache + Mooncake on a single node gave no benefit — stale-closed with the negative result unrebutted
- **WHO:** reporter; `xiaguan` (CONTRIBUTOR) responding
- **URL:** https://github.com/LMCache/LMCache/issues/1331
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE. The substantive negative result is the reporter’s and the maintainer’s, not the closure text.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** I tried both methods and found no benefit. What could be the reason? … "Experiment 2 found that the RDMA transmission protocol was slightly better than TCP, but not as good as Baseline." `xiaguan`: "If you're only using a single node, why not just use the local CPU backend?"
- **Date:** created 2025-08-13, bot-closed 2025-11-14
- **Evidence:** READ BODY

#### C83. vLLM selective KV cache offload (send part of the prompt to slower storage) — stale-closed after a maintainer redirect
- **WHO:** `ruocco` (author, llm-d/vLLM); `effi-ofer`, `michalmalka`
- **URL:** https://github.com/vllm-project/vllm/issues/39305
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE, with a substantive maintainer-side redirect earlier in the thread.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** `effi-ofer` (2026-04-12): "Once [multi-tier](https://github.com/vllm-project/vllm/issues/38260) is available, this can be used to offload the entire prompt but send a portion to slower storage (fs or object storage)." The author's own framing of the motivation is a negative observation about reuse: "Public traces like Mooncake or Alibaba [show](https://dl.acm.org/doi/pdf/10.1145/3773772) that 40-60% of the KV Cache is stored but never reused."
- **Date:** created 2026-04-08, bot-closed 2026-08-12
- **Evidence:** READ BODY

#### C84. vLLM topology-aware KV cache compression proposal — stale-closed with no maintainer reason
- **WHO:** `jianxinglee62-prog` (author); `jagmarques` (commenter)
- **URL:** https://github.com/vllm-project/vllm/issues/38725
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE. The only substantive technical reason on the page is a commenter’s, not a maintainer’s.
- **STATED REASON, VERBATIM:** "DeepSeek-V3's MLA compresses KV at the architecture level by using a shared latent vector. The residual KV cache is already smaller, so external compression may have diminishing returns. Worth benchmarking whether the gain is meaningful post-MLA vs. standard MHA models."
- **Date:** created 2026-04-01, bot-closed 2026-08-06
- **Evidence:** READ BODY

#### C85. vLLM KV-cache rollback / “step back” API in the engine core — maintainer rejection in-thread, then stale closure
- **WHO:** `robertgshaw2-redhat` (Red Hat, vLLM maintainer); proposer `alexbuiko-sketch`
- **URL:** https://github.com/vllm-project/vllm/issues/34698
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE after a maintainer rejection in-thread. The quoted maintainer sentence reproduces the page exactly, including its apparent typo.
- **STATED REASON, VERBATIM:** I think that adding an interface in the engine core to mutate existing sequences with a "step back" function is going to be accepted as it creates significant complexity into the core engine. However, I feel confident this feature could be implemented by leveraging vllm's prefix caching functionality. This is for example how beam search is implemented internally
- **Date:** created 2026-02-17, bot-closed 2026-09-06
- **Evidence:** READ BODY

#### C86. llama.cpp q4_0 symmetric KV cache performance regression — not reproduced; stale-closed
- **WHO:** `Zulf1qar` (reporter), `vibecodingengine` (independent test); closed by the **stale bot**
- **URL:** https://github.com/ggml-org/llama.cpp/issues/25422
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE.
- **STATED REASON, VERBATIM:** I tried to reproduce on RunPod RTX **5090** since RTX **5070** is not available there. I tested b9867, b9888, and b9890 with GLM-4.7-Flash-UD-Q4_K_XL.gguf, flash attention on, q4_0 K/V cache, --kv-unified, -c 98304, one server slot, and a fixed 15644-token synthetic prompt. I did not see a major regression on this setup. Reporter concedes upstream: "Ran few more tests on b9890 kept showing worse results, but I did just run b9924 and I got almost same result as b9867"
- **Date:** created 2026-07-07, bot-closed 2026-08-22
- **Evidence:** READ BODY

#### C87. llama.cpp KV cache in the RPC backend — the prompt cache does no useful work and its save path aborts the server; stale-closed
- **WHO:** `doornail` (reporter); closed by the **stale bot**
- **URL:** https://github.com/ggml-org/llama.cpp/issues/26128
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** --cache-ram 0 fully disables the prompt cache and the crash goes away. … "The server prompt cache only earns anything when conversations outnumber slots; with -np 4 and ≤4 conversations nothing is ever evicted, so it does no useful work while its save path aborts the server."
- **Date:** bot-closed 2026-09-13
- **Evidence:** READ BODY

#### C88. llama.cpp cross-request caching gives no benefit on SWA models (always invalidated) — stale-closed
- **WHO:** reporter; closed by the **stale bot**
- **URL:** https://github.com/ggml-org/llama.cpp/issues/24587
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** No actual benefit from cross-request caching (since it's always invalidated). Workaround: "Removing the `--ctx-cross-request-cache` flag".
- **Date:** bot-closed 2026-08-01
- **Evidence:** READ BODY

#### C89. TensorRT-LLM `enable_block_reuse=True` crash on Gemma 3 VSWA — stale-closed, never refuted
- **WHO:** filed by `stolorz`; closed by the **stale bot**
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/issues/13123
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE, not a maintainer decision. A maintainer did post a genuine reproduction attempt (Gemma-3-1b on RTX PRO 6000 Blackwell) before the issue went stale, i.e. the bug timed out rather than being refuted.
- **STATED REASON, VERBATIM:** "Issue has not received an update in over 14 days. Adding stale label." — and, the stale-bot closing text on the same page: "This issue was closed because it has been 14 days without activity since it has been marked as stale."
- **Date:** created 2026-04-16, bot-closed 2026-06-11
- **Evidence:** READ BODY

#### C90. vLLM Prefill Context Parallel (PCP) for MLA models — full review cycle, then stale-closed
- **WHO:** FENP (author), reviewed by LucasWilkinson (vLLM maintainer)
- **URL:** https://github.com/vllm-project/vllm/pull/28988
- **Closure status:** closed as not planned by the STALE BOT (ClosedEvent actor `github-actions Bot`), not by a maintainer — the substantive human blockers are quoted below but no maintainer closing statement exists.
- **STATED REASON, VERBATIM:** This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this pull request should remain open. Thank you! — followed by: "This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you!" - Substantive human blockers recorded before the bot closure, verbatim: "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @FENP ." and from the automated review: "My review has identified a critical issue in the attention correction logic where DCP and PCP corrections are applied in the wrong order, which will lead to incorrect results." and from a maintainer: "I find it quite messy that _prepare_inputs modifies num_scheduled_tokens_np and scheduler_output internally. Its not very clear to the reader here thats whats happening and thats why this recomputation/re-assignment is required. I think we should try harder to keep PCP more isolated for now". - Its own hardware evidence, verbatim from the PR body's Test Plan: "vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --prefill-context-parallel-size 2" and "We evaluated the performance of PCP and TP on DeepSeek-R1 and Kimi-K2 using 4K-length inputs on the H20-3e." Note that even at `--tensor-parallel-size 1`, the configuration is `--prefill-context-parallel-size 2`, i.e. two ranks, therefore two GPUs. Benchmark rows in the same body are labelled "PCP1TP8", "PCP2TP4", "PCP4TP2", "PCP8TP1".
- **Date:** created 2025-11-19, last substantive human review activity before the stale timer, closed 2026-05-09
- **Evidence:** READ BODY

#### C91. vLLM Prefill Context Parallel for GQA FlashInfer (split-out follow-up, plus its predecessor) — stale-closed unmerged
- **WHO:** pisceskkk and LookAround0301 (authors), reviewed by LucasWilkinson
- **URL:** https://github.com/vllm-project/vllm/pull/28723 | https://github.com/vllm-project/vllm/pull/26864
- **Closure status:** #28723 closed unmerged by the STALE BOT (no “not planned” badge, no human reason); its predecessor #26864 — which carries the authors’ own negative GQA+PCP measurement — was closed unmerged earlier.
- **STATED REASON, VERBATIM:** This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you! (preceded by "This pull request has been automatically marked as stale because it has not had any activity within 90 days.") - The substantive human blocker recorded earlier in the thread, verbatim: "Left some comments on #28988 which I think similarly apply here". — and, the predecessor PR #26864, where the technique’s own measurement is the reason it could not be justified as a default: "but the performance decrease compare to the base. I run the pcp test as above command，and client send request 8K/1K ，2k/1k，1k/1k input/ouput，the test result as follows： the 1K/1K base data: the 1K/1K pcp data the 2K/1K base data: the 2K/1K PCP data Is there any error in my testing method? Why isn't PCP effective?"
- **Date:** #28723 created 2025-11-14, closed 2026-04-01; #26864 created 2025-10-15, closed 2025-11-20
- **Evidence:** READ BODY

#### C92. vLLM NCCL graph-mixing optimisation for context parallel (~1 ms TPOT win) — stale-closed unmerged
- **WHO:** FENP
- **URL:** https://github.com/vllm-project/vllm/pull/32106
- **Closure status:** PR closed unmerged by the STALE BOT; no maintainer closing comment exists — the only closure reason is bot text.
- **STATED REASON, VERBATIM:** This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you! (preceded by the stale-bot marking: "This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this pull request should remain open. Thank you!")
- **Date:** created 2026-01-11, closed 2026-05-22
- **Evidence:** READ BODY

#### C93. vLLM Decode Context Parallel failed to run on H200 — stale-closed, never resolved
- **WHO:** haic0 (reporter); pisceskkk responded
- **URL:** https://github.com/vllm-project/vllm/issues/27544
- **Closure status:** closed as `NOT_PLANNED` by the STALE BOT — no maintainer reason statement, only bot text.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! then "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2025-10-27, closed 2026-06-08
- **Evidence:** READ BODY

#### C94. vLLM DCP regression between v0.10.2 and v0.11.0 for DeepSeek-R1 — stale-closed, never root-caused
- **WHO:** Harshith-umesh
- **URL:** https://github.com/vllm-project/vllm/issues/26942
- **Closure status:** closed as `NOT_PLANNED` by the STALE BOT only — explicitly not a maintainer decision.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! then "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2025-10-15, closed 2026-02-15
- **Evidence:** READ BODY

#### C95. vLLM FP8 KV cache + DCP for MLA, benchmarked on 8× and 16× RTX PRO 6000 — stale-closed unmerged
- **WHO:** grimulkan
- **URL:** https://github.com/vllm-project/vllm/pull/34795
- **Closure status:** closed as not planned by the STALE BOT; no human maintainer closing reason — explicitly bot-driven.
- **STATED REASON, VERBATIM:** This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this pull request should remain open. Thank you! then "This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you!"
- **Date:** created 2026-02-18, closed 2026-07-05
- **Evidence:** READ BODY

#### C96. SGLang context-parallel PRs — one produced incorrect long-context output, one was an unmaintained draft; both stale-closed
- **WHO:** SGLang contributor (PR #27276)
- **URL:** https://github.com/sgl-project/sglang/pull/27276 | https://github.com/sgl-project/sglang/pull/21865
- **Closure status:** both PRs closed unmerged by the STALE BOT; closure reason is bot text only in each case. Both rows are TITLE ONLY in the evidence.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Closing this because it has had no updates in 99 days. — the technical content of the PR body was recorded as: "the sparse path is selected by a CP-blind gate and then runs with global, contiguous query geometry, which is wrong once round-robin CP strides each rank's tokens — producing incorrect long-context output". — and, the second PR (#21865), closed as an unmaintained draft: Closing this because it is still a draft and has not been updated in 146 days. — the PR body was recorded as noting "no CUDA graph support for CP".
- **Date:** closed 2026-09-12
- **Evidence:** TITLE ONLY

#### C97. vLLM T-LRU conversation-aware KV cache eviction RFC — stale-closed; NOT a maintainer decision
- **WHO:** wenxinzhang0 on vLLM (RFC issue #37823)
- **URL:** https://github.com/vllm-project/vllm/issues/37823
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT — there is NO human maintainer reason on the page. Do not read this as a maintainer decision.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** opened Mar 22, 2026; stale-marked 2026-06-25; auto-closed 2026-07-27
- **Evidence:** READ BODY

#### C98. vLLM caching system-prompt token ids (to skip re-tokenisation) — stale-closed
- **WHO:** vLLM issue author (issue #29286 "[Performance]: cache system prompt token ids")
- **URL:** https://github.com/vllm-project/vllm/issues/29286
- **Closure status:** closed as `NOT_PLANNED` with labels `performance` and `stale`, BY THE AUTOMATED STALE BOT — no human maintainer reason on the page.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** opened 2025-11-24; auto-closed (NOT_PLANNED per page metadata)
- **Evidence:** READ BODY

#### C99. vLLM hybrid checkpoint ABI for non-KV prefix resume (RFC) — stale-closed
- **WHO:** CarlOskarRost on vLLM (RFC issue #40533)
- **URL:** https://github.com/vllm-project/vllm/issues/40533
- **Closure status:** closed as `NOT_PLANNED` with the `stale` label, BY THE AUTOMATED STALE BOT — no human maintainer reason on the page.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** opened 2026-04-21; stale-marked 2026-07-23; auto-closed 2026-08-25
- **Evidence:** READ BODY

#### C100. vLLM potential use-after-free in the KV block allocator under eviction pressure — stale-closed while the candidate fix PR remains open
- **WHO:** Yunzez (reporter); closed by the stale automation. Fix PR #37164 remains OPEN.
- **URL:** https://github.com/vllm-project/vllm/issues/37076
- **Closure status:** closed as `NOT_PLANNED` by `github-actions` (stale automation), NOT by a maintainer. The candidate fix PR #37164 is still OPEN and unmerged.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (posted by `github-actions`, with the prior verbatim bot comment "This issue has been automatically marked as stale because it has not had any activity within 90 days.")
- **Date:** created 2026-03-14; stale 2026-06-15; closed 2026-07-15. The technical content that was abandoned, verbatim from flutist's comment on 2026-03-16: "When many requests sharing a common prefix are scheduled in the same step, cache_full_blocks inserts newly allocated blocks into the prefix-cache hash table before the GPU forward pass runs. A subsequent request in the same step can hit one of these blocks via get_cached_block, read its uninitialized GPU memory, and produce wrong output tokens — even at temperature=0."
- **Evidence:** READ BODY

#### C101. vLLM stats metric for `available_kv_cache_memory` — stale-closed unanswered
- **WHO:** MML-coder (requester); closed by the stale automation
- **URL:** https://github.com/vllm-project/vllm/issues/26850
- **Closure status:** closed as `NOT_PLANNED`, label `stale`, by `github-actions` (stale automation), NOT by a maintainer.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (posted by `github-actions`; prior bot comment verbatim: "This issue has been automatically marked as stale because it has not had any activity within 90 days.")
- **Date:** created 2025-10-14; stale 2026-01-13; closed 2026-02-12. Contributing work referenced in the timeline includes commits titled verbatim "[Core] Add token-level KV cache metrics to V1 engine", but the issue itself was never resolved by a human.
- **Evidence:** READ BODY

### C.4 Reverted, withdrawn, or retracted by the author

#### C102. vLLM `--prefix-match-unit` documentation issue — withdrawn by its own author
- **WHO:** vLLM issue #54607, author sanjeevrg89 (self-closed)
- **URL:** https://github.com/vllm-project/vllm/issues/54607
- **Closure status:** closed as not planned (`stateReason=NOT_PLANNED`) by the author himself, not a maintainer and not a bot.
- **STATED REASON, VERBATIM:** "Closing this. The premise was wrong and the remaining justification did not survive checking either --prefix-match-unit exists, it does what I said could not be done, and it is documented …"
- **Date:** created 2026-08-31, closed 2026-09-01
- **Evidence:** READ BODY

#### C103. vLLM equal-block-sizes hybrid crash report — retracted by its author as a duplicate with a wrong premise
- **WHO:** vLLM issue #54199, author eak7824 (self-retracted)
- **URL:** https://github.com/vllm-project/vllm/issues/54199
- **Closure status:** closed as not planned (`stateReason=NOT_PLANNED`); the title itself reads “RETRACTED — duplicate of #53142 (my \"equal block sizes\" premise was wrong; see closing comment)”. Closed by the author.
- **STATED REASON, VERBATIM:** "**Retracting this report — the central premise is wrong, and #53142's mechanism explains our crash after all.**"
- **Date:** created 2026-08-28, closed 2026-08-29
- **Evidence:** READ BODY

#### C104. Attempted revert of the HMA GPU-side KV events feature — the revert PR itself was closed unmerged
- **WHO:** vllm-agent (automated agent account, head branch `auto-revert/pr-37688`); original feature by hickeyma
- **URL:** https://github.com/vllm-project/vllm/pull/39658
- **Closure status:** PR closed unmerged (`state=CLOSED`, `mergedTime=null`, `mergedBy=null`); the target feature PR #37688 remains MERGED. No human closure comment was retrievable — the only verbatim artifact is the commit message.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** "This reverts commit cc07dad789e9ae3f02ab6a4794800b8c1cc240c6."
- **Date:** created 2026-04-13, closed 2026-05-20
- **Evidence:** READ BODY

#### C105. A publicised “92.5% prompt-throughput collapse” and “memory paradox” for q4_0 KV — retracted by its own author
- **WHO:** dentity007, in the ggml-org/llama.cpp discussion #20969 (with attribution of the catch to u/audioen on r/LocalLLaMA)
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** self-retraction — the measurement that made q4_0 KV look catastrophic was wrong (RSS measurement, silent request failures); the corrected numbers reverse the conclusion.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** EDIT (April 1): The data below has been corrected. Original claims about a 92.5% prompt collapse and memory paradox were wrong (RSS measurement, silent request failures). See my correction reply below for accurate numbers. And the correction itself verbatim: "Correction to my post above: the benchmark data I shared was flawed. u/audioen on r/LocalLLaMA caught the methodology error and they were right." / "\"92.5% prompt throughput collapse at 64K\" -- Wrong. I measured throughput from requests that failed silently. Prompt throughput is identical across all cache types at all context lengths."
- **Date:** original post Mar 31, 2026; correction Apr 1, 2026
- **Evidence:** READ BODY

#### C106. SGLang MERGED REVERT of the `kv_cache_scheme` refactor for quantisation — it broke DeepSeek V3 FP4
- **WHO:** zhyncs (SGLang maintainer) — revert of PR #10132
- **URL:** https://github.com/sgl-project/sglang/pull/10935
- **Closure status:** MERGED REVERT — the revert landed; the refactor was undone.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** This reverts commit cd4da1f . and, under Motivation, "this breaks deepseek v3 fp4"
- **Date:** Sep 26, 2025
- **Evidence:** READ BODY

#### C107. Custom evictor implementations for the vLLM filesystem offload tier — withdrawn by the author as codebase pollution; the follow-up capacity-bound LRU attempt also closed unmerged
- **WHO:** varun-sundar-rabindranath (author, self-closed), vLLM PR #43725 "[FileSystemTierManager] Toy Evictors for FS Offloading"
- **URL:** https://github.com/vllm-project/vllm/pull/43725 | https://github.com/vllm-project/vllm/pull/52784
- **Closure status:** both PRs CLOSED unmerged by their own authors. The first has a verbatim reason; for the second (#52784) the page contains no closing comment — the last substantive content is Copilot’s automated review raising an eviction-never-triggers bug.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Decided we dont want to pollute the main codebase. (sic — copied character-for-character, including the missing apostrophe) — and, the second attempt (#52784), whose evidence that the first was abandoned is recorded in its body: Decided we dont want to pollute the main codebase. (author varun-sundar-rabindranath, Jun 24, 2026, posted immediately before his own close of #43725; the PR was created 2026-05-27 and closed 2026-06-24). Evidence that the first attempt was abandoned, verbatim from PR #52784's body: "A previous attempt ( #43725 , "Toy Evictors for FS Offloading") was closed by the author without merging." For #52784 itself, the verbatim stated reason is: NO STATED REASON FOUND — the page contains no closing comment; the only closure evidence is the timeline line "XiaYiHann closed this Aug 19, 2026" on a PR created 2026-08-18, and the last substantive page content is Copilot's automated review raising an eviction-never-triggers bug ("When capacity_bytes is set, submit_store() only records _store_job_keys when KV events are enabled. That means get_finished_jobs() can't account for stored bytes or update the LRU (store_keys is None), so eviction never triggers and disk usage can still grow without bound.").
- **Date:** #43725 created 2026-05-27, closed 2026-06-24; #52784 created 2026-08-18, closed 2026-08-19
- **Evidence:** READ BODY

#### C108. Dynamo progressive decode-affinity yield under KV pressure — author closed it after finding the existing soft-affinity support gave the same gain
- **WHO:** anish-shanbhag (NVIDIA Dynamo), PR #14443, opened 2026-09-08, closed 2026-09-08; maintainer comment from ishandhanani
- **URL:** https://github.com/ai-dynamo/dynamo/pull/14443
- **Closure status:** PR closed unmerged by its author (human reason, not a bot): the same performance improvement was already available via soft affinity.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Maintainer (ishandhanani): "We added a concept of hard/soft affinity in #13907 . Please leverage this and see if you can implement this PR as a custom policy" — Author (anish-shanbhag): "Thanks for flagging @ishandhanani , after some testing it looks like the same perf improvement can be achieved just by using soft affinity which is already supported, so am closing this PR"
- **Date:** opened Sep 8, 2026; closed Sep 8, 2026
- **Evidence:** READ BODY

#### C109. vLLM PoC “soft-pin recently-hit prefix-cache entries” — closed by the author as a workaround that would not fix the real bug
- **WHO:** manueldomke, vLLM
- **URL:** https://github.com/vllm-project/vllm/pull/42985
- **Closure status:** PR closed unmerged by the author (human, not a bot; not a maintainer decision), explicitly to avoid leaving a workaround in the review queue. (The S14 duplicate row for the same closure is merged here; its copy of the comment dropped the closing lines, the S9 copy is complete.)
- **STATED REASON, VERBATIM:** "Closing this too — same reasoning as #43302 (close comment). @stecasta's block-pool instrumentation on the #42948 thread (comment) shows the real bug is 131% pool overflow from the homogeneous lcm=256 physical block layout vs DSv4-Flash's {256, 64, 4, 8}-block-size KV groups. Under that overflow, some eviction every alloc cycle is mandatory; the recent-hit set in this PR can only delay the eviction of blocks that have already served a lookup hit, which by itself doesn't help the agentic-rotation case (per @stecasta's BS-sweep: 46% at BS=4, 21% at BS=8 — meaningful mitigation but not a fix). The real avenue is heterogeneous per-page-size pools so the bs=4 SWA indexer stops costing 64× its useful storage: #42082 (RFC) + #42374 (WIP). Closing this PR rather than leaving a workaround in the review queue. Thanks to everyone who looked at the earlier iterations. Refs: #42948, #42082, #42374."
- **Date:** created 2026-05-18, closed 2026-05-21
- **Evidence:** READ BODY

#### C110. vLLM popular-insert extension of the soft-pin — closed as superseded, and the superseding measurement later failed independent reproduction
- **WHO:** manueldomke, vLLM
- **URL:** https://github.com/vllm-project/vllm/pull/43191
- **Closure status:** PR closed unmerged by the author as superseded by #43302.
- **STATED REASON, VERBATIM:** "Superseded by #43302 . The protection-based approach in this PR can only mitigate the cascade collapse (max 46%/21% at BS=4/8 per stecasta's independent reproduction); the new PR closes it structurally via deliberate multi-storage with a pre-reserved pristine block pool, hitting 97%+ at the same workload. Closing."
- **Date:** created 2026-05-20, closed 2026-05-21
- **Evidence:** READ BODY

#### C111. vLLM ARC eviction for the GPU block pool, first attempt — closed by its author for a procedural reason and reopened as #27039
- **WHO:** albertoperdomo2, vLLM
- **URL:** https://github.com/vllm-project/vllm/pull/26921
- **Closure status:** CLOSED unmerged by the author (human) for a procedural reason, 7 minutes before reopening as #27039 (which merged).
- **STATED REASON, VERBATIM:** "I'll reopen the PR, messed up forcing the commit with sign off."
- **Date:** closed 2025-10-16; successor #27039 created 2025-10-16T16:21:38Z and merged 2025-11-12
- **Evidence:** READ BODY

#### C112. vLLM attention-sink-protected eviction policy request — withdrawn by its author after a collaborator’s clarification
- **WHO:** sahilmalik27 (issue author), vLLM
- **URL:** https://github.com/vllm-project/vllm/issues/36311
- **Closure status:** CLOSED (completed) by the issue author himself after a collaborator’s clarification — a withdrawn request, not a maintainer rejection.
- **STATED REASON, VERBATIM:** "Thanks for the quick clarification — you're right, and I should have read the block manager more carefully before opening this. vLLM only evicts free blocks, so sink tokens in active sequences are never at risk. The concern I was modeling (early token eviction corrupting attention distribution) applies to sliding-window KV cache modes like StreamingLLM, which vLLM doesn't implement. For standard PagedAttention, this is a non-issue. The underlying analysis (identifying sink tokens and spike channels) is still useful for quantization — knowing which hidden-state channels are outliers helps bitsandbytes/AWQ configs avoid clipping them. But that's a separate, narrower topic and doesn't belong in a KV eviction issue. Closing this. Apologies for the noise."
- **Date:** created and closed 2026-03-07
- **Evidence:** READ BODY

#### C113. Dynamo KVBM V2 transfer — merged revert, removing `BlockTransferHandlerV2` from the public API
- **WHO:** `jthomson04` (Dynamo)
- **URL:** https://github.com/ai-dynamo/dynamo/pull/5406
- **Closure status:** MERGED REVERT (`state=MERGED`, `mergedTime=2026-01-13T22:18:07Z`). The diff removes `BlockTransferHandlerV2` from the public API surface.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Commit subject: "Revert \"feat: KVBM V2 transfer (#4068)\"" — commit body: "This reverts commit 827b8c3e511295a53351b44b6af596c85f189454." The diff removes `BlockTransferHandlerV2` from the public API surface: `-pub use transfer::{BlockTransferHandler, BlockTransferHandlerV1, BlockTransferHandlerV2};` `+pub use transfer::BlockTransferHandler;`
- **Date:** created 2026-01-13, merged 2026-01-13
- **Evidence:** READ BODY

#### C114. Dynamo KVBM documentation claimed tiers and knobs that do not exist — correction PRs opened to retract the claims
- **WHO:** Dynamo maintainers (open PRs #14012, #14011), and a counter-PR #13867
- **URL:** https://github.com/ai-dynamo/dynamo/pull/14012
- **Closure status:** documentation/claim retraction: the advertised capabilities were never implemented; the correction PRs #14012 and #14011 were open at retrieval (a counter-PR #13867 also exists).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** PR #14012 title "docs(kvbm): drop the object-storage tier that has no implementation", body: "Nothing in the runtime reads any of them." PR #14011 title "fix(kvbm): remove the transfer batch size knob that nothing reads", body: "Nothing reads it."
- **Date:** 2026 (open at time of retrieval)
- **Evidence:** READ BODY

#### C115. LMCache merged revert of its MP adapter shim
- **WHO:** LMCache
- **URL:** https://github.com/LMCache/LMCache/pull/3111
- **Closure status:** merged revert, 2026-04-23.
- **STATED REASON, VERBATIM:** "This reverts only the adapter shim (restores the pre-#3100 signature requiring parallel_strategy)."
- **Date:** 2026-04-23
- **Evidence:** READ BODY

#### C116. SGLang AMD DeepSeek-V4 unified-KV pool sizing — merged revert, then relitigated and relanded the next day with gating
- **WHO:** `Liangsheng Yin` (revert author, merged)
- **URL:** https://github.com/sgl-project/sglang/pull/38163
- **Closure status:** MERGED REVERT (`state=MERGED`, `mergedTime=2026-09-06T00:28:47Z`); the change was relanded the next day by PR #38192 with gating. No narrative reason was retrievable in the revert PR (PR comment bodies did not render for the evidence agent).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Commit subject: "Revert \"[AMD][DSV4] Fix unified-KV pool sizing and SWA ring accounting (#30315)\"" — body: "This reverts commit 514b45fd3447a300c55a2532892a2c8539d3cd6d." No narrative reason was retrievable (PR comment bodies do not render).
- **Date:** created and merged 2026-09-06; reland merged 2026-09-07
- **Evidence:** READ BODY

#### C117. TensorRT-LLM merged revert of the FlashInfer→CuTeDSL MLA backend path about two weeks after it landed
- **WHO:** `pengbowang-nv`
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/pull/18653
- **Closure status:** MERGED REVERT (`state=MERGED`, `mergedTime=2026-09-08T08:29:00Z`), preserving the standalone CuTeDSL FMHA backend and later changes.
- **STATED REASON, VERBATIM:** "Remove the FlashInfer-to-CuTeDSL MLA backend path introduced by #17800 while preserving the standalone CuTeDSL FMHA backend and later DSA, Helix, combined-FMHA, and sysinfo changes."
- **Date:** created 2026-09-03, merged 2026-09-08 (reverting #17800, which had landed roughly two weeks earlier)
- **Evidence:** READ BODY

#### C118. SGLang zigzag prefill context parallelism — withdrawn (“temporarily unavailable”) for every current DeepSeek/GLM model
- **WHO:** SGLang project
- **URL:** https://docs.sglang.ai/advanced_features/dcp.html
- **Closure status:** feature disabled / withdrawn at the documentation level — a shipped technique pulled for the models that most need it, replaced by `interleave` with `--dp 1`. Not a PR closure. The quoted text lives in the SGLang cookbook pages; VERIFIER NOTE: the row’s headline DCP page does not contain the word “zigzag”, so the sourcing is imprecise.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Zigzag prefill CP (`--cp-strategy zigzag`) is temporarily unavailable for DeepSeek V3.2, GLM-5, GLM-5.1, GLM-5.2, and GLM-5.3. Use `interleave` for these models and keep `--dp 1`; interleave DSA CP does not support `--dp` greater than 1. - Also verbatim from GLM-5.3 (`docs/cookbook/autoregressive/GLM/GLM-5.3.mdx` line 246): "Zigzag prefill CP (`--cp-strategy zigzag`) is temporarily unavailable for GLM-5.3. For prefill CP on CUDA, use `interleave` with `--dp 1` as shown below." - Also verbatim from GLM-5 (`docs/cookbook/autoregressive/GLM/GLM-5.mdx` line 97): "**Prefill CP on CUDA**: Zigzag (`--cp-strategy zigzag`) is temporarily unavailable for GLM-5. Use `--enable-prefill-cp --cp-strategy interleave` with `--dp 1`."
- **Date:** docs current in the retrieved SGLang tree (2026-09); line numbers in the evidence file are from a local checkout and are not reproduced here
- **Evidence:** READ BODY

#### C119. vLLM revert of “[Nixl][PD] DCP support for MLA models” after it broke DeepSeek-V4-Flash — the revert PR was itself closed unmerged in favour of a narrower fix
- **WHO:** vllm-agent (author of the revert PR), 1 commit
- **URL:** https://github.com/vllm-project/vllm/pull/54386
- **Closure status:** revert PR closed unmerged; the narrower fix #54400 was merged the same day. The substantive reason is a human/automated-agent technical statement in the PR body, not a stale timer.
- **STATED REASON, VERBATIM:** "The regression is one gated assignment. In SparseAttnIndexer.cp_kv_cache_interleave_size , the memoization is conditional: if isinstance ( get_forward_context (). attn_metadata , dict ): self . _cp_kv_cache_interleave_size = value return value When attn_metadata is not a dict , the value is never cached, so every subsequent forward re-enters get_current_vllm_config() . Outside a set_current_vllm_config() context that raises. Caching unconditionally on first access — or resolving the value eagerly in Worker.initialize_from_config , as the docstring says already happens — should fix this without a revert. If a maintainer lands that, please close this PR."
- **Date:** closed 2026-08-30
- **Evidence:** READ BODY

#### C120. vLLM ACE — Attention-Weighted Context Eviction for multi-turn agentic tool results — withdrawn by its own author after benchmarking
- **WHO:** KikoCisBot, PR author, on vLLM (PR #42645 "[RFC] feat(context): ACE — Attention-Weighted Context Eviction for multi-turn tool results")
- **URL:** https://github.com/vllm-project/vllm/pull/42645
- **Closure status:** PR closed UNMERGED by the author after a self-run benchmark; the author states the design lost to doing nothing. Not stale-bot, not maintainer-closed — an author withdrawal with measured data.
- **STATED REASON, VERBATIM:** "Withdrawing this: I measured it against doing nothing, and it lost Closing this rather than leaving it open, because I no longer believe the design in it."
- **Date:** created 2026-05-14 (first author comment 2026-05-14T14:12:48Z); closed 2026-08-10 (closedTime 2026-08-10T05:36:27Z)
- **Evidence:** READ BODY

#### C121. SGLang stripping the Anthropic billing header so Claude Code turns can share a prefix — closed unmerged by its author
- **WHO:** joninco, PR author, on SGLang
- **URL:** https://github.com/sgl-project/sglang/pull/21064
- **Closure status:** PR closed UNMERGED by the author (joninco closed this May 10, 2026); no maintainer merge, the fix did not land.
- **STATED REASON, VERBATIM:** "Anthropic SDK clients (e.g. Claude Code) include an `x-anthropic-billing-header` block in the system prompt that contains a per-request conversation context hash (`cch=...`). Because this block is the first system prompt entry and changes on every request, it produces a different token prefix each time, defeating RadixAttention prefix caching for the entire conversation."
- **Date:** created 2026-03-21, closed 2026-05-10
- **Evidence:** READ BODY

#### C122. vLLM auto-revert of “[Mamba] enable prefix cache by default” — the revert PR was closed unmerged in favour of the forward fix
- **WHO:** vllm-agent (CI failure analyzer bot-authored PR), closed by noooop
- **URL:** https://github.com/vllm-project/vllm/pull/51098
- **Closure status:** revert PR closed UNMERGED (`mergedTime` null; “noooop closed this Aug 5, 2026”). The underlying failures were real and are recorded verbatim.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** The PR removed the and not model_config.is_hybrid guard in _set_default_chunked_prefill_and_prefix_caching_args (vllm/engine/arg_utils.py), so prefix caching now defaults to on for hybrid/Mamba models, and vllm/model_executor/models/config.py now unconditionally defaults mamba_cache_mode to align" instead of "all". Two consequences: "align" mode asserts scheduler_config.enable_chunked_prefill. Any hybrid-model configuration that explicitly passes enable_chunked_prefill=False — previously valid, since prefix caching was off — now fails hard at VllmConfig validation."
- **Date:** opened 2026-08-05, closed 2026-08-05
- **Evidence:** READ BODY

#### C123. vLLM multi-storage replication for shared-prefix KV blocks — closed by its author after an independent test showed it was a regression
- **WHO:** manueldomke (vLLM contributor); independent test by stecasta
- **URL:** https://github.com/vllm-project/vllm/pull/43302
- **Closure status:** PR closed unmerged by the author. The failure was caught by an independent reproduction (`stecasta`), not by the author’s own benchmark: 0% hit rate and +10–13% wall time versus the no-cache baseline.
- **STATED REASON, VERBATIM:** Pre-reserving 4 GB of pristine blocks (the mechanism in this PR) under that constraint makes the regression strictly worse: effective pool drops ~10%, eviction pressure rises, even the chat-template prelude survivors get wiped. Hence the 0% hit rate. The mechanism does demonstrably help the one-shared-prefix-across-many-sessions shape my reproducer tested (chat-completion with large system prompt, RAG with shared retrieval context), and the 4-step #42948 issue-body reproducer hits 99.95% on A.3 with this PR. But those are narrower than the agentic-rotation case that drives the real production pain, and shipping a connector that worsens the more-common workload to land a narrower one isn't acceptable. and "Thanks to @stecasta for the careful re-test that prevented this from being merged as a regression."
- **Date:** created May 21, 2026; closed May 21, 2026. The independent test result, verbatim from stecasta in issue #42948: "| BS | N | Vanilla hit | This PR hit | This PR wall vs nocache baseline | ... | 4 | 8 | 1.0% | **0.0%** | +10% slower (348 s vs 317 s) | | 8 |16 | 0.3% | **0.0%** | +13% slower (581 s vs 514 s) |" and "Both prefix_cache_hit_rate and external_prefix_cache_hit_rate reported 0% throughout."
- **Evidence:** READ BODY

### C.5 Negative results: measured, and it did not pay off

#### C124. Elastic KV cache — reclaiming the prefill-activation reserve; built, measured, and reported as a negative result
- **WHO:** Sathishkumar Sivashanmugam (single-author paper)
- **URL:** https://arxiv.org/abs/2608.23658
- **Closure status:** published negative result — the mechanism was built and explicitly judged not to pay off (the penalty the premise requires is ~1% between chunk sizes of 8192 and 32768; simply lowering `max_num_batched_tokens` recovers more KV).
- **STATED REASON, VERBATIM:** "Having built the mechanism, we test the premise it rests on and report an honest negative result. It only pays off if a small prefill chunk size badly hurts prefill latency. In a controlled experiment injecting long prompts into a live decode load, that penalty is small (median time-to-first-token differs by about 1% between chunk sizes of 8192 and 32768 tokens), because prefill is compute bound and decode consumes only about one token per sequence per step. Simply lowering max_num_batched_tokens recovers more KV than the controller does, at nearly equal latency. The reserve also dilutes under tensor parallelism, from 16% of KV at TP1 to 2.7% at TP4. We state precisely when reclaiming the reserve could still help, and release the mechanism as a reusable userspace elastic-VMM allocator."
- **Date:** submitted 24 Aug 2026
- **Evidence:** READ BODY

#### C125. External/NVMe KV caching falls below break-even on a fast GPU — the authors’ own admission rule
- **WHO:** Joseph Kanichai, Tiziano De Matteis, Animesh Trivedi — “Building py-kvcache: A Performance Characterization of External KV Caching for vLLM with NVMe SSDs” (arXiv 2609.11744); also carried as S11-ABANDONED-44, S13-ABANDONED-18 and S5-ABANDONED-8
- **URL:** https://arxiv.org/abs/2609.11744 | https://arxiv.org/html/2609.11744v1
- **Closure status:** published negative/limiting result for the cross-tier direction on fast hardware: external KV caching should be treated as a per-setup admission decision, not a general win.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Prefix caching can reduce the time to first token (TTFT) of long-context LLM requests by reusing previously computed key-value (KV) states, but for short prefixes or fast GPUs, recomputation can be faster than loading from an external cache. And the decisive measurement, verbatim: "Bailian trace replays improve TTFT on a weaker GPU, but on an H100 the average request falls below the break-even point and GPU memory alone retains enough prefixes. External KV caching should therefore be treated as a setup specific admission decision." — and, the same paper’s break-even and below-8k findings (the S5 row’s quotation had dropped the word “as”; corrected here): The same traces show a different outcome on the Snellius node. Their average request length is below the measured 6,203 token SSD break-even point for Qwen3 4B, on our setup, and the H100's larger GPU memory retains a large portion of the working set. As a result the TTFT of only using GPU prefix caching is identical to that of py-kvcache , while the native vLLM KV Offload implementation is 0.2 s higher. and "Below roughly 8k tokens the prefill is short enough that no storage device we measured can load the prompt faster than the GPU rebuilds it, and on a mid-range drive that threshold moves out to 32k. Therefore, external cache admission must depend on the model, GPU, SSD, and reusable prefix length rather than treating every hit as beneficial."
- **Date:** submitted 10 Sep 2026
- **Evidence:** READ BODY

#### C126. Exact composition of all cached recurrent states for hybrid prefix reuse — unnecessary and, on some architectures, harmful
- **WHO:** Yirui Liu, Ruoling Qi, Longwen Wang, Xuaner Wu, Jian Chen, Yuxin Jin, Jiawei Shao, Xuelong Li — "LinearKV"
- **URL:** https://arxiv.org/abs/2608.11231
- **Closure status:** published negative result against a specific competing technique (exact state composition, attributed to concurrent work HYPIC) inside the hybrid prefix-caching design space.
- **STATED REASON, VERBATIM:** "The algebraically principled alternative---composing all $K$ cached states into the exact full-prefix state, as concurrent work HYPIC does---is unnecessary and, on some architectures, even harmful. We compare the two across three hybrid models and three PIC selectors. On the two GDN models the two tie, both recovering most of full quality (up to $92\%$); on the Mamba-2 model, exact composition instead collapses under every selector---under EPIC, for instance, it recovers only $46.6\%$ of full quality, versus $86.8\%$ for a single cached block initializer. A single state initializer is also cheaper, cutting time-to-first-token to $0.46\times$ full prefill versus a further $5$--$17\%$ overhead for exact composition."
- **Date:** submitted 31 Jul 2026
- **Evidence:** READ BODY

#### C127. Repairing reused KV for non-prefix positions is often worse than not reusing at all (KVShareArena)
- **WHO:** Xi Shi, Qian Lou — "KVShareArena"
- **URL:** https://arxiv.org/abs/2609.10266
- **Closure status:** published negative result about the value of KV reuse outside the exact-prefix case, including a negative result for cache compression on agent reports and for trained adapters across checkpoints.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** We find that correcting positions, which needs no recomputation, is enough until a question needs several sources at once. There, only methods that pay, by re-encoding part of the cache or by training, recover half to two thirds of the gap; unrepaired caches can be worse than no cache. Cache-compression methods that are harmless on a single prompt fall significantly behind position correction on freshly written agent reports. And: "When a different checkpoint wrote the cache, training-free methods are barely affected, while an adapter trained on one checkpoint's caches loses quality."
- **Date:** submitted 9 Sep 2026
- **Evidence:** READ BODY

#### C128. Attention-score KV eviction — the selection signal contributes almost nothing (Random Attention)
- **WHO:** authors of "Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning" (arXiv 2609.03430), submitted 2026-09-03.
- **URL:** https://arxiv.org/abs/2609.03430
- **Closure status:** published negative result against the whole scored-selection family (H2O/SnapKV/Ada-KV/TriAttention-style scoring): random eviction within each head matches the strongest prior evictor while serving 32–43% higher throughput.
- **STATED REASON, VERBATIM:** We show that the selection signal contributes almost nothing. Random Attention keeps the prompt and evicts uniformly at random within each attention head, computing no score at all; across four models and six reasoning tasks it matches the strongest prior evictor while serving 32-43% higher throughput than it in vLLM deployment. Mechanism, VERBATIM: "1) the prompt is the fragile part of the cache, and most of the gap between selectors is just whether their selection signal happened to keep it; 2) the reasoning trace protects itself against eviction with redundancy at two levels, in the text (the model restates what it still needs as it works) and across attention heads (each keeps its own copy of the trace), so once the prompt is safe, a random draw retains enough copies of what the model still needs, and no score is required to pick them."
- **Date:** submitted 2026-09-03.
- **Evidence:** READ BODY

#### C129. “Protection Is (Nearly) All You Need” — structural protection dominates scoring; H2O/SnapKV/StreamingLLM/Ada-KV/QUEST collapse without it
- **WHO:** authors of arXiv 2605.18053 (38 pages, code and figure regeneration scripts released), submitted 2026-05-18.
- **URL:** https://arxiv.org/abs/2605.18053
- **Closure status:** published negative result: seven policies collapse to near-zero quality without structural protection, and with protection the simplified score-isolation variants are TOST-equivalent to LRU. Linear online credit estimation adds nothing (the paper’s own §8 “Negative Result”).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Seven policies (LRU, H2O, SnapKV, StreamingLLM, Ada-KV, QUEST, Random) share a prompt-boundary vulnerability: without structural protection, they collapse to near-zero quality on six pure-transformer models (F1$\leq$0.064). and "With protection, simplified score-isolation variants are TOST-equivalent to LRU at $K{=}32$ ($\Delta{=}0.02$)". Overall conclusion, VERBATIM: "Overall: protection dominates; scoring differences are secondary once boundaries are guarded; per-head allocation gives a further modest gain." Its attention-mass measurement, VERBATIM: "the position-0 sink holds ${\sim}75\%$ of prefix mass, while other boundary tokens sit near ${\sim}0.41{\times}$ uniform expectation, so attention scorers retain the sink but still drop structurally critical tokens." — and, the paper’s explicit §8 negative result on learned credit estimation: Our negative result from the original online-credit experiment, bounded to a 4-feature linear estimator, suggests a cautionary lesson for linear online credit estimators in KV cache management. When the easy" structural fix accounts for the entire quality gap, there may be insufficient signal for simple learned components to exploit." Section heading, verbatim: "8 Negative Result: Linear Credit Estimation Adds Nothing". Second negative finding: "H2O-faithful's nominal advantage ( $p = 0.043$ ) does not survive Holm–Bonferroni correction." (approximate — see verification note)
- **Date:** submitted 2026-05-18.
- **Evidence:** READ BODY

#### C130. Rate–distortion survey — attention-magnitude/recency retention signals fail uniformly and irreversibly, and repeated agent compaction is almost never measured
- **WHO:** authors of "What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in LLMs and Agents" (arXiv 2607.08032), submitted 2026-07-09.
- **URL:** https://arxiv.org/abs/2607.08032
- **Closure status:** published negative characterisation across four research communities; the paper turns the observation into a benchmark proposal rather than a fix.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Two patterns hold across the survey. At every layer the signal that decides what to keep is attention magnitude or recency, and it fails in the same way everywhere, by discarding, before the query is known and with no way to undo it, information the query later needs. - Second finding, VERBATIM (directly about this subsection's evaluation practice): "And while compression is measured carefully on single-turn long context, the repeated compaction that agents actually perform is almost never measured, and no benchmark holds one budget axis across all the layers at once."
- **Date:** submitted 2026-07-09.
- **Evidence:** READ BODY

#### C131. VarRate — SnapKV/Ada-KV-style token selection collapses 11–15 points under query-agnostic reuse
- **WHO:** authors of "VarRate: Training-Free Variable-Rate KV Cache Compression for Long-Context LLMs" (arXiv 2607.15498), submitted 2026-07-16.
- **URL:** https://arxiv.org/abs/2607.15498
- **Closure status:** published negative result on irreversible eviction, specifically naming SnapKV and Ada-KV; the paper abandons eviction in favour of rank allocation.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Two leading training-free families are both structurally limited: token-selection methods (SnapKV, Ada-KV) score importance from an observation window and evict low-scoring tokens, but eviction is irreversible -- so when the importance signal degrades under query-agnostic reuse, accuracy collapses by 11-15 points; uniform low-rank coding keeps every token but spends equal rank everywhere, wasting budget. We observe that both failures share one cure: rank should be allocated, not evicted. Result, VERBATIM: "Because no token is dropped, it degrades by only 3.5-5.5 points where query-aware selection collapses."
- **Date:** submitted 2026-07-16.
- **Evidence:** READ BODY

#### C132. “More GPUs or a Smaller Cache?” — no cost-equivalence crossover found; extra GPUs are largely wasted spend below a model-size wall
- **WHO:** authors of arXiv 2608.23962, submitted 2026-08-25.
- **URL:** https://arxiv.org/abs/2608.23962
- **Closure status:** published negative result against the “add tensor parallelism instead of compressing the cache” answer to KV memory pressure (compression cheaper by 1.20×–2.00×; crossover at roughly 36B parameters on an 80 GB card).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** We place tensor-parallel configurations (degree 1 to 8) and KV-compressed configurations (16/8/4-bit, keep-ratios down to 0.25) on one costnormalised axis, cost per million tokens against latency, using a profiled simulator calibrated on A100, A40, and H100 hardware, and we go looking for the cost-equivalence crossover. We do not find one. Across two models (Llama-2 at 7B and 70B), three GPU types, and every level of memory relief we could construct, compression is cheaper by 1.20x to 2.00x. Boundary, VERBATIM: "A 7B model on an 80 GB device cannot exhaust its KV budget within its own context window, and the boundary that decides between the strategies is model size relative to device memory, at roughly 36B parameters for an 80 GB card. Below that wall, compression dominates and extra GPUs are largely wasted spend". Counter-finding, VERBATIM: "Tensor parallelism is the only lever that improves latency (compression makes per-token latency worse, by 8 to 93%, through batching contention)".
- **Date:** submitted 2026-08-25.
- **Evidence:** READ BODY

#### C133. VaSE — eviction often yields worse accuracy than selection-based sparse attention; a few large-magnitude value states cause catastrophic failure when evicted
- **WHO:** authors of "Value-Aware Stochastic KV Cache Eviction for Reasoning Models" (arXiv 2606.03928), submitted 2026-06-02.
- **URL:** https://arxiv.org/abs/2606.03928
- **Closure status:** published negative result on the eviction-vs-full-cache trade-off, plus a documented failure mode of magnitude-agnostic scoring.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** KV cache eviction methods reduce this cost by evicting unimportant key-value pairs from the cache, yet they often yield worse accuracy than selection-based sparse attention alternatives, which keep the full KV cache. Failure mode, VERBATIM: "First, a small fraction of value states have abnormally large magnitudes, and evicting them causes catastrophic failure where models enter repetitive reasoning loops." - Corroborating independent account of the same runaway-degeneration failure, VERBATIM (from arXiv 2608.15797): "Under aggressive budgets, this not only lowers accuracy but can also cause runaway degeneration, where the model produces incoherent or repetitive tokens until reaching the length limit."
- **Date:** submitted 2026-06-02.
- **Evidence:** READ BODY

#### C134. QJL residual correction / TurboQuantProd keys — rejected on independent measurement; MSE beats the paper’s recommended Prod path
- **WHO:** scos-lab (`turboquant` reference implementation README) and `Arclabs001` in the ggml-org/llama.cpp discussion #20969 (verifier-corrected authorship)
- **URL:** https://raw.githubusercontent.com/scos-lab/turboquant/refs/heads/main/README.md | https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** technique rejected after measurement: the reproduction concludes the paper’s recommended Prod path should not be used for attention, with the variance cost quantified. VERIFIER CORRECTION: the first quoted fragment in the S4-ABANDONED-14 row was posted by `Arclabs001`, not by `scos-lab`; only the second fragment is scos-lab’s.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** The paper recommends TurboQuantProd for Keys (unbiased inner product) and TurboQuantMSE for Values. Our experiments show **MSE for both is better** with the table "| GPT-2, b=4 | MSE (both) | Paper (Prod keys) | | PPL change | +1.1% | +6.5% |" and "**Why:** TurboQuantProd's QJL residual correction adds variance. Softmax attention amplifies variance more than bias. Low variance (MSE) beats unbiasedness (Prod) in practice." — and, the quantified variance finding in the same thread: QJL eliminates bias but explodes variance. And for attention, variance is worse than bias! with the preceding measured rows "-0.02%" and "+368.6%". And: "To clarify: the 300% PPL increase was specifically from using TurboQuantProd (the QJL residual correction method from Theorem 2 of the paper) for Key quantization at b=3 on GPT-2 (head_dim=64)."
- **Date:** README retrieved 2026-09-13; the same author's restatement in the llama.cpp thread is dated Mar 28, 2026
- **Evidence:** READ BODY

#### C135. vLLM’s own comprehensive evaluation concludes TurboQuant k8v4 is not worth using and the aggressive presets are unfit for production
- **WHO:** vLLM team (official vLLM blog, "A First Comprehensive Study of TurboQuant: Accuracy and Performance")
- **URL:** https://vllm.ai/blog/2026-05-11-turboquant
- **Closure status:** published negative result from the project that shipped the technique — the maintainers’ own recommendation is to prefer FP8 and avoid the aggressive presets.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** TurboQuant k8v4 does not provide any significant advantage over FP8. This TQ variant only provides modest KV-cache savings (2.4x vs 2x), which are not worth the consistent negative impact on throughput and latency metrics. And: "Avoid TurboQuant k3v4-nc and 3bit-nc without thorough validation. These aggressive variants can cause drastic accuracy drops that reach up to 20 points on challenging math and coding benchmarks. In addition to accuracy, their consistent performance degradation due to complex dequantization steps renders them unsuitable for production deployments." And the headline: "FP8 ( --kv-cache-dtype fp8 ) remains the best default for KV-cache quantization." Throughput verbatim: "All TurboQuant variants are strictly below BF16: on Qwen3-30B (Figure 9), ranging from 80% (k8v4) to 73% (3bit-nc)".
- **Date:** published May 11, 2026; benchmarked on vLLM 0.20.2 (commit 6ec9bbec3)
- **Evidence:** READ BODY

#### C136. turbo3/turbo4 KV cache types judged worse than plain q4_0 on perplexity over all of wiki.test.raw
- **WHO:** `ubergarm`, in the ggml-org/llama.cpp discussion #20969 (verifier-corrected authorship; the evidence file named the addressee)
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** community measurement reporting the new format loses to an existing simpler format — no adoption followed. VERIFIER CORRECTION: the post is by `ubergarm` (who opens with “@Dampfinchen”), not by Dampfinchen.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** "I tested and turbo3 and turbo4 look much worse than even q4_0 in my testing: ikawrakow/ik_llama.cpp#1509 (comment) I'm running llama-perplexity over entire wiki.test.raw . command and method shown in the linked details."
- **Date:** post dated Mar 27, 2026
- **Evidence:** READ BODY

#### C137. TurboQuant’s advertised sub-4-bit compression does not survive per-block metadata on a real 27B hybrid model
- **WHO:** a contributor running the TurboQuant × MTP correctness battery, in the ggml-org/llama.cpp discussion #20969
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** measured shortfall against the advertised ratio (2.75× at 256K instead of the ≈5× full-bit-rate ratio; ~6 bits/element once per-block norms are counted), plus a demonstration that MTP consumes the freed headroom.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** TurboQuant compresses KV by 2.75× at 256 K (16.00 → 5.81 GiB). The full-bit-rate ratio is ≈ 5×, but the per-block norms in turbo3 push effective bits/element to ≈ 6. And: "MTP roughly halves the practical ceiling on a 4090. Even with the smaller turbo3 KV (3.16 GiB at 136 K), TQ+MTP at 136 K is already at 24.02 GiB / 24.56 GiB total. The MTP draft branch's own buffers (compute, n_ubatch -scaled allocations) eat the headroom that TurboQuant freed up."
- **Date:** post in the thread (thread started Mar 25, 2026)
- **Evidence:** READ BODY

#### C138. turbo3 V-cache degrades generation-trajectory scores badly even when output-level smoke tests look clean
- **WHO:** sztlink (measurement) as reported and qualified by another contributor in the ggml-org/llama.cpp discussion #20969
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** measured decode-path divergence from the f16 reference that output-level smoke tests do not detect (the gap grows with model size and quantisation aggressiveness).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** @sztlink 's data on this very thread shows turbo3 V-cache degrades trajectory scores significantly on 27B (≈ 58) and dramatically on 32B (≈ 25) even when outputs look fine on smoke tests. Related in-thread finding, verbatim: "GTM says q8_0/turbo3 is near-excellent (88.83); trajectory says it has materially broken generation paths (FAIL). The gap grows with model size and quantization aggressiveness."
- **Date:** posts in the thread (started Mar 25, 2026)
- **Evidence:** READ BODY

#### C139. GPUDirect Storage (cuFile/KvikIO) prototype for KV cache — slower than every other implementation tested
- **WHO:** t348575 et al., py-kvcache (arXiv 2609.11744), §6.1 "Selecting the Storage Interface"
- **URL:** https://arxiv.org/html/2609.11744v1
- **Closure status:** negative result in a published paper — the authors built the GDS prototype and rejected it.
- **STATED REASON, VERBATIM:** "We separately tested whether GPUDirect Storage could remove the intermediate CPU transfer. We implemented a simple prototype using KvikIO, which provides Python bindings to cuFile and can access GPU buffers through GPUDirect Storage [ 28 ] . In our configuration, this path was slower than all our other implementations."
- **Date:** paper submitted 2026-09-10
- **Evidence:** READ BODY

#### C140. io_uring / xNVMe / SPDK userspace NVMe paths did not improve end-to-end TTFT
- **WHO:** t348575 et al., py-kvcache (arXiv 2609.11744), §5 "I/O engine and request size" and §6.1
- **URL:** https://arxiv.org/html/2609.11744v1
- **Closure status:** negative result in a published paper — the faster I/O engines improved synthetic microbenchmarks but not vLLM TTFT, so they were dropped from the final design.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Correcting the backend improved storage microbenchmarks ( 8(a) ), but did not improve our vLLM TTFT. This result agreed with our prior characterization, since the workload consists of large, bandwidth limited reads and writes, so reducing overhead of I/O operations does not necessarily shorten the request's critical path. and "However, replacing the I/O path did not measurably improve TTFT in the LLM benchmark." and "High small-I/O IOPS and additional worker threads can improve a synthetic microbenchmark without improving end-to-end inference."
- **Date:** paper submitted 2026-09-10
- **Evidence:** READ BODY

#### C141. llm-d’s filesystem KV cache was deprecated mid-experiment and its staging allocation caused ~2.3× data amplification
- **WHO:** t348575 et al., py-kvcache (arXiv 2609.11744) §3.2/§5.1; llm-d project (`llm-d/llm-d-kv-cache`)
- **URL:** https://arxiv.org/html/2609.11744v1 | https://raw.githubusercontent.com/llm-d/llm-d-kv-cache/main/README.md
- **Closure status:** negative result plus upstream deprecation in a published paper; the standalone FS connector reached a final release and is now upstreamed into vLLM as the FS tier of the multi-tier offloading connector.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** However, this was deprecated while our experiments were ongoing, in favour of the secondary filesystem disk tier introduced directly into vLLM. and, on the measured defect, "The trace showed that llm-d copied and submitted data separately for each cache chunk. More importantly, its minimum staging-buffer allocation was larger than the KV data required by our test model. A request containing 4.4 GB of useful KV data consequently caused approximately 10 GB to be copied and read or written." — and, the llm-d README’s own sunset/upstream note: "**Now upstreamed into vLLM.** `llmd-fs-connector==0.23` (llm-d v0.8 / vLLM v0.23) is the **final release** — llmd-fs-backend is now the FS tier of vLLM's multi-tier offloading connector (`TieringOffloadingSpec`). All new features and support continue there"
- **Date:** paper submitted 2026-09-10
- **Evidence:** READ BODY

#### C142. SSD-backed KV cache restore is “no longer beneficial” on modern engines, and GPU Direct Storage does not rescue it (Tutti’s own measurements)
- **WHO:** Tutti authors (arXiv 2605.03375) — the S14 row for the same finding had no quote of its own and is merged here
- **URL:** https://arxiv.org/html/2605.03375v1 | https://arxiv.org/abs/2605.03375
- **Closure status:** negative results inside a published paper — SSD restore loses to DRAM (GPU bubbles >70%, ~80% with layer-wise transfers), GDS still relies on CPU intervention and leaves GPU bubble time above 70%; the paper reports a 98.3% hit-rate crossover below which the SSD tier is not beneficial.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Moreover, as LLM engines continuously optimize inference computation, restoring KV cache from SSDs is no longer beneficial due to severe I/O bottleneck. Supporting measurements verbatim from the same section: "restoring KV cache from SSDs performs much worse than from DRAM, causing GPU bubbles to exceed 70% of total inference latency in all cases." and "Applying layer-wise transfers on SSDs (SSD-LW) further reduces I/O granularity and increases the number of operations, inflating end-to-end latency and pushing GPU bubble time to around 80% of total inference latency." — and, the same paper’s GDS verdict in §2.2: GDS ( Nvidia, 2024 ) removes CPU-GPU copies through peer-to-peer DMA, but still relies on CPU intervention to initiate each I/O, incurring substantial software overhead and limiting I/O parallelism ( DeepSpeedAI, 2025 ; Li et al., 2025 ) . Even with GDS, GPU bubble time remains high at above 70%, indicating that eliminating the CPU from the data path alone hardly alleviates the mismatch between paged KV layouts and SSD access patterns. The abstract states the same verdict verbatim: "rendering the low-parallelism CPU a severe bottleneck even with GPU Direct Storage (GDS), which still relies on CPU intervention to initiate each I/O and thus remains CPU-centric."
- **Date:** paper submitted 2026-05-05
- **Evidence:** READ BODY

#### C143. Multi-SSD KV offloading collapses to the baseline on a single SSD (Swarm)
- **WHO:** Swarm authors (arXiv 2603.17803), §8.4
- **URL:** https://arxiv.org/html/2603.17803v1 | https://arxiv.org/abs/2603.17803
- **Closure status:** the paper’s own scaling result — no benefit in the single-device configuration, which is exactly the single-node single-local-disk case.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** SSD Number. Figure 18 shows throughput as the number of SSDs increases from 1 to 8. With a single SSD, Swarm falls back to the baseline. As the number increases, its throughput scales steadily and consistently surpasses all baselines. The paper's own framing of the underlying limit is verbatim in the abstract: "Solid-state drives (SSDs) provide a cost-effective alternative, but naive SSD-based paging is fundamentally bandwidth-bound due to limited PCIe throughput and per-device bandwidth constraints."
- **Date:** paper submitted 2026-03-18
- **Evidence:** READ BODY

#### C144. KV offloading degrades accuracy on context-intensive tasks (Yandex) — low-rank projection of keys and unreliable landmarks
- **WHO:** Yandex authors, "KV Cache Offloading for Context-Intensive Tasks" (arXiv 2604.08426)
- **URL:** https://arxiv.org/abs/2604.08426
- **Closure status:** negative empirical result in a published paper (significant degradation on both Llama 3 and Qwen 3); the authors nonetheless argue the technique is salvageable with better compression and key selection.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key reasons for poor accuracy: low-rank projection of keys and unreliable landmarks. Second bounded result verbatim: "Our observations suggest that some model-benchmark pairs require very large budgets (over 10%) to achieve near-lossless accuracy." The authors do NOT declare the technique dead — verbatim: "These are not conceptual problems with the offloading itself, but technical limitations that can be circumvented with better compression and key selection. Our 'view from trenches' shows that KV offloading is still a good candidate for production use across both easy and context-intensive tasks."
- **Date:** submitted 2026-04-09 (v5 2026-09-01)
- **Evidence:** READ BODY

#### C145. KV offload to host DRAM is memory-bound by construction on typical workloads — 99% of latency spent on transfers, GPUs at 28% of rated TDP
- **WHO:** authors of "Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading" (arXiv 2601.19910)
- **URL:** https://arxiv.org/abs/2601.19910
- **Closure status:** negative analytic + empirical result in a published paper: typical workloads exceed the critical cached-to-prefill token ratio by orders of magnitude.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** KV cache offloading enables long-context LLM inference by storing caches in CPU DRAM, but PCIe bandwidth limitations create severe bottlenecks. In this paper, we develops an analytical framework that derives κ_crit, the critical cached-to-prefill token ratio where execution becomes memory-bound and show typical workloads exceed this threshold by orders of magnitude. Empirical characterization reveals 99% of latency spent on transfers and serving offloaded requests results in GPU's consuming only 28% of their rated TDP, motivating our proposed optimizations for hardware interconnects, model architectures, and scheduling algorithms. Supporting figure verbatim (body §2.3, delegated retrieval): "HBM, tightly integrated on a silicon interposer, delivers TB/s, while CPU-GPU PCIe 5.0 provides only 64 GB/s—2% of HBM's bandwidth. Transferring a 50 GB KV cache takes 15 ms from HBM but 800 ms from CPU DRAM."
- **Date:** submitted 2025-12-16
- **Evidence:** READ BODY

#### C146. HERALD’s own measurement — host-memory KV fetch overtakes the forward pass at 8K context (3.12× slower; 7.61× at 32K)
- **WHO:** HERALD authors (arXiv 2606.21633), §2.3
- **URL:** https://arxiv.org/abs/2606.21633
- **Closure status:** the paper’s own negative characterisation of the PCIe tier, which is why it abandons per-step full-cache fetching.
- **STATED REASON, VERBATIM:** "However, the PCIe interconnect between host and device provides roughly two orders of magnitude lower bandwidth than HBM in GPUs, making the transfer cost a critical bottleneck. As illustrated in Figure 3 (b), at a batch size of B = 64, the per-step latency of fetching the full KV cache already surpasses the model forward latency starting from 8K context, where it is 3.12x slower. This gap widens drastically as the sequence extends, reaching 7.61x at 32K context. Even with a hypothetically perfect overlap of communication and computation, the per-step latency would still be limited by max(forward, fetch)."
- **Date:** submitted 2026-06-19 (v2 2026-08-03)
- **Evidence:** READ BODY

#### C147. Performing attention on the CPU in KV-offload systems is argued to be on the wrong side of the roofline (KVDrive)
- **WHO:** KVDrive authors (arXiv 2605.18071), §6.3
- **URL:** https://arxiv.org/abs/2605.18071
- **Closure status:** published rebuttal of a design family (“prior works advocate performing attention on the CPU in KV cache offloading systems”) — the paper argues that regime is bandwidth-roof-bound.
- **STATED REASON, VERBATIM:** "Prior works advocate performing attention on the CPU in KV cache offloading systems, arguing that CPU–GPU transfer overhead dominates GPU execution. We instead analyze why GPU-based attention is preferable in our system, using the roofline model. ... When the operational intensity falls below the threshold P, transferring data to the GPU yields no benefit, as performance is bounded by the CPU–GPU bandwidth roof. This is the regime assumed in prior studies." (approximate — see verification note)
- **Date:** submitted 2026-05-18
- **Evidence:** READ BODY

#### C148. SVD/low-rank compression of attention projections accelerates rank collapse on pretrained models (negative original finding)
- **WHO:** the "Phase Reversal" survey/experiment authors
- **URL:** https://arxiv.org/abs/2609.06341
- **Closure status:** paper reporting an explicitly negative original finding that undercuts the standard motivation for SVD-based KV/attention projection compression.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** we report an original finding that using SVD compression of attention projections actually has the opposite effect on the rank collapse of the network: while it strongly suppresses it at initialization, it accelerates on pretrained models (for GPT-2 124M, GPT-2 Medium 355M, and Pythia-160M) with minimal risk of object aliasing artifacts appearing (verified on all compression ratios) and is consistent across four rank estimation methods. The paper also states the problem framing verbatim: "Section VI identifies an open question the reviewed literature does not address: whether the compression methods of Sections III-B through III-D compound with, or counteract, the network's natural tendency toward rank collapse. ... The two regimes produce opposite answers."
- **Date:** submitted 6 Sep 2026 (v1).
- **Evidence:** READ BODY

#### C149. Pure low-rank (Tucker/SVD) KV backbones are insufficient on the value spectrum and degrade sharply past ~4×
- **WHO:** the JoLT authors, reporting their own ablation against the pure-low-rank alternative
- **URL:** https://arxiv.org/abs/2607.12550
- **Closure status:** paper finding that the low-rank-only variant does NOT pay off (it needs a quantised residual to be near-lossless), with downstream collapse documented (LLaMA GSM8K 18.5%/6.0%/1.5% at 4×/6×/8×).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** A pure low-rank backbone cannot reach near-lossless fidelity on its own, because truncation discards real energy, and on the flat value spectrum that energy is large. The residual recovers it. Additional verbatim: "Mistral degrades gracefully, losing roughly 4% perplexity per integer ratio step from 4× on, whereas LLaMA degrades sharply between 4× and 5×. We return to this split repeatedly, because it is the main caveat on an otherwise architecture-agnostic result." and "LLaMA GSM8K falls to 18.5%, 6.0%, and 1.5% at 4×, 6×, and 8× (Appendix F), the downstream face of its perplexity degradation". Also verbatim on the competing family: "A two-dimensional factorization must commit to one axis of redundancy and ignore the others, and a fixed-bit-width quantizer cannot reach the modest, intermediate compression ratios where quality is still free, because its smallest setting already overshoots them." (approximate — see verification note)
- **Date:** submitted 14 Jul 2026 (v1), revised 23 Aug 2026 (v2).
- **Evidence:** READ BODY

#### C150. Low-rank KV compression on the target model class — keys compress, values do not; Qwen3-4B is the sensitive case
- **WHO:** the KV-CoRE authors (benchmark paper, 5 domains / 16 languages)
- **URL:** https://arxiv.org/abs/2602.05929
- **Closure status:** benchmark finding that limits low-rank KV compression and specifically flags Qwen3-4B as substantially more sensitive than LLaMA-2-7B; the authors conclude future compression must be layer-aware.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Keys are consistently more compressible than values. and "We evaluate the performance degradation of compressed models using both perplexity (PPL) and GPT-score. Figure 3 presents PPL heatmaps of Qwen3-4B and LLaMA-2-7B on the Alpaca dataset across a grid of KV-cache compression ratios ... We can see that LLaMA-2-7B remains relatively stable, showing only modest PPL increases even under aggressive compression, whereas Qwen3-4B is more sensitive, exhibiting substantial degradation." Also verbatim on why uniform compression fails: "This heterogeneity indicates that future KV-cache compression should be layer-aware: applying a uniform compression ratio risks overly degrading high-rank layers while missing opportunities for more aggressive reduction in lower-rank ones."
- **Date:** submitted 5 Feb 2026 (v1), revised 7 Feb 2026 (v2).
- **Evidence:** READ BODY

#### C151. Post-training SVD of projection weights for KV savings — achievable compression is bounded, and compressing Q and K together is catastrophic
- **WHO:** the "Thin Keys, Full Values" authors
- **URL:** https://arxiv.org/abs/2603.04427
- **Closure status:** paper concluding the retraining-free variant does NOT pay off beyond moderate ratios (K-only SVD at rank 192 degrades by 26%), while the obvious stronger form destroys attention patterns.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Table 5 reveals a striking asymmetry. Compressing both W_Q and W_K is catastrophic—at rank 192, errors in Q and K compound through the softmax, producing a cross-term Δ_Q Δ_K^⊤ that destroys attention patterns. However, compressing K alone is far more forgiving: rank 384 (d_model/2) incurs only 2.0% degradation. and "Post-training SVD of W_K alone provides a viable, retraining-free path to moderate KV cache savings. However, the achievable compression is limited: K-only SVD at d_model/4 (rank 192) degrades by 26%, whereas training from scratch with d_select=d_model/4 costs only 4.3% on WikiText-103."
- **Date:** v1 submitted 16 Feb 2026; announced March 2026; the 28 March 2026 revision is v4, not v2 (verifier correction).
- **Evidence:** READ BODY

#### C152. MLA’s shipped “MQA-absorb-only” decoding design does not pay off off-H100 (hardware coupling)
- **WHO:** the GQLA authors, analysing the deployed MLA design
- **URL:** https://arxiv.org/abs/2605.15250
- **Closure status:** paper concluding that the shipped MLA decode path is a regression relative to its own goals on non-H100 hardware — three coupled drawbacks: hardware coupling to H100, loss of head-axis tensor parallelism, and zero MTP gain on commodity inference GPUs.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** We identify three coupled hardware drawbacks of MLA's MQA-absorb-only design: hardware coupling to H100, loss of head-axis tensor parallelism, and zero MTP gain on commodity inference GPUs. and "On the NVIDIA H100, whose BF16 roofline (Williams et al., 2009) ridges around 295 FLOPs/byte, the absorbed MQA path with the canonical configuration ... and single-token decoding lands its arithmetic intensity at ≈242 FLOPs/byte, just below the ridge. This perfect H100 fit, however, is the only operating point MLA exposes."
- **Date:** v1 submitted 14 May 2026; the 21 July 2026 revision is v3, not v2 (verifier correction).
- **Evidence:** READ BODY

#### C153. FP8 KV cache and NVFP4 KV cache PRs on SM120 — measured slower, or unmerged with critical bugs
- **WHO:** reported by piweb-labs in vLLM RFC #43235 (the abandoned artifacts are #41402 and #41404, which were not retrieved directly — the quote below is the third-party statement on the page that WAS retrieved)
- **URL:** https://github.com/vllm-project/vllm/issues/43235
- **Closure status:** VERIFIER CORRECTION: the abandonment premise is REFUTED — #41402 is CLOSED with `stateReason: COMPLETED` (not “closed without resolution”) and #41404 is a pull request, not an issue. The quoted third-party text is on an OPEN RFC page (#43235). The verifier also found the evidence file’s second quote misattributed: it comes from a ROCm/gfx950 (MI355X) report about `rope_kvcache` slowdowns, whose surrounding text says the opposite (“It was not a deadlock … a factor of 8 is indistinguishable from a hang from the outside”). What survives is the measured SM120 negative result quoted below.
- **STATED REASON, VERBATIM:** "MTP + TP>1 + concurrent>1 produces deadlocks (see #41402, #41404 — both closed without resolution)." — and, the same RFC page, on the FP8/NVFP4 paths (the evidence file’s third quotation, from issue #55687, is a different hardware stack and is dropped here per the verification report): "FP8 KV Cache is slower than BF16 on SM120 because there are no native FP8 block-scaled compute units. NVFP4 KV Cache PRs (#21601 in SGLang) remain unmerged with critical bugs."
- **Date:** created 2026-05-20; stale-marked 2026-08-19.
- **Evidence:** READ BODY (verifier correction applied)

#### C154. LMCache CacheBlend — non-prefix reuse decays back to the cold baseline under sustained novel RAG traffic
- **WHO:** `shawnguyen-tensormesh` (author and closer), LMCache issue #4663 (also filed as S7-ABANDONED-2)
- **URL:** https://github.com/LMCache/LMCache/issues/4663
- **Closure status:** closed as not planned by the issue author (comment “Closing here”, verifier-corrected) after the measured decay was reported: once L1 saturates, LRU evicts the warm document chunks and the speedup decays to ~1.0–1.1× within ~1–3 batches (~30–90 queries).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Once L1 saturates, LRU evicts the **warm document chunks** (the only thing the matcher can actually reuse), so the speedup **decays from a post-warm burst down to the cold baseline within a few dozen queries.** Nothing is functionally wrong; the store/lookup **asymmetry** is the defect. Production impact: any benchmark that re-warms before each measurement reports the burst (e.g. 4–6×) and completely misses that steady-state throughput under real sustained novel traffic falls back to ~1×. Also: "**Verdict:** **CONFIRMED** - reproduced live and grounded in source." and "Sustained uniform-novel: decays to **~1.0–1.1×** (cold floor) within ~1–3 batches (~30–90 queries)."
- **Date:** created 2026-08-20, closed 2026-08-21
- **Evidence:** READ BODY

#### C155. DCPP (dynamic chunk resizing for chunked-prefill pipeline parallelism) — measured to LOSE at 512K
- **WHO:** Yan Shi, Xiaochao Wang et al. (Huawei Technologies / Shanghai Jiao Tong University), the VPP paper's own negative result
- **URL:** https://arxiv.org/abs/2608.26523
- **Closure status:** paper concludes the technique does not pay off at long context: +10.5% compute, +36.1% exposed communication, 20.47 s residual bubble, a 4.6% TTFT regression and 4.4% throughput loss at 512K (it wins only in the 64K–256K band).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Existing approaches mitigate this imbalance through dynamic chunk resizing (Dynamic CPP, DCPP), but our measurements show that this trades scheduling overhead for load balancing, which becomes unfavorable on long sequences. and, with the numbers: "At 512K, where the optimal chunk-size budget is 24K, profiler traces show that CPP requires 22 invocations, whereas DCPP requires 97, or 4.4 × 4.4\times as many." and "However, these savings are outweighed by the cost of finer-grained execution. Computation time increases by 10.5% (+26.48 s) as smaller chunks reduce operator efficiency, while the increased number of chunk boundaries raises exposed communication by 36.1% (+6.01 s). Despite dynamic balancing, 20.47 s of bubble remains, and DCPP ultimately incurs a 14.32 s (4.6%) TTFT regression over CPP." and "DCPP outperforms CPP on 64K–256K sequences, delivering consistent 1.6–3.6% improvements in both throughput and TTFT. However, its advantage progressively diminishes with sequence length and reverses at 512K, resulting in a 4.4% throughput loss and a 4.6% TTFT increase." and the generalisation: "These observations yield a critical insight: DCPP trades scheduling overhead for load-balancing gains. As sequence length grows, however, DCPP requires progressively more invocations, and the resulting fragmentation overhead eventually outweighs the benefits of bubble reduction."
- **Date:** submitted 27 Aug 2026.
- **Evidence:** READ BODY

#### C156. Dropping a KV row and observing no accuracy loss as validation of an eviction policy — shown to be an unsound inference
- **WHO:** Zefeng Cai, Zerui Cai (Independent Researchers) — "Compute Globally, Materialize Locally: The Memory Contract of Sparse Event-KV"
- **URL:** https://arxiv.org/abs/2607.23693
- **Closure status:** published negative result that undercuts the standard validation methodology of long-context KV eviction / episodic-memory designs; the paper also shows that YaRN position extension widens rather than closes the null.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Long-horizon agents increasingly reuse their KV cache as memory: a serving system keeps a subset of cached entries and drops the rest. Eviction and episodic-memory schemes therefore rest on a premise rarely tested directly, that a retained event is still informative once the observations that produced it are gone. and the conclusion, verbatim: "One consequence follows for anyone who evicts. An ablation that drops a source event and observes no accuracy loss has not shown that the source was unnecessary; it may have retained a row that already carried the answer." and "For anyone who evicts the corollary is that dropping a source event and observing no accuracy loss does not show the source was unnecessary." - Second, separate negative inside the same paper (position-extension makes the null WORSE, not better): "16 16 of 20 20 conversations exceed Qwen3's native window, where restoring range with YaRN widens the null rather than closing it (both arms re-run together on one stack: LoCoMo − .048 -.048 [ − .103 , + .009 ] [-.103,+.009] without scaling, − .121 -.121 [ − .181 , − .068 ] [-.181,-.068] with it; paired per-conversation change − .074 -.074 , p = .02 p{=}.02 )."
- **Date:** arXiv:2607.23693v1 [cs.AI] 26 Jul 2026.
- **Evidence:** READ BODY

#### C157. Final-answer accuracy as the metric for KV compression at long context — an asymmetric diagnostic; the “answer–evidence gap”
- **WHO:** Mengting Ai, Jingrui He, Yue Guo (University of Illinois Urbana-Champaign) — "Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression"
- **URL:** https://arxiv.org/abs/2608.01631
- **Closure status:** published negative result whose own framing undercuts accuracy-only evaluation of long-context KV compression, including token-eviction families (H2O/SnapKV-class).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** KV cache compression is commonly evaluated by final-answer accuracy, implicitly assuming that preserving the answer also preserves the reasoning that supports it. We test this assumption for large reasoning models and show that it can fail: under compression, correct answers and the validity of their visible supporting rationales can be preserved at different rates. and "Across tasks, token-eviction methods can preserve competitive final-answer accuracy while substantially degrading chain support or perturbation faithfulness. We call this the answer-evidence gap." and "A coverage-preserving quantization control is substantially less affected, suggesting that the failure is tied less to KV memory reduction itself than to losing access to parts of the reasoning trace." and "we show that the final–answer accuracy is an asymmetric diagnostic for compressed reasoning. While collapsed accuracy reveals compression damage, preserved accuracy can hide evidence loss, creating an illusion of competence where correct answer lacks supported evidence." - Task-dependence, verbatim (which is what makes it usable as a scoping result): "Our results show that the answer–evidence gap is strongly task-dependent: on AIME, GPQA-Diamond, and MedCalc, compressed models can maintain competitive accuracy while reasoning validity or faithfulness substantially degrades. In contrast, on retrieval tasks such as RULER QA, where answers depend more directly on retaining exact supporting evidence, compression damage is more visible through accuracy collapse."
- **Date:** arXiv:2608.01631v1 [cs.CL] 03 Aug 2026.
- **Evidence:** READ BODY

#### C158. Fixed-parameter eviction policies under workload mismatch — degrade by up to 2.7× (SAECache’s own comparison)
- **WHO:** SAECache authors (arXiv 2605.18825), self-reported negative result about the fixed-parameter class
- **URL:** https://arxiv.org/abs/2605.18825
- **Closure status:** published paper reporting that the fixed-parameter alternative degrades badly — a negative result about a whole class of KV eviction policies, stated by the authors.
- **STATED REASON, VERBATIM:** "Through extensive evaluation across heterogeneous workloads, we demonstrate that SAECache achieves 1.4x-2.7x TTFT improvement over production-style baselines, while fixed-parameter alternatives can degrade by up to 2.7x under workload mismatch -- a failure mode our adaptive approach avoids entirely."
- **Date:** submitted 12 May 2026
- **Evidence:** READ BODY

#### C159. Dynamo KVBM 0.7.0 regression — TTFT 1.27 s → 8.23 s; worked around by the reporter, never fixed
- **WHO:** reporter (issue #4774)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/4774
- **Closure status:** reported regression, worked around by the reporter (server-side flag change), not fixed. VERIFIER CORRECTION: the issue was created 2025-12-05, not 2025-12-17.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** In 0.6.1 I was getting a TTFT of 1.27 seconds. Now I'm seeing a TTFT of 8.23 seconds. … "I tried --distributed-executor-backend=mp and I saw no improvement"
- **Date:** reported 2025-12-05 (verifier-corrected; no 2025-12-17 activity exists on the page)
- **Evidence:** READ BODY

#### C160. LMCache CPU offload measured as pure overhead next to vLLM’s own prefix cache
- **WHO:** `KuntaiDu` (vLLM & LMCache committer), `sammshen` (LMCache contributor)
- **URL:** https://github.com/LMCache/LMCache/issues/1109
- **Closure status:** closed `COMPLETED` — materially this is an explained-away negative result, not a fix: only 3 tokens were loaded from LMCache because vLLM’s prefix cache absorbed the hits, so the measurement was of LMCache overhead alone.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** `KuntaiDu` (2025-07-21T17:32:30Z): "Currently CPU memory size needs to be larger than GPU KV cache size in order for LMCache to show perf improvement." `sammshen` (2025-07-21T23:34:50Z): "The key thing to notice is the logs that shows that we are only loading 3 tokens from lmcache because vllm prefix caching is receiving cache hit" … "This means we are only measuring lmcache overhead and no benefits." … "lmcache will not excel here because vllm has a ton of space for prefix caching."
- **Date:** created 2025-07-21, closed 2025-07-29
- **Evidence:** READ BODY

#### C161. LMCache’s reported 25× speedup was attributable to vLLM, not to LMCache CPU offload
- **WHO:** reporter; `FeiGSSS` (LMCache maintainer) responding
- **URL:** https://github.com/LMCache/LMCache/issues/2334
- **Closure status:** `NOT_PLANNED`, closed by the AUTOMATED STALE BOT. The negative result is in the issue body and the maintainer reply, not in the closure.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Body: "Performance comparison between vLLM standalone and vLLM with LMCache CPU Offload shows minimal difference in measured TTFT improvements, despite the same 25x speedup being reported for both configurations." … "The 25x improvement seen in both cases appears to be coming from vLLM's internal KV cache mechanism rather than LMCache's CPU offload". `FeiGSSS`: "vLLM has its own CPU KVCache offload, which is slightly slower than LMCache (as reported in the paper). I think your results is expectable."
- **Date:** created 2025-12-31, closed 2026-04-06
- **Evidence:** READ BODY

#### C162. LMCache token-dropping / compression accuracy is within run-to-run noise
- **WHO:** LMCache (PR #4500 body)
- **URL:** https://github.com/LMCache/LMCache/pull/4500
- **Closure status:** the PR merged, but its own documented measurement (accuracy carries ~1–2/30 run-to-run noise from decode non-determinism) undercuts the accuracy claim.
- **STATED REASON, VERBATIM:** "Accuracy on this workload carries ~1–2/30 run-to-run noise from decode non-determinism"
- **Date:** 2026
- **Evidence:** READ BODY

#### C163. vLLM FP8 KV-cache — the measured failure modes before the fixes, and the residual “when to avoid FP8” list
- **WHO:** vLLM (AWS + Red Hat AI authors of the blog)
- **URL:** https://vllm.ai/blog/2026-04-22-fp8-kvcache
- **Closure status:** published negative result, partially remediated; residual limits are documented in the blog’s own “When to Avoid FP8 KV-Cache” section (short contexts, head_dim 256 prefill, 91%→13% NIAH at 128k on Hopper before the fix, gpt-oss-20b break-even beyond 700k tokens).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** On Hopper GPUs, the FP8 Flash Attention 3 kernel suffered from accumulation precision loss at long contexts. On a 128k needle-in-a-haystack task, FP8 accuracy dropped from 91% (BF16 baseline) to just 13% — a regression traced to imprecise FP32 accumulation in the Tensor Cores. "The FP8 ITL slope for models with sliding-window attention layers (e.g., gpt-oss-20b) was nearly identical to BF16 (96% of BF16 slope), meaning users gained almost no decoding speedup despite halving memory. The break-even point exceeded 700k tokens — well beyond most practical context lengths." Residual: "for head dimensions larger than 128, the prefill performance remains behind BF16." And the explicit avoidance list: "FP8 KV-cache quantization is not always the right choice. Consider staying with BF16 if: Your contexts are short (< ~7k tokens): FP8 has a small constant overhead (the intercept gap), so at short contexts BF16 may be slightly faster for ITL. Your model uses head_dim = 256 and prefill latency matters: The two-level accumulation overhead increases TTFT by ~1.6x at long contexts." Quantified in their Table 3: `gpt-oss-20b before (v0.10.2) FP8 741,565 96%` break-even tokens and slope. (approximate — see verification note)
- **Date:** published 2026-04-22
- **Evidence:** READ BODY

#### C164. Seven curated KV-compression mechanisms all rejected under a pre-registered protocol at matched mean cache
- **WHO:** arXiv 2605.14292
- **URL:** https://arxiv.org/abs/2605.14292
- **Closure status:** published negative result (pre-registered protocol, held-out confirmation) on long-form mathematical reasoning with two distilled-reasoning models at budgets b ∈ {64, 128}.
- **STATED REASON, VERBATIM:** "We study seven mechanisms across these five families under matched mean cache on long-form mathematical reasoning (MATH-500~\cite{hendrycks2021math}) with two distilled-reasoning models (Qwen-7B and Llama-8B variants of DeepSeek-R1-Distill~\cite{deepseek2025r1}) at budgets $b \in \{64, 128\}$. All seven were rejected."
- **Date:** submitted 2026-05-14, revised 2026-05-17
- **Evidence:** READ BODY

#### C165. Mixed-precision KV bit allocation can be worse than uniform quantisation — “distortion model mismatch”
- **WHO:** arXiv 2605.06675 (RateQuant)
- **URL:** https://arxiv.org/abs/2605.06675
- **Closure status:** published negative result about the popular technique; the paper’s own contribution is the fix. Applying one quantiser’s distortion model to another inverts the allocation order.
- **STATED REASON, VERBATIM:** "We show, however, that such mixed-precision allocation has a hidden pitfall: each quantizer follows a different distortion curve D(b)=alpha*beta^{-b}, and the decay rate beta varies from 3.6 to 5.3 across quantizer designs. Applying one quantizer's distortion model to another inverts the allocation order and makes performance worse than uniform quantization. We call this failure mode distortion model mismatch"
- **Date:** 2026-05
- **Evidence:** READ BODY

#### C166. Per-layer FP8 KV-quantisation error attribution is non-diagnostic — errors survive on cross-layer cancellation
- **WHO:** arXiv 2607.28699 (WitCert)
- **URL:** https://arxiv.org/abs/2607.28699
- **Closure status:** published negative result: in a 28-layer sweep no single layer’s pollution alone loses anything (0/28).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** aggressive schemes survive on cross-layer error cancellation, not per-step fidelity -- in a 28-layer sweep, no single layer's pollution alone loses anything (0/28) … "raw-cast fp8 from 22.8 back to 79.7 on hard RULER tasks"
- **Date:** 2026-07
- **Evidence:** READ BODY

#### C167. Per-layer eviction routing has null results on short inputs (MATH-500, LongBench TREC)
- **WHO:** arXiv 2604.17695 (MoE-nD)
- **URL:** https://arxiv.org/abs/2604.17695
- **Closure status:** published negative results, self-labelled “null results”: the solver picks keep=1.0 on most layers for short inputs, cleanly characterising when per-layer routing has headroom to help.
- **STATED REASON, VERBATIM:** "Two null results -- MATH-500 and LongBench's TREC -- share a principled cause (short inputs, solver picks keep=1.0 on most layers), cleanly characterizing when per-layer eviction routing has headroom to help."
- **Date:** 2026-04
- **Evidence:** READ BODY

#### C168. Sliding window is order-optimal among suffix-only cache policies — a negative result bounding a whole policy class
- **WHO:** arXiv 2605.25085
- **URL:** https://arxiv.org/abs/2605.25085
- **Closure status:** published negative result bounding the suffix-only policy class; whether recurrent or propagating cache summaries can beat the scaling is left open by the authors.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** our main result characterizes the per-token memory requirement of suffix-only cache policies: a sliding-window scheme attains distortion eps with window w = O(eps^{-1/alpha}), and -- under an additional two-sided Bayes-risk condition -- a converse shows w = Omega(eps^{-1/alpha}) is necessary within this policy class … "Whether recurrent or propagating cache summaries can beat this scaling is left open." (approximate — see verification note)
- **Date:** 2026-05
- **Evidence:** READ BODY

#### C169. The standard chunked-mean fast-weight write is silently wrong — C²−C spurious cross-token outer products per chunk
- **WHO:** arXiv 2605.22884 (Tensor Cache)
- **URL:** https://arxiv.org/abs/2605.22884
- **Closure status:** published negative result about a widely used approximation.
- **STATED REASON, VERBATIM:** "the common chunked-mean training shortcut A<-lambda A+eta(kbar ⊗ vbar) silently introduces C^2-C spurious cross-token outer products per chunk" (approximate — see verification note)
- **Date:** 2026-05
- **Evidence:** READ BODY

#### C170. Prefetching cannot hide CPU-offload transfers — PCIe-bandwidth bound, not prediction-quality bound (WiSP)
- **WHO:** arXiv 2606.21868 (WiSP)
- **URL:** https://arxiv.org/abs/2606.21868
- **Closure status:** published negative result explicitly about a “natural next step”: speculative transfers compete with demand transfers instead of hiding them. (Scope caveat from the evidence: the mechanism concerns experts, not KV.)
- **STATED REASON, VERBATIM:** "A natural next step is to predict future experts and prefetch them. We find that this does not help in single-stream decode: the bottleneck is PCIe bandwidth, not prediction quality, so speculative transfers compete with demand transfers instead of hiding them."
- **Date:** submitted 2026-06-20, revised 2026-08-30
- **Evidence:** READ BODY

#### C171. Three attempts to beat LRU for cross-request KV prefix caching all failed on 68,266 real Claude Code requests plus 23,608 Mooncake requests
- **WHO:** `gauravapiscean`, independent reproduction study
- **URL:** https://github.com/gauravapiscean/agentic-kv-cache | https://raw.githubusercontent.com/gauravapiscean/agentic-kv-cache/main/README.md
- **Closure status:** published null result on real traces; the leaf restriction both major engines implement bought 0.02pp on this workload. The author also documents a harness trap (an offline Belady oracle losing to LRU).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** I replayed **68,266 requests from 393 real Claude Code sessions** and **23,608 Mooncake requests** through a prefix-cache simulator, tried to beat the production baseline three different ways, and failed. Section heading, verbatim: "## 4. Three ways to beat LRU, three failures". Result verbatim: "Monotone negative. Every component made it worse, and the one I was most confident in — coherent eviction — was the worst." And: "**LRU-leaf is a stronger baseline than the literature treats it as.** I couldn't beat it with three independent mechanisms on real traces." A directly shipped-engine-relevant finding, verbatim: "**Incidental finding:** flat block LRU and radix-leaf-restricted LRU differ by **0.02pp** on this workload. The leaf restriction both major engines implement buys essentially nothing here." Also documents a harness trap: "In my first run, Belady — an *offline oracle* — lost to LRU. That's not a result, that's a broken harness, and it's worth publishing because I expect it to be common."
- **Date:** 2026 (traces dated 062126; the study is undated in the README but references 2026 datasets)
- **Evidence:** READ BODY

#### C172. vLLM’s own docs state that disaggregated prefill DOES NOT improve throughput
- **WHO:** vLLM project (the docs page's own warning)
- **URL:** https://docs.vllm.ai/en/latest/features/disagg_prefill.html
- **Closure status:** shipped feature with an explicit negative scope statement in the project’s own documentation — the technique is published and supported, but the headline benefit is disclaimed.
- **STATED REASON, VERBATIM:** "!!! note Disaggregated prefill DOES NOT improve throughput."
- **Date:** doc current in retrieved tree (vLLM main)
- **Evidence:** READ BODY

#### C173. “Beyond the Buzz: A Pragmatic Take on Inference Disaggregation” — disaggregation is not a universal solution
- **WHO:** arXiv 2506.05508 (authors' own conclusion)
- **URL:** https://arxiv.org/abs/2506.05508 | https://arxiv.org/html/2506.05508v1
- **Closure status:** published negative/limiting result — the paper’s headline finding undercuts multi-node disaggregation for a large part of the design space.
- **STATED REASON, VERBATIM:** "As demonstrated in Figure 1, disaggregation is not a universal solution. In the following sections, we examine the performance benefits of disaggregated inference serving across a broad design space."
- **Date:** arXiv 2506.05508, retrieved 2026-09
- **Evidence:** READ BODY

#### C174. Prefix caching for small reasoning models — measured to make 7B serving WORSE
- **WHO:** Qi Li et al. (HKUST-GZ), paper "Reasoning Language Model Inference Serving Unveiled: An Empirical Study"
- **URL:** https://arxiv.org/abs/2510.18672
- **Closure status:** published empirical study whose own experiments conclude the technique does not pay off in the 7B regime (increased latency).
- **STATED REASON, VERBATIM:** "However, for 7B models, prefix caching negatively impacts efficiency, leading to increased latency."
- **Date:** submitted 2025-10-21
- **Evidence:** READ BODY

#### C175. Immediate online KV compaction for agents hurts performance; delaying compaction recovers much of the gap
- **WHO:** authors of "Practical Online KV Cache Compaction for LLM Agents: An Empirical Study"
- **URL:** https://arxiv.org/abs/2608.00902
- **Closure status:** published empirical study whose headline result is that the naive (immediate) form of the technique hurts — the negative finding is the paper’s main contribution.
- **STATED REASON, VERBATIM:** "Experiments on BrowseComp-Plus and WideSearch show that immediate compaction often hurts performance, whereas delaying compaction to use the agent's future queries recovers much of the gap. Moreover, TE is often more robust than AM under imperfect proxies."
- **Date:** submitted 2026-08-02
- **Evidence:** READ BODY

#### C176. Learned importance scoring for KV compression does not beat random selection
- **WHO:** Brady Steele — “On the Limits of Learned Importance Scoring for KV Cache Compression” (the lead-author name was not independently confirmed by the verifier; the title and abstract quotes were)
- **URL:** https://arxiv.org/abs/2601.14279
- **Closure status:** published negative result — the paper’s own headline is that the learned scorer fails across 5 seeds, 4 retention levels and 3 tasks.
- **STATED REASON, VERBATIM:** "Despite architectural sophistication (multi-horizon lookahead, cross-attention), SIP does not outperform simple baselines, including random selection, across 5 seeds, 4 retention levels, and 3 tasks."
- **Date:** submitted 2026-01-13
- **Evidence:** READ BODY

#### C177. Nexus — the off-anchor KV splice boundary, and a drift gate that does not predict drift
- **WHO:** Mustafa Arslan, "Nexus: Depth-Adaptive KV-Cache Splicing and Retrieval-Decoupled Tool Routing for Agentic LLMs on Unified Memory"
- **URL:** https://arxiv.org/abs/2608.20397
- **Closure status:** published paper containing two explicit self-declared negative results that bound the technique; the KV-splice lever is the bounded part.
- **STATED REASON, VERBATIM:** "Two negative results bound the design: the off-anchor RoPE fidelity boundary, and the failure of a reference-free drift gate to predict drift (Spearman rho = 0.193)."
- **Date:** submitted 2026-07-01
- **Evidence:** READ BODY

#### C178. Permanent eviction for reasoning chain-of-thought is catastrophic (accuracy 0–2.5%), and the reproduced SOTA eviction method lands at 0–32%
- **WHO:** Aojie Yuan, Tianqi Shen, Dajun Zhang, "Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy for LLM Reasoning"
- **URL:** https://arxiv.org/abs/2605.09490
- **Closure status:** published negative result about the dominant approach, plus a failed reproduction of the then-SOTA eviction method.
- **STATED REASON, VERBATIM:** "The dominant response -- permanently evicting low-importance tokens -- is catastrophic for reasoning: accuracy collapses to 0-2.5% when half the cache is removed."
- **Date:** submitted 2026-05-10
- **Evidence:** READ BODY

#### C179. Query-aware KV-compression evaluation — the pay-off does not survive a matched-budget audit; SnapKV loses to a trivial baseline
- **WHO:** authors of "How Query Visibility Changes KV-Cache Compression Rankings: A Matched-Budget Audit" (arXiv 2607.11942)
- **URL:** https://arxiv.org/abs/2607.11942
- **Closure status:** published negative result: under the query-agnostic protocol only KeyDiff consistently beats a best-of-3 trivial baseline, and the most widely deployed method (SnapKV) loses to “keep the start and the recent window” on average (−0.066).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** under the agnostic protocol, of the five audited methods that share a common attention backend, only KeyDiff beats a best-of-3 trivial baseline consistently (31 of 36 cells), and the most widely deployed method, SnapKV, loses to keep the start and the recent window" on average (-0.066)." and "The per-method drop between the two protocols is ordered consistently with how visible the question is to each method's scoring signal, legible in its source code: from Delta=+0.198 for SnapKV (the question sits inside its 64-token observation window) down to Delta=+0.011 for KeyDiff (its score contains no query term at all)."
- **Date:** submitted 2026-07-11.
- **Evidence:** READ BODY

### C.6 Papers whose own limitations undercut their technique

#### C180. AVMP — the paper frames its own production impact as an unmeasured hypothesis
- **WHO:** AVMP authors, arXiv 2605.22416 (the quote is in §6.3 “Production hypothesis”, not in a limitations section — verifier correction to the evidence file’s label)
- **URL:** https://arxiv.org/abs/2605.22416
- **Closure status:** paper whose own text undercuts the technique — its central production claim is explicitly labelled a hypothesis to be tested, not a measured outcome.
- **STATED REASON, VERBATIM:** "We frame the production impact of AVMP as a hypothesis to be tested, not a measured outcome."
- **Date:** submitted 21 May 2026
- **Evidence:** READ BODY

#### C181. QAQ needs attention scores that do not exist at quantisation time — worked around with an assumption rather than solved
- **WHO:** Shichen Dong, Wen Cheng, Jiayu Qin, Wei Wang — the QAQ paper’s own text (arXiv:2403.04643). Verifier correction: the evidence file’s shorthand “Cheng et al.” follows the submitting author, not the first author.
- **URL:** https://arxiv.org/html/2403.04643v2
- **Closure status:** self-declared limitation inside the paper: the method’s required attention scores are not available at quantisation time, so it invokes the persistence-of-importance assumption. No 2025–2026 revision and no implementation in either engine tree.
- **STATED REASON, VERBATIM:** "Our quantization method necessitates the attention scores that quantify the degree to which future tokens attend to preceding tokens. However, these attention scores are not available at the time of quantization. To address this limitation, we invoke the persistence of importance theory, which posits that the attention scores ( i.e. , importance) of each token remain relatively constant ( i.e. , persistence) throughout the process of token generation."
- **Date:** v1 Mar 7, 2024; v2 Apr 12, 2024. No 2025-2026 revision, and QAQ has no implementation in either engine tree (see OPEN-12).
- **Evidence:** READ BODY

#### C182. Monolithic single-cartridge KV for a whole document collection — collapses to near-chance when composed, and the replacement does not cover mid-conversation insertion
- **WHO:** Eyuboglu et al. (2025), the original Cartridges design — falsified by the CAS paper (arXiv:2606.04557), and by CAS's own limitations section
- **URL:** https://arxiv.org/abs/2606.04557
- **Closure status:** published paper concluding the earlier “cartridges” technique does not pay off at collection scale, while its own limitations section concedes that mid-conversation cartridge loading would invalidate the previously computed KV, that compression tolerance varies by document type (100× TechQA vs ≤2× FinQA), and that retrieval quality is bounded by raw-text matching.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** A critical limitation of this approach is that cartridges are monolithic and non-compositional: encoding an entire collection into a single KV block does not scale, and naively mixing cartridges trained in isolation collapses performance to near chance. and "However, their approach relies on a single monolithic cartridge per collection, which does not scale to real-world document collections: loading all documents into one cache bloats the prefix with irrelevant tokens, while information-dense content (e.g., financial tables, clinical notes, technical specifications) cannot be highly compressed without loss." - Inverted-negative from the CAS paper's OWN limitations section (its technique does not cover mid-conversation cache insertion at all): "In multi-turn scenarios, however, a cartridge may need to be loaded mid-conversation—for example, when a user references a new document after several turns of dialogue. Prepending the cartridge at that point would invalidate the previously computed KV entries, requiring a full recomputation of the conversation history." And the compression-tolerance bound: "Compression tolerance varies with document complexity ( 100 × 100\times for TechQA, ≤ 2 × {\leq}2\times for FinQA), reaching within 5% of uncompressed baselines. Finally, we show that cartridges can be combined with retrieval, matching RAG performance while using 3–4 × \times fewer tokens, though for information-dense tasks such as FinQA, text RAG maintains an edge." And on retrieval quality: "This is a pragmatic choice, but it means retrieval quality is bounded by how well the query matches the raw document text rather than the compressed KV representation."
- **Date:** submitted 3 Jun 2026.
- **Evidence:** READ BODY

#### C183. Region-aware eviction decays — MemDecay’s own abstract names attention-score normalisation as the main limitation
- **WHO:** MemDecay authors (arXiv 2607.10582)
- **URL:** https://arxiv.org/abs/2607.10582
- **Closure status:** published paper whose own abstract undercuts part of the technique: accumulated-attention retention performs better on unpinned content, and the main limitation is identified as attention-score normalisation.
- **STATED REASON, VERBATIM:** "Accumulated-attention retention performs better on unpinned content, however, and ablations identify attention-score normalization as the main limitation of the current formulation. These results establish semantic prompt structure as a robust signal for KV-cache management while clarifying how it should be combined with attention-based importance."
- **Date:** submitted 12 Jul 2026
- **Evidence:** READ BODY

#### C184. CacheTTL’s TTL cost model assumes the tool-call distribution is stable — the paper’s own Appendix D concedes it may produce suboptimal TTLs under distribution shift
- **WHO:** Continuum/CacheTTL authors (arXiv 2511.02230), "Appendix D Limitations and Future Work"
- **URL:** https://arxiv.org/html/2511.02230v4
- **Closure status:** published paper’s own limitations section scoping the TTL derivation — a scoped failure mode, not a retraction.
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** Sensitivity of the TTL Cost Model: CacheTTL relies on a cost–benefit model that combines empirical tool-call CDFs, memory-usage estimates, and a memoryfulness" factor to derive optimal TTL values. While this design is principled, it assumes that tool-call distributions and workload characteristics are sufficiently stable for historical samples to be predictive. In highly volatile or adversarial workloads, such as agents whose tool latencies abruptly shift due to back-end contention or external API variability, the model may produce suboptimal TTLs, temporarily degrading scheduling efficiency. ... We leave handling sudden distribution shifts in agent as future work."
- **Date:** v4 HTML retrieved; v1 submitted 4 Nov 2025
- **Evidence:** READ BODY

#### C185. Star Attention — distributed attention is SLOWER than ordinary single-machine inference below 32K, by the paper’s own Appendix/Table 6
- **WHO:** NVIDIA (Acharya, Jia, Ginsburg), the paper's own Appendix C.1 / Table 6
- **URL:** https://arxiv.org/abs/2411.17116 | https://arxiv.org/html/2411.17116v1
- **Closure status:** published result that undercuts the technique’s own applicability range: for sequence lengths below 32K vanilla inference is faster, and the technique also permits accuracy loss by design (1.6–4.9% Multi-NIAH, 0.9–6.8% QA declines).
- **STATED REASON (composite: quoted fragments interleaved with retrieved framing text, not one continuous quotation):** For sequence lengths below 32K, vanilla inference is faster than the distributed attention mechanisms, primarily due to the GPU communication overhead incurred in the distributed setups. However, in long context scenarios i.e. on sequence lengths exceeding 32K tokens, Star Attention begins to demonstrate clear performance advantages. - The paper's own Table 6 numbers make the loss concrete for Llama3.1-8B-Instruct: `16K → Vanilla 7s, Ring 10s, Star 9s`; `32K → Vanilla 10s, Ring 12s, Star 10s`; `64K → Vanilla 18s, Ring 22s, Star 12s`; `128K → Vanilla OOM, Ring 53s, Star 20s`. - The technique also permits accuracy loss by design, verbatim: "The choice of block size is dependent on the user on how much accuracy can be traded for improved speed." and the reported quality gap, verbatim: "Notably, Star Attention achieves scores nearly identical to global attention in Single-NIAH tasks. However, in more complex tasks such as Multi-NIAH and QA, it shows slight decline in performance, with reductions ranging from 1.6% to 4.9% in Multi-NIAH and 0.9% to 6.8% in QA tasks."
- **Date:** arXiv 2411.17116v1, 2024-11-26
- **Evidence:** READ BODY


---

## D. What is ruled out BY HARDWARE for a single-GPU researcher

Screen applied: exactly one GPU — RTX PRO 6000 Blackwell 96 GB (sm120, driver 580) — 208 CPU cores, ample local disk, one node, no NVLink bridge, no InfiniBand, no cluster. Rows are de-duplicated: the same artifact or the same question appears once, and a row that merges several segments carries every URL those segments cited. Placement rule used below: D.1 = the binding constraint is the number of GPUs on one node; D.2 = the binding constraint is per-device or aggregate VRAM; D.3 = the binding constraint is a node boundary or an inter-node fabric (InfiniBand/RoCE/RDMA/MNNVL); D.4 = the binding constraint is a datacenter storage fabric or a distributed KV-store service; D.5 = the technique is published on multi-GPU hardware but the source states or ships a single-GPU mode or configuration. Corrections recorded in the adversarial verification reports override the raw evidence files (notably: CacheGen's testbed is a four-GPU A40 server, not one A40, which flips it into D.1). Line numbers taken from the local `/tmp/kvsrc` mirrors are not reproduced here — the verifier established that the vLLM mirror is an older snapshot (~0.11.2) whose line numbers have drifted on live main.

### D.1 Requires >1 GPU (tensor/pipeline/context parallelism, multi-GPU within one node)

| # | Technique | Hardware requirement | Why it is out of reach | URL | Evidence |
|---|---|---|---|---|---|
| D1 | eLLM elastic memory management (virtual tensor abstraction + ballooning) | 8× NVIDIA A100 80GB on one server, connected via NVLink, with 1 TB host RAM (plus a second 2× L40S 48GB testbed) | GPU count — the mechanism is a host-memory balloon across 8 NVLink-connected GPUs | [2506.15155](https://arxiv.org/abs/2506.15155) | READ BODY |
| D2 | RedKnot long-context serving with SegPagedAttention | One server with 8× NVIDIA H800 80GB, two Xeon Platinum 8468V, 2 TB DDR; four RDMA RoCE v2 NICs measuring ~200 Gbps one-way for the PD experiments | GPU count — all experiments run on a single 8-GPU node | [2606.06256](https://arxiv.org/abs/2606.06256) | READ BODY |
| D3 | vLLM hybrid block-size / hash-block-size assertion crash (DeepSeek-V4.1-Flash) | 8× H100 80GB (HBM3), CUDA 13, with `--tensor-parallel-size 8` and `--enable-expert-parallel` | GPU count — the reproduction as filed needs an 8-GPU TP8/EP8 deployment (the defect itself is an allocator/hashing bug, not intrinsically multi-GPU) | [vllm#56396](https://github.com/vllm-project/vllm/issues/56396) | READ BODY |
| D4 | GLM-5.3-Flash kpool 32-page split under hybrid block_size 1152 | 8× AMD MI325X (ROCm, gfx942) | GPU count — filed on an 8-GPU ROCm node | [vllm#55280](https://github.com/vllm-project/vllm/issues/55280) | READ BODY |
| D5 | vLLM decode context parallelism (`-dcp`) — incl. its structural precondition | Tensor-parallel size strictly greater than the model's total KV heads (≥9 GPUs for an 8-KV-head GQA shape; in practice 16); DCP never adds ranks, it subdivides the existing TP group, so `tp_size > 1` is a precondition | GPU count — with `tp_size = 1` the only admissible DCP size is 1 (off) | [vllm/config/model.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py) · [vllm#23734](https://github.com/vllm-project/vllm/pull/23734) · [context_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md) | READ BODY |
| D6 | vLLM prefill context parallelism (`--prefill-context-parallel-size`) | ≥2 ranks even at TP=1 — the merged feature's own Test Plan runs `--tensor-parallel-size 1 --prefill-context-parallel-size 2` | GPU count — N ranks are created for N>1 regardless of TP | [vllm#28988](https://github.com/vllm-project/vllm/pull/28988) · [vllm#28718](https://github.com/vllm-project/vllm/pull/28718) | READ BODY |
| D7 | vLLM general context parallelism (test bed) and ring-attention / partial-query–partial-KV prefill CP | Test defined as "single-node (4 GPUs)" or "2 node with 2 GPUs each"; the ring path splits a request into N chunks for N GPUs | GPU count — ≥4 GPUs, or 2 nodes × 2 GPUs | [test_context_parallel.py](https://github.com/vllm-project/vllm/blob/main/tests/distributed/test_context_parallel.py) · [context_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md) | READ BODY |
| D8 | SGLang decode context parallelism (`--dcp-size`) | A tensor-parallel group larger than the DCP size (`attn_tp_size % dcp_size == 0`); the low-latency `fi_a2a` backend additionally requires Blackwell and the whole DCP group inside one MNNVL domain; validated only on 4× GB300 at TP4/EP4 | GPU count — needs an existing TP group of ≥2 ranks (plus an MNNVL domain for the fast backend) | [SGLang DCP](https://docs.sglang.ai/advanced_features/dcp.html) | READ BODY |
| D9 | SGLang prefill context parallelism + LayerSplit | 8 ranks minimum (`--attn-cp-size 8`); the LayerSplit composition also needs PD disaggregation with the Mooncake transfer backend | GPU count — ≥8 ranks | [SGLang docs](https://docs.sglang.ai/) | READ BODY |
| D10 | vLLM RFC — independent context parallelism for the DSA indexer | ≥2 GPUs in the CP group; the RFC's operator benchmarks are on 2× NVIDIA H20 | GPU count — CP2 is defined over two ranks | [vllm#53684](https://github.com/vllm-project/vllm/issues/53684) | READ BODY |
| D11 | Striped Attention (ring attention for causal transformers) | One server with 8× NVIDIA A100 80GB connected with NVLink, or a TPU pod slice (4× v3 / 16× v4 chips) | GPU count — 8 GPUs on one NVLink server; no single-device form | [2311.09431](https://arxiv.org/abs/2311.09431) | READ BODY |
| D12 | 3DLS — 3D logic-stacked chiplet architecture for layer-wise KV transfer isolation | A logic-on-logic 3D-stacked chiplet package with vertical die-to-die interconnects, TP up to 16 and a 512 GB/s lateral D2D fabric; evaluated with a trace-driven in-house simulator for OPT-175B | Silicon — requires stacked-die hardware with TP=16; no deployable single-GPU form | [2607.01617](https://arxiv.org/abs/2607.01617) | READ BODY |
| D13 | Mamba1 NIXL P/D accuracy validation | 8× NVIDIA H100 box plus a 52B model for the full P_TP × D_TP sweep (`ai21labs/AI21-Jamba2-Mini`) | GPU count — 8 GPUs | [vllm#45019](https://github.com/vllm-project/vllm/pull/45019) | READ BODY |
| D14 | DP8 vs TP8 KV-capacity study for single-KV-head MLA | One 8× NVIDIA B200 node serving DeepSeek-V4-Flash-0731 (1M context, FP8 KV) | GPU count — the KV-capacity result is only defined under 8-way parallelism | [vllm#51454](https://github.com/vllm-project/vllm/issues/51454) | READ BODY |
| D15 | SGLang HiCache load-back profiling (overlap measurement and prefill stall behind the H2D burst) | 8× H100, TP=8, with multi-GiB load bursts and `io_backend=direct` | GPU count — 8-GPU TP8 deployment | [sglang#38470](https://github.com/sgl-project/sglang/issues/38470) · [sglang#38448](https://github.com/sgl-project/sglang/issues/38448) | READ BODY |
| D16 | SGLang HiCache prefetch finalization lag | 4P1D disaggregated (prefill/decode-disaggregated, multi-instance) deployment with Mooncake storage backend | GPU count — 4 prefill + 1 decode instances | [sglang#32724](https://github.com/sgl-project/sglang/issues/32724) | READ BODY |
| D17 | Dynamo KVBM disk→device descriptor fragmentation | Tensor-parallel 8 for the target model; the fix's benefit statement is expressed for a 48-layer TP8 layout | GPU count — TP8 is the stated target | [dynamo#12750](https://github.com/ai-dynamo/dynamo/issues/12750) | READ BODY |
| D18 | Tensor-parallelism vs KV-compression trade-off study ("More GPUs or a Smaller Cache?") | Not retrieved — the item was seen in a search-result list only; the title frames a multi-GPU versus compression comparison | Unverified — no hardware requirement in the evidence (do not treat the hardware claim as evidenced) | [2608.23962](https://arxiv.org/abs/2608.23962) | TITLE ONLY |
| D19 | TurboQuant throughput on the target model (Qwen3-4B) | 4× RTX PRO 6000 Blackwell with cudagraphs+compile (the accuracy table in the same PR is single-model, the throughput table is 4-GPU) | GPU count — the published throughput table is 4-GPU | [vllm#38479](https://github.com/vllm-project/vllm/pull/38479) | READ BODY |
| D20 | vLLM's official TurboQuant accuracy/throughput conclusions | 2× H100 (Qwen3-30B-A3B) and 4× H100 (Llama-3.3-70B) | GPU count and aggregate VRAM — published on >1 GPU and >96 GB total | [vLLM blog](https://vllm.ai/blog/2026-05-11-turboquant) | READ BODY |
| D21 | TurboQuant × speculative-decoding degenerate-output repro | 2× RTX 3090 (Ubuntu 22.04, driver 595.58.03, CUDA 12.9, vLLM image pinned to a digest) | GPU count — 2 GPUs; the reporter adds that the decisive MTP probe "won't fit on our hardware" | [vllm#40831](https://github.com/vllm-project/vllm/issues/40831) | READ BODY |
| D22 | KVQuant's headline 10-million-token context result | An 8-GPU serving system (the 1M-token result is a single A100-80GB and *is* single-GPU) | GPU count — only the 10M-token headline needs 8 GPUs | [2401.18079v6](https://arxiv.org/html/2401.18079v6) | READ BODY |
| D23 | The only found quantized-KV × prefix-caching bug report | 8× NVIDIA B300 (sm_103) with `-dp 4 --enable-expert-parallel`, a 397B NVFP4 checkpoint and `--max-model-len 65536` | GPU count — 8 B300 GPUs (and a model far beyond one 96 GB card) | [vllm#47349](https://github.com/vllm-project/vllm/issues/47349) | READ BODY |
| D24 | KVarN's throughput-superiority evidence at scale | TP=2 (2 GPUs) for the headline "+17.6% throughput" result on GLM-4.5-Air-FP8 (~106B) | GPU count — the headline result is TP=2 | [vllm#46613](https://github.com/vllm-project/vllm/issues/46613) | READ BODY |
| D25 | RedKnot-MLA (MLA offline/online reuse for DeepSeek-V4 long-context serving) | One local DeepSeek-V4-Flash checkpoint on eight NVIDIA H200 GPUs with TP=8 and DP=CP=PP=1 | GPU count — one full 8-GPU node | [2609.07008](https://arxiv.org/abs/2609.07008) | READ BODY |
| D26 | KV-Pipe (cross-layer KV sharing to rebalance pipeline stages) | 8× Ascend 910B NPUs with PP=2/4/8; the PP-focused evaluation is repeated on one node of 8× NVIDIA V100 | GPU count — 8 accelerators and pipeline-parallel degree >1 | [2608.15943](https://arxiv.org/abs/2608.15943) | READ BODY |
| D27 | TPLA (Tensor Parallel Latent Attention) | ≥2 GPUs; TTFT reported on two H800 GPUs for MoE-removed DeepSeek-V3-0324 and Kimi-K2-Base | GPU count — the method distributes the latent across devices and all-reduces | [2508.15881](https://arxiv.org/abs/2508.15881) | READ BODY |
| D28 | SM120 MLA-vs-NSA production characterization on RTX PRO 6000 Blackwell | 8× RTX PRO 6000 Blackwell 96 GB on PCIe Gen5 with TP>1, MTP and concurrent serving | GPU count — 8 GPUs, and "MTP + TP>1 + concurrent>1" is the reported deadlock trigger | [vllm#43235](https://github.com/vllm-project/vllm/issues/43235) | READ BODY |
| D29 | vLLM native MLA TP-replica dedup in native offloading | TP > 1 — the dedup only engages when the MLA latent KV is replicated across tensor-parallel ranks | GPU count — the feature does not exist at TP=1 | [vllm#47929](https://github.com/vllm-project/vllm/issues/47929) | READ BODY |
| D30 | SGLang HiCache host-memory sizing guard (`host_memory_budget_bytes`) | Multiple co-located ranks on one host (demonstrated at `--tp 8`); the guard divides free RAM by `ranks_per_host()` | GPU count — the double-charging race needs >1 rank on the host and cannot occur with a single rank | [sglang#38156](https://github.com/sgl-project/sglang/issues/38156) | READ BODY |
| D31 | CacheGen KV cache compression + streaming | "an NVIDIA A40 GPU server with four GPUs" — a 4× A40 evaluation (verifier correction: the earlier "single A40" reading deleted "with four GPUs" from inside the quotation and flipped this row's hardware classification) | GPU count — 4 GPUs, not one | [2310.07240](https://arxiv.org/abs/2310.07240) | TITLE + ABSTRACT ONLY |
| D32 | SGLang HiSparse | PD disaggregation mode, decode instance only — i.e. ≥2 GPUs (a prefill instance plus a decode instance) | GPU count — the KV-memory-saving path exists only under PD disaggregation | [SGLang docs](https://docs.sglang.ai/) | READ BODY |
| D33 | vLLM cross-instance KV lifetime over a P/D pair (PD-disaggregated multi-tier TTL; NIXL lease renewal) | At least two vLLM instances (one prefill, one decode), each running `OffloadingConnector` with `TieringOffloadingSpec` (CPU primary + p2p secondary tier); measurements on an idle H200 | GPU count — TTL/lease semantics only exist between a P and a D engine | [vllm#53128](https://github.com/vllm-project/vllm/issues/53128) · [vllm#41383](https://github.com/vllm-project/vllm/pull/41383) | READ BODY |
| D34 | Llumnix live KV migration for SLA differentiation, and the 2026 multi-tier SLA extension built on it | Multiple model instances (multi-GPU/multi-node); the extension is evaluated by simulation (Vidur) rather than on one GPU | GPU count — rescheduling and live migration are defined across ≥2 instances | [2406.03243](https://arxiv.org/abs/2406.03243) · [2608.16336](https://arxiv.org/abs/2608.16336) | READ BODY |
| D35 | CacheScout — agent-aware KV-cache runtime for multi-agent serving (`Learning Agent Execution for KV-Cache Management in Agentic Serving`) | A server with eight NVIDIA RTX PRO 6000 Blackwell GPUs, 96 GB each; the large-model arm runs Qwen3-235B-A22B-FP8 with tensor parallelism over four H200 GPUs | GPU count — 8 GPUs of the target card type | [2608.14624](https://arxiv.org/abs/2608.14624) | READ BODY |
| D36 | Pythia — workflow-predictable agent-native serving | A server with two AMD EPYC 7J13 CPUs, 1.7 TB DRAM and eight NVIDIA A100 80GB GPUs interconnected via NVLink | GPU count — 8 GPUs | [2604.25899](https://arxiv.org/abs/2604.25899) | READ BODY |
| D37 | EpiCache — long-term conversation KV management | A server with 8× NVIDIA A100-40GB (PCIe) and dual Xeon Platinum 8275CL, host-device transfers over PCIe 4.0 x16 | GPU count — 8 GPUs on a DGX A100 | [2509.17396](https://arxiv.org/abs/2509.17396) | READ BODY |
| D38 | py-kvcache — external NVMe KV caching for vLLM | A Snellius GPU node: 4× NVIDIA H100 (PCIe 5.0, 94 GiB HBM2e), dual AMD EPYC 9334, 768 GiB DRAM, 7.5 TB PCIe 5.0 SSD | GPU count — 4 GPUs | [2609.11744](https://arxiv.org/abs/2609.11744) | READ BODY |
| D39 | Multi-turn LLM conversations under the LRU policy — hit-ratio theory | 5× Ascend 910B2 NPUs in a P/D-disaggregated architecture (1 prefiller + 4 decoders), ~64 GB HBM each | GPU count — 5 NPUs; the verifier downgraded the body quote to TITLE ONLY (the "verbatim" sentence was a splice of two non-adjacent sentences) | [2609.02027](https://arxiv.org/abs/2609.02027) | TITLE ONLY |
| D40 | Online KV compaction for agents — latency measurement | NVIDIA H200 GPUs; the latency result uses SGLang 0.5.15.post1 with tensor parallelism of two, FP8 weights and an FP8 KV cache | GPU count — TP=2 for the measured configuration | [2608.00902](https://arxiv.org/abs/2608.00902) | READ BODY |
| D41 | Tutti — GPU-centric SSD-backed KV cache store | A server with two H100 GPUs (80 GB HBM) connected via NVLink and 4× Solidigm D7-PS1010 7.68 TB SSDs; the distributed-scale variant spans two PCIe root complexes | GPU count — 2 NVLink-connected GPUs minimum | [2605.03375](https://arxiv.org/html/2605.03375v1) | READ BODY |
| D42 | Swarm — co-activation-aware KV cache offloading across multiple SSDs | 8× NVIDIA H20 GPUs (96 GB HBM each), Intel Xeon Gold 6530, 1 TB DDR5 DRAM, up to 8 NVMe SSDs | GPU count — 8 GPUs | [2603.17803](https://arxiv.org/html/2603.17803v1) | READ BODY |
| D43 | SwiftCache — cross-model KV cache sharing | A server with four NVIDIA H20 GPUs (96 GB each), 64-core CPU, 128 GB host memory; GPUs interconnected by NVLink at 400 GB/s bidirectional | GPU count — 4 GPUs on one NVLink server (the paper states GPUs on different servers cannot use NVLink) | [2606.16135](https://arxiv.org/html/2606.16135v1) | READ BODY |
| D44 | Preble — distributed prompt scheduling co-optimizing KV reuse and load balancing | A four-NVIDIA-A6000 GPU cluster and an eight-NVIDIA-H100 GPU cluster; abstract: evaluation on "two to 8 GPUs" | GPU count — 2–8 GPUs across a cluster | [2407.00023v1](https://arxiv.org/html/2407.00023v1) · [2407.00023](https://arxiv.org/abs/2407.00023) | APPROXIMATE |
| D45 | AttentionStore / CachedAttention (multi-turn KV reuse over a KV hierarchy) | 4× NVIDIA A100 80GB (320 GB aggregate HBM) attached over PCIe Gen 4 — deliberately no NVLink — plus 128 GB DRAM and 10 TB of SSDs | GPU count — 4 GPUs; no single-GPU configuration is reported | [2403.19708](https://arxiv.org/abs/2403.19708) · [2403.19708v1](https://arxiv.org/html/2403.19708v1) | READ BODY |
| D46 | DASC — hybrid (KDA) recurrent-state checkpoint compression with TP-rank balancing | All formal experiments run in SGLang with tensor parallelism (TP) 8; the method explicitly balances compressed state checkpoints across TP ranks | GPU count — TP8 deployment | [2608.30386](https://arxiv.org/abs/2608.30386) | READ BODY |
| D47 | [SM120] DeepSeek-V4.1-Flash 1M-context field report and the DeepSeek-style sparse-MLA attention-sink blocker | 8× RTX PRO 6000 Blackwell Server Edition (sm120, 96 GB), TP=8, driver 580.126.09; Engram tables alone are ~189 GiB across the TP group, so one 96 GB card cannot hold the weights | GPU count — the working configurations are all 8-GPU | [vllm#56700](https://github.com/vllm-project/vllm/issues/56700) · [vllm#55757](https://github.com/vllm-project/vllm/issues/55757) | READ BODY |

### D.2 Requires >96 GB VRAM

Only two items in the gathered evidence are bound by memory capacity rather than by GPU count; every other large-memory result is also gated on device count and therefore sits in D.1.

| # | Technique | Hardware requirement | Why it is out of reach | URL | Evidence |
|---|---|---|---|---|---|
| D48 | TriAttention in TensorRT-LLM | NVIDIA B200 (SM100, 192 GB HBM); scope is limited to PyTorch, `KVCacheManagerV2` and SM100/B200, with TP configurations beyond TP1 unsupported | Aggregate VRAM and compute capability — 192 GB and SM100, while the target card is SM120 with 96 GB | [TensorRT-LLM#16957](https://github.com/NVIDIA/TensorRT-LLM/pull/16957) | READ BODY |
| D49 | Random Attention — serving-throughput measurements | One NVIDIA H200 with 143 GB HBM (vLLM v0.19.0, one job per GPU, budget 2048, 1k-token prompts); batch sizes are set to "the largest batch that fits one 143 GB H200" | Per-device VRAM — 143 GB exceeds the 96 GB card, and the reported batch sizes are defined by that memory | [2609.03430v1](https://arxiv.org/html/2609.03430v1) | READ BODY |

### D.3 Requires multi-node, NVLink, or InfiniBand

Rows here are bound by a node boundary or an inter-node fabric. Single-node NVLink-adjacent results whose binding constraint is device count appear in D.1 instead.

| # | Technique | Hardware requirement | Why it is out of reach | URL | Evidence |
|---|---|---|---|---|---|
| D50 | semi-PD — disaggregated computation with unified storage | Four separate server platforms of 8× datacenter GPUs each (A100 80GB SXM4; four nodes of A100 40GB SXM4; H200 141GB SXM5; A800 80GB SXM4), NVLink inside nodes, 200 Gbps cross-node; the paper's motivation also needs ≥2 GPUs to deploy one 8B model pair | Node count — four server platforms with cross-node links | [2504.19867](https://arxiv.org/html/2504.19867v1) | READ BODY |
| D51 | HMA-Serve — memory-heterogeneous prefill/decode across two vendors' accelerators | Real silicon: NVIDIA A100-80GB on the HBM side plus a four-chip Tenstorrent Blackhole p150 mesh (TT × 4) on the GDDR side, over a 100 Gb RoCE fabric, serving through vLLM 0.19.1 | Interconnect and second accelerator — a non-NVIDIA GDDR device plus a RoCE fabric | [2606.29986v1](https://arxiv.org/html/2606.29986v1) | READ BODY |
| D52 | Prefill-as-a-Service (PrfaaS) | Multiple loosely coupled clusters across datacenters: standalone compute-dense prefill clusters plus local PD clusters, transferring KVCache over commodity Ethernet; case study on an internal 1T-parameter hybrid model | Node count — spans datacenters | [2604.15039](https://arxiv.org/abs/2604.15039) | READ BODY |
| D53 | LMCache multi-node tensor parallelism | Multi-node TP — the issue asks for cross-node weight/KV placement; the multi-node `lmcache server` capability is described as an unmerged proposal (PR #3248) | Node boundary — requirement crosses hosts (title-level evidence only) | [LMCache#3266](https://github.com/LMCache/LMCache/issues/3266) | TITLE ONLY |
| D54 | SGLang hybrid Mamba state-pool sizing from `max_running_requests` | 2 nodes × 8 NVIDIA H20-3e, TP16 / EP16, Kimi-K3 + Kimi-K3-DSpark MXFP4-AFP8 | Node count and GPU count — 16 ranks across 2 nodes | [sglang#33004](https://github.com/sgl-project/sglang/issues/33004) | READ BODY |
| D55 | Cross-instance MLA attention redistribution ("Move the Query, Not the Cache") | A production cluster of 4× H100 SXM5 nodes, NVLink 4.0 intra-node (six bonded links per GPU pair; the 4-GPU HGX board carries no NVSwitch) and InfiniBand NDR-200 cross-node, NVSHMEM 26.3 with IBGDA; a cross-instance run is a 2-node × 4-PE job | Node boundary plus InfiniBand — the headline claim is explicitly about cross-node transfer | [2606.01502](https://arxiv.org/html/2606.01502v1) · [2606.01502](https://arxiv.org/abs/2606.01502) | READ BODY |
| D56 | LAGA — latent all-gather attention for MLA sequence parallelism | One node of 8× Ascend 910B (HCCL) for single-node work, and two nodes × 8 cards (16 ranks) spanning both nodes over RoCE at ≈11 GB/s for the multi-node SP group | Node count — 16 ranks across two nodes over RoCE | [2607.17644](https://arxiv.org/abs/2607.17644) | READ BODY |
| D57 | vLLM disaggregated prefill (NIXL / Mooncake connectors, two-engine reference deployment) | ≥2 GPUs / 2 vLLM instances (a prefiller and a decoder); RDMA for the transfer path; multi-node NVLink (MNNVL, GB-series) plus `--enable-cumem-allocator` or `--enable-sleep-mode` and `UCX_CUDA_IPC_ENABLE_MNNVL` for the peer-memory path | Node boundary and RDMA — the documented deployment launches two engines and the fast path needs MNNVL | [disagg_prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) · [nixl_connector_usage.md](https://github.com/vllm-project/vllm/blob/main/docs/features/nixl_connector_usage.md) | READ BODY |
| D58 | NIXL KV descriptor scale on GB200 / MNNVL | GB200-class multi-GPU nodes with MNNVL/`cuda_ipc` and RDMA | NVLink domain — GB200-class MNNVL fabric (title-level evidence only; the descriptor-scale finding itself was not read) | [vllm#55434](https://github.com/vllm-project/vllm/issues/55434) | TITLE ONLY |
| D59 | Helix Parallelism | GB200 NVL72-class hardware with inter-GPU NVLink; 405B and 671B models; KV histories of ≥1M tokens (evaluated in an in-house GB200 simulator, not on a single GPU) | NVLink domain and model scale — NVL72-class interconnect, evaluated by simulator | [2507.07120](https://arxiv.org/abs/2507.07120) | READ BODY |
| D60 | Star Attention | 8–32 A100 GPUs spread over multiple hosts, 4–16 parallel workers; 1M-token sequences used 32 GPUs and the 70B model at 128K also used 32 GPUs | Node count and GPU count — 32 GPUs across multiple hosts | [2411.17116](https://arxiv.org/abs/2411.17116) | READ BODY |
| D61 | LoongTrain (head-context parallelism) | A cluster of 8 GPU servers, each with 8 NVIDIA Ampere GPUs at 80 GB, intra-node NVLink, and 4× NVIDIA Mellanox HDR (200 Gb/s) InfiniBand NICs per node | Node count plus InfiniBand — 8 servers | [2406.18485](https://arxiv.org/abs/2406.18485) | READ BODY |
| D62 | Infinite-LLM — distributed KVCache across a cluster | A cluster of 32 A100 GPUs (4 nodes × 8) using a pooled GPU-memory strategy | Node count — 4 nodes / 32 GPUs, with no single-GPU mode | [2401.02669](https://arxiv.org/abs/2401.02669) | READ BODY |
| D63 | DistServe | A cluster with 4 nodes and 32 GPUs; each node has 8 NVIDIA SXM A100-80GB connected with NVLink; 25 Gbps cross-node bandwidth | Node count — 4 nodes (and ≥2 GPUs for the minimum prefill/decode split) | [2401.09670](https://arxiv.org/abs/2401.09670) | READ BODY |
| D64 | Splitwise | At least two machines (prompt-computation and token-generation) connected with InfiniBand; characterization used two DGX-A100 and two DGX-H100 VMs on Azure (the H100s at 400 Gbps) | Node count plus InfiniBand — two separate machines | [2311.18677](https://arxiv.org/abs/2311.18677) | READ BODY |
| D65 | NanoCP — request-level dynamic context parallelism | A cluster of NVIDIA H200 GPUs; each node has eight GPUs on a fully-connected NVLink fabric (900 GB/s bidirectional) plus eight 50 GB/s RDMA NICs for inter-node communication | Node boundary and InfiniBand — inter-node payload routing is over IB | [2605.21100](https://arxiv.org/abs/2605.21100) | READ BODY |
| D66 | NVIDIA Dynamo — KV-aware routing and KV Block Manager | Multiple GPUs or multiple nodes; the README states that for a single model on a single GPU the inference engine alone is sufficient, and its KV features target NVL72/NVLink fabrics | Node boundary — it is an orchestration layer above a cluster | [dynamo README](https://raw.githubusercontent.com/ai-dynamo/dynamo/main/README.md) | READ BODY |
| D67 | vLLM tensor/pipeline parallelism with InfiniBand and GPUDirect RDMA | Multiple GPUs; for multi-node, high-speed network adapters such as InfiniBand plus GPUDirect RDMA (documented example: `tensor_parallel_size=8`, `pipeline_parallel_size=` number of nodes) | Node boundary plus InfiniBand — multi-node TP/PP | [parallelism_scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html) | READ BODY |
| D68 | vLLM expert parallelism / DeepEP / NVSHMEM | An H200 or H20 node with 8 GPUs for the single-node recipe; multi-node requires DeepEP kernels, an InfiniBand/RoCE fabric and NVSHMEM; MNNVL systems for the FlashInfer NVLink A2A backends | Node boundary plus InfiniBand/RoCE — even the single-node recipe needs 8 GPUs | [expert_parallel_deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html) | READ BODY |
| D69 | LMCache — multi-node P2P KV sharing over NIXL | Multiple serving engines / multiple nodes; KV transfer over NVLink, RDMA or TCP through transport layers such as NIXL | Node boundary — multiple engines across nodes | [LMCache README](https://raw.githubusercontent.com/LMCache/LMCache/dev/README.md) | READ BODY |
| D70 | BlockLLM | A cluster of four servers: two with 2× A100 80GB and two with 4× A100 80GB (12 GPUs total), interconnected with a 100 Gbps network | Node count — four servers | [2404.18322](https://arxiv.org/abs/2404.18322) | READ BODY |
| D71 | vLLM KV-offload / eviction failure that only manifests on a multi-node cluster | 3 nodes × 8 H100 = 24 GPUs (TP8+DP3), with `--nnodes 3` and 512 GB `cpu_bytes_to_use` per configuration | Node count — 24 GPUs across 3 nodes | [vllm#54914](https://github.com/vllm-project/vllm/issues/54914) | READ BODY |
| D72 | psRL — training-time prefix sharing for agentic RL | A multi-node GPU cluster with 4× 80GB NVIDIA A100 per node, 200 Gb/s intra-node and 100 Gb/s RDMA cross-node; the industrial Qwen3-235B configuration spans 1536 GPUs (DP=4, TP=4, EP=8, PP=12) | Node boundary plus RDMA fabric — multi-node cluster | [2608.25683](https://arxiv.org/abs/2608.25683) | READ BODY |
| D73 | vLLM hybrid partial-prefix-checkpoint validation | Two 4-GPU nodes (TP8/EP8) running Kimi-K3 with FlashKDA, prefix caching and speculative decoding; NIXL/MooncakeStore PD coverage | Node count — two nodes for TP8/EP8 | [vllm#53614](https://github.com/vllm-project/vllm/pull/53614) | READ BODY |
| D74 | VPP — virtual-stage chunked-prefill pipeline parallelism up to 1M tokens | 16 Ascend 910C NPUs, all configurations using 8-way tensor parallelism with two pipeline ranks (TP8+CPP2, TP8+DCPP2, TP8+VPP2) | GPU count and node boundary — 16 NPUs, TP8+PP2 | [2608.26523](https://arxiv.org/abs/2608.26523) | READ BODY |
| D75 | Cartridges at Scale (CAS) — training modular KV caches over document collections | A cluster with NVIDIA H200 and B200 cloud GPUs plus persistent storage for cartridge rotation (the training pool holds a bounded number of cartridges on GPU, the rest offloaded) | Node boundary — multi-GPU cloud cluster with GPU↔persistent-storage swapping | [2606.04557](https://arxiv.org/abs/2606.04557) | READ BODY |

### D.4 Requires datacenter storage fabric or a distributed KV store service

| # | Technique | Hardware requirement | Why it is out of reach | URL | Evidence |
|---|---|---|---|---|---|
| D76 | Mooncake — KVCache-centric disaggregated architecture (KVCache-centric scheduler, disaggregated KVCache pool) | A compute-node cluster where each node is "8 NVIDIA-A800-SXM4-80GB GPUs, each with 80GB HBM, connected by NVLINK; equipped with RDMA network cards that supporting up to 800 Gbps of interconnect bandwidth between nodes"; each node deploys either a prefill or a decoding instance | Node count plus RDMA fabric — a distributed pool with separate prefill and decode clusters | [2407.00079v3](https://arxiv.org/html/2407.00079v3) · [2407.00079v4](https://arxiv.org/html/2407.00079v4) · [2407.00079v1](https://arxiv.org/html/2407.00079v1) · [2407.00079](https://arxiv.org/abs/2407.00079) | READ BODY |
| D77 | SGLang HiCache published benchmark configurations (Mooncake and DeepSeek 3FS backends) | Qwen3-235B-A22B-Instruct-2507 on 8× H800 GPUs with 8× mlx5 RDMA NICs using Mooncake; separately DeepSeek R1 on 8× H20-3e with `--tp 8` and `--hicache-storage-backend hf3fs` against a distributed 3FS filesystem | Node count plus storage fabric — 8-GPU nodes wired to a distributed storage backend | [lmsys.org HiCache blog](https://lmsys.org/blog/2025-09-10-sglang-hicache/) | READ BODY |
| D78 | SGLang HiCache cross-instance (L3) reuse | A cluster of SGLang instances all pointed at the same distributed L3 namespace (mooncake / hf3fs / nixl / aibrix); L2 is host memory owned by one instance's process, so two instances never read each other's L2, not even on the same node | Distributed service — cross-instance reuse requires a cluster-wide storage namespace, not merely more VRAM | [hicache_design.md](https://docs.sglang.io/docs/advanced_features/hicache_design.md) · [hicache_design](https://docs.sglang.io/docs/advanced_features/hicache_design) | READ BODY |
| D79 | SGLang PD disaggregation via the Mooncake TransferEngine | Separate prefill and decode node groups; HiCache can be enabled on both the prefill nodes and the decode nodes | Node count — separate prefill/decode node groups | [hicache_design](https://docs.sglang.io/docs/advanced_features/hicache_design) | READ BODY |
| D80 | MemServe / MemPool — elastic memory pool for KV and context | The design requires multiple serving instances across a cluster with RDMA-capable interconnect; what was actually built and measured is one NVIDIA DGX H800 server with 8× H800-80GB on intra-node NVLink, using NCCL send/recv and sockets because RDMA was never implemented | Distributed pool service — the design's requirement is a cross-instance KV pool over RDMA | [2406.17565](https://arxiv.org/abs/2406.17565) | READ BODY |
| D81 | PTStore — distributed prefix tensor store with replication | ALCF Polaris: 560 nodes, each with 512 GB DDR4, 32-core AMD Zen 3, two 1.6 TB SSDs and four NVIDIA A100 GPUs aggregating to 160 GB HBM per node, four NVLinks per node and a one-to-one GPU↔NIC mapping | HPC fabric — the point is aggregating KV memory across hundreds of nodes | [2607.22648](https://arxiv.org/abs/2607.22648) | READ BODY |
| D82 | CacheRoute — cache-aware routing as a fleet-level planner | 30 tensor-parallel-2 destinations on 60 H100 GPUs (each destination exposing 40,071 measured KV blocks, ~641K tokens); a separate 30× single-H100 mechanism testbed | Fleet scale — 60 GPUs behind a router | [2608.19677](https://arxiv.org/abs/2608.19677) | READ BODY |
| D83 | PrefixPlace — provable prefix-KV placement across a worker fleet | A multi-worker serving topology with cross-worker replica fetches, evaluated at 4/8/16 workers and measured across T4, L4 and A100 requester hardware | Multi-worker fleet — placement/solver results are defined over cross-worker replica fetches | [2608.01655](https://arxiv.org/abs/2608.01655) | READ BODY |
| D84 | HyperOffload — graph-driven hierarchical memory management | SuperNode architectures offering terabyte-scale shared memory pools across high-bandwidth interconnects; the compiler treats remote memory access as explicit operations in the computation graph | SuperNode memory pool — the software is "specifically designed for hierarchical SuperNode architectures" | [2602.00748](https://arxiv.org/abs/2602.00748) | READ BODY |
| D85 | SGLang RFC "Beyond Passive Byte Stores" (MORI-UMBP) | A multi-node cluster with RDMA-capable NICs (zero-copy RDMA, IBGDA/SDMA transports), AMD GPU/CPU/NIC affinity by design (AMD Infinity Storage, GPU-initiated NVMe), and a cluster-wide master KV placement directory over per-node capacity | Cluster-wide RDMA pool plus AMD-specific hardware affinity | [sglang#27898](https://github.com/sgl-project/sglang/issues/27898) | READ BODY |
| D86 | InstInfer / In-Storage Attention Offloading | Computational storage drives: an InstCSD built on a Daisyplus OpenSSD with a Xilinx ZU17EG UltraScale+ MPSoC FPGA, 2 GB DRAM, PCIe Gen3x4, 64 GB and four flash channels (the GPU side is a single A6000) | Specialist storage hardware — the paper itself notes "expensive FPGA chips, costing thousands of dollars" and a 64 GB / 4-channel limit | [2409.04992](https://arxiv.org/abs/2409.04992) | READ BODY |
| D87 | Photonic-CXL memory appliance for KV cache management | A photonic-CXL hybrid memory appliance delivering 32 TB of shared memory across 16 hosts via a switch-free full-crossbar passive fiber shuffle | Storage fabric — 16 hosts plus a photonic CXL fabric | [2607.27187](https://arxiv.org/abs/2607.27187) | READ BODY |

### D.5 Borderline: published on multi-GPU but with a stated single-GPU mode or configuration

| # | Technique | Hardware requirement | Why it is out of reach | URL | Evidence |
|---|---|---|---|---|---|
| D88 | LMCache paper evaluation (cross-engine KV store, PD disaggregation) | Single-node evaluation on an 8× H100 server (GMI Cloud); multi-node uses the same GPU count with a remote CPU-memory storage backend; the offloading result is quoted at TP=2 | Not out of reach for the software — only the published evaluation is multi-GPU; LMCache itself does not require multi-GPU | [2510.09665v1](https://arxiv.org/html/2510.09665v1) · [2510.09665](https://arxiv.org/abs/2510.09665) | APPROXIMATE |
| D89 | vLLM batched KV swap copy | Benchmark hardware: 8× H100 80GB HBM3, CUDA 12.8/12.9 | Not out of reach — the merged code path itself is single-GPU compatible; only the published numbers are 8-GPU | [vllm#38460](https://github.com/vllm-project/vllm/pull/38460) | READ BODY |
| D90 | TurboQuant 4-bit KV on Gemma 4 31B multimodal (Blackwell sm_120 blocker stack) | The reporter's setting is 4× RTX 5060 Ti (sm_120) at TP=4, motivated by the context not fitting one 16 GB card | Not intrinsically multi-GPU — the issue is KV-compression gating on sm_120; the multi-GPU setting is the reporter's configuration | [vllm#41403](https://github.com/vllm-project/vllm/issues/41403) | READ BODY |
| D91 | Cross-layer index-topk reuse measured benefit for DeepSeek-V4 | The cited benefit measurement is 8× Hopper, TP8+EP, with `index_topk_freq=2` run for a full working day in production | Not intrinsically multi-GPU — only the benefit measurement is 8-GPU; the requested code change is not | [sglang#36083](https://github.com/sgl-project/sglang/issues/36083) | READ BODY |
| D92 | Continuum / CacheTTL — multi-tier KV lifetime scheduling | 4× B200 for the 70B configuration, an 8× H100 node for the RL micro-benchmark, and Company A's internal H100 testbed (the page prints a typographic apostrophe) for a 500-task SWE-agent run; single-GPU rows also exist (1× B200, 1× A100) | Not out of reach — the paper reports single-GPU configurations alongside the multi-GPU ones | [2511.02230v4](https://arxiv.org/html/2511.02230v4) | READ BODY |
| D93 | Runtime Observability for Heterogeneous Attention Memory | All headline serving experiments run on a single node with 8× H200 (143 GB) GPUs; regression runs for smaller models use a separate RTX 4090 pool, including a single-GPU Qwen2.5-7B serving stack | Partially reachable — some observability sub-experiments do run single-GPU on an RTX 4090, but the headline serving results are 8×H200 | [2608.05863v2](https://arxiv.org/html/2608.05863v2) | READ BODY |
| D94 | vLLM tiered KV offloading (announcement benchmark) | The feature is hardware-agnostic, but every published number is Qwen3.6-35B-A3B on 2× NVIDIA H100 (TP=2) with a local-NVMe filesystem tier, 12K-token prompts, 8 rounds, concurrency 64 | Not out of reach for the feature — only the published performance claims were produced at TP=2 | [vLLM blog 2026-09-10](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading) | READ BODY |

Counter-cell recorded but deliberately not given a D row: vLLM's CPU/filesystem KV offloading tier (`OffloadingConnector` / `TieringOffloadingSpec`, KV to CPU pinned memory and secondary tiers via `cudaMemcpyAsync`, no RDMA/NVLink/second GPU in the documented path) carries no hardware requirement beyond one GPU and therefore is *not* ruled out by this screen — [kv_offloading_usage](https://docs.vllm.ai/en/latest/features/kv_offloading_usage.html) (READ BODY).

## Empty subsections (explicitly: nothing found, and the queries tried)

This map's rule is that a subsection with no genuine evidence says so rather than being padded. Below is, per subsection, exactly what came back empty and the queries that were tried. The full unabridged declarations, including every non-promoted lead URL, remain in the per-subsection evidence files (`evidence/kv-gapmap/S1.md` … `S14.md`, each with a `## EMPTY SUBSECTIONS` section).


### 1. paged/block KV management (`S1.md`)

No class came back empty. All four classes have at least four evidenced rows. Two caveats about evidence depth, recorded honestly rather than padded:

1. **Closed-unmerged PR rows were dropped for want of a stated reason.** I identified many closed-unmerged vLLM PRs in this subsection via the HTML issue-search endpoint (`is:pr is:unmerged label:kv-cache-manager`, and `is:pr is:unmerged "page size"` / `"block size"`). Examples retrieved as titles only: #46841 "[Bugfix][Core] Fix request-bound KV cache sizing" (CLOSED unmerged), #4762 "[Core][Bugfix]: fix prefix caching for blockv2" (CLOSED unmerged), #54949 "[XPU] env to control block size alignment for mamba" (CLOSED unmerged), #54735 "[Bugfix][DCP] Align SimpleCPU offload hybrid geometry" (CLOSED unmerged), #55622, #55134, #56320, #56223, #56222, #56221, #56205. For #46841 I retrieved the PR page and its rendered comment bodies and found **only** bot/AI text ("You have reached your Codex usage limits for code reviews." and "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @lesj0610 ."), not a human closure rationale. For #4762 I retrieved the PR body but no closure comment. Per hard rule 3 I did not emit ABANDONED rows for these, because I could not retrieve a verbatim stated reason. The blocker is structural: GitHub lazy-loads PR conversation bodies, and the REST API (`/repos/vllm-project/vllm/issues/{n}/comments`) returned HTTP 403 "API rate limit exceeded for 50.7.158.236" on every attempt across the whole session.
2. **#53178 and #42571 also lack a retrievable closure reason.** Both are `stateReason=NOT_PLANNED` in the page metadata, but the fetched payloads contain no maintainer comment explaining the closure (for #53178 only the RFC body is present; for #42571 only the environment block). They are omitted from ABANDONED for the same reason. #42571 ("[Bug]: KV Block double free when using eager SimpleCPUOffloading + Sliding window attention", created 2026-05-13, closed 2026-09-12) sits in the same 2026-09-12 bulk-closure batch as ABANDONED-5 and is likely also a stale-bot closure, but the page does not show the bot comment, so I am not asserting that.

Queries tried for the classes where I searched hardest:
- ABANDONED: `is:issue is:closed reason:"not planned" kv cache repo:vllm-project/vllm`; `is:issue is:closed reason:"not planned" kv cache repo:sgl-project/sglang`; `is:pr is:unmerged label:kv-cache-manager repo:vllm-project/vllm`; `is:pr is:unmerged "page size" repo:vllm-project/vllm`; `is:pr is:unmerged "block size" repo:vllm-project/vllm`; `is:pr is:unmerged "memory pool" repo:sgl-project/sglang`; `vLLM "wontfix" OR "not planned" KV cache block size allocator issue closed`; `vLLM PR "closed" unmerged "KV cache" block manager revert`; `llama.cpp KV cache defragmentation "not needed" removed unified cache`; `llama.cpp KV cache defragmentation removed defrag_thold deprecated`
- HW_RULED_OUT: `state+space+model+KV+cache+memory+management+serving`; `hybrid+attention+SSM+KV+cache+allocator+page`; `virtual+memory+KV+cache+GPU+LLM+inference`; plus direct `arxiv.org/html/<id>v1` setup-section extraction for 2506.15155, 2605.22416, 2606.06256, 2509.06261.


### 2. prefix caching (`S2.md`)

No class came back empty. Coverage caveats and items that could **not** be promoted to a numbered row (per the verbatim-quote rule) are listed here.

**Closed-unmerged PRs found but NOT emitted as ABANDONED rows (no retrievable verbatim closure reason):**
- https://github.com/vllm-project/vllm/pull/54609 — "WIP: Enable prefix caching for FI ReplaySSM" by askliar. State CLOSED, `merged_at` absent. Page payload contains only the timeline event "askliar closed this Sep 7, 2026" and no comment; the PR body returned by the GitHub search API is empty. Topic: prefix caching for the FlashInfer ReplaySSM path in hybrid Mamba models.
- https://github.com/vllm-project/vllm/pull/54953 — "WIP: Unify MTP/STP/Prefix Caching for ReplaySSM" by askliar. Same author, same closure date (2026-09-07), same absence of any stated reason. Topic: unifying MTP/STP/prefix-caching lifecycles for ReplaySSM.
- https://github.com/vllm-project/vllm/pull/54319 — "[Prefix Cache] Support fine-grained SWA hits", closed unmerged by the author on the day it was opened with no comment at all.
- https://github.com/vllm-project/vllm/pull/51640 — "[Security] Include cache_salt in HF3FS external cache keys", closed 2026-08-10 by the author on the day it was opened, `merged_at` absent, no closure comment. Topic: salting × cross-instance external KV cache keys (the exact question in OPEN-7).
- https://github.com/vllm-project/vllm/pull/52971 — "[Kimi K3][Pref] Reuse internal checkpoint blocks for partial prefix caching", closed unmerged 2026-08-22. The only reviewer statement retrievable is a scope request, not a rejection: "Thanks for building on #52789 . Since this PR currently overlaps with its checkpoint infrastructure, please keep it in draft until #52789 lands, then rebase it on top and remove the duplicated pieces." Its functionality was subsequently carried by #53614 (CLOSED-9).

**arXiv-search coverage limit:** `https://arxiv.org/search/?searchtype=all&query=<q>` returned HTTP 200 for the first query in this session and HTTP 429 for every subsequent one, even with 18–25 s spacing and 4 retries per query. Only one listing page (`prefix caching KV cache`, 50 results) was captured. The `export.arxiv.org/api/query` Atom endpoint returned HTTP 429 on every attempt. All other paper evidence below was obtained from individual `arxiv.org/abs/<id>` and `arxiv.org/html/<id>v1` retrievals, which were reliable. As a result the paper-level sweep is *not* exhaustive and is biased toward papers named in the one listing page that succeeded.

---


### 3. eviction/reuse policy (`S3.md`)

None of the four classes is empty. Two sub-topics inside the subsection produced no independently-evidenced item and are recorded here rather than fabricated:

1. **RazorAttention-specific production/abandonment evidence.** I retrieved the canonical paper (https://arxiv.org/abs/2407.15891, READ BODY, "we propose RazorAttention, a training-free KV cache compression algorithm, which maintains a full cache for these crucial retrieval heads and discards the remote tokens in non-retrieval heads") but found no merged engine support, no closed-unmerged PR, and no issue about it that I could retrieve. Its retrieval-head premise is answered only indirectly by RazorAttention-adjacent work (the `DuoAttentionPress` in kvpress, "split heads into retrieval heads (no compression) and streaming heads (StreamingLLM approach)"). No CLOSED/OPEN/ABANDONED/HW row is emitted for RazorAttention specifically.
2. **Scissorhands-specific artifact.** Scissorhands appears only inside other artifacts' text (the vLLM RFC #5751 list of eviction methods; the vLLM issue #55463 motivation list) and inside `kvpress`'s history. I did not retrieve a Scissorhands paper page or any Scissorhands-specific engine artifact, so no row is emitted.

Queries that returned nothing usable for these two: `RazorAttention retrieval head KV cache eviction 2026 follow-up`, `Scissorhands PyramidKV limitations negative results`, `Scissorhands KV cache eviction implementation vLLM`.

---


### 4. KV quantization (`S4.md`)

No class came back empty. All four classes have at least five genuinely evidenced rows.

Two targeted sub-topics returned NO directly-retrieved evidence and are therefore NOT represented as rows (rather than being fabricated):

1. **`kv-cache-dtype` calibration-free INT4 per-token-head accuracy on Qwen3-4B specifically.** No page was retrieved that reports GSM8K/logprob deltas for `int4_per_token_head` on Qwen3-4B; the shipped accuracy test uses `meta-llama/Llama-3.2-1B-Instruct` at `MAX_MODEL_LEN = 1024`, `max_tokens = 4`, `tensor_parallel_size = 1`, `backend = "TRITON_ATTN"` (read locally at `/tmp/kvsrc/vllm-main/tests/models/quantization/test_per_token_kv_cache.py`). Queries attempted are listed below.
2. **A revert of a KV-quantisation change in vLLM itself** (as opposed to SGLang's ABANDONED-11). Queries attempted did not surface one, and no such PR page was retrieved, so no row was emitted.

---


### 5. offloading & tiered storage (`S5.md`)

None of the four classes came back empty. Caveats on completeness and on items deliberately NOT emitted as rows:

- **RDMA KV stores as such**: no dedicated RDMA-KV-store artifact was emitted as its own row. The RDMA material I actually retrieved is Mooncake's Messenger (cited in HW-1), vLLM's NIXL-based P2P tier (cited in CLOSED-4 and CLOSED-13), and the 2026 papers that argue *against* whole-prefix RDMA fetch (only abstract-level retrieval; not emitted as rows).
- **InfiniGen** (verified: arXiv 2406.19707, "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management", submitted 28 Jun 2024): its setup is a **single** RTX A6000 48GB with 96GB DDR4-2666 (verbatim: "We run the experiments on a system equipped with an NVIDIA RTX A6000 GPU [ 44 ] with 48GB of memory and an Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory. PCIe 3.0 × 16 interconnects the CPU and GPU."). It is therefore **NOT** HW-ruled-out, and it is a KV-selection/prefetch technique rather than a storage-tier technique, so it is not a clean CLOSED row for this subsection either. Left out of the tables deliberately rather than misclassified.
- **CacheGen** (verified: arXiv 2310.07240) is **NOT** HW-ruled-out — its setup is one NVIDIA A40 GPU server with 384GB host memory ("Hardware settings: We use an NVIDIA A40 GPU server to benchmark our results. The server is equipped with 384GB of memory and two Intel(R) Xeon(R) Gold 6130 CPUs"). Emitted as CLOSED-10 only.
- **Pensieve** (verified: arXiv 2312.05516) is **NOT** HW-ruled-out — evaluated on one NVIDIA A100 PCIe 80GB (Microsoft Azure NC24ads_A100_v4). Emitted as CLOSED-11 only. Note: "Pensieve: A Tiered, Fully-Automated, and Cost-Effective Virtual Cluster..." is a *different, unrelated* Pensieve.
- **FlexGen** (verified: arXiv 2303.06865, "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU") is explicitly single-GPU, so it is not HW-ruled-out; not emitted as a row because I did not retrieve a body quote tying it to a currently-open question.
- **HCache** (verified: arXiv 2410.05004, "Fast State Restoration in LLM Serving with HCache"): its headline testbed is a **partially** ruled-out configuration — verbatim "4 × A100-40G SXM4 connected via NVLink. The host has 2 × AMD EPYC 7642 CPUs, 256G DDR4 memory, and 4 × Samsung PM9A3 4TB enterprise SSDs." — but the paper also states verbatim "For Llama2-7B/13B, we use a single A100 GPU to serve them." Because it does **not require** more than one GPU, it is deliberately NOT emitted as an HW_RULED_OUT row.
- **CXL-SpecKV** (arXiv 2512.11920): I did not retrieve its body, so its hardware is unverified. Not emitted — a TITLE ONLY row asserting a hardware requirement would be a guess.
- **TieredKV / SpecKV / DeepSpeed ZeRO-Inference**: no verified ID or retrieved setup section. No evidence, no row.
- **Klotski** (arXiv 2502.06888): verified to be an MoE inference paper, not KV-cache offloading. Off-topic; excluded.
- **NVMe/SSD tiers, GPUDirect Storage, host-memory bandwidth bottleneck**: all three are covered — GPUDirect Storage by ABANDONED-4 (KvikIO prototype slower than everything) and ABANDONED-14 (Tutti: GDS stays CPU-centric); host-memory bandwidth by ABANDONED-18 and ABANDONED-19; NVMe/SSD tiers by CLOSED-4, CLOSED-12, CLOSED-21, CLOSED-22, ABANDONED-5, ABANDONED-15, ABANDONED-16.
- **Closed-unmerged with NO stated reason (deliberately NOT emitted as an ABANDONED row)**: vLLM PR #52784 "[Feature][KV Offload] Add capacity-bound LRU eviction to FileSystemTierManager" (https://github.com/vllm-project/vllm/pull/52784) is recorded as closed unmerged on 2026-08-19 by its author XiaYiHann, but the page carries no closing comment and therefore no retrievable stated reason. Under the no-fabricated-quote rule it is reported here rather than as a row.
- **Automated-stale closures (bot text only — NOT presented as maintainer decisions)**: three KV-eviction items in the offload/prefix-cache area were closed by an inactivity bot with no human rationale, so they are recorded here rather than as ABANDONED rows. All three carry the same bot text; verbatim examples: "This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you!" (vLLM PRs #22236 "Workload-Aware KVCache Eviction Policy" https://github.com/vllm-project/vllm/pull/22236 and #27539 "[Core] Prefix cache: frequency- and cost-aware eviction (opt-in)" https://github.com/vllm-project/vllm/pull/27539) and "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" (vLLM issue #23641 "[RFC]: Frequency and Cost Aware Eviction Policy for Prefix Caching" https://github.com/vllm-project/vllm/issues/23641). This evidence came from delegated retrieval, not from pages I fetched myself.
- **Item deliberately not classified**: vLLM PR #26921 "[V1][KV Cache] Add ARC (Adaptive Replacement Cache) eviction policy" was closed and then superseded — the recorded author comment is "I'll reopen the PR, messed up forcing the commit with sign off." (https://github.com/vllm-project/vllm/pull/26921), and the replacement PR #27039 "Implement ARC KV cache eviction policy" was merged. This is an author slip, not an abandoned technique, so it gets no row. Delegated retrieval; I did not fetch these two pages myself.
- **Automated-stale closures**: the only stale-bot closure I retrieved in this subsection is vLLM #39766 ("stale — Over 90 days of inactivity", closed as not planned). It is emitted as a row only in the QUERIES/notes trail, not as an ABANDONED row, because the closure reason is bot text and not a maintainer decision. Every other ABANDONED row above carries either a human/author statement or a paper's own published conclusion.


### 6. compression/low-rank (`S6.md`)

None of the four classes came back empty. Specifically:
- CLOSED: 16 rows, all with retrieved sources.
- OPEN: 9 rows, all with retrieved sources.
- ABANDONED: 12 rows. **Caveat on quality, not quantity:** five of the twelve (ABANDONED-2, -9, -10, -12 and the bot half of -1) rest on stale-bot closure text rather than a human maintainer decision, and are labelled as such. Only ABANDONED-1 contains a genuine human maintainer objection ("The required data is large IMHO."), and ABANDONED-3..-8 are paper-level negative/limiting results rather than engine-level rejections. I found **no** human-maintainer "wontfix"/"not planned"/explicit-rejection artifact for: cross-layer KV merging (MiniCache-style) in vLLM or SGLang, latent/low-rank KV (Palu-style) in vLLM or SGLang, learned KV codebooks, or MLA upcycling (X-EcoMLA/TransMLA/MHA2MLA-style). The `label:wontfix` search on vllm-project/vllm returned zero results, so a "wontfix label" artifact may not exist in that repo at all.
- HW_RULED_OUT: 8 rows.

Searches that came back empty and are therefore reported as genuinely-no-result rather than omitted:
- `repo:vllm-project/vllm label:wontfix` → 0 results (GitHub search HTML, one attempt)
- arXiv full-text search `MiniCache` → 0 parsed results on the attempt that hit an arXiv 429; the record was instead confirmed by direct abstract fetch of a specific ID
- arXiv full-text search `You Only Cache Once` → 0 parsed results (arXiv 429 on that attempt); confirmed by direct abstract fetch
- arXiv full-text search `Dynamic Memory Compression retrofitting` → 0 parsed results (arXiv 429); confirmed by direct abstract fetch
- GitHub REST search API was unusable for this entire session: every call returned `{"message":"API rate limit exceeded for 50.7.158.236. ..."}` (HTTP 403), so all GitHub discovery was done through the HTML search page and direct issue/PR page fetches.

---


### 7. cross-request & cross-replica sharing (`S7.md`)

- **KV multicast / broadcast (one-to-many KV dissemination)**: no shipped artifact, merged PR, open issue, or paper found that implements KV cache multicast or broadcast to multiple consumers. The closest retrieved item was vLLM PR title "[Bugfix][CUDA] Fix the FlashInfer's fused all-reduce issue where NVLink multicast isn't available" (https://github.com/vllm-project/vllm/pull/55973), which is NVLink multicast for all-reduce, i.e. tensor-parallel communication, not KV dissemination — not emitted as a row. No ABANDONED row either: I found no closed-unmerged/wontfix artifact with a retrievable verbatim closure reason on this topic, so per the hard rules I emit nothing rather than a row without a quote. Queries run for this: `all:"KV cache" AND all:"multicast"`, `all:"KV cache" AND all:"broadcast"` (arXiv API — both rate-limited, see concerns), `kv cache broadcast OR multicast` (GitHub HTML search, vllm-project/vllm), `multicast OR broadcast kv cache` (GitHub HTML search, sgl-project/sglang), `cross-engine OR cross engine OR multicast` (GitHub HTML search, LMCache/LMCache).
- **A merged cross-*vendor* engine KV exchange (e.g. a shipped vLLM↔TensorRT-LLM KV format bridge)**: the only retrieved cross-engine artifact is LMCache's (vLLM and SGLang, via the LMCache layer), emitted as CLOSED-10. The TensorRT-LLM HTML search `kv cache sharing OR reuse across instances` returned no parsed result rows within the retrieval window, so nothing is emitted for TRT-LLM. Queries run: `cross-engine OR "cross engine" kv` (vllm-project/vllm), `cross-engine OR cross engine OR multicast` (LMCache/LMCache), `kv cache reuse across requests OR instances` (NVIDIA/TensorRT-LLM — returned zero parsed rows).
- **KV deduplication as a first-class cross-replica feature** (distinct from prefix-cache key determinism and from LMCache's fingerprint dedup): no standalone shipped "KV dedup service" artifact was found. Two partial answers are emitted instead: CLOSED-5 (identical content → identical keys cross-process) and CLOSED-9 (`--enable-dedup-content`).


### 8. long context (`S8.md`)

No class came back genuinely empty. Coverage notes and thin spots, stated honestly:

- **ABANDONED for *ring attention KV specifically***: no closed/unmerged PR or paper-level retraction of a ring-attention-for-KV proposal was found in vLLM, SGLang, or arXiv within the window. What exists instead is vLLM's design doc declaring the partial-Q/partial-KV ring path "under active development" (HW-3) — that is an OPEN status, not an abandonment, so no ABANDONED row was emitted for it. I did not find a maintainer rejection of ring attention to quote, and I will not invent one.
- **CLOSED for "RAG with many documents" at the *engine* level**: the only shipping artifacts I could verify are prefix caching (`--enable-prefix-caching`, `--prefix-match-unit`) and the research-side CAS/CoinRAG line. I found no merged vLLM/SGLang PR that specifically targets many-document RAG KV (e.g. per-document cache selection inside the engine), so that sub-question has no CLOSED row of its own; it is represented by CLOSED-3 (system-prompt bloat) and CLOSED-8 (CAS).
- **CLOSED for "attention sinks for long context" inside a *dense GQA long-context* backend on sm120**: sink support exists per backend (CLOSED-6) but the in-tree docs explicitly exclude sinks from the Blackwell head_size=256 FA4 kernel (`does not support logit soft capping, attention sinks, mm_prefix/R-SWA masking, DCP, or windowed encoder attention`), so on this target hardware the sink path is backend-conditional rather than universally available. Counted as a caveat on CLOSED-6, not as a separate row.
- **Thin: exact PR number for the `--enable-mamba-fine-grained-prefix-cache` merge.** CLOSED-3 is evidenced from the shipped docs page and the in-tree config source rather than from the merge PR, because I did not retrieve the merge PR page.
- **arXiv API sweeps returned no data**: `export.arxiv.org/api/query` answered HTTP 429 (rate-limited) throughout this session, and `arxiv.org/list/...` returned HTTP 404 for the 2606/2609 listing URLs I tried. All paper-level findings in this file therefore come from `arxiv.org/abs/<id>` and `arxiv.org/html/<id>v1`, which worked reliably.

---


### 9. KV lifetime/TTL economics (`S9.md`)

None of the four classes came back empty. Two honesty caveats rather than empty classes:

1. **No maintainer-authored "wontfix" / "not planned" rejection of a KV-lifetime proposal was found.** The only `NOT_PLANNED` stateReason in the ABANDONED set (#23641) and the two other stale closures (#22236, #27539) are all `github-actions` STALE BOT text, explicitly labelled as such. Every human closure of a KV-eviction PR found was an *author self-close*. Queries run for this: `repo:vllm-project/vllm label:wontfix kv` (returned total 0), `repo:vllm-project/vllm eviction policy in:title`, `repo:vllm-project/vllm "TTL" kv in:title,body`, plus reading the full label lists of #26921, #27039, #42985, #43191, #43725, #49114, #41383, #54327, #36311 and #52784 (no `wontfix`/`not planned` label on any of them).
2. **One ABANDONED topic has no retrievable stated reason and is therefore folded, not rowed.** PR #52784 (capacity-bound LRU eviction for the FS tier, author self-close 2026-08-19, one day after opening) is documented inside ABANDONED-3 with the explicit marker "NO STATED REASON FOUND". It is not given its own row because no verbatim reason exists on the page. Similarly, PR #35652 (SwapConnector) is folded into ABANDONED-2 with only its timeline line quoted, because it carries no closing comment.

Also not represented as rows, for the same reason: several 2026 HiCache PRs in SGLang (e.g. #39297, #39283, #39269, #39267, all created 2026-09-13 and still OPEN) were seen only as list entries in a search result and were not opened, so they are omitted rather than cited at TITLE ONLY.


### 10. disaggregated prefill/decode (`S10.md`)

No class is empty. Coverage notes and the honest weak spots:

- **SGLang is thin on purpose.** `/tmp/kvsrc/sglang-main` is a partial checkout (only `benchmark/`, `docker/`, and docs fragments present; no `python/sglang` tree), so no SGLang source-level verbatim grep was possible. SGLang evidence here is limited to its documentation page (CLOSED-7 URL) and to `GitHub` search results, and the SGLang-specific ABANDONED row (ABANDONED-7) is actually a Dynamo-side issue about the SGLang backend.
- **LMCache layerwise was dropped from ABANDONED.** LMCache PR #4221 ("feat(sglang): add LMCachePDConnector for PD disaggregation over NIXL") was closed unmerged and **by its own author** (`skaulintel closed this Jul 23, 2026`) only ~21 minutes after opening, with no comment on the page. I could retrieve the closure *event* but not a *stated reason*, so per the hard rules it is not emitted as an ABANDONED row. Its PR body does record a real layerwise-vs-PD incompatibility: "Add LMCachePDConnector, extending the non-layerwise LMCacheConnector because only engine.store (not store_layer) forwards transfer_spec to the storage manager / PDBackend." Related open LMCache layerwise correctness bugs exist (#4921, #4132, #4133) but are storage-tier layerwise issues rather than P/D-transfer ones.
- **No "smaller-model, single-GPU P/D" paper was found.** Every P/D or KV-transfer systems paper retrieved for this subsection assumed ≥2 accelerators. The RTX PRO 6000-specific search queries below returned nothing on-topic, so I did not manufacture a row.
- **The `Prefill-as-a-Service` paper note** surfaced through a third-party aggregator (a GitHub paper-notes issue) was not used as a source; the arXiv abstract page was retrieved directly instead and is what is cited.


### 11. negative results (`S11.md`)

- **TensorRT-LLM: no longer empty, but still thin.** My own repo-scoped search `repo:NVIDIA/TensorRT-LLM is:closed reason:"not planned" "kv cache"` returned HTTP 403 (shared-IP search rate limit) on every attempt. The delegated TensorRT-LLM investigation later supplied exactly **one** unambiguous maintainer "we will not do this" KV-cache decision (ABANDONED-55, KV reuse across LoRAs), one merged revert (ABANDONED-56), one stale-bot close (ABANDONED-57) and one relocation (ABANDONED-58). It also found **no** verified case where TensorRT-LLM maintainers concluded that KV quantization (FP8/INT4/INT8), eviction or offloading does not pay off — in that repository the FP8/NVFP4 KV work is shipping in the opposite direction. Two TensorRT-LLM-adjacent items were visible only as titles inside the *Dynamo* tracker (`#4387` "[FEATURE]: Layerwise KV Cache Transfer for Disaggregated TensorRT-LLM Serving" and `#729` "[FEATURE]: How to manage offload percentage", both closed `not_planned`); neither page was retrieved, so under the verbatim rule they are **not** emitted as rows.
- **HW_RULED_OUT items with a retrieved setup sentence for *papers*: EMPTY.** All seven HW rows are GitHub issues, not papers. The delegated arXiv investigation retrieved **only abstract pages, never full-text HTML**, and therefore could not certify any multi-GPU/NVLink/InfiniBand requirement from an experimental-setup section. Every hardware string it did retrieve was single-GPU and hence *in* scope (e.g. arXiv 2606.21868: "On a real 24 GiB RTX 3090, WiSP achieves up to 2.0x the decode throughput of static offload at the same memory budget when the model does not fit."; arXiv 2605.06675: "The entire calibration takes 1.6 s on a single GPU and adds zero overhead at inference time."). HW-6 and HW-7 are TITLE-ONLY leads with no evidenced requirement.
- **HuggingFace TGI ABANDONED items in the 2025-06 → 2026-09 window: EMPTY.** The delegated investigation reports: `"kv cache" "not planned"` → total_count 0; `"kv cache" offload` → 1 result (issue #991, closed not_planned but on 2024-03-27, **outside the window**); the 72 hits on `"prefix caching"` were "almost entirely log-line noise ('Disabling prefix caching because of VLM model')". Two scope-limit statements were seen as search-API text-match fragments on **open** issues (#3333: "Note that prefix caching is not supported for VLMs. The hardware also needs to support flashinfer or flash-attention2, so prefix caching is not supported on GPUs with a CUDA capability that is lower than 8.0."; #3110: "NotImplementedError: Vlm do not work with prefix caching yet") — but these are **support-scope limits, not abandoned optimizations**, and were not full-page reads. The TGI closed-unmerged PR enumeration failed with HTTP 403 on both queries, so this gap is real and not a true negative.
- **Papers that FAILED TO REPRODUCE a claimed KV-cache speedup: EMPTY** for the paper literature. The only reproduction-style negative result found is non-paper (ABANDONED-46, the agentic-kv-cache trace study). One paper is *adjacent* — arXiv 2503.24000 identifies "missing pieces in their performance measurement, which could hinder their adoption in practice" — but it does not frame itself as a failed reproduction.
- **A maintainer statement of the exact form "offloading gave no speedup because PCIe bandwidth bound" for LMCache or vLLM: EMPTY.** The delegated LMCache/Dynamo investigation states plainly: "I found no PR or issue stating *'offloading gave no speedup because PCIe bandwidth bound'*, *'compression hurts accuracy'*, *'we reverted the async loader'*, or *'not worth it for single node'* in those words." The closest genuine analogues are ABANDONED-17, -18, -22 (offload measures as pure overhead vs vLLM's own prefix cache) and ABANDONED-45 (PCIe-bound prefetching, but for MoE experts).

---


### 12. hardware-ruled-out (`S12.md`)

No class is empty for S12. All four classes (CLOSED, OPEN, ABANDONED, HW_RULED_OUT) have multiple evidenced rows. Three specific *sub-questions* did come back thin or empty, and are recorded here rather than fabricated:

1. **"Multi-node KV technique abandoned by a *paper's* own conclusion, with a stated reason in the paper's limitations section"** — I did not find a paper that explicitly abandons a multi-GPU KV technique in its own limitations section. What I found instead are (a) PR-level abandonments with maintainer/stale-bot reasons, and (b) papers whose results undercut their own technique (ABANDONED-13 Beyond the Buzz, ABANDONED-14 Star Attention, ABANDONED-12 the vLLM disagg-prefill doc note) plus MemServe's own admission that the multi-node transport was never implemented (HW-14). The Star Attention, Beyond-the-Buzz and MemServe rows are the closest genuine matches.
2. **"SGLang-side ABANDONED PR/issue with a maintainer's *stated wontfix* reason"** — I still could not retrieve one, even after a dedicated sub-agent sweep. What exists instead is: (a) three SGLang PRs closed unmerged with reasons that are **entirely bot text** or **entirely absent** — ABANDONED-18 (`#27276`, stale bot: "Closing this because it has had no updates in 99 days."), ABANDONED-19 (`#21865`, stale bot: "Closing this because it is still a draft and has not been updated in 146 days."), ABANDONED-20 (`#25846`, author self-close with no comment and an empty template); and (b) documentation-level retraction (ABANDONED-11, `--cp-strategy zigzag` "temporarily unavailable"). **No SGLang maintainer statement declaring a multi-GPU KV technique infeasible was found.** My own direct GitHub search API attempts were blocked by HTTP 403 (secondary rate limit) in every case; the sub-agent reported the same 403 on its SGLang `disaggregation in:title` queries. This area is explicitly **under-sampled**.
3. **"A hardware-ruled-out KV technique specific to *this* GPU generation (sm120 / RTX PRO 6000 Blackwell)"** — nothing I retrieved is gated on sm120 specifically. All hardware gates I found are of a different kind: GPU *count* (TP/DCP/PCP rank counts), interconnect class (NVLink/MNNVL/InfiniBand/RDMA), or model scale. The one sm120-adjacent artifact is vLLM PR `#34795`'s author benchmark table, which reports DCP on 8× and 16× "RTX 6000 Pro" — i.e. sm120 appears in that data only as a *multi-GPU* configuration (`TP 8 DCP 8` / `TP 16 DCP 16`), never at TP=1.

Queries that returned nothing usable for these gaps, verbatim as executed:
- `https://api.github.com/search/issues?q=repo:sgl-project/sglang+context+parallel+is:issue+is:closed&per_page=40&sort=created&order=desc` → HTTP 403
- `https://api.github.com/search/issues?q=repo%3Avllm-project%2Fvllm+disagg+in%3Atitle+type%3Apr+is%3Aunmerged&per_page=40&sort=created&order=desc` → HTTP 403
- `https://api.github.com/rate_limit` → confirmed `"core": {"limit": 60, "remaining": 0}`
- Per the research sub-agent's report (declared as un-retrieved rather than filled): SGLang `disaggregation in:title` (×2), vLLM `label:wontfix`, vLLM `nixl in:title ... is:unmerged` — all HTTP 403; and `repo:vllm-project/vllm label:not-planned` returns `total_count 0` because that label does not exist in vLLM.
- `decode context parallel` as an exact phrase in the arXiv API returned 0 usable results (per sub-agent report); the workable term is "context parallelism" + KV cache decoding.

---


### 13. single-GPU agentic/long-context (`S13.md`)

No class came back empty. Coverage notes and shortfalls, stated honestly:

- **No ABANDONED row exists for "KV reuse across turns with changing system prompts" that is a clean maintainer rejection.** The best evidence found is SGLang PR #21064 (ABANDONED-6, closed unmerged by its author, with a measured 4.8k → 20.8k prefix-hit improvement left on the table) and vLLM issue #29286 (ABANDONED-9, stale-bot closed). Neither is a maintainer saying "we will not do this". Searches for a maintainer-stated rejection on this specific topic returned nothing retrievable.
- **No ABANDONED row for "shared-prefix KV deduplication within one engine" as a distinct feature.** The closest material is block-hash dedup, which is simply how vLLM/SGLang work by default (CLOSED-2, CLOSED-1), plus the block-size divisibility defect surfaced in ABANDONED-7. No one appears to have proposed and abandoned a separate within-engine dedup feature under that name in the retrieved corpus.
- **ABANDONED-19 (SGLang PR #20088) lacks a verbatim closure reason.** The page shows `Closed` with 17 unmerged commits but posts no closure comment that could be retrieved. It is flagged inline rather than silently passed off as a reasoned abandonment.
- **TRT-LLM PR #15828 (ABANDONED-12) likewise has no closure comment**; the verbatim quote supplied is the last substantive maintainer comment and is explicitly labelled as such, not as a closure reason.
- **SGLang PR #33315 ("[srt] Batchable context forwards over cached prefix KV", closed unmerged 2026-08-26) was DROPPED** from ABANDONED because no verbatim closure reason could be retrieved from the page. It is recorded here so the lead is not lost: https://github.com/sgl-project/sglang/pull/33315 (retrieved, READ BODY of the PR chrome; the author closed it with no comment).
- **vLLM #36311 ("[Feature Request] Pluggable KV cache eviction policy with attention sink protection")** was retrieved and is cited inside ABANDONED-1 as the source of the ref_cnt==0 lesson, but is not emitted as its own row because its closure state and reason were not independently confirmed on the page beyond the quoting of it by the ACE author.

---


### 14. correctness/observability (`S14.md`)

None. All four classes have at least one evidenced entry.

One sub-topic was thin and should be flagged rather than padded: I found no *retrieved, verbatim-quoted* artifact that closes the loop on "KV eviction changes the measured quality of a model" as a first-class, shipped **metric** (as opposed to a published audit). CLOSED-15/CLOSED-16 and ABANDONED-10/ABANDONED-11 are papers; the shipped engine metrics in CLOSED-4/CLOSED-5 measure *cache* behaviour (hit rate, block lifetime/reuse), not output-quality deltas attributable to eviction. I did not find a merged PR that adds an output-quality-vs-eviction metric to vLLM or SGLang, and I am not asserting one exists.

---

## Search log

Consolidated, de-duplicated list of the queries actually run across S1–S14 and of the query text recorded in the S1–S14 verification reports. Per-ID page retrievals (arXiv `/abs/`, `/html/`; individual GitHub issue/PR pages) are logged inside each segment file and are not repeated here.

### arXiv

- `KV+cache+fragmentation`
- `KV+cache+page+size`
- `heterogeneous+page+sizes+KV+cache`
- `Mamba+attention+hybrid+KV+cache+management`
- `block+manager+KV+cache+LLM+serving`
- `Marconi+KV+cache+admission+eviction`
- `hybrid+attention+SSM+KV+cache+allocator+page`
- `state+space+model+KV+cache+memory+management+serving`
- `prefix+cache+block+granularity+LLM+serving`
- `virtual+memory+KV+cache+GPU+LLM+inference`
- `abs:"KV cache" AND abs:"fragmentation"` (Atom API)
- `abs:"KV cache" AND abs:"page size"` (Atom API)
- `abs:"Mamba" AND abs:"KV cache"` (Atom API)
- `abs:"RadixAttention"` (Atom API)
- `abs:"hybrid" AND abs:"KV cache" AND abs:"memory"` (Atom API)
- `abs:"PagedAttention"` (Atom API)
- `prefix caching KV cache`
- `cache-aware+routing+prefix+KV+cache+LLM+serving`
- `prefix+cache+eviction+salting+LLM+inference`
- `hybrid+model+prefix+caching+Mamba+state+reuse`
- `prefix+cache+hit+rate+LLM+serving`
- `quantized+KV+cache+prefix+caching`
- `cross-instance+KV+cache+sharing+prefix`
- `prefix+caching+KV+cache+LLM+inference`
- `abs:"prefix caching" AND abs:"KV cache"` (Atom API)
- `all:"RadixAttention"` (Atom API)
- `abs:"prefix cache" AND abs:"LLM inference"` (Atom API)
- `abs:"KV cache" AND abs:"eviction"` (Atom API)
- `abs:"attention sink"` (Atom API)
- `abs:"SnapKV"` (Atom API)
- `abs:"KV cache" AND abs:"quantization"` (Atom API)
- `all:"KIVI"` (Atom API)
- `all:"KVQuant"` (Atom API)
- `all:"QAQ" AND all:"KV cache"` (Atom API)
- `abs:"FP8" AND abs:"KV cache"` (Atom API)
- `abs:"per-token" AND abs:"KV cache" AND abs:"quantization"` (Atom API)
- `multi-head latent attention`
- `KV cache merging`
- `KV cache distillation`
- `KV cache tensor decomposition`
- `KV cache codebook`
- `low-rank KV cache compression`
- `cross-layer KV sharing`
- `You Only Cache Once`
- `MiniCache`
- `latent attention KV compression`
- `Dynamic Memory Compression retrofitting`
- `all:"multi-head latent attention"` (Atom API)
- `abs:"latent KV"` (Atom API)
- `abs:"KV cache merging" OR abs:"KV merging"` (Atom API)
- `all:electron` (Atom API connectivity probe)
- `all:"KV connector"` (Atom API)
- `all:"KV cache-aware routing"` (Atom API)
- `all:"cache-aware load balancing"` (Atom API)
- `all:"LoRA" AND all:"KV cache" AND all:"sharing"` (Atom API)
- `all:"KV cache deduplication"` (Atom API)
- `all:"KV cache" AND all:"multicast"` (Atom API)
- `all:"NIXL"` (Atom API)
- `all:"cross-engine" AND all:"KV cache"` (Atom API)
- `all:"KV cache transfer"` (Atom API)
- `all:"KV cache reuse"` (Atom API)
- `all:"KV cache-aware"` (Atom API)
- `all:"KV cache" AND all:"deduplication"` (Atom API)
- `all:"KV cache" AND all:"broadcast"` (Atom API)
- `all:"LoRA" AND all:"KV cache"` (Atom API)
- `all:"cross-engine" AND all:"KV"` (Atom API)
- `ti:"Preble"` (Atom API)
- `abs:"chunked prefill"` (Atom API)
- `abs:"ring attention"` (Atom API)
- `abs:"context parallelism"` (Atom API)
- `abs:"KV cache" AND abs:"long-context"` (Atom API)
- `abs:"retrieval-augmented generation" AND abs:"KV cache"` (Atom API)
- `abs:"system prompt"` (Atom API)
- `abs:"long-context benchmark"` (Atom API)
- `abs:"prefill-decode" AND abs:"disaggregation"` (Atom API)
- `abs:"million tokens"` (Atom API)
- `abs:%22KV+cache%22+AND+abs:%22TTL%22` (Atom API)
- `abs:%22KV+cache%22+AND+abs:%22eviction%22` (Atom API)
- `abs:%22KV+cache%22+AND+abs:%22admission%22` (Atom API)
- `abs:%22KV+cache%22+AND+abs:%22tiered%22` (Atom API)
- `abs:%22KV+cache%22+AND+abs:%22scheduling%22` (Atom API)
- `abs:%22preemption%22+AND+abs:%22LLM%22` (Atom API)
- `abs:%22SLA%22+AND+abs:%22KV+cache%22` (Atom API)
- `abs:%22token-level%22+AND+abs:%22KV+cache%22` (Atom API)
- `abs:%22prefix+cache%22+AND+abs%22hit+rate%22` (Atom API)
- `abs:%22KV+cache%22+AND+abs:%22cost-aware%22` (Atom API)
- `abs:%22recompute%22+AND+abs:%22KV+cache%22` (Atom API)
- `abs:%22KV+cache%22+AND+abs:%22lifetime%22` (Atom API)
- `KV+cache+TTL+eviction` (arxiv.org/search, `start=0&size=50`)
- `abs:"disaggregated prefill"` (Atom API)
- `all:"prefill-decode disaggregation"` (Atom API)
- `abs:"layerwise KV"` (Atom API)
- `abs:"hidden states" AND abs:"disaggregat"` (Atom API)
- `abs:"layer-wise" AND abs:"KV cache"` (Atom API)
- `abs:"prefill-decode disaggregation"` (Atom API)
- `ti:"disaggregat" AND abs:"KV cache"` (Atom API)
- `ti:"KV cache" AND abs:"does not"` (Atom API)
- `abs:"KV cache" AND abs:"no speedup"` (Atom API)
- `ti:"rethinking" AND abs:"KV cache"` (Atom API)
- `KV+cache+eviction+does+not+outperform` (arxiv.org/search HTML)
- `KV cache eviction` (arxiv.org/search/advanced, terms-0-field=all, size=25)
- `abs:"KV cache" AND abs:"agentic"` (Atom API)
- `abs:"KV cache"` (Atom API test)
- `abs:"prefix caching" AND abs:"agent"` (Atom API, timed out)
- `abs:"prefix cache" AND abs:"multi-turn"` (Atom API, no file produced)
- `KV cache determinism LLM inference` (arxiv.org/search UI)
- `prefix cache collision` (arxiv.org/search UI)
- `prefix cache hash collision LLM serving` (arxiv.org/search UI)
- `KV cache offloading disk SSD reuse prefill` (arxiv.org/search UI)
- `KV cache offloading SSD` (arxiv.org/search UI)
- `reproducibility nondeterminism LLM inference serving` (arxiv.org/search UI)
- `KV cache metrics observability serving` (arxiv.org/search UI)
- `KV cache eviction evaluation protocol flawed` (arxiv.org/search UI, no results)
- `prefix cache hit rate measurement LLM` (arxiv.org/search UI, no parseable results)
- `KV cache eviction benchmark quality evaluation` (arxiv.org/search UI, no parseable results)
- `abs:"KV cache" AND abs:"determinism"` (Atom API)

Query list not recorded for this segment: S5 has no arXiv *search* queries — it records only per-ID arXiv retrievals (each `/abs/` page title-matched). S12 and its verification report record no arXiv query of their own; only a sub-agent's four Atom-API sweeps, all reported as HTTP 429. S7, S10 and S14 verification reports re-attempted the Atom API/`/search` endpoints (see the reachability line below) without logging new query strings.

### GitHub issue / PR search

- `is:pr is:unmerged label:kv-cache-manager repo:vllm-project/vllm`
- `is:issue is:closed label:kv-cache-manager repo:vllm-project/vllm`
- `is:issue is:closed reason:"not planned" kv cache repo:vllm-project/vllm`
- `is:pr is:unmerged "page size" repo:vllm-project/vllm`
- `is:pr is:unmerged "block size" repo:vllm-project/vllm`
- `is:issue is:closed reason:"not planned" kv cache repo:sgl-project/sglang`
- `is:pr is:unmerged "memory pool" repo:sgl-project/sglang`
- `is:pr is:closed is:unmerged kv cache block repo:vllm-project/vllm`
- `repo:vllm-project/vllm "prefix caching" is:issue is:open`
- `repo:vllm-project/vllm "prefix caching" is:pr is:unmerged`
- `repo:vllm-project/vllm "prefix caching"`
- `repo:vllm-project/vllm "prefix cache" is:issue is:open`
- `repo:vllm-project/vllm prefix caching in:title`
- `repo:vllm-project/vllm "cache_salt"`
- `repo:sgl-project/sglang "prefix cache" is:issue`
- `repo:sgl-project/sglang "radix cache"`
- `repo:sgl-project/sglang prefix cache in:title`
- `repo:vllm-project/vllm prefix caching quantization`
- `repo:vllm-project/vllm prefix caching eviction`
- `repo:vllm-project/vllm ReplaySSM in:title`
- `repo:vllm-project/vllm "fine-grained SWA hits"`
- `repo:vllm-project/vllm "internal prefill checkpoints"`
- `repo:vllm-project/vllm prefix cache is:unmerged is:closed`
- `repo:vllm-project/vllm prefix cache label:stale`
- `repo:LMCache/LMCache prefix caching`
- `repo:ai-dynamo/dynamo "prefix cache" OR "kv router"`
- `repo:vllm-project/vllm prefix caching stale`
- `repo:vllm-project/vllm cache_salt in:title`
- `repo:vllm-project/vllm "fine-grained" hybrid lookup merged`
- `repo:ai-dynamo/dynamo "kv router" in:title`
- `repo:vllm-project/vllm prefix caching is:pr is:merged salt`
- `repo:vllm-project/vllm prefix cache is:closed label:stale`
- `repo:sgl-project/sglang "prefix cache" is:closed`
- `repo:ai-dynamo/dynamo kv-router is:pr is:merged`
- `repo:NVIDIA/TensorRT-LLM prefix cache kv reuse`
- `repo:vllm-project/vllm eviction in:title type:pr is:closed is:unmerged` (recorded as "query rejected"; re-run by the verifier and it succeeds)
- `repo:sgl-project/sglang eviction in:title type:pr is:closed is:unmerged`
- `repo:NVIDIA/TensorRT-LLM eviction in:title type:pr is:closed is:unmerged`
- `repo:vllm-project/vllm eviction type:issue is:open`
- `repo:sgl-project/sglang "eviction policy" type:issue is:open`
- `repo:vllm-project/vllm "sparse KV" type:issue is:open`
- `repo:vllm-project/vllm eviction type:issue is:closed reason:not planned`
- `repo:vllm-project/vllm "attention sink" type:issue is:closed reason:not planned`
- `repo:sgl-project/sglang "eviction policy" type:pr is:closed is:unmerged`
- `repo:vllm-project/vllm turboquant`
- `repo:vllm-project/vllm "kv cache" quantization in:title`
- `repo:vllm-project/vllm fp8 kv cache accuracy`
- `repo:vllm-project/vllm "prefix caching" kv cache quantization`
- `repo:vllm-project/vllm KIVI`
- `repo:vllm-project/vllm "per-channel" key quantization KV`
- `repo:vllm-project/vllm "kv-cache-dtype" "prefix caching"`
- `repo:vllm-project/vllm quantized kv cache "prefix caching" in:title`
- `repo:sgl-project/sglang kv cache quantization fp8`
- `repo:sgl-project/sglang kv cache quantization in:title`
- `repo:sgl-project/sglang fp8 kv cache`
- `repo:vllm-project/vllm hicache is:issue is:open`
- `repo:sgl-project/sglang hicache is:issue is:open`
- `repo:vllm-project/vllm "cpu offload" is:pr is:closed is:unmerged`
- `repo:vllm-project/vllm "cpu offload" is:issue is:closed`
- `repo:LMCache/LMCache is:pr is:closed is:unmerged`
- `repo:vllm-project/vllm swap_space is:issue is:closed`
- `repo:vllm-project/vllm swap_space is:pr`
- `repo:vllm-project/vllm OffloadingConnector is:pr is:merged`
- `repo:vllm-project/vllm "secondary tier" is:pr`
- `repo:sgl-project/sglang hicache is:issue is:closed label:"not planned"`
- `repo:vllm-project/vllm is:issue is:closed label:"not planned"`
- `repo:vllm-project/vllm is:issue is:closed state:not_planned offload`
- `repo:vllm-project/vllm "disk" "kv cache" is:issue is:closed`
- `repo:vllm-project/vllm NixlConnector is:pr is:merged`
- `repo:sgl-project/sglang is:issue is:closed label:"not planned" cache`
- `repo:sgl-project/sglang hicache is:pr is:closed is:unmerged`
- `repo:vllm-project/vllm is:issue label:"not planned" kv`
- `repo:vllm-project/vllm TieringOffloadingSpec is:pr`
- `repo:vllm-project/vllm MLA absorption`
- `repo:vllm-project/vllm MLA in:title`
- `repo:vllm-project/vllm "KV cache" compression in:title`
- `repo:vllm-project/vllm MLA latent`
- `repo:vllm-project/vllm low-rank KV reason:"not planned"`
- `repo:vllm-project/vllm Palu OR MiniCache OR YOCO`
- `repo:vllm-project/vllm "KV merging"`
- `repo:vllm-project/vllm cross-layer KV`
- `repo:vllm-project/vllm MiniCache`
- `repo:vllm-project/vllm cross-layer KV cache`
- `repo:vllm-project/vllm MLA closed:>2025-06-01 reason:"not planned"`
- `repo:vllm-project/vllm "KV cache" reason:"not planned"`
- `repo:vllm-project/vllm YOCO`
- `repo:sgl-project/sglang MLA reason:"not planned"`
- `repo:vllm-project/vllm TransMLA OR MHA2MLA OR upcycle`
- `repo:vllm-project/vllm "KV merging" OR "cache merging"`
- `repo:vllm-project/vllm codebook OR "vector quantization" KV`
- `repo:sgl-project/sglang cross-layer OR MiniCache OR YOCO`
- `repo:vllm-project/vllm "Dynamic Memory Compression" OR DMC`
- `repo:vllm-project/vllm "KV cache compression" is:open`
- `repo:sgl-project/sglang "absorb" MLA`
- `repo:vllm-project/vllm TurboQuant merged`
- `repo:vllm-project/vllm label:wontfix`
- `repo:vllm-project/vllm MLA is:unmerged is:pr`
- `repo:vllm-project/vllm "KV cache" is:unmerged is:pr`
- `repo:sgl-project/sglang "KV cache" is:unmerged is:pr`
- `repo:vllm-project/vllm "kv cache" "not planned"`
- `repo:vllm-project/vllm "kv cache" label:wontfix`
- `repo:vllm-project/vllm "kv cache" RFC`
- `repo:LMCache/LMCache "dedup"`
- `repo:ai-dynamo/dynamo "KV-aware"`
- `repo:vllm-project/vllm "kv connector" is:pr is:closed`
- `repo:vllm-project/vllm "deduplication"`
- `repo:vllm-project/vllm "KV cache" sharing replicas`
- `repo:sgl-project/sglang "hicache"`
- `repo:sgl-project/sglang "cache-aware"`
- `is:pr is:unmerged "kv connector"` (vllm-project/vllm, HTML issue search)
- `is:pr is:unmerged "kv cache" in:title`
- `is:pr is:unmerged mooncake`
- `is:pr is:unmerged nixl`
- `is:pr is:unmerged lmcache`
- `is:issue is:closed reason:"not planned" kv cache`
- `is:pr revert kv cache`
- `lora prefix cache sharing`
- `is:pr is:closed is:unmerged "kv"`
- `is:pr is:closed is:unmerged "connector"`
- `is:issue is:closed reason:"not planned" "disagg"`
- `is:issue is:closed reason:"not planned" "routing"`
- `kv cache broadcast OR multicast`
- `cross-engine OR "cross engine" kv`
- `deduplicate OR dedup kv cache`
- `lora kv cache reuse`
- `"cross-replica" OR "cross replica" OR "across replicas"`
- `is:issue is:closed "by design" kv`
- `is:issue is:closed "share the KV cache" OR "KV cache sharing"`
- `is:pr is:closed is:unmerged hicache OR cache` (sgl-project/sglang)
- `is:issue is:closed reason:"not planned" hicache OR radix OR cache`
- `cache-aware routing OR router kv`
- `multicast OR broadcast kv cache`
- `is:pr is:closed is:unmerged dedup OR sharing OR multicast` (LMCache/LMCache)
- `is:issue is:closed reason:"not planned"` (LMCache/LMCache)
- `cross-engine OR cross engine OR multicast` (LMCache/LMCache)
- `is:issue is:closed reason:"not planned" kv` (ai-dynamo/dynamo)
- `kv cache sharing OR reuse across instances` (NVIDIA/TensorRT-LLM)
- `prompt cache sharing across slots OR replicas` (ggml-org/llama.cpp)
- `is:issue is:closed reason:"not planned" long context`
- `is:issue is:closed reason:"not planned" KV cache`
- `is:issue is:closed reason:"not planned" RFC`
- `is:issue is:closed reason:"not planned" prefill`
- `is:issue is:closed reason:"not planned" 1M OR million`
- `is:issue is:closed "stale" KV cache long context`
- `is:issue "attention sink"`
- `is:issue "system prompt" cache`
- `is:issue prefix cache bloat OR "cache bloat"`
- `is:issue is:open "long context"`
- `is:issue is:open "sparse attention" long context`
- `is:issue "RFC" KV cache`
- `long prefill token threshold`
- `long-prefill-token-threshold`
- `is:pr is:closed is:unmerged KV`
- `is:pr is:closed is:unmerged "context parallel"`
- `is:pr is:closed is:unmerged ring`
- `is:pr is:closed is:unmerged attention sink`
- `is:pr is:closed is:unmerged chunked prefill` (vllm-project/vllm)
- `is:pr "attention sink"`
- `is:pr "long-prefill-token-threshold"`
- `context parallel OR ring attention` (sgl-project/sglang)
- `is:issue is:closed reason:"not planned" context parallel`
- `is:issue is:open context parallel`
- `is:issue RFC KV cache`
- `is:issue is:open long context KV`
- `is:pr is:closed is:unmerged KV cache`
- `is:pr is:closed is:unmerged chunked prefill` (sgl-project/sglang)
- `is:pr is:closed is:unmerged long context`
- `is:issue is:closed reason:"not planned" KvCacheRetentionConfig OR retention` (NVIDIA/TensorRT-LLM, 0 results)
- `repo:vllm-project/vllm+"chunked prefill"+in:title`
- `repo:vllm-project/vllm+label:wontfix` (total_count 0 — vLLM has no such label)
- `repo:vllm-project/vllm+is:pr+is:closed+is:unmerged+"long context"`
- `repo:vllm-project/vllm+"context parallel"+in:title`
- `repo:sgl-project/sglang+is:issue+"context parallel"`
- `repo:vllm-project/vllm+is:issue+is:closed+%22not+planned%22+%22long+context%22`
- `repo:vllm-project/vllm kv_block_lifetime`
- `repo:vllm-project/vllm preemption swap`
- `repo:vllm-project/vllm TTL in:title,body`
- `repo:vllm-project/vllm "TTL" kv in:title,body`
- `repo:vllm-project/vllm eviction in:title`
- `repo:vllm-project/vllm label:wontfix kv`
- `repo:vllm-project/vllm eviction policy in:title`
- `repo:vllm-project/vllm "swap space" V1`
- `repo:vllm-project/vllm swap preemption deprecated`
- `repo:vllm-project/vllm kv-cache-metrics in:title,body`
- `repo:vllm-project/vllm "cache hit rate" in:title,body`
- `repo:vllm-project/vllm preemption in:title`
- `repo:sgl-project/sglang hicache in:title`
- `repo:ggml-org/llama.cpp "context shift"`
- `repo:ggml-org/llama.cpp cache eviction OR TTL in:title`
- `repo:vllm-project/sglang KV cache OR eviction in:title` (validation error — wrong org; not retried)
- `vllm-project/vllm/pulls?q=is:pr kv cache eviction policy`
- `vllm-project/vllm/pulls?q=is:pr kv cache residency metrics`
- `vllm-project/vllm/pulls?q=is:pr preemption`
- `vllm-project/vllm/issues?q=is:issue kv ttl`
- `ggml-org/llama.cpp/issues?q=is:issue context shift`
- `ggml-org/llama.cpp/pulls?q=is:pr context shift`
- `sgl-project/sglang/pulls?q=is:pr hicache`
- `repo:vllm-project/vllm disaggregation in:title label:wontfix` (total 0)
- `repo:vllm-project/vllm "layerwise"`
- `repo:vllm-project/vllm nixl state:closed reason:not_planned`
- `repo:vllm-project/vllm "kv transfer" failure`
- `repo:sgl-project/sglang disaggregation state:closed reason:not_planned`
- `repo:LMCache/LMCache layerwise`
- `repo:vllm-project/vllm "hidden states" disaggregation`
- `repo:vllm-project/vllm "hidden states" transfer state:closed reason:not_planned`
- `repo:vllm-project/vllm "prefill/decode ratio" OR "prefill-decode ratio"`
- `repo:ai-dynamo/dynamo disaggregation state:closed reason:not_planned`
- `is:issue+disaggregation+reason:not+planned` (vllm-project/vllm HTML issue search — zero parsable links)
- `repo:vllm-project/vllm "context parallel" in:title`
- `repo:vllm-project/vllm disagg in:title type:pr is:unmerged`
- `repo:sgl-project/sglang context parallel is:issue is:closed`
- `repo:vllm-project/vllm "kv cache"+wontfix`
- `repo:vllm-project/vllm "kv cache"+"not planned"`
- `repo:vllm-project/vllm is:closed reason:"not planned" "kv cache"`
- `repo:vllm-project/vllm is:closed reason:"not planned" "prefix caching"`
- `repo:vllm-project/vllm is:pr is:closed is:unmerged "kv cache"`
- `repo:vllm-project/vllm is:pr is:merged revert kv`
- `repo:vllm-project/vllm "kv cache"+revert`
- `repo:vllm-project/vllm "kv cache"+reverted`
- `repo:vllm-project/vllm is:issue "kv cache" "no benefit"`
- `repo:vllm-project/vllm is:issue "prefix caching" "not worth"`
- `repo:vllm-project/vllm is:closed reason:"not planned" "kv offload"`
- `repo:vllm-project/vllm is:closed reason:"not planned" "kv cache quantization"`
- `repo:sgl-project/sglang is:closed reason:"not planned" "kv cache"`
- `repo:sgl-project/sglang is:closed reason:"not planned" "cache"`
- `repo:sgl-project/sglang is:pr is:closed is:unmerged "kv cache"`
- `repo:sgl-project/sglang is:pr is:merged revert`
- `repo:sgl-project/sglang "kv cache"+revert`
- `repo:sgl-project/sglang "radix cache"+wontfix`
- `repo:sgl-project/sglang is:issue "hicache"`
- `repo:sgl-project/sglang is:issue "kv cache" "no benefit"`
- `repo:ai-dynamo/dynamo is:closed reason:"not planned" kv`
- `repo:ai-dynamo/dynamo is:issue is:closed kv offload`
- `repo:ai-dynamo/dynamo is:issue wontfix`
- `repo:ai-dynamo/dynamo is:pr is:closed is:unmerged revert`
- `repo:ai-dynamo/dynamo in:comments "won't fix"`
- `repo:ai-dynamo/dynamo in:comments "no benefit"`
- `repo:ai-dynamo/dynamo in:comments "out of scope"`
- `repo:ai-dynamo/dynamo kvbm offload`
- `repo:ai-dynamo/dynamo is:issue is:closed "by design"`
- `repo:ai-dynamo/dynamo KVBM deprecat`
- `repo:ai-dynamo/dynamo "KVBM is no longer supported"`
- `repo:ai-dynamo/dynamo in:comments "no longer supported"`
- `repo:LMCache/LMCache is:closed reason:"not planned"`
- `repo:LMCache/LMCache is:issue is:closed kv cache offload`
- `repo:LMCache/LMCache is:issue wontfix`
- `repo:LMCache/LMCache in:comments "no benefit"`
- `repo:LMCache/LMCache in:comments reverted`
- `repo:LMCache/LMCache compression accuracy`
- `repo:LMCache/LMCache blend`
- `repo:LMCache/LMCache is:issue is:closed "does not help"`
- `repo:NVIDIA/TensorRT-LLM is:closed reason:"not planned" "kv cache"`
- `repo:NVIDIA/TensorRT-LLM is:issue "kv cache" wontfix`
- `repo:NVIDIA/TensorRT-LLM "kv cache" "not planned"`
- `repo:NVIDIA/TensorRT-LLM kv cache revert`
- `repo:NVIDIA/TensorRT-LLM "kv cache" "out of scope"`
- `repo:NVIDIA/TensorRT-LLM kv cache "no benefit"`
- `repo:NVIDIA/TensorRT-LLM quantization kv accuracy`
- `repo:NVIDIA/TensorRT-LLM is:issue is:closed label:"wontfix"`
- `repo:NVIDIA/TensorRT-LLM is:pr is:closed is:unmerged "kv cache" created:2025-06-01..2026-09-13`
- `repo:NVIDIA/TensorRT-LLM is:issue is:closed "kv cache" created:2025-06-01..2026-09-13`
- `repo:NVIDIA/TensorRT-LLM "kv cache" "won't fix"`
- `repo:NVIDIA/TensorRT-LLM is:pr is:closed is:merged created:2026-09-08..2026-09-09 "kv cache"` (control test)
- `repo:NVIDIA/TensorRT-LLM is:pr is:closed is:unmerged "kv cache"` (verifier re-run; 40/40 sampled items had `merged_at=null`)
- `repo:NVIDIA/TensorRT-LLM is:pr is:closed is:merged "kv cache"` (verifier re-run)
- `repo:ggml-org/llama.cpp is:closed reason:"not planned" "kv cache"`
- `repo:ggml-org/llama.cpp is:issue "cache reuse"`
- `repo:ggml-org/llama.cpp is:pr is:closed is:unmerged "kv cache"`
- `repo:ggml-org/llama.cpp "kv cache" "no benefit"`
- `repo:ggml-org/llama.cpp "kv cache" "not worth"`
- `repo:ggml-org/llama.cpp is:pr+25494 in:comments "no benefit"` (verification)
- `repo:huggingface/text-generation-inference is:closed reason:"not planned" cache`
- `repo:huggingface/text-generation-inference is:issue "prefix caching"`
- `repo:huggingface/text-generation-inference "kv cache" "not planned"`
- `repo:huggingface/text-generation-inference "kv cache" offload`
- `repo:huggingface/text-generation-inference is:pr is:closed is:unmerged "kv cache"`
- `repo:vllm-project/vllm+"prefix caching"+agentic`
- `repo:vllm-project/vllm+"multi-turn"`
- `repo:sgl-project/sglang+"prefix caching"`
- `repo:vllm-project/vllm+"prefix caching"+"not planned"`
- `repo:vllm-project/vllm+54607`
- `repo:vllm-project/vllm+determinism+in:title`
- `repo:vllm-project/vllm "prefix caching" batch invariant`
- `repo:vllm-project/vllm "prefix caching" deterministic in:title`
- `repo:vllm-project/vllm is:pr is:closed is:unmerged "prefix cach"`
- `repo:sgl-project/sglang "radix cache" deterministic`
- `repo:sgl-project/sglang "cache hit rate" in:title`
- `repo:vllm-project/vllm "hash collision"`
- `repo:vllm-project/vllm "kv cache" corruption`
- `repo:ggml-org/llama.cpp is:pr is:closed is:unmerged "slot save"`
- `repo:vllm-project/vllm "batch invarian" is:pr is:closed is:unmerged`
- `repo:sgl-project/sglang "deterministic" in:title is:issue`
- `repo:vllm-project/vllm "kv cache" metrics in:title`
- `repo:sgl-project/sglang "kv cache" corruption`
- `repo:vllm-project/vllm "block hash" collision`
- `repo:vllm-project/vllm is:pr is:closed is:unmerged "batch invarian"`
- `repo:vllm-project/vllm "cache hit rate" in:title`
- `repo:vllm-project/vllm "kv cache" revert`
- `https://api.github.com/search/issues?q=repo:vllm-project/vllm+%22hybrid+KV+cache%22`
- `https://api.github.com/search/issues?q=repo:sgl-project/sglang+radix+cache+memory+pool`
- `https://api.github.com/search/issues?q=repo:vllm-project/vllm+%22kv+cache%22+%22page+size%22`
- `https://api.github.com/search/issues?q=repo:vllm-project/vllm+%22block+size%22+kv+cache`
- `https://api.github.com/search/issues?q=repo:vllm-project/vllm+mamba+kv+cache+page`
- `https://api.github.com/repos/vllm-project/vllm/issues/46462/comments`
- `https://api.github.com/repos/vllm-project/vllm/issues/{46841,4762,53178}/comments`
- `https://api.github.com/repos/vllm-project/vllm/issues/54386/comments`
- `https://api.github.com/repos/vllm-project/vllm/pulls/49574`
- `https://api.github.com/repos/vllm-project/vllm/pulls/43729`
- `https://api.github.com/rate_limit` (used in S5, S10, S11, S12 and by the S5/S10/S12 verifiers)

Query list not recorded for this segment: none — every segment that ran GitHub search logged its queries. S6's REST-API section and S13's cached-corpus section record additional candidate-discovery work whose strings are the API queries listed above or local file scans.

### Engine docs and source greps

- `block_size` across `vllm/v1/core/` and `vllm/config/*.py`
- `grep -rn "TODO\|FIXME\|XXX\|NOTE:" vllm/v1/core/*.py`
- `HybridKVCacheCoordinator|KVCacheCoordinator|class .*Coordinator` in `vllm/v1/core/kv_cache_coordinator.py`
- `page_size|page_size_padded|page_size_bytes|unpadded_page_size_bytes` in `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/kv_cache_interface.py`
- `mamba_block_size` across `vllm/` (including `vllm/config/cache.py`, `vllm/config/vllm.py`, `vllm/engine/arg_utils.py`)
- `NotImplementedError` context in `vllm/v1/core/kv_cache_utils.py`
- `_try_get_full_allocation_fallback_groups` body
- `FIXME(Chen)` context in `vllm/v1/core/kv_cache_utils.py`
- `VLLM_KV_CACHE_LAYOUT` across `vllm/` and `docs/`
- `deprecat` in `vllm/config/cache.py`, `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/kv_cache_coordinator.py`
- `marconi` (case-insensitive) across the vLLM tree
- `admission` in `vllm/v1/core/*.py`
- `block_size must|supported block size|block_size not` across `vllm/`
- `grep -rn "enable_prefix_caching" --include=*.py --include=*.md .`
- `grep -rln "prefix cach" docs/`
- `grep -rn "TODO|FIXME|NotImplementedError|not supported" vllm/v1/core/kv_cache_manager.py vllm/v1/core/kv_cache_utils.py`
- `grep -rn "cache_salt" --include=*.py vllm/`
- `grep -rn "kv_cache_dtype_skip_layers|kv-cache-dtype-skip-layers"`
- `grep -rn "prefix_cache" vllm/v1/core/kv_cache_metrics.py vllm/v1/metrics/loggers.py`
- `grep -rli "prefix cach|radix" --include=*.py --include=*.md --include=*.mdx --include=*.sh .` (SGLang partial tree)
- `grep -rn -i "prefix cach|radix" docs/cookbook/autoregressive/Qwen/Qwen3-Next.mdx`
- `grep -rniE "evict" --include=*.py vllm/`
- `grep -rniE "evict" --include=*.py python/` (SGLang tree — no hits, `python/` absent)
- `grep -rniE "attention_sink|attention sink|h2o|streamingllm|snapkv|pyramidkv|scissorhand|razorattention|ada-kv|adakv" --include=*.py --include=*.md --include=*.rst /tmp/kvsrc/vllm-main`
- `grep -rniE "class BlockPool|free_block_queue|evict" vllm/v1/core/block_pool.py`
- `grep -rniE "eviction_policy|cache_policy" docs/`
- `grep -rniE "TODO|FIXME|XXX" vllm/v1/core/block_pool.py`
- `grep -rn "cpu_offload" --include=*.py --include=*.md --include=*.rst .`
- `ls -R vllm/distributed/kv_transfer`
- `grep -rni "hicache" --include=*.py --include=*.md .`
- `grep -rn "cpu_offload_gb\|cpu-offload-gb" --include=*.py vllm/`
- `grep -rln "offload" docs/`
- `sed -n '1,80p' vllm/distributed/kv_transfer/README.md`
- `sed -n '200,270p' vllm/distributed/kv_transfer/kv_connector/factory.py`
- `sed -n '1,90p' vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py`
- `sed -n '1,70p' vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`
- `grep -rn "disk_path\|disk_capacity_bytes\|use_page_cache\|disk_buffer_slots" vllm/v1/kv_offload/ vllm/v1/simple_kv_offload/`
- `grep -rn "TODO\|FIXME\|XXX\|not supported\|unsupported\|by design" vllm/v1/kv_offload/ vllm/v1/simple_kv_offload/`
- `grep -rn "swap_space" --include=*.py --include=*.md .`
- `grep -rn "kv_offloading_size\|kv-offloading\|kv_offloading_backend" --include=*.py --include=*.md .`
- `grep -rni "gpu.?direct\|GDS_MT\|cufile" --include=*.py --include=*.md .`
- `grep -rn "kv_cache_dtype" --include=*.py /tmp/kvsrc/vllm-main`
- `grep -rn "FP8_KV_CACHE_DTYPES\|KV_CACHE_DTYPE\|fp8_e4m3\|fp8_e5m2" --include=*.py vllm/`
- `grep -rn "per_token_head_scale\|per-token-head\|per_token_scale\|per-channel\|per_channel" --include=*.py vllm/`
- `grep -rn "kv_cache_dtype\|kv_cache_quant" --include=*.py /tmp/kvsrc/sglang-main/`
- `grep -rn "kv_cache_dtype\|fp8_e4m3\|fp8_e5m2" /tmp/kvsrc/sglang-main/docs/`
- `grep -rin "kivi\|kvquant\|kv-quant" /tmp/kvsrc/vllm-main/vllm/ /tmp/kvsrc/vllm-main/docs/`
- `grep -rin "kivi\|kvquant" /tmp/kvsrc/sglang-main/`
- `grep -rn "fp8_e5m2" vllm/vllm/`
- `grep -rln "enable_prefix_caching" --include=*.py tests/ | xargs grep -ln "kv_cache_dtype"`
- `grep -rn "kv_cache_dtype" docs/features/automatic_prefix_caching.md`
- `grep -ril "mla\|multi_head_latent\|multihead_latent" --include=*.py .`
- `grep -rn "TODO\|FIXME" --include=*.py vllm/ | grep -i "mla\|latent\|low.rank\|compress\|merge\|yoco\|cross.layer"`
- `grep -rni "yoco\|minicache\|palu\|cross_layer_kv\|kv_merg\|mla_absorb\|absorb" --include=*.py --include=*.md .`
- `grep -rn "kv_sharing_fast_prefill" --include=*.py --include=*.md .`
- `grep -rn -i "MLA\|latent attention" docs/design/attention_backends.md`
- `grep -rn -i "not supported\|unsupported\|does not support" docs/ | grep -i "mla\|latent\|kv sharing\|sliding"`
- `grep -rhoE "https://github\.com/(vllm-project/vllm|sgl-project/sglang)/(issues|pull)/[0-9]+"` over the whole vLLM tree
- `grep -rni "mla\|latent_attention\|kv_lora" --include=*.py python/` (SGLang partial tree — nothing returned)
- `grep -rn "register_connector\|KVConnectorFactory" --include=*.py vllm/`
- `sed -n '145,250p' vllm/distributed/kv_transfer/kv_connector/factory.py`
- `grep -rn "TODO\|FIXME\|XXX\|deprecated\|NotImplementedError\|does not support" --include=*.py vllm/distributed/kv_transfer/`
- `grep -rn "kv_events\|KVEvent\|publish_kv_events\|zmq" --include=*.py vllm/config/ vllm/distributed/kv_events*`
- `grep -rn "lora" --include=*.py vllm/v1/core/kv_cache_utils.py`
- `grep -rni "lora" --include=*.py vllm/v1/core/sched/scheduler.py`
- `sed -n '535,625p' vllm/v1/core/kv_cache_utils.py`
- `sed -n '60,130p' vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py`
- `find docs -iname "*kv*" -o -iname "*disagg*"`
- `find /tmp/kvsrc/sglang-main -iname "*hicache*" -o -iname "*hierarchical*"`
- `find /tmp/kvsrc/sglang-main -iname "*mem_cache*" -o -iname "*cache_controller*"`
- `enable_chunked_prefill` in `vllm/config/scheduler.py`, `vllm/config/vllm.py`, `vllm/engine/arg_utils.py`
- `long_prefill_token_threshold` across `vllm/`, `tests/`, `docs/`
- `prefix_match_unit` / `enable_mamba_fine_grained_prefix_cache` across `vllm/`
- `decode_context_parallel_size` / `dcp_*` / `prefill_context_parallel_size` / `-pcp` / `-dcp` in `vllm/engine/arg_utils.py`
- `context_parallel` file list across `vllm/`
- `ring` across `vllm/`
- `attention_sink` / `sinks` across `vllm/` and `csrc/`
- `TODO|FIXME|NOTE:` in `vllm/v1/attention/ops/dcp.py`, `pcp.py`, `vllm/v1/worker/gpu/pcp_manager.py`
- `not supported|not implemented|unsupported|limitation` in the same files
- `grep -rn "prefill_context_parallel_size" vllm/config/parallel.py`
- `grep -rn -E "dcp|decode_context_parallel" vllm/config/*.py vllm/engine/arg_utils.py`
- `grep -rn -i -E "nvlink|infiniband|rdma|multi-node|multinode|multiple GPUs|at least 2|at least two|2 GPUs|two GPUs|more than one GPU" docs/ --include=*.md`
- `find /tmp/kvsrc/vllm-main -iname "*context_parallel*" -o -iname "*disagg*"`
- `grep -ril "context parallel" docs/`
- `grep -rn -i -E "requirements|must have|at least|minimum|only support" docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`
- `grep -rn -i "zigzag" docs/`
- `grep -rn -o -E "validated on [^.]{0,80}|on [0-9]+x GB300[^.]{0,40}|4x GB300|8x [A-Z0-9]+" docs/cookbook/`
- `grep -rn -i "hisparse" docs/`
- `find vllm/distributed/kv_transfer -name '*.py'`
- `grep -rn 'TODO\|FIXME\|XXX\|not supported\|unsupported\|NotImplemented' vllm/distributed/kv_transfer/ --include=*.py`
- `grep -rn 'kv_load_failure_policy' --include=*.py .`
- `grep -rn 'supports_hma_config\|SupportsHMA' --include=*.py vllm/`
- `grep -rn 'hybrid kv cache manager\|disable_hybrid_kv_cache_manager' vllm/config/vllm.py`
- `grep -rn 'layerwise\|layer-wise' --include=*.py --include=*.md --include=*.rst vllm/distributed/kv_transfer/ docs/`
- `grep -rln 'layerwise\|layer_wise\|per_layer\|layer-by-layer' --include=*.py vllm/` and `grep -rln 'layerwise\|layer_wise' sglang-main/`
- `grep -rn 'HiddenStates\|hidden_states' vllm/distributed/kv_transfer/kv_connector/factory.py`
- `sed -n '95,175p' vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py`
- `find . -iname '*disagg*' -o -iname '*pd_transfer*' -o -iname '*nixl*'` in `/tmp/kvsrc/sglang-main`
- `git log --oneline -2000 | grep -iE 'revert' | grep -iE 'kv|cache'`
- `grep -rniE "(no (measurable )?(benefit|speedup|gain)|does not help|doesn't help|not worth|won't (fix|help)|no longer pursu|out of scope|we decided against|reverted)" --include=*.py --include=*.md --include=*.cu --include=*.cuh --include=*.h --include=*.cpp .` filtered by `kv|cache|prefix|quant|offload|evict`
- `grep -rniE --include=*.md "(no (measurable |significant )?(speedup|benefit|gain|improvement)|does not (help|improve|pay)|not worth|hurts accuracy|accuracy (loss|drop)|overhead (cancels|outweighs)|negligible (benefit|gain)|not recommended|we (do not|don't) (recommend|support))" docs/`
- `find vllm/v1/kv_offload -name '*.py'`; `grep -rln --include=*.md -iE "kv.?offload|tiering" docs/`
- `grep -rniE "\bttl\b|time.to.live" --include=*.py vllm/`
- `grep -rn "PreemptionMode\|preemption_mode\|scheduling_policy\|SchedulingPolicy" --include=*.py vllm/`
- `grep -rn "swap_space\|swap-space\|num_cpu_blocks\|cpu_blocks" --include=*.py vllm/`
- `grep -rn "preempt" --include=*.py vllm/v1/core/sched/scheduler.py`
- `grep -rniE "eviction|LRU" docs/ --include=*.md`
- `grep -rniE "\bswap\b" docs/ --include=*.md`
- `grep -rn "kv_cache_metrics_sample\|kv-cache-metrics-sample" --include=*.py vllm/`
- `grep -rn "kv_lease_duration\|decoder_kv_blocks_ttl\|VLLM_NIXL_" --include=*.py vllm/`
- `grep -rn "PREFIX_CACHE_RETENTION_INTERVAL\|retention_interval" --include=*.py vllm/`
- `grep -rniE "hicache|hierarchical cache|kv_cache_ttl|ttl" --include=*.py .`
- `grep -rniE "eviction|TTL|time-to-live" docs/`
- `grep -rn "deterministic\|determinism" --include="*.py" --include="*.md" --include="*.rst" -il`
- `grep -rn "VLLM_BATCH_INVARIANT" --include="*.py" --include="*.md" --include="*.rst" .`
- `grep -rni "hash collision\|collision" --include="*.py" --include="*.md" vllm/ docs/`
- `grep -rni "dump.*kv\|kv.*dump\|save.*kv_cache\|load.*kv_cache" --include="*.py" --include="*.md" --include="*.rst" vllm/ docs/`
- `grep -rn "prefix_cache_hit\|prefix_cache_queries\|gpu_prefix_cache" --include="*.py" vllm/v1/metrics/ vllm/v1/core/`
- `grep -rn "prefix_cache" docs/design/metrics.md`
- `grep -rn "prefix-caching-hash-algo\|prefix_caching_hash_algo\|sha256_cbor\|xxhash_cbor" --include="*.py" --include="*.md" .`
- `grep -rn "batch_invariant\|BATCH_INVARIANT" --include="*.py" vllm/v1/core/ vllm/v1/engine/ vllm/config/`
- `ls -R vllm/v1/kv_offload/`
- `grep -rn "enable_prefix_caching|prefix_caching_hash_algo|--no-enable-prefix" vllm/config/cache.py`
- `grep -rni "agentic" --include=*.py --include=*.md -l .`
- `grep -rni "session_id|session-level" --include=*.py -l vllm/`
- `grep -rn "enable_mamba_fine_grained_prefix_cache|enable-mamba-fine-grained" vllm/ docs/`
- `grep -rn "prefix_match_unit" vllm/config/cache.py`
- `grep -il "agentic|multi-turn|tool.call|prefix cach|session" kv_neg/abs/*.html`
- `grep -n "### |title = " kv_discovery/closing_vllm.txt | grep -i "agentic|multi-turn|prefix|session|turn|tool|system prompt|conversation|reason|thinking|radix"` (repeated over `closing_sglang.txt`, `closing_trt.txt`)
- python scan of `kv_discovery/raw/*.json` for the keywords `agentic`, `multi-turn`, `tool call`, `tool-call`, `toolcall`, `prefix cach`, `prefix-cach`, `session`, `system prompt`, `conversation`, `reasoning`, `chain-of-thought`, `thinking`, `cross-turn`, `chat`, `radix`, `turn`, `claude`, `coding agent`, `agent`
- Docs/raw endpoints consulted for engine behaviour: `docs.vllm.ai/en/latest/design/prefix_caching/`, `docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/`, `docs.vllm.ai/en/latest/features/disagg_prefill.html`, `docs.vllm.ai/en/latest/features/nixl_connector_usage/`, `docs.vllm.ai/en/latest/features/nixl_connector_compatibility/`, `docs.vllm.ai/en/latest/features/kv_offloading_usage.html`, `docs.vllm.ai/en/latest/features/batch_invariance/`, `docs.vllm.ai/en/latest/design/metrics/`, `docs.vllm.ai/en/latest/usage/reproducibility/`, `docs.vllm.ai/en/latest/features/per_request_metrics/`, `docs.vllm.ai/en/latest/configuration/optimization/`, `docs.vllm.ai/en/latest/features/automatic_prefix_caching/`, `docs.vllm.ai/en/v0.28.0/features/quantization/quantized_kvcache/`, `docs.sglang.io/docs/advanced_features/{hicache,hicache_design,hicache_best_practices,hicache_storage_runtime_attach_detach,radix_eviction_policy,session_radix_cache,quantized_kv_cache,dcp,pd_disaggregation,deterministic_inference,observability}.md`, `docs.sglang.ai/advanced_features/{dcp,pd_disaggregation,dp_dpa_smg_guide,server_arguments,pipeline_parallelism,hicache_design}.html`, `docs.dynamo.nvidia.com/dynamo/v-0-8-1/user-guides/tuning-disaggregated-performance`, `docs.lmcache.ai/`, `raw.githubusercontent.com/vllm-project/vllm/main/docs/**`, `raw.githubusercontent.com/sgl-project/sglang/main/docs/**`, `raw.githubusercontent.com/LMCache/LMCache/dev/README.md`, `raw.githubusercontent.com/ai-dynamo/dynamo/main/README.md`, `raw.githubusercontent.com/llm-d/llm-d-kv-cache/main/README.md`, `raw.githubusercontent.com/scos-lab/turboquant/refs/heads/main/README.md`, `raw.githubusercontent.com/NVIDIA/kvpress/main/README.md`, `github.com/vllm-project/vllm/blob/main/**`, `lmsys.org/blog/2025-09-10-sglang-hicache/`, `vllm.ai/blog/*`, `llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d`
- Local mirrors inspected (read-only, greps above): `/tmp/kvsrc/vllm-main` (an older vLLM snapshot — the verifier establishes it is a ~0.11.2 tree, so its line numbers have drifted on live main) and `/tmp/kvsrc/sglang-main` (a partial checkout with only `3rdparty/ assets/ benchmark/ docker/ docs/ LICENSE README.md` — no `python/` sources)

### Web search engines

- `vLLM pull request closed unmerged KV cache block size allocator not planned`
- `vLLM issue wontfix hybrid KV cache manager mamba page size abandoned`
- `llama.cpp KV cache defragmentation removed defrag_thold deprecated`
- `vLLM PR closed unmerged block manager V1 KV cache allocator rejected`
- `SGLang RadixAttention memory pool issue closed not planned`
- `TensorRT-LLM KV cache block size paged removed`
- `vLLM issue 24280 decouple attention backend block size KVCacheManager`
- `vLLM prefix caching ineffective mamba hybrid block size 528 issue`
- `KV cache paged attention block size tuning benchmark 16 vs 32 vLLM`
- `llama.cpp "defrag" KV cache removed no longer needed issue`
- `llama.cpp pull request remove KV cache defragmentation`
- `vLLM PR closed without merging KV cache block allocator wontfix`
- `vLLM issue "not planned" KV cache block size hybrid mamba`
- `SGLang pull request closed unmerged radix cache memory pool allocator`
- `vLLM "wontfix" OR "not planned" KV cache block size allocator issue closed`
- `vLLM PR "closed" unmerged "KV cache" block manager revert`
- `llama.cpp KV cache defragmentation "not needed" removed unified cache`
- `prefix caching KV reuse LLM inference 2026 paper cache-aware routing`
- `prefix cache hit rate cross-instance KV cache sharing vLLM SGLang 2026`
- `vLLM KV cache eviction policy ARC LRU PR merged 2026`
- `KV cache eviction negative result does not improve accuracy paper 2026`
- `H2O heavy hitter KV cache eviction implementation rejected production engine`
- `SnapKV vLLM not merged PR KV cache eviction`
- `StreamingLLM attention sink vLLM SGLang implementation shipped`
- `Ada-KV RazorAttention implementation open source vLLM`
- `submodular KV cache eviction selection Lagrangian budget allocation paper`
- `query-agnostic KV cache selection paper 2026`
- `Scissorhands PyramidKV limitations negative results`
- `vLLM issue KV cache eviction not planned wontfix`
- `"KV cache" eviction accuracy degradation worse than full cache benchmark critique 2026`
- `attention score based KV eviction fails long context negative result`
- `QUEST query-aware KV cache selection paper`
- `KV cache eviction "does not" improve "future work" limitation 2026`
- `vLLM RFC "Sparse KV cache management framework" github issue`
- `vLLM RFC "Support KV Cache Compaction" github`
- `vLLM RFC sparse KV cache framework per-head eviction issue`
- `SGLang HiCache eviction policy LRU documentation`
- `TensorRT-LLM KV cache eviction attention sink gpt-oss implementation`
- `llama.cpp KV cache eviction context shift --cache-reuse discussion`
- `vLLM attention sink gpt-oss sinks shipped configuration`
- `KV cache eviction requires multi-GPU setup 8xA100 experiments H2O SnapKV`
- `"KV cache" eviction paper "8×A100" OR "8xA100" OR "4xA100" experimental setup 2026`
- `KV cache compression paper limitation "single GPU" not evaluated larger models`
- `NVIDIA kvpress KV cache compression library supported methods documentation`
- `vLLM PR attention sink support gpt-oss merged sink tokens`
- `LMCache compactor KV cache compaction status abandoned`
- `SGLang attention sink support PR merged gpt-oss`
- `RazorAttention retrieval head KV cache eviction 2026 follow-up`
- `Ada-KV adaptive budget allocation head-wise eviction NeurIPS 2025 result`
- `attention sink phenomenon explanation 2026 paper counterexample`
- `submodular KV cache selection budget allocation paper 2026 arxiv`
- `KV cache eviction attention sink evaluation requires A100 80GB multi-GPU experimental setup 2026 paper`
- `long context KV cache compression paper experiments 8 GPUs tensor parallel setup section`
- `KV cache eviction "two A100" OR "4 x A100" OR "8 x H100" long context evaluation`
- `KV cache eviction paper requires H100 Hopper only kernel FlashAttention-3`
- `"attention sink" KV cache paper 2026 requires multi-GPU NVLink evaluation`
- `KV cache compression survey open problems 2026 arxiv "From Tensor Buffer to Distributed Memory Hierarchy"`
- `vLLM disaggregated prefill decode KV eviction requires separate GPUs two instances`
- `Cloudflare kvcompress Ada-KV vLLM integration blog`
- `attention sink necessary or artifact paper 2026 negative result "sink" not needed`
- `TurboQuant Zandieh ICLR 2026 KV cache quantization arxiv`
- `QJL 1-bit KV cache quantization 2406.03482 attention quality`
- `vLLM kv-cache-dtype int4_per_token_head per-token-head scales`
- `"TurboQuant" arxiv KV cache quantization Hadamard Lloyd-Max 2025`
- `vLLM issue kv cache int4 quantization per token head accuracy`
- `QJL removed KV cache quantization hurts attention softmax variance`
- `vLLM blog TurboQuant KV cache quantization throughput 40 52% accuracy`
- `vLLM issue quantized KV cache prefix caching incompatible fp8`
- `KVarN variance-normalized KV cache quantization arxiv 2606.03458`
- `vLLM blog "The State of FP8 KV-Cache and Attention Quantization in vLLM"`
- `vLLM blog fp8 kv cache e4m3 e5m2 skip layers validation Hopper Blackwell`
- `Pensieve KV cache arXiv tiered storage LLM inference`
- `arXiv 2026 KV cache offloading host memory bandwidth bottleneck negative result`
- `"HyperOffload" KV cache offload scheduling arXiv 2026`
- `KV cache offloading NVMe SSD tiered 2026 arXiv single GPU`
- `LMCache vLLM CPU offload limitation "does not" future work 2026`
- `vllm github issue MiniCache cross-layer KV cache compression not planned`
- `vllm pull request Palu low-rank KV cache compression closed unmerged`
- `sglang github issue MLA absorption limitation wontfix`
- `H2O heavy hitter oracle KV cache arxiv 2306`
- `Quest query-aware sparsity KV cache arxiv 2406`
- `arxiv 2026 long context KV cache survey 1M tokens`
- `vLLM chunked prefill default enable v1 arxiv paper`
- `arxiv 2026 KV cache eviction does not help long context negative result`
- `arxiv 2026 long context KV cache compression accuracy loss limitation`
- `RULER benchmark KV cache compression fails at 128K arxiv 2025`
- `attention sink long context 2026 arxiv follow-up`
- `arXiv 2602 Vegas self-speculative decoding verification-guided sparse attention`
- `arxiv 2026 ring attention beyond single node bottleneck limitation negative`
- `arxiv 2026 context parallelism inference long context KV cache sharding`
- `arxiv 2026 chunked prefill 1M token inference serving`
- `arxiv 2026 attention sink does not help long context sliding window negative`
- `arxiv 2026 "attention sink" long context KV cache paper`
- `arxiv 2026 RAG many documents KV cache prefill cost`
- `vLLM chunked prefill default v1 PR merged which version`
- `arXiv 2026 KV cache TTL eviction scheduling LLM serving`
- `arXiv 2026 KV cache preemption recompute swap scheduling`
- `arXiv 2026 cost-aware KV cache retention SLA LLM serving`
- `arXiv 2026 negative result KV cache eviction does not improve prefix caching`
- `arXiv 2026 KV cache swap versus recompute preemption measurement`
- `arXiv 2026 token-level KV cache lifetime expiry per-token`
- `arXiv 2026 cache hit rate economics LLM inference cost per token`
- `KV cache admission control LLM serving arXiv 2026 memory pressure`
- `arXiv 2026 preemption policy LLM serving recompute cost victim selection`
- `arXiv 2026 KV cache tiered GPU CPU disk hierarchy single GPU offloading`
- `arxiv 2026 prefill decode disaggregation KV transfer layerwise`
- `hidden state transfer instead of KV cache transfer prefill decode disaggregation arxiv`
- `KV cache transfer failure handling disaggregated prefill vLLM issue`
- `NIXL vLLM KV connector disaggregated prefill 2026`
- `arXiv Prefill-as-a-Service KVCache Next-Generation Models Cross-Datacenter Qin`
- `layerwise KV cache transfer disaggregated prefill decode performance overhead paper 2026`
- `hidden state transfer vs KV cache transfer disaggregation bandwidth comparison paper`
- `prefill decode disaggregation does not help negative result overhead study paper`
- `NVIDIA Dynamo disaggregated serving prefill decode ratio tuning docs`
- `SGLang PD disaggregation documentation NIXL transfer`
- `vLLM NixlConnector compatibility matrix limitations`
- `vLLM KV cache offloading no speedup PCIe bandwidth bound negative result`
- `KV cache eviction H2O SnapKV does not outperform sliding window baseline paper`
- `KV cache quantization no end-to-end speedup memory bound decode paper 2026`
- `prefix caching no benefit criticism paper KV cache offload CPU not worth it 2026`
- `llama.cpp key-value cache quantization quality loss "not worth" flash attention issue`
- `TensorRT-LLM KV cache quantization accuracy regression issue closed not planned`
- `text-generation-inference prefix caching not supported wontfix issue`
- `"KV cache" offloading "no benefit" OR "not worth" single GPU PCIe bandwidth 2026 paper`
- `vLLM issue "we decided against" OR "out of scope" prefix caching kv cache offload`
- `"KV cache" eviction paper 2026 "no better than" LRU baseline negative result`
- `KV cache compression "does not" speed up decode memory-bound arxiv 2026 limitations`
- `Dynamo KVBM "no longer supported" removed kv block manager deprecation`
- `ai-dynamo KVBM deprecated removed kv cache offloading 2026`
- `LLM inference batch invariance non-determinism KV cache vLLM github issue`
- `prefix cache hash collision vLLM issue`
- `vLLM KV cache offload to disk filesystem tiering`
- `KV cache observability metrics hit rate vLLM SGLang`
- `vLLM issue "batch invariance" "prefix caching" not supported`
- `vLLM issue KV cache dump to disk save restore feature closed not planned`
- `SGLang issue deterministic batch invariant output prefix cache`
- `vLLM issue cache hit rate metric wrong misleading`
- `paper "KV cache" nondeterminism batch invariance 2026 arxiv`
- `arxiv 2026 "prefix caching" correctness collision LLM serving evaluation`
- `"cache hit rate" LLM serving metric flawed measurement paper 2026`
- `arxiv paper KV cache compression evaluation flawed baseline unfair comparison 2026`
- `vllm github issue prefix caching "by design" won't fix cache hit rate`
- `vllm pull request closed unmerged "kv cache" disk persist`
- `sglang issue deterministic inference radix cache still broken 2026`
- `vllm issue "kv cache" observability metrics misleading eviction quality benchmark`
- `arxiv paper 2026 negative result prefix caching does not help latency evaluation`
- `arxiv 2026 "KV cache" eviction evaluation "does not" improve accuracy measurement flaw`
- `sglang issue hierarchical cache disk HiCache correctness`
- `vllm issue kv cache events observability metrics missing`

Query list not recorded for this segment: S7, S12 and S13 do not record web-search query strings (S13 used a local cached corpus for discovery and re-fetched every cited item live; S7 and S12 used direct retrieval and GitHub/arXiv searches).

**Endpoint reachability as recorded in the evidence files.** The arXiv Atom API (`export.arxiv.org/api/query`) was **not reachable** — HTTP 429 (and HTTP 000/timeouts) on every attempt, in S1, S2, S3, S4, S6, S7, S8, S9, S10, S11, S13 and S14, and it still answered HTTP 429 during the S8/S13/S14 verification sessions. The arXiv HTML search endpoint (`arxiv.org/search/`) was **intermittently reachable**: S1 used it successfully; S2 got HTTP 200 on the first query and HTTP 429 on the next six; S11 saw HTTP 200 (and later 400/429 on the advanced-search path); S14 got 200 but several queries returned no parseable results; S9 recorded 429 at run time yet the verifier re-ran the same URL and got HTTP 200. arXiv `/list/` browse pages were **not reachable** (HTTP 404, e.g. `cs.DC/2606`, `cs.DC/2609`). The GitHub REST search API (`api.github.com/search/issues`) was **partially reachable**: many queries returned totals (S2, S5, S7, S8, S10, S11, S12, S14) but HTTP 403 "API rate limit exceeded" recurred across S2, S4, S5, S6, S7, S8, S9, S11, S12, S13 and S14 — the S6 verifier notes at least one REST search did succeed (a cached `label:wontfix → total_count 0` response), and the S12 verifier reproduced a 403 with `api.github.com/rate_limit` showing `core: {limit: 60, remaining: 0}`. The GitHub REST **core** endpoints (`/repos/.../issues/N/comments`, `/repos/.../pulls/N`) were **not reachable** (HTTP 403 throughout S1, S10, S12). GitHub **HTML** pages (`github.com/...` issue, PR and search pages, plus `.patch`/`.diff` endpoints) were **reachable** (HTTP 200) throughout, and are how issue bodies, timelines and closure events were read. Engine documentation hosts (`docs.vllm.ai`, `docs.sglang.ai`/`docs.sglang.io`, `docs.dynamo.nvidia.com`, `docs.lmcache.ai`) and `raw.githubusercontent.com` were **reachable**, with specific **404s** recorded for `docs.vllm.ai/llms.txt`, `docs.sglang.io/backend/server_arguments.html`, `docs.sglang.io/.../hicache.html` (reported 200 by the file, 404 on the verifier's two fetches), `docs.nvidia.com/dynamo/latest/user-guides/disaggregated-serving.html`, `docs.lmcache.ai/developer_guide/layerwise.html`, `docs.vllm.ai/en/latest/serving/context_parallelism.html`, `docs.vllm.ai/en/latest/serving/distributed_serving.html`, `docs.vllm.ai/en/latest/features/kv_events/`, `docs.sglang.ai/advanced_features/{context_parallelism,context_parallel,prefill_context_parallel}.html`, `docs.sglang.ai/backend/deterministic_inference.html` and `raw.githubusercontent.com/atlarge-research/py-kvcache/main/README.md`. The **`web_fetch` tool was unavailable** for this investigation — S4's verifier, S6's verifier and S7's method note all record that every fetch went through a sanctioned `curl --retry` proxy helper and that `web_fetch` was not used, and S11 records the task-supplied note that `web_fetch` is broken in this environment; the **`web_search` tool was reachable** and used for discovery in S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11 and S14 (with every cited URL then re-retrieved directly). Local source mirrors were **readable but stale or partial**: `/tmp/kvsrc/vllm-main` is an older vLLM snapshot (the S13 verifier identifies it as a ~0.11.2 tree, so its line numbers have drifted on live main) and `/tmp/kvsrc/sglang-main` is a partial checkout with no `python/` sources.
