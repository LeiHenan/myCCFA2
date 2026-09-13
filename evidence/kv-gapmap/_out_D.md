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
