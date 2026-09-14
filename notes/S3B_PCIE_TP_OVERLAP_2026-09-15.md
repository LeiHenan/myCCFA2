# S3b — Prior-art / occupancy check: is the PCIe TP penalty "latency-determined and therefore overlap-determined"?

**Date:** 2026-09-15
**Target claim under adjudication:** *the PCIe TP penalty is latency-determined and therefore overlap-determined — and specifically that TP collectives at decode shapes can be hidden under compute on a host-staged (no-P2P) PCIe box; hence the published "TP unusable / PP is the only viable strategy on PCIe" conclusion is conditionally wrong when the collective is overlapped.*

---

## 1. Verdict

**`OCCUPIED`**

Three independent artifacts each remove a leg of the claim, and one of them removes the mechanism itself:

1. **PRESERVE** (arXiv 2501.08192v2) already publishes the target's *exact mechanism* — hide the TP all-reduce not under arithmetic but under **memory traffic** (weight/KV prefetch into on-chip cache "during the communication operations"), motivated by the identical observation that accelerators "mostly remain idle" during collectives. Up to 1.6× end-to-end. This is the target's central idea, in print.
2. **SiFAR** (arXiv 2607.08973, MICRO 2026) runs the **matched control** the target proposes and reports it does *not* work at small batch: fused vs unfused GEMM+AllReduce for Llama-3.1-8B down-proj at TP=8 → "the fused GEMM has similar or worse latency, because there is not enough computation to overlap with communication." §2.6 states it as a general law: "in the low-latency inference regime … there is not enough computation to hide any communication overheads."
3. **FlashInfer #4393 is MERGED** and already ships the *other* resolution on the exact fabric the target names — "intra-node PCIe machines without NVLink … every peer transfer crosses the CPU root complex". It attacks latency directly (12 KiB: 5.3 µs vs NCCL 205.1 µs, 38.8×), and SGLang PR #34528 measures the end-to-end payoff on that fabric (TPOT 21.14 ms → 13.62 ms at TP8). Latency reduction, not overlap, is the shipped answer.

**Deciding artifact:** **PRESERVE** for the mechanism, **SiFAR §6.4** for the empirical refutation at the target's own batch regime. If forced to name one: **SiFAR §6.4** — it is a controlled A/B that tests overlap at decode batch sizes and reports a null/negative result.

**Residual that survives:** a *measurement* gap, not a *claim* gap — nobody has published the overlap A/B on a **GeForce-class card with peer access disabled entirely** (8×4090), nor a **TP-vs-PP crossover** curve. §4 below. Both are evaluation gaps inside an occupied claim space, and the nearest-neighbour evidence predicts the measurement would come back negative, not positive.

**Correction to the target's own framing (material):** the claim says the reversal holds "when the model is **small enough** that the latency tax is a small fraction of the token floor." This is inverted. The tax ≈ 2·L·t_ar is *flat* in model size, while the batch-1 weight-read floor scales with model size, so the tax fraction **shrinks as the model grows** (4B: ~3.95 ms vs ~8–11 ms ≈ 36–49%; 70B: ~3.95 ms vs ~140 ms ≈ 3%). The correct condition is *large enough*. If the surviving experiment is framed on "small model", it is framed on the worst case.

---

## 2. Prior-art table

State verified by fetching the URL myself on 2026-09-15 unless marked otherwise.

### 2a. The mechanism — "hide the collective under non-arithmetic work at decode"

| # | URL | State I verified | Verbatim quote | (a) lat/bw at decode | (b) overlap A/B | (c) TP/PP crossover | Control? |
|---|---|---|---|---|---|---|---|
| P1 | https://arxiv.org/abs/2501.08192 · https://arxiv.org/html/2501.08192v2 | **Published** (arXiv v2, 26 May 2025; EuroSys-conference header on v2). Title read back: *"PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving"* | *"Prior work addressed this issue by overlapping communication with compute, but has severe limitations due to the data dependencies between these operations. In this paper, we propose PRESERVE, a novel framework that prefetches model weights and KV-cache from off-chip HBM memory to the on-chip cache of AI accelerators **during the communication operations**"*; *"During these communication phases, the accelerators mostly remain idle, leading to reduced device utilization."* | Partial — names decode + PCIe: *"scaling beyond a single server still requires performing allreduce calls over slower networks such as PCIe or InfiniBand… the execution time of allreduce calls may take a large portion of the overall time"*. No latency-vs-bandwidth split at decode shapes. | **YES** — Fig. 1 is a three-way A/B: (A) vanilla, (B) GEMM+AllReduce fused, (C) PRESERVE. Up to **1.6×** e2e. | NO — "Pipeline Parallel" occurs **0** times in full text | **HAS-CONTROL** (fused vs unfused vs prefetch, same models) — but on *"commercial AI accelerators"* (Ascend/TPU-class NPUs), **not PCIe GPUs** |

### 2b. The refutation of the mechanism at the target's batch regime

| # | URL | State I verified | Verbatim quote | (a) | (b) | (c) | Control? |
|---|---|---|---|---|---|---|---|
| S1 | https://arxiv.org/abs/2607.08973 · https://arxiv.org/html/2607.08973v1 | **Published** (arXiv v1, 9 Jul 2026; MICRO 2026 header, Athens). Title read back: *"SiFAR: Synchronization-Free All-Reduce for Low-Latency LLM Inference"* | **§6.4:** *"Figure 26 compares fine-grained compute-communication fusion from ParallelKittens (Sul et al., 2025) against an unfused implementation for the down-projection GEMM of Llama-3.1-8B at TP=8. At small batch sizes, the fused GEMM has similar or worse latency, because there is not enough computation to overlap with communication. Fusion helps only at very large batch sizes, such as BS=2048, where the GEMM has enough work to hide communication. In the regime that SiFAR targets, overlap provides little benefit, so SiFAR reduces All-Reduce latency directly instead of trying to hide it."* | **YES** — *"synchronization overheads due to barriers before and after communication, accounting for 32–62% of the latency for small payloads we observe in low-batch serving"* | **YES, and negative** — fused vs unfused GEMM, BS sweep | NO | **HAS-CONTROL** — matched fused/unfused, same kernel family, same TP degree. Fabric is **8×H200 NVLink**, not PCIe |
| S2 | same as S1 (abstract + §2.5) | Published | *"Existing oneshot and twoshot algorithms incur overheads from barriers before and after communication."* … **§2.6:** *"Unlike prior approaches that rely on compute-boundedness from large batch sizes to overlap communication with computation, our solution directly reduces All-Reduce overheads in the low-latency inference regime, where batch sizes are minimal, and there is not enough computation to hide any communication overheads."* **§2.5:** *"With small batches, each GEMM only issues a small number of threadblocks, which can execute concurrently on the GPU, making computation insufficient to hide any communication latency."* | YES | YES (negative) | NO | HAS-CONTROL |

### 2c. Already shipped on the exact fabric (no-NVLink, host-staged/root-complex PCIe)

| # | URL | State I verified | Verbatim quote | (a) | (b) | (c) | Control? |
|---|---|---|---|---|---|---|---|
| F1 | https://github.com/flashinfer-ai/flashinfer/pull/4393 | **MERGED** (page state `"state":"MERGED"`) | *"Add a custom all-reduce for intra-node PCIe machines without NVLink … Adds `pcie_ipc_comm`: a CUDA-IPC all-reduce for machines where every peer transfer crosses the CPU root complex — no NVLink, no multicast."* … *"At 4 and 8 ranks `vllm_custom_all_reduce` launches nothing at all without NVLink: every arm of its dispatch except `world_size == 2` sits inside `if (full_nvlink_)` … so on this fabric the call falls through and returns."* … *"What is left is NCCL, and NCCL leaves a large amount on the table at these sizes"* … *"Copy-engine transfers are only ~24% faster than kernel writes here, so the data plane stays in the kernel rather than moving to DMA."* | **YES** — *"NCCL leaves a large amount on the table at these sizes"* + staged-push analysis | Partial — A/B is PCIe-IPC vs NCCL (two *collective implementations*), not comm-hidden vs comm-serialized | NO | **HAS-CONTROL** (matched NCCL baseline) but it is a *latency-reduction* control, not an *overlap* control |
| F2 | https://github.com/sgl-project/sglang/pull/34528 | **OPEN** (state `"state":"OPEN"`; opt-in, `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` **default off**) | *"Motivation On switch-free intra-node hosts — no NVLink, no multicast, every peer transfer crossing the CPU root complex — none of SGLang's custom all-reduce backends apply. … **NCCL's ring is bandwidth-optimal but latency-poor at decode message sizes, and it shows up directly in TPOT.**"* … *"These kernels win by latency at small messages; a prefill chunk is three orders of magnitude larger, and there NCCL's ring is the better algorithm."* Measured, 8 ranks, TP8, 8k ctx: `NCCL only` TTFT 1910 ms / **TPOT 21.14 ms** / 35.02 tok/s → `decode-sized` 1849 ms / **13.62 ms** / 48.05 tok/s. Raw: 12 KiB **5.3 µs vs 205.1 µs (38.8×)**; 96 KiB 15.3 vs 584.9 (38.2×); 100 MiB 8.5 ms vs 50.4 ms (5.9×). | **YES** — the sharpest statement of (a) found anywhere, on the target fabric | NO — replace-the-collective, not hide-the-collective | NO | **HAS-CONTROL** (matched NCCL, same image/token IDs) |
| F3 | https://github.com/local-inference-lab/rtx6kpro/blob/61842c4f5c3489dc4edaba7639a2f0656b587f9e/optimization/pcie-oneshot-allreduce.md | **Public repo doc, not peer-reviewed** (fetched raw at pinned commit) | *"During tensor-parallel decode, each layer performs AllReduce operations on small messages (typically 16–256 KB for attention/MoE layers). **At these sizes, NCCL's ring protocol overhead dominates — the actual data transfer time is negligible compared to setup, synchronization, and protocol negotiation.**"* … *"the nvidia driver defaults to **SysMem staging** for GPU-to-GPU memory accesses from CUDA kernels. This makes the PCIe oneshot allreduce ~15× slower than NCCL"* | **YES** — explicit latency-vs-bandwidth decomposition at decode shapes on PCIe | NO — same "make the collective faster" route | NO | **HAS-CONTROL** (NCCL at matched sizes). Note: results on RTX PRO 6000 Blackwell, 4/8-GPU; **net loss at 8 GPU cross-socket (−4.3% to −15.7%)** |

### 2d. The published claim the target wants to reverse, and its nearest critics

| # | URL | State I verified | Verbatim quote | (a) | (b) | (c) | Control? |
|---|---|---|---|---|---|---|---|
| B1 | https://sapi.bntu.by/jour/article/view/799/0 · https://rep.bntu.by/items/1a84e739-9ecc-4017-91fd-221745335bee | **Published, peer-reviewed** — «№ 1 (2026)», DOI 10.21122/2309-4923-2026-1-54-59, PDF `54-59.pdf` (1.68 MB). English alt-title on the repository record: *"Parallelism strategies as a key factor for deploying Large Language Models on consumer gpus"* | *"Результаты однозначно указывают на **непригодность тензорного параллелизма для систем без NVLink из-за критических задержек синхронизации**. Доказано, что конвейерный параллелизм является единственной жизнеспособной стратегией для PCIe-кластеров, обеспечивая высокую пропускную способность, несмотря на наличие периодов простоя («пузырей»)…"* (Results unambiguously indicate the unsuitability of TP for systems without NVLink **due to critical synchronization latencies**. PP is proven the only viable strategy for PCIe clusters…). Setup: monolithic RTX A6000 vs 2× RTX 3090, vLLM, DeepSeek-R1-Distill-Llama-14B; throughput, TTFT, TPOT, power stability. | **PARTIAL, and in the target's favour** — the paper itself attributes failure to **latency of synchronization**, not bandwidth. So (a)'s *conclusion* is already in the literature for PCIe; only the measurement is missing. | NO | **NO crossover reported** — blanket verdict; PP "only viable" despite bubbles | **NO-CONTROL** for the causal claim — no comm-hidden arm, no overlap A/B; a monolithic-vs-distributed comparison only |
| B2 | https://pssg.cs.umd.edu/blog/2025/beyond-nccl/ | **Published group blog** (PSSG, UMD; companion paper + NVRAR/YALIS repos linked). Updated 16 Sep 2025 | *"its auto-regressive token-by-token nature — issuing many small, latency-sensitive messages — dominates end-to-end latency"* … *"the prefill All-Reduce message size is 620 MiB (very large message regime), whereas **the decode All-Reduce message size is 256 KiB (small message regime)**"* … *"But are these algorithms optimal for the small-message, low-latency regime of decode?"* … *"While NCCL excels within a node for small messages, its performance drops once communication crosses nodes."* | **YES** — clean two-regime decomposition at decode | NO — NVRAR is a faster collective | NO | **HAS-CONTROL** (NCCL baseline); fabric is **inter-node Slingshot/IB**, not PCIe |
| B3 | https://github.com/pytorch/pytorch/pull/24800 | **CLOSED** (stale-bot; not merged) | *"This mode delays gradients allreduce till the end of the backward pass, i.e., there will be no overlap between gradient communication and local backward computation. **This will be slower than the default mode**, but should have less constraints on input models."* | NO | Indirectly: it proves **overlap is the *default*** in PyTorch DDP, and de-overlapping is the opt-out | NO | HAS-CONTROL by construction (default vs delayed) |
| B4 | https://github.com/NVIDIA/TransformerEngine/blob/ed4d06908dbe3a584aca794d3995075a53f9151f/docs/developer/distributed/comm_gemm_overlap.rst | **MERGED / in-tree docs** at pinned commit | *"Transformer Engine can overlap NCCL collective communication with GEMM computation to hide communication latency. **This is critical for scaling TP across nodes where inter-node bandwidth is limited.**"* … *"In tensor parallelism, a column-parallel linear requires an all-gather before GEMM, and a row-parallel linear requires a reduce-scatter after GEMM."* Overlap modes shipped: `BULK_OVERLAP_AG`, `BULK_OVERLAP_RS`, `SPLIT_PIPELINED_AG_P2P`, `SPLIT_PIPELINED_RS`, `SPLIT_PIPELINED_RS_P2P`, `ATOMIC_GEMM_RS`, `ATOMIC_GEMM_AG_P2P` — **no all-reduce overlap mode** | NO | **YES** — canonical AG+GEMM / GEMM+RS pipelined overlap, with comm-stream vs compute-stream timelines | NO | **HAS-CONTROL** (with/without overlap timelines). Training/SP pattern; **AG+RS, not all-reduce** |
| B5 | https://github.com/sgl-project/sglang/issues/8728 | **CLOSED** | *"The current community implementation of overlap is TBO, which uses deepep to achieve computation and communication hiding. However, TBO has bugs in its application #8234, and its performance improvement is limited. In a scenario with an input length of 3500, the improvement is only 5-10%."* Table 1 row — **GEMM+AllReduce**: TransformerEngine **No**; Flux **Plan**; Triton-Distributed **"Plan, the integration of gemm and allreduce has been achieved."** Motivation is the **prefill** stage on K2 TP8; *"communication accounts for more than the computation."* | Partial (prefill, NVLink single machine) | Yes — a *comparison of available overlap libraries*, not a measurement of overlap benefit | NO | **UNVERIFIABLE** for the table (vendor/community claims, no numbers shown) |
| B6 | https://arxiv.org/abs/2606.01927 | **Published** (arXiv, submitted 1 Jun 2026). Title read back: *"Scaling LLM Inference Beyond Amdahl`s Limits via Eliminating Non-Scalable Overheads"* | *"Tensor parallelism (TP) is necessary to fit modern models but scales sub-linearly as the TP degree t grows, due to cross-GPU communication and non-scalable runtime work… **We identify and validate an empirical optimal TP degree t_e that balances these effects.**"* … *"raises the attainable t_e by shrinking the non-scalable portion via overlap of scheduling and I/O with compute and sequence-parallel sampling"* | NO (Amdahl framing, not latency/bandwidth) | YES — overlap of scheduling/IO with compute; up to 1.9× throughput | **Partial (c)** — an **optimal TP *degree***, i.e. a crossover in *degree*, not TP-vs-PP and not PCIe-indexed | **HAS-CONTROL** (vLLM baseline) |

### 2e. Checked and *not* relevant (recorded to prevent re-work)

| URL | State verified | Why it drops out |
|---|---|---|
| https://arxiv.org/abs/2508.20274 — *"Predictable LLM Serving on GPU Clusters"* (27 Aug 2025) | Published; title read back | PCIe fabric is only a *noisy-neighbour* substrate (MIG, placement, MPS quotas, cgroup I/O). No TP collective, no overlap A/B, no TP/PP crossover |
| https://aclanthology.org/2025.findings-emnlp.957/ — Jin et al., *"Distributed LLM Serving on Consumer-Grade GPUs by Reconciling Computation and Communication"* (Findings of EMNLP 2025); arXiv 2507.05043 *"MoLink"* | Published (ACL Anthology + arXiv titles read back, both match) | Contribution is **prefill/decode transmission scheduling + chunking** across weakly-connected consumer GPUs: *"It splits the prolonged transmission volume of prefill requests into smaller chunks and carefully scheduling their transmission."* No TP allreduce latency decomposition, no overlap A/B, no TP/PP crossover |
| https://arxiv.org/abs/2506.22033 — *"SiPipe: Bridging the CPU-GPU Utilization Gap for Efficient Pipeline-Parallel LLM Inference"* | Published; title read back | Improves **PP** (CPU sampling, token-safe execution, structure-aware transmission) to close PP bubbles. Assumes PP; does not compare TP vs PP on PCIe |
| https://arxiv.org/abs/2511.13940 — *"ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels"* | Published; title read back | Supplies the *fused GEMM* that SiFAR §6.4 uses as the overlap arm. Generic overlap framework; *"up to 2.33× speedup for data- and tensor-parallel workloads"* — no PCIe-specific or decode-shape TP result |
| https://arxiv.org/abs/2205.05198 — *"Reducing Activation Recomputation in Large Transformer Models"* (Megatron-LM) | Published; title read back | Sequence parallelism, **training/activation-memory** motivation. Establishes SP (which converts the all-reduce into AG+RS, enabling overlap) but reports no inference-decode collective measurement |
| https://arxiv.org/abs/2510.27641 vs https://arxiv.org/abs/2602.07223 | Both fetched; titles read back | **SpecAttn name-collision trap confirmed and avoided.** 2510.27641 = *"SpecAttn: Speculating Sparse Attention"* (31 Oct 2025); 2602.07223 = *"Vegas: Self-Speculative Decoding with Verification-Guided Sparse Attention"* (6 Feb 2026). **Different papers.** Neither is relevant to TP overlap |
| https://github.com/noonghunna/qwen36-dual-3090/blob/master/docs/INTERNALS.md | Community doc, fetched | Closest community statement of a **batch-size** effect on PCIe: *"All-reduce on PCIe Gen 4 (~32 GB/s practical) is ~3-5× slower than NVLink"*; *"All-reduce overhead approximately cancels the memory-bandwidth halving"*; *"single-stream TPS per card ≈ same as single-card TPS. The TP=2 win is concurrent throughput — when 2-4 streams run simultaneously, all-reduce overhead amortizes across the larger batch"*; *"per-stream TPS gain from TP=2 is small (~5%)"*. **Uses the bandwidth framing the target says is wrong** — and reports TP=2 *improving* with batch, i.e. the opposite direction from a "decode-shape" story |

---

## 3. What is genuinely taken

**"Overlap TP collectives with compute" is standard practice, and has been for years.** Say it plainly:

- PyTorch DDP's *default* is overlapped gradient all-reduce; PR #24800 is the opt-out, and its own description says the mode that removes overlap "will be slower than the default mode."
- NVIDIA TransformerEngine ships seven named comm/GEMM overlap algorithms with documented comm-stream/compute-stream timelines (B4).
- PRESERVE's related work dates allreduce/GEMM fusion to **Hoefler & Lumsdaine 2008** and cites five systems (Rashidi 2021; Punniyamurthy 2024; Chang 2024; Wang 2023) — *"prior studies have proposed fusing matmul and allreduce operations to overlap compute and communication"* (P1).
- ParallelKittens (2511.13940) generalizes it as a framework.
- PRESERVE (P1) is the *specific* variant the target needs — hide the collective under **memory traffic**, not arithmetic — and is titled and abstracted around exactly that.

**What that does to the claim's novelty:**

| Element of the target | Status |
|---|---|
| "TP all-reduce on PCIe is a latency event, not a bandwidth event" | **Fully occupied.** SiFAR (32–62% of latency is barriers), F1/F2 on the exact no-NVLink fabric ("bandwidth-optimal but latency-poor"), F3 ("protocol overhead dominates — data transfer time is negligible"), B2 (620 MiB prefill vs 256 KiB decode), and even B1's own causal attribution ("критических задержек синхронизации"). |
| "Therefore it should be overlap-determined" | **Occupied as a mechanism (P1), and empirically contradicted at the target's batch regime (S1/S2).** PRESERVE already does "hide the allreduce under memory work". SiFAR's matched control says fusion at small batch is "similar or worse". |
| "TP collectives at decode shapes can be hidden under compute on a host-staged PCIe box" | **No direct measurement found — but this is the weakest form of the claim, and the closest evidence points the other way.** The one thing genuinely absent is the A/B on a GeForce-class no-P2P box. |
| "The BNTU conclusion is conditionally wrong" | **Not sustainable as stated.** BNTU is not confounded by a missing overlap arm in the way the target hopes: B1's own stated cause is latency, and the shipped answer to latency on that fabric (F1 merged, F2 measured 21.14 → 13.62 ms TPOT) is *remove the latency*, not *hide it*. Reversing B1 requires showing overlap beats a 38× lower-latency collective — a much harder bar than "BNTU forgot to overlap." |

Net: the target is a **conjunction of an occupied observation and an occupied (and partly refuted) mechanism**, plus one unmeasured fabric/shape cell. That is an evaluation gap, not a contribution.

---

## 4. What remains, if anything

**Nearest neighbour X does A; under condition C it misses B; doing B differs because…**

- **X = PRESERVE (P1).** Does A = *hide the TP all-reduce under memory traffic instead of under arithmetic* — the target's mechanism, in print, up to 1.6× e2e.
- **C = the miss:** PRESERVE is evaluated on *"commercial AI accelerators"* (Ascend/TPU-class NPUs) with design-space exploration over L2 capacity. It never considers the **NVIDIA consumer/GeForce no-P2P regime**, where the collective is not merely slow but must **bounce through host DRAM** (no peer access at all), has no on-chip cache to prefetch into at the relevant granularity, and where the transport resource (PCIe + host DRAM + copy engine) is *disjoint* from the decode bottleneck resource (GPU HBM + SM). SiFAR's stated failure mechanism — *"each GEMM only issues a small number of threadblocks … computation insufficient to hide any communication latency"* — is an **SM/threadblock-scarcity** argument. That argument is fabric-specific, and on a host-staged box it may not bind, because the collective runs over copy engines and PCIe rather than contending for the same SMs.
- **Doing B differs because** the two governing ratios move in *opposite* directions on PCIe and neither has been measured there: (i) *favourable* — resource disjointness (PCIe/host vs HBM/SM) means the comm need not steal the decode bottleneck; (ii) *unfavourable* — t_ar/t_compute is ~140× worse than on NVLink at equal size, so there is far more latency to hide per unit of compute. Published work resolves this only for NVLink (S1: negative) and for NPUs (P1: positive). The sign on host-staged PCIe is genuinely unknown.

**First-order magnitude estimate — label each number:**

| Quantity | Value | Source |
|---|---|---|
| 2-rank all-reduce, 5 KB, 8×4090 | ~0.055 ms (~0.09 GB/s) | **measured** (project background) |
| Collective tax, 36 layers × 2 | ~3.95 ms/token, flat in batch | **estimated** (arithmetic on the above) |
| Batch-1 weight-read floor, 4B model | ~8–11 ms/token | **measured** (project background) |
| Tax as fraction of token time, 4B | ~36–49% | **estimated** |
| Upside **if the collective were fully hidden** | ceiling ≈ 26–33% token time | **estimated** — an upper bound, not achievable |
| Realistic hideable fraction | ~3–8% token time | **estimated, weakly reasoned.** Reason it is far below the ceiling: every all-reduce is a hard data dependency for the next layer, so at least one chunk's latency stays exposed per layer; and on PCIe the per-chunk latency is nearly size-independent (F3, F2), so *chunking does not shrink the exposed tail* — splitting 55 µs into 4 chunks leaves ~45 µs exposed, not ~14 µs. This is the single most important thing to check empirically before investing. |
| Competing measured bar the overlap route must beat | **−4.6% to −5.6% full-model step time** (F1/F2, TP2/TB8, PCIe Gen4, DeepSeek-V4-Flash, matched 127-step captures) and **TPOT 21.14 → 13.62 ms** (F2, TP8 8k) — obtained by *pure latency reduction*, not overlap | **measured** |
| SiFAR's own upper bound on the value of the collective (NVLink) | removing All-Reduce entirely = +18% (TP=2) / +43% (TP=8) throughput, Llama-3.1-8B, 8×H200 | **measured** |

**Cheapest decisive experiment** (on the existing 8×4090 box; ~1 day, no new hardware):

On a 2-rank TP group at batch 1, hidden=2560, bf16, 36 layers, CUDA-graph-captured, ≥5 reps of ≥500 tokens, measure token time for four arms:

- **(A) serialized** — current: row-parallel GEMM → `all_reduce` (sync) → next layer.
- **(B) hidden** — `dist.all_reduce(..., async_op=True)` on a second stream, launched immediately as the row-parallel GEMM's last tile is written, with the *next* layer's GEMM already launched (i.e. the delayed-collective schedule); plus a 2- and 4-chunk variant.
- **(C) deleted upper bound** — feed the pre-all-reduce tensor straight to the next op, exactly as SiFAR does, to get the true headroom: `(A − C)/A`.
- **(D) faster-collective control** — F3's PCIe oneshot / any non-NCCL path, to see whether latency reduction and overlap are additive or redundant.

**Kill condition (either one ends it):**
1. `(A − C)/A < 5%` — the collective is already too small a share of token time for *any* overlap scheme to matter in this regime; or
2. `(A − B)/A < 5%`, or `(A − B) < (A − D)` — overlap loses to simply making the collective faster, which is what the merged artefacts in §2c already do, on this exact fabric, for free.

A note on the second condition: given that (i) F1 is merged, F2 is measured on the target fabric, and (ii) chunking cannot shrink a latency-dominated exposed tail, condition 2 is the likely outcome. **Run (C) and (D) first** — they are both cheap and each can terminate the project without building the overlap schedule.

---

## 5. Search log

### Surfaces used
`web_search` (multi-query batches); `curl -sL --compressed --retry 4 --retry-delay 2 --retry-all-errors -A "<Chrome/126 macOS UA>"` against: `arxiv.org/abs/<id>`, `arxiv.org/html/<id>v<N>`, `raw.githubusercontent.com`, `github.com/{o}/{r}/pull|issues/<n>`, `patch-diff.githubusercontent.com/raw/.../pull/<n>.diff`, `sapi.bntu.by`, `rep.bntu.by`, `aclanthology.org`, `pssg.cs.umd.edu`, `mlsys.org`, `blog.hotdry.top`. Local project surfaces: `kv_tp_research/C7_C8_prior_art_report.md`, `.s3b_dir34/`, `.s3b_numdrift/`.

Method: every arXiv ID cited was fetched at `arxiv.org/abs/<id>` and its **`<h1 class="title mathjax">` read back** before use. GitHub PR/issue **state was read from the page's embedded `"state":"…"` JSON**, and bodies extracted from `class="comment-body"` / `class="markdown-body"` blocks (the `"body":"…"` embedded-JSON route yielded nothing on these pages).

### Exact queries (web_search)
1. `overlapping all-reduce with compute tensor parallelism PCIe no NVLink decode latency`
2. `tensor parallelism vs pipeline parallelism PCIe crossover batch size measurement`
3. `Megatron-LM sequence parallelism overlapped all-reduce communication hidden under compute`
4. `tensor parallelism PCIe no NVLink overlap communication compute hidden decode batch size 1`
5. `TP vs PP crossover point PCIe NVLink interconnect LLM inference measurement study`
6. `critique tensor parallelism unsuitable PCIe pipeline parallelism only viable`
7. `MoLink disaggregated serving consumer GPUs tensor parallelism latency`
8. `PowerInfer 2 PCIe tensor parallelism consumer GPU no NVLink`
9. `host-staged allreduce NCCL PCIe no P2P latency dominated small message decode`
10. `"all-reduce" overlap tensor parallel decode inference PCIe consumer GPU hidden latency 2026`
11. `chunked pipelined allreduce overlap GEMM tensor parallelism inference latency hiding`
12. `sequence parallelism all-gather reduce-scatter overlap tensor parallel communication hidden`
13. `tensor parallel allreduce overlap compute PCIe 4090 RTX consumer GPU decode hiding latency experiment`
14. `"tensor parallelism" PCIe "pipeline parallelism" crossover batch size model size measurement study 2026`
15. `overlap allreduce with weight loading memory bound decode LLM inference PCIe bandwidth contention`
16. `Kog inference engine KCCL tensor parallelism consumer GPU communication overlap 3000 tokens`
17. `delayed allreduce one layer lagged communication overlap tensor parallel inference`
18. `"pipeline parallelism" "tensor parallelism" PCIe 3090 4090 benchmark TPOT TP vs PP comparison consumer`
19. `TP vs PP crossover batch size fiber interconnect "tensor parallel" "pipeline parallel" when to choose measurement A100 PCIe no NVLink`
20. `"GEMM+AllReduce" fusion overlap inference decode low batch prototype`
21. `TensorRT-LLM vLLM PCIe no P2P allreduce latency TPOT measurement 8x 4090`
22. `PRESERVE prefetching model weights KV-cache distributed LLM serving tensor parallel`
23. `tensor parallelism communication overhead PCIe hidden under compute weight prefetch decode single batch experiment 2026`
24. `Tensor Parallelism vs Pipeline Parallelism which is better PCIe only flawed benchmark critique`

### What I could NOT verify (absence claims are "I did not find it", never "it does not exist")

1. **The BNTU full text.** I verified title, authors, DOI, journal issue, English alternative title and abstract at **two independent surfaces** (`sapi.bntu.by` and `rep.bntu.by`), but I did **not** download `54-59.pdf` (1.68 MB) and therefore have **not read their per-configuration numbers**. I can **not** confirm or deny whether BNTU reports a batch/degree crossover internally, nor at what concurrency their throughput comparison was run. Their abstract states a blanket verdict; the body may be more conditional. **This is the highest-value unread document in this report.**
2. **PRESERVE's ACM "Brief Announcement" record** (`dl.acm.org/doi/pdf/10.1145/3816782.3819214`) appeared in search results but I did not fetch it (dl.acm.org is Cloudflare-gated here). All PRESERVE quotes are from **arXiv 2501.08192v2**, title read back. Venue line is from the v2 HTML header ("EuroSys: European Conference on Computer Sys…", truncated in the rendered text) and is reported here as **unconfirmed in full**.
3. **Whether FlashInfer `pcie_ipc_comm` (F1) works on GeForce-class cards with peer access disabled entirely.** The PR scopes itself to *"the whole RTX PRO 6000 / RTX 6000D class"* and to fabrics where peer transfers "cross the CPU root complex" — i.e. slow-but-present root-complex peer writes. It makes **no claim** about 8×4090 (GeForce P2P disabled). So F1 is *adjacent* to, not *identical to*, the target's box. I did not find any artifact that measures **overlap** on a true no-peer-access GeForce box.
4. **Kog / KCCL.** The Chinese secondary blog (below) asserts Kog's KCCL gets linear TP scaling via delayed communication on **consumer GPUs**; press coverage I saw attributes Kog's 3,000 tok/s result to **AMD MI300X** (Infinity Fabric, not PCIe). I did not fetch `kog.ai` and could not verify any overlap implementation. Treated as **UNVERIFIABLE**.
5. **`blog.hotdry.top` (2026-05-30), "消费级GPU上突破内存带宽瓶颈：LLM推理的计算-通信重叠策略"** — fetched successfully. It describes the target's exact schedule ("delay layer N's all-reduce until layer N+1 has started; overlap in the background") for consumer GPUs, but it is an **unsigned secondary blog** carrying implausible technical claims (e.g. topology-aware placement of weights into specific GDDR6X banks to raise effective bandwidth 15–25%). No measurements, no artifact, no control. I record it only as evidence that **the idea circulates in consumer-GPU discourse**, not as prior art. There may be an unindexed primary source behind it that I did not find.
6. **A counter-paper critiquing BNTU, or any paper arguing "TP is fine on PCIe."** I did not find one. B1 appears to stand uncited-and-uncriticised in this specific respect.
7. **Global GitHub code search** (`github.com/search`) — not attempted to completion; known 429 in this environment. Repo-scoped issue search and direct PR/issue URLs worked and were used.
8. **Any work reporting a TP-vs-PP crossover as a function of batch, model size, or degree on PCIe.** I did not find it. The nearest items are B5's community note that TP=2's win is *concurrent throughput* (a batch effect, bandwidth-framed), B6's optimal TP *degree*, and F2's message-size crossover — none of which is a TP-vs-PP crossover on PCIe.
