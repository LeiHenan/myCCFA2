# Prior-Art Verification: Where Decode-Phase Time and Memory Bandwidth Actually Go

Verification date: 2026-09-15. All titles below were read back at `https://arxiv.org/abs/<id>` via `curl`.
Route health observed during this task is recorded at the end.

---

## HEADLINE ANSWER

The folk breakdown **"decode time ≈ weights read + KV cache read, both bandwidth-bound"** has been
**measured, and explicitly corrected as "true but incomplete"** in the literature.

Three artifacts carry the correction:

| Artifact | What it does |
|---|---|
| **2605.30571** *Memory-Bound but Not Bandwidth-Limited* | Measures the folk model directly as `t_floor = (W+K)/B_peak` and shows only **27%** of it is realized on H100 vs **81%** on L4. Isolates **launch-side CPU overhead** as the missing term via a pre-registered CUDA Graphs A/B. |
| **2607.29575** *SLIM* | States prior bandwidth attributions were *"supported by conceptual arguments or high-level performance observations"* and relocates the saturation cause to **decode-phase attention kernels**, with Nsight Systems/Compute profiling. |
| **2512.01644** *A Systematic Characterization of LLM Inference on GPUs* | Shows decode's kernels have **heterogeneous** bottlenecks: FFN-Up memory-bound (58% Memory Dependency) but O-Proj/FFN-Down **execution-bound** (51% Execution Dependency, ~130 live registers). LayerNorm sync stalls rise 11.0% → 27.7%. |

**The strongest single refutation of "weights + KV, co-equally"** is 2605.30571's own per-block
byte accounting: at ctx=2048 the KV contribution is **4.2 MB** out of **470 MB** per block
(≈0.9%); SwiGLU MLP weights alone are **407.4 MB (≈87%)**. KV only becomes comparable at
much longer contexts.

---

# PART A — THE 9 CANDIDATE IDs (all read back)

## A1. VERIFIED, ON-TOPIC

```
URL: https://arxiv.org/abs/2402.16363
STATE: arXiv abs page, v6 (submitted 26 Feb 2024, last revised 1 May 2024). HTML full text also fetched at https://arxiv.org/html/2402.16363v6
TITLE_READBACK: [2402.16363] LLM Inference Unveiled: Survey and Roofline Model Insights
VERBATIM_QUOTE: From the abs-page abstract: "This framework identifies the bottlenecks when deploying LLMs on hardware devices and provides a clear understanding of practical problems, such as why LLMs are memory-bound, how much memory and computation they need, and how to choose the right hardware." Also from the abs-page abstract: "The analyze tool, LLM-Viewer, is open-sourced."
CLAIM: Canonical statement of the roofline/folk framing: LLM inference is memory-bound; introduces a roofline framework (LLM-Viewer) to derive the memory-vs-compute split analytically.
HAS_MEASUREMENT: NO measurement. This is a MODEL/ESTIMATE paper. I grepped the v6 HTML full text (186,449 chars): the string "validat" has 0 hits; "measure" has 1 hit and it is inside a bibliography entry. All splits are derived from roofline arithmetic on hardware spec sheets, not from counters.
HAS_CONTROL: no
RELEVANCE_TO_DIRECTION6: This is the likely upstream SOURCE of the folk "weights + KV, bandwidth-bound" narrative — cite it as the model to be contrasted against, not as evidence for it.
```

```
URL: https://arxiv.org/abs/2501.08219
STATE: arXiv abs page, v4 (submitted 14 Jan 2025, last revised 24 Feb 2026)
TITLE_READBACK: [2501.08219] Characterizing LLM Inference Energy-Performance Tradeoffs across Workloads and GPU Scaling
VERBATIM_QUOTE: From the abs-page abstract: "At the hardware level, the decode phase dominates inference time (77-91%) and is largely insensitive to GPU frequency. Consequently, reducing GPU frequency from 2842 MHz to 180 MHz achieves an average of 42% energy savings with only a 1-6% latency increase."
CLAIM: Decode is 77-91% of end-to-end inference time and is frequency-insensitive (i.e. bandwidth/latency-limited, not compute-limited). Useful as the *phase-level* attribution ("decode dominates") but says nothing about weights-vs-KV inside decode.
HAS_MEASUREMENT: YES — measured, on five decoder-only LLMs (1B-32B) across four NLP benchmarks, controlled offline setup, with measured latency and energy under DVFS sweeps. Numbers: decode = 77-91% of time; 42% energy saving; 1-6% latency increase; 180-2842 MHz.
HAS_CONTROL: yes — frequency sweep is a controlled A/B at fixed workload
RELEVANCE_TO_DIRECTION6: Establishes "decode dominates" as measured fact, but does NOT decompose decode into weights vs KV; cannot support the byte-split claim either way.
```

```
URL: https://arxiv.org/abs/2211.05102
STATE: arXiv abs page, v1 (submitted 9 Nov 2022). Authors confirmed on page: Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean.
TITLE_READBACK: [2211.05102] Efficiently Scaling Transformer Inference
VERBATIM_QUOTE: From the abs-page abstract: "We develop a simple analytical model for inference efficiency to select the best multi-dimensional partitioning techniques optimized for TPU v4 slices based on the application requirements." And: "we achieve a low-batch-size latency of 29ms per token during generation (using int8 weight quantization) and a 76% MFU during large-batch-size processing of input tokens, while supporting a long 2048-token context length on the PaLM 540B parameter model."
CLAIM: Analytical (roofline-style) model of latency/ MFU for partitioning; the low-batch-size decode case is explicitly framed as weight-traffic-dominated, and multiquery attention is used to shrink KV growth for long context.
HAS_MEASUREMENT: MIXED — the headline 29 ms/token and 76% MFU are measured on TPU v4; the partitioning selection is from an analytical model. No DRAM byte counters.
HAS_CONTROL: yes — new Pareto frontier compared against the FasterTransformer benchmark suite.
RELEVANCE_TO_DIRECTION6: Primary-source analytical model that predates the folk summary; useful as an example of the "model predicts, do not cite as measurement" class.
```

```
URL: https://arxiv.org/abs/2401.08671
STATE: arXiv abs page (DeepSpeed-FastGen). NOTE: the prompt offered an alternative attribution "or DistServe" — that alternative is WRONG for this ID.
TITLE_READBACK: [2401.08671] DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference
VERBATIM_QUOTE: From the abs-page abstract: "This paper introduces DeepSpeed-FastGen, a system that employs Dynamic SplitFuse, a novel prompt and generation composition strategy, to deliver up to 2.3x higher effective throughput, 2x lower latency on average, and up to 3.7x lower (token-level) tail latency, compared to state-of-the-art systems like vLLM."
CLAIM: Scheduling/throughput system. Does not attribute decode time to a weights-vs-KV split.
HAS_MEASUREMENT: YES for throughput/latency (2.3x, 2x, 3.7x); NO bandwidth/byte attribution.
HAS_CONTROL: yes — compared against vLLM.
RELEVANCE_TO_DIRECTION6: NOT relevant to the byte-attribution question. Discard from the direction-6 evidence table.
```

```
URL: https://arxiv.org/abs/2401.09670
STATE: arXiv abs page (DistServe)
TITLE_READBACK: [2401.09670] DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
VERBATIM_QUOTE: From the abs-page abstract: "DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding computation. Existing LLM serving systems colocate the two phases and batch the computation of prefill and decoding across all users and requests. We find that this strategy not only leads to strong prefill-decoding interferences but also couples the resource allocation and parallelism plans for both phases."
CLAIM: Prefill and decode have different resource profiles and should be placed on different GPUs. It attributes *interference* and *resource coupling* to colocation — a phase-level attribution, not a weights-vs-KV byte split.
HAS_MEASUREMENT: YES — 7.4x more requests or 12.6x tighter SLO vs SOTA, >90% of requests within latency constraints.
HAS_CONTROL: yes — compared against state-of-the-art colocated systems.
RELEVANCE_TO_DIRECTION6: Supports "decode ≠ prefill" but does NOT quantify weights-vs-KV bytes. Peripheral.
```

```
URL: https://arxiv.org/abs/2405.04532
STATE: arXiv abs page (QServe)
TITLE_READBACK: [2405.04532] QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving
VERBATIM_QUOTE: From the abs-page abstract: "We uncover a critical issue: existing INT4 quantization methods suffer from significant runtime overhead (20-90%) when dequantizing either weights or partial sums on GPUs." And: "We also make fused attention memory-bound, harnessing the performance gain brought by KV4 quantization."
CLAIM: A first-order CORRECTION of a different mis-attribution: the claimed bottleneck for INT4 serving is NOT bandwidth but **dequantization overhead on low-throughput CUDA cores**. Also asserts attention can be made memory-bound by KV4.
HAS_MEASUREMENT: YES — measured speedups: Llama-3-8B 1.2x on A100 / 1.4x on L40S; Qwen1.5-72B 2.4x on A100 / 3.5x on L40S vs TensorRT-LLM; and a measured 20-90% dequant overhead.
HAS_CONTROL: yes — matched comparison against TensorRT-LLM.
RELEVANCE_TO_DIRECTION6: Strong precedent for the *genre* "measured result contradicts the assumed bandwidth story" — a good template for how Direction 6 should frame its correction.
```

## A2. VERIFIED BUT MUST BE DISCARDED (ID exists, described title does not match)

```
URL: https://arxiv.org/abs/2407.05858
STATE: arXiv abs page
TITLE_READBACK: [2407.05858] Fast On-device LLM Inference with NPUs
MISMATCH: The prompt described this ID as "about LLM inference roofline / analytical model". That is FALSE. The ID resolves to an on-device NPU inference paper. DISCARD — do not cite as a roofline/analytical-model reference.
```

```
URL: https://arxiv.org/abs/2402.15627
STATE: arXiv abs page
TITLE_READBACK: [2402.15627] MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs
MISMATCH: Correctly suspected irrelevant. This is a TRAINING-systems paper, not inference/decode. DISCARD for Direction 6.
```

```
URL: https://arxiv.org/abs/2407.00088
STATE: arXiv abs page
TITLE_READBACK: [2407.00088] T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge
MISMATCH: The prompt described this ID as "QServe or similar W4A8 quantization". That is FALSE — QServe is 2405.04532 (verified above). 2407.00088 is T-MAC (CPU, table-lookup low-bit). DISCARD.
```

---

# PART B — THE FIVE REQUESTED TOPICS

## Topic 1 — Per-operator / per-layer decode profiling: attention vs MLP/FFN time & bytes

```
URL: https://arxiv.org/abs/2512.01644 and https://arxiv.org/html/2512.01644v1
STATE: arXiv abs page + arXiv HTML v1 full text (369,827 bytes fetched; 87,005 chars of text extracted). v1, 1 Dec 2025, cs.AR. Authors read off the page: Haonan Wang, Xuxin Xiao, Mingyu Yan, Zhuoyuan Zhu, Dengke Han, Duo Wang, Wenming Li, Xiaochun Ye, Cunchen Hu, Hongyang Chen, Guangyu Sun.
TITLE_READBACK: [2512.01644] A Systematic Characterization of LLM Inference on GPUs
VERBATIM_QUOTE (from the HTML full text, Section 4.2.1 "Operator-Level Bottleneck Migration", caption of Figure 4 and surrounding prose): "In the Decode phase, FFN cost per step is roughly constant O(1), while Attention grows linearly O(n) due to full KV-cache traversal. Thus, under typical contexts, large models (e.g., 32B) are FFN-dominated due to high fixed costs, while small models (e.g., 8B) are Attention-dominated. As context length extends further, Attention's linear growth eventually exceeds FFN's fixed cost, causing both model scales to become Attention-dominated in extremely long contexts."
VERBATIM_QUOTE 2 (from the HTML full text, same section): "Within LLM architectures, the FFN and Attention modules collectively account for the majority of inference latency, representing 94% to 96% of the total execution time across phases and scenarios."
VERBATIM_QUOTE 3 (from the HTML full text, Section 5.2.2 "Decode Phase Stalls"): "Decode's stall profile shifts fundamentally: Pipeline Busy plunges to 3 ∼ 4%, while Memory Dependency dominates."
VERBATIM_QUOTE 4 (from the HTML full text, Section 5.2.2, the key qualification): "Further analysis reveals heterogeneous bottlenecks among Decode's GEMM kernels: FFN-Up is memory-bound, with Memory Dependency reaching 58% due to GEMV-like computation patterns that cannot hide HBM weight-loading latency. O-Proj/FFN-Down are execution-bound, with Execution Dependency reaching 51% due to extreme register pressure (≈130 live registers) that limits schedulable warps and prevents HMMA latency hiding."
VERBATIM_QUOTE 5 (Section 5.3.2 "Operator-Level Memory Patterns"): "Attention Kernels suffer severe locality degradation, with L2 hit rates dropping 81.9% (Chat) and 73.3% (Summary), while DRAM utilization increases up to 76.4%." ... "FFN Kernels exhibit a bottleneck shift, with DRAM utilization rising 62% and arithmetic intensity plummeting from ∼95 to ∼8 FLOP/byte."
VERBATIM_QUOTE 6 (Section 5.2.2, the non-GEMM overhead term): "Additionally, LayerNorm shows significantly increased Synchronization stalls (from 11.0% to 27.7%), revealing: [18] Small-batch reduction operations in Decode exacerbate synchronization overhead, as early-finishing warps wait at barriers, creating tail latency effects."
CLAIM: Decode's FFN-vs-Attention dominance is CONTEXT-DEPENDENT, with a crossover: FFN-dominated at typical contexts for large models, Attention-dominated at long contexts (linear KV traversal). Critically, decode's kernels are NOT uniformly bandwidth-bound — O-Proj/FFN-Down are execution-bound on register pressure, and LayerNorm sync stalls nearly triple.
HAS_MEASUREMENT: YES — hardware counters. Reports measured AI (prefill ≈55-100, decode ≈1-10 FLOP/byte; attention AI reaching 319.3 for Llama3-8B and 382.1 for Qwen2.5-32B in long-context prefill), measured stall-cause percentages, DRAM bandwidth utilization, L2 hit rates, and median per-kernel Prefill→Decode deltas (DRAM utilization +48.2%, L2 hit rate -54.0% in Chat). AttnCore DRAM utilization +38.1% in decode under long context while prefill -9.3%. Granularity is per-KERNEL (QKV-Proj, AttnCore, O-Proj, FFN-Up, Act(SiLU), FFN-Down, LayerNorm) — exactly the attention-vs-MLP split requested.
HAS_CONTROL: yes — Prefill vs Decode is a matched within-paper control at identical model/hardware, and Chat vs Summary (short vs long context) is a matched context-length control.
RELEVANCE_TO_DIRECTION6: THE best single citation for per-operator decode attribution and for the claim that "bandwidth-bound decode" is an over-generalization. Directly supplies the attention-vs-MLP bandwidth split with context-length dependence.
```

```
URL: https://arxiv.org/html/2605.30571v1 (HTML full text) and https://arxiv.org/abs/2605.30571
STATE: arXiv HTML v1 full text (500,724 bytes fetched) + abs page. v1, 28 May 2026, cs.AR. Single author: Josef Chen (KAIKAKU).
TITLE_READBACK: [2605.30571] Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode
VERBATIM_QUOTE (HTML full text, Section 4.1 / Figure 2, the per-decoder-block byte table for Qwen-2.5-7B, ctx=2048): "MLP weights = 407.4 MB ≈ 87% of block bytes ... HBM bytes per block weights ≈ 466 MB MLP 407.4 attn 58.8 KV 4.2 MB Per block: ≈ 470 MB HBM read per decode step. 28 blocks: ≈ 13.16 GB per step. At H100 3.35 TB/s peak: floor = 3.93 ms. Observed: 14.83 ms. R_floor = 0.27 (27% of peak)."
VERBATIM_QUOTE 2 (HTML full text, Figure 2 caption): "The SwiGLU MLP dominates the weight footprint at ≈407 MB per block out of ≈470 MB total. The KV cache contribution at ctx = 2048 is ≈4.2 MB per block, small relative to weights but growing linearly with context."
VERBATIM_QUOTE 3 (HTML full text, the per-operator byte list): "Q proj 3584→3584 weights 25.7 MB | K proj 3584→512 weights 3.7 MB | V proj 3584→512 weights 3.7 MB | RoPE apply (Q, K) cos/sin tables, negligible | SDPA attention Q[1,28,128]·K,V[1,4,ctx,128] KV cache load ctx=2048 → ≈4.2 MB (key + value, bf16) | O proj 3584→3584 weights 25.7 MB | residual + RMSNorm negligible | SwiGLU MLP gate_proj 3584→18944 135.8 MB | up_proj 3584→18944 135.8 MB | down_proj 18944→3584 135.8 MB"
CLAIM: Gives the exact per-operator byte split inside one decode step. At ctx=2048 the KV cache read is ~1% of decode bytes; the SwiGLU MLP weights are ~87%. This directly refutes any framing that treats "weights read" and "KV cache read" as comparable terms at moderate context.
HAS_MEASUREMENT: The byte table is ANALYTIC (analytic byte counts), but it is paired with MEASURED step time (median over 30 measured decode steps). Measured numbers: H100 ctx=2048 p50 14.83 ms; measured ratio R_floor = 0.270.
HAS_CONTROL: yes — CUDA Graphs A/B with pre-registered falsification thresholds, N=10 fresh sessions.
RELEVANCE_TO_DIRECTION6: Supplies the numeric weights-vs-KV byte split per operator — the concrete answer to "how much of decode is actually KV traffic".
```

```
URL: https://arxiv.org/html/2607.28824v1 (abs page fetched to verify title) — secondary/weak
STATE: arXiv abs page verified. v1.
TITLE_READBACK: [2607.28824] Characterizing LLM Kernel Access and Memory Interaction in Multi-Partition NUMA GPUs
VERBATIM_QUOTE (from the abs-page abstract): "we analyze performance-critical LLM kernel implementations spanning weight projection, mixture-of-experts, and attention variants of state-of-the-art serving engines to present a characterization of data access patterns in multi-partition GPUs."
CLAIM: Workgroup-level data access/sharing patterns per LLM kernel category (weight projection, MoE, attention). Relevant to *operand* traffic attribution, but on multi-partition NUMA GPUs, not standard HBM decode.
HAS_MEASUREMENT: YES but via a cycle-level simulator plus memory-trace analysis, NOT on-hardware DRAM counters. Search snippet suggested "The weight matrix dominates the memory footprint at 2 MB per WG" but I did NOT read that sentence on a page I fetched, so I do not report it as verbatim.
HAS_CONTROL: unclear
RELEVANCE_TO_DIRECTION6: Peripheral. Operand-level attribution on an unusual (multi-partition) substrate.
```

## Topic 2 — Roofline analyses comparing PREDICTED bytes (weights + KV) to MEASURED bandwidth

```
URL: https://arxiv.org/abs/2605.30571 and https://arxiv.org/html/2605.30571v1
STATE: as above (v1, 28 May 2026, cs.AR)
TITLE_READBACK: [2605.30571] Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode
VERBATIM_QUOTE (from the abs-page abstract): "This workload is usually described as memory-bandwidth-bound. Each decode step streams model weights and the active KV cache, so latency should scale with peak HBM bandwidth. We show that this account is true but incomplete."
VERBATIM_QUOTE 2 (from the abs-page abstract): "On the headline Qwen-2.5-7B ctx=2048 cell, an L4 reaches roughly 81 percent of its analytic memory floor, while an H100 reaches only 27 percent. Physical-AI decode is memory-dominated, but faster memory does not translate into proportional latency gains."
VERBATIM_QUOTE 3 (HTML full text, Section 3.4 "The observed-over-floor ratio"): "t_floor(G,M,ctx) = (W(M) + K(M,ctx)) / B_peak(G), where W(M) is the bf16 weight footprint of the model in bytes ... and K(M,ctx) = 2 · n_layers · n_kv_heads · d_head · ctx · 2 bytes is the per-step KV bytes touched (the leading 2 covers K and V, the trailing 2 covers bf16 element size)."
VERBATIM_QUOTE 4 (HTML full text, Table 8): "Table 8: Cross-GPU per-step accounting, Qwen-2.5-7B ... bw_util is analytic streamed bytes per step (weights + active KV slab) divided by spec-sheet B_peak time; unprofiled uses the median production step time as denominator, profiled uses torch.profiler step time. cpu/cuda is the fraction of profiler wall time spent on the CPU side launching and synchronising kernels relative to GPU kernel time. [H100 2048: p50 unprof 14.83 ms, p50 prof 54.38 ms, bw_util(unprof) 26.9%, cpu/cuda 0.75] [L4 2048: 62.96, 63.67, 79.4%, 0.35]"
VERBATIM_QUOTE 5 (HTML full text, Section 8, the honesty caveat): "Nsight Compute (ncu) was blocked on Modal cloud containers: ncu requires host driver access we did not have. We use torch.profiler with analytic byte counts as the substitute."
CLAIM: THE direct test of the folk model. It constructs exactly `t_floor = (W+K)/B_peak` (predicted bytes / peak spec bandwidth) and compares to measured step time. Result: the fraction of the analytic floor actually achieved FALLS as peak bandwidth rises (H100 27%, A100 31%, L40S 72%, L4 81%). The residual is not explained by more traffic — it is per-kernel CPU launch overhead (~30 µs launch vs ~10 µs compute per kernel on H100, 10 kernels/block, 28 blocks → ≈11.0 ms launch + 3.8 ms compute = 14.83 ms).
HAS_MEASUREMENT: YES — extensively. 44 valid cells (3 models × 4 GPUs × 4 context lengths, 4 L4 OOMs excluded). Measured: R_floor per cell from median of 30 measured decode steps. Table 1 R_floor, Qwen-2.5-7B: ctx 2048 → H100 0.270, A100-80GB 0.311, L40S 0.723, L4 0.810; ctx 4096 → 0.271 / 0.369 / 0.711 / 0.722; ctx 8192 → 0.274 / 0.312 / 0.666 / OOM; ctx 16384 → 0.235 / 0.243 / ... Measured CUDA Graphs speedup: H100 1.259x, 95% bootstrap CI [1.253, 1.267], cross-session CV 0.9%; L4 only 1.028x. Measured quantization: bnb-nf4 59.36 ms/step, AutoAWQ+Marlin 45.24 ms/step, GPTQ+ExLlamaV2 17.36 ms/step, vs bf16 baseline 62.32 ms/step — i.e. common 4-bit paths do NOT recover the expected 4x weight-traffic reduction.
HAS_CONTROL: YES — and it is the strongest control design in this whole set: a pre-registered CUDA Graphs A/B that "touches the launch term and only the launch term", with an explicit null prediction on L4 (which was confirmed: 1.028x). Also uses torch.profiler instrumentation-bias control (H100 inflates 16.97→54.38 ms under profiler, 3.2x; L4 63.0→63.67 ms, 1.01x) and reports unprofiled bw_util as headline.
RELEVANCE_TO_DIRECTION6: THE central artifact. It is the paper that shows the "decode = (weights+KV)/bandwidth" formula is *measurably* incomplete, and quantifies by how much (3x overestimate of achievable speedup on fast silicon).
```

```
URL: https://arxiv.org/abs/2607.29575 and https://arxiv.org/html/2607.29575v1
STATE: arXiv abs page + arXiv HTML v1 full text (263,131 bytes fetched; 86,183 chars). v1, 31 Jul 2026. Authors: Pol G. Recasens, Ferran Agullo, Yue Zhu, Chen Wang, Jordi Torres, Josep Ll. Berral (UPC + IBM Research).
TITLE_READBACK: [2607.29575] SLIM: Saturation-Aware Lightweight Performance Modeling for LLM Serving
VERBATIM_QUOTE (from the abs-page abstract / HTML abstract, identical text): "Previous studies have attributed this behavior to HBM/DRAM bandwidth limitations, but the underlying causes have primarily been supported by conceptual arguments or high-level performance observations. As our first contribution, we present a detailed GPU characterization using hardware profiling techniques, demonstrating that throughput saturation originates in the attention kernels during the decode phase."
VERBATIM_QUOTE 2 (abs abstract): "Specifically, we show that their nearly constant arithmetic intensity as active-context lengths increases—not merely larger batch sizes—drives DRAM-bandwidth saturation, while the achieved compute throughput remains far below the hardware limit."
VERBATIM_QUOTE 3 (HTML full text, Section IV Methodology): "In offline mode, used for the low-level GPU profiling in Section V with Nsight Systems and Nsight Compute, we instantiate vLLM directly in Python, injects synthetic requests into the scheduler and then runs the prefill and decode phases with explicit llm_engine.step() calls."
CLAIM: An explicit CORRECTION paper: it states the prior DRAM-bandwidth attribution rested on "conceptual arguments or high-level performance observations", and replaces it with a profiling-backed claim that ATTENTION KERNELS in decode (whose arithmetic intensity stays nearly constant as active context grows) are what saturate DRAM — not batch size per se. Matmul kernels by contrast DO benefit from larger active contexts.
HAS_MEASUREMENT: YES — hardware profiling with **Nsight Systems and Nsight Compute** on vLLM 0.15.1, offline mode with explicit `llm_engine.step()` for controlled profiling. Also reports MAPE reductions of 79.3% vs baselines LLMVisor and Imai et al., and identifies "up to 55 GB of GPU memory allocation that can be avoided". Note: I grepped for literal `dram__` counter names and found 0 hits, so the paper reports Nsight-derived metrics but I did not read a literal `dram__bytes` counter name on the page.
HAS_CONTROL: yes — held-out evaluation across unseen input length, unseen output length, unseen model, and unseen model+output length without refitting; plus baselines LLMVisor and Imai et al.
RELEVANCE_TO_DIRECTION6: Provides the "conceptual arguments" critique verbatim — extremely useful rhetorical anchor — and relocates decode saturation to the attention kernel specifically.
```

```
URL: https://arxiv.org/html/2503.06433v1 (HTML full text) and https://arxiv.org/abs/... (ar5iv also resolves)
STATE: arXiv HTML v1 full text (216,928 bytes via ar5iv; 245,103 bytes via arxiv.org/html). v1, 09 Mar 2025, cs.DC. ICML-style.
TITLE_READBACK: [2503.06433] Seesaw: High-throughput LLM Inference via Model Re-sharding
VERBATIM_QUOTE (HTML full text, Appendix A.1 "Runtime Break-Down" — this is the folk breakdown stated as an equation): "The runtime of each decoding layer can be divided into three components: 1) data movement (T_dm) from GPU global memory (HBM) to compute units, which includes transferring weights (T_dm^linear) and KV cache (T_dm^attn), 2) computation T_comp, including T_comp^linear and T_comp^attn, and 3) communication cost T_nw (nw stands for network), primarily arising from the all-reduce operation in tensor parallelism."
VERBATIM_QUOTE 2 (same section): "Based on the roof-line model, the runtime of each layer can be approximated as T_L = max(T_dm^linear, T_comp^linear) + max(T_dm^attn, T_comp^attn) + T_nw."
VERBATIM_QUOTE 3 (same section, "Data Movement."): "The runtime of data movement can be approximated as transferred data volume divided by the bandwidth, which is the HBM bandwidth for GPUs. For linear layers, the transferred data is mostly weight matrices, of which the size is 2W bytes, which is constant."
CLAIM: States the folk attribution explicitly and formally: per-decode-layer time = max(weight-move, linear-compute) + max(KV-move, attention-compute) + network, with data movement = volume / HBM bandwidth. This is the *canonical written form* of "decode ≈ weights read + KV cache read, both bandwidth-bound".
HAS_MEASUREMENT: MIXED. The A.1 break-down is explicitly an APPROXIMATION from the roofline model, NOT instrumented counters ("can be approximated as"; the word "measured" appears in the paper only for end-to-end stage throughput and speedup breakdown on A10/L4 GPUs — e.g. "we measured the runtime of each stage" for TP4/PP4/TP2PP2 on CodeLLaMA-34B, arxiv-summarization, four A10 GPUs). No per-layer DRAM byte counters.
HAS_CONTROL: yes for the systems evaluation (matched parallelism configurations TP4/PP4/TP2PP2 with and without chunked prefill).
RELEVANCE_TO_DIRECTION6: HIGH VALUE AS THE NAMED TARGET. This is a citable, verbatim instance of the folk decomposition being used as an approximation without byte-level instrumentation — i.e. the "assumed, not measured" baseline that Direction 6 can position against.
```

```
URL: https://arxiv.org/abs/2512.22066
STATE: arXiv abs page. v1, 26 Dec 2025.
TITLE_READBACK: [2512.22066] Prefill vs. Decode Bottlenecks: SRAM-Frequency Tradeoffs and the Memory-Bandwidth Ceiling
VERBATIM_QUOTE (from the abs-page abstract): "Our simulation methodology combines OpenRAM for energy modeling, LLMCompass for latency simulation, and ScaleSIM for systolic array operational intensity."
VERBATIM_QUOTE 2 (abs abstract): "We quantitatively explore the memory-bandwidth bottleneck, demonstrating that while high operating frequencies reduce prefill latency, their positive impact on memory-bound decode latency is capped by the external memory bandwidth."
CLAIM: Decode latency is capped by external memory bandwidth; memory bandwidth acts as a "performance ceiling" and rising compute frequency only helps until the workload becomes memory-bound. Reasserts (does not challenge) the memory-bound decode account.
HAS_MEASUREMENT: NO — it is a SIMULATION study (OpenRAM + LLMCompass + ScaleSIM), not on-hardware measurement. Also reports an "optimal hardware configuration ... high operating frequencies (1200MHz-1400MHz) and a small local buffer size of 32KB to 64KB".
HAS_CONTROL: yes within the simulated design space (parameter sweep of SRAM size × frequency).
RELEVANCE_TO_DIRECTION6: A useful example of the *uncorrected* folk position still being published from simulation — contrast against 2605.30571/2607.29575 which measure.
```

## Topic 3 — Papers that explicitly CORRECT or REFUTE a common mis-attribution

```
URL: https://arxiv.org/abs/2605.30571 (abstract) — see Topic 2 entry for full detail
TITLE_READBACK: [2605.30571] Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode
VERBATIM_QUOTE (HTML full text, Section 1 Introduction, the named "standard account"): "The standard account of batch-1 decode is that it is HBM-bandwidth-bound: each decode step streams the model weights and the per-layer KV cache through global memory once, arithmetic intensity is too low to keep the SMs busy, and step time is approximately (W + K)/B_peak. This account motivates a sizeable fraction of recent serving work: KV cache compression, weight quantisation and KV offloading."
VERBATIM_QUOTE 2 (same paragraph, immediately after): "This paper is a measurement study, not a modeling paper."
CLAIM: Names the folk account, states it "is true but incomplete", and identifies the missing term as CPU-side per-kernel launch overhead that becomes visible only on high-bandwidth GPUs. Implication: KV-cache compression and weight quantization — the two techniques the folk account motivates — do not deliver proportional latency gains on fast silicon (measured: L4 quantized paths fail to recover the expected 4x weight-traffic reduction).
HAS_MEASUREMENT: YES — see Topic 2. 44 cells, R_floor 0.235-0.810 depending on GPU/context, CUDA Graphs A/B 1.259x [1.253, 1.267] on H100 vs 1.028x on L4.
HAS_CONTROL: YES — pre-registered falsification, matched A/B, and an explicit predicted null on L4 that was confirmed.
RELEVANCE_TO_DIRECTION6: The clearest existing refutation. Direction 6 should cite this as the closest prior art and differentiate on what remains unmeasured (on-hardware DRAM counters; the launch overhead here is inferred from torch.profiler because ncu was blocked).
```

```
URL: https://arxiv.org/abs/2607.29575 — see Topic 2 entry
TITLE_READBACK: [2607.29575] SLIM: Saturation-Aware Lightweight Performance Modeling for LLM Serving
VERBATIM_QUOTE (abs abstract): "Previous studies have attributed this behavior to HBM/DRAM bandwidth limitations, but the underlying causes have primarily been supported by conceptual arguments or high-level performance observations."
CLAIM: The single most quotable sentence in this set for the "prior work did not actually measure it" argument. It explicitly characterizes the prior bandwidth attribution as conceptually argued rather than measured.
HAS_MEASUREMENT: YES (Nsight Systems + Nsight Compute on vLLM).
HAS_CONTROL: yes (held-out model/length scenarios + model baselines).
RELEVANCE_TO_DIRECTION6: Second pillar of the correction literature.
```

```
URL: https://arxiv.org/abs/2512.01644 — see Topic 1 entry
TITLE_READBACK: [2512.01644] A Systematic Characterization of LLM Inference on GPUs
VERBATIM_QUOTE (HTML full text, Section 5.2.2): "Further analysis reveals heterogeneous bottlenecks among Decode's GEMM kernels: FFN-Up is memory-bound, with Memory Dependency reaching 58% due to GEMV-like computation patterns that cannot hide HBM weight-loading latency. O-Proj/FFN-Down are execution-bound, with Execution Dependency reaching 51% due to extreme register pressure (≈130 live registers) that limits schedulable warps and prevents HMMA latency hiding."
VERBATIM_QUOTE 2 (HTML full text, Section 5.2.2): "Additionally, LayerNorm shows significantly increased Synchronization stalls (from 11.0% to 27.7%), revealing: [18] Small-batch reduction operations in Decode exacerbate synchronization overhead, as early-finishing warps wait at barriers, creating tail latency effects."
CLAIM: A measured, counter-based challenge to "decode is bandwidth-bound" as a blanket statement: within a single decode step, different operators are bound by different things (HBM weight-loading latency for FFN-Up; register-pressure instruction dependency for O-Proj/FFN-Down; barrier synchronization for LayerNorm). This is the "activation/non-GEMM overhead is non-negligible" evidence.
HAS_MEASUREMENT: YES — warp-level issue-stall analysis, DRAM bandwidth utilization, L2 hit rates, per-kernel medians. Pipeline Busy in decode collapses to 3~4%; both phases spend 70%-80% of cycles stalled (GPU Execution: Prefill 27%, Decode 24%).
HAS_CONTROL: yes — Prefill vs Decode matched at identical model/hardware; Chat vs Summary matched context-length control; single-GPU A100 vs Jetson AGX Orin edge comparison.
RELEVANCE_TO_DIRECTION6: Supplies the "not all decode traffic is uniform" measured correction, including an explicit non-GEMM (LayerNorm) overhead term the folk model omits entirely.
```

```
URL: https://arxiv.org/abs/2405.04532 — see Part A
TITLE_READBACK: [2405.04532] QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving
VERBATIM_QUOTE (abs abstract): "We uncover a critical issue: existing INT4 quantization methods suffer from significant runtime overhead (20-90%) when dequantizing either weights or partial sums on GPUs."
CLAIM: Corrects the implicit assumption that reducing weight bytes proportionally reduces decode time — the real cost migrates to dequantization on low-throughput CUDA cores.
HAS_MEASUREMENT: YES (20-90% overhead; 1.2x-3.5x measured speedups).
HAS_CONTROL: yes (vs TensorRT-LLM).
RELEVANCE_TO_DIRECTION6: Supporting precedent that byte-reduction ≠ time-reduction in decode.
```

```
URL: https://arxiv.org/abs/2606.07713
STATE: arXiv abs page. v1, 5 Jun 2026.
TITLE_READBACK: [2606.07713] Attention at the Theoretical Minimum: A Mathematics of Arrays Framework for Memory-Optimal Transformer Kernels
VERBATIM_QUOTE (from the abs-page abstract): "Its standard implementation incurs quadratic memory traffic in the sequence length n, and DRAM accesses cost 100--1000× more energy than arithmetic operations on contemporary hardware, so any analysis focused solely on FLOP counts fundamentally mischaracterises the bottleneck."
VERBATIM_QUOTE 2 (abs abstract): "The DNF achieves O(n_dk + n_dv) data movement versus O(n^2 + n_dk + n_dv) for the standard implementation"
CLAIM: A FLOP-centric analysis mischaracterizes the bottleneck; attention's problem is quadratic DATA MOVEMENT, not arithmetic. Correction is against FLOP-counting, not against the bandwidth account.
HAS_MEASUREMENT: NO on hardware — "is verified numerically against PyTorch at full double-precision floating-point on concrete inputs" is a numerical-correctness check; the performance claim is a projection: "A predictive performance model projects 2--100× speedup and 2--50× energy reduction".
HAS_CONTROL: unclear
RELEVANCE_TO_DIRECTION6: Useful only as a framing citation ("DRAM accesses cost 100-1000x more energy than arithmetic") — it is a theory/projection paper, NOT a measurement of decode traffic.
```

## Topic 4 — KV-cache read volume as an explicit function of context length during decode

```
URL: https://arxiv.org/abs/2605.30571 and https://arxiv.org/html/2605.30571v1
TITLE_READBACK: [2605.30571] Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode
VERBATIM_QUOTE (HTML full text, Section 3.4): "K(M,ctx) = 2 · n_layers · n_kv_heads · d_head · ctx · 2 bytes is the per-step KV bytes touched (the leading 2 covers K and V, the trailing 2 covers bf16 element size)."
VERBATIM_QUOTE 2 (HTML full text, Section 3.3 and the OOM footnote, worked values as a function of ctx): "Qwen-2.5-7B at bf16 has W = 15.23 GB and per-token KV bytes 2·28·4·128·2 = 56 KB; at ctx = 8192 that is 0.47 GB of KV, totalling 15.70 GB. Mistral-7B at bf16 has W = 14.50 GB and 128 KB per-token KV; at ctx = 8192 that is 1.07 GB of KV, totalling 15.57 GB."
VERBATIM_QUOTE 3 (HTML full text, Figure 2 caption): "The KV cache contribution at ctx = 2048 is ≈4.2 MB per block, small relative to weights but growing linearly with context."
CLAIM: KV read volume is explicitly linear in ctx (56 KB/token/layer-set for Qwen-2.5-7B GQA with 4 KV heads), and the paper measures contiguous context lengths 2048 / 4096 / 8192 / 16384 across four GPUs, so the KV term's growth against the constant weight term is directly observable in the measured R_floor trends (e.g. H100 Qwen-2.5-7B: 0.270 at 2048 → 0.235 at 16384).
HAS_MEASUREMENT: YES for step time (median of 30 measured decode steps per cell); the KV byte volume itself is ANALYTIC (explicitly called "analytic byte counts"), since ncu was unavailable.
HAS_CONTROL: yes (CUDA Graphs A/B; four-GPU cross-comparison; four context lengths).
RELEVANCE_TO_DIRECTION6: Direct source for "KV read volume as a function of context length", with the honest caveat that the byte volume is computed, not counter-read.
```

```
URL: https://arxiv.org/html/2607.09052v1 and https://arxiv.org/abs/2607.09052
STATE: arXiv HTML v1 full text (454,891 bytes fetched) + abs page. v1, 10 Jul 2026, cs.LG. Authors: Alexander Tian, Aditya Ghai, Sanjit Neelam, Zaal Vasania, Akshay Mishra (MatX).
TITLE_READBACK: [2607.09052] COBS: Cumulant Order Block Sparse Attention
VERBATIM_QUOTE (HTML full text, Section 7.5 "KV cache read traffic study"): "We report the KV cache read traffic per decode step and layer (Table 3). The accounting uses a 32k context with H = 4 KV heads of dimension D = 128 and the same L = 32, top-k = 16, and 256-token window from setup."
VERBATIM_QUOTE 2 (HTML full text, Table 3 caption): "Table 3: Per-decode-step, per-layer KV cache read traffic at 32k, by branch (KiB; 1 KiB = 1024 bytes)."
VERBATIM_QUOTE 3 (HTML full text, Table 3 rows, per-layer KiB): "Dense (full attention) – – – 65,536 → 65,536 KiB, 18.29× less vs dense is n/a; OSA (mass oracle) 33,792 / 1024 / 512 / 1024 → 36,352 KiB (1.80× less than dense); NSA MLP 1024 / 1024 / 512 / 1024 → 3584 KiB (18.29× less than dense); NSA Quest 3072/1024/512/1024 → 5632 KiB (11.64× less); COBS full-space r=4 (bf16) 5120/1024/512/1024 → 7680 KiB (8.53× less); COBS full-space r=6 (bf16) 7168/1024/512/1024 → 9728 KiB (6.74× less); COBS full-space r=4 (FP4) 2112/1024/512/1024 → 4672 KiB (14.03× less); COBS full-space r=6 (FP4) 2656/1024/512/1024 → 5216 KiB (12.56× less); COBS (subspace s≈85, r=4, FP4) 1767/1024/512/1024 → 4327 KiB (15.15× less than dense, 1.21× more than NSA MLP)."
VERBATIM_QUOTE 4 (HTML full text, Figure 1 discussion): "attention ... reading the key–value (KV) cache: at each decode step, attention reads the keys and values of every past token, so decoding is limited by memory bandwidth rather than compute and leaves hardware underutilized."
CLAIM: Provides a per-decode-step, per-layer KV cache read traffic accounting at a fixed 32k context, decomposed BY BRANCH (summary keys / summary values / local window / fine-grained full K,V). Dense = 65,536 KiB per layer per decode step; sparse methods are 6.7x-18.3x lower. Also asserts the standard framing that decode attention is bandwidth-limited.
HAS_MEASUREMENT: NO on-hardware counters — this is an explicit ACCOUNTING (arithmetic byte counts), and the paper itself flags this: the Limitations list includes "KV read accounting". The accuracy claims (RULER 32k mean score 0.2999 NSA baseline → 0.8195 COBS, dense 0.9040) ARE measured.
HAS_CONTROL: yes — controlled NSA baseline comparison and dense-attention reference at matched 32k context, top-k=16, L=32, GQA-4, D=128.
RELEVANCE_TO_DIRECTION6: The clearest worked example of "KV cache read volume per decode step per layer as an explicit function of context" (at fixed 32k) with a branch-level breakdown — but it is an accounting model, so it cannot itself settle the measured-bandwidth question.
```

## Topic 5 — Attention-vs-MLP bandwidth split in decode

Covered by the three artifacts above; summarized here for directness:

- **2512.01644** — the actual measured split. Attention kernels: L2 hit rate drops 81.9% (Chat) / 73.3% (Summary), DRAM utilization up to +76.4%, and a further **+38.1% DRAM utilization in Decode under long-context (Summary)**. FFN kernels: DRAM utilization **+62%**, arithmetic intensity falling from **~95 to ~8 FLOP/byte**. Dominance crossover is context- and model-scale-dependent (8B Attention-dominated, 32B FFN-dominated at typical context; both Attention-dominated at very long context).
- **2605.30571** — the byte split: per block at ctx=2048, **MLP 407.4 MB vs attn 58.8 MB vs KV 4.2 MB** (total ≈470 MB); MLP ≈87% of block bytes.
- **2607.29575** — attention is the operator that saturates DRAM in decode (nearly constant arithmetic intensity as active context grows), whereas matmuls gain utilization from larger active context.
- **2607.09052** — attention-side KV read accounting: 65,536 KiB/layer/step dense at 32k.

Corroborating phase-level (not intra-decode) statements:
- **2311.18677 Splitwise** — abs abstract: "Based on our extensive characterization, we find that there are two main phases during an LLM inference request: a compute-intensive prompt computation, and a memory-intensive token generation, each with distinct latency, throughput, memory, and power characteristics."
- **2403.02310 Sarathi-Serve** — abs abstract: "Prefill iterations have high latency but saturate GPU compute due to parallel processing of the input prompt. In contrast, decode iterations have low latency but also low compute utilization because a decode iteration processes only a single token per request."
- **2605.19775 Understanding Inference Scaling for LLMs** — abs abstract: "our analysis reveals that data parallelism is throughput efficient for small models but hits a capacity trap on reasoning workloads as KV-cache fragmentation forces early throttling resulting in sub-optimal compute utilization." (capacity-bound, not bandwidth-bound, framing)
- **2608.06723 HYMELL** — abs abstract: "HYMELL models LLM execution through a three-level hierarchy: analytical estimation of primitive operations, ML prediction of higher-level components, and an end-to-end model that captures system-level overheads across both prefill and decode phases. ... for LLaMA 3 8B, it attains less than 5% error for both prefill and decode phases." (a modeling paper; useful as a baseline, not as measurement)

---

# PART C — CONFIRMED SEARCH FABRICATIONS / MISMATCHES (4 cases this session)

This project's warning about fabricated IDs and mismatched titles is confirmed again. Do NOT trust any
search-returned (ID, title) pair without curl readback.

1. **2407.05858** — prompt reported "LLM inference roofline / analytical model". Actual title:
   `Fast On-device LLM Inference with NPUs`. **MISMATCH.**
2. **2407.00088** — prompt reported "QServe or similar W4A8 quantization". Actual title:
   `T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge`. **MISMATCH.**
   (Real QServe = 2405.04532, verified.)
3. **2401.08671** — prompt offered the alternative "or DistServe". Actual title:
   `DeepSpeed-FastGen: ...`. The alternative was wrong; the first title was right.
   (Real DistServe = 2401.09670, verified.)
4. **2604.05438** — `web_search` returned this ID under the title
   *"Top-K Retrieval with Fixed-Size Linear-Attention Completion: Backbone- and KV-Format-Preserving
   Attention for KV-Cache Read Reduction"*. Readback at `https://arxiv.org/abs/2604.05438` gives:
   `[2604.05438] Residual-Mass Accounting for Partial-KV Decoding` (submission history shown on page:
   v1 Tue, 7 Apr 2026; v2 Thu, 7 May 2026). **MISMATCH — the search title is fabricated or belongs to
   a different work.** Do not cite the "Top-K Retrieval with Fixed-Size Linear-Attention Completion"
   title as 2604.05438.

Also note: **2402.15627** resolves correctly to `MegaScale: Scaling Large Language Model Training to
More Than 10,000 GPUs` — it is a TRAINING paper and should be discarded for Direction 6, as suspected.

---

# PART D — FETCH ROUTE HEALTH (observed this session)

| Route | Result |
|---|---|
| `https://arxiv.org/abs/<id>` via curl with the prescribed UA | **WORKS** — used for all 30+ title readbacks. Reliable. |
| `https://arxiv.org/html/<id>v1` via curl | **WORKS** — full text, useful for verbatim section quotes (used for 2512.01644, 2605.30571, 2607.09052, 2607.29575, 2402.16363v6). |
| `https://ar5iv.labs.arxiv.org/html/<id>` via curl | **WORKS** — alternate full-text route; note `<title>` is often the first section name, not the paper title, so do not use ar5iv for title readback. |
| `https://api.semanticscholar.org/graph/v1/paper/search` | **FAILS** — HTTP 429 "Too Many Requests" on every attempt. |
| `https://www.semanticscholar.org/paper/...` | **FAILS** — HTTP 202 with 0 bytes (bot challenge). |
| `https://lite.duckduckgo.com/lite/?q=...` | **FAILS** — bot challenge page ("Select all squares containing a duck"). |
| `http://export.arxiv.org/api/query` | **FAILS** — curl exit code 28 / HTTP 000 (connection failure), consistent with the reported 429. |
| `https://arxiv-org.ezproxy.obspm.fr/...` (mirror seen in search results) | **FAILS for HTML** — returns a 751-byte "Cookie Required" page. |
| `https://arxiv.org/list/cs.LG/recent` | HTTP 200 (reachable) but not a search interface — no query support. |
| `https://api2.openreview.net/notes/search` | HTTP 200 (reachable). |
| `https://huggingface.co/papers` | HTTP 200 (reachable). |

Total distinct arXiv URLs fetched successfully this session: **35+** (well over the 20 requested).

Notable in-paper tooling limitation worth quoting in the Direction 6 write-up: **2605.30571** reports
that Nsight Compute could not be used at all in its environment —
"Nsight Compute (ncu) was blocked on Modal cloud containers: ncu requires host driver access we did not
have. We use torch.profiler with analytic byte counts as the substitute." This means the strongest
existing correction paper still does **not** have direct `dram__bytes` counter evidence, which is a
live gap Direction 6 can occupy.

---

# PART E — BOTTOM LINE FOR DIRECTION 6

1. The folk breakdown **has been written down explicitly** — best citation is **2503.06433 (Seesaw),
   Appendix A.1**, with the equation `T_L = max(T_dm^linear, T_comp^linear) + max(T_dm^attn, T_comp^attn) + T_nw`,
   and **2402.16363 (LLM Inference Unveiled)**, which supplies the roofline justification but contains
   **zero measurement validation**.
2. It **has been measured and corrected** — **2605.30571** constructs exactly `(W+K)/B_peak` and finds
   the realized fraction collapses from 81% (L4) to 27% (H100); **2607.29575** says prior bandwidth
   attributions were "supported by conceptual arguments"; **2512.01644** shows decode operators are
   heterogeneously bound (FFN-Up memory-bound at 58% Memory Dependency, O-Proj/FFN-Down execution-bound
   at 51% Execution Dependency, LayerNorm sync stalls 11.0%→27.7%).
3. What is **still open**: no artifact in this set reports direct on-hardware DRAM byte counters
   (`dram__bytes` / NVML) reconciled against the predicted weights+KV volume. Every KV-volume figure
   found (2605.30571, 2607.09052, 2503.06433, 2402.16363) is **analytic/accounting**, and the one paper
   that tried `ncu` was blocked. The 2605.30571 correction attributes the gap to CPU launch overhead;
   it does not measure DRAM traffic. **That reconciliation is the defensible novelty for Direction 6.**
