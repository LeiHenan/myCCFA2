# Question B5 — Controlled cross-GPU-generation comparison for LLM inference

Date: 2026-09-14. Subagent B5 report.

**Overlap check against parent's already-covered list** (RoundPipe 2604.27085, 2311.03687, Albireo 2606.01927, CUDAHercules 2605.08467, LLMQ 2512.15306): **zero overlap**. Everything below is new.

All arXiv IDs below were verified by fetching `https://arxiv.org/abs/<id>` and reading `<meta name="citation_title">` back. IDs are marked VERIFIED / UNVERIFIED explicitly.

---

## 1. Candidate papers

### 1.1 Cross-generation, same system, unified stack — LLM inference serving

**A. SlideSparse: Fast and Flexible (2N-2):2N Structured Sparsity**
- arXiv **2603.05232** — **VERIFIED** (title read back)
- https://arxiv.org/abs/2603.05232 | submitted 2026-03-05 (v1)
- Coverage: **6 GPUs, 4 architecture generations**, one code path, one Docker stack (vLLM 0.13.0, CUDA 12.9, PyTorch 2.9.0), integrated into vLLM.
- Verbatim (arXiv HTML v1, §5.1):
  > "We test on six NVIDIA GPUs across three architecture generations: Datacenter: A100 (80GB, Ampere), H100 (80GB, Hopper), B200 (180GB, Blackwell); Consumer: RTX 4090 (24GB, Ada Lovelace), RTX 5080 (16GB, Blackwell); Embedded: DGX Spark GB10 (128GB, Blackwell)."
- Verbatim (Appendix D.1) — note it says *four* generations and gives sm numbers:
  > "Our evaluation spans six NVIDIA GPU platforms across four architecture generations: • Datacenter: A100 80GB (Ampere, sm80), H100 80GB (Hopper, sm90), B200 180GB (Blackwell, sm100) • Consumer: RTX 4090 24GB (Ada Lovelace, sm89), RTX 5080 16GB (Blackwell, sm120) • Embedded: DGX Spark GB10 128GB (Blackwell, sm121, aarch64)"
- Per-generation numbers exist, e.g. (§5.3 / D.4): "On RTX 4090 (FP8 prefill), 6:8 reaches 1.18–1.19×"; "H100 FP8: Qwen-14B achieves 1.24–1.31×"; "B200 FP8: reaches 1.23–1.28×"; and RTX 4090 "speedups dropping to 0.10–0.27× at certain M values—likely due to API implementation issues rather than fundamental performance limitations."

**B. Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures**
- arXiv **2604.09048** — **VERIFIED**
- https://arxiv.org/abs/2604.09048 | submitted 2026-04-10
- Coverage: **10 GPUs, 5 architectures**, 50 LLMs, >5,000 experiments, one reproducible benchmark (vLLM), batch + server scenarios. This is the largest controlled per-generation LLM-inference measurement set I found.
- Verbatim (§4, Table 2 lead-in):
  > "For hardware, we select a wide range of NVIDIA GPUs, spanning five different architectures and including server-class and consumer-grade devices (Table 2), enabling the analysis of architectural design trade-offs that affect energy efficiency."
- Verbatim (Table 2 rows, abridged): "Tesla V100 SXM2 Volta … 2018 | Tesla T4 Turing … 2018 | A100 SXM4 Ampere … 2020 | GeForce RTX 3090 Ampere … 2020 | A30 PCIe Ampere … 2021 | **GeForce RTX 4090 Ada Lovelace … 2022** | L40S Ada Lovelace … 2022 | L4 Ada Lovelace … 2023 | **H100 NVL Hopper** … 2023 | **H200 NVL Hopper** … 2024"
- Per-generation result tables: Table 3 (energy/token ranking), Table 4 (server-scenario ranking), Tables 5–6 (TTFT p95 + mean power, per GPU, per model-size category).

**C. Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures**
- arXiv **2504.11750** — **VERIFIED**
- https://arxiv.org/abs/2504.11750 | submitted 2025-04-16
- Coverage: 3 platforms, PCIe A100 (Ampere) / PCIe H100 (Hopper) / GH200, same 4 LLM workloads, fine-grained kernel-trace profiling.
- Verbatim (abstract):
  > "This paper presents an in-depth analysis of LLM inference behavior on loosely-coupled (PCIe A100/H100) and closely-coupled (GH200) systems."
- Verbatim (Table IV / contributions):
  > "The three systems include two LC systems (PCIe-connected A100, H100 GPUs) and a CC GH200 system."
- Caveat: generation is **confounded** with CPU (AMD EPYC + A100 vs Intel Xeon + H100 vs Grace + GH200), so it is not a clean generation ablation. Its axis is really coupling, not generation.

### 1.2 Papers whose *finding* is that the effect is generation-dependent

**D. Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode** ← *closest thing to question (b)*
- arXiv **2605.30571** — **VERIFIED**
- https://arxiv.org/abs/2605.30571 | submitted 2026-05-28
- Coverage: **4 GPUs spanning 3 generations** — H100 SXM5 (Hopper), A100-80GB SXM4 (Ampere), L40S (Ada), L4 (Ada) — under "a controlled bf16 SDPA setup", batch-1 decode, ctx 2048–16384.
- Verbatim (abstract):
  > "We measure batch-1 decode for three 7 to 8B-class GQA transformers across four NVIDIA GPUs: H100 SXM5, A100-80GB SXM4, L40S and L4."
- Verbatim (abstract) — **the generation-dependence claim**:
  > "The achieved fraction of peak HBM bandwidth falls as peak bandwidth rises. On the headline Qwen-2.5-7B ctx=2048 cell, an L4 reaches roughly 81 percent of its analytic memory floor, while an H100 reaches only 27 percent."
- Verbatim (abstract) — **the "does not transfer" claim**:
  > "On H100 at ctx=2048, CUDA Graphs improves decode latency by 1.259x across N=10 fresh sessions … On L4, the same intervention gives only 1.028x. This isolates a launch-side overhead that becomes visible on fast GPUs but remains mostly hidden on slower, bandwidth-bound GPUs."

**E. APEX4: Efficient Pure W4A4 LLM Inference via Intra-SM Compute Rebalancing**
- arXiv **2606.08761** — **VERIFIED**
- https://arxiv.org/abs/2606.08761 | submitted 2026-06-07 (v1), online 2026-06-26
- Coverage: "controlled benchmarks across four GPUs from Ampere and Ada architectures" (RTX 3090, A100, L40S, A40). **No Hopper/Blackwell** — but this is an explicit *controlled hardware-variable* study whose stated conclusion is a generation/architecture-dependent sign flip.
- Verbatim (abstract):
  > "Through controlled benchmarks across four GPUs from Ampere and Ada architectures, we identify the Tensor Cores to CUDA Cores throughput ratio (ρ) as the primary hardware indicator: the W4A4-g128 kernel yields 2.0–2.5× speedup on RTX 3090 (ρ=16) yet degrades to 0.43–0.47× on A100 (ρ=64) in compute-bond scenarios, establishing W4A4 viability as platform-dependent rather than universally infeasible."

**F. FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling**
- arXiv **2603.05451** — **VERIFIED**
- https://arxiv.org/abs/2603.05451 | submitted 2026-03-05
- This is the clearest *hardware-generation-as-the-design-axis* paper for LLM attention kernels: the entire contribution is "FA3's Hopper design does not transfer to Blackwell".
- Verbatim (abstract):
  > "While FlashAttention-3 optimized attention for Hopper GPUs through asynchronous execution and warp specialization, it primarily targets the H100 architecture. The AI industry has rapidly transitioned to deploying Blackwell-based systems such as the B200 and GB200, which exhibit fundamentally different performance characteristics due to asymmetric hardware scaling: tensor core throughput doubles while other functional units (shared memory bandwidth, exponential units) scale more slowly or remain unchanged."

### 1.3 Benchmark / measurement papers giving per-generation inference numbers (question c)

**G. WattGPU: Predicting Inference Power and Latency on Unseen GPUs and LLMs**
- arXiv **2607.02391** — **VERIFIED**
- https://arxiv.org/abs/2607.02391 | submitted 2026-07-02
- Explicit **cross-hardware generalization** as the formal object of study (`leave-one-GPU-out`), 8 GPUs / 42 LLMs, and a head-to-head against the roofline baseline.
- Verbatim (abstract):
  > "Our approach leverages only publicly available LLM metadata and GPU specifications, eliminating the need for hardware access or profiling while enabling generalization to unseen NVIDIA server-grade GPUs and LLMs. We evaluate our models using rigorous leave-one-GPU-out and leave-one-LLM-out cross-validation on a dataset of 42 open-source LLMs (0.1B--27B parameters) and 8 GPUs…"

**H. Where Do the Joules Go? Diagnosing Inference Energy Consumption**
- arXiv **2601.22076** — **VERIFIED**
- https://arxiv.org/abs/2601.22076 | submitted 2026-01-29
- Two generations (Hopper H100 + Blackwell B200), 1,858 configs, 46 models — large controlled measurement, but only 2 generations and no Ada.
- Verbatim (abstract):
  > "we begin by presenting a large-scale measurement study of inference time and energy across the generative AI landscape with 46 models, 7 tasks, and 1,858 different configurations on NVIDIA H100 and B200 GPUs."

**I. Benchmarking and Dissecting the Nvidia Hopper GPU Architecture**
- arXiv **2402.13499** — **VERIFIED**
- https://arxiv.org/abs/2402.13499 | submitted 2024-02-21
- Microbenchmark paper, not LLM inference, but it *is* a controlled 3-generation hardware comparison of AI function units.
- Verbatim (abstract):
  > "we conduct conventional latency and throughput comparison benchmarks across the three most recent GPU architectures, namely Hopper, Ada, and Ampere."

### 1.4 Quantization ablated by generation (question d)

**J. Microarchitecture Is Destiny: Performance and Accuracy of Quantized LLMs on Consumer Hardware**
- **No arXiv ID found.** OpenReview forum id `SzQGRR65c0`; PDF at https://openreview.net/pdf?id=SzQGRR65c0 (I could **not** fetch it — 403 challenge). **ID/venue UNVERIFIED.**
- Title obtained from OpenReview's own search API (`api2.openreview.net/notes/search?...&source=forum`), which returned `id SzQGRR65c0 | title: Microarchitecture Is Destiny: Performance and Accuracy of Quantized LLMs on Consumer Hardware`.
- Coverage per the public reviews (verbatim, fetched from `api2.openreview.net/notes/search`):
  > "This paper empirically studies how quantized large language models (LLMs) perform across multiple GPU generations. The authors evaluate four NVIDIA architectures (Pascal, Turing, Ampere, and Ada Lovelace) using three model families (Qwen 2.5, DeepSeek-R1, QwQ-32B) and three datasets (GSM8K, SQuAD, CMMLU)."
  > "The experiments span four GPU generations, multiple model sizes, and diverse benchmarks, offering strong empirical coverage."
  > "once VRAM capacity suffices to load a model, inference throughput is primarily determined by Tensor Core generation and memory bandwidth"
- Note: Pascal/Turing/Ampere/Ada — **no Hopper, no Blackwell**. Also reviewed as having "limited relevance for modern LLM deployment".

---

## 2. Verdict

**Does a controlled cross-generation hardware comparison for LLM inference exist? — YES, but not one that spans Ada → Hopper → Blackwell on the inference *serving-system* axis.**

- **Closest overall: SlideSparse (2603.05232).** It is the only paper I verified that runs *one* LLM-inference system (vLLM integration, one Docker image, CUDA 12.9) across **Ada sm89 + Hopper sm90 + Blackwell sm100 + Blackwell sm120** and reports per-generation numbers, with an explicit per-architecture discussion section. It is an anonymous ICML-style submission (no venue/authorship confirmed) and its subject is sparse-GEMM kernels rather than multi-GPU serving, so it does not answer the PCIe-only 8×4090 question.
- **Closest for a generation-dependent *scientific claim*: 2605.30571.** It is the cleanest instance of "the same technique's benefit depends on the GPU generation, and a Hopper-era conclusion does not transfer" — CUDA Graphs gives 1.259× on H100 but 1.028× on L4, and the fraction-of-peak-bandwidth achieved *falls* as peak bandwidth rises (L4 81% vs H100 27%). Four GPUs, three generations (Ampere/Ada/Hopper), controlled setup. **No Blackwell.**
- **Broadest measurement: Watt Counts (2604.09048)** — 10 GPUs across Volta/Turing/Ampere/Ada/Hopper, 50 LLMs, one benchmark, per-GPU tables, and an explicit "newer generations do not always win" finding. **No Blackwell, and no RTX 5090.**
- **Most explicit hardware-axis-as-thesis: APEX4 (2606.08761)** — isolates a single architectural ratio ρ (Tensor-Core:CUDA-Core throughput) and shows a 2.0–2.5× win becoming a 0.43–0.47× loss across architectures. Ampere + Ada only.
- **Explicit generation-transfer claim: FlashAttention-4 (2603.05451)** — a full system justified by "Hopper design does not transfer to Blackwell due to asymmetric hardware scaling". Hopper + Blackwell only.

**How close is the closest thing?** Two of the three legs you care about exist, but never in one paper:
- Ada↔Hopper is covered (2605.30571, 2604.09048, 2504.11750, 2402.13499).
- Hopper↔Blackwell is covered (2603.05451, 2601.22076).
- Ada↔Hopper↔Blackwell in one controlled inference system is covered **only** by SlideSparse (2603.05232), and only for sparse/quantized GEMM kernels inside vLLM — not for multi-GPU serving, not for the PCIe-vs-NVLink axis, and not on 8×RTX 4090.

No paper I found measures the *same* LLM inference/serving configuration across **Ada sm89 AND Hopper sm90 AND Blackwell sm100/sm120** as a controlled hardware variable in a peer-reviewed venue with an artifact. I did not find the specific "8× RTX 4090 PCIe-only, 2 NUMA nodes, Ada vs Hopper vs Blackwell" ablation — nothing in my harvest touches 2-NUMA-node inference serving.

---

## 3. Could NOT verify / not found

**Not found (I did not find it):**
- No paper ablating **GPU generation** for **speculative decoding** across 3+ generations. Closest seen (title only, from the HF index, not read): Sequoia (2402.12374 — *verified* title "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding", but I did not read its evaluation section, so I cannot claim its generation coverage).
- No paper ablating **GPU generation** for **KV-cache compression / offloading** across 3+ generations.
- No **MLPerf Inference hardware-scaling analysis** verified. I did not successfully fetch any mlcommons.org surface; treat MLPerf as **unsearched**.
- No paper on **SGLang/vLLM/TensorRT-LLM default-config cross-generation ablation**.
- Nothing covering **2 NUMA nodes** + multi-GPU inference.

**Could not fetch / blocked surfaces:**
- `arxiv.org/search/?...` → HTTP 406 (hard blocked; also `arxiv.org/list/<cat>/<YYMM>` → 406). Full-text search on arXiv was unavailable.
- `export.arxiv.org/api` → HTTP 429 (as warned).
- `api.github.com` → not used (as instructed).
- `openreview.net/api2 .../notes?forum=SzQGRR65c0` and `openreview.net/pdf?id=SzQGRR65c0` → HTTP 403 challenge, repeated 5× with backoff and a cookie jar. Therefore the "Microarchitecture Is Destiny" **abstract and venue are UNVERIFIED**; only the title (from OpenReview search API) and the review text (from OpenReview search API) were actually read.
- `semanticscholar.org` web + `api.semanticscholar.org` → HTTP 429 / 202 throughout.
- `searx.be` → anti-bot captcha. `mojeek.com` → 403. `html.duckduckgo.com` → 202 challenge.
- `bing.com/search` HTML → worked twice, then began returning locale-poisoned unrelated results (Japanese/Chinese spam) for the same query string; **not trusted, not used for any claim in this report**.

**Surfaces that DID work (and how):**
- `https://arxiv.org/abs/<id>` and `https://arxiv.org/html/<id>v1` via curl — used for all verification and all quotes.
- `https://huggingface.co/api/papers/search?q=...` — primary discovery surface. 15 queries → 984 unique papers (batch 1).
- `https://api2.openreview.net/notes/search?query=...&source=forum` — discovery + title/abstract retrieval.
- `web_search` tool — used only to discover the OpenReview PDF URL; every arXiv ID it suggested was independently verified or discarded.

**Exact queries run (batch 1, via `huggingface.co/api/papers/search`):**
`LLM inference evaluated on H100 and RTX 4090 and B200` · `LLM serving benchmark Hopper Ada Blackwell comparison` · `speculative decoding evaluation multiple GPU architectures H100 4090` · `KV cache compression evaluated across GPU architectures generations` · `attention kernel benchmark across NVIDIA generations Hopper Ada Blackwell` · `hardware software co-design ablation GPU generation LLM` · `MLPerf inference benchmark hardware scaling analysis GPU generations` · `roofline model arithmetic intensity across GPU generations LLM` · `does speedup transfer across GPU architectures LLM inference` · `GPU architecture dependent performance conclusion generalization LLM` · `quantization evaluation Hopper Ada Blackwell RTX 4090 H100` · `8x RTX 4090 PCIe inference serving without NVLink` · `consumer GPU vs datacenter GPU LLM inference comparison 4090 H100` · `L40S L4 inference benchmark comparison H100` · `cross-generation GPU performance portability LLM inference`

**Exact queries run (via `web_search` tool):** `controlled comparison LLM inference across GPU generations Ada Hopper Blackwell paper` · `quantized LLM inference across multiple NVIDIA GPU generations Pascal Turing Ampere Ada paper` · `LLM inference benchmark H100 RTX 4090 B200 same system comparison arxiv` · `"Microarchitecture Is Destiny" quantized LLMs consumer hardware paper` · `"Microarchitecture Is Destiny: Performance and Accuracy of Quantized LLMs on Consumer Hardware"`

**Second batch (15 further queries) was cancelled mid-run** on the parent's wrap-up instruction; ~15 more queries covering generation-dependent optimization, performance portability, PCIe bottlenecks, TP without NVLink, disaggregated prefill/decode across generations, and microbenchmark comparison were launched but produced no completed result file. Treat those as **incomplete, not negative**.

**Absence claims are stated as "I did not find it", not "it does not exist."**
