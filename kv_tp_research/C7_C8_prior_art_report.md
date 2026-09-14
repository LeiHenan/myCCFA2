# C7 / C8 prior-art report — KV-cache-driven TP on 24 GB PCIe-only boxes

Scope: 8× RTX 4090, 24 GB, PCIe-only, no NVLink, 2 NUMA nodes. Date of work: 2026-09-14.
Every arXiv ID below was verified by fetching `https://arxiv.org/abs/<id>` and reading the title back.
Every quote is text I saw in a fetched page.

---

## (a) Work that NAMES the causal chain: KV/context → TP degree up → all-reduce cost

### A1. vLLM docs — `docs/serving/context_parallel_deployment.md`  ★ strongest framework-doc naming
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/serving/context_parallel_deployment.md (fetched 2026-09-14, HTTP 200)
Also live at: https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md
Date: current `main` (local clone pinned at commit `2d9c30a7b545e09fd118b2b9e92b5702028d7c1d`, 2026-09-13)
Verbatim:
> "For a model with `H` kv-heads, a request with `T` tokens in the context needs to store `H * T` key/value tensors in the KV cache.
> 1. If one GPU can hold them all, and the performance is good enough, then no parallelization is needed.
> 2. If one GPU cannot hold them all, or we want to hold more requests in the KV cache, we can first shard the KV cache along the `H` dimension, that's the plain tensor parallel sharding. It's as simple as adding `-tp <num_gpus>` to the command line.
> 3. Since `H` is limited (determined by the model architecture), when we continue to increase the tensor parallel size, the KV cache for each GPU will be duplicated for `tp_size / H` times."

and, on the cost side, verbatim:
> "With larger dcp size, the KV cache duplication is reduced, but the communication overhead increases."
> "In short, for decode context parallel, try to increase `-tp` size until you get satisfactory performance, and then add `-dcp` to reduce the KV cache duplication."

Bearing: this is the ONLY framework doc I found that instructs users to raise TP **because of KV capacity** (step 2 is explicitly "one GPU cannot hold them all"), and in the same breath prices the extra sharding in communication. It is NVLink-agnostic — it never mentions PCIe.
NOT in the parent's list.

### A2. vLLM docs — `docs/configuration/optimization.md` (inverse framing + the cost, one paragraph)
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/configuration/optimization.md (fetched, HTTP 200)
Verbatim (preemption warning users are told to read):
> "Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory."
Verbatim (the remedy list):
> "Increase `tensor_parallel_size`. This shards model weights across GPUs, allowing each GPU to have more memory available for KV cache. However, increasing this value may cause excessive synchronization overhead."
Bearing: the official vLLM answer to "not enough KV cache space" is *raise TP*, immediately qualified by synchronization overhead. That is the C7 coupling in two sentences, with no interconnect differentiation.
NOT in the parent's list.

### A3. vLLM blog — "Efficient Decode Context Parallelism with vLLM for Long Context Workloads"
URL: https://vllm.ai/blog/2026-08-07-decode-context-parallelism (fetched, HTTP 200)
Date: August 7, 2026. Authors: Seonghee Lee, Sungsoo Ha, Omri Almog (NVIDIA), Lucas Wilkinson (Red Hat AI).
Verbatim:
> "Agent-trace benchmarks now run from 64K all the way to 1M tokens and their KV caches are correspondingly large. Under a baseline tensor-parallel (TP) setup, this KV cache is partitioned by attention head, which puts a hard floor on how much it can shrink."
> "In both cases the duplicated KV cache eats into GPU memory, leaving very little room to serve additional requests. This caps the number of concurrent requests the system can handle, driving down throughput and pushing up cost per token."
> "On systems with high-bandwidth GPU-to-GPU interconnects, this helps preserve interactive responsiveness while serving many long-context agents at once."
Bearing: context length → KV cache → the TP default becomes capacity-inadequate; and the fix is scoped to "high-bandwidth GPU-to-GPU interconnects" — the 24 GB PCIe box is exactly the case the sentence excludes.
NOT in the parent's list.

### A4. arXiv 2405.08944 — "Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis"  ★ the PCIe term
URL: https://arxiv.org/abs/2405.08944 — VERIFIED title; full text https://arxiv.org/html/2405.08944v1 (fetched, HTTP 200)
Date: submitted 14 May 2024.
Verbatim:
> "We give a detailed analysis of how all additional computational costs, compared to 4K context, trace back to one single source: the large size of the KV cache."
> "Tensor Parallelism utilizes multiple devices for accelerating inference with negligible communication overhead. Linearly increasing the number of devices to 2, 4, and 8 introduces more HBM space, thus linearly increasing concurrency."
> "(4) context switching is PCIE bound : offloading user 1's KV cache to the CPU DDR and loading user 2's KV cache to the HBM is bounded by the PCIE bandwidth."
> "context switching overhead = (user 1's KV cache + user 2's KV cache) / PCIE bandwidth"
Bearing: formalizes (i) KV size as the single root cause of long-context deployment cost, (ii) TP as the lever that linearly buys HBM/KV room (the inverse framing), and (iii) a PCIe-bandwidth-bound term whose magnitude is the KV cache size. Its TP model assumes "negligible communication overhead" (A100 NVLink), i.e. the assumption that fails on a PCIe-only box. Closest thing to a formula for the C8 half.
NOT in the parent's list.

### A5. arXiv 2608.23962 — additional verbatim beyond the parent's summary
The parent already has this paper. Two quotes worth having because they are the explicit "compulsory / forced" naming:
> "When an LLM serving deployment becomes memory-bound, whether because the context is long, the batch is large, or both, the KV cache stops fitting in one GPU and something has to give."
> "Across degrees, the all-reduce term erodes benefit as hypothesised. Scaling efficiency falls to 31.9% at TP = 8 on W1, so high degrees never sit on the cost frontier unless feasibility forces them (Fig. 5)."
> "Above the wall, tensor parallelism is compulsory and compression cannot substitute"
> "On one side the tensor-parallel degree is swept (1, 2, 4, 8): each step divides per-GPU weight and KV memory, adds all-reduce communication, and multiplies cost by the device count."
Caveat I verified: the full text contains **0 occurrences of "PCIe", "NVLink", "interconnect" and "bandwidth"** — the interconnect is a generic all-reduce term, calibrated on A100/A40/H100.

### A6. Megatron-LM docs — `context_parallel.md` (same coupling shape, training/activation memory)
URL: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/refs/heads/main/docs/user-guide/features/context_parallel.md (fetched, HTTP 200)
Verbatim:
> "An LLM can hit an out-of-memory (OOM) error on long contexts (long sequence lengths) because activation memory grows about linearly with sequence length. Recomputing activations in backward can avoid OOM but adds significant overhead (about 30 percent with full recomputation). Increasing TP (tensor model parallelism) can also fix OOM, but it can make compute in layers such as Linear too short to hide communication latency. Scaling to more GPUs with larger TP can hit that overlap limit even when OOM is not the driver."
Bearing: NVIDIA stating, in framework docs, that long context → OOM → raise TP → communication latency becomes unhidden. Applies to activation memory in training, NOT to KV cache in serving — so it is the same causal shape but a different resource.
NOT in the parent's list.

---

## (b) 24 GB-regime MULTI-GPU SERVING studies (measured)

### B1. `mtecnic/vllm-topology-bench` — 4× RTX 3090 (24 GB), no NVLink, measured TP=4 vs 2×TP=2  ★ best 24 GB PCIe multi-GPU serving measurement
URL: https://github.com/mtecnic/vllm-topology-bench (fetched; README.md and METHODOLOGY.md via raw, HTTP 200)
Hardware/date: 4× RTX 3090 24 GB (Ampere sm_86), no NVLink; vLLM 0.21.0; model Qwen3.6-35B-A3B-AWQ; `max-model-len 8192`; accessed 2026-09-14 (repo has no explicit publication date).
Verbatim:
> "4× NVIDIA GeForce RTX 3090 (24 GB, Ampere sm_86), **no NVLink** — inter-GPU traffic crosses **PCIe**."
> "**TP ≥ 2 is mandatory** for this model on 3090s." / "Lowering `max-model-len` doesn't help — the weights are the wall, not the KV."
> "`1× TP=4` **saturates ~600 tok/s**; `2× TP=2` keeps climbing past **1,400**. Same 4 GPUs, same model, same workload." (aggregate decode +136% at concurrency 128; TTFT 1.4 s vs 3.9 s)
> "## 🔬 Why (it's the interconnect, not the KV cache)"
> "The lever is **interconnect**: tensor parallelism does an **all-reduce every single layer**, and with **no NVLink that crosses PCIe**. `TP=4` pays that tax across 4 GPUs → saturates."
> "**No-NVLink PCIe** interconnect penalizes tensor parallelism more than an NVLink system would; results are specific to this class of hardware."
> "**Benchmark `max-model-len` (8192) < production (131072)** — chosen so all modes are comparable (and so TP=1 could even be attempted). **Longer contexts would further favor whichever topology has more KV headroom.**"
Bearing: this is the sharpest artifact for the whole question, and it cuts both ways. At 8K context on 24 GB PCIe cards the forcing resource is **weights, not KV** — and higher TP is measured to be *much slower*. The authors explicitly flag that at production context lengths KV headroom would favour the higher-TP layout, i.e. the regime where the C7 coupling bites is **acknowledged but not measured** here.
NOT in the parent's list.

### B2. `noonghunna/qwen36-dual-3090` — `docs/INTERNALS.md`, 2× RTX 3090 24 GB, PCIe-only
URL: https://github.com/noonghunna/qwen36-dual-3090/blob/master/docs/INTERNALS.md (blob HTTP 200; raw file fetched)
Context: "Run Qwen3.6-27B at full 262K context with 4-stream concurrency on 2× consumer 24 GB RTX 3090"; README: "**NVLink bridge NOT required** … PCIe-only is the tested path."
Verbatim:
> "Reality on PCIe-only consumer Ampere: per-stream TPS gain from TP=2 is small (~5%)."
> "All-reduce on **PCIe Gen 4** (~32 GB/s practical) is ~3-5× slower than NVLink (~600 GB/s on H100, ~200 GB/s on 3090 with bridge)"
> "All-reduce overhead approximately cancels the memory-bandwidth halving"
Bearing: quantifies the C8 claim on 24 GB consumer cards at 262K context — the per-stream benefit of raising TP on PCIe is ~5%, because the all-reduce cancels the memory-bandwidth saving.
NOT in the parent's list.

### B3. Kurochka, Basharymau, Youzhanka — "Parallelism strategies as a key factor for deploying Large Language Models on consumer GPUs"  ★ peer-reviewed, PCIe consumer multi-GPU
URL: https://sapi.bntu.by/jour/article/view/799/0 (fetched, HTTP 200); DOI https://doi.org/10.21122/2309-4923-2026-1-54-59
Venue/date: *System analysis and applied information science* (Системный анализ и прикладная информатика), 2026;(1):54–59. Authors: К. С. Курочка, Ю. С. Башаримов, Ю. Д. Ёвженко (Gomel State Technical University).
Verbatim (Russian abstract, as printed):
> "Методы исследования включали проведение серии вычислительных экспериментов для сравнения монолитной архитектуры (NVIDIA RTX A6000) и распределенной системы (2x NVIDIA RTX 3090) с использованием фреймворка vLLM. Анализировалось влияние тензорного (Tensor Parallelism) и конвейерного (Pipeline Parallelism) параллелизма на ключевые метрики: пропускную способность, задержку (TTFT, TPOT) и стабильность энергопотребления при запуске модели DeepSeek-R1-DistillLlama-14B. Результаты однозначно указывают на непригодность тензорного параллелизма для систем без NVLink из-за критических задержек синхронизации. Доказано, что конвейерный параллелизм является единственной жизнеспособной стратегией для PCIe-кластеров…"
Translation: experiments compared a monolithic RTX A6000 against a distributed 2× RTX 3090 system in vLLM, measuring TP vs PP on throughput, TTFT, TPOT and power stability for DeepSeek-R1-Distill-Llama-14B; the results "unambiguously indicate the unsuitability of tensor parallelism for systems without NVLink due to critical synchronization delays", and PP is "the only viable strategy for PCIe clusters".
Bearing: a peer-reviewed, 24 GB-class, multi-GPU, PCIe-only serving study concluding TP is unusable without NVLink — the C8 claim in published form. It is TP-vs-PP, **not** KV-driven TP, and it is 2 GPUs, not 8.
NOT in the parent's list. (I could not open the PDF: no PDF text extractor available in this environment; the abstract and citation block above are from the landing page I fetched.)

### B4. `local-inference-lab/rtx6kpro` — "PCIe Oneshot AllReduce for Inference" (NUMA / 4-vs-8 GPU, not 24 GB)
URL: https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/61842c4f5c3489dc4edaba7639a2f0656b587f9e/optimization/pcie-oneshot-allreduce.md (fetched, HTTP 200). NOTE: `blob/main/...` returns 404; only the SHA-pinned raw URL worked.
Hardware: ASUS ESC8000A-E13P, 4× RTX PRO 6000 Blackwell (**96 GB**, so not the 24 GB regime).
Verbatim:
> "Provides **~7% faster decode throughput on 4 GPU (same NUMA)** but **does not help on 8 GPU cross-socket** where Infinity Fabric latency makes the system-scope barrier too expensive."
> "During tensor-parallel decode, each layer performs AllReduce operations on small messages (typically 16–256 KB for attention/MoE layers)."
> 8 GPU cross-socket, GLM-5 TP=8 MTP: c=1 ctx=0 "**-4.3%**", c=1 ctx=16k "**-5.2%**", c=1 ctx=32k "**-15.7%**"
Bearing: the only measurement I found that ties TP-collective cost to **NUMA topology and to rising context length** (penalty grows −5.2% → −15.7% from 16k to 32k as payloads cross the NCCL crossover). Directly relevant to the 2-NUMA-node part of the target box, though the GPUs are 96 GB not 24 GB.
NOT in the parent's list.

---

## (c) Framework documentation: does anything tell users to pick TP from a KV/context budget?

Fetched and checked:
| Doc | URL (fetched) | Verdict |
|---|---|---|
| vLLM `context_parallel_deployment.md` | raw.githubusercontent.com/vllm-project/vllm/main/docs/serving/context_parallel_deployment.md | **YES** — step 2 shards KV when "one GPU cannot hold them all"; per-case studies (DeepSeek-R1 `-tp 8` → 8× KV duplication; Kimi-K2 `-tp 16` → 16× duplication; Qwen3-235B `-tp 8` → 2×) |
| vLLM `optimization.md` | …/docs/configuration/optimization.md | **YES, indirect** — preemption warning names `tensor_parallel_size` as the KV-capacity remedy |
| vLLM `parallelism_scaling.md` | …/docs/serving/parallelism_scaling.md | Interconnect-driven, not context-driven: "If the model fits within a single node but the GPU count doesn't evenly divide the model size … if the GPUs on the node do not have NVLINK interconnect (e.g. L40S), leverage pipeline parallelism instead of tensor parallelism for higher throughput and lower communication overhead." |
| TensorRT-LLM "Deciding Model Sharding Strategy" | https://nvidia.github.io/TensorRT-LLM/0.20.0rc0/performance/performance-tuning-guide/deciding-model-sharding-strategy.html | **NO context/KV criterion.** "Consequently, a general rule of thumb is that if your GPUs have fast connections between them like NVLink then tensor parallel is likely a good choice. However if the communication will go over slow connections (across nodes for example) pipeline parallel is likely better." |
| SGLang server arguments / docs | https://docs.sglang.ai/advanced_features/server_arguments.html | **NO rule found** tying TP to context/KV; TP guidance is "To enable multi-GPU tensor parallelism, add `--tp 2`" |

So: exactly one framework doc (vLLM's DCP page) tells users to raise TP because the KV cache does not fit; none of the docs surveyed conditions the TP-vs-PP choice on context length.

---

## (d) Verdict

**C7 — is the coupling named/formalized?** *Partly, and only in pieces; never as one named chain; never on PCIe.*
- The **inverse half** (TP raises per-GPU KV headroom / TP is the standard way to fit a long-context cache) is stated plainly and repeatedly: vLLM `optimization.md` and `context_parallel_deployment.md`, arXiv 2405.08944 ("linearly increasing the number of devices to 2, 4, and 8 introduces more HBM space"), arXiv 2608.23962 ("the standard way to fit a model or a long-context cache that exceeds one GPU").
- The **forward half** (long context → KV does not fit → you must raise TP) is stated as a *constraint* in vLLM's DCP doc and in 2608.23962 ("tensor parallelism is compulsory"; "unless feasibility forces them"), and 2608.23962 prices it ("each step divides per-GPU weight and KV memory, adds all-reduce communication, and multiplies cost by the device count").
- The **closing link** (that all-reduce traffic then runs over a PCIe-only link and is therefore disproportionately expensive *exactly when you are forced to raise TP*) is **not** made by any paper. In 2608.23962 the full text has zero occurrences of "PCIe"/"NVLink"/"interconnect". 2405.08944 has a PCIe-bound term but assumes TP communication is "negligible". The vLLM/SGLang/NVIDIA blogs that discuss KV-driven sharding all scope their results to NVLink domains (Helix → GB200; SGLang PP blog → H20 cross-node).
- Closest single artifact to the whole chain: **arXiv 2608.23962** for the TP-vs-KV-capacity cost axis, and **vLLM `context_parallel_deployment.md`** for the operational rule "shard KV with TP, then pay more communication".
- No source I found uses the terms "KV cache pressure" (as a defined term), "KV-budget-driven sharding", "memory-capacity-forced parallelism", or "context-driven TP".

**C8 — 24 GB regime + PCIe all-reduce cost rising exactly when TP is forced.** *The regime is measured, the coincidence is documented, but the two are never joined.*
- Measured 24 GB multi-GPU PCIe serving exists: `mtecnic/vllm-topology-bench` (4× 3090, TP=4 vs 2×TP=2, +136%), `noonghunna/qwen36-dual-3090` (2× 3090, TP=2 ≈ +5% per-stream), and the peer-reviewed Kurochka et al. 2026 (2× 3090, "TP unsuitable without NVLink"). All three conclude **use less TP on PCIe**.
- But the one that comes closest to the coupling states the opposite binding resource at its tested context: "the weights are the wall, not the KV" (`vllm-topology-bench`, 8K context), with the long-context case explicitly left as an untested caveat ("Longer contexts would further favor whichever topology has more KV headroom").
- The only source measuring the collective cost as a function of **NUMA + context length** is the rtx6kpro PCIe-oneshot writeup (−5.2% at 16k → −15.7% at 32k on 8 GPU cross-socket) — but on 96 GB Blackwell cards.
- Nothing found quantifies TP scaling efficiency vs TP degree on **8× 24 GB PCIe** at long context. That is an open measurement.

---

## (e) Could NOT verify / not found (exact queries + surfaces)

Surfaces used: `web_search` tool; Hugging Face paper search (curl, `huggingface.co/papers?q=`); Hacker News Algolia API (curl); direct curl of arxiv.org/abs + arxiv.org/html; direct curl of vendor blogs and GitHub blob/raw pages; local clones of vLLM / SGLang / TensorRT-LLM at `/Users/leihenan/Desktop/myProject/research/`.
Surfaces blocked/failed: `arxiv.org/search` → HTTP 406; Bing HTML → returned unrelated localized results, discarded; DuckDuckGo HTML/lite → bot challenge; Startpage → Anubis challenge; Mojeek → 403; searx.be → captcha; `api.semanticscholar.org` → 429; `export.arxiv.org/api` → not used per instructions; **reddit.com / old.reddit.com** → network-blocked, not opened; **GitHub repo-scoped issue search** → the `/issues?q=` page no longer embeds results (no `total_count`, no result JSON) and `api.github.com` is rate-limited to zero, so repo-scoped search was abandoned; individual issue pages (e.g. vLLM #34018) fetch fine via the embedded `"body":"` JSON.

Exact queries that returned nothing on point:
- `"KV-budget-driven sharding"` (web_search; HF papers) → no source uses the term.
- `"memory-capacity-forced parallelism"` (web_search; HF papers) → nothing; results were generic memory-optimization papers.
- `"context-driven TP"` / `"context-driven tensor parallelism"` (web_search; HF papers) → nothing on point.
- `"KV cache pressure"` (web_search; HN Algolia `tags=story`; HF papers) → no paper or vendor doc defines it as a term; the phrase appears only loosely, closest formal use being vLLM RFC #34018 "Reduces memory pressure for long-context workloads (256K+ tokens)".
- `"KV cache" "tensor parallel" forced higher TP degree long context PCIe all-reduce blog` (web_search) → no matching blog.
- `all-reduce latency measurement tensor parallel decode PCIe Gen4 x16 microbenchmark LLM inference overhead` (web_search) → nothing but the rtx6kpro item already listed.
- `8x RTX 4090 tensor parallel benchmark tokens/s PCIe NVLink comparison LLM serving` (web_search) → no 8× 4090 TP-scaling measurement found.
- HF paper searches `KV budget driven sharding`, `memory capacity forced parallelism`, `context driven tensor parallelism`, `24GB VRAM tensor parallelism long context serving`, `pipeline parallel vs tensor parallel PCIe consumer GPU LLM inference` → no paper whose abstract ties context/KV budget to the required TP degree on PCIe.

UNVERIFIED items (flagged, do not cite as confirmed):
- **Mnemosyne** ("Mnemosyne: Parallelization Strategies for Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations"): Hugging Face lists it under **arXiv 2409.17264**, but fetching `https://arxiv.org/abs/2409.17264` returns "**No Request Left Behind: Tackling Heterogeneity in Long-Context LLM Inference with Medha**", and the string "Mnemosyne" does not appear on that page. **I could not confirm any arXiv ID for Mnemosyne — treat its ID as UNVERIFIED.**
- Two more HF-search title/ID pairings were demonstrably wrong when checked: HF listed 2511.09557 as "LLM Inference Beyond a Single Node…" but the abs page reads "**Understanding and Improving Communication Performance in Multi-node LLM Inference**"; HF listed 2504.08791 as "PRIMA.CPP: Speeding Up 70B-Scale…" but the abs page reads "**Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters**". Practical consequence: **Hugging Face paper-search titles are not reliably paired with their IDs — verify every one at arxiv.org/abs.**
- `localai.computer/multi-gpu-methodology` returned HTTP 429 (Vercel checkpoint); not opened.
- https://github.com/local-inference-lab/rtx6kpro/blob/main/optimization/pcie-oneshot-allreduce.md → 404 (only the SHA-pinned raw URL resolved).

Other verified-but-secondary IDs fetched during the sweep (titles confirmed, not quoted in this report): 2404.09526 LoongServe (elastic sequence parallelism, "Restricted by static parallelism strategies…"), 2506.20187 LeoAM (long-context KV on a single commodity GPU), 2312.12456 PowerInfer, 2406.02532 SpecExec, 2410.00531 TPI-LLM, 2509.16495 Shift Parallelism, 2405.05329 KV-Runahead, 2511.09557 (title above), 2211.05102 Efficiently Scaling Transformer Inference.
