# HW_RULED_OUT Evidence Report — KV Cache Offloading & Hierarchical/Tiered Storage

Target researcher profile: exactly ONE GPU (RTX PRO 6000 Blackwell 96GB, sm120, driver 580),
208 CPU cores, ample local disk, single node, no multi-node, no clusters.

Ruling-out criterion applied: requires >1 GPU, >96GB VRAM, NVLink, InfiniBand/RDMA across
nodes, multi-node/multi-host cluster, or other hardware the target researcher does not have
(e.g. computational storage drives / FPGA).

All URLs below were actually retrieved during this session. All arXiv IDs were verified by
fetching `https://arxiv.org/abs/<id>` and confirming the `<title>` tag. Every verbatim quote is
copied from text extracted from the retrieved page.

Note on arXiv HTML math rendering: in `arxiv.org/html/*` pages, LaTeX `\times` renders as a
`<math alttext="\times">` element and appears as `[MATH]` in naive text extraction. Verified
against raw HTML for HCache; `[MATH]` = `×`. Quotes below show `×` and this is flagged per item.

---

# PART A — HW_RULED_OUT ITEMS (beyond the parent's already-covered list)

Parent already covered: Mooncake 2407.00079, SGLang HiCache blog, LMCache 2510.09665,
HyperOffload 2602.00748, vLLM PR #38460. Those are NOT repeated here.

## A1. InstInfer / InstAttention — In-Storage Attention Offloading

- **Technique:** InstInfer (published version titled "InstAttention: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference")
- **arXiv:** 2409.04992 — verified title: "InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference"
- **Hardware requirement (exact):** Computational Storage Drives (CSDs) built on a Daisyplus OpenSSD board with a Xilinx ZU17EG UltraScale+ MPSoC (mid-range FPGA + four-core ARM + 2GB DRAM), connected via PCIe Gen3x4 as an NVMe drive; in-storage attention engine and NFC filters implemented on the FPGA at 285MHz; engine also ported to a Xilinx Zynq7045 FPGA. The GPU side is a *single* NVIDIA A6000 (48GB) — the ruling-out factor is the CSD/FPGA hardware, NOT GPU count.
- **URLs retrieved:**
  - https://arxiv.org/abs/2409.04992
  - https://arxiv.org/html/2409.04992v1
- **Verbatim evidence (READ BODY, from §V-A System Deployment):**
  > "InstCSD is built on a Daisyplus OpenSSD, the latest version of a representative CSD device in the OpenSSD project [ 29 , 57 ] . It employs a Xilinx ZU17EG UltraScale+ MPSoC as its processor, which contains a mid-range FPGA chip with a four-core ARM processor and 2GB DRAM. The device is connected to the host through PCIe Gen3x4 lanes as an NVMe drive."
- **Supporting verbatim (same section):**
  > "The hardware SparF Attention engine and NFC filters are implemented on the FPGA part, clocked at a frequency of 285MHz, while the FTL runs as software on the ARM processor."
  > "While the Daisyplus OpenSSD platform provides an effective environment for deploying a real CSD, it presents several challenges that hinder its practicality for widespread adoption. Notably, the platform features expensive FPGA chips, costing thousands of dollars [ 57 ] , and is equipped with a limited storage capacity of only 64GB and merely 4 flash channels. In addition, OpenSSD only supports legacy motherboards like Z97, which lags far behind the current hardware environment [ 29 ] ."
  > "To reflect the real computing capability of CSD, we migrated the InstCSD implementation to the Xilinx Zynq7045 [ 64 ] , a more economically viable FPGA with an SoC that is prevalently utilized in edge computing scenarios."
- **GPU-side verbatim:** "Experimental results show that for a 13B model with an NVIDIA A6000 GPU [ 45 ] , the throughput for long-sequence inference is improved by up to 11.1 [MATH] , compared to FlexGen." (`[MATH]` = `×`)
- **Evidence level:** READ BODY

## A2. HCache — "Fast State Restoration in LLM Serving with HCache"

- **Technique:** HCache (recomputes KV cache from saved hidden states; tiered restoration using host DRAM/SSD)
- **arXiv:** 2410.05004 — verified title: "Fast State Restoration in LLM Serving with HCache"
- **Hardware requirement (exact):** 4× A100-40G SXM4 connected via NVLink, 2× AMD EPYC 7642 CPUs, 256G DDR4, 4× Samsung PM9A3 4TB enterprise SSDs. Total HBM = 160GB across 4 GPUs.
- **URLs retrieved:**
  - https://arxiv.org/abs/2410.05004
  - https://arxiv.org/html/2410.05004v1
- **Verbatim evidence (READ BODY, §6 Testbed):**
  > "Unless otherwise stated, all our experiments are conducted on a server equipped with 4×A100-40G SXM4 connected via NVLink. The host has 2× AMD EPYC 7642 CPUs, 256G DDR4 memory, and 4× Samsung PM9A3 4TB enterprise SSDs."
- **Supporting verbatim (model/GPU mapping — partial single-GPU path exists):**
  > "For Llama2-7B/13B, we use a single A100 GPU to serve them. For OPT-30B, we run it on 4 A100 GPUs with tensor parallelism unless otherwise stated."
- **Supporting verbatim (NVLink dependence for the 4-GPU path):**
  > "Since the GPU is connected via the fast NVlink interconnect, the all-gather operation will only incur a small overhead compared with the transmission part."
- **Evidence level:** READ BODY
- **Caveat:** The default testbed is 4×A100+NVLink, so the headline evaluation is ruled out. The 7B/13B single-A100 configuration is a subset.

## A3. BlockLLM — Multi-tenant Finer-grained Serving for LLMs

- **Technique:** BlockLLM (block-wise model/KV distribution across servers, NCCL-based cross-server KV transfer)
- **arXiv:** 2404.18322 — verified title: "BlockLLM: Multi-tenant Finer-grained Serving for Large Language Models"
- **Hardware requirement (exact):** 4 servers, 12× A100 80GB total (2 servers × 2 A100-80GB + 2 servers × 4 A100-80GB), servers interconnected with 100Gbps network, NCCL as communication backend. Multi-node cluster.
- **URLs retrieved:**
  - https://arxiv.org/abs/2404.18322
  - https://arxiv.org/html/2404.18322v1
- **Verbatim evidence (READ BODY, §7.1 Setup):**
  > "Cluster. Our cluster has four servers. Two are equipped with two A100 GPUs with 80GB memory, and the other two have four A100 GPUs with 80GB memory. The servers are interconnected with 100Gbps network."
- **Supporting verbatim:**
  > "We have implemented a prototype of BlockLLM. It is compatible with HuggingFace models. We use NCCL as the communication backend to transfer data among servers."
  > "Such an arrangement leverages the full potential of high-capacity intra-server connections, such as NVLink interconnects, for the transfer of requests and KV cache, rather than resorting to the constrained inter-server links."
  > "We implement and evaluate BlockLLM, a multi-tenant serving system. We evaluate BlockLLM in a 12-A100 cluster."
- **Evidence level:** READ BODY

## A4. AttentionStore / CachedAttention

- **Technique:** AttentionStore (hierarchical KV caching across HBM / host DRAM / SSD for multi-turn conversations). "AttentionStore" is the system name; the paper title is "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention".
- **arXiv:** 2403.19708 — verified title: "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention"
- **Hardware requirement (exact):** 4× NVIDIA A100 80GB (320GB total HBM), 128GB DRAM, 10TB SSDs, GPUs connected to host via PCIe Gen 4, NCCL for cross-GPU synchronization.
- **URLs retrieved:**
  - https://arxiv.org/abs/2403.19708
  - https://arxiv.org/html/2403.19708v1
- **Verbatim evidence (READ BODY, §4.1 Experimental Setup → Testbeds):**
  > "Testbeds. All our experiments are performed on 4 NVIDIA A100 GPUs, each with 80GB HBM. The system is equipped with 128GB DRAM and 10TB SSDs. GPUs are connected to the host via PCIe Gen 4."
- **Supporting verbatim:**
  > "NCCL library [ 31 ] is applied for synchronization of the parallel GPU workers."
  > "For example, we evaluate the inference time of the LLaMA-65B model using 4 NVIDIA A100 GPUs and observe that prefilling 2K tokens of a prompt consumes about 360 ms."
- **Note:** One later long-context experiment uses a single GPU: "we deploy the Mistral-7B model [ 20 ] with a maximum 32K context window on one A100 GPU with 80GB HBM, employing a GQA factor of 8". The primary/majority testbed is 4×A100.
- **Evidence level:** READ BODY

## A5. CXL-SpecKV — Disaggregated FPGA Speculative KV-Cache

- **Technique:** CXL-SpecKV (disaggregated FPGA-based speculative KV-Cache for datacenter LLM serving)
- **arXiv:** 2512.11920 — verified title: "CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving"
- **Hardware requirement (exact):** NOT RETRIEVED. Not read beyond the title.
- **URL retrieved:** https://arxiv.org/abs/2512.11920
- **Verbatim evidence:** NONE — body not read. Title alone indicates FPGA + CXL disaggregated memory, neither of which the target researcher has.
- **Evidence level:** TITLE ONLY
- **Do not treat the hardware requirement as verified for this item.**

---

# PART B — SYSTEMS THAT ARE **NOT** RULED OUT (run on a single GPU)

These are explicitly single-GPU. This distinction matters and two of them contradict the
prior assumption in the task brief.

## B1. InfiniGen — SINGLE GPU (CONFIRMED, contradicts prior assumption)

- **arXiv:** 2406.19707 — verified title: "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management"
- **Hardware:** ONE NVIDIA RTX A6000 (48GB), Intel Xeon Gold 6136, 96GB DDR4-2666, PCIe 3.0 × 16.
- **URLs retrieved:**
  - https://arxiv.org/abs/2406.19707
  - https://arxiv.org/html/2406.19707v1
- **Verbatim evidence (READ BODY, §5.1 Experimental Setup → Model and System Configuration):**
  > "We run the experiments on a system equipped with an NVIDIA RTX A6000 GPU [ 44 ] with 48GB of memory and an Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory. PCIe 3.0 × 16 interconnects the CPU and GPU."
- **Verdict:** NOT ruled out. Fits within 1 GPU / 96GB VRAM. Baselines used are CUDA UVM and FlexGen (both CPU-offload), confirming the offloading regime is CPU-GPU within one node.

## B2. CacheGen — SINGLE GPU (CONFIRMED, contradicts prior assumption)

- **arXiv:** 2310.07240 — verified title: "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving"
- **Hardware:** ONE NVIDIA A40 GPU server, 384GB memory, two Intel Xeon Gold 6130 CPUs.
- **URLs retrieved:**
  - https://arxiv.org/abs/2310.07240
  - https://arxiv.org/html/2310.07240v1
- **Verbatim evidence (READ BODY, §5.1 → Hardware settings):**
  > "Hardware settings: We use an NVIDIA A40 GPU server to benchmark our results. The server is equipped with 384GB of memory and two Intel(R) Xeon(R) Gold 6130 CPUs with Hyper-threading and Turbo Boost enabled by default."
- **Verdict:** NOT ruled out. Note this is an older/smaller Ampere GPU; the target researcher's single RTX PRO 6000 Blackwell 96GB exceeds it.

## B3. Pensieve — SINGLE GPU (CONFIRMED)

- **arXiv:** 2312.05516 — verified title: "Stateful Large Language Model Serving with Pensieve"
- **Hardware:** ONE NVIDIA A100 PCIe GPU 80GB, 24 non-multithreaded AMD EPYC Milan cores, 220GB system memory.
- **URLs retrieved:**
  - https://arxiv.org/abs/2312.05516
  - https://arxiv.org/html/2312.05516v1
- **Verbatim evidence (READ BODY, §6.1 Experimental Setup → System Environment):**
  > "We evaluate Pensieve on a Microsoft Azure NC24ads_A100_v4 instance, which is equipped with 24 non-multithreaded AMD EPYC Milan processor cores, 220 GB system memory, and one NVIDIA A100 PCIe GPU with 80GB memory. We configure each system to allocate 40GB GPU memory to KV cache for fair comparison. Pensieve and all baselines use CUDA version 11.8 and PyTorch 2.0."
- **Verdict:** NOT ruled out — single GPU, single node, CPU/GPU tiered KV cache.
- **DISAMBIGUATION:** The paper "Pensieve: A Tiered, Fully-Automated, and Cost-Effective Virtual Cluster..." is a DIFFERENT, unrelated Pensieve (virtual-cluster/networking work), not a KV-cache paper. The KV-cache Pensieve is 2312.05516, verified above.

## B4. ShadowKV — SINGLE GPU

- **arXiv:** 2410.21465 — verified title: "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference"
- **Hardware:** A single A100 GPU; value cache offloaded to CPU memory.
- **URLs retrieved:**
  - https://arxiv.org/abs/2410.21465
  - https://arxiv.org/html/2410.21465v1
- **Verbatim evidence (READ BODY, abstract):**
  > "we demonstrate that it can support up to 6 [MATH] larger batch sizes and boost throughput by up to 3.04 [MATH] on an A100 GPU without sacrificing accuracy, even surpassing the performance achievable with infinite batch size under the assumption of infinite GPU memory." (`[MATH]` = `×`; singular "an A100 GPU")
- **Verbatim evidence (READ BODY, body):**
  > "During KV selection, ShadowKV loads approximately [MATH] bytes using the GPU memory bandwidth [MATH] . For value cache fetching, it loads [MATH] bytes using the PCIe bandwidth [MATH ] [ 42 ] ."
- **Evidence limitation:** The paper does not contain a standalone explicit hardware-setup sentence naming GPU count/CPU/RAM. The single-GPU claim rests on the abstract's "an A100 GPU" and the CPU-offload design. I did NOT read the GitHub repo (https://github.com/bytedance/ShadowKV) to confirm minimum hardware.
- **Verdict:** NOT ruled out on available evidence.

## B5. IMPRESS — SINGLE GPU (FAST'25, not on arXiv)

- **Technique:** IMPRESS — "An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference". Three tiers: GPU memory, CPU memory, disk.
- **Venue:** USENIX FAST '25. NOT an arXiv paper — no arXiv ID exists for it.
- **Hardware:** ONE NVIDIA A100 GPU 80GB HBM, 2× AMD EPYC 7763 (64 cores), 128GB DRAM, ONE 2TB Intel SSD (~5GB/s read), PCIe 4.0 ×16.
- **URLs retrieved:**
  - https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf
  - https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress
- **Verbatim evidence (READ BODY, §6.1 Experimental Setup → Models and system configuration).** Extracted from the USENIX PDF. The `×` multiplication glyphs are typeset in a math font and are dropped by PDF text extraction; they are restored below and marked. The PDF is two-column justified and hyphenates across line breaks; hyphenation is rejoined:
  > "We conduct tests using three open-source OPT models of different scales (i.e., OPT-6.7B, OPT-13B, and OPT-30B). Our experiments are performed on a server with 2[×] AMD EPYC 7763 CPUs (64 cores), 128 GB DRAM, one NVIDIA A100 GPU with 80GB HBM, and one 2TB Intel SSD whose measured read throughput is around 5GB/s. The GPU and CPU are connected via PCIe 4.0[×]16."
  Raw extractor output for the same span (spaces/`×` lost): `"on a server with 2AMD EPYC 7763 CPUs (64cores), 128 GB DRAM, one NVIDIA A100 GPU with 80GBHBM, and one 2TB Intel SSD whose measured read through-put is around 5GB/s. The GPU and CPU are connected viaPCIe 4.0Datasets and metrics."`
- **Supporting verbatim (cache allocation, confirms single-GPU 10GB GPU cache):**
  > "To prevent runtime out-of-memory errors and ensure only a portion of the prefix KVs are cached (the other KVs reside on SSD), we allocate 10GB of GPU cache and 32GB of CPU cache for prefix KVs, leaving the remaining GPU and CPU memory for storing model weights, the KV cache used during the decoding phase, and the input data."
- **Verdict:** NOT ruled out. This is a GPU/CPU/SSD three-tier prefix-KV storage system evaluated on exactly one GPU plus one local SSD — the target researcher's exact shape (one GPU, ample local disk, no cluster).
- **Caveat:** The paper's own baselines (AttentionStore, AS+H2O) are not open-sourced; the authors implement IMPRESS on top of FlexGen. IMPRESS itself states: "As none of the existing state-of-the-art offloading inference systems (e.g., IMPRESS, AttentionStore) are open-sourced, we build ContiguousKV upon a well-known, publicly available LLM offloading inference framework, FlexGen [ 32 ] ." (this sentence is from the ContiguousKV paper, see B6).

## B6. ContiguousKV — single-GPU-scale successor to IMPRESS (context)

- **arXiv:** 2601.13631 — verified title: "ContiguousKV: Accelerating LLM Prefill with Granularity-Aligned KV Cache Management"
- **URL retrieved:** https://arxiv.org/html/2601.13631v1
- **Relevance:** positions itself against IMPRESS and AttentionStore; reports IMPRESS read amplification. Built on FlexGen.
- **Verbatim (READ BODY):**
  > "As none of the existing state-of-the-art offloading inference systems (e.g., IMPRESS, AttentionStore) are open-sourced, we build ContiguousKV upon a well-known, publicly available LLM offloading inference framework, FlexGen [ 32 ] ."
  > "IMPRESS [ 6 ] : The current state-of-the-art offloading-based inference system, which selectively loads partial keys for importance identification and employs a score-based cache policy."
- **Note:** I did NOT retrieve a hardware/testbed sentence for ContiguousKV. Do not treat its hardware as verified.

## B7. Klotski — OFF-TOPIC (not a KV-cache offloading paper)

- **arXiv:** 2502.06888 — verified title: "Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline"
- **URL retrieved:** https://arxiv.org/abs/2502.06888
- **Finding:** Klotski is a Mixture-of-Experts inference pipelining paper, NOT a KV-cache offloading / tiered-storage paper. It does not belong in this gap map. Body not read.
- **Evidence level:** TITLE ONLY

## B8. KVLink — borderline (multi-GPU fine-tuning, no verified inference hardware)

- **arXiv:** 2502.16002 — verified title: "KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse"
- **URLs retrieved:**
  - https://arxiv.org/abs/2502.16002
  - https://arxiv.org/html/2502.16002v1
- **Only hardware statement found (READ BODY, §3.1 Experiment Setup → Implementation):**
  > "We adopt Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct as the backbone models, fine-tuning them for 6,000 steps using a global batch size of 64 across 8 A100 GPUs."
- **Finding:** The method requires a fine-tuning stage on 8× A100 GPUs. No separate standalone evaluation-hardware sentence was found in the retrieved HTML. KVLink is primarily a KV *reuse* / link-token method rather than an offloading/tiered-storage method.
- **Evidence level:** READ BODY (for the 8×A100 fine-tuning sentence only)
- **Verdict:** Borderline — the training stage is multi-GPU; inference hardware unverified.

---

# PART C — ID CORRECTIONS AND NON-FINDINGS

## C1. arXiv ID correction: 2407.09486

- **2407.09486 is NOT CacheGen and NOT LMCache.**
- Verified title at https://arxiv.org/abs/2407.09486: "ENOVA: Autoscaling towards Cost-effective and Stable Serverless LLM Serving".
- **Correct CacheGen ID: arXiv 2310.07240** (verified title: "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving").
- **Correct LMCache ID: arXiv 2510.09665** (verified title: "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference").

## C2. Items NOT found / NOT verified (stated plainly)

- **TieredKV:** I did NOT find or verify any arXiv ID for a paper named "TieredKV". No evidence gathered. Do not cite.
- **DeepSpeed-Inference / ZeRO-Inference offloading:** I did NOT retrieve the paper or any hardware setup section. No evidence gathered. Do not cite.
- **"Pensieve: A Tiered, Fully-Automated, and Cost-Effective Virtual Cluster...":** not retrieved. This is a different Pensieve unrelated to KV cache; the KV-cache Pensieve is 2312.05516 (verified, see B3).
- **NVMe/SSD KV cache tiers 2025-2026 / GPUDirect Storage for KV cache / RDMA KV stores:** I did NOT complete verification of any specific additional paper in this class. The only two SSD-tier systems I fully verified are IMPRESS (B5) and HCache (A2). CXL-SpecKV (A5) is TITLE ONLY.
- **SpecKV:** no arXiv ID verified. The only related item seen was "CXL-SpecKV" (2512.11920, TITLE ONLY, see A5).
- **"Infinigen"** (alternate spelling in the brief) = InfiniGen, 2406.19707, verified, see B1.

## C3. Tooling limitations encountered (affects coverage)

- `https://export.arxiv.org/api/query` returned **HTTP 429** persistently (6 retries with 15s backoff, all 429). The official arXiv API was effectively unavailable.
- `https://arxiv.org/search/?...` intermittently returned **HTTP 429** as well; successful searches were paced.
- `https://api.semanticscholar.org/graph/v1/paper/search` returned **HTTP 429**.
- `https://dblp.org/search/publ/api` returned non-JSON / timed out.
- ID discovery therefore relied partly on the `web_search` tool for *leads only*; every ID used in this report was independently confirmed by fetching `https://arxiv.org/abs/<id>` and matching the `<title>`.
- No `pdftotext`/`pypdf`/`mutool` on this host; PDF text was extracted with a purpose-written pure-Python FlateDecode+TJ extractor. Expect lost spacing around math glyphs and ligatures (e.g. `ﬁ`) in PDF-derived quotes; this is flagged at B5.

---

# APPENDIX — Every URL retrieved in this session

arXiv abstract pages (all HTTP 200, all titles verified):
1. https://arxiv.org/abs/2407.00079 — Mooncake (verified)
2. https://arxiv.org/abs/2406.19707 — InfiniGen (verified)
3. https://arxiv.org/abs/2407.09486 — ENOVA (verified; MISMATCH vs. brief)
4. https://arxiv.org/abs/2303.06865 — FlexGen (verified)
5. https://arxiv.org/abs/2306.14048 — H2O (verified)
6. https://arxiv.org/abs/2403.19708 — CachedAttention / AttentionStore (verified)
7. https://arxiv.org/abs/2310.07240 — CacheGen (verified)
8. https://arxiv.org/abs/2312.05516 — Pensieve (verified)
9. https://arxiv.org/abs/2510.09665 — LMCache (verified)
10. https://arxiv.org/abs/2410.21465 — ShadowKV (verified)
11. https://arxiv.org/abs/2409.04992 — InstInfer (verified)
12. https://arxiv.org/abs/2410.05004 — HCache (verified)
13. https://arxiv.org/abs/2502.16002 — KVLink (verified)
14. https://arxiv.org/abs/2404.18322 — BlockLLM (verified)
15. https://arxiv.org/abs/2502.06888 — Klotski (verified; off-topic)
16. https://arxiv.org/abs/2512.11920 — CXL-SpecKV (verified; TITLE ONLY)
17. https://arxiv.org/abs/2506.20187 — "Breaking the Boundaries of Long-Context LLM Inference: Adaptive KV Management on a Single Commodity GPU" (verified title; body NOT read)

arXiv full HTML (all HTTP 200):
18. https://arxiv.org/html/2407.00079v1 — Mooncake
19. https://arxiv.org/html/2406.19707v1 — InfiniGen
20. https://arxiv.org/html/2310.07240v1 — CacheGen
21. https://arxiv.org/html/2312.05516v1 — Pensieve
22. https://arxiv.org/html/2510.09665v1 — LMCache
23. https://arxiv.org/html/2410.21465v1 — ShadowKV
24. https://arxiv.org/html/2409.04992v1 — InstInfer
25. https://arxiv.org/html/2410.05004v1 — HCache
26. https://arxiv.org/html/2502.16002v1 — KVLink
27. https://arxiv.org/html/2404.18322v1 — BlockLLM
28. https://arxiv.org/html/2403.19708v1 — CachedAttention / AttentionStore
29. https://arxiv.org/html/2601.13631v1 — ContiguousKV

Non-arXiv:
30. https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf — IMPRESS (FAST'25) full paper
31. https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress — IMPRESS presentation page

Search endpoints used (ID discovery only; every hit independently verified):
32. https://arxiv.org/search/?searchtype=all&query=CacheGen&size=25
33. https://arxiv.org/search/?searchtype=all&query=LMCache&size=25
34. https://arxiv.org/search/?searchtype=all&query=IMPRESS+KV+cache&size=25
35. https://arxiv.org/search/?searchtype=title&query=IMPRESS&size=50
36. https://export.arxiv.org/api/query?search_query=ti:%22CacheGen%22&max_results=10 (HTTP 429)
37. https://export.arxiv.org/api/query?search_query=ti:%22LMCache%22&max_results=10 (HTTP 429)
38. https://api.semanticscholar.org/graph/v1/paper/search?query=LMCache&fields=title,externalIds,year,abstract&limit=5 (HTTP 429)
39. https://dblp.org/search/publ/api?q=LMCache&format=json&h=8 (non-JSON/blocked)
