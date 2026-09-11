# R4 — Adversarial Prior-Art Screen: "Compute-before-Attention" / Predictive KV Prefetch

**Claims under test:** ⑤ compute-before-attention / predictive attention prefetch; ⑥ KV prefetching (`P(KV_i accessed at t+1 | Q_t)`).
**Screen date:** 2026-09-11.
**Evidence rule applied:** every source cited below was **fetched by me**, and every quotation is **verbatim from the fetched body** (title + quoted sentence both seen). Where I could not fetch a body, or where the number is the author's model rather than a measurement, that is stated explicitly. Secondary summaries are marked as such and never used as the basis of a verdict.

---

## T5 — VERDICTS

**Claim ⑤ — NARROWED.**
The *non-predictive* half of ⑤ is published with the identical value proposition ("fewer memory stalls", not fewer FLOPs) and measured on real Hopper hardware — Dong et al., AAAI-26, prefetches KV HBM→L2 with `cp.async.bulk.prefetch.L2` and reports stall-cycle reduction — so "predict which pages and prefetch into a faster level" survives **only** as the conjunction *attention-aware prediction → on-chip (L2/SMEM) destination*, which I found **no occupier of**; and the single paper that tried exactly that conjunction (Levy, Intel, `arXiv:2603.13430` §5.3) reports it **essentially failing**.

**Claim ⑥ — NARROWED (near-degenerate; residual identical to ⑤'s).**
The predict-then-prefetch loop is published at least four times (InfiniGen OSDI'24, OasisKV, SparDA, FlashMemory-DeepSeek-V4) and is even env-flag-gated in a modified SGLang — but **every** instance lands in the **host/remote → HBM** tier, and ⑥'s stated virtue that "a wrong prediction costs only bandwidth, never accuracy" is **contradicted in print** by OasisKV, so ⑥'s residual delta is the same on-chip slot as ⑤ and the two are not separable contributions.

**Neither V1 nor V2 fires.** V1 fails because the *exact* mechanism (prediction feeding an on-chip prefetch) is not published anywhere I could find. V2 fails because the closest measured end-to-end delta against a deployed baseline is **+15% / +7% / −2…−5%** versus FlashAttention-3 (not <5%). See T4 for why the screener's earlier 2–8% figure was never a measurement.

---

## T2 — THE SEPARATION (the point of this screen)

The prior pass's error was **O7: conflating two different decisions.** They do come apart cleanly. Both tables below are populated only from sources I fetched.

### (a) STORAGE-TIER prefetch — source is host DRAM / remote node / SSD, destination is GPU HBM

| Work | Venue / ID | What exactly is predicted | Source → Destination | Evidence |
|---|---|---|---|---|
| **InfiniGen** | OSDI 2024, `arXiv:2406.19707` | next attention layer's important tokens, "speculated by performing a minimal rehearsal with the inputs of the current layer and part of the query weight and key cache of the subsequent layer" | host memory → GPU | measured, "up to 3.00x" |
| **OasisKV** | `arXiv:2608.08097` v1 (Aug 2026) | next decode step's top-K KV blocks, from speculative-decoding draft tokens | host DRAM / remote memory → HBM | measured on 8×H100, 1.69× / 2.1× |
| **SparDA** | `arXiv:2606.04511` (NVIDIA, Jun 2026) | "the KV blocks needed by the next layer", via a per-layer Forecast projection | CPU → GPU | measured, up to 1.7× decode |
| **FlashMemory-DeepSeek-V4** | `arXiv:2606.09079` | "which ~10–15% of chunks the next 64 tokens will attend to" | GPU ↔ CPU offload/recall | measured, 2.8× on 8×H20; env-flag-gated fork |
| **OPKV** | MLSys 2026 **oral** | none per-step (plugin `select`/`recall`; optimises recall overhead) | CPU memory → GPU, "on demand" | measured, 1.3–1.8× |
| **vLLM #48445** | GitHub **RFC, OPEN** | layer N+1's block selection, from layer N (SparDA-style Forecast) | fires "through the existing KVConnector path" → PCIe/offload tier | **not shipped**; no branch, no PR |
| **vLLM PR #49123** | GitHub **PR, OPEN / NOT MERGED** (`"state":"OPEN"`, `"mergedTime":null`) | **none yet** — "This PR does not add the Forecast projection model implementation yet." | CPU→GPU, via the existing offload connector path | **scaffold only**; experimental, unmerged |
| **vLLM #52113 / #51428** | GitHub **RFCs, OPEN** | none — session-lifecycle / prefix-cache hints | prefix cache / offload | **not shipped** |
| **Span Queries** | MLSys 2026, `arXiv:2511.02749` | none — reorders requests for prefix-cache locality | prefix cache | measured, 10–20× TTFT |

### (b) COMPUTE-SIDE prefetch — source is HBM, destination is on-chip (L2 / BEOL / SMEM)

| Work | Venue / ID | Prediction step? | HBM → | Evidence |
|---|---|---|---|---|
| **Dong et al.** | **AAAI-26**, DOI `10.1609/aaai.v40i25.39224`, `arXiv:2504.06319` | **NO.** Algorithm 1 prefetches `bt[block_idx + w]` — the next block in the *already-known* block table. Address is deterministic. | **L2** (`cp.async.bulk.prefetch.L2`) | **measured** on H20 |
| **PRESERVE** | `arXiv:2501.08192` | **NO** — scheduled inside allReduce comm windows | "the on-chip cache" | measured, up to 1.6× |
| **Lee et al.** | `arXiv:2508.08457` / IEEE 11421351 | **NO** — "KV-cache prefetch opportunity" is chosen by *idle HBM bandwidth + on-chip space*, not by prediction; dense layer-by-layer addresses | 512 MB M3D BEOL on-chip | **simulated** (Timeloop), TPUv6e-like; 8.06× decode |
| **Levy** | `arXiv:2603.13430` (Intel, cs.AR) | attempted and **failed** — see §5.3 quote below | LLC (reservation + LRU, not prefetch) | measured traces + NCU |
| **Kelle** | MICRO 2025, `arXiv:2510.16040` | **NO** — eDRAM is the *primary* KV store, so there is nothing to fetch from HBM; contribution is eviction/recomputation/refresh control | (KV resident on-chip by construction) | simulated accelerator, 3.9× |

**The empty cell:** attention-aware prediction feeding an **on-chip** prefetch. Nothing I fetched occupies it.

**Headwind worth noting from the architecture side (HPCA 2026, verbatim):** the architecture community is currently moving *away* from predictors for sparse attention, because the predictor itself is the cost. Huizheng Wang et al., "PADE: A Predictor-Free Sparse Attention Accelerator via Unified Execution and Stage Fusion", HPCA 2026 Main Conference:

> "Sparse attention methods alleviate this by skipping low-relevance token pairs. However, current approaches lack practicality due to **the heavy expense of added sparsity predictor**, which severely drops their hardware efficiency. ... We propose PADE, a **predictor-free** algorithm-hardware co-design for sparse attention acceleration."

(PADE predicts *which token pairs to compute*, not which pages to fetch — it is not an occupier of the empty cell. But a screen that recommends "add a predictor" must answer why the predictor's own overhead is not the dominant cost, which is exactly the objection PADE was written to fix. PADE's numbers are on a synthesised accelerator vs. H100, not a deployed serving baseline.)

**Control group — attention-aware KV *selection* with no prefetch** (these do *not* belong in either prefetch table, and the prior screen should not count them as crowding): Quest, QUOKA, SeerAttention, CompactAttention (`arXiv:2605.16839`), TinyServe (`arXiv:2509.12211`).

---

## T3 — IS IT A FLAG?

**Compute-side (HBM→L2/SMEM): NO flag exists.** The AAAI-26 method is a CUDA-kernel modification using a Hopper-specific PTX instruction, not a config knob; I found no config flag, env var, or hook in vLLM / SGLang / TensorRT-LLM / FlashInfer that moves KV HBM→L2 on prediction.

**Storage-tier predictive version: the TIMING HOOK exists, the PREDICTION does not.** A real per-layer hook is present and in use — vLLM **PR #49123** "Add SparDA lookahead KV connector" states verbatim: *"Uses the existing `wait_for_layer_load()` per-layer connector hook, which matches the timing needed for layer-ahead prefetch."* But that PR is **OPEN and NOT MERGED** (raw page state `"state":"OPEN"`, `"mergedTime":null`), is self-described as *"experimental"*, and states plainly: *"**This PR does not add the Forecast projection model implementation yet.** It establishes the dedicated connector path and metadata surface for validating layer-ahead prefetch behavior before wiring in model-driven block prediction."* It also targets the **offload** path (*"The goal is to reduce KV offload stalls during long-context decoding"*), i.e. CPU→GPU. So: a storage-tier layer-ahead *scaffold* is being built in the open, with no predictor and no merge.

**Closest thing to "it's a flag":** FlashMemory-DeepSeek-V4 ships env flags in a **modified SGLang fork** — `SGLANG_DECODE_SWAP_P`, `SGLANG_PATHP_INDEX_K_OFFLOAD`, `SGLANG_PATHP_SCORE_RESIDENT`, `SGLANG_PATHP_PAGE_RECALL` — with "gate-off = exact baseline DeepSeek-V4 behavior". This is a fork, not upstream, and it is CPU↔GPU.

**Plainly: ⑤/⑥ cannot be expressed today via an existing config or hook.** The prediction→on-chip-prefetch combination requires a new kernel plus a new scheduling point.

---

## T4 — MAGNITUDE, DONE PROPERLY

### What IS measured

**(1) Attention-kernel stall share — Dong et al., AAAI-26, Table 1** (H20, Llama2-7B, batch 64, 4096 output tokens, vLLM **XFormers** backend):

> Compute throughput 23.35% · Memory throughput 47.10% · L1 cache hit rate (load) 0.75% · L2 cache hit rate (load) 0.06% · CPI 27.68 cycles · Stall Long Scoreboard 21.34 cycles

> "the average Cycles Per Instruction (CPI) reaches 27.68, with 21.34 cycles (77%) consumed by 'Stall Long Scoreboard' events, forming an absolute performance bottleneck."

After prefetch: stall cycles 21.34 → 4.13 (Llama2-7B) etc., "driving 65.8%–67.5% reductions in attention kernel CPI."
**Limit:** this is a share of **CPI inside the attention kernel**, on the **weak XFormers backend**, on an **MHA** model. It is *not* a share of decode wall-clock, and the same paper measures only **+15% / +7% / −2…−5% vs FlashAttention-3** on GQA models.

**(2) Per-kernel stall-cause breakdown in decode — Wang et al., `arXiv:2512.01644`** (cs.AR, vLLM 0.9.2, A100):

> "Decode's stall profile shifts fundamentally: Pipeline Busy plunges to 3~4%, while Memory Dependency dominates."

> "the dominant stall transitions from Execution Dependency (32.8% in Prefill) to **Memory Dependency (32.7% in Decode)**, consistent with intensive KV-cache operations."

> "**FFN-Up is memory-bound, with Memory Dependency reaching 58%** due to GEMV-like computation patterns that cannot hide HBM **weight-loading** latency."

> "both phases exhibit low GPU Execution (Prefill: 27%, Decode: 24%), meaning 70%–80% of cycles are spent stalled"

> "DM kernels dominating execution time in both phases" (DM = dense matmul; FA = fused attention)

**This is the most important measurement for the screen, and it cuts against the claim's premise:** inside decode, the *larger* memory-stall pool belongs to **weight** loading in FFN GEMMs (58%), not to KV access in attention (32.7%). Its Table 3 also asserts the operator switch to Attention at "extremely long context (>16K-32K+)" but attributes that to *prior* studies (Nawrot et al. 2025; Jiang et al. 2024), i.e. it is a citation, not their own measurement.

**(3) DSA access-pattern volatility — Levy, `arXiv:2603.13430`** (H200, NCU):
> Table 2: Dense vs Sparse — SM Utilization 23.5% → 10.9%; HBM BW Utilization 36.0% → 7.3%
> Table 3 (measured access patterns): persistence 1.82 steps (mean), lookback 3.29 top-k steps, new lookups 0.55, inter-layer 0.36
> Figure 9: "the average fetched KV page is only 35% utilized"

**(4) OasisKV fetch-cap ablation (measured):** throughput 2,178 → 824 tok/s as per-step traffic rises 0.30 → 5.05 GB — "The compute per step is unchanged across these rows, so every lost token of throughput is paid in bytes moved."

### What is NOT measured

**No one has published a measurement of the KV-memory-stall share of decode wall-clock at long context, against a deployed baseline (FA3 / production vLLM).** The two near-misses are (1) an attention-kernel-internal CPI share on a weak backend with an MHA model, and (2) a per-kernel stall-cause breakdown in which weight loading, not KV, is the bigger memory-stall pool.

**That absence is itself the finding.** Claim ⑤'s headline value proposition — "not fewer FLOPs, fewer memory stalls" — has never been quantified as a fraction of decode time. So **both** the screener's earlier 2–8% ceiling **and** any claim of a large win are, today, unmeasured. The prior 2–8% was the screener's own arithmetic and is not a valid kill; but neither does any published measurement support a large number.

### Bounds (explicitly labelled as bounds, with assumptions)

- **BOUND (author's roofline model, not a measurement):** Levy Table 4 — LL cache reserved 0 / 5 / 10 / 15 / 20 MB ⇒ modelled slowdown 1.87 / 1.67 / 1.5 / 1.3 / 1.15 (LLaMA-3.1-70B, 20 layers/GPU, 100 tok/s/user). The paper itself calls this "a roofline model".
- **BOUND (author's arithmetic, not a measurement):** OasisKV §3.2 — "for Qwen3-8B ... a decode step takes ≈17 ms, so a ≈64 GB/s PCIe link admits only ≈118 tokens of newly active KV per request per step before decoding stalls." Assumes a 2K active context and 64 GB/s PCIe.
- **MEASURED DEPLOYED-BASELINE DELTA (the closest thing to a ceiling for the non-predictive form):** AAAI-26 vs FlashAttention-3 — **+110%** (Llama2-7B, 32:32 MHA), **+15%** (Llama3-8B, 32:8 GQA), **+7%** (Qwen2.5-14B, 40:8 GQA), **−2% to −5%** (Qwen2.5-7B, 28:4 GQA). The paper's own explanation: "prefetching benefits decay linearly with KV head reduction."

---

## T1 — CLOSEST COMPUTE-SIDE OCCUPIER (verbatim)

**Dong, Miao, Li, Zheng, Wang, Wu, Lyu — "Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching", AAAI-26, `arXiv:2504.06319` (v1 8 Apr 2025, v2 8 Nov 2025; AAAI DOI `10.1609/aaai.v40i25.39224`, published 2026-03-14).**

> "we propose an L2 Cache-oriented asynchronous KV Cache prefetching method to break through the memory bandwidth bottleneck in LLM inference through computation-load overlap. ... our method proactively prefetches required KV Cache into GPU L2 cache, enabling high-speed L2 cache hits for subsequent accesses and **effectively hiding HBM access latency within computational cycles**."

> "NVIDIA CUDA introduces an L2 cache-oriented asynchronous prefetch mechanism through the PTX instruction `cp.async.bulk.prefetch.L2`."

**Why this is the closest occupier and why it is still not the claim:**
- **Level:** HBM → **L2**. Compute-side, not storage-tier. The value proposition is **explicitly fewer memory stalls** (stall cycles 21.34 → 4.13), **not fewer FLOPs**. This is the O7-sensitive part: the prior pass's "storage-tier only" framing was **wrong** — a compute-side L2 KV prefetch with this exact value proposition is **published and measured**.
- **But there is no prediction.** Algorithm 1 loops `block_idx` over `[s,e)` and prefetches `bt[block_idx + w]`, the next block **of the block table it is already iterating**. The address is statically known; the contribution is latency hiding by software pipelining, i.e. classic prefetching, not "predicting which KV pages the next query will access."

**The exact prediction→on-chip mechanism: none found.**

### The one direct attempt, and it failed — verbatim

**Noam Levy (Intel), "Dynamic Sparse Attention: Access Patterns and Architecture", `arXiv:2603.13430`, cs.AR, 13 Mar 2026, §5.3 "Prediction of top-k":**

> "An inviting optimization is to predict the top-`k` values ahead of when they are needed, to hide the latency of bringing these pages from HBM behind computation. We attempted a learned approach, training a predictor that uses hidden states from previous tokens. However, **we achieved results only slightly better than keeping the previous step's top-`k` in memory, essentially failing at this approach.** We interpret this as an indication that DSA's fine-grained top-`k` selection is highly volatile and strongly influenced by the current-step query. Other approaches to predicting top-`k` may still be promising, but we have not explored them."

And from the same paper's abstract:

> "its token-dependent selection pattern introduces a system-level challenge: **the KV working set is fragmented, volatile, and difficult to prefetch**"

**Independent corroboration** — OasisKV (`arXiv:2608.08097`), Figure 4:

> "the previous-token proxy recovers the true top-20 blocks unreliably, and its accuracy varies widely across layers, so a system built on it either misses important blocks or must widen its budget to compensate."

So **two independent primary sources** agree that the naive conditioner — the previous step's query/hidden state — is a bad predictor. What *does* work, per OasisKV, is a **lookahead/draft-token** signal:

> "future important tokens can be predicted accurately in advance using lookahead tokens drafted by speculative decoding (SD)"

and per SparDA, a **next-layer Forecast projection**. Both of those working predictors were pointed at the **CPU/remote → HBM** tier, never at L2/SMEM.

### Claim ⑥'s stated virtue is false in print

OasisKV, §3.1:

> "**A miss is not free**, as the system must either drop the block, **losing attention context and accuracy**, or issue a corrective on-demand fetch, which puts retrieval back on the critical path, the exact cost prefetching set out to remove."

Its measured fetch-cap ablation confirms this is a real trade, not a hypothetical: tightening the cap raises accuracy 74.9 → 77.4 while throughput falls 2,178 → 824 tok/s. So a wrong prediction at the storage tier costs **accuracy or critical-path latency**, not "only bandwidth". (A wrong prediction at the *on-chip* tier would cost only bandwidth, since the data is still addressable — which means ⑥'s virtue is true only in the on-chip tier that nobody occupies. That is the sharpest way to state ⑥'s surviving delta.)

---

## SMALLEST SURVIVING DELTA

**Predict the decode attention KV block set from a signal other than the previous step's query (draft/lookahead tokens, or next-layer rehearsal), and land it in L2/SMEM rather than host DRAM — measured end-to-end against FlashAttention-3 / production vLLM, per-GQA-ratio.**

Everything on either side of that sentence is already occupied:
- on-chip prefetch without prediction — published and measured (AAAI-26), and simulated at scale (Lee et al.);
- prediction without on-chip — published four times and flag-gated in a fork;
- on-chip KV *residency* without prefetch — published (Levy);
- the naive predictor — published as a **failure**.

**Risk to the delta, stated honestly:** the measured envelope is unforgiving. AAAI-26 already captures most of the L2 benefit without predicting anything, and its own GQA ablation goes **negative at 7:1** ("prefetching benefits decay linearly with KV head reduction"). Levy's modelled ceiling for on-chip KV reservation is 1.87 → 1.15 slowdown. Any new work must beat *that* baseline, not the XFormers one, and must show a per-GQA-ratio win.

---

## WHAT I COULD NOT VERIFY

1. **Architecture-venue proceedings sweep (HPCA 2026, ISCA 2025/26, MICRO 2025, ASPLOS 2025/26, MLSys 2025/26) is INCOMPLETE — and this is the single biggest hole in the screen.** I enumerated **MLSys 2026** via its own program plus an accepted-paper index, and I spot-checked **HPCA 2026** (PADE, AQPIM — individual detail pages fetch, the session index does not). I did **not** enumerate ISCA 2025/26, MICRO 2025, or ASPLOS 2025/26 programs. A delegated sweep of these venues **was stopped before it finished**; its own closing note was *"ACM DL and CSDL are blocked"* — so `dl.acm.org` (ASPLOS, MICRO, ICS, SC) and the IEEE/CSDL proceedings pages are **not retrievable from this session**, which is a hard retrieval limit, not evidence of absence. **The "empty cell" claim in T2 is therefore provisional on ISCA/MICRO/ASPLOS.** I did fetch one ASPLOS'26-adjacent item (near-storage processing, `arXiv:2502.09921`) only via a search-result snippet, not its body — not counted.
2. **`usenix.org` PDFs could not be fetched** (content-type `application/pdf` rejected by the fetch tool, and the OSDI'24 InfiniGen PDF was not retrieved). InfiniGen is therefore verified from its **arXiv** abstract (`arXiv:2406.19707`, comments field "OSDI 2024"), not from the USENIX proceedings PDF.
3. **The AAAI-26 PDF was not retrieved as PDF**; the AAAI landing page (abstract, DOI, pagination) and the **arXiv v1 HTML full text** were retrieved. The three headline numbers vs. FA3 come from Figure 3 prose, not from a table I could re-read.
4. **OpenReview abstracts were blocked** (browser check on `/forum/`, HTTP 403 on `api2.openreview.net`). The MLSys 2026 span-queries abstract was obtained from `arXiv:2511.02749` instead; OPKV's abstract came from the MLSys virtual site (primary venue page).
5. **vLLM/SGLang #21846 and the `--hicache-storage-prefetch-policy` flag from the earlier pass were NOT re-verified by me.** I verified #48445, #52113, #51428 only. Treat the earlier pass's citations as unconfirmed.
6. **FlashMemory-DeepSeek-V4 is a third-party repository** (`libertywing/FlashMemory-Deepseek-V4`), not an official DeepSeek release. OasisKV cites the same work as "(Wang et al., 2026)". I verified the README and the arXiv abstract, **not** that the env flags behave as documented.
7. **`arXiv:2608.08097` (OasisKV) is a v1-only preprint** (submitted 8 Aug 2026, one version) carrying an ACM ISBN placeholder `978-1-4503-XXXX-X/2018/06` — i.e. ACM-submission formatting, **no accepted venue confirmed**. No withdrawal or retraction marker in the raw HTML. Same for `2606.09079` (FlashMemory, "Technical report. 11 pages"), `2603.13430`, `2606.04511`, `2512.01644`. Raw-HTML retraction greps returned clean for all of them.
8. **`arXiv:2508.08457` / IEEE 11421351:** the IEEE Xplore body was **not** retrievable (JS-gated); verified via the arXiv abstract + HTML full text, whose thanks-note reads "Submitted to IEEE MICRO Special Issue ... for review". Publication status in IEEE MICRO **unconfirmed**.
9. **No published measurement exists** isolating the KV-memory-stall share of decode wall-clock at long context (see T4) — this is a finding, not a retrieval failure, but it does mean no V2-style kill was even constructible for the predictive variant.

---

## SOURCE LIST (all fetched; primary unless noted)

| # | Source | What it establishes |
|---|---|---|
| 1 | `arXiv:2504.06319` (abs + v1 HTML), AAAI DOI `10.1609/aaai.v40i25.39224` | compute-side HBM→L2 KV prefetch, **no prediction**; Table 1 stalls; vs-FA3 deltas |
| 2 | `arXiv:2608.08097` (abs + full HTML) | lookahead SD prediction; host/remote→HBM; miss-is-not-free quote |
| 3 | `arXiv:2606.04511` (abs) + `NVlabs/SparDA` README | next-layer Forecast → CPU→GPU |
| 4 | `arXiv:2606.09079` (abs) + FlashMemory README | next-64-token chunk prediction; CPU offload; SGLang env flags |
| 5 | `arXiv:2406.19707` (abs) | InfiniGen; host-memory prefetch |
| 6 | `arXiv:2603.13430` (abs + v1 HTML) | LLC reservation; **failed top-k predictor**; NCU utilization; access-pattern stats |
| 7 | `arXiv:2512.01644` (abs + v1 HTML) | decode stall-cause breakdown; FFN-Up 58% weight-bound vs AttnCore 32.7% KV |
| 8 | `arXiv:2501.08192` (abs) | PRESERVE; off-chip HBM → on-chip cache |
| 9 | `arXiv:2508.08457` (abs + v1 HTML) | packing-prefetch scheduler → BEOL on-chip; **simulated** |
| 10 | `arXiv:2511.02749` (abs) | Span Queries; prefix-cache locality, not compute-side |
| 11 | `mlsys.org/virtual/2026/oral/3844` | OPKV, MLSys 2026 oral; CPU offload + recall |
| 12 | GitHub vLLM **#48445** (HTML) | RFC OPEN; lookahead prediction over KVConnector; "triggered a layer early today" open question |
| 13 | GitHub vLLM **#52113**, **#51428** (HTML) | RFCs OPEN; session-lifecycle/prefix-cache; no attention prediction |
| 14 | `arXiv:2605.16839` (ar5iv HTML) | CompactAttention; KV *selection*, no prefetch |
| 15 | `arXiv:2509.12211` (abs) | TinyServe; query-aware page *selection*, no prefetch |
| 16 | `jianyuh.github.io` MLSys 2026 notes; `awesome-papers` mlsys-2026.md | **secondary** — used only to locate OPKV/span-queries, both then verified at the venue |
| 17 | GitHub vLLM **PR #49123** (HTML, raw state JSON) | `SparDALookaheadConnector`; **OPEN/UNMERGED**; uses existing `wait_for_layer_load()` hook; **no Forecast model** |
| 18 | `2026.hpca-conf.org` PADE detail page | HPCA 2026 "predictor-free" sparse-attention accelerator (anti-predictor trend) |
| 19 | `arXiv:2510.16040` (abs) | Kelle, MICRO 2025; eDRAM-primary KV store; no prefetch |
