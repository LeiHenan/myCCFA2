# Verification-aware attention / KV access for speculative decoding — evidence recon
Date: 2026-09-14. Every entry below was fetched with `curl` and the title read back from the page.
Nothing here is from a search snippet.

## 0. CRITICAL CHECK — arXiv 2605.19893 — VERIFIED, both titles confirmed

| | |
|---|---|
| **v1** (19 May 2026 14:24:27 UTC) | **"SpecSA: Bridging Speculative Decoding and Sparse Attention for Efficient LLM Inference"** |
| **v2** (20 May 2026 15:53:57 UTC) | **"SSV: Sparse Speculative Verification for Efficient LLM Inference"** |
| Authors | Zhibin Wang, Ziyu Zhong, Nuo Shen, Yuhang Zhou, Rong Gu, Sheng Zhong |
| Subject | cs.OS (Operating Systems) |
| Comments/venue | **none present** — no venue or comment field on either abs page |
| URLs | https://arxiv.org/abs/2605.19893 (200), `/2605.19893v1` (200), `/2605.19893v2` (200), https://arxiv.org/html/2605.19893v2 (200, 309 KB) |

Verbatim (v2 abstract):
> "speculative verification relies on cross-query commonality, whereas dynamic sparse attention assigns query-specific sparse layouts. This mismatch limits KV-block reuse, amplifies NSA's branch-wise overheads"

Verbatim (v2 §3.1 / §4.1):
> "We observe that nearby verifier queries often select overlapping KV blocks, but decoding-oriented sparse kernels process each query independently and reload these blocks"
> "a tree node can be grouped either with its siblings (candidates for the same position that are likely synonymous) or with its parent and children"
> "the selected-block overlap ratio between adjacent verifier queries typically remains around 50%–90% across most layers"

Mechanism: exact **merged-schedule** variant = union of per-query selected blocks, dedup on chip, load once, row-wise mask to −∞ to keep per-query semantics; **approximate shared-index** variant = one representative query (longest prefix) supplies the layout for the group. Cross-query grouping is *overlap-driven within a group of C adjacent queries*, i.e. **overlap-based, not nesting-based**.

**1. Does SSV cite 2512.21911? NO.** I extracted the entire bibliography from the fetched v2 HTML. Zero hits for `2512.21911`, "Sparse Computation in Verification", or "Jikai". The complete set of arXiv IDs in SSV's reference list is: 1803.05457, 1905.12322, 2004.05150, 2106.06899, 2302.01318, 2307.08691, 2401.10774, 2401.15077, 2407.21783, 2502.20766, 2503.01840, 2512.02556, 2512.16391, 2602.03560, 2603.12201, 2605.19893. **The two papers are mutually independent — this is the strongest signal that the framing is emerging in parallel, not being cited through.**

## 2. Papers that frame speculative verification queries as a correlated/grouped batch sharing a KV working set

`Title | Venue/Date | URL | VERIFIED | verbatim | nesting vs overlap`

1. **Accelerate Speculative Decoding with Sparse Computation in Verification** | 26 Dec 2025, Comments "Pre-print" | https://arxiv.org/abs/2512.21911 | VERIFIED | *"tokens within the same step retrieve highly similar blocks, with an average overlap exceeding 0.8 in most layers... we use the first token to perform block retrieval, while the other tokens reuse its retrieved blocks during verification"* | **OVERLAP** — measures the overlap directly, then collapses it to one representative (first draft token). Grouping is positional.
2. **SSV: Sparse Speculative Verification for Efficient LLM Inference** (v2; v1 = SpecSA) | 20 May 2026, cs.OS, no venue | https://arxiv.org/abs/2605.19893 | VERIFIED | *"grouping adjacent queries into a single thread block to amortize KV-block loads"* | **OVERLAP** — explicit union/merge of selected block sets; two variants (exact merge vs shared-index).
3. **Vegas: Self-Speculative Decoding with Verification-Guided Sparse Attention** (v1 = "SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding", 6 Feb 2026) | v2 29 May 2026, Comments **"Accepted to ICML'26"** | https://arxiv.org/abs/2602.07223 | VERIFIED | *"maximizing the coverage of attention weights across all draft tokens (including both accepted and discarded ones) yields much higher acceptance probabilities"*; Fig. 7: *"The pairwise overlap ratio of the top-1024 prefix tokens selected by each draft token... diminishes as the positional gap between draft tokens increases"* | **OVERLAP** — treats the γ+1 verifier queries as one group jointly optimized (Eq. 2 sums logits over all t = 1..γ+1). Direction is inverse: verification logits *select KV for the next draft*.
4. **vLLM issue #47763** (engineering report, not a paper) | opened 6 Jul 2026 by yifjiang | https://github.com/vllm-project/vllm/issues/47763 | VERIFIED | *"the selected sparse K/V blocks are streamed from HBM once per draft token — verify KV-read traffic scales ~decode_query_len×"*; *"a shared-KV kernel would load the union of selected blocks per request-tile and mask per position"* | **OVERLAP + prefix** — names the union-and-mask fix and states "the shared prefix + overlapping selections are where the reuse lives". Independent convergent arrival at the SSV design.

## 3. Sparse-attention KV *selection* applied specifically to the verification pass

| Title | Date | URL | Verdict | Verbatim | Nesting vs prefix |
|---|---|---|---|---|---|
| Accelerate Speculative Decoding with Sparse Computation in Verification | 26 Dec 2025 | https://arxiv.org/abs/2512.21911 | VERIFIED | *"each step must verify multiple candidate tokens simultaneously... This multi-token verification introduces new challenges for efficiently selecting relevant context"* — Quest-style block retrieval driven by q₀ | OVERLAP (first-token proxy) |
| SpecAttn: Speculating Sparse Attention | 31 Oct 2025, Comments "Accepted to NeurIPS 2025 Workshop on Structured Probabilistic Inference & Generative Modeling" | https://arxiv.org/abs/2510.27641 | VERIFIED | *"exploit the attention weights already computed by the draft model during speculative decoding to identify important tokens for the target model, eliminating redundant computation"* | draft→target transfer; not cross-query grouping |
| SpecPV: Improving Self-Speculative Decoding for Long-Context Generation via Partial Verification | v1 2 Dec 2025, v2 29 Aug 2026 | https://arxiv.org/abs/2512.02337 | VERIFIED | *"performs fast verification using partial key-value states (KV) and periodically applies full verification to eliminate accumulated errors"* | partial-KV verification; not cross-query grouping |
| Vegas | 6 Feb 2026 / 29 May 2026 | https://arxiv.org/abs/2602.07223 | VERIFIED | see §2 | OVERLAP (inverse direction) |

## 4. Prefix-sharing / IO-aware tree attention (nesting, applied to speculative verification)

| Title | Venue/Date | URL | Verdict | Verbatim | Nesting vs prefix |
|---|---|---|---|---|---|
| DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference | v1 30 Mar 2024, v4 7 Mar 2025; Comments "accepted by ICLR'25" | https://arxiv.org/abs/2404.00242 | VERIFIED | *"DeFT-Flatten groups the prefix's KV cache with all shared queries, ensuring the prefix KV cache is only loaded once, significantly reducing redundant loading"* | **NESTING / prefix only** — no KV *selection*, no sparse layout. Benchmarked on "speculative decoding with 64 queries and a prompt with 4k tokens". |
| Hydragen: High-Throughput LLM Inference with Shared Prefixes | v1 7 Feb 2024, v2 13 May 2024 | https://arxiv.org/abs/2402.05099 | VERIFIED | *"an hardware-aware exact implementation of attention with shared prefixes... can be applied to tree-based prompt sharing patterns"* | **prefix only** — abstract has **0** occurrences of speculative/draft/verification. |
| Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware | 19 Jul 2026, Param Chordiya | https://arxiv.org/abs/2607.17283 | VERIFIED | *"speculative decoding pays off only when verification is genuinely batch-parallel and the draft/target latency gap is real"* | IO/hardware-angle on verification; no KV-set grouping. |

## 5. Foundation sparse-attention baselines — ALL VERIFIED, **NONE applied to verification**

Abstracts were fetched and grepped case-insensitively for `speculative|draft token|verification`:
**0 hits in every single one.** They are long-context prefill/decode methods only.

| Name | arXiv ID | Title read back | Date |
|---|---|---|---|
| MoBA | 2502.13189 | MoBA: Mixture of Block Attention for Long-Context LLMs | 18 Feb 2025 |
| Quest | 2406.10774 | Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference | v1 16 Jun 2024, v2 26 Aug 2024 |
| MInference | 2407.02490 | MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention | v1 2 Jul 2024, v2 30 Oct 2024 |
| SeerAttention | 2410.13276 | SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs | v1 17 Oct 2024, v4 17 Feb 2025 |
| XAttention | 2503.16428 | XAttention: Block Sparse Attention with Antidiagonal Scoring | 20 Mar 2025 |
| SnapKV | 2404.14469 | SnapKV: LLM Knows What You are Looking for Before Generation | v1 22 Apr 2024, v2 17 Jun 2024 |
| H2O | 2306.14048 | H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | v1 24 Jun 2023, v3 18 Dec 2023 |
| StreamingLLM | 2309.17453 | Efficient Streaming Language Models with Attention Sinks | v1 29 Sep 2023, v4 7 Apr 2024 |
| NSA | 2502.11089 | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | v1 16 Feb 2025, v2 27 Feb 2025 |

Every claimed ID above resolved to the claimed title. Baseline IDs in this area are reliable; the
*search snippets* about them were not (see §7).

## 6. Adjacent verified work (spec decoding × tree / KV / quantization)

| Title | Venue/Date | URL | Verbatim / note |
|---|---|---|---|
| MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding | v1 20 Aug 2024, v5 2 Apr 2025 | https://arxiv.org/abs/2408.11049 | cited by Vegas as the sparse-attention self-spec-dec baseline |
| SpecExtend: A Drop-in Enhancement for Speculative Decoding of Long Sequences | v1 27 May 2025, v4 19 Jan 2026 | https://arxiv.org/abs/2505.20776 | Vegas: "conditions its KV selection on the last accepted token, which is indeterminate during the attention kernel execution" |
| TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding | v1 18 Apr 2024, v3 4 Aug 2024 | https://arxiv.org/abs/2404.11912 | hierarchical draft/verify |
| QSpec: Speculative Decoding with Complementary Quantization Schemes | v1 15 Oct 2024, v3 2 Oct 2025 | https://arxiv.org/abs/2410.11305 | spec decoding × KV quantization; treats them as complementary knobs, not as correlated verification queries |
| TAPS: Target-Aware Prefix Tree Selection for Diffusion-Drafted Speculative Decoding | 30 May 2026 | https://arxiv.org/abs/2606.00487 | "verification is prefix-conditioned" — tree topology/budget, **not** KV set structure |
| SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs | 30 Nov 2025, Comments "Accepted by ASPLOS 2026" | https://arxiv.org/abs/2512.00722 | DLM attention heads drive retrieval; adjacent to SpecAttn |
| Bole: Efficient Tree Speculation for Hybrid-Attention Language Models | 3 Aug 2026 | https://arxiv.org/abs/2608.01651 | tree speculation, hybrid attention |
| From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning (SpecGuard) | Findings of ACL 2026, July 2026, pp. 17457–17471 | https://aclanthology.org/2026.findings-acl.864/ | "verification-aware" **by name only**; step-level correctness via attention grounding score, no KV-set grouping |
| Verification-Aware Training for Speculative Decoding | 31 Aug 2026 | https://arxiv.org/abs/2608.30135 | title verified; training-side |

## 7. COULD NOT VERIFY (exact URLs + HTTP status)

- `https://arxiv.org/search/?searchtype=all&query=<any>&start=0` → **HTTP 406**, 0 bytes (every query; also without the trailing slash, and with browser Accept/Accept-Language headers).
- `http://export.arxiv.org/api/query?search_query=...` → **HTTP 429**, body "Rate exceeded."
- `https://api.semanticscholar.org/graph/v1/paper/search?query=...` → **HTTP 429**.
- `https://api.openalex.org/works?search=...` → **HTTP 429**, "Insufficient budget. This request costs $0.001 but you only have $0 remaining. Resets at midnight UTC."
- `https://www.semanticscholar.org/paper/...` (Vegas and sparse-verification pages) → **HTTP 202**, bot challenge.
- `https://html.duckduckgo.com/html/?q=...` and `https://lite.duckduckgo.com/lite/?q=...` → **HTTP 202**, "Select all squares containing a duck" challenge.
- `https://www.bing.com/search?q=...` → HTTP 200 but returned a ja-JP shell with no usable result bodies.
- **Not verified because I did not fetch it:** the arXiv version of "Accelerate Speculative Decoding with Sparse Computation in Verification" was verified at 2512.21911; the Semantic Scholar mirror page was not reachable. `github.com/platformxlab/vegas` README was fetched (200) and is the source for the 2602.07223 BibTeX.

**Networking note for downstream agents:** only `arxiv.org/abs/*`, `arxiv.org/html/*`, `icml.cc/virtual/*`, `aclanthology.org/*`, and `github.com/*` were reachable. All
programmatic search APIs (arXiv search, export API, S2, OpenAlex) and both HTML search engines were blocked. Discovery must go through the `web_search` tool, then be confirmed by fetching `arxiv.org/abs/<id>`.
