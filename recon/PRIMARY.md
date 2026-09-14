# Verification-aware attention / KV access for speculative decoding — primary-source baseline

Date of reconnaissance: 2026-09-14. All IDs below were verified by fetching `arxiv.org/abs/<id>` and
reading the title back off the fetched page. All quotes are verbatim from the fetched HTML full text
(`arxiv.org/html/<id>v<N>`) or, where noted, from fetched source code.

Working dir: `/Users/leihenan/Desktop/myProject/recon/`

---

## 1. Medusa — arXiv 2401.10774

- **Title:** Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
- **Venue/Date:** Submitted 19 Jan 2024; v3 14 Jun 2024. No venue in the arXiv Comments field
  (only a code URL). v3 fetched.
- **URL:** https://arxiv.org/abs/2401.10774 | full text https://arxiv.org/html/2401.10774v3
- **Verified state:** VERIFIED (title + full text fetched).

**Verbatim (paper, §2.1.2 Tree Attention):**

> "To guarantee that each token only accesses its predecessors, we devise an attention mask that
> exclusively permits attention flow from the current token back to its antecedent tokens. The
> positional indices for positional encoding are adjusted in line with this structure."

> "Within this tree, only a token's predecessors are seen as historical context, and our attention
> mask ensures that the attention is only applied on a token's predecessors. By employing this mask
> and properly setting the positional indices for positional encoding, we can process numerous
> candidates simultaneously without the need to expand the batch size."

> "This attention mechanism diverges from the traditional causal attention paradigm. Within this
> framework, only tokens from the same continuation are regarded as historical data."

**Verbatim (paper, §1):**

> "we propose generating multiple candidate continuations using the Medusa heads and verifying them
> concurrently through a simple adjustment to the attention mask."

**Verbatim (paper, §3.3.1, the cost statement):**

> "The decline in speed in Fig. 4(b) is attributed to the increased overhead introduced by the
> compute-bound. While a more complex tree can improve acceleration, it does so at the cost of speed
> due to intensive matrix multiplications for linear layers and self-attention."

**Code-level evidence (fetched 2026-09-14):**
`https://raw.githubusercontent.com/FasterDecoding/Medusa/main/medusa/model/utils.py`
`generate_medusa_buffers()` builds the mask as a dense N×N matrix:

```python
medusa_attn_mask = torch.eye(medusa_len, medusa_len)
medusa_attn_mask[:, 0] = 1
...
        medusa_attn_mask[j + start + 1, ancestor_idx] = 1
...
"medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
```

and `medusa_model.py` comments it as
> "# Cache medusa buffers (the fixed patterns for tree attention)".

**KV-set structure exploited:** ONLY the shared prefix (column 0 of the mask, `[:, 0] = 1`).
No nesting/overlap exploitation: the mask is a **dense** `medusa_len × medusa_len` additive mask
applied over the whole flattened tree; non-ancestor positions are masked out, not skipped. The
KV reads for non-ancestor tree positions still happen. Medusa's "sparse tree" ablation (§3.3.1,
Appendix C) means *fewer candidate nodes*, NOT sparse attention over KV.

---

## 2. SpecInfer — arXiv 2305.09781

- **Title:** SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based
  Speculative Inference and Verification
- **Venue/Date:** Submitted 16 May 2023; v4 1 Apr 2024. **Comments: ASPLOS'24** (verified).
- **URL:** https://arxiv.org/abs/2305.09781 | full text https://arxiv.org/html/2305.09781v4
- **Verified state:** VERIFIED.

**Verbatim (§4.2 Tree-based Parallel Decoding) — the KV-conflict statement:**

> "A key challenge SpecInfer must address in computing tree attention is managing key-value cache."

> "However, when computing tree attention, different sequences in a token tree may include
> conflicting key-value caches. For example, for the speculated token tree in Figure 4, two token
> sequences (t2,t3,t4,t5) and (t2,t3,t8,t9) have different keys and values for the third and fourth
> positions."

**Verbatim — the redundant-computation statement:**

> "A straightforward approach to supporting key-value cache is employing the sequence-based decoding
> of existing LLM inference systems and using a different key-value cache for each sequence of a
> token tree, as shown on the left of Figure 4. However, this approach is computationally very
> expensive and involves redundant computation, since two token sequences sharing a common prefix
> have the same attention outputs for the common prefix due to the causal mask in Equation 3."

**Verbatim — the DSF shared-cache statement:**

> "Instead of caching the keys and values for individual token sequences of a token tree, SpecInfer
> reuses the same key-value cache across all token sequences by leveraging a depth-first search
> mechanism to traverse the token tree"

**Verbatim — THE decisive passage on what the kernel actually does:**

> "We introduce topology-aware casual mask to fuse tree attention computation of all tokens in a
> single kernel. To batch attention computation, SpecInfer uses a tree topology instead of the
> original sequence topology to store the keys and values of all tokens in a token tree in the
> key-value cache."

> "This approach allows SpecInfer to fuse the attention computation into a single kernel but also
> results in attention scores that violate the causal dependency (e.g., t7's attention computation
> uses all previous tokens, including t5 which is not in t7's token sequence). To fix the attention
> scores for these pairs, SpecInfer updates the causal mask based on the token tree's topology.
> This approach computes the exact same attention output as incremental decoding, while resulting in
> much fewer kernel launches compared to sequence-based decoding."

**Verbatim — the summary claim (§6.5):**

> "The improvement is realized by (1) eliminating redundant attention computation for sequences with
> a shared prefix, and (2) fusing tree attention of all tokens in a single kernel through the
> topology-aware casual mask (see §4.2)."

**Definition of tree attention (Def. 4.1):**
> "For a token tree N and an arbitrary node u ∈ N, its tree attention is defined as the output of
> computing the original Transformer-based sequence attention on S_u (i.e., the token sequence
> represented by u)"

**KV-set structure exploited:** shared prefix ONLY, at the level of *not recomputing* the prefix,
plus one fused kernel. The quoted t7/t5 passage is explicit that the single fused kernel
**over-computes** attention over the whole tree KV buffer and then repairs the scores with a
topology mask. Node-level KV-set nesting is NOT exploited to skip work.

---

## 3. EAGLE / EAGLE-2 / EAGLE-3

### 3a. EAGLE — arXiv 2401.15077
- **Title:** EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
- **Venue/Date:** Submitted 26 Jan 2024; v2 4 Feb 2024 (v3 4 Mar 2025). No venue in Comments.
- **URL:** https://arxiv.org/abs/2401.15077 | full text https://arxiv.org/html/2401.15077v2
- **Verified state:** VERIFIED.

**Verbatim (§3.3 Verification phase):**
> "Employing tree attention, the target LLM computes the probability of each token in the
> tree-structured draft through a single forward pass."

**Verbatim (§4.3.1 Tree attention — the cost statement):**
> "EAGLE, similar to SpecInfer and Medusa, employs tree attention, where both the generation and
> validation of drafts are tree-structured. ... The implementation of tree draft and verification in
> EAGLE results in an approximate increase of 0.6-0.8 in the average acceptance length and about
> 0.3-0.5 in the speedup ratio. Compared to chain draft and verification, tree draft and
> verification do not increase the number of forward passes in the model (both the target LLM and
> the draft model), but they do increase the number of tokens processed per forward pass.
> Consequently, the improvement in the speedup ratio is less pronounced than the increase in
> average acceptance length."

> "Tree attention consumes more computational resources."

**KV-set structure exploited:** shared prefix only. No nesting claim. EAGLE explicitly measures the
cost of tree verification as "more tokens processed per forward pass" and "more computational
resources" — i.e. it treats attention cost as a function of token count, not of KV access structure.

### 3b. EAGLE-2 — arXiv 2406.16858
- **Title:** EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees
- **Venue/Date:** Submitted 24 Jun 2024; v2 30 Jun 2024. No venue in Comments.
- **URL:** https://arxiv.org/abs/2406.16858 | full text https://arxiv.org/html/2406.16858v2
- **Verified state:** VERIFIED.

**Verbatim (§4.2, attention mask for verification):**
> "Afterwards, we flatten the selected tokens into a one-dimensional sequence to serve as the input
> for the verification phase. To ensure consistency with vanilla autoregressive decoding, we also
> need to adjust the attention mask. In vanilla autoregressive decoding, each token can see all
> preceding tokens, resulting in a lower triangular attention matrix. When using a draft tree,
> tokens from different branches should not be visible to each other. Therefore, the attention mask
> must be adjusted according to the tree structure to ensure that each token can only see its
> ancestor nodes."

**Verbatim (§4.1, draft-model side):**
> "Thanks to tree attention, the draft model can simultaneously input all tokens from the current
> layer and compute the probabilities for the next tokens in a single forward pass, thereby
> expanding all tokens in the current layer."

**KV-set structure exploited:** shared prefix only. "Flatten" + "adjust the attention mask" is
explicitly the densify-then-mask pattern.

### 3c. EAGLE-3 — arXiv 2503.01840
- **Title:** EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test
- **Venue/Date:** Submitted 3 Mar 2025; v3 23 Apr 2025. No venue in Comments.
- **URL:** https://arxiv.org/abs/2503.01840 | full text https://arxiv.org/html/2503.01840v3
- **Verified state:** VERIFIED.

**Verbatim (§2 / §3, verification):**
> "In the verification stage, EAGLE uses tree attention to parallelize the verification of the draft
> tree."

**Verbatim (§3.1 — closest thing in the EAGLE family to exploiting mask structure for compute; this
is the DRAFT model's training-time test, NOT the target verification pass):**
> "All attention masks are diagonal, except when the original training data is used as the key.
> Using matrix multiplication in this case would result in significant computational waste, so we
> can use vector dot products to calculate the attention score only for the corresponding positions."

**KV-set structure exploited:** shared prefix only for verification. No verification-pass KV
access-cost analysis at all; the paper's attention-mechanism surgery is confined to the draft model.

---

## 4. Sequoia — arXiv 2402.12374

- **Title:** Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding
- **Venue/Date:** Submitted 19 Feb 2024; v3 5 Jul 2025. No venue in Comments.
- **URL:** https://arxiv.org/abs/2402.12374 | full text https://arxiv.org/html/2402.12374v3
- **Verified state:** VERIFIED.

**Verbatim (abstract):**
> "Finally, Sequoia introduces a hardware-aware tree optimizer that maximizes speculative performance
> by automatically selecting the token tree size and depth for a given hardware platform."

**Verbatim (§4.1 Hardware-Aware Optimization — what it actually optimises):**
> "Letting G(n,d) denote the expected number of tokens generated by verifying the Sequoia tree of
> size n and depth d (computed via dynamic programming), t(n) denote the (hardware-dependent) amount
> of time it takes the target model to verify n tokens divided by the time to verify 1 token, and c
> denote the (hardware-dependent) time to draft 1 token divided by the time to verify 1 token, the
> speedup attained by Sequoia can be expressed as Speedup(n,d) = G(n,d) / (t(n) + d·c)."

> "We measure t(n) and c empirically for each type of model and inference hardware, and then search
> over possible values of n, d to find the pair that gives the largest speedup."

**Verbatim (§4.1, the choice it makes):**
> "Sequoia selects much larger trees in the offloading setting (768 tokens) than in the on-device
> setting (64 to 128 tokens)."

**NEGATIVE RESULT (important):** the strings `attention`, `tree attention`, `mask`, `KV cache`,
`key-value` do **not** occur anywhere in the Sequoia HTML full text outside the reference list.
Verified by grep over the fetched text. The only `sibling` occurrence is inside the DP
correctness proof (Appendix), where `S_v` is "set of sibling indices" used to compute the expected
number of generated tokens — pure tree-topology/probability bookkeeping, not attention.

**KV-set structure exploited:** NONE. Sequoia's "hardware-aware" part is a **scalar throughput model
t(n)** — verify time as a function of token count `n` — plus a search over (size, depth). It does not
model, measure, or exploit KV access structure or query correlation at all. It assumes verify cost
is a function of `n` alone.

---

## 5. Hydra / Lookahead Decoding / REST

### 5a. Hydra — arXiv 2402.05109
- **Title:** Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding
- **Venue/Date:** Submitted 7 Feb 2024; v2 7 Oct 2024. No venue in Comments.
- **URL:** https://arxiv.org/abs/2402.05109 | full text https://arxiv.org/html/2402.05109v2
- **Verified state:** VERIFIED.

**Verbatim (§3, verification):**
> "We query the base model for these conditional probabilities in a single forward pass by packing
> all of the tree's tokens into a single input sequence, and manipulating the attention mask to
> ensure that each token can only attend to its parents in the tree."

**Verbatim (§5, the strongest counter-evidence in the corpus that verify KV access is a bottleneck):**
> "Speculative decoding techniques are typically evaluated in the batch-size-1 setting. When there
> is only a single sequence in the batch, decoding is extremely memory bandwidth-bound, and large
> numbers of FLOPs can be consumed in the verification step of speculative decoding 'for free'
> without significantly increasing the latency per decoding step."

> "However, at larger batch sizes it is easier for verification to become compute-bound, and the
> number of tokens verified per sequence per step must be more tightly controlled to avoid
> saturating the GPU's compute capacity and entering the regime where speculative decoding becomes
> unprofitable."

**KV-set structure exploited:** shared prefix only. "Packing all of the tree's tokens into a single
input sequence, and manipulating the attention mask" = densify-then-mask.

### 5b. Lookahead Decoding — arXiv 2402.02057
- **Title:** Break the Sequential Dependency of LLM Inference Using Lookahead Decoding
- **Venue/Date:** Submitted 3 Feb 2024. No venue in Comments.
- **URL:** https://arxiv.org/abs/2402.02057 | full text via ar5iv
  https://ar5iv.labs.arxiv.org/html/2402.02057 (arXiv HTML **not** available — see COULD NOT VERIFY)
- **Verified state:** VERIFIED (abs page + ar5iv full text).

**Verbatim (§3.3, the custom verification attention pattern):**
> "At execution, the lookahead and verification branches can be integrated into one decoding step to
> leverage parallel processing. This requires a designated attention mask, as shown in Fig. 2 (b).
> This attention mask is straightforwardly derived following the principle that each token is only
> visible to the tokens with a larger position index than itself ... The tokens in the lookahead
> branch are not visible to the tokens in the verification branch, and vice versa."

**Verbatim (§3.3 — earliest explicit statement that a causal-mask kernel is unsuitable for verification):**
> "FlashAttention can vastly accelerate the training and inference of LLMs by saving memory I/O on
> the slow memory hierarchy. It forces a causal mask (e.g., Fig. 2 (a)) to avoid all token
> interactions outside a lower triangular scope, which is not suitable for Lookahead Decoding as we
> take a more subtle attention mask (e.g., Fig. 2 (b)) for different W, N, and G. To solve this, we
> hardcode Lookahead Decoding's attention pattern with adjustable W, N, and G in FlashAttention.
> Applying FlashAttention to Lookahead Decoding brings about 20% end-to-end speedup compared to a
> straightforward implementation on top of native PyTorch in our experiments (§5.2)."

**Verbatim (§3.2 — explicit that Lookahead's verification queries are NOT tree-structured):**
> "Previous research (Miao et al. 2023) has developed efficient tree-based verification for
> speculative decoding with sampling support, where multiple draft sequences derived from a token
> tree can be verified in parallel. However, it does not apply to Lookahead Decoding as our
> verification works on disjoint n-grams instead of trees."

**KV-set structure exploited:** NONE by construction — verification candidates are *disjoint* n-grams
sharing only the committed prefix. Lookahead is nonetheless a genuine systems contribution to
verification attention: it is the only paper in this set that states FlashAttention's causal mask is
unsuitable and hardcodes a custom verification attention pattern inside FlashAttention.

### 5c. REST — arXiv 2311.08252
- **Title:** REST: Retrieval-Based Speculative Decoding
- **Venue/Date:** Submitted 14 Nov 2023; v2 4 Apr 2024. **Comments: NAACL 2024, camera ready** (verified).
- **URL:** https://arxiv.org/abs/2311.08252 | full text https://arxiv.org/html/2311.08252v2
- **Verified state:** VERIFIED.

**Verbatim (§3.3 Draft verification of REST):**
> "In REST, multiple draft sequences may be retrieved from the datastore. While one might initially
> approach the drafts independently and feed them into the LLM as distinct sequences in a batch,
> practical observations reveal that many drafts share common prefixes. This leads to redundant
> computation of Transformer layers on these shared prefixes across different sequences, resulting
> in a waste of computational power. To optimize the efficiency, we construct a pseudo sequence from
> the subtree using breadth-first search. By definition, it can be immediately obtained that each
> draft constitutes a sub-sequence of this pseudo sequence, and any shared prefix appears only once.
> To correctly execute LLM on this pseudo sequence, we implement a carefully designed attention mask
> in each attention layer, ensuring that the computation of each token precisely reflects its
> dependencies in the original draft sequence. This attention strategy is also known as tree
> attention."

**Verbatim (§1):**
> "This sequence then undergoes verification by the LLM through a single forward pass, aided by a
> meticulously designed attention mask known as tree attention."

**KV-set structure exploited:** shared prefix only, and only to the extent of emitting each shared
prefix once in the BFS pseudo-sequence. Explicitly framed as "any shared prefix appears only once",
not as per-node KV-set containment.

---

## 6. Nearest neighbour: arXiv 2605.19893 (SpecSA → SSV)

- **v1 title (VERIFIED):** SpecSA: Bridging Speculative Decoding and Sparse Attention for Efficient
  LLM Inference — https://arxiv.org/abs/2605.19893v1, submitted 19 May 2026.
- **v2 title (VERIFIED):** SSV: Sparse Speculative Verification for Efficient LLM Inference —
  https://arxiv.org/abs/2605.19893v2, last revised 20 May 2026.
- **Subjects (v1):** Operating Systems (cs.OS). Both abstracts read identically except for the
  system name; the paper was retitled and renamed in one day.
- **Full text:** https://arxiv.org/html/2605.19893v2
- **Verified state:** VERIFIED (both v1 and v2 abs pages, plus v2 full text).

**Verbatim (abstract):**
> "Speculative decoding and dynamic sparse attention are two complementary approaches for
> accelerating long-context LLM inference: the former amortizes target-model execution across
> multiple verifier queries, while the latter reduces each query's KV-cache working set. Directly
> combining them, however, exposes a structural mismatch: speculative verification relies on
> cross-query commonality, whereas dynamic sparse attention assigns query-specific sparse layouts."

**Verbatim (§3.1, the three insights):**
> "The per-query selectivity of dynamic sparse attention conflicts with the cross-query commonality
> of speculative verification."

> "(1) Cross-query reuse is hidden behind query-specific routing. We observe that nearby verifier
> queries often select overlapping KV blocks, but decoding-oriented sparse kernels process each query
> independently and reload these blocks, while prefill-oriented kernels may fetch blocks unused by
> any verifier query. A practical verifier must exploit this overlap without losing the option to
> preserve exact per-query selected-block semantics."

**Verbatim (§3.1, the measurement — this is the empirical core):**
> "Tree-based speculative verification processes multiple draft positions in the same target-model
> pass. Although dynamic sparse attention gives each query its own selected-block set, verifier
> queries that are close in position often have semantically similar contexts and therefore select
> many of the same KV blocks. Our profiling confirms this trend: as shown in Figure 2, under an 8K
> context, the selected-block overlap ratio between adjacent verifier queries typically remains
> around 50%–90% across most layers. This overlap suggests grouping positionally close queries for
> joint execution, while the kernel can either preserve each query's own layout or use a faster
> approximate shared layout."

**Verbatim (§1, contribution list):**
> "First, SSV introduces an overlap-aware kernel design (Section 4) that flattens draft trees with
> selected traversal orders and groups up to C adjacent verifier queries in one thread block. Each
> group can use an exact merged schedule that deduplicates overlapping selected blocks while
> preserving per-query NSA semantics, or an approximate shared-index layout that reuses one
> representative query's sparse layout for higher KV reuse."

**Verbatim (§4.1 Cross-Query Grouping — the two traversal orders):**
> "While grouping is trivial for a flat draft sequence, tree-based speculation introduces structural
> differences. Specifically, a tree node can be grouped either with its siblings (candidates for the
> same position that are likely synonymous) or with its parent and children (candidates across
> nearby positions with strong semantic proximity). In practice, SSV reorganizes the draft tree into
> a flattened batch using two traversal orders: breadth-first search (BFS) to enforce sibling-based
> grouping, and depth-first search (DFS) to enforce parent/child-based grouping."

> "Because the optimal grouping strategy depends heavily on the specific verification setting and
> realized query ordering, no single tree-local rule is universally superior. Therefore, rather than
> using a static heuristic, SSV delegates this choice to the throughput-aware planner (Section 6),
> which adaptively selects the best traversal order based on offline profiles and runtime states."

**Verbatim (§4.1, the exact merged-schedule variant — the decisive mechanism):**
> "(2) On-chip Merging: It concatenates these per-query indices, sorts them, and performs a linear
> scan to deduplicate identical entries. This creates a unified schedule containing the union of all
> selected blocks while tracking query-to-block ownership.
> (3) Shared Loading and Masking: Each unique KV block in the merged schedule is loaded from HBM
> exactly once and shared across the group. To preserve exact semantics, the kernel applies a
> row-wise mask during attention computation: logits are masked to −∞ if the key block does not
> belong to a specific query's original selected set."

> "This design effectively amortizes the HBM load costs of overlapping blocks while remaining
> semantically equivalent to independent query execution."

**Verbatim (§4.1, the approximate variant's justification):**
> "Prior work (Neelam et al., 2025) has shown that leveraging the same index for neighboring tokens
> barely affects model quality. Therefore, this variant aims to maximize reuse by forcing all C
> queries in the group to share the same selected block set."

**Verbatim (§5.2, the negative result on grouping gains):**
> "At a short draft length (γ=4), SSV without reuse layers yields modest 1.05×–1.07× speedups over
> vanilla NSA on both models. ... For larger draft lengths (γ≥64), grouped-query execution brings
> additional independent gains. For instance, on the 1B model, the exact variant (C=2) and
> approximate variant (C=4) reach up to 1.10× and 1.23× speedups, respectively. The 8B model follows
> the same trend, albeit with slightly more modest gains, up to 1.02× and 1.12×."

**Setup (verified from §5 / §6, matching the parent's description):** NVIDIA H100 PCIe GPU; batch
size 1; bfloat16; NSA config l=32, d=16, l'=64, n=16, sliding window 512, 32 query heads, 8 KV heads,
head dim 64; sequence lengths 16K/32K/64K; draft lengths γ ∈ {4, 64, 128}; 1B and 1.2B NSA target
models pretrained on Children-Stories-Collection; also NSA-adapted Llama3-1B and Llama3-8B; EAGLE-3
integration with 16K context, temperature 0.0.

**KV-set structure exploited:** **approximate measured overlap only (50–90% between adjacent
verifier queries), NOT exact nesting.** Both variants are union-then-mask: the exact variant builds
the *union* of selected blocks and re-masks per query (same densify-then-mask pattern as SpecInfer,
transposed to block granularity); the approximate variant forces a single shared index set and
accepts quality loss. SSV is the only work found that *measures* cross-query KV working-set overlap
in verification, and it explicitly reports that the exact schedule yields small gains (1.10× on 1B,
1.02× on 8B) while the aggressive approximation is what pays (1.23× / 1.12×).

---

## 7. Cascade Inference (FlashInfer) — non-arXiv, verified

- **Title:** Cascade Inference: Memory Bandwidth Efficient Shared Prefix Batch Decoding
- **Venue/Date:** FlashInfer blog post, 2024-02-02 (JSON-LD `datePublished` = 2024-02-02).
- **Authors (from page JSON-LD):** Zihao Ye (UW), Ruihang Lai (CMU), Bo-Ru Lu (UW), Chien-Yu Lin
  (UW), Size Zheng (UW & PKU), Lequn Chen (UW), Tianqi Chen (CMU & OctoAI), Luis Ceze (UW & OctoAI)
- **URL:** https://flashinfer.ai/2024/02/02/cascade-inference.html
- **Verified state:** VERIFIED (fetched).

**Verbatim:**
> "In this blog post, we introduce Cascade Inference, which simply decouples attention of shared
> prefix and unique suffixes, and enables storing shared KV-Cache in GPU shared memory (SMEM for
> short) for fast access in multiple requests."

> "Cascade Inference allow us to maximize memory reuse for common prefix, thus making the attention
> computation much more memory efficient."

> "The idea of Cascade Inference can be generalized to multiple levels (we only show two levels in
> this blog post) and multiple shared prefixes, the multi-level, multi shared-prefix Cascade
> Inference has been integrated to MLC-Serving"

> "Recently, SGLang (a domain-specific language for programming LLMs) proposes RadixAttention, where
> the KV-Cache is organized as a radix tree structure and the attention can be further accelerated
> with multiple-level Cascade Inference."

**Title/venue-level caveats:** the phrase "tree attention" does NOT appear in this blog post.

**On the origin of the term "tree attention":** the earliest usage I verified in this corpus is REST
(arXiv 2311.08252, Nov 2023), which writes "tree attention (Cai et al., 2023; Miao et al., 2023;
Spector and Re, 2023)". REST's reference-list entry, fetched verbatim, is:

> "Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, and Tri Dao. 2023. Medusa: Simple framework
> for accelerating llm generation with multiple decoding heads. https://github.com/FasterDecoding/Medusa ."

So REST cites the **2023 GitHub repository**, not the arXiv paper (2401.10774 is Jan 2024) — there is
no chronological error. Note also that Medusa's own paper (§2.1.2) cites SpecInfer (Miao et al. 2023)
and Spector & Re (2023) for tree attention while claiming the top-down tree construction as its own
distinction. The term is co-attributed across SpecInfer / Medusa-repo / Spector & Re; I did not find
a single verifiable origin.

**KV-set structure exploited:** shared prefix across *requests*, organised as a hierarchy of levels.
Because prefixes nest, the multi-level generalisation is structurally capable of expressing nested
KV-set containment — but it is stated for served requests, not for the verification queries of one
speculative step. RadixAttention (SGLang) is the closest existing structure: a radix tree over KV
blocks is exactly a nested-set structure.

---

## 8. Hydragen — arXiv 2402.05099 (the strongest prior art found)

**ID correction, important:** Hydragen is **arXiv 2402.05099**, NOT 2401.14351.
I verified 2401.14351 and it is an unrelated paper ("ServerlessLLM: Low-Latency Serverless
Inference for Large Language Models"). The ID 2402.05099 was taken from the fetched reference list of
the FlashInfer paper (arXiv 2501.01005v2) and then independently confirmed against
`arxiv.org/abs/2402.05099`.

- **Title:** Hydragen: High-Throughput LLM Inference with Shared Prefixes
- **Authors (from FlashInfer ref list):** Juravsky, Brown, Ehrlich, Fu, Ré, Mirhoseini
- **Venue/Date:** Submitted 7 Feb 2024; v2 13 May 2024. No venue in Comments.
- **URL:** https://arxiv.org/abs/2402.05099 | full text https://arxiv.org/html/2402.05099v2
- **Verified state:** VERIFIED.

**Verbatim (abstract):**
> "In this work, we introduce Hydragen, a hardware-aware exact implementation of attention with
> shared prefixes. Hydragen computes attention over the shared prefix and unique suffixes separately.
> This decomposition enables efficient prefix attention by batching queries together across
> sequences, reducing redundant memory reads and enabling the use of hardware-friendly matrix
> multiplications."

> "Hydragen generalizes beyond simple prefix-suffix decomposition and can be applied to tree-based
> prompt sharing patterns, allowing us to further reduce inference time on competitive programming
> problems by 55%."

**Verbatim (§1, the cost being attacked):**
> "We identify that FlashAttention and PagedAttention redundantly read the prefix's keys and values
> from GPU memory when computing attention, regardless of whether the prefix is redundantly stored.
> In order to eliminate these redundant reads, we present Hydragen, an exact implementation of
> attention that is specialized for shared prefixes."

**Verbatim (§2.4):**
> "if multiple sequences share a common prefix, the keys and values corresponding to the prefix
> tokens will be identical across sequences."

**Verbatim (§3.2 Inter-Sequence Batched Prefix Attention — the merge rule):**
> "Queries do not affect each other when computing attention, therefore if two sets of queries
> attend over identical keys and values, they can be merged into a single attention operation with a
> larger number of queries."

> "Note that we are unable to apply inter-sequence batching when computing attention over suffixes,
> since the keys and values in each sequence's suffix are not identical. Suffix attention is
> therefore computed normally, with a single query per sequence."

**Verbatim (§3.3 Hierarchical Sharing — THE key passage):**
> "Additionally, sharing may be more fine-grained than a simple prefix-suffix decomposition, with the
> overlap between sequences forming a tree structure where each node contains a token sequence that
> is shared by all descendants (see Figure 2 for an example). These forms of sharing are increasingly
> relevant as LLMs are applied in more complicated inference/search algorithms"

> "Hydragen naturally generalizes to these richer forms of sharing as well. To apply Hydragen to a
> tree of sequences, we replace attention decomposition over the prefix and suffix with attention
> decomposition at every vertex in the tree. We can then use inter-sequence batching across levels of
> the tree, so that the keys and values associated with one node in the tree are shared across the
> queries of all descendant nodes."

> Figure 2 caption: "An example of a batch of sequences with a hierarchical sharing pattern. This
> diagram depicts the setting of Section 4.4, which solves competitive programming problems using a
> few-shot prompt and by sampling many candidate solutions per problem. The few-shot prompt (orange)
> is globally shared across all sequences in the batch. However, the descriptions of each problem
> (green and blue) are only shared across the candidate solutions corresponding to that problem."

**Verbatim (§3.4, the regime limitation):**
> "In order for Hydragen to meaningfully improve decoding speed in a particular setting, attention
> must be a major contributor to decoding time. For example, with small batch sizes or short sequence
> lengths, decoding speed is often bottlenecked not by attention, but by reading the parameters of
> the model from GPU memory. The benefits of Hydragen in this scenario will therefore be minimal."

**KV-set structure exploited:** **EXACT nesting, at every tree vertex.** §3.3 is an explicit
statement that a node's KV is shared by all descendants and that a per-vertex attention
decomposition plus inter-level query batching is applied. The §3.2 merge rule ("if two sets of
queries attend over identical keys and values, they can be merged") is precisely the primitive that
speculative verification queries need.

**Why it is not the same ground:** Hydragen's tree is a tree of *independent sequences in a serving
batch* (candidate solutions to competitive-programming problems, Tree-of-Thoughts). It is a
throughput optimisation at large batch. It is NOT applied to the k verification queries of a single
speculative step — no speculative-decoding experiment appears in the paper. And §3.4 concedes the
benefit vanishes at small batch / short sequence, i.e. the canonical speculative-decoding regime.
The structural machinery exists and is published; the application to verification queries does not.

---

## 9. FlashInfer — arXiv 2501.01005 (tree attention as a sparse mask)

- **Title:** FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving
- **Venue/Date:** Submitted 2 Jan 2025; v2 21 Apr 2025. No venue in Comments field.
- **URL:** https://arxiv.org/abs/2501.01005 | full text https://arxiv.org/html/2501.01005v2
- **Verified state:** VERIFIED.

**Verbatim (§3.1.2 / §5.1, how tree attention is represented):**
> "Beyond page tables and radix trees, sparse matrices can also effectively represent various
> attention mechanisms, such as Tree Attentions used in speculative decoding (Cai et al., 2024;
> Miao et al., 2024; Chen et al., 2024) and importance masks applied to KV-Cache (Tang et al.,
> 2024)."

**Verbatim (§1, listing tree decoding as a workload driver):**
> "As multiple requests are processed, opportunities for prefix-reuse emerge, and the introduction of
> tree decoding in speculative scenarios creates additional attention patterns (Cai et al., 2024;
> Miao et al., 2024; Chen et al., 2024)."

**Verbatim (§3.1.2, composable formats — shared prefix handled separately from tree masks):**
> "Our composable format design allows for the decomposition of the KV cache sparse matrix based on
> prior knowledge. For instance, if certain requests share a prefix, the corresponding rows and
> columns in the KV cache form a dense submatrix. We can then use a block sparse matrix with a larger
> B_r to store these submatrices efficiently."

**Verbatim (§5.1, what FlashInfer adds over the shared-prefix line of work):**
> "Recent works like RelayAttention (Zhu et al., 2024b), Hydragen (Juravsky et al., 2024),
> ChunkAttention (Ye et al., 2024), and Parrot (Lin et al., 2024) explore shared prefix decoding
> attention but require separate KV-Cache management for prefixes and suffixes. In contrast,
> FlashInfer's composable formats support multi-level, multiple-prefix decoding with unified page
> table management, enabling seamless integration into LLM serving frameworks without modifying
> memory management modules."

**KV-set structure exploited:** tree attention is handled as a **sparse mask over a flat KV
buffer** — i.e. correctness by masking, which is the same densify-then-mask shape as SpecInfer, at
block granularity. Shared-prefix decomposition is handled by the *separate* composable-format
mechanism and is described for *requests* ("if certain requests share a prefix"), not for tree nodes
within one verification step. The paper places "Tree Attentions" and "importance masks applied to
KV-Cache" in the same category — masks, not KV-set structure. No claim that tree-node KV sets nest.

---

## 10. ChunkAttention — arXiv 2402.15220 (verified, not yet quoted)

- **Title:** ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition
- **Venue/Date:** Submitted 23 Feb 2024 (from abs page). No venue read.
- **URL:** https://arxiv.org/abs/2402.15220
- **Verified state:** ID + title VERIFIED by fetching the abs page. **Full-text claims NOT verified**
  in this reconnaissance (time-boxed). It appears in FlashInfer's §5.1 list of shared-prefix
  decoding attention works requiring separate prefix/suffix KV management. Treat as a lead.

---

## 11. arXiv 2512.21911 — "Accelerate Speculative Decoding with Sparse Computation in
Verification" (THE decisive prior art; predates SSV by ~5 months)

- **Title:** Accelerate Speculative Decoding with Sparse Computation in Verification
- **Venue/Date:** Submitted 26 Dec 2025. Comments field: "Pre-print". Subjects: cs.CL.
- **URL:** https://arxiv.org/abs/2512.21911 | full text https://arxiv.org/html/2512.21911v1
- **Verified state:** VERIFIED (abs page title + full text re-fetched by me).
- **Note:** this paper was found by a delegated breadth search, then **independently re-verified by
  me** against the abs page and a fresh full-text fetch. Quotes below are from my own fetch.

**Verbatim (abstract):**
> "Speculative decoding accelerates autoregressive language model inference by verifying multiple
> draft tokens in parallel. However, the verification stage often becomes the dominant computational
> bottleneck, especially for long-context inputs and mixture-of-experts (MoE) models."

> "This work systematically adopts different sparse methods on the verification stage of the
> speculative decoding and identifies structured redundancy across multiple dimensions. Based on
> these observations, we propose a sparse verification framework that jointly sparsifies attention,
> FFN, and MoE components during the verification stage to reduce the dominant computation cost.
> The framework further incorporates an inter-draft token and inter-layer retrieval reuse strategy
> to further reduce redundant computation without introducing additional training."

**Verbatim (§2.2, heading "KV Cache Eviction in Sparse Verification" — THE measurement and the
mechanism):**
> "For speculative decoding, each step of the decoding process involves multiple tokens as input.
> Therefore, it is necessary to analyze the retrieval differences of these tokens to better adapt to
> these strategies. To this end, we perform speculative decoding on Llama3.1-8B and analyze the
> overlap of cache blocks retrieved by different draft tokens based on their respective query values."

> "In the experiments, a tree-structured draft is adopted, with 60 draft tokens in total. We measure
> the overlap ratio of retrieved blocks between token pairs at different positional distances. The
> overlap is calculated as the intersection of the retrieved blocks for the two tokens divided by the
> total number of blocks."

> "The results show that tokens within the same step retrieve highly similar blocks, with an average
> overlap exceeding 0.8 in most layers. Moreover, tokens that are closer in position tend to retrieve
> more similar blocks. On the other hand, in the draft tree, the first token is the accepted token
> sampled from the previous step, making it more important than the others since it influences the
> verification of all subsequent tokens. Taking into account both aspects, we use the first token to
> perform block retrieval, while the other tokens reuse its retrieved blocks during verification."

**Verbatim (§3.1):**
> "Existing sparse attention methods primarily focus on the decoding stage by applying KV cache
> eviction to reduce memory and computation overhead. However, such strategies are designed for
> standard autoregressive decoding, where each step processes only a single token. In speculative
> decoding, by contrast, each step must verify multiple candidate tokens simultaneously, often
> involving several to dozens of tokens. This multi-token verification introduces new challenges for
> efficiently selecting relevant context while maintaining verification accuracy."

**Verbatim (§3.3 Inter-layer Retrieval Reuse):**
> "Through empirical analysis of the retrieved blocks across transformer layers, we observe that
> although the retrieval patterns in shallow layers tend to be irregular and diverse, the block
> selection in middle and deeper layers exhibits strong inter-layer similarity, particularly among
> adjacent layers. This observation suggests that performing retrieval at all layers is redundant and
> leads to unnecessary computational overhead."

**KV-set structure exploited:** **cross-query KV-block overlap, measured and then exploited
directly** — but by *approximation*: one representative (first) draft token's retrieved block set is
reused for all other draft tokens in the step. This is the same idea SSV (2605.19893) later calls the
"approximate shared-index variant". **2512.21911 does NOT implement an exact union/merged schedule
with per-query masking**, which is the one piece of SSV that appears to remain novel.

**Consequence for the research-ground claim:** the observation "verification queries share KV
structure" and the exploitation of it were **published on 26 Dec 2025**, five months before SSV
(19 May 2026). SSV is a later, NSA-specialised refinement, not the first mover. The overlap figure
reported here (>0.8) and SSV's (50–90%) are consistent.

---

## 12. arXiv 2602.07223 — SpecAttn → Vegas (verification as a *source* of KV selection, ICML'26)

- **v1 title (VERIFIED):** SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding —
  https://arxiv.org/abs/2602.07223v1, submitted 6 Feb 2026.
- **v2 title (VERIFIED):** Vegas: Self-Speculative Decoding with Verification-Guided Sparse
  Attention — https://arxiv.org/abs/2602.07223v2, 29 May 2026. **Comments: Accepted to ICML'26**
  (verified from the v2 abs page).
- **Full text:** https://arxiv.org/html/2602.07223v1 (v1 HTML, which carries the SpecAttn title)
- **Verified state:** VERIFIED (both abs pages, v2 Comments field, v1 full text).

**Verbatim (v1 abstract):**
> "Previous works have shown that self-speculative decoding with sparse attention, where tokens are
> drafted using a subset of the KV cache and verified in parallel with full KV cache, speeds up
> inference in a lossless way. However, this approach relies on standalone KV selection algorithms to
> select the KV entries used for drafting and overlooks that the criticality of each KV entry is
> inherently computed during verification."

> "SpecAttn identifies critical KV entries as a byproduct of verification and only loads these
> entries when drafting subsequent tokens. This not only improves draft token acceptance rate but
> also incurs low KV selection overhead, thereby improving decoding throughput."

**Verbatim (v1 §2.3 / §1):**
> "Since the full KV cache is accessed only once in the verification stage, the overhead can be
> amortized across multiple accepted draft tokens to achieve a significant speedup."

**KV-set structure exploited:** NONE in the cross-query sense — this is the **inverse** direction.
Verification is deliberately kept **full-KV** and is used as the *oracle* that selects which KV
entries the drafting phase may load. It is strong counter-evidence: a 2026 ICML-accepted paper
treats the verification pass's full-KV access as an *amortised, desirable* property ("the full KV
cache is accessed only once in the verification stage, the overhead can be amortized"), not as a
cost to be sparsified.

**Retitle pattern worth noting:** both 2605.19893 (SpecSA→SSV, 19→20 May 2026) and 2602.07223
(SpecAttn→Vegas, by 29 May 2026) were renamed within 2026. The two v1 names "SpecSA" and "SpecAttn"
are near-collisions in the same niche.

---

## 13. MagicDec — arXiv 2408.11049 (the regime argument for verification KV cost)

- **Title:** MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with
  Speculative Decoding
- **Venue/Date:** Submitted 20 Aug 2024; v5 2 Apr 2025 (v5 full text fetched). No venue in Comments.
- **URL:** https://arxiv.org/abs/2408.11049 | full text https://arxiv.org/html/2408.11049v5
- **Verified state:** VERIFIED (abs page title + v5 full text).

**Verbatim (abstract):**
> "Speculative decoding (SD) is a widely used technique to reduce latency losslessly, but the
> conventional wisdom suggests that its efficacy is limited to small batch sizes. In MagicDec, we
> show that surprisingly SD can achieve speedup even for a high throughput inference regime for
> moderate to long sequences. ... We leverage draft model with sparse KV cache to address the KV
> bottleneck, which scales with both sequence length and batch size."

**Verbatim (§3, Insight 1):**
> "KV Cache Is The Dominant Bottleneck In Large batch size Long-context Regime: In long-context and
> large batch size regime, the KV cache outgrows the memory footprint of the model parameter and
> continues to increase with batch size."

**Verbatim (§3, Insight 2 — the amortisation claim, and the regime boundary):**
> "SD Can Improve Throughput Only Beyond a Critical Sequence Length: While existing research
> suggests that SD is inefficient for large batches due to high verification costs, this limitation
> only applies to very short sequences. Because with short sequences, increasing the batch size makes
> computational costs the primary bottleneck, which is prohibitive for an efficient verification
> process. However, once sequences exceed a certain critical length (which varies based on the model
> and hardware), the KV loading cost becomes the dominant factor, even for large batches. At this
> point, SD becomes effective again because the computational overhead of verification becomes less
> significant compared to the KV loading costs, which can be amortized across the tokens to be
> verified."

**Verbatim (§1):**
> "For small batches, the main performance bottleneck is the parameter loading cost, which can be
> amortized by the verification process across the tokens to be verified at the expense of increased
> computation. However, with large batches, LLMs become compute bound, making verification
> significantly costly because of its compute-hungry nature."

**KV-set structure exploited:** the *committed/shared* KV is amortised across verified tokens — i.e.
the shared-prefix amortisation, quantified against a critical sequence length. **MagicDec does not
exploit per-query KV-set nesting**: it sparsifies the *draft* model's KV cache, not the target
verification pass. But it supplies the regime argument that makes verification KV access matter at
all: at long context the KV loading cost dominates and is amortised across verified tokens; at short
sequence it does not and verification is compute-bound.

**This is the pivot of the whole baseline.** Hydra (Feb 2024) says verification FLOPs are "free" at
bs=1 because decoding is memory-bandwidth-bound. MagicDec (Aug 2024) says that reasoning only holds
below a critical sequence length; above it, KV loading dominates and SD becomes worthwhile again.
SSV (May 2026) and 2512.21911 (Dec 2025) both operate in the long-context regime (16K–64K) where
KV access is the thing being optimised.

---

## 14. DeFT — arXiv 2404.00242 (the sharpest negative evidence) — independently verified by me

- **Title:** DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference
- **Venue/Date:** Submitted 30 Mar 2024; v4 7 Mar 2025. **Comments: "Update DeFT-v4, accepted by
  ICLR'25"** (verified from the abs page Comments field).
- **URL:** https://arxiv.org/abs/2404.00242 | full text https://arxiv.org/html/2404.00242v4
- **Verified state:** VERIFIED by me (abs page Comments + fresh full-text fetch + grep of each quote).

**Verbatim (§Related Work, on SpecInfer — the decisive sentence):**
> "In SpecInfer (Miao et al., 2023), as shown in Figure 6b, a bit mask is utilized to record the
> causal information among queries of a token tree. Each token t_i in queries will have a 64-bit Int
> as a bit mask, where j-th bit means the causal relationship between query of t_i and KV cache of
> t_j. The advantage of this mask design is that it greatly reduces IO, but it results in the maximum
> number of tree tokens being only 64, which is not practical for scenarios with tree-structured KV
> cache. What's more, **it is not IO-aware for KV cache as it will load KV cache of the entire tree
> for each query.**"

**Verbatim (§3, DeFT's own mechanism — the closest any work comes to nesting):**
> "We resort to alternative KV-Guided Grouping approach: **by grouping each node's KV cache with all
> the queries that share it, the partitioning can be made prefix-aware, therefore reducing memory
> access to the KV cache.**"

**Verbatim (§2 / §3, on the baseline):**
> "This method calculates attention for all queries simultaneously in a single Streaming Multiprocessor
> (SM), with the aid of a dense causal mask (DCM)."

> "Medusa (Cai et al., 2024) materializes the dense causal mask (DCM) in HBM to record the causal
> information between n_q tokens in queries and n_kv tokens in the KV cache, thereby introducing a
> significant IO cost for loading this n_q × n_kv-sized mask to shared memory."

**KV-set structure exploited:** DeFT is the **closest existing work to exploiting set nesting**. The
set of queries that share node v's KV is exactly subtree(v), and those query sets nest along every
root→leaf path. DeFT groups node KV with the queries that share it, so each node's KV is loaded once
— i.e. it *does* exploit the containment structure to deduplicate **loads**. But it stops short of
nesting-for-sparsity: it still computes attention output for every node and its own text never
frames the sets as nested (a delegated grep for nest/subset/superset/ancestor/descend over the full
text returned no semantic hits). And its own statement of the state of the art is that prior tree
attention "will load KV cache of the entire tree for each query."

---

## 15. Tree-attention as a systems/kernel contribution — summary (full detail in `kernels.md`)

Full evidence file: `/Users/leihenan/Desktop/myProject/recon/kernels.md`.
Venues verified by me from abs-page Comments fields: FlashInfer = "Accepted by MLSys 2025";
DeFT = "accepted by ICLR'25"; PAT (2511.22333) = ASPLOS'26 per the delegated report (venue field not
re-verified by me).

**Verified headline:** no source found — FlashInfer, xgrammar, vLLM, SGLang, or any paper — claims
tree attention can **skip** attention computation or KV reads for nodes whose KV sets are subsets of
other nodes. The universal mechanism is a **dense (optionally bitpacked) custom/topology-aware mask**
over a **single shared prefix KV list**. What is exploited is **KV-load deduplication** across
queries (IO reuse), not arithmetic skipping.

Selected verified artifacts (all fetched; see `kernels.md` for URLs and quotes):

- **FlashInfer** (2501.01005, MLSys 2025): no `tree_attn` symbol is exported by `flashinfer/__init__.py`;
  the paper represents tree attention as a sparse matrix/mask. Decisive: the merge operator
  `s(I∪J) = log(exp s(I) + exp s(J))` is valid only for **disjoint** index sets, so nested sets would
  be double-counted — the cascade machinery is structurally a **union over a partition**, not a
  nesting-aware path.
- **xgrammar**: its "adaptive/compressed mask" is a **vocabulary/logits bitmask**
  (`include/xgrammar/matcher.h`, `TraverseDraftTree`: "fills the token bitmask for each node based on
  grammar constraints"). It does contain genuine set-nesting — but for the **automaton stack**, not KV.
- **SGLang**: `python/sglang/srt/speculative/eagle_info.py` allocates
  `mask_numel = (paged_kernel_lens_sum * self.draft_token_num + (self.draft_token_num**2) * batch_size)`
  — a **dense** prefix×draft mask — and passes a **single `kv_indices` array per sequence** shared
  across all top-k branches. There is no per-tree-node KV list, so node KV-set nesting is not even
  expressible in the current data layout.
- **vLLM**: no tree-attn backend (`vllm/v1/attention/backends/tree_attn.py` and
  `vllm/v1/spec_decode/tree_utils.py` both HTTP 404). Issue #18327 ("[Feature]: Tree-Attention Support
  for Speculative Decoding", opened 2025-05-19) was **closed as NOT_PLANNED**; PR #17560
  ("[WIP][V1][Spec Decode] EAGLE tree-attention") was **closed unmerged** with
  "- [ ] Attention metadata & attention mask" and "- [ ] KV cache & CUDA graph (future PRs)" unchecked.
- **Papers**: SpecInfer 2305.09781; DeFT 2404.00242 (ICLR'25); CoDec 2505.17694 — **renamed**: v1 was
  "FlashForge: Ultra-Efficient Prefix-Aware Attention for LLM Decoding", same ID, no separate ID;
  PAT 2511.22333 (ASPLOS'26); Bole 2608.01651 (hybrid-attention tree speculation — "materialize a full
  state for every proposal node"); SpecSA/SSV 2605.19893.

**Excluded as non-evidence:** a text titled "Branch-Shared KV Fragments: Memory-Efficient Key-Value
Cache Sharing for Branchy Decoding", hosted in a raw GitHub corpus with **no venue and no arXiv ID**,
self-declaring "This draft was AI-generated from automated research artifacts... No human reviewer has
endorsed its claims", CPU/NumPy only, sub-unity result stated by its own author as "within measurement
noise". It is not citable. Recorded here only so it is not mistaken for prior art later.

**Additional systems artifact I verified directly (POST-dates the kernels.md sweep):**

ONNX Runtime PR #32340, "[CUDA] Add speculative decoding to paged XQA", author `tianleiwu`,
https://github.com/microsoft/onnxruntime/pull/32340 — page JSON `"state":"MERGED"` (VERIFIED, fetched).
Verbatim from the PR body:
> "Add paged XQA kernels for multi-token speculative verification at head size 256 and group size 6,
> covering INT8, FP8 and native FP16/BF16 paged KV caches."

> "Generate the linear causal mask from `cumulative_seqlens_q` on device, including ragged batches and
> zero-query requests, without an operator schema change."

and a documented GPU-compiler bug in the mask builder:
> "So `clamped == 32` was true for every value, every mask word became `0xffffffff`, and there was
> effectively no intra-block causal mask — each draft token could attend to its own future. The
> emitted PTX is correct (`setp.eq.s32 %p2, %r8, 32`); only the SASS is wrong."

**KV-set structure exploited:** none. This is a **linear (chain) causal mask** for multi-token
verification, generated on device, not a tree mask and not a nesting-aware KV access. It is included
because it is a current, merged, production kernel for speculative verification and it corroborates
the mask-centric pattern — plus it is direct evidence that correct verification-mask construction is
still error-prone in shipping code.

