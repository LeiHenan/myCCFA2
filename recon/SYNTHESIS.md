# Synthesis — verification-aware attention / KV access for speculative decoding

Date: 2026-09-14. Primary-source detail in `PRIMARY.md`; kernel/systems detail in `kernels.md`.
Every quote is verbatim from a page I or a delegated agent fetched, with IDs verified by reading the
title back off `arxiv.org/abs/<id>`.

---

## Q1. What is the strongest existing claim that verification queries share KV structure?

**The strongest *structural* claim (exact nesting): Hydragen, arXiv 2402.05099, §3.3
"Hierarchical Sharing"** — but it is made about serving batches, not about verification.

> "the overlap between sequences forming a tree structure where each node contains a token sequence
> that is shared by all descendants"

> "To apply Hydragen to a tree of sequences, we replace attention decomposition over the prefix and
> suffix with attention decomposition at every vertex in the tree. We can then use inter-sequence
> batching across levels of the tree, so that the keys and values associated with one node in the
> tree are shared across the queries of all descendant nodes."

Backed by the merge primitive in §3.2:
> "Queries do not affect each other when computing attention, therefore if two sets of queries attend
> over identical keys and values, they can be merged into a single attention operation with a larger
> number of queries."

This is an *exact* statement of KV-set containment (node v's KV is in every descendant's KV set)
plus an exact algorithm (per-vertex decomposition + inter-level query batching). It is the machinery
a verification tree needs. It is not applied to speculative verification, and §3.4 concedes the
benefit requires an attention-bound regime:
> "with small batch sizes or short sequence lengths, decoding speed is often bottlenecked not by
> attention, but by reading the parameters of the model from GPU memory. The benefits of Hydragen in
> this scenario will therefore be minimal."

**The strongest *empirical* claim about verification queries specifically: arXiv 2512.21911
(26 Dec 2025), §2.2** — this is the decisive one, and it predates SSV by ~5 months.

> "For speculative decoding, each step of the decoding process involves multiple tokens as input.
> Therefore, it is necessary to analyze the retrieval differences of these tokens to better adapt to
> these strategies."

> "In the experiments, a tree-structured draft is adopted, with 60 draft tokens in total. ... The
> results show that tokens within the same step retrieve highly similar blocks, with an average
> overlap exceeding 0.8 in most layers. Moreover, tokens that are closer in position tend to retrieve
> more similar blocks. ... we use the first token to perform block retrieval, while the other tokens
> reuse its retrieved blocks during verification."

That is: measured cross-query KV-block overlap **>0.8 in most layers**, inside the verification step,
plus an implemented exploitation of it.

**The strongest *IO-dedup* claim inside tree attention: DeFT, arXiv 2404.00242 (ICLR'25), §3:**
> "by grouping each node's KV cache with all the queries that share it, the partitioning can be made
> prefix-aware, therefore reducing memory access to the KV cache."

The query set sharing node v's KV is exactly subtree(v); those sets nest along root→leaf paths. DeFT
exploits this for **load deduplication** while still computing attention for every node.

**Corroborating claim inside the sparse-attention route: SSV, arXiv 2605.19893v2 (20 May 2026), §3.1:**
> "under an 8K context, the selected-block overlap ratio between adjacent verifier queries typically
> remains around 50%–90% across most layers."

**Consistency check:** 2512.21911 measures >0.8 overlap (Llama3.1-8B, 60-token tree, dense-attention
KV-block retrieval); SSV measures 50–90% (NSA block selection, 8K context). The two independent
measurements agree. The phenomenon is real and reproduced.

---

## Q2. What is the strongest counter-evidence that this is already fully exploited?

**Counter-evidence 1 — nobody skips work; the field's own statement of the art is densify-then-mask.**
DeFT, on the immediate prior art (arXiv 2404.00242v4, verified):
> "it is not IO-aware for KV cache as it will load KV cache of the entire tree for each query."

SpecInfer's own description of its fused kernel (arXiv 2305.09781v4, verified) says the same:
> "results in attention scores that violate the causal dependency (e.g., t7's attention computation
> uses all previous tokens, including t5 which is not in t7's token sequence). To fix the attention
> scores for these pairs, SpecInfer updates the causal mask based on the token tree's topology."

Corroborated at kernel level: SGLang allocates a **dense** prefix×draft mask and passes a **single
`kv_indices` array per sequence**, so per-tree-node KV sets are not expressible in the current data
layout; FlashInfer's merge operator `s(I∪J) = log(exp s(I) + exp s(J))` is valid only for **disjoint**
index sets, so the existing cascade primitive cannot represent nested sets without double-counting;
vLLM's tree-attention PR was closed unmerged with the attention-mask and KV-cache items unchecked.

**Counter-evidence 2 — where the exact structure *has* been exploited in verification, the payoff was
small; the payoff came from approximation.** SSV's own ablation (§5.2, verified):
> "At a short draft length (γ=4), SSV without reuse layers yields modest 1.05×–1.07× speedups over
> vanilla NSA on both models. ... on the 1B model, the exact variant (C=2) and approximate variant
> (C=4) reach up to 1.10× and 1.23× speedups, respectively. ... The 8B model follows the same trend,
> albeit with slightly more modest gains, up to 1.02× and 1.12×."

The **exact** merged schedule — the piece that actually preserves per-query selected-block semantics —
is worth **1.02×–1.10×**. The larger gain (1.12×–1.23×) comes from the **approximate** shared-index
variant, which "trading exact selection semantics for lower scheduling overhead" and which SSV itself
restricts to "regimes with sufficient concurrent queries". SSV's bigger reported gains come from the
orthogonal refresh/reuse schedule (1.24×–1.28×), not from cross-query grouping. So the *exact*
structural reuse is nearly exhausted in that setup.

**Counter-evidence 3 — 2512.21911, the earlier paper, also declined the exact route.** Its mechanism
is the approximate one ("use the first token to perform block retrieval, while the other tokens reuse
its retrieved blocks"), i.e. the same approximation SSV later calls C=4. Neither paper built an exact
per-query nested-set schedule before SSV; and SSV's exact variant is its weakest component.

**Counter-evidence 4 — a 2026 ICML-accepted paper treats full-KV verification as an asset, not a cost.**
Vegas / SpecAttn (arXiv 2602.07223v2, "Accepted to ICML'26"; v1 titled SpecAttn), verified:
> "Since the full KV cache is accessed only once in the verification stage, the overhead can be
> amortized across multiple accepted draft tokens to achieve a significant speedup."

> "SpecAttn identifies critical KV entries as a byproduct of verification and only loads these entries
> when drafting subsequent tokens."

The verification pass is deliberately left **full-KV** and used as the selection oracle.

**Counter-evidence 5 — the regime is contested, so "verification KV access is a bottleneck" is not
universally true.** Hydra (arXiv 2402.05109v2, verified):
> "When there is only a single sequence in the batch, decoding is extremely memory bandwidth-bound,
> and large numbers of FLOPs can be consumed in the verification step of speculative decoding 'for
> free' without significantly increasing the latency per decoding step."

MagicDec (arXiv 2408.11049v5, verified) supplies the counter-boundary:
> "once sequences exceed a certain critical length (which varies based on the model and hardware), the
> KV loading cost becomes the dominant factor, even for large batches. At this point, SD becomes
> effective again because the computational overhead of verification becomes less significant compared
> to the KV loading costs, which can be amortized across the tokens to be verified."

So: at short context, verification attention is cheap and the whole premise is weak; at long context
(SSV: 16K–64K; 2512.21911 targets long context) it dominates. Both sparse-verification papers sit
exclusively in the long-context regime.

**Counter-evidence 6 — the two closest works do not cite each other.** SSV (2605.19893v2) contains no
occurrence of "2512.21911", of the title "Accelerate Speculative Decoding with Sparse Computation in
Verification", or of any of that paper's distinctive author names (Jikai Wang, Jianchao Tan, Yuxuan
Hu, Juntao Li, Min Zhang) — verified by grep over the full fetched text. Overlapping claims are
therefore being made in parallel without a settled prior-art baseline, which cuts both ways: the
ground is not securely occupied, but it is actively contested.

---

## Bottom line (no recommendations, as requested)

- The **observation** that verification queries share KV structure is **published**: measured at >0.8
  block overlap (2512.21911, 26 Dec 2025) and 50–90% (SSV, 20 May 2026).
- The **approximate exploitation** of it in verification is **published**: 2512.21911 (Dec 2025,
  reuse-the-first-token's blocks) and SSV (May 2026, shared-index C=4). SSV additionally contributes
  the **exact union-merge schedule** (C=2), which appears to be the one genuinely new mechanism.
- The **exact nesting-aware** treatment exists and is published, but **outside verification**:
  Hydragen §3.3 (hierarchical tree sharing, exact, Feb 2024) and DeFT §3 (KV-Guided Grouping, ICLR'25).
- **No source** found claims that a node whose KV set is a subset of another's can **skip** attention
  computation or KV reads. Universal mechanism: dense (optionally bitpacked) topology-aware mask over
  a single shared KV list, plus load deduplication.
- The **strongest counter-evidence to a large remaining prize** is SSV's own ablation: the exact
  schedule yields 1.02×–1.10× while the approximation yields 1.12×–1.23×, and the paper's headline
  gains come from an orthogonal reuse schedule. The strongest counter-evidence *for* remaining ground
  is DeFT's "load KV cache of the entire tree for each query" plus FlashInfer's disjoint-only merge
  operator, i.e. the kernel primitives for exact nested-set attention do not exist yet.

---

## Quote- fidelity caveat

Quotes were taken from fetched HTML stripped of tags (`sed -e 's/<[^>]*>/ /g'`). Where a passage
contains inline math, arXiv's HTML interleaves MathML and its LaTeX annotation, so the stripped text
renders e.g. `t 7 t_{7} ’s attention computation` for what the paper typesets as `t7's attention
computation`. Quotes containing math are therefore **normalised to the rendered reading**; no words
were added, removed, or reordered. Every non-math quote was verified byte-exact by grep against the
saved stripped text (Medusa, SpecInfer, EAGLE 1/2/3, Sequoia, Hydra, Lookahead, REST, Hydragen, DeFT,
SSV, 2512.21911).

---

## What I could NOT verify (with exact queries / URLs used)

### A. Retrieval infrastructure that was blocked (so no systematic keyword sweep was possible)

| Target | Result |
|---|---|
| `https://arxiv.org/search/?searchtype=all&query=tree+attention+kernel+speculative+decoding` | **HTTP 406, size 0** |
| `https://arxiv.org/search/?searchtype=all&query=verification-aware+attention+speculative+decoding` | **HTTP 406, size 0** |
| `https://arxiv.org/search/?query=sparse+speculative+verification&searchtype=all` | **HTTP 406, size 0** |
| `https://arxiv.org/search/?searchtype=all&query=tree+attention+speculative+decoding` (with `--compressed`, full browser `Accept`/`Accept-Language`, Chrome/126 UA) | **HTTP 406, size 0** — headers did not help |
| `http://export.arxiv.org/api/query?search_query=all:%22tree+attention%22+AND+all:%22speculative+decoding%22&max_results=25` | **HTTP 429, body exactly `Rate exceeded.`** |
| `https://api.semanticscholar.org/graph/v1/paper/search?...` | **HTTP 429** (per delegated agent) |
| `https://html.duckduckgo.com/html/?q=%22tree+attention%22+...` | HTTP 202, bot CAPTCHA page (per delegated agent) |
| `https://api.github.com/...` | rate-limited to zero (per the task brief) |
| `https://github.com/search?q=repo%3Aflashinfer-ai%2Fflashinfer+tree_attn&type=code` | HTTP 200 but "Sign in to search code on GitHub"; `result_count: 0` |

**Consequence:** I could not run a systematic arXiv keyword sweep. Discovery used `web_search`
strictly as a **lead-finder** and by **citation-chasing** through fetched bibliographies; every ID
and title in this report was then confirmed by fetching `arxiv.org/abs/<id>` and reading the title
back. Note that the search index is *stale*, not necessarily fabricated: it returned
"SpecSA" for 2605.19893 (the true v1 title) and "FlashForge" for 2505.17694 (the true v1 title).
Both are correct-as-of-v1 and would be wrong as current titles.

### B. Kernel sources not located (claims rest on indirect evidence)

- **SGLang `tree_attn` CUDA kernel**: not found at any probed path —
  `sgl-kernel/csrc/attention/tree_attn.cu`, `sgl-kernel/csrc/speculative/tree_attn.cu`,
  `sgl-kernel/csrc/speculative/eagle_utils.cu`, `sgl-kernel/csrc/eagle/tree_attn.cu`,
  `python/sglang/srt/layers/attention/tree_attn.py`, `python/sglang/srt/speculative/eagle_worker.py`,
  `github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc` — all **HTTP 404**.
  Its mask/KV behaviour is therefore inferred **only** from the Python call sites
  (`TreeMaskMode`, the `mask_numel` allocation, the single `kv_indices` array), which are all
  consistent with a dense mask but are not the kernel itself.
- **FlashInfer `tree_attn`**: absence is inferred from `flashinfer/__init__.py` exports plus the
  paper's "sparse matrices ... represent ... Tree Attentions" statement — **not** from a code search,
  which is unavailable anonymously.
- **vLLM**: `vllm/v1/attention/backends/tree_attn.py`, `vllm/v1/spec_decode/tree_utils.py`,
  `vllm/v1/worker/gpu/spec_decode/eagle.py` — all **HTTP 404**.
- **xgrammar docs**: `docs/api/python.rst`, `docs/advanced.rst`, `docs/key_concepts.rst`,
  `docs/technical_report.md` — all **HTTP 404** (used `include/xgrammar/matcher.h` and
  arXiv 2411.15100 instead).

### C. IDs and venues I corrected or could not confirm

- **Hydragen is NOT arXiv 2401.14351.** I verified 2401.14351 and it is "ServerlessLLM: Low-Latency
  Serverless Inference for Large Language Models". Hydragen = **2402.05099** (verified). The wrong ID
  was supplied in the task brief; do not propagate it. `arxiv.org/html/2402.05099v3` → **HTTP 404**
  (v3 does not exist); v1 and v2 both work.
- **Hydragen is not "Tri Dao et al."** Verified author list: Juravsky, Brown, Ehrlich, Fu, Ré,
  Mirhoseini.
- **Lookahead Decoding has no arXiv HTML.** `arxiv.org/html/2402.02057v2` returned 7,715 bytes of
  MathJax boilerplate with no paper content. I used `https://ar5iv.labs.arxiv.org/html/2402.02057`
  (404,716 bytes) instead. **Caveat:** ar5iv is a derived rendering, not arXiv's own HTML, so the
  Lookahead quotes rest on a derived source.
- **Venues NOT stated on the arXiv abs page, therefore NOT verified by me:** Sequoia 2402.12374,
  Lookahead 2402.02057, EAGLE 2401.15077, EAGLE-2 2406.16858, EAGLE-3 2503.01840, Hydra 2402.05109,
  Medusa 2401.10774 (Comments field contains only a code URL), Hydragen 2402.05099,
  ChunkAttention 2402.15220, PAT 2511.22333, SSV/SpecSA 2605.19893 (Subjects: cs.OS; no Comments).
- **Venues verified from the abs-page Comments field:** SpecInfer = ASPLOS'24; REST = NAACL 2024;
  DeFT = "accepted by ICLR'25"; FlashInfer = "Accepted by MLSys 2025";
  Vegas/SpecAttn 2602.07223v2 = "Accepted to ICML'26".
- **PAT's ASPLOS'26 venue** appeared only in the delegated agent's report; the abs page showed no
  Comments field, so I did **not** confirm it.
- **RelayAttention and Parrot: arXiv IDs NOT verified.** I have only their venues from FlashInfer's
  *fetched* bibliography: RelayAttention = ACL 2024 Long Papers, pp. 4945–4957 (Zhu, Wang, Zhang,
  Lau), doi `10.18653/V1/2024.ACL-LONG.270`; Parrot = OSDI'24, pp. 929–945, USENIX
  (`https://www.usenix.org/conference/osdi24/presentation/lin-chaofan`). Their own pages were never
  fetched, so any claim about their attention/KV mechanism is **unverified**.
- **ChunkAttention 2402.15220** — ID and title verified via the abs page; **full-text claims not
  read**. It appears in FlashInfer §5.1's shared-prefix list. Treat as a lead.
- **"Efficient Speculative Decoding for Llama at Scale" 2508.08192** — title and abstract only;
  full text not fetched, so its tree-attention statements are unverified beyond the abstract.
- **Gumiho (ICML 2025)** — surfaced as a lead; arXiv ID **never verified**; deliberately excluded
  from evidence.
- **Sprinter 2502.04557** — ID/title/date verified (v3, 8 Jul 2025). Its relevance is limited: it
  replaces target-model verification with a learned approximate verifier and makes **no**
  attention/KV-access claim. Included only as an adjacent "avoid verification cost" line of work.
- **MagicDec 2408.11049** — ID/title/date verified; quotes taken from v5 HTML.

### D. Coverage limits I did not close

- **EAGLE-3 and Medusa full texts were grepped, not read end-to-end.** I found no explicit
  attention/KV *cost measurement* in EAGLE-3 and no KV-access quantification in Medusa beyond its
  "compute-bound" remark, but I cannot claim exhaustively that none exists in their appendices.
- **Sequoia v3 HTML contains no occurrence of `attention`, `mask`, `KV cache`, or `key-value`
  outside its reference list** — verified by grep. If such material exists in a version or appendix
  I did not fetch, I did not see it.
- **SSV's non-citation of 2512.21911** was established by grepping the fetched v2 full text for the
  ID, the exact title, and five distinctive author names (Jikai, Jianchao, Yuxuan, Juntao, and
  "Min Zhang") — all zero hits. A citation could in principle be present in a form none of those
  strings match, but that is unlikely.
- **No paper was found** that applies exact nested-KV-set attention (Hydragen §3.3 / DeFT §3 style) to
  the verification queries of a single speculative step. This is a **negative result from an
  incomplete sweep**, not a proof of absence: with arXiv search blocked (406) and the API 429'd, the
  search was citation-driven and cannot be exhaustive.
