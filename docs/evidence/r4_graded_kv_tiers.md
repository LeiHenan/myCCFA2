# R4 — Adversarial Prior-Art Screen: *Graded* Speculative-KV State Machines Keyed on Commit Probability

**Screen date:** 2026-09-11 · **Screener:** delegated subagent (r4) · **Parent:** `session-28b40a51-488b-473c-898a-edeaf30fe8cd`
**Claim under test (restated verbatim from brief):** *"Speculative KV cache with graded state keyed on commit probability. Split the KV cache into more than two states — e.g. Committed / Speculative / Uncertain — and let `P(commit)` drive promotion and demotion between tiers (resident HBM → compressed → host → drop), rather than a binary committed-vs-speculative split."*

---

## 0. Method, and what "verified" means here

Every source marked **[V]** below was fetched by me in this session, and I saw its title *and* the quoted sentence in the fetched body. Fetch paths that worked on this host:

- `export.arxiv.org/abs/<id>` and `export.arxiv.org/pdf/<id>` — **worked** where `arxiv.org/abs` hung or 000'd.
- `arxiv.org/html/<id>vN` — worked with `--max-time 90`; `arxiv-org.ezproxy.obspm.fr/html/...` — **worked for abs-style HTML but returned a "Cookie Required" interstitial for full-text HTML** (751-byte body); `ezproxy` full text must therefore be treated as unusable.
- `raw.githubusercontent.com` — worked. GitHub HTML with a browser User-Agent — worked for issues/PRs, but `curl` without a per-request `--max-time` wedged (one call was killed at 60 s).
- `techrxiv.org/doi/full/10.36227/techrxiv.177101038.80960856/v1` — **HTTP 403, Cloudflare interstitial** ("Just a moment...") on every attempt, with and without a browser UA. TransKV's abstract was therefore verified through a secondary index that reproduces the publisher's deposit metadata (**scite.ai**, which serves `citation_abstract` + `citation_title` + `citation_doi` from the Crossref/IEEE deposit). **I did not read the TransKV PDF body.** This is flagged again in §6.
- **pip is unusable in this sandbox** (`Errno 1: Operation not permitted: '/Users/leihenan/Library/Python/3.14'`, `--break-system-packages` also refused). PDFs were extracted with a hand-written FlateDecode + text-operator extractor (`/tmp/r4/pdftxt.py`). Consequence: any PDF whose text is set with CID/Type0 fonts without ToUnicode maps is **unreadable** for me. Two papers hit that wall (§6).
- The arXiv index is incomplete for 2026 systems work, as the brief warned. I additionally pulled the **OSDI '26 technical sessions** page as a primary proceedings source, and the **ACL 2026** proceedings index.

---

## 1. T1 — Is a *graded* speculative-KV state machine keyed on commit probability already done?

### 1.1 Verdict on T1

**No occupier found.** I found no work in which `P(commit)` / acceptance probability is the *signal that selects a memory tier* for KV state. What exists is a set of adjacent mechanisms that each hold one piece of the claim and none hold the predicate:

| Work | States / tiers | **Driving signal** | Is it `P(commit)`? |
|---|---|---|---|
| TransKV [V-abstract] | binary: committed vs. packed speculative buffer | token acceptance event (yes/no), post-hoc | **No — binary** |
| vLLM `TieringOffloadingSpec` [V-source] | primary CPU tier + N secondary tiers; promotion/demotion | block-hash **lookup HIT** in a secondary tier, triggered by `lookup()` during scheduling | **No — content-address hit, not probability** |
| vLLM KV-offload "speculative promotions" (#49952) [V] | reserved-vs-running blocks in primary tier | scheduler **admission** (`token_budget`, `max_num_running_reqs`) | **No — admission, not acceptance** |
| OasisKV [V] | GPU resident set vs. host/remote full-KV tiers, per-head | **lookahead-token attention** → predicted important blocks; LRU beyond that | **No — attention score from draft tokens** |
| CONF-KV [V] | loose vs. tight retention budget + FP16/INT8 mixed storage | **next-token distribution (confidence/entropy/margin)** | **No — output confidence, not `P(commit)`; also not speculative** |
| TensorRT-LLM retention policy [V] | continuous priority score (int) per block, `TokenRangeRetentionConfig` | user-configured priority per token range + duration | **No — config, not probability** |
| VIA-SD [V] | **three** verification tiers (direct accept / slim-verifier / full model) | per-token **confidence** | **Closest state-machine analogue, but it tiers *compute*, not KV bytes** |
| MiKV / QuantSpec / QSpec / Don't-Waste-Bits | precision tiers | attention importance / per-token features | No (already known to the brief) |

Two of these deserve emphasis because they are the nearest misses, and both were fetched this session.

**(a) TransKV — the binary occupier (verbatim, from the publisher deposit as re-served by scite):**

> "However, speculative decoding induces temporary KV growth for uncommitted draft tokens, which interacts poorly with block-granular KV allocation and token-budget schedulers. Production documentation explicitly notes that draft tokens consume extra KV-cache pages and count toward global token limits, limiting observed speedups at higher load [12, 14]. We propose TransKV, a transactional KV-cache abstraction that separates stable committed KV state from a packed speculative KV buffer. Speculative KV writes remain uncommitted until acceptance is known; only the accepted prefix is committed into the paged cache and rejected KV is discarded without rollback. We provide a formal memory and scheduling analysis showing speculative KV pressure reduces from block-sized pages to token-sized buffers. This reduction yields significant concurrency improvements in capacity-limited prefix-shared branching workloads (e.g., up to 1.78× at 𝐵=16, 𝑚=2). On real GPUs (Kaggle Tesla P100 16 GiB and Colab Tesla T4 15 GiB), TransKV increases achievable branch concurrency by up to 1.78× at 𝐵=16, 𝑚=2 and by 1.60× at 𝐵=16, 𝑚=4, with exact output equivalence by construction."

Title: *Transactional KV Caching for Speculative Decoding under Paged KV Memory*, Anay Dongre (Cal Poly Pomona), DOI `10.36227/techrxiv.177101038.80960856/v1`, year 2026, type `posted-content` (preprint).
**Driving signal: the acceptance event itself, applied once, at commit time. Exactly two states. No intermediate tier, no probability, no demotion path.** The claim's delta survives this: TransKV never has a state in which a KV row is *known to be uncertain* and is *placed in a cheaper tier because of that*.

**(b) vLLM's "speculative promotion" is a different sense of the word — do not let the name mislead.** PR #49952 [V] is titled *"[Bugfix][KV Offload] Reserve primary-tier headroom so speculative promotions don't starve running stores"*. Its body reads:

> "With tiered KV offload, promotions from a secondary tier could claim every block of the primary (CPU/DRAM) pool on behalf of requests that were still queued, so offloads of the requests actually running failed to allocate. This adds a small headroom reserve so speculative promotions yield the last blocks."
> "…`lookup()` runs from the waiting loop for every queued request (`get_num_new_matched_tokens`, scheduler.py:751), and `allocate_slots()` only decides admission at scheduler.py:946; a request that defers is `continue`d past rather than `break`ed on (scheduler.py:756-762)…"

Here "speculative" means *promoted before admission is known*, not *speculatively decoded*. The tier transition is driven by scheduler admission, not by `P(commit)`. Companion PR #50014 (*"Protect unread promotions from eviction by other speculative promotions"*) is **still open**, i.e. the primary tier's promotion/eviction interaction in shipped vLLM is not even fully settled.

**(c) OasisKV [V] is the nearest *shipped-system-shaped* relative** and I want to be explicit about why it is not the claim. Title: *OasisKV: Scaling In-Decode KV Cache Beyond HBM with Lookahead Sparse Prefetching* (Can Xiao, Sukmin Cho, et al., Microsoft). From the body I fetched:

> "We observe that future important tokens can be predicted accurately in advance using lookahead tokens drafted by speculative decoding (SD). OasisKV employs an efficient attention background pipeline to identify important KV blocks. They are then prefetched from higher-capacity memory tiers (e.g., host or remote memory) and staged in HBMs before being used in the next decode step."

and its eviction rule:

> "It then ranks resident blocks outside the predicted set by their last-selected step and chooses the least-recently-selected entries as eviction targets. Each admitted non-resident block is paired with one eviction target, and these CPU-source/GPU-destination pairs form the transfer plan."

**Driving signal: the *attention mass* the drafted lookahead tokens place on a block — a relevance score, not a commit probability.** Demotion is LRU-outside-predicted-set. So OasisKV does use speculative tokens as a *predictor*, which is the same input the claim would use, but the predicate is "will attention need this?" (relevance), not "will this token be committed?" (acceptance). That is a real distinction and it is the distinction the claim lives on — but it is also a *narrow* one, and OasisKV's measured result (1.69× over dense vLLM on reasoning at 0.1 pt accuracy loss; up to 2.1× multi-GPU long-context) sets the bar a graded-state paper would have to beat while using a strictly weaker signal.

**(d) The three-state analogue that already exists is VIA-SD** (ICML 2026, arXiv 2606.12243) [V]:

> "Draft tokens are processed hierarchically: direct acceptance for high-confidence cases, slim-verifier regeneration for medium-confidence cases, and full-model verification for uncertain cases."

That is a **>2-state machine driven by confidence** — but its tiers are *verification compute paths*, not memory tiers. It is the strongest evidence that "graded speculative state keyed on a probability" is an accepted, publishable idea at a top venue; it is *not* evidence that the KV-memory version is occupied.

### 1.2 What I searched, and what came back empty

Queries run (all via `web_search`, 20+ distinct phrasings): `graded/multi-level/three-state KV cache states`; `"commit probability"` × {`KV cache`, `tier promotion`, `speculative`}; `"acceptance probability"` × {`drives tier`, `tier eviction`, `promotion demotion`}; `KV cache tiering by token acceptance likelihood`; `speculative KV cache demotion compressed host drop`; `confidence-gated KV cache tier promotion`; `KV cache state machine accepted unaccepted tiers`; `probability-weighted speculative KV`. **Not one returned a paper, issue, PR, or release note whose mechanism matched.** The recurring returns were the same occupied cluster the brief already lists (TransKV, SpecMem, MiKV, QuantSpec, OasisKV, tiering docs).

Proceedings checks performed directly, not inferred:

- **OSDI '26 technical sessions page [V]**, fetched in full (583 KB HTML, 237 K chars of text). It has a named session *"KV Cache and Long Context"*. Papers in it: **Strata** (hierarchical context caching across GPU HBM / CPU memory / SSDs; "up to 5× over vLLM-LMCache"), **ECHO** (KV offloading with lossless prefetching for native sparse attention), **DirectKV** (zero-copy KV offloading on GH200/GB200). All three tier *committed* KV and are driven by recency/reuse/predictability of index scores — **none mentions speculative token state, commit probability, or a graded state machine.** I also grepped the full page for `speculat` and the only hits were unrelated (SPEX for tree-of-thought, Osprey/`libDSE` speculative execution for confidential computing and durable workflows, SERENO repurposing speculative decoding for yield points).
- **ACL 2026 proceedings index [V]** (225 K chars): zero hits for `speculative KV`, `tiered KV`, `hierarchical KV`, `commit probability`. (SpecCache, ACL 2026 long.859 [V], is *speculative-model-guided KV recomputation*, not tiering: "employs deep-layer hidden-state norms from a speculative model as a proxy to guide the critical token selection for target large model.")
- **MLSys/ASPLOS/ISCA/ATC:** I did **not** get a clean proceedings page. `usenix.org/conference/atc26/technical-sessions` returned **404**, and I did not find a usable ASPLOS/ISCA '26 accepted-papers listing within the fetch budget. The brief's warning about the arXiv index applies to me here too: **this is a genuine coverage hole** (see §6). The one MLSys '26 item I did confirm as real is Cascade / arXiv 2506.20675 [V] ("Utility-Driven Speculative Decoding for Mixture-of-Experts", MLSys 2026 per vLLM issue #44506 [V]) — that is `utility = tokens_accepted / verification_cost(k)`, a **depth** controller, not a memory-tier controller.

### 1.3 One more negative worth recording

`arXiv 2506.01986` (SpecMemo) [V, full HTML/ar5iv text read] is the paper I expected to be the occupier, because its whole thesis is memory footprint of speculation. It is not: it prunes the *attention tree* and models the *pre-allocated buffer* size. Its own framing:

> "Tree-based speculative decoding methods offer improved acceptance rates (α) … While tree-based speculative decoding offer longer sequences in a single decoding pass, it often causes greater overall computational and memory overhead due to rejected tokens on the tree."
> "Both Medusa and Eagle rely on pre-allocated KV caches that match the model's full context length—an inefficient design for memory-constrained environments. Recent KV cache compression techniques … are non-trivial to extend to speculative decoding: **Each speculative decoding head must retain high numerical precision to pass cumulative verification.**"

That last sentence is an *argument against* the claim's compressed tier for speculative rows — from the paper that has thought hardest about speculative memory. Worth carrying into T4.

---

## 2. T2 — The population question: what fraction of KV **bytes** is held by uncommitted speculative tokens?

### 2.1 The honest answer

**No measurement of this quantity exists in any source I could fetch.** I could not find, anywhere, a published or shipped-system number of the form "X% of KV bytes are draft/uncommitted tokens." I found:

- expressions and buffer sizes for the *draft attention-tree* scratch space,
- per-token acceptance rates,
- concurrency deltas attributable to speculative KV pressure,
- an API that forces the maximum draft window into every request's KV accounting,

but **not the population ratio itself**. Per the brief, I am reporting that absence as a finding rather than computing it.

Two independent confirmations that this is a *community-wide* gap, both fetched this session:

1. **The KV-management survey** (arXiv 2607.02574, *From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving*, Jie Li, Tongyang Wang, Yong Chen, Texas Tech) [V, full HTML] enumerates **seven missing KV-specific measurements** (MG1–MG7) and states of the metadata/reuse family:

   > "MG4: Lifetime / reuse distance — ✗ ✗ ✗ ✗ ✗" (no surveyed archetype reports it)
   > "Reuse-distance and block-lifetime distributions are also absent (MG4), which leaves eviction policies tuned by intuition rather than workload evidence."

   And it elevates exactly our question to an open design goal:

   > **"DG4—Speculative decoding × KV cache.** Speculative decoding creates tentative KV columns that may be accepted or rolled back. Existing distributed-KV systems mostly assume linear append, so they do not specify whether branch KV should be transferred, shared, or discarded before verification. RDMA-based transfer and CXL-style shared memory have different rollback costs; MG1 and MG3 are needed to model the resulting trade-off."

   That is a survey telling me the population question is *uninstrumented*, and that the design question is *unanswered*. It is a secondary source for the existence of the mechanism, but a primary source for the state of measurement.

2. **The claim's own antecedent** (TransKV) asserts the pressure exists only qualitatively: "speculative decoding induces temporary KV growth for uncommitted draft tokens" — no bytes, no ratio, no fraction. Its quantitative claim is a *concurrency* delta under an extreme, non-serving workload (prefix-shared branching at B=16, m=2, on a 16 GiB P100).

### 2.2 The closest things to a population measurement that I *did* verify

These bound the population without reporting it as a ratio. I list them with their exact operating points, because the brief is right that a 5-token draft window and a 64-node tree differ by orders of magnitude — and every number below is small-window.

| Source [V] | Measured quantity | Value | What it constrains |
|---|---|---|---|
| TransKV (deposit abstract) | concurrency gain from token-granular speculative KV, prefix-shared branching | **1.78× @ 𝐵=16, 𝑚=2**; **1.60× @ 𝐵=16, 𝑚=4** | The *effect size* of removing speculative KV pressure entirely. If speculative KV were a tiny slice of bytes, this would be ≈1.0×. It is not ≈1.0× — but the regime is one where the committed prefix is short and shared, which maximises the speculative share. |
| **KVBuffer** (arXiv 2605.19049, Zou & Zhong) | "increase the maximum number of serving requests by **5x** for speculative decoding when verifying **four draft tokens**" | 5× request count | Same shape as TransKV, but for **linear attention**, where "the state is much larger than the per-token key and value" — the authors say so explicitly. **Not transferable to softmax attention.** |
| **PayPal EAGLE3 study** (arXiv 2604.19767) | per-token acceptance rate, production commerce agent, 2×H100, 40 configs, concurrency 1–32, T∈{0, 0.5} | **"acceptance rates remain stable at approximately 35.5% for gamma=3 across all conditions"**; **"gamma=5 … approximately 25% acceptance rate"** | This is the closest thing to a *population* input in the wild: at γ=3, **~two of every three drafted KV rows are never committed.** The *count* of uncommitted rows is large; their *persistence* is what is short. |
| **SpecMemo** (arXiv 2506.01986) | candidate-sequence buffer allocation per query, Medusa tree mask | **"Tree mask built by SpecMemo named 1-10-16-17 in Figure 12 occupies 19.5 MB buffer per query, while allocating 390 MB across 20 queries … Compared to the last mask (provided by Medusa) with 55 MB buffer allocation per query, which accumulates 1.1 GB"** | Draft-tree scratch is **55 MB/query** for a stock Medusa mask on a 7B/13B-class setup. This is the buffer, not necessarily the committed-tier KV, so it does **not** convert to a ratio — but it is the only per-query draft-memory figure I saw stated in MB. |
| **vLLM `scheduler.py`** (main, read directly) [V] | how speculative tokens enter KV accounting | `allocate_slots(..., num_lookahead_tokens=self.num_lookahead_tokens)` at line ~731–734, with `self.num_lookahead_tokens = self.num_spec_tokens` fixed "for lifetime of engine"; `padded_num_tokens = 1 + self.num_spec_tokens` | The engine reserves/accounts for the **full** draft window on **every** request, not `E[accepted]`. Confirms the worst-case (not expected) population is what hardware sees. |
| **TensorRT-LLM docs** [V] | engine's own statement of the phenomenon | **"The purpose of this stage is to make the KV cache and scheduler aware of the fact that speculative decoding will occur. Draft tokens take up extra KV cache pages and count towards the executor's `max_num_tokens` limit."** | Vendor-confirmed, still unitless. Also: "we simply attach the maximum number of draft tokens to each request" — again the **max**, not `E[accepted]`. |
| **vLLM PR #48037** [V] | activation, not KV, but the same failure mode | "That copy can be **~4GB** for a standard Qwen3-8B DFlash model deployment"; before: "Available KV cache: 52.02 GiB"; after: "47.94 GiB" | A ~4.1 GiB swing in **KV pool sizing** caused by speculative decoding's expanded logits copy. Note this is *activation* memory crowding out KV, not uncommitted KV itself — but it is the single largest verified speculative-decoding memory number I found, and it shows the draft path can move the KV budget by ~8%. |

### 2.3 Why I refuse to turn this into a formula

The brief is explicit that arithmetic is not a valid kill, and it is right for a second reason: the ratio is *pathologically* sensitive to the operating point. It scales as `(draft tree nodes) / (batch × context)`. A 5-token window at batch 32 / 2 K context and a 64-node tree at batch 1 / 512 context differ by well over an order of magnitude, and **the field has not published the ratio at either end**. Anyone who hands you a single number here — mine included — is doing arithmetic, not measurement. I will therefore state the *shape* and not a value: the population is **crowd-out-bounded**, not **capacity-bounded**, and §4(a) explains why that distinction decides the claim.

---

## 3. T3 — Is it a flag? Can graded speculative-KV tiering be expressed today via existing config/hooks?

### 3.1 Answer: **the machinery is a flag; the predicate is not. Roughly 80% config-expressible, 0% probability-expressible.**

I verified all four of these knobs in shipped code/docs, not from memory.

| Piece of the claim | Expressible today? | Evidence [V] |
|---|---|---|
| **Resident HBM ↔ compressed** | **Yes, per-layer granularity.** | `vllm/config/cache.py`, `CacheConfig.kv_cache_dtype_skip_layers`: *"Layer patterns to skip KV cache quantization. Accepts layer indices (e.g., '0', '2', '4') or attention type names (e.g., 'sliding_window')."* Plus `skip_page_size_padded` for page-size alignment of skipped layers, and `cache_dtype: CacheDType = "auto"`. So a 2-tier *precision* split is a config list today; a *graded* split is a config list with more entries. (A fuller per-layer RFC exists — vLLM issue #23161 — which I saw referenced but **did not fetch**, so I do not quote it.) |
| **HBM → host → NVMe** | **Yes, N tiers, shipped.** | `docs/features/kv_offloading_usage.md`: `TieringOffloadingSpec` — *"multi-tier. A CPU primary tier plus one or more secondary tiers."* Diagram is `GPU ↔ CPU primary ↔ S0/S1/SN`. "The list is ordered: tier 0 is consulted before tier 1." |
| **Promotion AND demotion** | **Yes, both, named as such.** | `vllm/v1/kv_offload/tiering/manager.py`: *"3. Staged promotion — Chunks in secondary tiers must be promoted to the primary tier before GPU can access them"*; `TransferJob.is_promotion` distinguishes *"True: secondary → primary (promotion) / False: primary → secondary (cascade)"*. |
| **Custom policy hook** | **Yes — this is the real "flag".** | `docs/.../kv_offloading_usage.md`: `eviction_policy` (*"built-in `lru`/`arc`, or a custom `CachePolicy` name"*) and `cache_policy_module_path` — *"Python import path for a custom `CachePolicy` not in the built-in registry… No import or registration call needs to run before the server starts."* Interface is `vllm/v1/kv_offload/cpu/policies/base.py:CachePolicy` with `get/insert/remove/touch` over `OffloadKey`. |
| **`P(commit)` as the driver** | **No.** | Nothing in the tiering path can see a per-token commit probability. `CachePolicy.touch(keys, req_context)` receives an `OffloadKey` and a `ReqContext` — no acceptance signal, and the KV scheduler's speculative window is not exposed to the offload connector. |
| **Per-token *priority* within a request** | **Yes — in TensorRT-LLM, not vLLM.** | TensorRT-LLM KV-cache docs [V]: *"Blocks are assigned priority in line with the retention policy of the request. Blocks with lower priority scores will be freed preferentially to blocks with higher priority. The retention policy is a list of `TokenRangeRetentionConfig` objects, each specifying priority for a given range of tokens, such as 'assign priority X to tokens 10 through 61'. You can also assign a duration in milliseconds for this to remain in effect."* Plus `decode_retention_policy` / `decode_duration_ms` for generated tokens, and a documented default priority of 35. |

### 3.2 Why "it's a flag" is true and still not a kill

Because the *policy class* is pluggable but the *state* is not. Concretely, what is missing is one join:

1. The offload tiering path operates on **committed chunks** (the docs' own section header is *"offloading completed KV blocks to slower but larger tiers … as they are produced"*, and the unit is a **chunk** = *"a fixed-size piece of KV data covering a group of tokens"*, default one GPU block). vLLM issue #43996 [V] shows directly that speculative lookahead rows are **not** addressable as ordinary committed blocks: *"P keeps its full `num_lookahead_tokens`, so at block boundaries P allocates one more block than D. The connector's prefix-cache trimming then drops the first data block instead of the extra lookahead block at the end… D ends up with the wrong data: KV for a speculated token instead of the first block of prompt KV."*
2. The only per-request offload limit is `max_offload_tokens`, which is **ordinal, not probabilistic**: *"Only the first `max_offload_tokens` tokens of the request are offloaded; blocks beyond that point are skipped on the store path."*

So implementing the claim is: write a `CachePolicy` subclass (free, out-of-tree) **plus** plumb an acceptance signal from the spec-decode proposer/verifier into `ReqContext` and address sub-block uncommitted rows (a fork-level change to `vllm/v1/core/sched/scheduler.py` + `kv_cache_manager` + the model runner). That is a *small systems paper*, not a config diff — which is exactly the line the brief draws between (V1) "shipped and the delta is a flag" and everything else. **This claim is on the "not a flag" side of the line.**

---

## 4. T4 — What would make this fail?

### 4(a) Failure mode A: the uncommitted population is too small for grading to matter

**Evidence for the failure mode (strong but indirect):**

- The population is **crowd-out-bounded, not capacity-bounded**. By construction, an uncommitted row exists for at most `(1 + k)` token positions per sequence (or the frontier of a draft tree), and lives for **one verification step**. Its byte footprint is therefore `O(batch × window)`, while the committed tier is `O(batch × context)`. The ratio *must* shrink as 1/context. At a 2 K context and a 5-token window the ratio is a few tenths of a percent; the ratio only becomes material when context is short — which is precisely the regime where HBM pressure is low and tiering has nothing to win.
- Both TransKV's 1.78×/1.60× and KVBuffer's 5× were obtained in exactly that short-context, capacity-starved regime (prefix-shared branching; linear attention), and on **hardware and workloads that are not the deployed baseline** (P100 16 GiB, T4 15 GiB). Neither number transfers to an H100/B200 deployment with 2 K+ contexts.
- The strongest *direct* evidence available: **no one has published the ratio**, in a literature whose survey [V] explicitly flags reuse-distance/lifetime distributions as an unmeasured gap (MG4) and calls speculative × KV "not specified" (DG4). If the population were large and consequential, the number would exist. Note the direction of this inference: I am using the *absence of measurement* as evidence about the *likely size*, which is weaker than a measurement — and I flag it as such.

**Evidence against the failure mode:**

- The 35.5% acceptance rate at γ=3 [V] means **~65% of every drafted KV row is uncommitted in count terms.** The rows are not rare; only their lifetime is short. A grading scheme that *retains* rejected-but-likely-reusable rows (rather than discarding them) changes what "lifetime" means — and that is arguably the claim's real value proposition, not a byte-ratio argument.
- vLLM #48037 [V] shows speculative decoding shifting the KV pool from **52.02 GiB → 47.94 GiB** on a standard Qwen3-8B DFlash deployment. An ~8% swing in how much KV the engine can hold is a *systems-relevant* magnitude, even if the uncommitted rows themselves are a sliver.

### 4(b) Failure mode B: grading costs more than it saves

**Evidence that the overhead is small (which cuts *against* this failure mode):**

- **CONF-KV [V] measured the exact overhead family** and found it cheap: *"In our GPT-2 profile, compaction is 0.22 ms per step at the observed eviction rate and metadata updates are 0.11 ms, together smaller than the attention savings from the reduced cache."* Also, the confidence estimator itself is three scalars (normalized entropy, log-prob margin, top-token mass) with a fixed weight vector.
- **Promotion/demotion machinery is already paid for.** vLLM's tiering manager maintains `PendingPromotion`, `pending_primary_stores`, `ref_cnt`-as-eviction-protection and staged promotions as shipped, tested code — the *transfer* cost of a tier move is not a new cost the claim would introduce.
- **Re-quantization on commit is not required by the claim.** The claim's compressed tier can be a *copy* destination rather than a *replacement*, so "commit" can be satisfied by discarding the compressed copy. The dominant cost is then copy bandwidth, and drafts already write to a scratch buffer (SpecMemo's 55 MB/query Medusa mask [V]); a graded scheme may be re-using scratch rather than adding it.

**Evidence that the overhead bites (which supports this failure mode):**

- **SpecMemo's precision constraint [V]:** *"Each speculative decoding head must retain high numerical precision to pass cumulative verification."* This is the sharpest technical objection I found. Speculative rows are read by the verifier under cumulative acceptance — lossy storage of an uncommitted row can change the accept/reject outcome and therefore the output distribution, which would break the losslessness that every speculative-decoding deployment (and TransKV's own claim — *"with exact output equivalence by construction"*) depends on. **A compressed tier for uncommitted rows is only lossless if it stores the exact bytes**, which means a compressed KV tier for speculative rows is functionally an *offload* tier, not a precision tier. That collapses the "resident → compressed → host → drop" ladder to "resident → host → drop" for the speculative state, i.e. **it removes one of the claim's tiers, not because it is occupied, but because it is lossy.**
- vLLM's own promotion path is fragile enough that the fix is still in review: #49952 (closed, unmerged-to-main as far as I can tell from the page state "Closed") and #50014 (open). Adding a second admission policy on top of an unsettled one is a real integration cost.
- OasisKV [V], which spent a whole pipeline on lookahead-driven tier decisions, had to bound transfers *hard* to keep the overlap window: *"Each admitted non-resident block is paired with one eviction target… OasisKV admits at most Θ pairs per decoding step."* Per-token grading multiplies the number of transfer decisions by the draft window.

### 4(c) A third failure mode the brief did not list, which I think is the decisive one

**Grading only pays if it changes a decision that a binary split gets wrong.** Under a binary split, the uncommitted row is either kept exactly (HBM) or dropped. Under grading, the row is moved to compressed/host and possibly brought back. But the *value* of bringing a row back is bounded by how often it gets committed — and if it *does* get committed, the committed tier is the right place for it anyway, one step later, at which point the existing (already-shipped, content-addressed, LRU/ARC-managed) tiering handles it. **The window in which grading has an advantage over "keep it exactly for one step, then let the normal tiering take over" is one verification step wide.** That is the smallest-defensible-delta question, and it is why I do not think this is a wrong claim — I think it is a claim whose entire value is concentrated in one step of pipeline.

---

## 5. T5 — VERDICT

### **NARROWED.**

One sentence: **the exact mechanism is not occupied — no fetched source lets `P(commit)` select a KV memory tier, and the nearest occupiers (TransKV's binary commit, vLLM's admission-triggered "speculative promotions", OasisKV's lookahead-attention prefetch, CONF-KV's confidence-gated budget) each hold one piece and miss the predicate — but the surviving delta is narrow because the uncommitted population is unboundedly small as context grows, nobody has measured it, and at least one of the claim's three tiers (compressed) is arguably *unusable* for uncommitted rows without breaking losslessness.**

Why not OPEN: the three-state machine and confidence-as-control-signal are both already published (VIA-SD, ICML 2026); the tiering machinery and the promotion/demotion vocabulary are shipped; and the nearest relative (OasisKV) already feeds *speculative draft tokens* into a *tier placement* decision, just with a different predicate. The claim's novelty has been reduced from "use speculation to manage KV memory" to "use the *acceptance* statistic rather than the *attention* statistic as the placement predicate."

Why not DEAD: no source measured the population; no source measured grading's cost; and the survey that audited the field says both questions are open (DG4, MG4). I have **no measurement** that a graded scheme loses to the deployed baseline by any margin, let alone <5%. Under the brief's own kill rules, neither V1 nor V2 is satisfied.

### 5.1 The smallest defensible delta

> **A three-state speculative KV tier driven by a *calibrated, per-position* commit probability, where the middle tier is a lossless host-side staging buffer for the top-`m` draft-tree frontier only, and the tier assignment is derived from the observed per-position acceptance profile of the deployed drafter (`α₁ > α₂ > … > α_k`, monotone by construction) rather than from token identity, attention mass, or a global cap.**

Three things make this the minimal surviving claim:
1. The **predicate** is `P(commit)`, and it is *positional* — which is the one form of it that existing work provably does not use (OasisKV uses attention; vLLM uses admission; TransKV uses the realized accept event).
2. The **middle tier is lossless** (host staging, not compression), which sidesteps SpecMemo's cumulative-verification precision objection *and* concedes the "compressed" tier of the original claim.
3. The **frontier-only** restriction keeps the metadata path O(1) per step instead of O(tree nodes) — directly answering OasisKV's per-step transfer-bound lesson.

### 5.2 The experiment that would settle it

**Two measurements, in order; the first can kill it without building anything.**

**E1 — the population measurement (a day of work, and it is the missing number in the literature).** Instrument a stock `vllm serve` with EAGLE-3 (the brief's own deployed baseline) and add a counter for *KV bytes allocated to uncommitted draft tokens* vs *KV bytes held by committed tokens*, sampled per scheduling step, swept over {context length} × {batch size} × {`--speculative-config` num_speculative_tokens ∈ 3, 5, 7} — plus one draft-**tree** configuration to get the orders-of-magnitude span the brief asks about. Report the ratio distribution, not the mean. **Falsifier: if the 95th-percentile ratio is below ~5% at every operating point that fits HBM at all, grading cannot matter and the claim is DEAD on population grounds** — and I would accept that as a measurement-based kill under V2, not an arithmetic one. This also produces the `α₁…α_k` profile E2 needs, so it is not throwaway work.

**E2 — only if E1 survives.** Implement the middle tier as an out-of-tree `CachePolicy` (the docs promise no fork is needed for the policy itself) plus the minimum plumbing to expose the acceptance signal, and compare against **three** baselines, not one: (i) stock vLLM + EAGLE-3 (the deployed baseline), (ii) **TransKV's own binary split** — the occupier must be beaten, not the null, and (iii) OasisKV's lookahead-attention placement, which uses the *same* speculative input with a *different* predicate and therefore isolates the predicate's contribution. Report end-to-end goodput and P99 TPOT at matched HBM budget, and **explicitly report the metadata + copy cost on the critical path** (CONF-KV's 0.22 ms + 0.11 ms is the number to beat). Falsifier: end-to-end gain over (i) < 5% at matched memory, or a loss to (iii), which would mean the acceptance statistic is a *worse* placement signal than the attention statistic OasisKV already uses — in which case the delta is not just narrow, it is negative.

---

## 6. Could-not-verify (read this before citing anything above)

1. **TransKV's paper body.** `techrxiv.org` returned **HTTP 403 / Cloudflare "Just a moment..."** on every attempt (with and without a browser UA). I verified the title, author, DOI, year, and abstract through **scite.ai's report page for the DOI**, which re-serves the publisher's `citation_title` / `citation_abstract` / `citation_doi` metadata. **I never saw the PDF/Section 1.** Consequences: (i) I cannot confirm whether TransKV has an intermediate state anywhere in its full text — the brief's binary characterisation is confirmed *by the abstract only*; (ii) I cannot confirm or deny a TransKV v2; (iii) its "1.78× / 1.60×" numbers are as-deposited, not reproduced. **A Crossref TITLE query was not needed** (the DOI resolved through the index), but the versioned-DOI risk remains: `.../v1` is what the deposit names, and I did not enumerate versions.
2. **arXiv IDs I could not fetch at all.** `2606.29223` (*Depth Exploration for LLM Decoding* — the one paper whose abstract mentions a "commit position" and "collapses the exploration lattice to retain only reusable branch states") failed on `export.arxiv.org/pdf` (**000**) and on `arxiv-org.ezproxy.obspm.fr` (**cookie interstitial**). I read only its abstract, via the export mirror's `/abs`. **Its full text is the single most likely place a graded-state mechanism could be hiding**, because "collapse the exploration lattice to retain only reusable branch states" is structurally a multi-state KV lifecycle. I saw **no** statement in the abstract that commit probability drives tiering, so I did not count it — but this should be closed before the claim is published.
3. **arXiv 2606.12243 (VIA-SD) full text** — read abstract only (HF papers page + ICML 2026 poster page, both [V]). I did **not** confirm it tiers KV *memory*; I assert only that its verification tiers are compute paths, based on the abstract's own wording.
4. **Two PDFs were text-unreadable** with my extractor (CID/Type0 fonts, no ToUnicode): I fetched but could not read CXL-SpecKV (**arXiv 2512.11920**, cited by 2604.26968 as *"a disaggregated FPGA speculative KV-cache for datacenter serving"*) and parts of HiSpec. CXL-SpecKV matters because 2604.26968 distinguishes it from itself as *"uses speculative prefetching without Bayesian modeling or a deeper tier hierarchy"* — a claim about an occupier that I am repeating from a secondary source, **not verified by me**.
5. **Proceedings coverage is incomplete, and this is my biggest exposure.** The brief told me to check MLSys/OSDI/SOSP/ASPLOS/ISCA/ATC/SC directly. What I actually covered: **OSDI '26** (full technical-sessions page, grepped in full) and **ACL 2026** (full index, grepped). What I did **not** cover: **ASPLOS '26, ISCA '26, ATC '26 (the USENIX URL 404'd), SOSP '26, SC '26, and MLSys '26**. I saw one ISCA '26 item in passing (*Revelator*, speculative address translation — unrelated domain) and one MLSys '26 poster reference (Cascade, via vLLM issue #44506). **Absence of an occupier in ASPLOS/ISCA/ATC/SOSP/SC/MLSys is NOT established by this screen.** Given that hardware-architecture venues are where a "graded KV state machine in HBM" would most plausibly be published, this gap is material.
6. **Not verified, seen only as search-result titles or citations:** Nightjar's DOI `10.1016/j.sysarc.2026.103889` (the search result pointed at a *different* DOI-shaped URL, `S1383762126002079`, and I did not reconcile them); MemSpec beyond its abstract (LCTES 2026, `10.1145/3814943.3816174`); Dynamo KVBM's G1–G4 tiers and Mooncake/LMCache/SGLang HiCache (I read a Dynamo *SGLang HiCache integration* doc reference but not the KVBM architecture itself); vLLM issue #23161 (per-layer KV dtype RFC); vLLM PR #51007 (`module_path` for out-of-tree secondary tiers — I verified the *capability* through the official docs' `module_path` key instead); `TrendingKV`/`Hard-KV`/`SPECTRE`/`Polestar`/`VeriCache` full texts (abstracts only; VeriCache and Polestar were both assessed as *not* occupying the claim on the strength of their abstracts plus, for VeriCache, one equation fragment from a locally cached HTML copy that I did **not** fetch fresh this session).
7. **One source in my T2 table is explicitly synthetic and I excluded it**: a HuggingFace dataset artifact proposing a "KV-pressure speculation governor" (EMA acceptance rate → speculative depth, not KV tiering). Its own text says *"All results are synthetic. Scientific closure requires backend-integrated measurements"* and that the low-acceptance-waste benefit *"remains unquantified."* It is **not** evidence for or against the claim, and it is AI-generated per its own provenance notice — I mention it only so the parent knows I saw it and ruled it out rather than missed it.
8. **The `arxiv-org.ezproxy.obspm.fr` mirror is not usable for full text** (cookie interstitial, 751-byte body). Several search results in this session surfaced `.ezproxy.obspm.fr/html/...` links that *look* like full text; anything cited from those by another screener should be re-verified.
9. **Population ratio: no measurement exists that I could find.** I want this stated plainly in the record, because the brief says its absence is itself a finding: **I could not find any published or shipped-system measurement of the fraction of KV bytes held by uncommitted speculative tokens, at any draft-window size, from any source.** The closest published artefacts are (a) TransKV's concurrency delta, (b) KVBuffer's 5× for linear attention, (c) SpecMemo's 55 MB/query draft buffer, (d) the 35.5% per-token acceptance rate, and (e) vendor documentation confirming the phenomenon without units. Any ratio used to argue for or against this claim will be arithmetic until someone runs E1.

---

## 7. Sources fetched this session, with the verification level reached

**[V-body]** = I fetched the body and saw the title and the quoted sentence.
**[V-abstract]** = I fetched the abstract page and saw title + abstract, but not the full text.
**[V-deposit]** = title + abstract read from a metadata index re-serving the publisher's deposit (publisher site 403'd).
**[V-cached]** = the page was **not** re-fetched over the network in this session; I read a previously-saved local copy in this workspace (`archive/`) and confirmed the document's own self-identifying header (e.g. `<title>Speculative Decoding — TensorRT LLM</title>`, `# KV Cache System`) plus the quoted sentence inside it. Marked separately because a stale local copy is a weaker warrant than a fresh fetch — the two TensorRT-LLM citations in §2.2 and §3.1 rest on this level, and so does the SpecMemo full-text quote in §1.3 (fetched fresh in a prior session, cached at `archive/_r3_scratch/priorart/ar5iv_2506.01986.txt`, re-read this session).

| # | Source | Level | Used for |
|---|---|---|---|
| 1 | vLLM `docs/features/kv_offloading_usage.md` (raw, main) | V-body | T3: `TieringOffloadingSpec`, `cache_policy_module_path`, `spec_module_path`, `eviction_policy`, `max_offload_tokens`, secondary tier types |
| 2 | vLLM `vllm/v1/kv_offload/tiering/manager.py` (raw, main) | V-body | T3: staged promotion, `is_promotion` direction, `PendingPromotion`, ref_cnt eviction protection |
| 3 | vLLM `vllm/v1/kv_offload/cpu/policies/base.py` (raw, main) | V-body | T3: `CachePolicy` ABC (`get`/`insert`/`remove`/`touch`), `ChunkStatus` |
| 4 | vLLM `vllm/config/cache.py` (raw, main) | V-body | T3: `kv_cache_dtype_skip_layers`, `skip_page_size_padded` |
| 5 | vLLM `vllm/v1/core/sched/scheduler.py` (raw, main) | V-body | T2/T3: `num_lookahead_tokens` fixed for engine lifetime; `allocate_slots(num_lookahead_tokens=...)`; `padded_num_tokens = 1 + num_spec_tokens` |
| 6 | vLLM PR #49952 (HTML, browser UA) | V-body | T1: "speculative promotions" = pre-admission promotions; starvation of stores; priority debate with orozery |
| 7 | vLLM issue #44506 (HTML) | V-body | T1: static `num_spec_tokens`, no per-request k; Cascade/MLSys 2026; prior dynamic-k PR #26504 |
| 8 | vLLM issue #43996 (HTML) | V-body | T3: lookahead block = extra block at boundary; prefix-cache trim drops wrong block |
| 9 | vLLM PR #48037 (HTML) | V-body | T2: ~4 GB expanded logits copy; 52.02 → 47.94 GiB KV pool |
| 10 | TensorRT-LLM *Speculative Decoding* doc (local cached copy at `archive/prior_art/axA/trtllm.html`) | V-cached | T2: "Draft tokens take up extra KV cache pages and count towards the executor's max_num_tokens limit" |
| 11 | TensorRT-LLM *KV Cache System* doc (local cached copy at `archive/kv_evidence/trtllm_kvcache.md`) | V-cached | T3: `TokenRangeRetentionConfig`, per-token-range priority, `decode_retention_policy`, default priority 35 |
| 12 | TransKV deposit via scite.ai report page for `10.36227/techrxiv.177101038.80960856/v1` | V-deposit | T1/T2: binary committed-vs-speculative; 1.78×/1.60× |
| 13 | arXiv 2607.02574 *From Tensor Buffer to Distributed Memory Hierarchy* (full HTML) | V-body | T1/T2: Table 11 / MG1–MG7; DG4 speculative × KV; DG3 tiered eviction; P6 / CXL-SpecKV |
| 14 | arXiv 2608.08097 *OasisKV* (PDF, fetched + extracted) | V-body | T1/T2/T4: lookahead-driven tier prefetch; LRU-outside-predicted eviction; per-step admit cap; 1.69×/2.1× |
| 15 | arXiv 2605.24786 *CONF-KV* (PDF, fetched + extracted) | V-body | T1/T4: confidence-gated budget; 0.22 ms compaction + 0.11 ms metadata; mixed FP16/INT8; "orthogonal to speculative decoding" |
| 16 | arXiv 2604.19767 PayPal EAGLE3 study (abstract) | V-abstract | T2: 35.5% acceptance @ γ=3; 25% @ γ=5 |
| 17 | arXiv 2506.01986 *SpecMemo* (full text; fresh-fetched in a prior session, cached at `archive/_r3_scratch/priorart/ar5iv_2506.01986.txt`, re-read this session) | V-cached | T1/T2/T4: 55 MB/query Medusa mask, 19.5 MB pruned, 1.1 GB/20 queries; "Each speculative decoding head must retain high numerical precision to pass cumulative verification" |
| 18 | arXiv 2606.12243 *VIA-SD* (abstract; ICML 2026) | V-abstract | T1: the three-state, confidence-driven analogue |
| 19 | arXiv 2605.19049 *KVBuffer* (abstract) | V-abstract | T2: 5× request count at 4 draft tokens, linear attention only |
| 20 | OSDI '26 technical sessions (full page) | V-body | T1: KV Cache and Long Context session — Strata, ECHO, DirectKV; no speculative-state occupier |
| 21 | ACL 2026 proceedings index (full page) | V-body | T1: zero hits for speculative-KV / tiered-KV / commit-probability |
| 22 | arXiv 2607.14107 *Polestar*; 2604.26412 *KVShot*; 2604.05250 *DualDiffusion*; 2606.29223 *DEX*; 2605.09490 *Not All Thoughts Need HBM*; 2609.03949 *VestigeKV*; 2410.11305-family; 2502.10424 *QuantSpec*; 2404.11912 *TriForce*; 2510.01336 *HiSpec*; 2605.17613 *VeriCache*; 2604.26968 *Predictive Multi-Tier* | V-abstract (mostly) | Checked-and-excluded as occupiers; TriForce/HiSpec PDFs extracted and grepped for tier/commit |

**Bottom line for the parent:** the mechanism is not occupied, so this is not a V1 kill; but the deployment regime that would make it matter is unmeasured, the middle tier may be structurally lossy for speculative rows, and the claim's advantage over shipped binary handling is one verification step wide. Run **E1** before spending any more screening effort — one day of vLLM instrumentation either produces the ratio that makes the paper necessary or produces the measurement that ends it.
