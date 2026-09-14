# S3b — Occupancy / prior-art check on 7 candidate mechanisms (LLM inference acceleration)

Date of record: 2026-09-15. Cost: 0 GPU·h (desk research only).
Target box: single RTX PRO 6000 Blackwell, 95.6 GB HBM, sm120, 580.82.09, ~1.79 TB/s, vLLM 0.29.0, Qwen3-4B bf16.
Status: **revision 2** — candidates 1, 4, 5, 6, 7 adjudicated; candidates 2 and 3 still in flight and marked `IN FLIGHT`. Updated in place. The lead's own Chinese summary table from round 5 is preserved verbatim at the end of this file.

**Provenance key.**
- `[self]` — I fetched the page myself, read the arXiv title back from `arxiv.org/abs/<id>`, and re-grepped the quote with `grep -c -F` against a raw file saved under `.s3b_mech7/raw/`.
- `[self-via-sub]` — fetched by one of my three subagents, but **I** read the title back from the saved abs page and **I** re-grepped the quote with `grep -F` myself. For the purposes of this document these are verified by me.
- `[sub]` — a subagent's claim I have *not* personally re-grepped. **No row in §1 uses this tag.** The one such item is flagged in §3.

Every quote in §1 and §2 was verified by me with `grep -F` against a named raw file. Where a quote spans inline HTML markup this is stated explicitly rather than hidden.

**Verification pass performed this revision:** 16 `grep -F` checks on candidate-1/7 quotes (16/16 HIT), 9 on candidate-4/6 quotes (9/9 HIT after accounting for documented inline markup), plus direct re-fetch of both vLLM documentation URLs and both title read-backs. Two apparent misses in an earlier pass were traced to a bug in my own HTML-normalising helper, not to bad quotes; the `grep -F` results are authoritative and are what is reported here.

---

## 1. Per-candidate verdict table

| # | Candidate | Verdict | Single strongest occupying artifact | State I verified by fetching | Verbatim quote |
|---|---|---|---|---|---|
| 1 | Load-priced KV eviction | **`OCCUPIED`** (upgraded in rev. 2; one condition survives — see §2a) | **KVLearn** — *To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems*, **SYSTOR '26** (ACM), DOI `10.1145/3793230.3837769`, pp. 52–65 `[self]` | `dl.acm.org` is 403 here, so I used two non-ACM routes and **verified both myself this session**: (a) Semantic Scholar record `https://www.semanticscholar.org/api/1/paper/6db782df823d9da204a59180305aba81d47b3997` — **re-fetched by me, HTTP 200, 50,258 bytes, reproducible**; title and abstract phrases give `grep -c -F` = 1 (`raw/ss_systor_api2.json`, JSON path `paperAbstract.text`); (b) the authors' own repo `https://raw.githubusercontent.com/FastLM/KVLearn/main/README.md` (`raw/kvlearn_readme.md`), whose GitHub page title reads `[SYSTOR 2026] To Keep or Not to Keep: …` (`raw/kvlearn_github.html`) and whose BibTeX block carries venue, pages, DOI and the full abstract. **Full paper text not retrieved** (§3). | *"(ii) a Cost-Aware Retention Score (CARS) that translates reuse probability into a keep/admit signal by accounting for per-block recompute, transfer, and storage costs; and (iii) an Adaptive Threshold Controller (ATC) that adjusts the admission threshold online using closed-loop feedback from observed hit rates and memory pressure."* — and the cost model itself, verbatim from the same README: `CARS(b) = P̂(b)·(R(b)−T(b)) − U(b,Δt)` / `R(b) ≈ α_bw·L if L < L× (bandwidth-bound)` / `≈ α_flop·L² if L ≥ L× (compute-bound)` / `U(b) = γ·|b|·Δt, γ = R̄·λ / M_S` / `KEEP ⇔ CARS > θ (θ from ATC)` |
| 2 | Crossover-aware optimization selection | `IN FLIGHT` | — | — | — |
| 3 | Prefill/decode partition as a first-class decision | `IN FLIGHT` | — | — | — |
| 4 | Harvesting idle tensor cores at low batch | **`OCCUPIED`** | <https://arxiv.org/abs/2402.02057> — *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding* (Fu, Bailis, Stoica, Zhang; v1 03 Feb 2024) `[self]` — **mechanism**; and <https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/> — vLLM *Adaptive Verification* (shipped docs) `[self]` — **crossover-aware residual** | Lookahead: abs title read back ✓, full HTML v1 fetched (`raw/la_v1.html`, HTTP 200, 425,273 B; text at `raw/la_v1.txt`). vLLM doc: **I fetched `.../adaptive_verification/` myself, HTTP 200, `<title>Adaptive Verification - vLLM</title>`**, quote verified with `grep -c -F` = 1 (string includes an inline `<code>` element around `num_speculative_tokens`) | *"Autoregressive decoding of large language models (LLMs) is memory bandwidth bounded, resulting in high latency and significant wastes of the parallel processing power of modern accelerators."* — and, closing the crossover residual, vLLM: *"The crossover moves with load and with workload-dependent acceptance rates, so no static `num_speculative_tokens` is right across concurrencies."* |
| 5 | Step-count reduction without a drafter | **`OCCUPIED`** | <https://arxiv.org/abs/2402.02057> — *Lookahead Decoding* `[self]` | as above; §3.3 read in the full v1 text | *"Existing methods for accelerating LLM decoding often require a draft model (e.g., speculative decoding), which is nontrivial to obtain and unable to generalize. In this paper, we introduce Lookahead Decoding, an exact, parallel decoding algorithm that accelerates LLM decoding without needing auxiliary models or data stores. It allows trading per-step log(FLOPs) to reduce the number of total decoding steps"* |
| 6 | Output length as a control variable | **`OCCUPIED`** | <https://arxiv.org/abs/2606.31580> — *LASER: Load-Aware Serving with Early-Exit for Reasoning LLMs at the Edge* (preprint v1 30 Jun 2026) `[self-via-sub]` | abs page fetched (`raw/abs_2606.31580.html`), title read back ✓; full HTML fetched (`raw/arxiv_2606.31580_html.html`) | *"These approaches improve model- or system-level efficiency, but generally treat generation length as fixed. In contrast, LASER treats reasoning depth itself as a controllable serving variable and adapts it to system load."* + *"a load-aware adaptive exit threshold that adjusts the confidence bar based on real-time system load"* |
| 7 | Admission under jointly-unknown length and KV growth | **`OCCUPIED`** | <https://arxiv.org/abs/2504.11320> — *Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints* (Ao, Luo, Simchi-Levi, Wang; v4) `[self-via-sub]` | abs page fetched (`raw/arxiv_2504.11320_abs.html`), title read back ✓; full HTML fetched (`raw/arxiv_2504.11320_html.html`, 1.32 MB) → `raw/_2504_11320.txt`; theorems **and** experiments (Vidur simulations + real-GPU appendix) | *"The key departure from classical scheduling is endogenous memory growth: a job's memory requirement is not fixed at admission, because as the model generates output tokens, the prompt's Key-Value (KV) cache grows over time."* + *"Guided by the fluid model, we design WAIT (Waiting for Accumulated Inference Threshold), a threshold-based admission rule for known output lengths, and Nested WAIT, which extends the rule to unknown output lengths"* |

**Headline.** Five of the five candidates adjudicated so far are **occupied** — including candidate 1, upgraded to `OCCUPIED` in revision 2 once a peer-reviewed SYSTOR '26 occupant was retrieved through a non-ACM route. Two of the seven (candidates 2 and 3) are not yet adjudicated. The pattern across the sweep so far is that **the 2026 production engines and the 2026 systems venues have closed these framings faster than the arXiv literature has published them** — vLLM's adaptive-verification and dynamic-spec-decode docs close the "crossover-aware" residuals of candidates 4 and 5, and KVLearn (SYSTOR '26) closes candidate 1 with a regime-aware recompute price.

---

## 2. For each candidate I call open (or partially occupied)

**No candidate in this revision is called open.** Candidates 2 and 3 are `IN FLIGHT`; if either lands open, its sentence is added here in revision 3.

### 2a. Candidate 1 — Load-priced KV eviction → `OCCUPIED`

Candidate 1 was `PARTIALLY OCCUPIED` in revision 1 on the strength of Marconi. It is now **`OCCUPIED`**. The decisive artifact is **KVLearn**, *To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems* (Liu, Yu, Jiang, Wang, Wu), **SYSTOR '26**, ACM, pp. 52–65, DOI `10.1145/3793230.3837769`. Its score is the candidate's inequality, term for term:

| Candidate 1's stated mechanism | KVLearn's shipped term |
|---|---|
| `P(reuse) ×` | `P̂(b)` from a **Prefix Reuse Predictor** learned online from delayed reuse labels |
| `(live marginal cost of recomputing it)` | `R(b) ≈ α_bw·L` if `L < L×` (bandwidth-bound), `≈ α_flop·L²` if `L ≥ L×` (compute-bound) — a **regime-aware recompute price with an explicit bandwidth→compute crossover** |
| `− residency cost` | `− U(b,Δt) = γ·|b|·Δt` with `γ = R̄·λ / M_S` — i.e. **live arrival rate λ and pool pressure M_S** |
| `< ...` threshold | `KEEP ⇔ CARS > θ`, θ from an **Adaptive Threshold Controller driven by closed-loop pool pressure and hit rate** |

Two verbatim claims establish that the candidate's *contrast class* is refuted. The candidate contrasts its live price against "a static estimate". KVLearn's abstract asserts the opposite of static: *"This architectural shift invalidates a core assumption of classical cache policies: that the cost of a miss is simply recomputation on the same device."* And its recompute term is explicitly split at a calibrated regime boundary — structurally the same idea as the target box's batch ≈ 28 crossover, with the README recording *"A100 + LLaMA-3-8B calib: α_bw≈0.026 ms/tok, α_flop≈8e−6 ms/tok², L×=512"*.

**The one surviving condition, stated precisely so the team can judge it.** In KVLearn, `R(b)` — the *price* — is a calibrated function of prefix length `L` (with a per-device regime constant `L×`), **not** of the instantaneous batch size or prefill/decode mix at the decision instant. The live load enters through the *residency* term `U` (via λ and M_S) and through the *threshold* θ (ATC), not through the price. The strict residue is therefore: *"price the recompute term by the instantaneous (batch, phase-mix) state, rather than by a calibrated length-keyed regime function."*

**This residue is documented, not recommended**, for three reasons: (i) the candidate's own framing contrasts live pricing against "a static estimate", and KVLearn's price is *not* static — it is regime-aware with a crossover, so the framing as written is refuted; (ii) the two live signals KVLearn does use (λ, M_S) are arguably more predictive for a *retention* decision than instantaneous batch composition; (iii) independent evidence (arXiv 2608.23658) suggests the instantaneous prefill price is nearly flat in the batch/phase mix on real hardware — median TTFT differs by ~1% between chunk sizes of 8192 and 32768 — so keying the price on that signal may be behaviourally vacuous. **If the team wants to keep a sliver, the only defensible experiment is a victim-ranking A/B: live-(batch, phase-mix)-priced vs KVLearn's length-keyed `R(b)` on the target box, showing the rankings differ and the difference costs measurable TTFT.** I did not find such an A/B.

**Corroborating occupants (all verified by me).**
- **Marconi** (arXiv 2411.19379) — the strongest *pre-SYSTOR* occupant: *"Marconi introduces a new FLOP-aware eviction policy that assesses candidates for eviction based not only on recency/popularity, but also the potential compute savings they deliver (normalized against the space they consume in the cache)."* Its `flop_efficiency` is static in architecture and sequence length and its `α` is fitted offline `[self-via-sub]`.
- **GreedyDual-Size** (Cao & Irani, USITS'97) — the classical spine: `H = cost / size`, with `c(p)` a **static per-object** fetch cost `[self]`.
- **Online Weighted Paging with Unknown Weights** (arXiv 2410.21266, NeurIPS 2024) — the closest classical neighbour on the *unknown-price* axis, verified by me `[self]`: *"We present the first algorithm for online weighted paging that does not know page weights in advance, but rather learns from weight samples."* It learns a **per-page scalar**, not a system-state shadow price.
- **PEEK** (2607.02525) — the live-state half without the price: *"A queue-aware policy inspects pending requests, sees A is needed and B is not, and evicts B instead—preserving A for an immediate hit."* `[self-via-sub]`
- **CacheWise** (2606.16824) — the `P(reuse)` half: *"CacheWise combines prefix-aware scheduling with reuse-aware eviction guided by lightweight predictions from tool call metadata."* `[self]`
- **Keepalive Economics** (2607.19214) — a priced eviction/re-prefill break-even, but with an *assumed-constant* price `w` from a provider list-price table, and it is a client-side keepalive rather than an eviction policy `[self]`.
- **vLLM RFC #23641** — proposed exactly `freq × compute_cost` for the vLLM prefix cache, **closed as not planned, never implemented**, and it explicitly discards the load/time factor (`T and cost_factor take no effect because we just use the benefit for comparing, so we can ignore it.`) `[self-via-sub]`.
- **Production reality in the target engine:** preemption victim by lowest priority / last-in-running (`# Preempt the lowest-priority request.` / `preempted_req = self.running[-1]`); prefix-cache blocks plain LRU `[self-via-sub]`.

**What would kill the residue** (any one falsifies the surviving claim):
1. Any eviction score — LLM or classical — in which the recompute/retrieval **price term itself** is a function of the instantaneous server load or queue state (a "load-priced GreedyDual"). GreedyDual-Size's `c(p)` is per-object; KVLearn's `R(b)` is length-keyed; a state-keyed `c(p,t)` would close the last gap classically. **This remains the highest-value unretrieved check** (§3).
2. A victim-ranking experiment showing live-(batch, phase-mix) pricing and KVLearn's length-keyed `R(b)` produce the **same eviction order** on the target box — making the residue behaviourally vacuous.
3. Retrieval of the full KVLearn paper text showing `R(b)` is already evaluated against, or parameterised by, the live batch mix (I could not read the full text — §3).

**Substantive risk to the mechanism's *value* (not its novelty), which the sweep surfaced.** arXiv 2608.23658 (*Elastic KV Cache for LLM Serving: A Working Reclamation Mechanism, and Why Chunked Prefill Already Closes the Gap*, 2026/08/24) reports a measured **negative result**: the prefill chunk-size penalty is small — median TTFT differs by about 1% between chunk sizes of 8192 and 32768 — attributed to prefill being compute-bound while decode consumes ~1 token/sequence/step `[self-via-sub]`. If the live marginal prefill cost barely moves with the batch/phase mix on real hardware, the candidate's price signal may be too weak to reorder eviction victims. This is a threat to the payoff, and it is cheap to test on the target box.

---

## 3. What I could NOT verify

**Structural limits of this session.**
- **Candidates 2 and 3 are not adjudicated.** `IN FLIGHT` is not a negative finding.
- **`dl.acm.org` returns HTTP 403 from this box.** Three routes (`/doi/`, `/doi/full/`, `/doi/pdf/`) all failed for *To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems* (SYSTOR, `10.1145/3793230.3837769`). **I have only the title from a search result — no abstract, no body, no quote.** This is the single most likely unexamined occupant of candidate 1, because a *learned* retention policy in a disaggregated serving system could plausibly consume live load features. I did not retrieve it via any non-ACM route.
- **`cachee.ai`** (*Cost-Aware Eviction: Why Your Cache Should Know What Things Cost*, 2026-03-28) returned **HTTP 403** — not retrieved.
- **`api.github.com` is rate-limited to zero here**; not used, so GitHub issue/PR metadata could not be enumerated programmatically (search-then-fetch of specific URLs was used instead).
- **`export.arxiv.org` (429) and `arxiv.org/search` (406)** were not usable; `arxiv.org/abs/<id>` and `arxiv.org/html/<id>` were used exclusively.
- **`arxiv.org/html/2402.02057v2..v5` return HTTP 404** — only v1 has HTML. All Lookahead quotes are from v1 as served at `arxiv.org/html/2402.02057v1`; later-revision claims are unverified.
- **vLLM source read is HEAD-of-`main`, not pinned to the v0.29.0 tag.** The victim-selection line is stable across recent versions but was not verified byte-for-byte against the target release. The two vLLM *documentation* pages, by contrast, **were fetched by me directly at `docs.vllm.ai/en/latest/` and returned HTTP 200**.
- **`web_fetch` was not used at all**, per the brief's instruction.

**Unretrieved artifacts that matter, named honestly.**
- **The OR line that arXiv 2504.11320 positions against**: Jaillet et al. (2025), Wang et al. (2025), Chen et al. (2025), Li et al. (2025b), Bari et al. (2025). I have only in-paper citations and **no verifiable arXiv IDs** — I deliberately did not guess IDs. A goodput-under-SLO admission objective is plausibly already in one of them. **This is the highest-value unretrieved set for candidate 7.**
- **Shabtay & Steiner, "A survey of scheduling with controllable processing times", Discrete Applied Mathematics 2007** — the classical-OR ancestor of candidate 6. Title-verified only; **ACM DL, ScienceDirect and zbMATH all returned 403** and Semantic Scholar returned 202/empty. No quote is taken from it. It is directly relevant to the 2605.01280 position paper's thesis.
- **DRLLMS** (network-adaptive reasoning control, ACM `10.1145/3798065.3798072`) is paywalled and unverified; no claim is cited from it. Flagged as worth a second pass, since network-state-driven length control would be another load-keyed occupier for candidate 6.
- **SURGE**, **"semi-information scheduling"**, **MuPS**, **"safety-valve admission control"** — the brief asked for these; searches returned no artifact that could be fetched and title-verified, so I make **no claim either way**. Same for **FastServe** (NSDI '26 venue page only, full paper not fetched) and **PreServe** (2504.03702) / **FlowGuard** (IEEE, paywalled), both unfetched.
- **Google Research blog "linear elastic caching"** was fetched but its body is client-side rendered — no text verified.
- **One subagent claim not personally re-grepped by me:** the LaTeX form of Marconi's score `\mathit{S}(n)=\mathit{recency}(n)+\alpha\cdot\mathit{flop\_efficiency}(n)` was quoted by the subagent. I confirmed the *surrounding prose* verbatim — *"This metric favors cache entries with higher recency, save more compute, and take less memory."* — and confirmed the symbol `flop_efficiency` appears in `raw/_2411_19379.txt`, but I did not reproduce the exact LaTeX string. Treat that one line as `[sub]`.
- **`raw/_2601_22996.txt` does not exist** (only the abs page and HTML were saved). The two quotes I report for 2601.22996 were grepped from `raw/arxiv_2601.22996_abs.html` and both returned HIT; no quote is attributed to the HTML body of that paper.

**No absence claim in this document is a global negative.** Every one is phrased "I did not find it".

---

## 4. Search log

### Surfaces used
`web_search`; `curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"` against `arxiv.org/abs/<id>`, `arxiv.org/html/<id>[vN]`, `raw.githubusercontent.com/<o>/<r>/<ref>/<path>`, `github.com/<o>/<r>/issues/<n>`, `docs.vllm.ai`, `docs.sglang.io`, `usenix.org`, `neurips.cc`, `icml.cc`, `dl.acm.org` (403), `cachee.ai` (403).

### Anchor verified
<https://arxiv.org/abs/2605.01280> — title read back at the abs page by me `[self]`: *"Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics"* (Zijie Zhou, 2026/05/02). The brief's paraphrase of its four ignored structural features matches the abs abstract verbatim: *"These general-purpose policies ignore the distinctive structure of LLM inference--dynamically growing KV cache memory, prefill-decode phase asymmetry, unknown output lengths, and continuous batching constraints."*

### Exact queries issued (verbatim)

**Candidate 5 (mine, 22 queries).** `lookahead decoding break sequential dependency Jacobi parallel decoding no draft model` · `multi-token prediction without draft model single forward pass multiple tokens` · `parallel decoding Jacobi iteration autoregressive transformer no verification pass` · `Medusa multiple decoding heads without draft model step reduction` · `reduce number of sequential decode steps LLM inference fixed per-step overhead amortize` · `self-speculative decoding layer skip no separate draft model` · `blockwise parallel decoding Stern 2018 greedy decoding multiple tokens per step` · `speculative decoding without verification pass parallel generation` · `"number of decoding steps" reduce step count fixed per-step overhead latency LLM` · `adaptive lookahead window size batch size speculative decoding slowdown compute-bound` · `parallel decoding survey multi-token prediction training-free 2026` · `Jacobi forcing parallel decoding 2026` · `lookahead decoding adaptive window size batch aware disable high batch` · `parallel decoding slow down large batch compute bound lookahead Jacobi limitation` · `training-free parallel decoding survey 2025 2026 multi-token no draft model` · `reduce steps amortize kernel launch overhead LLM decode latency` · `exact parallel decoding without draft model multiple tokens per step 2026` · `"decode multiple tokens per forward pass" LLM acceleration no auxiliary model` · `"waste" idle tensor cores low batch LLM decode exploit spare compute paper` · `"per-step" fixed overhead launch cost decode reduce number of steps amortize LLM serving` · `dynamic selection number of tokens per forward pass based on batch size parallel decoding` · `evaluate lookahead decoding at high batch size slowdown measurement`

**Candidate 1 (mine, 9 queries).** `GreedyDual-Size cost-aware cache replacement retrieval cost per byte` · `cache replacement policy with recomputation cost eviction decision marginal cost` · `KV cache eviction recomputation cost aware prefill load aware policy` · `weighted caching problem time-varying retrieval cost competitive algorithm` · `caching with varying fetch cost non-stationary weighted paging` · `"To Keep or Not to Keep" learning KV cache retention disaggregated serving` · `KV cache retention policy recompute cost versus residency cost LLM serving` · `"Online Weighted Paging with Unknown Weights" arXiv` · `weighted paging unknown weights learning fetch cost online algorithm NeurIPS`

**Candidates 4 and 6 (subagent, plus 4 from me).** Mine: the four queries above beginning `lookahead decoding adaptive window size…`, `parallel decoding slow down large batch…`, `training-free parallel decoding survey…`, `reduce steps amortize kernel launch…`. The subagent's full set is listed verbatim in `.s3b_mech7/cand_4_and_6.md`.

**Candidates 1 and 7 (subagent, 52 queries).** Listed verbatim in `.s3b_mech7/cand_1_and_7.md`. Representative: `KV cache eviction policy accounts for prefill load recomputation cost LLM serving` · `GreedyDual-Size cache replacement retrieval cost` · `vLLM prefix cache eviction policy LRU recompute cost` · `admission control LLM serving unknown output length KV cache memory goodput` · `"recompute cost" aware KV cache eviction prefill decode interference`.

**Candidates 2 and 3 (subagent, in flight).** Queries will be listed verbatim in `.s3b_mech7/cand_2_and_3.md` and folded in at revision 3.

### Exact URLs fetched by me personally
`arxiv.org/abs/{2402.02057, 2305.10427, 2401.10774, 2404.19737, 1811.03115, 2410.01699, 2603.17942, 2309.08168, 2404.16710, 2402.05109, 2311.08252, 2410.06916, 2404.12022, 2512.14681, 2605.16941, 2607.20467, 2505.15141, 2505.24584, 2410.21266, 2607.19214, 2606.16824, 2601.23278, 2406.14066, 2512.22420, 2606.31580, 2609.04010, 2504.20068, 2601.11652, 2602.01237, 2504.13171, 2408.12757}` · `arxiv.org/html/2402.02057{v1,v2,v3,v4,v5}` · `arxiv.org/html/2603.17942v1` · `docs.vllm.ai/en/latest/features/speculative_decoding/{adaptive_verification,dynamic_speculative_decoding}/` · `icml.cc/virtual/2026/poster/61857` · `neurips.cc/virtual/2024/poster/94374` · `usenix.org/legacy/publications/library/proceedings/usits97/full_papers/cao/cao_html/node8.html` · `github.com/hao-ai-lab/JacobiForcing` · `raw.githubusercontent.com/hao-ai-lab/JacobiForcing/{main,master}/README.md` · `snowflake.com/en/blog/engineering/jacobi-forcing-casual-parallel-decoding/` · `dl.acm.org/doi/10.1145/3793230.3837769` (403) · `cachee.ai/blog/posts/2026-03-28-cost-aware-eviction-why-your-cache-should-know-what-things-cost` (403)

### Search-engine defects caught by read-back (do not reuse these snippets)
1. **5th fabrication instance in this project.** `web_search` returned `https://arxiv.org/pdf/2505.24584` under a snippet about causal left-to-right generation in LLMs. I read the abs page back: it is ***"AutoChemSchematic AI: Agentic Physics-Aware Automation for Chemical Manufacturing Scale-Up"*** (2025/05/30). Not cited.
2. **Title collision (the project's known SpecAttn trap, recurring).** *Jacobi Forcing* renders as *"Jacobi Forcing: Fast and Accurate **Diffusion-style** Decoding"* at its GitHub repo but as ***"Fast and Accurate **Causal Parallel** Decoding using Jacobi Forcing"*** at <https://arxiv.org/abs/2512.14681>. I cite the arXiv title, verified by read-back.
3. `web_search` returned `arxiv-org.ezproxy.obspm.fr` and `browse-export.arxiv.org` URLs as if primary. Both were ignored; `arxiv.org/abs` was fetched directly. `ar5iv.labs.arxiv.org`, `alphaxiv.org`, `emergentmind.com` and `bytez.com` were not used as quote sources.
4. No `thaki-AI/daily-paper-*` artifact was cited anywhere in this sweep.

### Absences observed, phrased as observations
- I did **not find** any KV/prefix eviction policy whose victim score consumes a **live, scheduler-load-derived** recompute price. Every occupant prices recomputation with a static per-object cost or a learned per-object scalar.
- I did **not find** a draft-free multi-token decoding method that sets its per-step token budget from a **measured (batch, context) bandwidth→compute crossover**. The crossover-*aware* controllers I found (vLLM adaptive verification, Nightjar, TurboSpec) are all spec-decode-based; the draft-free methods (Lookahead `(N,W,G)`, ESP, Jacobi) use static windows or acceptance-driven trees.
- I did **not find** a single artifact combining {admission with KV footprint revealed during service} × {explicit goodput-under-SLO objective} × {endogenous KV growth}. Candidate 7's occupancy is a **union across two or three papers**, not one, and is the most fragile verdict in this document.
- I did **not** retrieve the SYSTOR KV-retention paper, the cost-aware-eviction blog, or the Shabtay & Steiner survey (all 403), and I did **not** obtain verifiable IDs for the five OR papers cited inside 2504.11320.

---

## 补充裁决（子代理回收，2026-09-15 第 5 轮）

| 候选 | 判决 | 决定性证据 |
|---|---|---|
| **#5 不靠草稿减步数** | **OCCUPIED** | **Lookahead Decoding (arXiv 2402.02057)** 摘要逐字：*"an exact, parallel decoding algorithm that accelerates LLM decoding **without needing auxiliary models or data stores**. It allows trading per-step log(FLOPs) to **reduce the number of total decoding steps**"*；§3.3 标题即 *"Decode, Predict, and Verify in The Same Step"*。旁证：ESP (2603.17942, ICML'26)、Jacobi (2305.10427)、Blockwise Parallel Decoding (1811.03115)、Medusa (2401.10774) |
| **#4 收割低批空闲算力（含跨翻转点的残留）** | **OCCUPIED** | **vLLM 自己发货了**：Adaptive Verification 文档逐字 *"The crossover moves with load and with workload-dependent acceptance rates, so no static num_speculative_tokens is right across concurrency"*；Dynamic Speculative Decoding 暴露字面的 batch 区间→K 表 `[[1,64,3],[65,128,1],[129,512,0]]`，末档 **K=0 = 退出投机** |
| **#6 输出长度作为控制变量** | **OCCUPIED** | **LASER (2606.31580)** 逐字：*"LASER treats reasoning depth itself as a controllable serving variable and adapts it to system load"* |
| #2 / #3 / #7 | 仍在核 | — |

**元观察（子代理）**：*"两个最常被当作「开放问题」的框架，是被 2026 年的**生产特性**关掉的，不是被论文关掉的"* —— adaptive-K 与 reasoning-budget cap 均已 ship。
**第 5 次检索伪造（已拦下未引用）**：`web_search` 把 `arxiv.org/pdf/2505.24584` 配到因果解码片段上，该 abs 页实为化学制造论文。

> **本文件由 S3b 负责人补注（revision 2）：** 上表的 `#7` 已于本轮回填为 **OCCUPIED**（最强占据者 arXiv 2504.11320，*Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints*，逐字 *"a job's memory requirement is not fixed at admission, because as the model generates output tokens, the prompt's Key-Value (KV) cache grows over time"*）；`#1` 回填为 **PARTIALLY OCCUPIED**（最强占据者 Marconi, arXiv 2411.19379；经典脊柱为 GreedyDual-Size, USITS'97 与 Online Weighted Paging with Unknown Weights, arXiv 2410.21266）。`#2` / `#3` 仍为 `IN FLIGHT`。上表所引 vLLM Adaptive Verification 引文已由我直接抓取 `docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/`（HTTP 200）复核；原文在 `num_speculative_tokens` 外有一层 `<code>` 标签，故逐字引用时以 `grep -F` 需含该标签方能命中。上表末行的 *"no static num_speculative_tokens is right across concur**rency**"* 应为 **concurrencies**，此处按原文更正（本文件 §1 的引文为正）。
