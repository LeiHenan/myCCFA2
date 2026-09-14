# S3b — Adversarial Prior-Art / Occupancy Check: Candidates 1 and 7

Agent: delegated S3b subagent (`70db0ba3-0c65-4fd9-8d30-55861c44dc78` is parent).
Workspace: `/Users/leihenan/Desktop/myProject`. All raw HTML saved under `.s3b_mech7/raw/`.
Every arXiv ID cited below had its title read back from `arxiv.org/abs/<id>` **in this session** (see §Verification ledger).
`api.github.com` was never used. `export.arxiv.org` and `arxiv.org/search` were never used.

**Note on the raw directory:** `.s3b_mech7/raw/` already contained ~130 files from other subagents when I started.
Files I fetched in this session are listed in the Verification ledger at the bottom; quotes below point only at files I fetched.

**HOW TO RE-VERIFY EVERY QUOTE (I ran this myself; 39/39 pass):**
All quotes are `grep -F`-verifiable against the named raw file. For the `.txt` derivatives and the plain-text source files,
`grep -F '<quote>' <file>` works directly. For the HTML/XML-ish files (`*.html`), a few quotes straddle a newline, so use:

```sh
tr '\n' ' ' < raw/<file>.html | grep -F '<quote>'
```

Two quotes are stored with HTML entities and must be grepped in entity form (I say so inline):
`a job&#39;s memory footprint grows linearly...` (arXiv 2601.22996 abs page) and
`driven by the scheduler&#39;s one-step-ahead view...` (arXiv 2608.23658 abs page).
One source has a ligature-broken PDF text layer (Hyperbolic Caching ATC'17) — flagged in its row.

---

# CANDIDATE 1 — "Load-priced KV eviction"

**Mechanism under test:** evict a KV block when `P(reuse) × (live marginal cost of recomputing it) < residency cost`,
where the recompute price is read from the **scheduler's current load state** (prefill competing with decode for the same SMs)
rather than a static/offline estimate.

## 1. VERDICT

### `PARTIALLY OCCUPIED`

The **inequality form itself is classical and is already instantiated for prefix/KV caches**. What I did *not* find is any
artifact that re-reads the recompute price from the **live scheduler load state** (current batch size / phase mix / queue depth)
at eviction time. Every occupant I found prices recomputation with a **static per-object cost** (FLOPs, bytes, `size^alpha`, download latency).

### Single strongest occupying artifact

**Marconi: Prefix Caching for the Era of Hybrid LLMs** — https://arxiv.org/abs/2411.19379
(also fetched as full HTML: https://arxiv.org/html/2411.19379v3)

- **State verified by me:** preprint, abs page fetched by me → `raw/arxiv_2411.19379_abs.html`; title read back ✓
  (`citation_title" content="Marconi: Prefix Caching for the Era of Hybrid LLMs"`, authors Pan/Wang et al., MLSys-lineage system paper with
  artifact appendix). Full HTML body fetched by me → `raw/arxiv_2411.19379_html.html`, tag-stripped to `raw/_2411_19379.txt`.
  Has real experiments (end-to-end results, "Fine-Grained Analysis of FLOP-Aware Eviction", microbenchmarks/ablations).
- **Verbatim quote** (from `raw/_2411_19379.txt` line 499; string also present in the raw HTML `raw/arxiv_2411.19379_html.html`):

  > `To enable more holistic management of cache entries, Marconi introduces a new FLOP-aware eviction policy that assesses candidates for eviction based not only on recency/popularity, but also the potential compute savings they deliver (normalized against the space they consume in the cache).`

  Corroborating verbatim, same file line 508 (eviction score and its meaning):

  > `This metric favors cache entries with higher recency, save more compute, and take less memory.`

  And line 510 (the eviction loop):

  > `During eviction, Marconi iteratively removes nodes with the lowest utility score until there is enough space to accommodate the new request’s states.`

  And the abs page (`raw/arxiv_2411.19379_abs.html`), which states the **reuse-probability** half explicitly:

  > `Key to Marconi are its novel admission and eviction policies that more judiciously assess potential cache entries based not only on recency, but also on (1) forecasts of their reuse likelihood across a taxonomy of different hit scenarios, and (2) the compute savings that hits deliver relative to memory footprints.`

- **How close:** very close. Marconi's eviction score, verbatim from `raw/_2411_19379.txt` (the file's own LaTeX rendering — grep this exact string):

  > `\mathit{S}(n)=\mathit{recency}(n)+\alpha\cdot\mathit{flop\_efficiency}(n)`

  (rendered in the paper as `S(n) = recency(n) + α·flop_efficiency(n)`) is
  `P(reuse)-decayed-by-recency × (compute-savings / memory-footprint)` — i.e. **the candidate's inequality with the recompute
  term measured in FLOPs**. The one distinguishing condition it misses: `flop_efficiency` is a **static function of the model
  architecture and sequence length** (Total FLOPs across layers ÷ memory of all states), and `α` is fitted **offline/retrospectively**
  from a bootstrap workload sample. Nothing in the eviction score reads the current batch size, the current prefill/decode mix,
  or measured step time. So it does not price *marginal* recomputation at current load.

## 2. Further artifacts (table)

| # | Artifact / URL | State verified by me | Verbatim quote (raw file to grep) | What it covers |
|---|---|---|---|---|
| 1 | **Cao & Irani, "Cost-Aware WWW Proxy Caching Algorithms" (GreedyDual-Size), USITS 1997** — https://www.usenix.org/legacy/publications/library/proceedings/usits97/full_papers/cao/cao_html/node8.html | Classical peer-reviewed paper; page fetched by me → `raw/gds_usenix_node8.html`; abstract page also fetched → `raw/gds_cao_irani_full.html` | line 48: `` GreedyDual algorithm by setting <I>H</I> to <I>cost</I>/<I>size</I> upon an access to a `` · line 54: `the downloading latency if the goal is to minimize average latency, and` | **Strongest CLASSICAL occupant.** Evict `argmin H`, `H(p)=L+c(p)/s(p)`: this *is* "retrieval cost per unit residency, decayed by recency". Its `c(p)` is a **static per-object fetch cost**; the paper never makes `c` a function of server load. |
| 2 | **vLLM RFC #23641 "[RFC]: Frequency and Cost Aware Eviction Policy for Prefix Caching"** — https://github.com/vllm-project/vllm/issues/23641 | GitHub issue fetched by me → `raw/vllm_issue_23641.html`; **Closed as not planned**, labels `feature request`, `stale`; **never implemented** | `compute_cost = cost_factor * cost_func(size)` and `We should evict a prefix if its retention benefit is smallest, that is to say it has the minimum ` + backticked `freq * compute_cost`. | Exact `freq × compute_cost` eviction for the **vLLM prefix cache**. Explicitly *discards* the time/load factor: "T and cost_factor take no effect because we just use the benefit for comparing, so we can ignore it." `cost_func = size^alpha`. So: cost-priced eviction proposed for the exact target engine, with **load explicitly factored out**, and **not implemented**. |
| 3 | **PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving** — https://arxiv.org/abs/2607.02525 | Preprint, abs page fetched by me → `raw/arxiv_2607.02525_abs.html`, title read back ✓ (`citation_title" content="PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving"`, date 2026/05/10); full HTML fetched → `raw/arxiv_2607.02525_html.html`; has SGLang+vLLM experiments up to 4×H100 | `raw/_2607_02525.txt` line 1465: `A queue-aware policy inspects pending requests, sees A is needed and B is not, and evicts B instead—preserving A for an immediate hit.` (also RAW-OK in `raw/arxiv_2607.02525_html.html`) | **The live-state half of the candidate, without the cost half.** Its eviction hook is driven by an auxiliary radix tree over the **pending queue** — it protects blocks ancestral to *queued demand*. Victim choice depends on live scheduler state, but the criterion is demand/presence, **not** a priced marginal recompute cost. |
| 4 | **vLLM 0.29-class scheduler source (victim selection)** — https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py | Source file fetched by me → `raw/vllm_scheduler.py` (115 KB) | line 747: `# Preempt the lowest-priority request.` · line 754: `preempted_req = self.running[-1]` | **Implementation reality in the target engine.** vLLM picks the preemption/recompute victim by **lowest priority / last-in-running (FCFS)** — recompute cost is not priced at all. No `P(reuse)`, no cost term. |
| 5 | **vLLM KV block eviction (prefix cache)** — https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/kv_cache_utils.py | Source file fetched by me → `raw/vllm_kv_cache_utils.py` | lines 255–257: `and then freed, it will be appended back with the eviction order:` / `1. The least recent used block is at the front (LRU).` | Confirms the target engine's block evictor is **plain LRU + tail-first tiebreak** — exactly the "LRU" the candidate proposes to replace. |
| 6 | **SGLang Radix Cache Eviction Policies (vendor doc)** — https://docs.sglang.io/docs/advanced_features/radix_eviction_policy | Vendor documentation page fetched by me → `raw/sglang_radix_eviction.html` | `Policy Evicts first Use when lru (default) The prefix unused for longest.` … `priority The prefix belonging to the lowest-priority request, then the least recent.` and `A policy only scores. It does not decide how much to free` | The competing engine ships `lru / lfu / slru / priority`. `priority` follows **request priority**, not load-priced recompute cost. No cost term anywhere in the documented scoring inputs (`Hit count`, `Request priority`). |
| 7 | **vLLM RFC #37003 "Context-Aware KV-Cache Retention API (Prioritized Evictions)"** — https://github.com/vllm-project/vllm/issues/37003 | GitHub issue fetched by me → `raw/vllm_issue_37003.html` | `The orchestrator defines policy; vLLM executes it.` … `Every system that exploits reuse structure, recomputation cost, or workflow topology beats LRU substantially.` | Shows the community is aware of "recomputation cost is invisible to LRU", but the proposed mechanism is an **external priority/TTL directive API** (orchestrator-supplied), not a load-priced eviction rule. It cites MARCONI, Continuum, KVFlow, and the Alibaba trace study as the cost-aware prior art — useful as *their* occupancy map. |
| 8 | **Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints** — https://arxiv.org/abs/2504.11320 | Preprint v4 (2026/06/13), abs page fetched by me → `raw/arxiv_2504.11320_abs.html`, title read back ✓; full HTML fetched → `raw/arxiv_2504.11320_html.html` | `raw/_2504_11320.txt`: `the scheduler must control the composition of the GPU-resident workload across prefill and decode stages when iteration time is memory dependent and overflow causes eviction and restart` | **The "live batch/phase mix determines prefill cost" modelling half already exists** — their iteration-time model (Eq. 1) is an explicit function of the *mixed prefill/decode batch composition*, with a worked example of prefill-only / mixed / decode-only batches. But they use this price for **admission and batch composition**, never for eviction victim selection; their eviction is treated as an overflow event ("overflow can evict in-progress requests"), not a priced decision. |
| 9 | **TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling** — https://arxiv.org/abs/2510.02758 | Preprint, abs page fetched by me → `raw/arxiv_2510.02758_abs.html`, title read back ✓ | (search snippet, not used as a quote) the paper's I/O cost model is `t_IO = t_evict_queueing + t_evict + t_load_queueing + t_load` | Neighbour: it *does* model **queueing-dependent** evict/load times, but for GPU↔CPU KV **transfer scheduling under preemption**, not for choosing an eviction victim on recompute economics. |
| 10 | **Hyperbolic Caching (Blankstein et al., USENIX ATC'17)** — https://www.usenix.net/system/files/conference/atc17/atc17-blankstein.pdf | Peer-reviewed paper PDF fetched by me → `raw/hyperbolic_caching_atc17.pdf`; text extracted → `raw/_hyperbolic.txt`. **Caveat: this PDF's text layer is ligature-split and control-character-laden, so no clean prose quote exists.** The greppable fragment is given verbatim below. | `(it) (attempts) (to) (incorporate) (cost) (into) (LR) (U,)` — which reads, in the paper's prose, "…it attempts to incorporate cost into LRU, requiring a re-design." | Classical cost-aware caching lineage generalising GreedyDual/LRU/Frequency; confirms the cost-aware-caching family. Its utility mapping is **not load-dependent**, but I flag it here as **quote-verified only at fragment level** — treat this row as weaker evidence than rows 1–9. |

**Families I checked that are NOT occupants** (different mechanism): H2O / SnapKV / Scissorhands / TOVA / "heavy hitter" are
**token-level, attention-score** KV pruning policies chosen for *accuracy under a budget*, not block-level victim selection under
memory pressure, and none price recomputation. I did not find a load-priced eviction variant in that family either.

## 3. Nearest-neighbour sentence + kill condition (candidate is PARTIALLY occupied)

> **Nearest neighbour X does A; under condition C it misses B; we do B, therefore when C holds the conclusion differs.**

Nearest neighbour **Marconi (arXiv 2411.19379) does A = evict by `recency-decayed reuse-likelihood × (compute savings per byte)`,
and GreedyDual-Size (USITS'97) does A = evict by `recency-decayed (retrieval cost / size)`; under condition C = the marginal cost
of recomputing a block depends on the current (batch, phase-mix) state — which the target box exhibits, since its non-bandwidth
component climbs 2.79 ms → 12.72 ms from n=1 to n=64 (slope ≈ 0.158 ms/sequence/step) and the box crosses from bandwidth-bound to
compute-bound at batch ≈ 28 — they miss B = a recompute price that is a **function of live scheduler load** rather than a static
per-object FLOP/size constant; we do B (read the recompute price from the scheduler's current load state), therefore when C holds
the victim ranking differs, and the difference is observable exactly in the regime where the box is near or above crossover.**

**What would kill it (concrete falsification conditions):**
1. Any prefix/KV-cache eviction policy whose eviction score consumes a **runtime-measured, load-dependent** prefill cost —
   e.g. current running-batch size, queue depth, measured step time, or prefill/decode ratio — rather than a static FLOP/byte/size formula.
   (MARCONI and RFC #23641 are static; PEEK is live but unpriced. A paper combining them kills this.)
2. A classical caching paper in which the retrieval cost `c(p, t)` is explicitly a function of the **server's current load or queue state**
   (a "load-priced GreedyDual"). GreedyDual-Size's `c(p)` is per-object; if a variant exists with `c` state-dependent, this is occupied classically.
3. An experiment showing the victim ranking under live pricing is **identical** to MARCONI/GreedyDual-Size ranking on the target box —
   i.e. that the load dependence never changes which block is evicted, making the mechanism behaviourally vacuous.
4. Note as a live risk: **arXiv 2608.23658** ("Elastic KV Cache for LLM Serving: A Working Reclamation Mechanism, and Why Chunked
   Prefill Already Closes the Gap", 2026/08/24, abs page fetched by me → `raw/arxiv_2608.23658_abs.html`) reports a measured
   **negative result** that "the prefill chunk-size penalty is small (median TTFT differs by about 1% between chunk sizes of 8192
   and 32768 tokens), because prefill is compute bound and decode consumes only about one token per sequence per step". If the
   *live* marginal prefill cost barely moves with the batch/phase mix on real hardware, the candidate's price signal may be too
   weak to change eviction decisions — this is a **substantive threat to the mechanism's value**, not to its novelty.

## 4. What I could NOT verify (Candidate 1)

- **ACM SYSTOR, "To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems"** —
  `https://dl.acm.org/doi/10.1145/3793230.3837769` returned **HTTP 403** on three attempts (`/doi/`, `/doi/full/`, `/doi/pdf/`;
  saved 403 bodies at `raw/acm_systor.html`, `raw/acm_systor_kv_retention.html`, `raw/acm_systor_full.html`, `raw/acm_systor_pdf.html`).
  I have **only the search-result title**, no abstract, no body, no quote. **This is the single most likely unexamined occupant of
  Candidate 1** (a *learned* retention policy in a disaggregated serving system could plausibly consume live load features).
  It must be retrieved via a non-ACM route before Candidate 1 is treated as clear.
- **Google Research blog, "Optimizing cloud economics with linear elastic caching"** —
  `https://research.google/blog/optimizing-cloud-economics-with-linear-elastic-caching/` fetched (`raw/google_linear_elastic_caching.html`),
  but the article body is client-side rendered; the fetched HTML contains only site chrome. No text verified. Candidate relevance: cache
  **residency pricing**, i.e. the "residency cost" half of the inequality.
- **IEEE Xplore, "FlowGuard: Slack-Aware Overload Control for Multi-Agent LLM Serving"** (`ieeexplore.ieee.org/document/11662939`) — paywalled, not fetched.
- **`arxiv.org/search` (HTTP 406) and `export.arxiv.org` (HTTP 429)** — unusable, per instruction; not attempted beyond confirming avoidance.
- **`api.github.com`** — rate-limited to zero here; not used, so I could not enumerate vLLM issue/PR metadata programmatically
  (search-then-fetch of specific issue URLs was used instead).
- **`raw.githubusercontent.com` HEAD-of-`main`** — the vLLM source I read is current `main`, **not pinned to the v0.29.0 tag**.
  The victim-selection line (`self.running[-1]`) is stable across recent versions, but I did not verify the v0.29.0 tree byte-for-byte.
- **`arxiv.org/abs/2607.16892` v1** — I fetched the abs page (title ✓) and the HTML body; the HTML body I parsed is 67 KB of
  extracted text (likely the v1 HTML). I did not diff versions.

---

# CANDIDATE 7 — "Admission under jointly-unknown length and KV growth"

**Mechanism under test:** admission control for LLM serving where **both** the output length **and** the resulting KV footprint are
unknown at admission and **grow together**, with the objective being **goodput under a memory cap**; the service requirement (memory)
is revealed *during* service and is **correlated with the duration**.

## 1. VERDICT

### `OCCUPIED`

This is the more heavily occupied of the two. The **specific structural feature — memory requirement not fixed at admission because
the KV cache grows with generated tokens, coupled with unknown response length and a hard memory cap, driving an admission decision —
is explicitly and formally the subject of a 2025–2026 OR/queueing literature**, and two 2026 theory papers state the "jointly unknown
length + growing footprint" structure as their first sentence. The **goodput-under-SLO wording** is separately occupied.
Importantly, the strongest occupants are **classical/OR** artifacts (MIT OR group + a queueing-theory line), which per the occupancy
standard is the strongest kind of occupation.

### Single strongest occupying artifact

**Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints** — https://arxiv.org/abs/2504.11320
(Ruicheng Ao, Gan Luo, David Simchi-Levi, Xinshang Wang — MIT IDSS / PKU / Alibaba Group)

- **State verified by me:** preprint **v4**, arXiv:2504.11320v4 [cs.LG], dated 13 Jun 2026 (updated Aug 24, 2026 rendering);
  abs page fetched by me → `raw/arxiv_2504.11320_abs.html`, title read back ✓
  (`citation_title" content="Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints"`);
  full HTML body fetched by me → `raw/arxiv_2504.11320_html.html` (1.32 MB), tag-stripped to `raw/_2504_11320.txt`.
  Has formal theorems (asymptotic guarantees, high-probability memory bounds) **and** experiments: Vidur simulations for
  Llama-2-7B on A100 on synthetic + `lmsys-chat-1m` traces, plus **real-GPU validation** in appendix H.3 ("Simulator vs GPU",
  "Single-type workload on GPU", "Real dataset on GPU").
- **Verbatim quote** (from `raw/_2504_11320.txt` line 401; the same sentence is RAW-OK in `raw/arxiv_2504.11320_html.html`):

  > `The key departure from classical scheduling is endogenous memory growth: a job’s memory requirement is not fixed at admission, because as the model generates output tokens, the prompt’s Key-Value (KV) cache grows over time.`

- **Second verbatim quote**, from the **abstract block of the saved abs page** `raw/arxiv_2504.11320_abs.html` (same sentence also at
  `raw/_2504_11320.txt` line 276), which is the admission-under-unknown-length claim:

  > `Guided by the fluid model, we design WAIT (Waiting for Accumulated Inference Threshold), a threshold-based admission rule for known output lengths, and Nested WAIT, which extends the rule to unknown output lengths by regulating how requests advance across decode-stage segments.`

  and, later in the same abstract block of `raw/arxiv_2504.11320_abs.html`:

  > `Nested WAIT uses an additional safety buffer of moderate scale to hedge against memory-overflow-induced evictions under unknown output lengths.`

- **Third verbatim quote** — the guarantees sentence appears in the **v4 HTML body**, NOT on the abs page. Grep `raw/_2504_11320.txt`:

  > `for unknown output lengths, Nested WAIT obtains guarantees on throughput, latency, and eviction avoidance under an additional safety buffer`

  (The abs-page abstract phrases the same guarantee differently: `Both algorithms approximate the fluid benchmark asymptotically under the stated memory conditions.` I record both because the wording differs between the abs page and the v4 HTML.)

- **How close:** this is the candidate's mechanism. It is **admission control**, under a **hard GPU-resident KV-cache capacity `C`**,
  with the memory requirement **endogenous to and growing with service**, with the **output length unknown at admission** and learned
  "on-the-fly classification at segment boundaries" (i.e. revealed during service), and it carries **throughput/latency/eviction-avoidance**
  guarantees plus measurements. The one condition it does *not* state in the candidate's exact words is "goodput" as an SLO-attainment
  objective — its benchmark is the **fluid-optimal throughput** and the **fluid stability region**. (Its own related-work paragraph names
  the distinction precisely: "Jaillet et al. (2025) and Wang et al. (2025) assume lengths are known at arrival, Chen et al. (2025) considers
  adversarial unknown lengths, while our Nested WAIT algorithm handles stochastic unknown lengths through on-the-fly classification at
  segment boundaries.") The goodput-under-SLO wording is supplied by the artifact in row 1 below.

## 2. Further artifacts (table)

| # | Artifact / URL | State verified by me | Verbatim quote (raw file to grep) | What it covers |
|---|---|---|---|---|
| 1 | **Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty** — https://arxiv.org/abs/2607.16892 | Preprint, abs page fetched by me → `raw/arxiv_2607.16892_abs.html`, title read back ✓ (`citation_title" content="Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty"`, Cheng/Do/Nguyen, date 2026/07/18); HTML body fetched → `raw/arxiv_2607.16892_html.html`; trace-driven eval on BurstGPT/Azure/ShareGPT | abs page: `the KV cache must be reserved upon request arrival, while the output token length remains unknown until generation completes` · `raw/_2607_16892.txt` line 175: `goodput—the rate of admitted-and-on-time requests` | **Supplies the exact "goodput under a memory cap with unknown length" framing**, with an explicit **rejection decision** (the rejected portion is a decision variable: `π_{i,0} = 1 − Σ_k π_{i,k}`, with a `C_rej` cost term) and SLO-violation metrics. Misses the *online-revelation* part: the decision is a **static per-class reservation quantile** from a Wasserstein-DRO critical fractile, not a rule that updates as the footprint is revealed during service. Preprint. |
| 2 | **Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference** — https://arxiv.org/abs/2601.22996 | Preprint dated 2026/01/30, abs page fetched by me → `raw/arxiv_2601.22996_abs.html`, title read back ✓ (`citation_title" content="Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference"`); full HTML fetched → `raw/arxiv_2601.22996_html.html`; has numerical experiments on real request traces | abs page line 30: `where a job&#39;s memory footprint grows linearly with the number of decoded tokens.` and `yet the response lengths of requests are inherently unknown` | **States the candidate's structural premise as its opening sentence** and gives the first constant-competitive non-clairvoyant policy (GSA, ratio ≤ 61.92; GBA clairvoyant counterpart). Misses: objective is **total completion time / makespan** in an offline batch setting (plus an online-arrivals extension), with **no admission/rejection decision** — jobs are restarted, not refused, and there is no goodput/SLO objective. |
| 3 | **General Non-Clairvoyant KV-Cache Scheduling via Regime-Aware Routing** — https://arxiv.org/abs/2607.09248 | Preprint dated 2026/07/10, abs page fetched by me → `raw/arxiv_2607.09248_abs.html`, title read back ✓ (`citation_title" content="General Non-Clairvoyant KV-Cache Scheduling via Regime-Aware Routing"`); full HTML fetched → `raw/arxiv_2607.09248_html.html` | `raw/_2607_09248.txt` line 179: `Each request has a known prompt length but an unknown response length, and its memory footprint comprises a fixed prompt component together with a response component that grows with each decoded token.` (RAW-OK in the saved HTML) | Same structural premise, extended to **online arrivals** with makespan + total-completion-time guarantees. Contains a `Greedy admission scan until no more jobs can be accepted` (line 984) — but that scan admits **attempts into memory**, not requests into service with refusal. **Zero occurrences of "goodput" and "reject"** in the fetched body (I grepped: `goodput` 0, `reject` 0). |
| 4 | **Learning to Admit Optimally in an M/M/k/k+N Queueing System with Unknown Service Rate** — https://arxiv.org/abs/2202.02419 | Preprint dated 2022/02/04, abs page fetched by me → `raw/arxiv_2202.02419_abs.html`, title read back ✓ | abs page line 30: `we study admission control for such systems with unknown service rate` … `Every served job yields a fixed reward but incurs a per unit time holding cost` | **The classical/OR neighbour.** Admission control (block-or-admit) maximizing long-run average reward with a **learned** policy and finite-time regret bounds, under **unknown service requirement**. Misses: the service requirement is a **rate** (a property of the job, sampled once), **not a memory footprint that grows with the job's own progress**, and there is **no correlation between footprint and duration** — exactly the structural feature the candidate claims. |
| 5 | **Scorpio: Serving Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference** — https://arxiv.org/abs/2505.23022 | Preprint, abs page fetched by me → `raw/arxiv_2505.23022_abs.html`, title read back ✓ (`citation_title" content="Scorpio: Serving Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference"`) | abs page: `Our core insight is to exploit SLO heterogeneity for adaptive scheduling across admission control, queue management, and batch selection.` … `a TTFT Guard, which employs least-deadline-first reordering and rejects unattainable requests` | Occupies **"admission control + goodput + SLO" for LLM serving** with experiments (up to 14.4× goodput). Misses the *joint unknown length/KV-growth* feature — its admission rule is deadline-based, and it does not model the memory footprint as jointly revealed. |
| 6 | **CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control** — https://arxiv.org/abs/2601.22705 | Preprint, abs page fetched by me → `raw/arxiv_2601.22705_abs.html`, title read back ✓ (`citation_title" content="CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control"`) | abs page line 31: `We argue that mitigating this pathology requires moving beyond reactive, request-level cache management to proactive, agent-level admission control.` … `CONCUR adapts a cache-aware control algorithm to dynamically adjust the number of active agents using runtime cache signals.` | Occupies **live-cache-signal-driven admission control** with experiments (4.09× on Qwen3-32B). Misses: it regulates **agent-level concurrency** (a closed-loop admission *count*), not per-request admission under an unknown length/footprint; no output-length uncertainty model; no goodput-under-SLO objective. |
| 7 | **FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference (USENIX NSDI '26)** — https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang | Venue page fetched by me → `raw/fastserve_nsdi26.html`; title read back ✓ from `<title>` and `citation_title` meta | (page is a venue/abstract landing page; I did not fetch the paper body) | Places the **preemption/scheduling** line (FastServe, and the semi-information scheduling lineage the candidate names) in a peer-reviewed 2026 venue. Adjacent to Candidate 7 but not an admission-control-under-unknown-footprint occupant on the evidence I fetched. |
| 8 | **Elastic KV Cache for LLM Serving: A Working Reclamation Mechanism, and Why Chunked Prefill Already Closes the Gap** — https://arxiv.org/abs/2608.23658 | Preprint dated 2026/08/24, abs page fetched by me → `raw/arxiv_2608.23658_abs.html`, title read back ✓ | abs page: `Our elastic KV cache lends the reserve to the KV pool during decode and returns it before prefill, driven by the scheduler's one-step-ahead view of the next batch.` | Neighbour on the **memory-cap-side** of Candidate 7 (elastic KV capacity driven by scheduler lookahead) with a **negative result** and real measurements. It does not do admission control; included because its finding bears on how much headroom a memory-cap admission rule can actually win. |
| 9 | **InferCept: Efficient Intercept Support for Augmented Large Language Model Inference** — https://arxiv.org/abs/2402.01869 | Preprint, abs page fetched by me → `raw/arxiv_2402.01869_abs.html`, title read back ✓ | abs page: `This paper presents InferCept, the first LLM inference framework targeting augmented LLMs and supporting the efficient interception of LLM generation.` | Checked as a possible "unknown-length admission" occupant; it is a **recomputation-avoidance** system (37–40% of forwarding time is recomputation), not admission control. Non-occupant; logged to record the negative. |
| 10 | **MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing** — https://arxiv.org/abs/2506.02006 | Preprint dated 2025/05/24, abs page fetched by me → `raw/arxiv_2506.02006_abs.html`, title read back ✓ | abs page: `pressure-aware KV cache resizing, which dynamically adjusts KV cache capacity in response to memory pressure` | Neighbour: load-reactive KV **capacity** adaptation (reduces SLO violations 92.45%). Not admission control, no unknown-output-length model. Non-occupant; logged to record the negative. |

## 3. Nearest-neighbour sentence + kill condition (candidate is OCCUPIED, so stated for completeness)

> **Nearest neighbour X does A; under condition C it misses B; we do B, therefore when C holds the conclusion differs.**

Nearest neighbour **Ao–Luo–Simchi-Levi–Wang (arXiv 2504.11320) does A = threshold-based admission under a hard GPU-resident KV
capacity where the memory requirement "is not fixed at admission" and grows with generated tokens, extended by Nested WAIT to
**unknown output lengths** with on-the-fly classification; under condition C = the objective is stated as **goodput subject to
latency SLOs** rather than fluid-optimal throughput plus eviction avoidance, it misses B = an explicit goodput-under-SLO admission
objective; we do B, therefore when C holds the conclusion differs.** This residual is narrow and is itself already supplied for LLM
serving by **2607.16892** (goodput + SLO + rejection cost + unknown output length + KV reserved at admission) and **2505.23022**
(goodput + admission control + rejection). **On the evidence I fetched, no single artifact holds all three of {online admission with
footprint revealed during service} × {explicit goodput-under-SLO objective} × {endogenous KV growth} simultaneously — that union is the
only surviving sliver, and the lead should treat it as fragile.**

**What would kill it (concrete falsification conditions):**
1. **A single paper combining the two halves.** Any paper with an admission/rejection decision, an explicit goodput-under-SLO objective,
   and an endogenous-KV-growth-with-unknown-length model kills the union sliver. 2504.11320 is one section away from doing so; if a
   v5 adds a goodput/SLO objective, Candidate 7 is fully occupied with no residual.
2. **The ~4 papers named in 2504.11320's own related work that I did NOT retrieve** — Jaillet et al. (2025), Wang et al. (2025),
   Chen et al. (2025), and Li et al. (2025b)/Bari et al. (2025). These are the OR line that 2504.11320 positions against; a goodput
   admission objective is very plausibly already in one of them. **This is the highest-value unretrieved set for Candidate 7.**
3. **An experiment showing the "correlated memory-and-duration" structure yields no decision-theoretic difference** from a classical
   admission rule that assumes a known/i.i.d. service requirement on the target box (e.g. because KV/token ≈ 147 KB and the output-length
   distribution is well-approximated by a known quantile). If the correlation never changes the admit/reject boundary, the novelty claim
   is behaviourally vacuous.

## 4. What I could NOT verify (Candidate 7)

- **Jaillet et al. (2025), Wang et al. (2025), Chen et al. (2025), Li et al. (2025b), Bari et al. (2025)** — cited *inside*
  arXiv 2504.11320's related work as the online-algorithm/OR line for LLM inference scheduling under known/adversarial/predicted output
  lengths. I have only the in-paper citations, **no arXiv IDs I could verify**; I deliberately did not guess IDs. These are the most
  important unretrieved artifacts for Candidate 7.
- **`arxiv.org/search` (HTTP 406)** — unusable, so citation-graph expansion from 2504.11320 could not be automated.
- **`api.github.com` (rate-limited to zero)** — no programmatic GitHub metadata.
- **PreServe** (`arxiv.org/abs/2504.03702`) and **FlowGuard** (`ieeexplore.ieee.org/document/11662939`, paywalled) — surfaced by search
  as overload-control/LLM-serving neighbours; not fetched, so I make no claim about them.
- **SURGE** and **"semi-information scheduling"** — the candidate brief asked me to check these; my queries for them did not return an
  identifiable artifact I could fetch and title-verify, so I have **no verified finding** either way. Same for **MuPS** and
  **"safety-valve admission control"**: searches returned only blogs/vendor pages, nothing I could occupy with.
- **FastServe full paper body** — only the NSDI '26 venue page was fetched; the paper text is not in my raw set.

---

# Verification ledger (every arXiv ID I cite, read back this session)

| arXiv ID | `citation_title` read back from `arxiv.org/abs/<id>` | Saved abs file | Match? |
|---|---|---|---|
| 2504.11320 | `Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints` | `raw/arxiv_2504.11320_abs.html` | ✓ |
| 2607.16892 | `Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty` | `raw/arxiv_2607.16892_abs.html` | ✓ |
| 2607.02525 | `PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving` | `raw/arxiv_2607.02525_abs.html` | ✓ |
| 2411.19379 | `Marconi: Prefix Caching for the Era of Hybrid LLMs` | `raw/arxiv_2411.19379_abs.html` | ✓ |
| 2601.22996 | `Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference` | `raw/arxiv_2601.22996_abs.html` | ✓ |
| 2607.09248 | `General Non-Clairvoyant KV-Cache Scheduling via Regime-Aware Routing` | `raw/arxiv_2607.09248_abs.html` | ✓ |
| 2202.02419 | `Learning to Admit Optimally in an $M/M/k/k+N$ Queueing System with Unknown Service Rate` | `raw/arxiv_2202.02419_abs.html` | ✓ |
| 2505.23022 | `Scorpio: Serving Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference` | `raw/arxiv_2505.23022_abs.html` | ✓ |
| 2601.22705 | `CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control` | `raw/arxiv_2601.22705_abs.html` | ✓ |
| 2402.01869 | `InferCept: Efficient Intercept Support for Augmented Large Language Model Inference` | `raw/arxiv_2402.01869_abs.html` | ✓ |
| 2608.23658 | `Elastic KV Cache for LLM Serving:A Working Reclamation Mechanism, and Why Chunked Prefill Already Closes the Gap` | `raw/arxiv_2608.23658_abs.html` | ✓ |
| 2506.02006 | `MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing` | `raw/arxiv_2506.02006_abs.html` | ✓ |
| 2510.02758 | `TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling` | `raw/arxiv_2510.02758_abs.html` | ✓ |
| 2511.02230 | `Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live` | `raw/arxiv_2511.02230_abs.html` | ✓ (not used as occupant; cited by vLLM RFC #37003) |
| 2506.02634 | `KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider` | `raw/arxiv_2506.02634_abs.html` | ✓ (not used as occupant) |
| 2507.07400 | `KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows` | `raw/arxiv_2507.07400_abs.html` | ✓ (not used as occupant) |

**Defect log (search-engine output I refused to cite):**
- `web_search` returned **`browse-export.arxiv.org/pdf/2607.16892`** and **`arxiv-org.ezproxy.obspm.fr/html/...`** as result URLs.
  I ignored both and fetched `arxiv.org/abs/...` / `arxiv.org/html/...` directly. No fabricated-ID incident occurred this session,
  but the ezproxy/browse-export surfaces are unreliable proxies and were not used as evidence.
- I did **not** encounter or cite any `thaki-AI/daily-paper-*` artifact.
- Searches surfaced **`www.emergentmind.com/papers/...`**, **`ar5iv.labs.arxiv.org`**, and **`alphaxiv.org`** mirrors. I used none of
  them as the source of a quote; every quote above comes from a primary URL I fetched (`arxiv.org`, `usenix.org`, `docs.sglang.io`,
  `github.com`, `raw.githubusercontent.com`).

---

# Search log

## Exact `web_search` queries issued (verbatim, in order)

1. `KV cache eviction policy accounts for prefill load recomputation cost LLM serving`
2. `GreedyDual-Size cache replacement retrieval cost`
3. `vLLM prefix cache eviction policy LRU recompute cost`
4. `admission control LLM serving unknown output length KV cache memory goodput`
5. `"recompute cost" aware KV cache eviction prefill decode interference`
6. `"admission control" LLM serving "unknown output length" memory cap`
7. `size-based admission control unknown job sizes queueing`
8. `SGLang RadixAttention eviction policy prefix cache`
9. `admission control LLM serving SLO unknown output length goodput memory`
10. `overload control LLM serving request dropping protect SLO`
11. `semi-information scheduling unknown job size LLM FastServe`
12. `GreedyDual-Size cost-aware cache replacement retrieval cost`
13. `KV cache eviction policy prefill-aware cost of recomputation live load`
14. `"evict or recompute" KV cache decision LLM serving`
15. `load-aware cache eviction scheduling state LLM inference`
16. `prefix cache eviction policy accounts for current prefill queue load`
17. `MARCONI KV cache eviction cost-aware recomputation`
18. `KV cache eviction policy scheduling-aware marginal cost live load pricing`
19. `admission control GPU inference memory-aware reject requests goodput`
20. `"safety valve" admission control LLM inference serving`
21. `prefill decode interference aware eviction policy KV cache LLM`
22. `KV cache eviction priced by current prefill queue latency`
23. `cache replacement policy dynamic retrieval cost varies with system load`
24. `"recomputation cost" "live" load state eviction LLM serving 2026`
25. `admission control without knowing service time queueing theory size-based`
26. `SURGE SLO-aware scheduling LLM serving semi-information unknown output length`
27. `MuPS multi-model serving admission control memory goodput`
28. `goodput maximization admission control unknown job size memory constraint`
29. `LLM serving admission control jointly unknown output length and KV cache growth`
30. `goodput SLO memory cap admission control generative inference 2026`
31. `request scheduling unknown memory footprint grows during service queueing`
32. `admission control endogenous memory growth unknown service requirement`
33. `eviction policy recompute cost depends on current batch composition LLM serving`
34. `KV cache retention value live marginal compute cost prefill decode mix`
35. `non-clairvoyant scheduling unknown memory footprint LLM`
36. `admission control capacity unknown service requirement grows with job duration`
37. `SLO-aware admission control LLM serving reject requests protect goodput paper`
38. `"admission control" "KV cache" reject request LLM inference memory cap experiment`
39. `unknown output length SLO LLM serving admission rejection 2026`
40. `KV cache eviction policy uses system load state dynamic pricing`
41. `"load-aware" OR "state-dependent" cache eviction LLM inference server`
42. `H2O SnapKV TOVA heavy hitter KV cache eviction attention score memory pressure`
43. `"prefill" "decode" interference KV cache eviction policy decision 2026`
44. `"To Keep or Not to Keep" KV cache retention disaggregated LLM serving`
45. `cache eviction policy recomputation cost system load aware classical caching theory`
46. `"cost-aware" cache eviction policy where cost varies over time server load`
47. `admission control known service requirement assumption classical queueing relaxation`
48. `eviction decision priced by scheduler load state recompute cost varies batch mix`
49. `KV cache eviction policy where recompute cost depends on current prefill batch`
50. `queue-aware cost-aware eviction LLM serving recomputation priced live`
51. `KV cache eviction recompute cost depends on current batch composition serving`
52. `cache eviction policy state dependent fetch cost varies with server load queueing`

(One `web_search` call mid-session returned a transport error — `DeepSeek returned an unprocessable response body: TypeError: terminated`
— for the batch containing queries 48–50; it was re-issued as queries 51–52.)

## Exact URLs I fetched (all via `curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "<Chrome UA>"`)

**HTTP 200 and quoted above**
1. `https://github.com/vllm-project/vllm/issues/23641` → `raw/vllm_issue_23641.html`
2. `https://github.com/vllm-project/vllm/issues/37003` → `raw/vllm_issue_37003.html`
3. `https://arxiv.org/abs/2504.11320` → `raw/arxiv_2504.11320_abs.html`
4. `https://arxiv.org/html/2504.11320v4` → `raw/arxiv_2504.11320_html.html`
5. `https://arxiv.org/abs/2607.16892` → `raw/arxiv_2607.16892_abs.html`
6. `https://arxiv.org/html/2607.16892` → `raw/arxiv_2607.16892_html.html`
7. `https://arxiv.org/abs/2607.02525` → `raw/arxiv_2607.02525_abs.html`
8. `https://arxiv.org/html/2607.02525` → `raw/arxiv_2607.02525_html.html`
9. `https://arxiv.org/abs/2411.19379` → `raw/arxiv_2411.19379_abs.html`
10. `https://arxiv.org/html/2411.19379v3` → `raw/arxiv_2411.19379_html.html`
11. `https://arxiv.org/abs/2601.22996` → `raw/arxiv_2601.22996_abs.html`
12. `https://arxiv.org/html/2601.22996` → `raw/arxiv_2601.22996_html.html`
13. `https://arxiv.org/abs/2607.09248` → `raw/arxiv_2607.09248_abs.html`
14. `https://arxiv.org/html/2607.09248` → `raw/arxiv_2607.09248_html.html`
15. `https://arxiv.org/abs/2202.02419` → `raw/arxiv_2202.02419_abs.html`
16. `https://arxiv.org/abs/2505.23022` → `raw/arxiv_2505.23022_abs.html`
17. `https://arxiv.org/abs/2601.22705` → `raw/arxiv_2601.22705_abs.html`
18. `https://arxiv.org/abs/2402.01869` → `raw/arxiv_2402.01869_abs.html`
19. `https://arxiv.org/abs/2608.23658` → `raw/arxiv_2608.23658_abs.html`
20. `https://arxiv.org/abs/2506.02006` → `raw/arxiv_2506.02006_abs.html`
21. `https://arxiv.org/abs/2510.02758` → `raw/arxiv_2510.02758_abs.html`
22. `https://arxiv.org/abs/2511.02230` → `raw/arxiv_2511.02230_abs.html`
23. `https://arxiv.org/abs/2506.02634` → `raw/arxiv_2506.02634_abs.html`
24. `https://arxiv.org/abs/2507.07400` → `raw/arxiv_2507.07400_abs.html`
25. `https://docs.sglang.io/docs/advanced_features/radix_eviction_policy` → `raw/sglang_radix_eviction.html`
26. `https://www.usenix.org/legacy/publications/library/proceedings/usits97/full_papers/cao/cao_html/node8.html` → `raw/gds_usenix_node8.html`
27. `https://www.usenix.org/legacy/publications/library/proceedings/usits97/full_papers/cao/cao_html/cao.html` → `raw/gds_cao_irani_full.html`
28. `https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang` → `raw/fastserve_nsdi26.html`
29. `https://www.usenix.net/system/files/conference/atc17/atc17-blankstein.pdf` → `raw/hyperbolic_caching_atc17.pdf`
30. `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py` → `raw/vllm_scheduler.py`
31. `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/kv_cache_utils.py` → `raw/vllm_kv_cache_utils.py`
32. `https://research.google/blog/optimizing-cloud-economics-with-linear-elastic-caching/` → `raw/google_linear_elastic_caching.html` (body unverified, JS-rendered)

**Non-200 / unusable**
33. `https://dl.acm.org/doi/10.1145/3793230.3837769` → **403** → `raw/acm_systor_kv_retention.html`
34. `https://dl.acm.org/doi/full/10.1145/3793230.3837769` → **403** → `raw/acm_systor_full.html`
35. `https://dl.acm.org/doi/pdf/10.1145/3793230.3837769` → **403** → `raw/acm_systor_pdf.html`

## Derived (tag-stripped) text files I created for grep-ability

`raw/_2504_11320.txt`, `raw/_2607_16892.txt`, `raw/_2607_02525.txt`, `raw/_2607_09248.txt`,
`raw/_2411_19379.txt`, `raw/_hyperbolic.txt`. Each is a mechanical strip of tags/entities from the corresponding saved raw HTML;
every quote attributed to one of these was additionally confirmed to be present in the saved raw HTML itself.
