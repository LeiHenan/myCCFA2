# S3b — Adversarial Prior-Art / Occupancy Check: Candidates 1 and 7

Agent: delegated S3b subagent (`70db0ba3-0c65-4fd9-8d30-55861c44dc78` is parent).
Workspace: `/Users/leihenan/Desktop/myProject`. All raw HTML saved under `.s3b_mech7/raw/`.
Every arXiv ID I cite had its title read back from `arxiv.org/abs/<id>` or from a saved abs page **in this session**;
three IDs (2410.21266, 2606.16824, 2607.19214) arrived on **abs pages the lead had already fetched**, and I read their titles back
from those saved files rather than re-fetching (see §Verification ledger).
`api.github.com` was never used. `export.arxiv.org` and `arxiv.org/search` were never used.

**Note on the raw directory:** `.s3b_mech7/raw/` already contained ~130 files from other subagents (and the lead) when I started.
Files I fetched in this session are listed in the Verification ledger at the bottom; quotes point only at files I fetched,
**except** for the three lead-supplied abs pages named in the ledger, which are marked as such.

**HOW TO RE-VERIFY EVERY QUOTE (I ran this myself; 44/44 pass in the current file):**
All quotes are `grep -F`-verifiable against the named raw file. For the `.txt` derivatives, the `.py`/`.md`/`.json` sources and the
plain-text files, `grep -F '<quote>' <file>` works directly. For the HTML/XML-ish files (`*.html`), a few quotes straddle a
newline, so use:

```sh
tr '\n' ' ' < raw/<file>.html | grep -F '<quote>'
```

For the two KVLearn quotes the named files are `raw/ss_systor_api.json` (the abstract record — it is one long JSON line, so plain
`grep -F` works) and `raw/kvlearn_readme.md` (a fenced code block — plain `grep -F` works).
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

### `PARTIALLY OCCUPIED` — **borderline; treat as effectively OCCUPIED unless a measurable delta is demonstrated**

The **inequality form is classical and is already instantiated for KV/prefix caches**, and — as of the KVLearn finding below — a
**peer-reviewed, open-source, experimentally-evaluated** system already ships `P(reuse) × (recompute cost) − residency cost > θ`
with **online-learned P(reuse)**, a **regime-aware recompute price that has the same bandwidth-bound → compute-bound crossover that
motivates this candidate**, and a **closed-loop controller on live pool pressure**. The candidate's stated distinguishing feature —
*the recompute price is read from the scheduler's current load state rather than a static/offline estimate* — is strictly **still not
done**: KVLearn's `R(b)` is a calibrated function of prefix length, and its live signal enters the *residency* term and the
*threshold*, not the recompute price. But that residual is now razor-thin and must be defended empirically, not rhetorically.

### Single strongest occupying artifact

**KVLearn — "To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems"**, SYSTOR 2026 (ACM),
DOI [10.1145/3793230.3837769](https://doi.org/10.1145/3793230.3837769).

- **State verified by me — retrieved via a NON-ACM route, because `dl.acm.org` returns HTTP 403 from this box.** I obtained:
  (a) **full abstract + metadata** via the Semantic Scholar Graph API,
  `https://www.semanticscholar.org/api/1/paper/6db782df823d9da204a59180305aba81d47b3997` → saved `raw/ss_systor_api.json`
  (HTTP 200, 50 KB; the abstract lives in the JSON field `paper.paperAbstract`; `openAccessInfo.status = "GOLD"`, `license = "CCBY"`,
  `pubDate = "2026-09-02"`, venue `Proceedings of the 19th ACM International Systems and Storage Conference`);
  (b) the **authors' own published cost model** via their artifact repo —
  `https://raw.githubusercontent.com/FastLM/KVLearn/main/README.md` → saved `raw/kvlearn_readme.md` (HTTP 200), and the repo landing
  page `https://github.com/FastLM/KVLearn` → saved `raw/kvlearn_github.html` whose `<title>` reads
  `[SYSTOR 2026] To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems` (independent title confirmation).
  **Peer-reviewed venue paper, GOLD open access, with a public code artifact.**
  **I did NOT obtain the paper's full text** (see §4) — the quotes below are from the abstract record and the authors' code repo.

- **Verbatim quote 1** — the mechanism, from `raw/ss_systor_api.json` (grep `translates reuse probability into a keep/admit signal`):

  > `(ii) a Cost-Aware Retention Score (CARS) that translates reuse probability into a keep/admit signal by accounting for per-block recompute, transfer, and storage costs; and (iii) an Adaptive Threshold Controller (ATC) that adjusts the admission threshold online using closed-loop feedback from observed hit rates and memory pressure.`

  Corroborating, same JSON (grep `invalidates a core assumption of classical cache policies`):

  > `This architectural shift invalidates a core assumption of classical cache policies: that the cost of a miss is simply recomputation on the same device.`

- **Verbatim quote 2 — the exact cost model, from the authors' README** `raw/kvlearn_readme.md` (grep `CARS(b) = `). This is what makes
  it the decisive occupant:

  > `CARS(b) = P̂(b)·(R(b)−T(b)) − U(b,Δt)`

  with, in the same file's "Cost model" block:

  > `R(b)  ≈ α_bw·L          if L < L×   (bandwidth-bound)`
  > `      ≈ α_flop·L²       if L ≥ L×   (compute-bound)`
  > `U(b)  = γ · |b| · Δt    γ = R̄·λ / M_S`
  > `KEEP  ⇔  CARS > θ       (θ from ATC)`

  and, in the same file's defaults block:

  > `A100 + LLaMA-3-8B calib: `α_bw≈0.026 ms/tok`, `α_flop≈8e−6 ms/tok²`, `L×=512``

  and, in the same file, the per-length policy consequence:

  > `For super-linear prefill (`q>1`), the optimal reuse threshold `P*_H` **decreases** with prefix length — longer visual blocks admit at lower predicted reuse.`

  and the ATC component row (line 7 of `raw/kvlearn_readme.md`):

  > `| **ATC** | Adaptive Threshold Controller — adapts `θ` from pool pressure and hit rate |`

- **How close — and why this is the decisive artifact.** KVLearn's eviction/admission rule *is* the candidate's inequality,
  `P(reuse) × (cost of a miss) − residency cost`, with an online-learned reuse probability (`P̂(b) = f_θ(x(b))`, "updated online from
  delayed reuse labels") and a **regime-aware recompute price `R(b)` that switches from `α_bw·L` to `α_flop·L²` at a crossover
  `L× = 512` tokens**. That is **structurally the same bandwidth-bound → compute-bound crossover that motivates this candidate's box**
  (measured crossover at batch ≈ 28). Its **residency term is explicitly load-dependent** — `γ = R̄·λ / M_S` carries the arrival rate λ
  and the pool size `M_S` — and its threshold is driven by **closed-loop feedback from live pool pressure and hit rate**.
  **The one thing it does not do:** its recompute price `R(b)` is a **calibrated structural function of prefix length**
  (`α_bw`, `α_flop` are fixed per-device calibration constants), **not a read of the instantaneous (batch, prefill/decode-phase-mix)
  schedule state**. The candidate's live signal and KVLearn's live signal differ in *where they enter the decision* — KVLearn puts
  liveness in the residency term and the threshold; the candidate puts it in the recompute price itself.

- **Why I demoted the previous strongest occupant (Marconi) to a table row:** MARCONI prices recomputation with a *static* FLOP-per-byte
  ratio and fits its balance weight offline; KVLearn is closer on the distinguishing feature (regime-aware price + online loop +
  live-pressure feedback), peer-reviewed at SYSTOR 2026, and open-sourced. Marconi remains a relevant row (see the table) but is no
  longer the strongest occupant.

## 1b. THE CRUX, ADJUDICATED: live shadow price vs. learned/sampled scalar

The adjudication the lead asked for, stated as a three-way contrast over the *recompute-price* term specifically:

| Artifact | What its price term actually is | Live? |
|---|---|---|
| **GreedyDual-Size** (USITS'97) | `c(p)` = a **static per-object fetch cost** (download latency / network cost / 1) | **No** — fixed per object |
| **Online Weighted Paging with Unknown Weights** (NeurIPS 2024, [2410.21266](https://arxiv.org/abs/2410.21266)) | a **per-page scalar weight learned by repeatedly sampling the fetch cost** | **Learned, not contemporaneous** |
| **KVLearn** (SYSTOR 2026) | `R(b) = α_bw·L` or `α_flop·L²` — **calibrated constants × prefix length**, with a bandwidth→compute crossover | **Price: no. Residency + threshold: yes** (`γ = R̄·λ/M_S`; ATC tracks pool pressure) |
| **Candidate 1** | the price itself is read from the **current scheduler load state** (batch size / prefill-decode mix) | **Yes — this is the whole claim** |

**Verdict on the crux:** the candidate is *not* the same as "learned weights" (2410.21266 learns a scalar; the candidate reads a
contemporaneous system state) and it is *not* the same as "static cost" (GreedyDual-Size/MARCONI). But **KVLearn already has a
live load term and a live control loop in the same decision**, which means the candidate can no longer claim liveness *per se* —
only liveness **of the recompute price specifically**, and it must show that this changes decisions beyond what
`γ = R̄·λ/M_S` plus a pressure-tracking threshold already achieves.

- **Second-strongest / still-relevant occupant — Marconi: Prefix Caching for the Era of Hybrid LLMs** —
  https://arxiv.org/abs/2411.19379 (also fetched as full HTML: https://arxiv.org/html/2411.19379v3).
  I keep its detail here because it is the artifact the lead is most likely to be challenged on after KVLearn.
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
| 10 | **Online Weighted Paging with Unknown Weights** (NeurIPS 2024) — https://arxiv.org/abs/2410.21266 | **Preprint abs page fetched by the LEAD** (not by me) → `raw/abs_2410.21266.html`; I read the title back from that saved file: `citation_title" content="Online Weighted Paging with Unknown Weights"` ✓, date 2024/10/28; NeurIPS 2024 poster page also fetched by me → `raw/nips24_weighted_paging.html` | `we present the first algorithm for online weighted paging that does not know page weights in advance, but rather learns from weight samples` — and the sentence that fixes the gap: `in multi-level caching architectures, the expected cost of fetching a memory block is a function of its probability of being in a mid-level cache rather than the main memory. This complex property cannot be predicted in advance; over time, however, one may glean information about page weights through sampling their fetching cost multiple times.` | **The nearest CLASSICAL neighbour, and the sharpest statement of the gap.** Weighted paging = `P(reuse) × fetch-cost < residency` with provable `O(log k)` competitiveness (Bansal–Buchbinder–Naor FOCS'07). Here the page **weight IS the recompute/fetch price**, and the 2024 advance is to **learn it from repeated fetch-cost samples**. That is a *per-page scalar learned over time* — **not** a contemporaneous system-state shadow price. Candidate 1 must be stated as the latter, not the former. Peer-reviewed (NeurIPS 2024), theory-only (no systems experiments). |
| 11 | **CacheWise: Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents** — https://arxiv.org/abs/2606.16824 | **Preprint abs page fetched by the LEAD** → `raw/abs_2606.16824.html`; I read the title back ✓ (`citation_title" content="CacheWise: Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents"`, arXiv:2606.16824v1 [cs.DC], 2026/06/15); I additionally fetched the **full HTML body** → `raw/arxiv_2606.16824_html.html`, tag-stripped to `raw/_2606_16824.txt`; implemented in vLLM (~2,500 LOC), evaluated on real CATraces coding-agent traces | `raw/_2606_16824.txt`: `An ideal eviction policy therefore selects j^{*}=\operatorname*{arg\,max}_{j\,\in\,\mathcal{S}_{t},\;j\neq i}\tau_{j}` and `reuse-aware KVCache eviction choosing the block with the highest predicted reuse probability, rather than purely recency-based heuristics like LRU` (also: `CacheWise replaces the default LRU eviction policy`) | **Occupies the `P(reuse)` half only — and it is NOT cost-priced at all.** Its victim rule is pure reuse-*timing* ordering (`argmax τ_j`, a Belady approximation), with `E[τ_i(t)]` refreshed every `N_rebuild = 3` engine iterations. The word "cost" appears only as motivation (`not all evictions are equally costly`), then is *resolved into time* (`evicting from a session with large τ_i is comparatively cheap`) rather than into a price. So: live-refreshed reuse prediction, **zero** recompute-price term. Shipped in vLLM. |
| 12 | **Keeping the Cache Warm Pays: Keepalive Economics for Agentic Workloads** — https://arxiv.org/abs/2607.19214 | **Preprint abs page fetched by the LEAD** → `raw/abs_2607.19214.html`; I read the title back ✓ (`citation_title" content="Keeping the Cache Warm Pays: Keepalive Economics for Agentic Workloads"`, 2026/07/21); I additionally fetched the **full HTML body** → `raw/arxiv_2607.19214_html.html`, tag-stripped to `raw/_2607_19214.txt` | abstract: `the strategy breaks even against a re-prefill at idle ~tau(w/r - 1)` and `since cache residency is priced per read rather than per token-hour, a keepalive-saturated tier gives LRU eviction nothing to rank`; body (`raw/_2607_19214.txt`): `Parameters driving keepalive economics (list prices, July 2026): cached-read ratio r r and re-prefill ratio w w relative to input price` | **A priced eviction-vs-re-prefill break-even analysis — but with a CONSTANT price.** `w` (the re-prefill price) is a **provider list-price ratio** taken from a table (Anthropic w=1.25, w=2.00, etc.), and the horizon `I_max ≈ τ(w/r − 1)` uses constant `w, r`. It is also a **client-side keepalive** strategy, not a server eviction policy. So it prices recomputation but never makes the price load-dependent. Preprint, with a measurement harness across four providers. |

**Also checked but weak evidence (not used as an occupant):** *Hyperbolic Caching* (Blankstein et al., USENIX ATC'17),
`raw/hyperbolic_caching_atc17.pdf` + `raw/_hyperbolic.txt`. Peer-reviewed cost-aware caching lineage generalising
GreedyDual/LRU/Frequency, but I flag it because **this PDF's text layer is ligature-split and control-character-laden, so no clean
prose quote exists** — the only greppable fragment I can offer is `(it) (attempts) (to) (incorporate) (cost) (into) (LR) (U,)`
(reads "…it attempts to incorporate cost into LRU, requiring a re-design."). Its utility mapping is not load-dependent.
**Treat this as weaker evidence than rows 1–12.**

**Families I checked that are NOT occupants** (different mechanism): H2O / SnapKV / Scissorhands / TOVA / "heavy hitter" are
**token-level, attention-score** KV pruning policies chosen for *accuracy under a budget*, not block-level victim selection under
memory pressure, and none price recomputation. I did not find a load-priced eviction variant in that family either.

## 3. Nearest-neighbour sentence + kill condition (candidate is PARTIALLY occupied)

> **Nearest neighbour X does A; under condition C it misses B; we do B, therefore when C holds the conclusion differs.**

Nearest neighbour **KVLearn (SYSTOR 2026) does A = make keep/evict a first-class cost-optimization decision via
`CARS(b) = P̂(b)·(R(b)−T(b)) − U(b,Δt) > θ`, with `P̂` learned online, a **regime-aware recompute price that switches
`α_bw·L → α_flop·L²` at a bandwidth/compute crossover (`L× = 512`)**, a residency term that already carries load
(`γ = R̄·λ/M_S`), and a threshold controller closing the loop on **live pool pressure**; under condition C = the marginal cost of
recomputing a block depends on the current (batch, phase-mix) state rather than only on the block's own length and the pool's
aggregate pressure — which the target box exhibits, since its non-bandwidth component climbs 2.79 ms → 12.72 ms from n=1 to n=64
(slope ≈ 0.158 ms/sequence/step) and the box crosses from bandwidth-bound to compute-bound at batch ≈ 28 — it misses B = a recompute
price that is a **function of the live scheduler phase mix at the decision instant** rather than a calibrated structural constant;
we do B, therefore when C holds the victim ranking differs, and the difference is observable exactly in the regime where the box is
near or above crossover.**

> **Adversarial caveat the lead should carry into any write-up:** this sentence is now doing real work, because KVLearn already
> supplies liveness (λ-loaded residency + pressure-tracking θ) and already supplies a bandwidth↔compute-crossover price. The claim
> therefore narrows to *where* liveness enters. If a reviewer replies "KVLearn's ATC already tracks the load you are describing,"
> the candidate has no answer short of the experiment in kill-condition 3 below.

**What would kill it (concrete falsification conditions):**
1. Any prefix/KV-cache eviction policy whose **recompute-price** term itself consumes a **runtime-measured, load-dependent** prefill
   cost — e.g. current running-batch size, queue depth, measured step time, or prefill/decode ratio — rather than a calibrated
   `length → FLOPs` or `length → bytes` formula. **KVLearn already occupies the residency/threshold half of this; only the price half is open.**
2. A classical caching paper in which the retrieval cost `c(p, t)` is explicitly a function of the **server's current load or queue state**
   (a "load-priced GreedyDual"). GreedyDual-Size's `c(p)` is per-object; weighted paging's weight is *learned*; if a variant exists
   with `c` state-dependent, this is occupied classically.
3. **The decisive experiment the lead must run:** show that re-pricing at the live phase mix changes the **victim ranking** versus
   (a) KVLearn's `CARS` with a fixed calibrated `R(b)` and only its `γ = R̄·λ/M_S` load term, and (b) MARCONI/GreedyDual-Size ranking.
   If the rankings coincide on the target box, the mechanism is behaviourally vacuous and the candidate should be abandoned.
4. Note as a live risk: **arXiv 2608.23658** ("Elastic KV Cache for LLM Serving: A Working Reclamation Mechanism, and Why Chunked
   Prefill Already Closes the Gap", 2026/08/24, abs page fetched by me → `raw/arxiv_2608.23658_abs.html`) reports a measured
   **negative result** that "the prefill chunk-size penalty is small (median TTFT differs by about 1% between chunk sizes of 8192
   and 32768 tokens), because prefill is compute bound and decode consumes only about one token per sequence per step". If the
   *live* marginal prefill cost barely moves with the batch/phase mix on real hardware, the candidate's price signal may be too
   weak to change eviction decisions — this is a **substantive threat to the mechanism's value**, not to its novelty.

## 4. What I could NOT verify (Candidate 1)

- **ACM SYSTOR / KVLearn full text — PARTIALLY RESOLVED.** `dl.acm.org` returned **HTTP 403** on **four** attempts
  (`/doi/`, `/doi/full/`, `/doi/pdf/`, `/doi/proceedings/10.1145/3793230`; saved 403 bodies at `raw/acm_systor.html`,
  `raw/acm_systor_kv_retention.html`, `raw/acm_systor_full.html`, `raw/acm_systor_pdf.html`, `raw/acm_systor_proceedings.html`), and
  `scilit.com` also returned **403** (`raw/scilit_cachewise.html`). **However I retrieved the abstract and the full cost model by
  non-ACM routes** — the Semantic Scholar Graph API (`raw/ss_systor_api.json`) and the authors' own repo
  (`raw/kvlearn_readme.md`, `raw/kvlearn_github.html`). **What is still NOT verified:** the paper's *body* — its experimental setup,
  whether `R(b)` is ever re-evaluated from live scheduler state inside the paper (vs. the README's calibrated constants), and the
  precise definition of `R̄` in `γ = R̄·λ/M_S`. The README is the **authors' own artifact** but is **not** the peer-reviewed text;
  a claim that turns on `R(b)` being static should be re-checked against the paper body before it is relied on. The GOLD CC-BY
  status and DOI mean a library or an unblocked network should resolve it.
- **KVLearn `R̄` semantics** — I could not determine from the README whether `R̄` in `γ = R̄·λ/M_S` is a global mean recompute cost
  (making `γ` essentially static) or a running mean (making the residency term genuinely live). This matters for the crux: if it is a
  running mean over current arrivals, KVLearn's liveness is even stronger than I have credited.
- **Whether a KVLearn arXiv preprint exists** — two searches found none; the artifact appears to be SYSTOR-only. I did not guess an arXiv ID.
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

**Added after the lead's follow-up (titles read back from the lead's saved abs pages — I did not re-fetch them):**

| arXiv ID | `citation_title` read back | Saved abs file (fetched by the LEAD) | Match? |
|---|---|---|---|
| 2410.21266 | `Online Weighted Paging with Unknown Weights` | `raw/abs_2410.21266.html` | ✓ (2024/10/28 — nearest classical neighbour) |
| 2606.16824 | `CacheWise: Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents` | `raw/abs_2606.16824.html` | ✓ (I additionally fetched the full body myself) |
| 2607.19214 | `Keeping the Cache Warm Pays: Keepalive Economics for Agentic Workloads` | `raw/abs_2607.19214.html` | ✓ (I additionally fetched the full body myself) |

**Non-arXiv occupant verified this session (the decisive one):**

| Artifact | Title read back from | Saved files | Match? |
|---|---|---|---|
| SYSTOR 2026, DOI 10.1145/3793230.3837769 | `To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems` — read back from `raw/ss_systor_api.json` (`paper.title.text`) **and** independently from `raw/kvlearn_github.html` `<title>` | `raw/ss_systor_api.json`, `raw/kvlearn_readme.md`, `raw/kvlearn_github.html` | ✓ |

**Defect log (search-engine output I refused to cite):**
- `web_search` returned **`browse-export.arxiv.org/pdf/2607.16892`** and **`arxiv-org.ezproxy.obspm.fr/html/...`** as result URLs.
  I ignored both and fetched `arxiv.org/abs/...` / `arxiv.org/html/...` directly. No fabricated-ID incident occurred this session,
  but the ezproxy/browse-export surfaces are unreliable proxies and were not used as evidence.
- I did **not** encounter or cite any `thaki-AI/daily-paper-*` artifact.
- Searches surfaced **`www.emergentmind.com/papers/...`**, **`ar5iv.labs.arxiv.org`**, **`alphaxiv.org`**, and **`scilit.com`** mirrors.
  I used none of them as the source of a quote; every quote above comes from a primary URL I fetched (`arxiv.org`, `usenix.org`,
  `docs.sglang.io`, `github.com`, `raw.githubusercontent.com`) or from the Semantic Scholar Graph API record for the one DOI that is
  403-walled here. I did fetch `ar5iv.labs.arxiv.org/html/2606.16824` (`raw/ar5iv_cachewise.html`, HTTP 200) but used it only as a
  cross-check, not as the source of any quote.

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

## Follow-up queries issued after the lead's leads (verbatim, in order)

53. `"To Keep or Not to Keep" learning KV cache retention disaggregated serving arxiv`
54. `Online Weighted Paging with Unknown Weights NeurIPS 2024 learned page weights`
55. `CacheWise KVCache management coding agents reuse-aware eviction tool call metadata`
56. `KVLearn learning-based retention framework keep evict cost-optimization disaggregated LLM serving abstract`
57. `"To Keep or Not to Keep" Liu Yu Jiang Wang SYSTOR 2026 KV cache retention abstract`
58. `KVLearn prefix reuse predictor cost-aware retention score CARS adaptive threshold controller`
59. `KVLearn arxiv preprint FastLM disaggregated KV cache retention`

## Exact URLs I fetched (all via `curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "<Chrome UA>"`)

### Follow-up fetches (after the lead's leads)

33. `https://arxiv.org/html/2607.19214` → `raw/arxiv_2607.19214_html.html` (body; proves the re-prefill price is a constant list price)
34. `https://arxiv.org/html/2606.16824` → `raw/arxiv_2606.16824_html.html` (body; proves CacheWise's victim rule is `argmax τ_j`, unpriced)
35. `https://papers.nips.cc/paper_files/paper/2024/hash/8f9d459c19b59b5400ce396e0f8c23e0-Abstract-Conference.html` → `raw/nips24_weighted_paging.html`
36. `https://www.semanticscholar.org/api/1/paper/6db782df823d9da204a59180305aba81d47b3997` → `raw/ss_systor_api.json` — **the non-ACM route that recovered the SYSTOR/KVLearn abstract and metadata**
37. `https://raw.githubusercontent.com/FastLM/KVLearn/main/README.md` → `raw/kvlearn_readme.md` — **the authors' own CARS/R/U/ATC cost model**
38. `https://github.com/FastLM/KVLearn` → `raw/kvlearn_github.html` — independent `<title>` confirmation of the SYSTOR 2026 paper
39. `https://ar5iv.labs.arxiv.org/html/2606.16824` → `raw/ar5iv_cachewise.html` (cross-check only; **not** a quote source)
40. `https://www.semanticscholar.org/paper/6db782df823d9da204a59180305aba81d47b3997` → `raw/ss_systor_retention.html` (**HTTP 202, 0 bytes — empty**)
41. `https://www.scilit.com/publications/e478c3369d83c15a3c88197997a9a95a` → `raw/scilit_cachewise.html` (**HTTP 403**)
42. `https://dl.acm.org/doi/proceedings/10.1145/3793230` → `raw/acm_systor_proceedings.html` (**HTTP 403**)

### Main-round fetches

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
36. `https://dl.acm.org/doi/proceedings/10.1145/3793230` → **403** → `raw/acm_systor_proceedings.html` (follow-up)
37. `https://www.scilit.com/publications/e478c3369d83c15a3c88197997a9a95a` → **403** → `raw/scilit_cachewise.html` (follow-up)
38. `https://www.semanticscholar.org/paper/6db782df823d9da204a59180305aba81d47b3997` → **202 with 0 bytes** → `raw/ss_systor_retention.html`
    (the HTML paper page is empty; the **API** endpoint #36 above is the one that works)

**Not a quote source (listed for completeness):** `https://ar5iv.labs.arxiv.org/html/2606.16824` → `raw/ar5iv_cachewise.html`, HTTP 200,
used only as a cross-check of the CacheWise body I had already fetched from `arxiv.org`. No quote in this document comes from it.

## Derived (tag-stripped) text files I created for grep-ability

`raw/_2504_11320.txt`, `raw/_2607_16892.txt`, `raw/_2607_02525.txt`, `raw/_2607_09248.txt`,
`raw/_2411_19379.txt`, `raw/_hyperbolic.txt`, `raw/_2607_19214.txt`, `raw/_2606_16824.txt`.
Each is a mechanical strip of tags/entities from the corresponding saved raw HTML;
every quote attributed to one of these was additionally confirmed to be present in the saved raw HTML itself.

## Files saved but NOT fetched by me (supplied by the lead; I only grepped them)

`raw/abs_2410.21266.html`, `raw/abs_2606.16824.html`, `raw/abs_2607.19214.html`, `raw/gds_usenix.html`.
For these four I verified the `citation_title` / page content **inside the saved file** and did not re-fetch the URL.
Note `raw/gds_usenix.html` (lead's copy) and `raw/gds_usenix_node8.html` (my copy) are two separate fetches of the same USITS'97 page
(different md5); the quotes in this document cite **my** copy `raw/gds_usenix_node8.html`, and the lead's copy carries the same text.
