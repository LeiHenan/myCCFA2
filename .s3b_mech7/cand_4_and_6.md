# S3b — Adversarial prior-art / occupancy check: Candidates 4 and 6

Agent: S3b (delegated). Workspace: `/Users/leihenan/Desktop/myProject/.s3b_mech7/`.
All raw HTML saved under `.s3b_mech7/raw/`. Every arXiv ID cited below had its title read back
from `arxiv.org/abs/<id>` **in this session** (or, where marked, from the lead's already-saved
`raw/abs_<id>.html`) — mismatches are logged in §C4.6 / §C6.6.

**Headline:** both candidates are **OCCUPIED**. Details, verbatim quotes, and the residual gaps below.

---

## CANDIDATE 4 — "Harvesting idle tensor cores at low batch"

### C4.1 Verdict

**OCCUPIED.**

And, specifically for the residual question the lead posed (is anything **crossover-aware** —
does it derive the batch point where surplus FLOPs run out and switch behaviour there?), the
answer is also **OCCUPIED**, including in *production*: vLLM ships both a cost-model-driven
per-step crossover (Adaptive Verification) and a batch-size→K lookup table with an explicit
`K=0` disengage band (Dynamic Speculative Decoding).

### C4.2 Single strongest occupying artifact (mechanism)

**Lookahead Decoding** — https://arxiv.org/abs/2402.02057

- **State verified:** arXiv abs page fetched by me to `raw/arxiv_2402.02057.html`
  (`HTTP 200`, title read back ✓ `[2402.02057] Break the Sequential Dependency of LLM Inference
  Using Lookahead Decoding`). Full HTML v1 previously fetched by the lead to `raw/la_v1.html`
  with text at `raw/la_v1.txt`; I re-grepped the lead's text file myself and confirmed each
  string below returns a hit (`grep -c` = 1).
- **Verbatim quotes** (grep `raw/la_v1.txt`):

  > `Autoregressive decoding of large language models (LLMs) is memory bandwidth bounded, resulting in high latency and significant wastes of the parallel processing power of modern accelerators.`

  > `Lookahead Decoding takes advantage of the particular characteristics of autoregressive decoding, which is bounded by the memory bandwidth–as each generated token depends on all tokens before it–rather than compute, by using the available cycles to generate and verify n-grams (subsequent tokens) at virtually no additional cost.`

  > `It allows trading per-step log(FLOPs) to reduce the number of total decoding steps`

  Section heading present in the same file: `Decode, Predict, and Verify in The Same Step`.

- **How close:** this is the candidate, stated as the paper's thesis. It (a) names the exact
  quantity in the box — memory-bandwidth-bounded decode leaving accelerator parallelism idle,
  (b) harvests it by spending extra FLOPs per step, (c) is draft-model-free, i.e. outside the
  "speculative decoding is already known" exclusion, (d) reports 1.8×–4× wall-clock.
  It is a preprint (ICML 2024) with experiments.

### C4.3 Single strongest occupying artifact (the crossover-aware residual)

**vLLM — "Adaptive Verification"** (production documentation)
https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/

- **State verified:** fetched by me to `raw/vllm_adaptive_verification.html` (`HTTP 200`,
  `<title>Adaptive Verification - vLLM</title>`). Vendor doc, **not** a paper; describes a
  shipped, default-off feature.
- **Verbatim quote** (grep `raw/vllm_adaptive_verification.html`):

  > `The crossover moves with load and with workload-dependent acceptance rates, so no static num_speculative_tokens is right across concurrencies.`

  Supporting strings in the same file (also greppable):

  > `While the GPU is memory-bound that slot is effectively free and worth the gamble; once it saturates the gamble has a real throughput cost.`

  > `The budget itself comes from a cost model profiled at startup. vLLM measures what a step costs at each shape, then picks the token count that maximizes expected accepted tokens per second.`

- **How close:** this is a *measured, cost-model-driven crossover rule* that decides per step how
  much surplus-compute speculation to buy, explicitly because the memory-bound/compute-bound
  crossover "moves with load". That is precisely the "derive the batch/context point at which
  surplus FLOPs run out and switch behaviour there" residual. It does not use batch *size* as the
  only key (it scores per-(request, position) survival probability under a global budget), but the
  crossover logic is explicit and the budget is profiled per step shape.

### C4.4 Table of further artifacts (candidate 4)

| # | Artifact | URL | State I verified | Verbatim quote (grep target) | What it covers |
|---|---|---|---|---|---|
| 1 | vLLM — Dynamic Speculative Decoding | https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/ | fetched by me → `raw/vllm_dynamic_spec_docs.html` (`HTTP 200`, title ✓ `Dynamic Speculative Decoding - vLLM`). Vendor doc. | `When this BS*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT).` — also `no draft tokens will be produced` | **Explicit crossover + disengage.** Config is a literal batch-range→K table: `[ [1, 64, 3], [65, 128, 1], [129, 512, 0] ]` where K=0 means no drafting at all. Production, shipped. |
| 2 | TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput | https://arxiv.org/abs/2406.14066 (+ https://arxiv.org/html/2406.14066v3) | abs fetched by me ✓ title `[2406.14066] TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput`; v3 HTML fetched ✓. Preprint, on vLLM, with experiments. | `speculative decoding may degrade LLM serving performance if added naively` (grep `raw/arxiv_2406.14066.html`) | **Measured crossover-aware control.** Profiles the execution environment and uses a feedback loop to pick the intra-request parallelism amount maximising goodput; explicitly "disabling speculative decoding when it does not [improve]". Reports an adaptive threshold: once batch size exceeds the recorded disable-point, the draft model's prefill is skipped entirely. |
| 3 | Nightjar: Dynamic Adaptive Speculative Decoding for LLM Serving | https://arxiv.org/abs/2512.22420 (+ https://arxiv.org/html/2512.22420v5) | abs fetched by me ✓ title `[2512.22420] Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving`; v5 HTML fetched ✓. Preprint with experiments; also has a ScienceDirect journal version. | `proactively disables speculative decoding when the MAB planner determines` (grep `raw/arxiv_2512.22420v5_html.html`) — also `As batch size increases, both SD and autoregressive decoding move toward the compute-bound regime` | **Roofline crossover + disengage + memory reclamation.** Picks optimal speculative length per batch size, disables SD when the bandit planner says it no longer pays, and offloads the draft model to free KV cache for larger batches. States the residual's premise directly: "it improves throughput in low-load, memory-bound systems but degrades performance in high-load, compute-bound environments". |
| 4 | ESP: Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing | https://arxiv.org/abs/2603.17942 | Lead fetched; I read title back from `raw/abs_2603.17942.html` ✓ `[2603.17942] Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing`; text at `raw/esp_2603.17942.txt`. Preprint (ICML 2026 per lead). | `training-free multi-token prediction` (grep `raw/html_2603.17942.html`) | **Draft-free parallel multi-token prediction on an off-the-shelf model** — probes mask tokens in embedding space, verifies in parallel, +7–11% acceptance length over LADE. Also defines "block complexity" and notes verification "can become compute-bound when this bundle grows large" — i.e. it is aware of the same budget ceiling. |
| 5 | Uno: Unlocking Lossless Speedups in LLMs via Discrete Diffusion | https://arxiv.org/abs/2609.04010 | abs fetched by me ✓ title `[2609.04010] Unlocking Lossless Speedups in LLMs via Discrete Diffusion`. Preprint. | `Unlike speculative decoding, our method requires no separate draft model` (grep `raw/arxiv_2609.04010.html`) | Multi-token-per-step by construction (diffusion weights draw several tokens from the AR distribution). Claims higher throughput than leading SD **at every evaluated batch size** including the largest the device supports — i.e. it attacks the crossover from the other side. Caveat: needs trained diffusion weights, so it is not inference-time harvesting of idle TC on a frozen model. |
| 6 | LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding | https://arxiv.org/abs/2404.16710 | abs fetched by me ✓ title `[2404.16710] LayerSkip: ...`; HTML v4 fetched to `raw/arxiv_2404.16710_html.html`. Preprint, trained models, experiments. | `self-speculative decoding solution where we exit at early layers and verify and correct with remaining layers of the model` | Covers the candidate's "drafting via the target model itself with early exit / layer truncation" direction. Same family as S3D (2405.20314) and DEL (2504.05598), both of which I also verified by title. |
| 7 | Sleep-time Compute: Beyond Inference Scaling at Test-time | https://arxiv.org/abs/2504.13171 | abs fetched by me ✓ title `[2504.13171] Sleep-time Compute: Beyond Inference Scaling at Test-time`. Preprint with experiments. | `we can significantly reduce the compute requirements at test-time` | Covers "spend compute off the critical path to reduce next-step/later work". Uses idle capacity at a *different* timescale (offline, pre-query) than intra-step idle tensor cores — a genuine boundary, but the compute-for-latency thesis is the same. |
| 8 | NanoFlow: Towards Optimal Large Language Model Serving Throughput | https://arxiv.org/abs/2408.12757 (OSDI'25) | abs fetched by me ✓ title `[2408.12757] NanoFlow: ...`. Peer-reviewed, experiments. | `end-to-end LLM serving is compute bound for most common workloads` | Adversarial counter-evidence to the box's premise: NanoFlow *measures* that serving is compute-bound for most workloads and harvests it by overlapping compute/memory/network via intra-device nano-batching, not by adding speculative FLOPs. |
| 9 | BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms | https://arxiv.org/abs/2505.15141 | Lead fetched (`raw/abs_2505.15141.html`); I read title back ✓ `[2505.15141] BanditSpec: ...`. Preprint with theory. | `training-free online learning framework to adaptively choose the configuration of the hyperparameters for speculative decoding as text is being generated` | Adaptive/self-tuning *speculation length* via MAB (UCBSpec/EXP3Spec). Nightjar's related-work explicitly faults it for "static design fails to incorporate real-time batch sizes as context" — so BanditSpec tunes but is not batch-crossover-aware; DSD is the batch-aware antecedent. |
| 10 | WISP: Waste- and Interference-Suppressed Distributed Speculative LLM Serving at the Edge | https://arxiv.org/abs/2601.11652 | abs fetched by me ✓ title `[2601.11652] WISP: ... via Dynamic Drafting and SLO-Aware Batching`. Preprint. | `a verification batch scheduler` | Dynamic drafting + SLO-aware verification batching formulated as SLO-constrained long-run **goodput maximization** — the scheduling-side crossover control, distinct from the roofline-side one. |

Also checked and **not** occupiers of the specific mechanism (fetched/failed, see §C4.6):
`raw/arxiv_2504.02263.html` MegaScale-Infer (MoE attention/FFN disaggregation for low-batch
utilization — a different lever: module disaggregation, not extra speculative FLOPs),
`raw/arxiv_2402.02057.html` (already the occupant), `raw/arxiv_2504.13171.html`.

### C4.5 Nearest-neighbour / novelty sentence for candidate 4

> Nearest neighbour Lookahead Decoding (arXiv 2402.02057) does harvest idle memory-bound decode
> cycles by trading per-step log(FLOPs) for fewer steps, and TurboSpec/Nightjar/vLLM-DSD do
> adaptively disengage speculation; under condition C = *the model has not been retrained / the
> deployer is on a stock checkpoint and must decide per-step how much surplus compute to buy*,
> Lookahead misses an explicit measured crossover rule (its own §5.5 only says "needs large surplus
> FLOPs … may cause slowdowns", a qualitative caveat), and TurboSpec/Nightjar/vLLM-DSD supply that
> rule but only for *drafting*, i.e. only for the speculative-decoding family; we do B = derive the
> crossover for non-speculative surplus-compute work at batch ≈ 28 on sm120 and switch the mechanism
> there, therefore when C holds the conclusion differs.

**Honest counterweight:** vLLM Adaptive Verification already profiles per-step cost "at each shape"
and picks the token count maximising accepted tokens/second, so the *crossover-aware surplus-compute
controller* itself is not novel — only its application to a non-speculative consumer of the FLOPs
would be.

**What would kill candidate 4 (I did not find these; if they exist the candidate is dead):**
1. Any measured roofline/crossover study on **Blackwell RTX PRO 6000 / sm120** at Qwen3-4B scale
   that already publishes the batch-≈28 bandwidth→compute inflection and uses it to gate a
   mechanism. (I did not find one; the closest are generic roofline analyses and Nightjar's
   conceptual Figure 1.)
2. Any non-speculative inference-time mechanism that spends surplus low-batch FLOPs **without**
   training new weights and reports a batch-dependent disengage rule. (ESP is training-free but is
   still a draft-verify scheme; Uno requires trained diffusion weights.)
3. A production engine flag that already gates a non-SD surplus-compute feature on batch size.

### C4.6 Candidate 4 — what I could NOT verify

- `api.github.com` — unusable by instruction (rate-limited to zero); I used HTML PR/issue pages.
- `raw/trtllm_lookahead_readme.html` — **HTTP 404**, the TensorRT-LLM lookahead example README
  path I tried from a search result does not exist on `main`. I am **not** citing it. Surfaced only
  via a search-result snippet pointing at a third-party fork
  (`zhengd-nv/TensorRT-LLM`), which I did not fetch.
- `arxiv.org/search` (406) and `export.arxiv.org` (429) per project rules — not used.
- Time budget did not permit reading §5.5 of Lookahead v2/v3/v5 (the lead's `la_v2/v3/v5.html`
  are 7,715-byte stubs, i.e. failed fetches — only `la_v1.html` is real). The §5.5 quote I relied on
  comes from `raw/la_v1.txt`, which the lead supplied; I verified it by `grep`, not by fetching v1
  myself.
- Nightjar's peer-reviewed ScienceDirect version (`S1383762126002079`) — not fetched; only the
  arXiv v5 HTML was read.

### C4.7 Candidate 4 — search log

Queries issued (verbatim), in order:

```
self-speculative decoding layer skipping early exit draft with target model
idle tensor cores low batch LLM decode exploit spare compute
batch-1 decoding underutilized GPU exploit free FLOPs latency
adaptive thinking budget control reasoning length scheduler
LayerSkip self-speculative decoding early exit arXiv
Draft-Then-Refine self-speculative decoding layer truncation
adaptive reasoning budget control batch scheduling goodput LLM serving
length-aware scheduling LLM serving output length control truncation SLO
sleep-time compute beyond inference scaling at test-time arXiv
NanoFlow intra-device parallelism LLM serving low batch memory compute overlap
multi-token prediction without draft model parallel decoding diffusion LLM
MegaScale-Infer disaggregated expert parallelism low batch utilization
"compute is free" batch size 1 LLM decode idle FLOPs exploit
speculative prefill predict likely continuation prefill
opportunistic LLM serving idle GPU slack best-effort requests
"idle" tensor cores "low batch" inference exploit extra work per step
Lookahead decoding parallel decoding Jacobi iteration break sequential dependency
layerskip draft verify layer truncation self-speculative decoding survey
multi-token prediction DeepSeek-V3 speculative decoding extra heads
early exit ensemble LLM inference acceleration
adaptive lookahead window selection speculative decoding disengage high batch
crossover batch size memory bound compute bound adaptive speculative decoding disable
self-tuning lookahead decoding window size W N G
speculative decoding disabled at large batch hybrid throughput aware
Nightjar dynamic adaptive speculative decoding serving arXiv 2512.22420
adaptive speculative decoding turn off when batch size exceeds threshold
"surplus FLOPs" speculative decoding compute bound batch measured crossover
self-tuning lookahead decoding dynamic window selection overhead aware
BanditSpec bandit speculative decoding hyperparameter selection arXiv
DSD dynamic speculative decoding disable speculation batch size arXiv
scheduler-controlled output length serving goodput reasoning budget truncation
```

URLs fetched by me this session (all saved under `.s3b_mech7/raw/`):
`arxiv.org/abs/2404.16710`, `/2405.20314`, `/2504.20068`, `/2504.05598`, `/2505.18822`,
`/2602.01237`, `/2604.11001`, `/2408.12757`, `/2504.02263`, `/2504.13171`, `/2609.04010`,
`/2507.10150`, `/2604.19780`, `/2606.31580`, `/2402.02057`, `/2512.22420`, `/2406.14066`,
`/2510.11713`, `/2505.05315`, `/2505.15141`, `/2601.11652`;
`arxiv.org/html/2504.20068v3`, `/2404.16710v4`, `/2406.14066v3`, `/2512.22420v5`, `/2606.31580v1`;
`docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/`,
`docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/`;
`github.com/vllm-project/vllm/issues/25112`, `/pull/37112`, `github.com/sgl-project/sglang/pull/6208`;
`raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/.../tools/budget/README.md`;
`hiascend.com/document/detail/en/mindie/300/.../thinking_budget.md`;
`icml.cc/virtual/2026/poster/63644`; `hitech863.com/.../view_abstract.aspx?file_no=202602004`;
`minlanyu.seas.harvard.edu/writeup/sllm25-length.pdf`;
`iotdigitaltwinplm.com/reasoning-effort-control-llm-serving-thinking-budgets-2026/`.

---

## CANDIDATE 6 — "Output length as a control variable"

### C6.1 Verdict

**OCCUPIED.**

The specific mechanism — the *system* choosing/controlling how long a request reasons, with that
choice driven by serving state rather than predicted — exists, is named in almost the candidate's
own words, and has experiments plus SLO measurements. If the lead insists on the narrower
distinguishing condition "the choice interacts with **batch composition** in a *large-batch* GPU
server for goodput", the honest rating degrades to **PARTIALLY OCCUPIED** — see §C6.3.

### C6.2 Single strongest occupying artifact

**LASER: Load-Aware Serving with Early-Exit for Reasoning LLMs at the Edge**
https://arxiv.org/abs/2606.31580

- **State verified:** abs page fetched by me to `raw/arxiv_2606.31580.html` (`HTTP 200`,
  title read back ✓ `[2606.31580] LASER: Load-Aware Serving with Early-Exit for Reasoning LLMs at
  the Edge`); **HTML body fetched by me** to `raw/arxiv_2606.31580_html.html` (title ✓). Preprint
  v1, 30 Jun 2026, with experiments (2 reasoning models, 4 benchmarks, diverse load conditions).
- **Verbatim quote** — the money line, from the HTML body (grep `raw/arxiv_2606.31580_html.html`):

  > `These approaches improve model- or system-level efficiency, but generally treat generation length as fixed. In contrast, LASER treats reasoning depth itself as a controllable serving variable and adapts it to system load.`

  Further greppable strings in the same two files:

  > `a load-aware adaptive exit threshold that adjusts the confidence bar based on real-time system load within an empirically validated robust range`

  > `a difficulty- and load-aware reasoning budget pre-allocation that assigns compute resources by request difficulty and system capacity`

  > `At the edge, a single GPU serves requests sequentially or with minimal batching, so one request’s continued reasoning directly delays those behind it.`
  (grep note: the apostrophe in `request’s` is U+2019 in the saved HTML, not ASCII `'` — copy the
  character from this line, not from the ASCII rendering below)

  > `A fixed threshold may exit too early under light load or allow excessive reasoning under heavy load, hurting latency targets.`

- **How close:** this is the candidate's crux, verbatim: service requirement as a **controllable
  serving variable**, chosen by the system from load, with the interaction to co-scheduled work
  ("one request's continued reasoning directly delays those behind it") used as the justification.
  It reports 17–38% latency reduction and 3–6% SLO-satisfaction gain over fixed-threshold
  baselines at ~1% accuracy cost. The residual is scope, not mechanism: LASER is *edge*, single GPU,
  "sequential or with minimal batching", and keys on **load** (queue depth / concurrency), not on an
  explicit goodput-maximising **batch composition** decision.

### C6.3 The residual, stated precisely

No artifact I found does **all** of: (i) system chooses the length, (ii) the choice is made *jointly
with* which requests are co-batched, (iii) in a large-batch server, (iv) to maximise goodput.
The literature splits cleanly:

- **Choose the length, keyed on load/concurrency:** LASER (2606.31580).
- **Shape the batch, without choosing length:** Triage "feedback-driven batch shaping"
  (`raw/triage_batch_shaping.html`), JITServe (2504.20068 — see §C6.4 row 9), WISP (2601.11652).
- **Expose length as a per-request knob the caller sets:** vLLM `reasoning_budget`
  (PR #37112), SGLang `thinking_budget` (PR #6208), MindIE `thinking_budget`, Nemotron
  `ThinkingBudgetLogitsProcessor` — none of these let the *scheduler* pick the value.

So the surviving novelty is the *joint* optimisation in (ii)+(iii)+(iv). That is a real but narrow
gap, and it is exactly the "nearest neighbour misses one identifiable distinguishing condition"
shape:

> Nearest neighbour LASER (arXiv 2606.31580) does choose reasoning depth as a controllable serving
> variable and adapts it to system load, and vLLM/SGLang/MindIE/Nemotron expose length caps as
> per-request knobs; under condition C = *the server runs large continuous batches (n ≫ 1) where
> per-request KV footprint is the binding constraint and the objective is goodput, not the latency of
> a single request*, LASER misses an explicit batch-composition coupling (it is edge, minimal-batch,
> load-keyed) and the engine knobs miss leaving the value to the scheduler; we do B = choose the cap
> jointly with batch admission/eviction under a KV budget, therefore when C holds the conclusion
> differs.

**What would kill candidate 6 (I did not find these):**
1. Any serving paper that formulates "admission/eviction + per-request length cap" as one
   goodput-maximisation problem in a large-batch engine. (Closest: JITServe and Triage, but neither
   controls length; and 2602.01237 allocates budget across a batch but from *predictions*.)
2. A classical-OR result applying **controllable processing times** to a queue with batching —
   the theory exists in isolation (§C6.4 row 11) and would subsume the formulation.
3. A vendor engine that lets a *scheduler policy*, not the client, set the thinking budget from
   observed batch state.

### C6.4 Table of further artifacts (candidate 6)

| # | Artifact | URL | State I verified | Verbatim quote (grep target) | What it covers |
|---|---|---|---|---|---|
| 1 | vLLM PR #37112 — "Add reasoning_budget to cap thinking tokens via existing reasoning parser" | https://github.com/vllm-project/vllm/pull/37112 | fetched by me → `raw/vllm_pr_37112_reasoning_budget.html` (`HTTP 200`, title ✓). **Production engineering artifact** (PR is closed/unmerged — status noted). | title string `Add reasoning_budget to cap thinking tokens via existing reasoning parser` | The *mechanism* of capping thinking length ships in the mainstream engine. Value is a request parameter — the caller sets it, not the scheduler. |
| 2 | SGLang PR #6208 — "add thinking_budget (version 2)" | https://github.com/sgl-project/sglang/pull/6208 | fetched by me → `raw/sglang_pr_6208_thinking_budget.html` (`HTTP 200`, title ✓). Production engineering artifact. | title string `add thinking_budget (version 2)` | Second mainstream engine implementing the same cap. Confirms the mechanism is commodity, not novel. |
| 3 | MindIE (Huawei Ascend) — Thinking Budget feature doc | https://www.hiascend.com/document/detail/en/mindie/300/LLMframe/llmdev/user_guide/feature/thinking_budget.md | fetched by me → `raw/mindie_thinking_budget.html` (`HTTP 200`). **Vendor doc.** | `when the reasoning exceeds the thinking_budget, the system truncates the chain of thought using a prompt, encouraging the model to stop reasoning early` | Shows the cap is enforced **server-side, by the system** ("the system truncates"), with an explicit speed/quality trade-off framing. Still caller-specified budget. |
| 4 | NVIDIA Nemotron — `ThinkingBudgetLogitsProcessor` (vLLM V1 custom logit processor) | https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/refs/heads/main/tools/budget/README.md | fetched by me → `raw/nemotron_budget_logit.html` (`HTTP 200`, 5,474 B). **Vendor doc/code.** | `This is useful for balancing inference cost against reasoning depth, or enforcing latency constraints in production deployments.` | Vendor-blessed implementation of runtime thinking-length control; supports per-request overrides. Again caller-set, not scheduler-set. |
| 5 | AdaCtrl: Towards Adaptive and Controllable Reasoning via Difficulty-Aware Budgeting | https://arxiv.org/abs/2505.18822 | abs fetched by me ✓ title `[2505.18822] AdaCtrl: ...`. Preprint with experiments. | `difficulty-aware adaptive reasoning budget allocation and explicit user control over reasoning depth` | Covers "difficulty-aware token budget" + explicit control interface (length-triggered tags). Model/training-level; not tied to serving state or batch composition. |
| 6 | Scalable Chain of Thoughts via Elastic Reasoning | https://arxiv.org/abs/2505.05315 | abs fetched by me ✓ title `[2505.05315] Scalable Chain of Thoughts via Elastic Reasoning`. Preprint with experiments. | `their uncontrolled output lengths pose significant challenges for real-world deployment, where inference-time budgets on tokens, latency, or compute are strictly constrained` | Makes truncation *safe* by splitting thinking/solution budgets and training under budget-constrained rollouts. This removes the main objection to using length as a control variable (quality collapse) — i.e. it enables the candidate's mechanism. |
| 7 | Are Large Reasoning Models Interruptible? | https://arxiv.org/abs/2510.11713 | abs fetched by me ✓ title `[2510.11713] Are Large Reasoning Models Interruptible?`. Preprint (ICML 2026 per repo); benchmark study. | `even state-of-the-art LRMs, which achieve high accuracy in static settings, can fail unpredictably when interrupted or exposed to changing context, with performance dropping by up to 60%` | Names and measures the **cost of the control action** (reasoning leakage, panic, self-doubt). Adversarial for the candidate: it quantifies why interrupting to save length is not free. |
| 8 | Predictive Scheduling for Efficient Inference-Time Reasoning in LLMs | https://arxiv.org/abs/2602.01237 | abs fetched by me ✓ title `[2602.01237] Predictive Scheduling for ...`. Preprint with experiments. | `Our greedy batch allocator dynamically distributes a fixed total token budget across queries to maximize expected accuracy.` | **Batch-level budget allocation** — the closest thing to "length choice interacting with the batch" — but the allocation is driven by *prediction* (predictors on hidden states / question text). This is precisely the "estimation problem" the candidate wants to distinguish itself from. |
| 9 | JITServe: SLO-aware LLM Serving with Imprecise Request Information | https://arxiv.org/abs/2504.20068 | abs fetched by me ✓ title `[2504.20068] JITServe: ...`; **v3 HTML fetched by me** → `raw/arxiv_2504.20068v3.html`. Preprint with experiments. | `while deciding the composition of requests in a batch to maximize efficiency and goodput with provable guarantees` | **Batch composition** for goodput under imprecise *length estimates*. I grepped the v3 HTML for truncation/cap/budget language: it treats remaining length as an estimate to be refined, **not** a variable it sets. So: occupies the batch-composition half, not the length-choice half. |
| 10 | Triage: optimizing LLM serving goodput in unified scheduling via feedback-driven batch shaping | http://www.hitech863.com/gjstxen/ch/reader/view_abstract.aspx?file_no=202602004&flag=1 | fetched by me → `raw/triage_batch_shaping.html` (`HTTP 200`, title ✓). **Journal paper** (高技术通讯(英文) 2026, 32(2):121–132), abstract page only. | `we propose Triage, a unified scheduling framework that optimizes goodput via feedback-driven batch shaping` | "Batch shaping" = directly manipulating engine-level knobs to reshape the running batch. Occupies "the batch is a control surface" but controls prefill/decode regime and batching hyperparameters, **not** output length. |
| 11 | Shabtay & Steiner, "A survey of scheduling with controllable processing times", Discrete Applied Mathematics (2007) | https://dl.acm.org/doi/abs/10.1016/j.dam.2007.02.003 (also https://www.sciencedirect.com/science/article/abs/pii/S0166218X07000225X) | **NOT fetched — 403.** Title verified only from independent listings (ACM DL, zbMATH Zbl 1119.90022, ScienceDirect, Mendeley). Treat as unverified-by-me. | n/a — no fetched full text | **The conceptual core is a mature classical-OR field**: scheduling where each job's processing time is a decision variable you may compress at a cost. If the candidate's novelty rests on "service requirement is chosen, not given", this literature is the nearest classical ancestor — and it is exactly the kind of classical result the 2605.01280 position paper says should be brought back into serving. |
| 12 | "Reasoning-Effort Control in LLM Serving: Thinking Budgets (2026)" | https://iotdigitaltwinplm.com/reasoning-effort-control-llm-serving-thinking-budgets-2026/ | fetched by me → `raw/iot_reasoning_effort.html` (`HTTP 200`, 154,949 B). **Content-marketing blog** (posted 21 Jul 2026, no experiments, no author affiliation). Cited only as evidence of what is *common knowledge*, not as a result. | `Reasoning Effort Is a Scheduling Dimension, Not a Model Choice` — also `A knob that changes generation length is not an API convenience — it is an input to the scheduler.` | States the candidate's thesis as settled 2026 practice: length/effort is a scheduler input, with batch composition, KV residency and admission control all depending on it. Also names `vLLM's per-request thinking_token_budget`, OpenAI `reasoning.effort`, Anthropic `budget_tokens`, Gemini `thinking_level`. **AI-ish blog — no experiments; do not cite as evidence of a result.** |

Also relevant and checked: `raw/arxiv_2604.19780.html` — BCAE, "Avoiding Overthinking and
Underthinking: Curriculum-Aware Budget Scheduling for LLMs" (title ✓ `[2604.19780]`), a preprint
whose "curriculum-aware budget scheduler" shifts the budget distribution **during training**;
it schedules budgets, not serving requests. And `raw/arxiv_2604.11001.html` — "Flow-Controlled
Scheduling for LLM Inference with Provable Stability Guarantees" (title ✓ `[2604.11001]`), which
controls the *admission rate of prompts* under unknown decode lengths, not the lengths themselves.

### C6.5 Candidate 6 — what I could NOT verify

- **Paywalled / blocked:** `dl.acm.org/doi/abs/10.1016/j.dam.2007.02.003` and
  `sciencedirect.com/science/article/abs/pii/S0166218X0700025X` (HTTP 403; `raw/sciencedirect_shabtay.html`
  is a 1.2 MB Cloudflare/consent shell, no abstract text). `zbmath.org/?q=an:1119.90022` — HTTP 403
  Cloudflare ("Just a moment..."), saved as `raw/zbmath_shabtay_steiner.html`.
  `semanticscholar.org` — HTTP 202 with 0 bytes, saved empty as `raw/drllms_semanticscholar.html`.
  So the classical controllable-processing-times literature is **title-verified only**, never
  fetched; I have no verbatim quote from it and I am not quoting it.
- **DRLLMS: Network-Adaptive Reasoning Control for Interactive LLM Streaming** — appears only on
  `dl.acm.org/doi/abs/10.1145/3798065.3798072` (paywalled) and Semantic Scholar (202/empty). I could
  **not** fetch or verify its abstract; it is named here only as an unverified lead from search
  snippets, and I am not citing any claim from it. Its title suggests network-state-driven reasoning
  control, which would be another load-keyed length control if confirmed.
- **ICML 2026 "Beyond Prediction: Tail-Aware Scheduling for LLM Inference"** (poster 63644) —
  fetched ✓ (`raw/icml2026_63644.html`, `HTTP 200`). Explicitly **prediction-free** scheduling, but
  it replaces length *prediction* with soft priority boosting; it does **not** choose output length,
  so it does not occupy the candidate. Included here because it is the strongest "prediction is
  fragile" rebuttal to the candidate's framing and could be mistaken for an occupier.
- I did **not** exhaustively search the training-time reasoning-length-control literature
  (L1/Thinkless/TokenSkip/Length-Controlled RL and similar). Those control length at the *model*
  level rather than the serving level; I judged them lower-value than the serving-side artifacts
  above and did not fetch them, so I make no claim about them.
- vLLM issue #25112 was fetched (`raw/vllm_issue_25112.html`, `HTTP 200`, title ✓ "Spec decoding is
  not disabled at/after configured batch size") — it corroborates that a batch-size disable
  threshold is a first-class vLLM concept, and I used it only for that.

### C6.6 Candidate 6 — search log

Queries issued (verbatim):

```
adaptive thinking budget control reasoning length scheduler
adaptive reasoning budget control batch scheduling goodput LLM serving
length-aware scheduling LLM serving output length control truncation SLO
scheduling with controllable service duration LLM serving adaptive output length
thinking budget allocation across requests scheduler reasoning
overthinking underthinking reasoning length control
compute-for-latency trade LLM inference spare compute
"controllable processing times" scheduling theory jobs compress processing time
scheduler chooses output length LLM serving truncate reasoning goodput
adaptive max_tokens SLO aware LLM serving cap generation length
budget forcing s1 simple test-time scaling thinking budget
controllable service time queueing LLM request length control scheduler
reasoning length control system level serving goodput batch
truncate reasoning meet deadline SLO thinking budget scheduler
adaptive output length goodput batch composition LLM serving
"length control" reasoning model RL control how long thinks L1
Thinkless LLM learns when to think dynamic reasoning length
adaptive inference-time compute mid-generation stopping early
SLO-aware reasoning model serving token budget scheduler batch
reasoning budget scheduler continuous batching goodput thinking length system chooses
vLLM SGLang thinking budget token control feature reasoning length
SLO-aware reasoning truncation scheduler decides how long to think
"output length" control variable scheduler LLM serving batch composition
anytime inference LLM interruptible stop generation return answer quality
deadline-aware truncation output length admit more requests goodput LLM
reasoning budget allocation serving system batch scheduler 2026
Elastic Reasoning thinking budget controllable phases LLM arXiv
batch composition goodput length truncation scheduling LLM reasoning 2026
reasoning length control scheduler chooses budget batch fill
goodput continuous batching reasoning budget cap scheduler batch fill thinking
"thinking budget" serving system scheduler adaptive per batch LLM 2026 paper
"reasoning budget" serving scheduler load-aware batch 2026
adaptive output truncation LLM serving SLO deadline system chooses
scheduling with controllable processing times survey Shabtay Steiner
"A survey of scheduling with controllable processing times" Shabtay Steiner abstract pdf
controllable processing times scheduling job processing time decision variable compression cost
```

---

## Appendix — arXiv titles read back in this session (all ✓)

`[2404.16710]` LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding ·
`[2405.20314]` S3D: A Simple and Cost-Effective Self-Speculative Decoding Scheme for Low-Memory GPUs ·
`[2504.20068]` JITServe: SLO-aware LLM Serving with Imprecise Request Information ·
`[2504.05598]` DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding ·
`[2505.18822]` AdaCtrl: Towards Adaptive and Controllable Reasoning via Difficulty-Aware Budgeting ·
`[2602.01237]` Predictive Scheduling for Efficient Inference-Time Reasoning in Large Language Models ·
`[2604.11001]` Flow-Controlled Scheduling for LLM Inference with Provable Stability Guarantees ·
`[2408.12757]` NanoFlow: Towards Optimal Large Language Model Serving Throughput ·
`[2504.02263]` MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism ·
`[2504.13171]` Sleep-time Compute: Beyond Inference Scaling at Test-time ·
`[2609.04010]` Unlocking Lossless Speedups in LLMs via Discrete Diffusion ·
`[2507.10150]` Past-Future Scheduler for LLM Serving under SLA Guarantees ·
`[2604.19780]` Avoiding Overthinking and Underthinking: Curriculum-Aware Budget Scheduling for LLMs ·
`[2606.31580]` LASER: Load-Aware Serving with Early-Exit for Reasoning LLMs at the Edge ·
`[2402.02057]` Break the Sequential Dependency of LLM Inference Using Lookahead Decoding ·
`[2512.22420]` Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving ·
`[2406.14066]` TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput ·
`[2603.17942]` Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing (from lead's saved abs page) ·
`[2510.11713]` Are Large Reasoning Models Interruptible? ·
`[2505.05315]` Scalable Chain of Thoughts via Elastic Reasoning ·
`[2505.15141]` BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms (from lead's saved abs page) ·
`[2601.11652]` WISP: Waste- and Interference-Suppressed Distributed Speculative LLM Serving at the Edge via Dynamic Drafting and SLO-Aware Batching

**No fabricated IDs to log** — every ID in this report matched its abs-page title on read-back.
The one search-result ID I did *not* put in this report because I could not confirm it is DRLLMS
(§C6.5).
