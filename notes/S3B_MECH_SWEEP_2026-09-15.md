# S3b — Occupancy / prior-art check on 7 candidate mechanisms (LLM inference acceleration)

Date of record: 2026-09-15. Cost: 0 GPU·h (desk research only).
Target box: single RTX PRO 6000 Blackwell, 95.6 GB HBM, sm120, driver 580.82.09, ~1.79 TB/s, vLLM 0.29.0, Qwen3-4B bf16.
Status of this document: **IN PROGRESS — this revision is a first partial write issued on request.**
Candidates 2, 3, 4, 6, 7 are marked `NOT YET CHECKED` in §1 and are being worked in parallel; §2–§4 will be extended in place.

**Provenance key.** `[self]` = I fetched the page myself in this session and read the title back where it is an arXiv item. `[sub-A]` = verified by my candidate-1/7 subagent. `[sub-B]` = verified by my candidate-2/3 subagent. `[sub-C]` = verified by my candidate-4/6 subagent. Every quote below is copied from a raw HTML file saved under `.s3b_mech7/raw/`; the file name is given next to each so the claim can be re-grepped. Rows not marked `[self]` are folded in from a subagent and are **not yet re-grepped by me** — that re-verification pass is pending and is listed in §3.

---

## 1. Per-candidate verdict table

| # | Candidate | Verdict | Single strongest occupying artifact | State I verified by fetching | Verbatim quote |
|---|---|---|---|---|---|
| 1 | Load-priced KV eviction | `PARTIALLY OCCUPIED` **(interim)** | <https://arxiv.org/abs/2410.21266> — *Online Weighted Paging with Unknown Weights* (NeurIPS 2024, v1 28 Oct 2024) `[self]` | abs page fetched by me; title read back ✓; abstract read | *"In the weighted variant of this problem, each page has its own fetching cost; a substantial line of work on this problem culminated in an (optimal) $O(\log k)$-competitive randomized algorithm, due to Bansal, Buchbinder and Naor (FOCS'07). Existing work for weighted paging assumes that page weights are known in advance, which is not always the case in practice. … We present the first algorithm for online weighted paging that does not know page weights in advance, but rather learns from weight samples."* |
| 2 | Crossover-aware optimization selection | `NOT YET CHECKED` | — | — | — |
| 3 | Prefill/decode partition as a first-class decision | `NOT YET CHECKED` | — | — | — |
| 4 | Harvesting idle tensor cores at low batch | `NOT YET CHECKED` | — | — | — |
| 5 | Step-count reduction without a drafter | **`OCCUPIED`** | <https://arxiv.org/abs/2402.02057> — *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding* (Fu, Bailis, Stoica, Zhang; v1 03 Feb 2024) `[self]` | abs page **and** full HTML v1 fetched by me (`https://arxiv.org/html/2402.02057v1`, HTTP 200, 425,273 B); title read back ✓. Note: `html/2402.02057v2..v5` return HTTP 404 — only v1 has HTML. | *"Existing methods for accelerating LLM decoding often require a draft model (e.g., speculative decoding), which is nontrivial to obtain and unable to generalize. In this paper, we introduce Lookahead Decoding, an exact, parallel decoding algorithm that accelerates LLM decoding without needing auxiliary models or data stores. It allows trading per-step log(FLOPs) to reduce the number of total decoding steps"* |
| 6 | Output length as a control variable | `NOT YET CHECKED` | — | — | — |
| 7 | Admission under jointly-unknown length and KV growth | `NOT YET CHECKED` | — | — | — |

Raw-file pointers for §1 quote re-verification:

- Candidate 1: `.s3b_mech7/raw/abs_2410.21266.html` (`<meta name="citation_title" content="Online Weighted Paging with Unknown Weights">`); classical neighbour `.s3b_mech7/raw/gds_usenix.html` (USENIX USITS'97, Cao & Irani, *Cost-Aware WWW Proxy Caching*).
- Candidate 5: `.s3b_mech7/raw/abs_2402.02057.html`, `.s3b_mech7/raw/la_v1.html` (text dump `.s3b_mech7/raw/la_v1.txt`).

---

## 2. Candidate 5 — `OCCUPIED` (worked to completion; no §3 novelty sentence is claimed)

Candidate 5 asks for methods that emit >1 token per forward pass **without** a separate draft model **or** an added verification pass, targeted at the per-step fixed overhead. I attempted to prove this exists and it does, on the exact qualifier.

**Strongest occupant — Lookahead Decoding, arXiv 2402.02057.** Verbatim passages, all greppable in `.s3b_mech7/raw/la_v1.txt`:

1. (abstract) *"Autoregressive decoding of large language models (LLMs) is memory bandwidth bounded, resulting in high latency and significant wastes of the parallel processing power of modern accelerators."* — this is the candidate-4 premise (idle compute at low batch) stated as the paper's own motivation.
2. (abstract) *"Existing methods for accelerating LLM decoding often require a draft model (e.g., speculative decoding), which is nontrivial to obtain and unable to generalize. In this paper, we introduce Lookahead Decoding, an exact, parallel decoding algorithm that accelerates LLM decoding without needing auxiliary models or data stores. It allows trading per-step log(FLOPs) to reduce the number of total decoding steps"* — draft-model-free **and** step-count reduction, in one sentence.
3. (body) *"Lookahead Decoding takes advantage of the particular characteristics of autoregressive decoding, which is bounded by the memory bandwidth–as each generated token depends on all tokens before it–rather than compute, by using the available cycles to generate and verify n-grams (subsequent tokens) at virtually no additional cost."* — "using the available cycles" is the idle-compute-harvesting mechanism verbatim.
4. Section heading: *"3.3 Decode, Predict, and Verify in The Same Step"*, and §3.3 body: *"At execution, the lookahead and verification branches can be integrated into one decoding step to leverage parallel processing."* — this defeats the "no added verification pass" qualifier: verification is fused into the same step, so Lookahead is not excluded by it.
5. (body §1 contributions) *"We design Lookahead Decoding, a new lossless, parallel decoding algorithm to accelerate LLM inference without needing any auxiliary component."*

**Why the qualifiers do not carve out novelty.** Every clause of candidate 5 is met by an artifact that already exists, is peer-reviewed (ICML 2024), and reports measured end-to-end speedups: *"Our implementation of Lookahead Decoding can speed up autoregressive decoding by up to 1.8x on MT-bench and 4x with strong scaling on multiple GPUs in code completion tasks."*

**Corroborating occupants in the same slot** (each title read back from its own `arxiv.org/abs/<id>` page by me this session; each saved as `.s3b_mech7/raw/abs_<id>.html`):

| URL | State verified | Verbatim quote | What it covers |
|---|---|---|---|
| <https://arxiv.org/abs/2603.17942> — *Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing* (2026/03/18; ICML 2026 poster #61857, poster page <https://icml.cc/virtual/2026/poster/61857> fetched `[self]`, saved `.s3b_mech7/raw/icml_esp.html`) | preprint + ICML 2026 accepted; abs title read back ✓; full HTML fetched (`.s3b_mech7/raw/html_2603.17942.html`, 460,440 B) | *"Traditional autoregressive decoding generates one token per step, leaving substantial compute underutilized."* / *"enabling parallel future-token prediction without modifying weights or relying on draft models"* / *"yielding lossless decoding while significantly reducing model calls and increasing token throughput"* / Appendix G.1: *"Our method achieves the highest reduction in model forward passes"* | **2026, draft-free, training-free, explicitly counts model forward calls as the metric** (`"model calls ∝ 1/τ"`). Beats LADE by 7–11% acceptance length on LLaMA3 / 7–8% on Qwen3. |
| <https://arxiv.org/abs/2305.10427> — *Accelerating Transformer Inference for Translation via Parallel Decoding* (2023/05/17) | abs fetched, title read back ✓ | *"We propose to reframe the standard greedy autoregressive decoding of MT with a parallel formulation leveraging Jacobi and Gauss-Seidel fixed-point iteration methods for fast inference. This formulation allows to speed up existing models without training or modifications while retaining translation quality."* | Jacobi-style parallel decoding — the exact example the brief names. No draft model, no verification pass. |
| <https://arxiv.org/abs/1811.03115> — *Blockwise Parallel Decoding for Deep Autoregressive Models* (2018/11/07) | abs fetched, title read back ✓ | *"we propose a novel blockwise parallel decoding scheme in which we make predictions for multiple time steps in parallel then back off to the longest prefix validated by a scoring model … achieving iteration reductions of up to 2x over a baseline greedy decoder with no loss in quality"* | The 2018 origin of the mechanism; the "iteration reduction" metric is step-count reduction verbatim. |
| <https://arxiv.org/abs/2410.06916> — *SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration* (2024/10/09) | abs fetched, title read back ✓ | *"SWIFT does not require auxiliary models or additional training, making it a plug-and-play solution"* | Draft-free; layer-skipping instead of a draft model. |
| <https://arxiv.org/abs/2404.12022> — *Parallel Decoding via Hidden Transfer for Lossless Large Language Model Acceleration* (2024/04/18) | abs fetched, title read back ✓ | *"we propose a novel parallel decoding approach, namely hidden transfer, which decodes multiple successive tokens simultaneously in a single forward pass"* | Multiple tokens per single forward pass, lossless. |
| <https://arxiv.org/abs/2512.14681> — *Fast and Accurate Causal Parallel Decoding using Jacobi Forcing* (2025/12/16, hao-ai-lab, ICML 2026) | abs fetched, title read back ✓ | *"multi-block decoding with rejection recycling, which enables up to 4.5x higher token acceptance count per iteration and nearly 4.0x wall-clock speedup, effectively trading additional compute for lower inference latency"* | **"trading additional compute for lower inference latency"** — the candidate-4/5 trade stated as the design objective. Note: the GitHub repo renders the title as *"Jacobi Forcing: Fast and Accurate Diffusion-style Decoding"*, which differs from the arXiv title — a title-collision instance worth recording. |
| <https://arxiv.org/abs/2401.10774> — *Medusa* (2024/01/19) | abs fetched, title read back ✓ | *"While methods such as speculative decoding have been suggested to address this issue, their implementation is impeded by the challenges associated with acquiring and maintaining a separate draft model."* / *"By leveraging parallel processing, Medusa substantially reduces the number of decoding steps required."* | Heads on the frozen backbone instead of a separate draft model; test-time verification fused into the same step. |
| <https://arxiv.org/abs/2404.19737> — *Better & Faster Large Language Models via Multi-token Prediction* (2024/04/30) | abs fetched, title read back ✓ | — (title read back only; abstract not transcribed) | Trained-in multi-token prediction. |

**Residual I did *not* find occupied** (stated as an observation, not as novelty): Lookahead's own limitations section states the crossover condition qualitatively but supplies **no rule** for it. Verbatim (`.s3b_mech7/raw/la_v1.txt`): *"Since the LLM decoding is memory bandwidth-bound rather than compute-bound, these extra FLOPs only turn into a limited wall-clock slowdown for each step. Given this, Lookahead Decoding needs large surplus FLOPs to obtain high speedups. Running in compute-bound environments (e.g., serving with a large batch size) may cause slowdowns."* The paper's `(N, W, G)` window is a static hyper-parameter; ESP (2603.17942) adapts the tree by *acceptance probability*, not by measured surplus FLOPs. **I did not find** a draft-free multi-token method that sets its per-step token budget from the measured (batch, context) crossover. This is a thin residual and is *not* offered as a novelty claim here — the mechanism the candidate describes is occupied.

**Search log (candidate 5).** Queries issued verbatim: `lookahead decoding break sequential dependency Jacobi parallel decoding no draft model`; `multi-token prediction without draft model single forward pass multiple tokens`; `parallel decoding Jacobi iteration autoregressive transformer no verification pass`; `Medusa multiple decoding heads without draft model step reduction`; `reduce number of sequential decode steps LLM inference fixed per-step overhead amortize`; `self-speculative decoding layer skip no separate draft model`; `blockwise parallel decoding Stern 2018 greedy decoding multiple tokens per step`; `speculative decoding without verification pass parallel generation`; `"number of decoding steps" reduce step count fixed per-step overhead latency LLM`; `adaptive lookahead window size batch size speculative decoding slowdown compute-bound`; `parallel decoding survey multi-token prediction training-free 2026`; `Jacobi forcing parallel decoding 2026`; `lookahead decoding adaptive window size batch aware disable high batch`; `parallel decoding slow down large batch compute bound lookahead Jacobi limitation`; `training-free parallel decoding survey 2025 2026 multi-token no draft model`; `reduce steps amortize kernel launch overhead LLM decode latency`; `exact parallel decoding without draft model multiple tokens per step 2026`; `"decode multiple tokens per forward pass" LLM acceleration no auxiliary model`; `"waste" idle tensor cores low batch LLM decode exploit spare compute paper`; `"per-step" fixed overhead launch cost decode reduce number of steps amortize LLM serving`; `dynamic selection number of tokens per forward pass based on batch size parallel decoding`; `evaluate lookahead decoding at high batch size slowdown measurement`.

URLs I fetched myself for candidate 5: `arxiv.org/abs/{2402.02057, 2305.10427, 2401.10774, 2404.19737, 1811.03115, 2410.01699, 2603.17942, 2309.08168, 2404.16710, 2402.05109, 2311.08252, 2410.06916, 2404.12022, 2512.14681, 2605.16941, 2607.20467, 2505.15141, 2505.24584}`, `arxiv.org/html/2402.02057{v1,v2,v3,v4,v5}`, `arxiv.org/html/2603.17942v1`, `icml.cc/virtual/2026/poster/61857`, `github.com/hao-ai-lab/JacobiForcing`, `raw.githubusercontent.com/hao-ai-lab/JacobiForcing/{main,master}/README.md`, `snowflake.com/en/blog/engineering/jacobi-forcing-casual-parallel-decoding/`.

**Search-engine defects caught this session (candidate 5).** `web_search` returned `https://arxiv.org/pdf/2505.24584` as a snippet about causal left-to-right generation; I read the abs page back and it is *"AutoChemSchematic AI: Agentic Physics-Aware Automation for Chemical Manufacturing Scale-Up"* (2025/05/30) — a **5th fabrication/mismatch instance** in this project. It is not cited. `2410.01699` (*"Accelerating Auto-regressive Text-to-Image Generation with Training-free Speculative Jacobi Decoding"*, verified) is a domain-adjacent occupant only: it targets text-to-image, not text LLMs, and is labelled as such wherever it might otherwise be mistaken for LLM prior art. `2605.16941` (*Roll Out and Roll Back: Diffusion LLMs are Their Own Efficiency Teachers*) and `2607.20467` (*DC-Leap*) are diffusion-LLM items — a different model class from autoregressive Qwen3-4B — and are recorded, not cited as occupants.

---

## 3. What I could NOT verify (this revision)

- **Candidates 2, 3, 4, 6, 7 have not been adjudicated yet** in this revision. `NOT YET CHECKED` is not a negative finding.
- **Subagent-sourced rows are not yet re-grepped by me.** Any row that later appears marked `[sub-A]`/`[sub-B]`/`[sub-C]` is a claim by a subagent that has not yet been independently re-verified against its saved raw file.
- **ACM Digital Library is HTTP 403 from this box.** `https://dl.acm.org/doi/10.1145/3793230.3837769` (*To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems*, SYSTOR) was **not retrieved** — I have only its title and a search snippet. Do not treat it as read.
- **`cachee.ai` blog is HTTP 403** (*Cost-Aware Eviction: Why Your Cache Should Know What Things Cost*, 2026-03-28) — not retrieved.
- **`api.github.com` is rate-limited to zero here** and was not used. `export.arxiv.org` (429) and `arxiv.org/search` (406) were avoided per project rules.
- **`arxiv.org/html/2402.02057v2..v5` return HTTP 404**; only v1 HTML exists. Any claim about a *later* revision of Lookahead Decoding is therefore unverified here; all quotes are from v1 as served at `arxiv.org/html/2402.02057v1`.
- **No absence claim in this document is a global negative.** Every one is "I did not find it".

---

## 4. Search log (all candidates — this revision)

**Surfaces used.** `web_search`; `curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "<Chrome 126 UA>"` against `arxiv.org/abs/<id>`, `arxiv.org/html/<id>[vN]`, `raw.githubusercontent.com/...`, `github.com/<o>/<r>[/blob/...]`, `icml.cc/virtual/...`, `neurips.cc`, `papers.nips.cc`, `usenix.org/legacy/...`, `snowflake.com`. `web_fetch` deliberately not used. `api.github.com`, `export.arxiv.org`, `arxiv.org/search`, `dl.acm.org`, `cachee.ai` all unusable from this box (see §3).

**Queries issued so far (verbatim).**
1. `lookahead decoding break sequential dependency Jacobi parallel decoding no draft model`
2. `multi-token prediction without draft model single forward pass multiple tokens`
3. `parallel decoding Jacobi iteration autoregressive transformer no verification pass`
4. `Medusa multiple decoding heads without draft model step reduction`
5. `reduce number of sequential decode steps LLM inference fixed per-step overhead amortize`
6. `self-speculative decoding layer skip no separate draft model`
7. `blockwise parallel decoding Stern 2018 greedy decoding multiple tokens per step`
8. `speculative decoding without verification pass parallel generation`
9. `"number of decoding steps" reduce step count fixed per-step overhead latency LLM`
10. `adaptive lookahead window size batch size speculative decoding slowdown compute-bound`
11. `parallel decoding survey multi-token prediction training-free 2026`
12. `Jacobi forcing parallel decoding 2026`
13. `lookahead decoding adaptive window size batch aware disable high batch`
14. `parallel decoding slow down large batch compute bound lookahead Jacobi limitation`
15. `training-free parallel decoding survey 2025 2026 multi-token no draft model`
16. `reduce steps amortize kernel launch overhead LLM decode latency`
17. `exact parallel decoding without draft model multiple tokens per step 2026`
18. `"decode multiple tokens per forward pass" LLM acceleration no auxiliary model`
19. `"waste" idle tensor cores low batch LLM decode exploit spare compute paper`
20. `"per-step" fixed overhead launch cost decode reduce number of steps amortize LLM serving`
21. `dynamic selection number of tokens per forward pass based on batch size parallel decoding`
22. `evaluate lookahead decoding at high batch size slowdown measurement`
23. `GreedyDual-Size cost-aware cache replacement retrieval cost per byte`
24. `cache replacement policy with recomputation cost eviction decision marginal cost`
25. `KV cache eviction recomputation cost aware prefill load aware policy`
26. `weighted caching problem time-varying retrieval cost competitive algorithm`
27. `caching with varying fetch cost non-stationary weighted paging`
28. `"To Keep or Not to Keep" learning KV cache retention disaggregated serving`
29. `KV cache retention policy recompute cost versus residency cost LLM serving`
30. `"Online Weighted Paging with Unknown Weights" arXiv`
31. `weighted paging unknown weights learning fetch cost online algorithm NeurIPS`

**Anchor verified.** <https://arxiv.org/abs/2605.01280> — title read back at the abs page by me `[self]`: *"Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics"* (Zijie Zhou, 2026/05/02). The brief's paraphrase of its four ignored structural features matches the abs abstract verbatim: *"These general-purpose policies ignore the distinctive structure of LLM inference--dynamically growing KV cache memory, prefill-decode phase asymmetry, unknown output lengths, and continuous batching constraints."*

**Fetched raw artifacts (this revision).** 18 arXiv abs pages for candidate 5 + `2410.21266` + `2607.19214` + `2606.16824` + `2601.23278`; `la_v1.html`; `html_2603.17942.html`; `icml_esp.html`; `gh_jacobiforcing.html`; `jf_main.md`; `gds_usenix.html`. All under `.s3b_mech7/raw/`.

**Absences observed (phrased as observations).** I did **not find** a draft-free parallel-decoding artifact whose per-step token budget is chosen from a measured bandwidth→compute crossover. I did **not find** a live-load-priced KV eviction policy (one whose recompute price is a shadow price read from the scheduler's current prefill/decode mix). I did **not** retrieve the SYSTOR KV-retention paper or the cost-aware-eviction blog (403).
