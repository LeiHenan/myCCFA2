# Speculative Decoding — Draft Budget / Speculation-Length Adaptation & Acceptance-Rate Behaviour
Prior-art / gap-mapping sweep. Compiled 2026-09-13. Window prioritised: 2025-06 → 2026-09.

Legend: **READ BODY** = page/abstract content retrieved and read. **TITLE ONLY** = only a search-result title/snippet was seen.

---

## A. CLOSED (someone shipped or published a working answer)

| # | Question it answers | Who closed it | Artifact (paper/PR/flag) | URL | Status | READ BODY or TITLE ONLY |
|---|---|---|---|---|---|---|
| A1 | Can speculation depth K be selected automatically per step from batch size instead of a static `num_speculative_tokens`? | ekagra-ranjan (vLLM) | vLLM PR #32374 `[V1][Spec Decode] Add Dynamic SD` → `num_speculative_tokens_per_batch_size` | https://github.com/vllm-project/vllm/pull/32374 | **Merged** (vLLM main, 2026) | READ BODY |
| A2 | Goodput-optimal K: `goodput = AL / ITL`, AL=f(K), ITL=f(K,BS), offline-profiled (BS,K) ITL table with linear interpolation, warmup then live acceptance-rate | ekagra-ranjan | vLLM PR #32374 (design section + `generate_config.py` + `dynamic_speculative_config.json`) | https://github.com/vllm-project/vllm/pull/32374 | Merged | READ BODY |
| A3 | How do I actually use a K-vs-batch-size schedule? | vLLM docs | `num_speculative_tokens_per_batch_size` schema `[start_bs, end_bs, optimal_K]` | https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ | Shipped (docs dated July 7, 2026) | READ BODY |
| A4 | Does dynamic K survive full CUDA graphs? | vLLM (MRV2) | PR #45953 `[MRV2][SD] Make Dynamic SD comatible with Full Cuda Graphs`; `VLLM_USE_V2_MODEL_RUNNER=1` | https://github.com/vllm-project/vllm/pull/45953 | Merged | TITLE ONLY |
| A5 | Per-step, per-(request, position) adaptive **verification** budget — score each draft slot by survival probability, admit best until budget spent (slots compete *across* requests) | benchislett (Red Hat), Lucas Wilkinson | vLLM PR #47808 `[Spec Decode] DSpark confidence-scheduled verification`, flag `enable_adaptive_verification` | https://github.com/vllm-project/vllm/pull/47808 | **Merged 2026-08-12** | READ BODY |
| A6 | Same, officially explained with the cost model: `B = argmax_B (1 + Σ survival of B best slots) / cost(T+B)`; costs profiled at startup, monotonic-forced, np.argmax over cumsum | vLLM Team | vLLM blog `Adaptive Verification in vLLM: DSpark confidence-scheduled verification` | https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification | Published 2026-08-14 | READ BODY |
| A7 | Same, operator-facing docs incl. limitations | vLLM docs | `Adaptive Verification` page | https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/ | Shipped (page dated Aug 25, 2026) | READ BODY |
| A8 | Can EMA-of-acceptance-length drive `speculative_num_steps` at runtime, with hysteresis? | SGLang | `speculative_adaptive` + `adaptive_spec_params.py`; formula `target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)`; defaults `ema_alpha=0.2, update_interval=5, warmup_batches=10, down_hysteresis=-0.25, up_hysteresis=0.0` | https://github.com/sgl-project/sglang/blob/c69844f0/python/sglang/srt/speculative/adaptive_spec_params.py | Shipped (source of record) | READ BODY |
| A9 | Per-batch-size independent adaptive state (own EMA, own candidate steps, own hysteresis per BS range) | maodoudou168; merged by Qiaolin-Yu | SGLang PR #24055 `[SPEC][5/N] feat: batchsize-aware support for adaptive speculative_num_steps` | https://github.com/sgl-project/sglang/pull/24055 | **Merged** (`merged commit 6b18095`) | READ BODY |
| A10 | Per-request acceptance statistics returned to the client (mean acceptance length, draft acceptance rate, histogram) | matthewkotila / vLLM | PR #48915 `[Frontend][Core][Spec Decode] Per-request acceptance stats in OpenAI API responses`; flag `--per-request-spec-decode-metrics {none,summary,detailed}` | https://github.com/vllm-project/vllm/pull/48915 · https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/ | Merged / Shipped | READ BODY (docs) |
| A11 | MoE per-request utility-driven K with a K=0 disable arm: `utility = tokens_accepted / verification_cost(k)`, test phase probes k, set phase locks K* | Saxena et al. (Georgia Tech / NVIDIA) | Cascade, MICRO 2025 / MLSys 2026 poster, arXiv 2506.20675 | https://arxiv.org/abs/2506.20675 · https://mlsys.org/virtual/2026/poster/10189 | Published | READ BODY |
| A12 | Closed-loop goodput controller for intra-request parallelism (profiles env, feedback algorithm, predicts goodput) | Liu et al. (UC Berkeley / SJTU) | TurboSpec / SmartSpec, arXiv 2406.14066 | https://arxiv.org/abs/2406.14066 | Published (v3 27 Jul 2025) | READ BODY |
| A13 | Optimal candidate length K is an MDP whose optimal policy is a **threshold policy**; train an acceptance-prediction head, stop when P(≥1 rejection) > threshold | Huang, Guo, Wang (Princeton) | SpecDec++, COLM 2025, arXiv 2405.19715 | https://arxiv.org/abs/2405.19715 | Published (COLM 2025) | READ BODY |
| A14 | Training-free early draft stopping via entropy-based lower bound on acceptance probability; robust at high sampling temperature | Agrawal, Jeon, Lee (Meta) | AdaEDL, arXiv 2410.18351 | https://arxiv.org/abs/2410.18351 | Published (NeurIPS 2024 ENLSP workshop) | READ BODY |
| A15 | Speculation amount should be a function of **sequence length**, not just batch: critical-sequence-length threshold + analytic model for optimal drafting strategy | Sadhukhan et al. | MagicDec, arXiv 2408.11049 | https://arxiv.org/abs/2408.11049 | Published | READ BODY |
| A16 | Log-linear scaling law for acceptance rate vs **decoding batch size** (plus pretraining tokens and draft capacity); instantiates Scylla | Yan et al. | Scaling Laws for Speculative Decoding, arXiv 2505.07858 | https://arxiv.org/abs/2505.07858 | Published | READ BODY |
| A17 | Task type is a **stronger predictor of acceptance than tree depth**; only chat consistently exceeds 1.0 accepted token/step; entropy–acceptance correlation negative but weak (rho ∈ [−0.20, −0.15]) | Saif Mahmoud | Acceptance Dynamics Across Cognitive Domains in Speculative Decoding, arXiv 2604.14682 | https://arxiv.org/abs/2604.14682 | Published | READ BODY |
| A18 | Theoretical upper bound on SD speedup (oracle proposed length = actual accepted length), and oracle method-combination headroom up to 2.2× | Xiaoxuan Liu, Jiaxiang Yu, Jongseok Park, Ion Stoica, Alvin Cheung | Speculative Decoding: Performance or Illusion?, arXiv 2601.11580v2 (v1 31 Dec 2025, v2 18 Mar 2026); MLSys 2026 | https://arxiv.org/abs/2601.11580 · https://mlsys.org/media/mlsys-2026/Slides/3782.pdf | Published (MLSys 2026) | READ BODY (arXiv HTML) |
| A19 | Per-step γ selection from draft-model confidence/entropy via a small MLP maximizing expected tokens per step; optimal γ shifts with **quantisation level** | Shikhar Shukla | SpecKV, arXiv 2605.02888 | https://arxiv.org/abs/2605.02888 | Published | READ BODY |
| A20 | Marginal-gain criterion for dynamic speculation length with diffusion drafters: extend only when acceptance gain > extra verification cost; monotone convergence proof | Lin et al. | LibraSpec, arXiv 2608.08721 | https://arxiv.org/abs/2608.08721 | Published | READ BODY |
| A21 | Online entropy-based controller selects speculation length by expected **step-wise efficiency** (long-context, sparse-KV self-speculation) | Liu et al. | SparseSpec-L, arXiv 2607.27735 | https://arxiv.org/abs/2607.27735 | Published | READ BODY |
| A22 | **Predict the accepted length of a cache draft before verification** and use the payoff estimate to pick the draft source | Su et al. (Intel Labs) | Hybrid Verified Decoding, arXiv 2606.01019 | https://arxiv.org/abs/2606.01019 | Published | READ BODY |
| A23 | Joint tree depth × width verification as conditional optimal transport; prefix acceptance probabilities act as dynamic scaling factors guiding horizontal draft selection | Weng, Hu, Yairi | UniVer, arXiv 2605.04543 | https://arxiv.org/abs/2605.04543 | Published | READ BODY |
| A24 | Adaptive speculation in vLLM via multi-armed bandits over speculative hyperparameters, with stopping-time regret bounds | — | BanditSpec / UCBSpec / EXP3Spec, arXiv 2505.15141 | https://arxiv.org/abs/2505.15141 | Published (ICML) | READ BODY |
| A25 | Bandit-based dynamic speculative decoding (meta-bandit over parameter-free dynamic strategies, training-free) | — | TapOut, arXiv 2511.02017 | https://arxiv.org/abs/2511.02017 | Published | READ BODY |
| A26 | Per-batch-size optimal speculation length via MAB + proactive speculation disable + draft-model offload | — | Nightjar, arXiv 2512.22420 | https://arxiv.org/abs/2512.22420 | Published (JSA) | READ BODY |
| A27 | Optimal stopping / threshold policy under delay drift, with UCB-SpecStop regret bounds | — | Delay-Adaptive Speculation Control, arXiv 2606.20591 | https://arxiv.org/abs/2606.20591 | Published | READ BODY |
| A28 | Confidence-aware adaptive token **tree** — "Acceptance Funnel": root needs width, deeper layers need depth; prune-while-expanding under a budget | — | TALON, arXiv 2601.07353 | https://arxiv.org/abs/2601.07353 | Published | READ BODY |
| A29 | Optimal tree **shape** (depth, width, verification scope) chosen against a hardware-profiled latency objective | — | Yggdrasil, NeurIPS 2025, arXiv 2512.23858 | https://arxiv.org/abs/2512.23858 | Published | READ BODY |
| A30 | Provably optimal best-first tree construction + hardware-calibrated online budget controller | — | Bastion, arXiv 2605.29727 | https://arxiv.org/abs/2605.29727 | Published | READ BODY |
| A31 | Per-request draft length under PIM, monitoring cumulative acceptance probabilities | Runze Wang et al. (HUST/UNSW) | SADDLE, HPCA 2026 main conference | https://2026.hpca-conf.org/details/hpca-2026-main-conference/112/Adaptive-Draft-Sequence-Length-Enhancing-Speculative-Decoding-Throughput-on-PIM-Enab | Published (HPCA 2026) | READ BODY |
| A32 | Production per-request confidence-scheduled verification in a frontier model system (DeepSeek-V4) | DeepSeek (32 authors) | DSpark, arXiv 2607.05147 | https://arxiv.org/abs/2607.05147 | Published | READ BODY |
| A33 | Adaptive thinking-budget routing (train two no-think drafts, accept on agreement, predict budget from draft entropy on disagreement) | Lee et al. (Korea Univ.) | DART, EMNLP 2026 Findings, arXiv 2606.23181 | https://arxiv.org/abs/2606.23181 | Published | READ BODY |
| A34 | Online draft-model adaptation to the query distribution (acceptance lift 0.1–0.65) | Liu et al. | Online Speculative Decoding / Disco, arXiv 2310.07177 | https://arxiv.org/abs/2310.07177 | Published (canonical) | READ BODY |
| A37 | No-regret **drafter selection** per query (reward = acceptance probability or expected acceptance length), competitive with the best drafter in hindsight | Liu et al. | Not-a-Bandit: Provably No-Regret Drafter Selection, arXiv 2510.20064 | https://arxiv.org/abs/2510.20064 | ICLR 2026 | READ BODY |
| A38 | Entropy → acceptance-rate proxy; dynamic draft length for long-form / reasoning generation | Zhang et al. | SVIP / "Draft Model Knows When to Stop", arXiv 2411.18462 | https://arxiv.org/abs/2411.18462 | EMNLP 2025 | READ BODY |
| A39 | Sample-adaptive draft **block size** predicted once from the prefill representation | — | BlockPilot, arXiv 2606.31315 | https://arxiv.org/abs/2606.31315 | Preprint | READ BODY |
| A40 | Goodput-optimal speculation under multi-SLO serving | Li et al. | AdaServe, arXiv 2501.12162 | https://arxiv.org/abs/2501.12162 | Preprint | READ BODY |
| A41 | Goodput-optimal speculation, edge-cloud distributed | — | WISP, arXiv 2601.11652 | https://arxiv.org/abs/2601.11652 | Preprint | READ BODY |
| A42 | Request-level speculation length via confidence-prior verifier under load/SLO fluctuation | — | AdaSpec (SpecServe), arXiv 2503.05096 | https://arxiv.org/abs/2503.05096 | ACM SoCC 2025 | READ BODY |
| A43 | Batch-level draft-token selection maximizing total throughput at fixed capacity | Wu et al. | TETRIS, arXiv 2502.15197 | https://arxiv.org/abs/2502.15197 | Preprint | READ BODY |
| A44 | Goodput-optimal config search. Verbatim: "goodput is maximised by the smallest, fastest draft model at device-dependent speculative lengths (K*=2-10)"; "no single fixed configuration can simultaneously optimise all objectives" | — | ConfigSpec, arXiv 2604.09722 | https://arxiv.org/abs/2604.09722 | Preprint | READ BODY |
| A45 | Benchmark exposing batch-size-dependent optimal draft lengths. Verbatim: "identifying batch-size dependent optimal draft lengths and biases in low-diversity data" | — | SPEED-Bench, arXiv 2604.09557 | https://arxiv.org/abs/2604.09557 | ICML 2026 | READ BODY |
| A46 | Adaptive γ controller over RL reasoning rollouts, with system-aware SD toggle | — | EfficientRollout, arXiv 2606.18967 | https://arxiv.org/abs/2606.18967 | Preprint | READ BODY |
| A47 | Benchmark of SD for test-time scaling. Verbatim: "simple n-gram-based methods effectively capture repetitive patterns" | — | arXiv 2509.04474 | https://arxiv.org/abs/2509.04474 | Preprint | READ BODY |
| A48 | The **original** SGLang adaptive `speculative_num_steps` for EAGLE topk=1 (root of the SGLang adaptive line) | alphabetc1 | SGLang PR #21599 `[SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1` | https://github.com/sgl-project/sglang/pull/21599 | **Merged** | READ BODY (state + title) |
| A49 | SGLang adaptive spec scope/limits, operator-facing, incl. a negative caveat. Verbatim: **"If your workload is already stable and one static setting is well tuned, adaptive mode may not help much"**; **"At high batch sizes, narrower ladders (e.g., [1, 2] or [1] ) often outperform wide ones"**; only EAGLE/EAGLE3 and only `--speculative-eagle-topk 1`, else "SGLang falls back to static speculative settings" | SGLang | `--speculative-adaptive` / `--speculative-adaptive-config` docs | https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding | Shipped docs | READ BODY |
| A50 | Adaptive spec on NPU | EanWang211123 | SGLang PR #25644 `[Speculative] [NPU] Adaptive-SD NPU support` | https://github.com/sgl-project/sglang/pull/25644 | **Merged** | TITLE ONLY |
| A51 | Adaptive spec conflicts with hybrid GDN (Qwen3.5) | EanWang211123 | SGLang PR #23331 `[BugFix] Resolve adaptive speculative decoding conflicts for Qwen3.5 (hybrid GDN)` | https://github.com/sgl-project/sglang/pull/23331 | **Merged** | TITLE ONLY |
| A52 | Dynamic draft length in TensorRT-LLM, one-model spec-decode path | zheyuf (NVIDIA) | TRT-LLM PR #10860 `[TRTLLM-10319][feat] Dynamic draft length on spec decode one-model path` | https://github.com/NVIDIA/TensorRT-LLM/pull/10860 | **Merged** | TITLE ONLY |
| A53 | Dynamic draft length in TensorRT-LLM, stage 1 | zheyuf (NVIDIA) | TRT-LLM PR #8194 `[TRTLLM-8136][feat] Dynamic draft length in spec decode (stage 1)` | https://github.com/NVIDIA/TensorRT-LLM/pull/8194 | **Merged** | TITLE ONLY |
| A54 | Correct CUDA graph selection under dynamic spec decode (TRT-LLM) | ziyixiong-nv | TRT-LLM PR #7728 `[fix] Use the correct cuda graph for dynamic spec dec` | https://github.com/NVIDIA/TensorRT-LLM/pull/7728 | **Merged** | TITLE ONLY |
| A55 | Per-request K on the DSpark path exists in SGLang as first-class runtime metadata. Verbatim: **"SGLang main ships per-request-K DSpark verification (srt/speculative/ragged_verify.py , dspark_components/dspark_planner.py )"** … **"verify_lens is produced by a planner separate from the proposer and treated as first-class runtime metadata."** | SGLang | Quoted by LJX-xixi in vLLM RFC #48202 | https://github.com/vllm-project/vllm/issues/48202 | Shipped in SGLang main (reported) | READ BODY (of the reporting thread) |
| A35 | vLLM Dynamic SD not compatible with data parallelism — engine auto-disables the schedule under DP | vLLM docs | DSD docs `Limitations` | https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ | Shipped behaviour (by design) | READ BODY |
| A36 | CPU-backend DSD stream-override bug — fixed by a different route | jmamou | PR #47160 (closed) / #47162 (merged) `[CPU] Remove speculative decoding stream overrides from CPUModelRunner` | https://github.com/vllm-project/vllm/pull/47012 (see closing comment) | Superseded & merged via #47162 | READ BODY (via #47012) |

**Note on PRISM and HELIOS (task lead).** Both are **NOT** adaptive-speculation-length work — flag as a red herring in the lead list:
- PRISM = *Parametrically Refactor Inference for Speculative Decoding Draft Models*, MLSys 2026 Oral, https://mlsys.org/virtual/2026/oral/3789 — a **draft-model architecture** paper (decouple capacity from inference cost; reports >2.6× throughput on an optimised engine). Does not do adaptive K. READ BODY.
- HELIOS = *Adaptive Model And Early-Exit Selection for Efficient LLM Inference Serving*, MLSys 2026 Oral, https://mlsys.org/virtual/2026/oral/3846, arXiv 2504.10724 — **early-exit LLM model switching**, not speculation-length adaptation. READ BODY.

---

## B. OPEN (explicitly unsolved, with evidence someone is still asking)

### B1. Per-request / non-uniform proposal lengths inside one batch
- **The contract question**: "representing different effective proposal lengths across requests within the same batch."
- Who: LJX-xixi, vLLM RFC #48202 (Jul 10 2026). https://github.com/vllm-project/vllm/issues/48202 — **now Closed**, closed silently with no maintainer comment; the RFC was not answered.
- What is missing (author's own words): "The missing piece is not a policy but a contract: per-request proposal-length metadata and its semantics across the proposer, model runner, scheduler accounting, async scheduling, and metrics. This is also the interface boundary left unresolved by several earlier adaptive-speculation attempts ( #28693 , #26504 , #44885 , #43522 )."
- Also: "modifying the per-request DraftTokenIds lists changes effective truncation when async scheduling is disabled, but does not change verification or scheduler accounting on the default async path".
- The deferred item it inherits from, PR #35301 (still **Open**): "a per-request policy would be more efficient but requires variable-length draft outputs and more complex KV-cache bookkeeping." https://github.com/vllm-project/vllm/pull/35301
- The original closed attempt: PR #26504 (closed unmerged) — https://github.com/vllm-project/vllm/pull/26504
- Live duplicate-ish request: issue #17984 `[Feature]: Per-sequence speculative decoding` — **Closed as not planned**, labels `feature request`, `stale`. https://github.com/vllm-project/vllm/issues/17984

### B2. Where should the K decision live — batch-keyed array index vs policy object?
- Who: seongyun1104, vLLM RFC #54749 (Sep 1 2026, **Open**). https://github.com/vllm-project/vllm/issues/54749
- Verbatim: "On current main ( 504bb8b0c3 ) the dynamic-SD K decision is one array index… Six different signals are being proposed for this one decision right now. Batch size is what ships." and "I am not asking for any of them to be adopted. I am asking whether the K decision should keep being a batch-keyed array index, given that the next six proposals all need it not to be."
- The six signals named: context length (#48944), token entropy (#54082), marginal gain (LibraSpec, 2608.08721), expert activation cost (EcoSpec, 2607.12696), step-wise efficiency (SparseSpec-L, 2607.27735), learned verification payoff (Hybrid Verified Decoding, 2606.01019).
- Verbatim on the shipped controller's blind spot: "So context is not absent from the shipped controller. It is a deployment-wide scalar set before serving , not an input the policy reads per step — which is the distinction this RFC exists to ask about". And: "the person who owns this decision has implemented it twice, three months apart, in two subsystems, and both times context was priced once at startup rather than read as an input. @LucasWilkinson — is that a deliberate boundary?"
- Explicit statement of what is missing: "nothing in the engine currently measures the quantity these policies need. Acceptance is exported in aggregate, but per- (batch bucket, ctx bucket, K) draft and verify latency, first-rejection position, committed tokens per verification, and K-keyed graph hit/miss are not." and "If only one thing comes out of this thread, that telemetry is the one I would pick".

### B3. Context-length-dependent K (the ctx axis)
- Who: seongyun1104. vLLM PR #48944 `[Spec Decode] Context-length-aware K in DSD (RFC #48627)` — **Open**, `needs-rebase`. https://github.com/vllm-project/vllm/pull/48944
- Sibling SGLang implementation: PR #31716 `[Spec] Add a ctx axis to the adaptive spec _route (BS × ctx → slot)` — **Open**. https://github.com/sgl-project/sglang/pull/31716
- Verbatim gap: "Under memory-bound decode with a shared prefix (RAG serving, agentic sessions, batched summarization), the optimal K depends on ctx as well as batch. A batch-only lookup has to pick one K per BS, either leaving ctx-amortization on the table for long sequences or over-drafting short ones."
- Verbatim on the SGLang side: "End-to-end tok/s comparison against a batch-only palette on real workload traces is out of scope here and will be reported in a follow-up once a workload harness is agreed on."
- Related: #54801 `num_speculative_tokens_per_seq_len` prompt by #54691; RFC #48627.

### B4. Per-request ability to disable/skip speculation (long-context collapse)
- Who: zxman0126, vLLM issue #52258 (Aug 14 2026, **Open**). https://github.com/vllm-project/vllm/issues/52258
- Verbatim: "long-document summarization (20K–73K Chinese meeting transcripts) collapses to 5–8% acceptance, making speculation net-negative (decode drops below the no-spec baseline). With no per-request toggle in V1 and the dynamic schedule ( num_speculative_tokens_per_batch_size ) keyed on batch size only, there is currently no way to keep speculation for short requests while disabling it for long ones on one engine".
- Verbatim no-op finding: "the config plumbing survives — SpeculativeConfig.max_model_len is still accepted… but we could not find any runtime consumer under vllm/v1/ , and empirically the field is a no-op."
- Hardware note in-thread: "unified-memory hardware (GB10 128GB) cannot co-load a second engine as a workaround."
- Follow-up bug on RTX PRO 6000 / sm_120: issue #52756 `[Bug]: V1 MTP max_model_len skip still schedules drafts with async scheduling` (Aug 18 2026, **Open**) — https://github.com/vllm-project/vllm/issues/52756 — verbatim: "The eligibility check also uses the batch-wide common_attn_metadata.max_seq_len . Therefore, one over-limit sequence makes speculation stop for every request in the same GPU batch".

### B5. MoE adaptive speculation depth — not upstreamed
- Who: bananighosh, vLLM issue #44506 `[Feature][Spec-Decode]: Cascade: Utility-Driven Adaptive k for MoE Speculative Decoding in V1` (Jun 4 2026, **Open**, labels `feature request` + `stale`). https://github.com/vllm-project/vllm/issues/44506
- Verbatim: "In vllm/v1/core/sched/scheduler.py , num_spec_tokens is set once at engine init and never changes… Today, there exists no mechanism to vary k per-request or disable speculation for a specific request mid-flight without restarting the engine."
- Verbatim: "In vllm/v1/spec_decode/llm_base_proposer.py , propose() has no per-request state. There is nowhere to track acceptance history, utility, or current k per request — the entire batch gets the same num_spec_tokens ."
- Parallel RFC, still **Open**: vLLM #46295 `[RFC]: Adaptive Speculation Depth for MoE Models via Per-Iteration Utility Budgeting` (Jun 21 2026) — https://github.com/vllm-project/vllm/issues/46295

### B6. An accurate, lightweight acceptance-length predictor
- Who: the Illusion paper authors, as their stated future work. https://arxiv.org/abs/2601.11580
- Verbatim: "Taken together, these results point to a promising direction for future work: developing an accurate yet lightweight predictor capable of adapting to varying levels of acceptance behavior across workloads, requests and token positions."
- Same gap stated independently by TurboSpec/SmartSpec: "We leave it as future work to develop an accurate and efficient predictor for accepted length." https://arxiv.org/abs/2406.14066
- And by SVIP: "The acceptance rate bound in Equation 6 could be overly conservative… we aim to investigate tighter bounds on the acceptance rate". https://arxiv.org/abs/2411.18462
- And SpecDec++: "We leave for future work the adaptation of our analysis and technique to the large-batchsize, long-context settings." https://arxiv.org/abs/2405.19715

### B7. No telemetry to calibrate a K controller from inside the engine
- Who: seongyun1104. vLLM PR #54748 `[Spec Decode] Make the dynamic SD schedule observable` — **Open**, `needs-rebase`. https://github.com/vllm-project/vllm/pull/54748
- Verbatim: "A served dynamic-SD config is opaque in two directions… The resolved schedule is never printed … Steps the schedule sent to K=0 increment no counter".
- Verbatim on the consequence: "a dynamic schedule that is working as configured and has stopped speculating because the batch is large" vs "speculative decoding that is silently not running at all" "produce identical metrics".
- Verbatim on why: "I have been measuring dynamic-SD schedules ( #48627 , #48944 ) and trying to calibrate a cost model that picks K. The coefficients were guesses and the model picked the wrong tier; there is no counter that would have told me so from inside vLLM."

### B8. Whether the cost model itself can find the right tier
- Who: seongyun1104, RFC #54749 (verbatim): "A cost model over the same interface gets that cell wrong… Run with the coefficients it ships, sweeping only the acceptance prior across 0.35–0.75 (the campaign observed 44–47 % in the K=3 tier), it selects K=0 at the high-batch tier at every context . It reproduces the batch-only schedule and would not have found the cell worth 29–36 %." https://github.com/vllm-project/vllm/issues/54749

### B9. Adaptive speculation with EAGLE topk > 1
- Who: elwhyjay, SGLang PR #24054 `[Speculative] Support topk>1 in adaptive speculative decoding` — **Open** (8 commits). https://github.com/sgl-project/sglang/pull/24054
- Corroborating shipped restriction, verbatim from source: `"speculative_eagle_topk={...} (only topk=1 is supported)"` in `adaptive_unsupported_reason()`. https://github.com/sgl-project/sglang/blob/c69844f0/python/sglang/srt/speculative/adaptive_spec_params.py

### B10. Latency–throughput joint optimisation of speculation + batching
- Verbatim: "extending the framework to the latency-throughput joint optimization space, where speculative decoding and batch scheduling are solved together, remains an open line of research." — Yggdrasil, https://arxiv.org/abs/2512.23858

### B11. Reasoning / thinking-phase vs answer-phase speculation budget
- **No paper found** that budgets speculation separately for the thinking phase vs the answer phase. Closest evidence:
  - Illusion paper's n-gram position-decay observation (thinking→answer transition) — https://arxiv.org/abs/2601.11580
  - SVIP's QwQ long-form-reasoning draft-length result — https://arxiv.org/abs/2411.18462
  - DART (thinking *token* budget routing, not speculation budget) — https://arxiv.org/abs/2606.23181
  - Benchmark gap stated verbatim: "its efficacy in the structured, repetition-rich context of test-time scaling remains largely unexplored" — arXiv 2509.04474, https://arxiv.org/abs/2509.04474

### B12. RL-rollout long-tail speculation
- Who: vLLM issue #41821 `[RFC]: Adaptive throughput/latency profile for RL rollout long-tail` — https://github.com/vllm-project/vllm/issues/41821
- Also named as a DSD use case in PR #32374: "During RL rollout where we start off with high BS but then end up with small BS due to very few long tail request".

### B14. Re-activation starvation after speculation is disabled
- Verbatim: **"Because these methods depend on historical speculative observations, disabling speculation also stops the collection of fresh acceptance statistics, which can make later reactivation less reliable."** — Nightjar, https://arxiv.org/abs/2512.22420

### B15. Verification cost as a function of batch size inside a controller
- Verbatim: **"TETRIS studies batch speculative decoding and shows that effective draft depth interacts with batching and hardware saturation; our framework could incorporate this by letting verification cost depend on batch size c_v(b), which we leave for future work."** — Delay-Adaptive Speculation Control, https://arxiv.org/abs/2606.20591

### B16. Contextual / non-stationary bandits for speculation
- Verbatim: **"Another direction is to explore contextual bandits, where the environment reveals additional information that can be leveraged to reduce the learning burden."** — BanditSpec, https://arxiv.org/abs/2505.15141

### B17. Tree depth-vs-width budget rule is model-dependent and not yet principled
- The only concrete published rule found, verbatim: **"for models with lower alignment like Qwen3-8B, a lower threshold (μ=0.01) yields the best performance by encouraging a 'shallow-and-wide' search to ensure coverage. In contrast, for highly aligned models like DSL-8B, a higher threshold (μ=0.04) is preferred to form 'deep-and-narrow' chains that maximize the speculation length."** — TALON, https://arxiv.org/abs/2601.07353
- Corroborating negative on static depth: see C31.
- Independent finding that depth may not even be the dominant axis: task type > tree depth (A17, arXiv 2604.14682).

### B13. User-side: "why is my acceptance rate this low / why no throughput gain"
- vLLM Forums thread 528, `Spec decode with eagle get very low Draft acceptance rate` (opened Apr 25 2025; still indexed). User reports K=7, temperature 0.5, Llama-3-8B + EAGLE: "Draft acceptance rate: 0.077 , System efficiency: 0.166… why the acceptance rate is so low, what did i do wrong?" https://discuss.vllm.ai/t/spec-decode-with-eagle-get-very-low-draft-acceptance-rate/528
- Related thread listed on that page: `[Spec Decode] Why does the acceptance rate look close to the paper, but the throughput is still not high?` (V1 Feedback, Apr 21 2025) — https://discuss.vllm.ai/t/spec-decode-why-does-the-acceptance-rate-look-close-to-the-paper-but-the-throughput-is-still-not-high/ (indexed from thread 528's related-topics block; TITLE ONLY)

### B18. Entropy-gated dynamic K ∈ {0, K} via real-time Shannon hysteresis — the most recent asking in the window
- Who: Creepybits, vLLM issue #54082 (Aug 27 2026). **Open**, labels `feature request`, `speculative-decoding`. https://github.com/vllm-project/vllm/issues/54082
- Verbatim: **"Standard speculative decoding in vLLM operates with a static draft length ( k = constant ). However, during high-entropy generation steps (e.g., complex reasoning forks, multi-lingual syntax, or high-perplexity code tokens), the token acceptance rate β → 0 . This leads to frequent draft rejection rollbacks, net throughput degradation ( S < 1 ), and wasted compute FLOPs/energy."**
- What is missing — the author asks for the hook point: **"Looking forward to feedback from the community and maintainers on the cleanest architectural hook point in vLLM v0.28+"**, proposing to toggle pre-captured CUDA graphs for k=0 vs k=K and to fuse entropy into sampling kernels "to prevent CPU-GPU host synchronization overhead".

### B19. Dynamic pruning of draft **trees** — no maintainer answer
- Who: supertanziang, discuss.vllm.ai thread 2630 (2026-05-08, 2 posts). https://discuss.vllm.ai/t/what-is-the-recommended-way-to-support-dynamic-pruning-for-speculative-decoding-draft-trees/2630
- Verbatim (asking): **"Would vLLM maintainers be open to supporting dynamic pruning of the draft tree at runtime, where low-confidence branches can be skipped based on token probabilities? I would like to understand whether this direction fits vLLM's speculative decoding roadmap, and what the preferred implementation approach would be before starting a PR."**
- Verbatim (current state): **"the number of draft tokens is effectively fixed by the static tree topology, regardless of how confident or uncertain the draft model is at each node."**
- The **only** reply is from the RunLLM bot, not a maintainer: **"Dynamic pruning—where branches are skipped based on token or path probabilities—is not yet implemented"**. No maintainer answered.
- The corresponding GitHub issue, vLLM #41823 `[Feature]: Support Dynamic Pruning for Speculative Decoding Draft Trees in EAGLE-3`, was **closed as not planned** (see C32).

### B20. Host-side planning of verify lengths conflicts with GPU-side confidences
- Verbatim, on the per-request-K design (from RFC #48202's thread): **"@jessiewei7 I don't think planning verify_lens on the host is a good strategy, as we would not be able to use the speculator-generated confidence scores without synchronizing the gpu and cpu."** https://github.com/vllm-project/vllm/issues/48202
- Same synchronisation objection is the stated reason PR #43522 was refused (see C2). This is a recurring, unresolved architectural constraint on adaptive K.

### B21. Only a static `num_speculative_tokens` is exposed to users on some stacks; higher K may just fail
- discuss.vllm.ai thread 2447, `Qwen3.5-27B-FP8 Speculative Decoding` (2026-03-12). https://discuss.vllm.ai/t/qwen3-5-27b-fp8-speculative-decoding/2447
- Verbatim (asking): **"The only way that works for me is {"method": "mtp", "num_speculative_tokens":1} . Increasing num_speculative_tokens to 2 results in error. Is that expected?"** and **"With MTP my throughput benchmark result went from 120 requests per minute to 140 . Which is about 16%. Not bad for adding one config parameter but also not as good as 1.5x to 2x performance gains I saw online."**
- Maintainer reply (benchislett): **"RunLLM is wrong here. You should be able to use it with MTP for multiple tokens, and that will be the most performant way to do it. Please create a github issue and include the error log so we can triage"**. Community explanation: **"Qwen3.5 uses hybrid linear attention throughout. Its conv_states and recurrent_states do not have a sequence_length dimension, so they cannot be selectively accepted the way a traditional KV cache can."**

### B22. Structured output / beginning-of-sequence has no delayed or per-position speculation start
- discuss.vllm.ai thread 1096, `Improving Speculative Decoding for Beginning Tokens & Structured Output` (2025-07-16). https://discuss.vllm.ai/t/improving-speculative-decoding-for-beginning-tokens-structured-output/1096
- Verbatim (asking): **"My idea is to allow users to customarily specify that the first 1/n tokens are generated by the main (or "original") model , and only then initiate speculative decoding ."**
- Verbatim (bot reply, confirming the gap): **"Speculative decoding in vLLM begins immediately after the prefill phase, and all sequences in a batch must have the same proposal length or zero; per-sequence or delayed speculative start is not supported yet"**.

### B23. Library/in-process hosts cannot reach the speculative API at all
- Who: thermodynamic-retard, llama.cpp issue #27089 (Aug 14 2026). **Open**, no labels. https://github.com/ggml-org/llama.cpp/issues/27089
- Verbatim: **"The core llama.h API exposes no speculative params… DSpark (and MTP) support is implemented in common/speculative.cpp / the server layer only."** and **"we ... are currently forced to use our own draft→verify loop, which cannot use DSpark's confidence head."**
- Related: llama.cpp PR #26575 `spec: respect safe draft caps before block decode` (devesssi) — **Open**, 1 commit. https://github.com/ggml-org/llama.cpp/pull/26575 (TITLE ONLY)

---

## C. ATTEMPTED-AND-ABANDONED (verbatim quotes required — all READ BODY)

### C1. Adaptive verifier step-length for DFlash/PARD (D-Cut-style) — REJECTED BY MAINTAINER
- PR #44885, EanWang211123. https://github.com/vllm-project/vllm/pull/44885 — **Closed unmerged**.
- benchislett (Member), Jun 8 2026, verbatim: **"As discussed here, the surface area of this feature is too significant to justify the marginal and hardware-specific gains #35301"**
- This is the single clearest maintainer statement that adaptive speculation length was judged not worth its complexity.

### C2. Dynamic verification for DFlash — REJECTED ON SYNCHRONISATION GROUNDS
- PR #43522, lrioxh. https://github.com/vllm-project/vllm/pull/43522 — **Closed unmerged**.
- benchislett (Member), Jun 22 2026, verbatim: **"I feel strongly that we must fundamentally avoid synchronization whenever possible. I do not feel comfortable supporting this feature at this time."**

### C3. Elastic Speculation: Adaptive Draft Length + Confidence-Based Early Exit — STALE-CLOSED
- PR #28693, yuz207 (IluvatarLabs). https://github.com/vllm-project/vllm/pull/28693 — **Closed unmerged**, 30 commits.
- github-actions bot, May 20 2026, verbatim: **"This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days."** — then closed by benchislett the same day.

### C4. Per-sequence dynamic speculative decoding (`DynamicProposer`, `--acceptance-rate-threshold`) — STALE-CLOSED
- PR #26504, yang926. https://github.com/vllm-project/vllm/pull/26504 — **Closed unmerged**, 19 commits.
- hmellor (Member), verbatim, first: **"Blocking as there are a lot of basic Python issues in the simple parts of this PR"**, then later: **"Closing as stale"**
- Note the method name shipped in the PR was `eagle_dynamic`.

### C5. Dynamic Speculation Length (DSL) with Confidence-Threshold Early Exit — RFC CLOSED AS NOT PLANNED
- Issue/RFC #36657. https://github.com/vllm-project/vllm/issues/36657 — **Closed as not planned**, labels `RFC`, `stale`.
- The implementation PR #35301 by jmamou remains **Open** but its own description defers per-request policy: **"a per-request policy would be more efficient but requires variable-length draft outputs and more complex KV-cache bookkeeping."** https://github.com/vllm-project/vllm/pull/35301

### C6. `[Benchmarks] Add speculative K schedule tuner` — AUTHOR SELF-CLOSED NEXT DAY
- PR #49163, RichApple123. https://github.com/vllm-project/vllm/pull/49163 — **Closed** 1 day after opening, branch deleted.
- Author, Jul 21 2026, verbatim: **"Closing this draft while re-scoping the contribution around vLLM's upstream Dynamic SD architecture and existing adaptive-K work. The offline tuner is not the original online-adaptation objective, and I do not want to advance it independently without a clearer non-overlapping role. The branch and benchmark artifacts remain available for reference."**

### C7. `[Spec Decode] Make DFlash draft compute honor dynamic K` — AUTHOR SELF-CLOSED, ALGORITHMIC OBJECTION
- PR #49164, RichApple123. https://github.com/vllm-project/vllm/pull/49164 — **Closed** 1 day after opening.
- Author, Jul 21 2026, verbatim: **"Closing this draft after re-evaluating the algorithmic assumption. For non-causal DFlash, shortening the physical draft block is not equivalent to computing the full trained block and truncating verification: later mask positions can influence earlier draft logits and acceptance. The maintainer concern is valid, and this should not be presented as removal of unused compute. I am preserving the branch and benchmark artifacts for reference, but will not continue this design as a general DFlash optimization. Thank you for catching this."**

### C8. MRV2 Adaptive Speculative Decoding — Initial Support — SUPERSEDED BY ITS OWN SUCCESSOR
- PR #48692, benchislett. https://github.com/vllm-project/vllm/pull/48692 — **Closed unmerged** (17 commits).
- benchislett (Member, author), Aug 5 2026, verbatim: **"Closing this PR, we will move forward and try to merge #47808 ."**
- Its own negative measurement, verbatim: **"Since verification is so cheap on an 8B model, no significant speedup is measurable; that is not the objective of this PR."**
- It also declares its scope limit: **"This PR does not implement online dynamic scheduling of the verification budget; instead, we rely on user-provided num_speculative_tokens_per_batch_size to determine what size pool of draft tokens to verify at each batch size."**
- A reviewer (cyLi-Tiger) raised a CUDA-graph efficiency objection that was never resolved in this PR, verbatim: **"SGLang's DSpark integration handles per-request variable verify lengths by packing the ragged batch into a compact varlen buffer and keying the graph purely on the total token count… the approach here, where decode graphs are rounded up to multiples of the scheduled query length"** — example given: a 17-real-token verify batch needing the 32-token FULL graph here vs an 18-token tier in SGLang.

### C9. CPU DSD fix — CLOSED BY ITS OWN AUTHOR AS UNNECESSARY
- PR #47012, jmamou. https://github.com/vllm-project/vllm/pull/47012 — **Closed**.
- Author, Jul 1 2026, verbatim: **"Closing this PR — the fixes here are no longer needed after #47162 ."**
- The underlying bug report stays **Open**: issue #47014 `[CPU][Bug] Dynamic Speculative Decoding crashes on CPU backend` (Jun 29 2026), verbatim root cause: "The CPU model runner overrides ( _copy_draft_token_ids_to_cpu and _get_draft_token_ids_cpu in vllm/v1/worker/cpu_model_runner.py ) assume a static number of speculative tokens." https://github.com/vllm-project/vllm/issues/47014

### C10. TTFT regression with EAGLE3 — CLOSED AS NOT PLANNED
- Issue #39790. https://github.com/vllm-project/vllm/issues/39790 — **Closed as not planned**, labels `performance`, `stale`.
- User's measurements, verbatim: "Mean TPOT: ~17.11 ms (a 39.3% improvement, which is great)" but "Mean TTFT: ~158.44 ms (a 114% increase)" and "P99 TTFT: ~1078.03 ms (a 331% increase)."
- The eventual partial fix PR #41481 notes 3 of 4 cost sources remain untouched, verbatim: **"The original report enumerates 4 first-call cost sources; this PR addresses 1 of the 4 (the utils.py kernels). Remaining sources called out by @ccs668899 — FlashAttention drafter-shape autotune, drafter CUDA module loading (~188 ms), lazy drafter KV/hidden-state allocation — are not touched here"**. https://github.com/vllm-project/vllm/pull/41481 (Open)

### C11. NEGATIVE: DSD arms pay a large baseline tax vs no-spec — −31% at short context
- Issue #49986, seongyun1104. https://github.com/vllm-project/vllm/issues/49986 — **Open**, no labels.
- Verbatim: **"I found that every arm using a speculative_config pays a substantial throughput tax vs no-spec under production defaults , and the gap is large enough at short ctx that it dominates the aggregate spec-bench signal"**
- Measured (H100 NVL 94GB, prefix_repetition c=256): ctx=400 → no-spec 2711.7 vs static K=3 2139.8 (−21%) vs 3-item batch schedule 1875.6 (**−31%**) vs 2D 6-cell 1890.7 (−30%). At ctx=900: 1987.1 vs 1453.5 (−27%).
- Verbatim on the residual: **"In an eager-mode control ( enforce_eager=True , both arms lose graphs equally) at ctx=1900, static K=3 was still 6% slower than no-spec (0.94×), not faster as the mechanism would predict. So there is a residual DSD-mode overhead beyond the cudagraph-mode difference — probably drafter forward on K=0 steps, spec bookkeeping, admission cost — that needs to be decomposed."**
- Also documents the forced downgrade: `"Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE for reliability."`

### C12. NEGATIVE: catastrophic aggregate-throughput collapse at the batch-size threshold
- Issue #49548, tobby168 (Jul 23 2026). https://github.com/vllm-project/vllm/issues/49548 — **Open**, no labels.
- Verbatim: **"With the schedule [[1,4,2],[5,512,0]] … 8 concurrent 180-token requests dropped from ~232 tok/s aggregate (static num_speculative_tokens: 2 ) to 24–157 tok/s , with wall-clock 40–60 s for a workload that completes in ~6.3 s under the static config"**
- Verbatim: **"The naive expectation is that at batch ≥ 5 (K=0) throughput should fall back to roughly the non-spec decode rate. Instead it is ~1.5–10× worse than a plain non-spec run would be."**
- Hypotheses are source-referenced, incl. H1 verbatim: **"At K=0 the MTP drafter still runs a full draft-model forward every step… So 'spec disabled at batch ≥ 5' is not equivalent to a non-spec run: every decode step at high batch still pays a full MTP forward plus all input prep"**, and H3: **"len(num_scheduled_tokens) counts all scheduled requests including prefills/chunks, so a single prefill joining 4 decodes bumps the count to 5 and silently disables spec for the whole step."**

### C13. NEGATIVE: DSD + MTP 12–25% throughput penalty and a full-CUDA-graph capture crash
- Issue #48494, seongyun1104 (Jul 13 2026). https://github.com/vllm-project/vllm/issues/48494 — **Open**, no labels.
- Verbatim: **"at client concurrency 256 we see DSD mode incur a 12–25% throughput penalty vs no-spec (oscillating run-to-run) with TTFT p50 inflating ~6×, even with an all-K=0 table producing near-zero draft tokens."**
- Verbatim on the trigger: **"→ The presence of the batch-size table is the trigger; K=0 tiers are not required."** (static `num_speculative_tokens: 3` capture completes; any DSD table crashes at `InputBatch.make_dummy` assert.)
- Also documents that the documented tested set is narrow: the DSD docs "state the feature is 'only tested with Eagle and Eagle-3'; this report adds the MTP data point."

### C14. NEGATIVE: native MTP can be slower than a no-MTP CUDA-graph baseline despite good acceptance
- Issue #47277. https://github.com/vllm-project/vllm/issues/47277 — referenced by both #48494 and #49548 as the canonical statement of the "per-step overhead erases the benefit" pattern. TITLE ONLY (fetch failed repeatedly; title + verbatim quotations of it from #48494 and #49548, both READ BODY).

### C15. NEGATIVE (author's own): confidence early stopping did not improve wall clock
- vLLM RFC #48202, `Non-goals`, verbatim: **"Honest measurement from the prototype: on Qwen3-4B on a single L20X up to batch 64, confidence early stopping did not improve wall clock over fixed full blocks at any threshold we tried; the DSpark backbone drafts the whole block in one pass regardless of truncation, and in this regime per-step cost is nearly flat in verified tokens while lost acceptance is paid in extra decoding steps. These results show that per-request truncation is not universally profitable"**
- https://github.com/vllm-project/vllm/issues/48202 (Closed, no maintainer comment)
- Same RFC also names the graveyard: "This is also the interface boundary left unresolved by several earlier adaptive-speculation attempts ( #28693 , #26504 , #44885 , #43522 )."

### C16. NEGATIVE (engine's own): DSD's K=0 still pays a drafter prefill
- vLLM PR #32374, author's own caveat, verbatim: **"DSD has some overhead of running the draft model to prefill even though its not used during decode even though DSD would assign K=0. This is fine because the setup can change BS in future so having all tokens prefilled in draft model is needed."**
- Also, verbatim scope admission: **"There are many ways to extend this strategy like resetting AR after some steps but those are left for future work. The purpose of the PR is to have at least something working in vLLM."**
- https://github.com/vllm-project/vllm/pull/32374

### C17. NEGATIVE (paper): speculation is impractical in MoEs; static-K profiling is ineffective
- Cascade, arXiv 2506.20675, verbatim: **"we show that speculation is ineffective for MoEs: draft tokens collectively activate more weights, increasing data movement and verification time by 2-3x. When token throughput gains fail to offset this overhead, speculation causes slowdowns up to 1.5x, making it infeasible."** and **"despite widespread use in dense LLMs, speculation remains impractical in leading MoEs."**
- Verbatim against static-K tuning: **"we find that every workload experiences slowdown for at least one K, and three (math, extract, math+extract) suffer slowdown across all K-values"**; **"profiling an optimal static-K per task is ineffective under realistic, mixed-request serving scenarios"**.
- Verbatim against prior dynamic-K: **"such schemes maximize ETR and cannot anticipate when no speculation (K=0) is optimal"**; **"current dynamic-K policies are infeasible for MoEs."**
- https://arxiv.org/abs/2506.20675

### C18. NEGATIVE (paper): gains vanish at high load / high QPS
- TurboSpec / SmartSpec, arXiv 2406.14066, verbatim: **"at a request rate greater than 16, proposing 3 tokens yields performance degradation"** and **"when the request rate exceeds 12, proposing 5 tokens offers no performance improvements"**.
- Verbatim: **"For large batch sizes, not speculate altogether can result in higher goodput. This occurs as the cost of unsuccessful speculations increases significantly with larger batch sizes, outpacing the potential gains."**
- Verbatim (sub-agent, separate fetch): **"on L40S at QPS=16, even minimal speculation (k=1) degrades performance by 14% (from 13.7s to 15.9s)"**; **"the notably poor performance of k=5 under high QPS conditions … highlights the limitations associated with overly aggressive speculative decoding under heavy workloads"**.
- Tree-width negative, verbatim: **"fixed top2/top3 Medusa quickly explode the batch size."**
- https://arxiv.org/abs/2406.14066

### C19. NEGATIVE (paper): exploration-based adaptive speculation fails under memory constraints
- MemSpec, arXiv 2608.10362, verbatim: **"We identify a fundamental limitation of adaptive speculative decoding on memory-constrained edge devices, namely the mismatch between draft selection and draft availability. We show that exploration-based methods incur excessive switching overhead, as selecting a better draft often requires loading non-resident models, limiting end-to-end throughput gains."** and **"Exploration-based adaptation fails to improve throughput under memory constraints."**
- Conclusion verbatim: **"the primary bottleneck in adaptive speculative decoding on edge devices is not draft selection quality, but ensuring timely availability of effective drafts."**
- https://arxiv.org/abs/2608.10362

### C20. NEGATIVE (paper): better token predictors need not yield better schedulers
- arXiv 2608.14787, verbatim: **"better token predictors need not yield better schedulers"**; **"marginal survival has higher positionwise AUROC than raw confidence at most positions, yet neither learned signal dominates online"**; **"our results show no observed online dominance by the learned signals, not a universal ordering."**
- https://arxiv.org/abs/2608.14787

### C21. NEGATIVE (paper): bandits do not always beat the best fixed baseline
- TapOut, arXiv 2511.02017, verbatim: **"the overall performance of bandits does not always exceed the best baseline method, and the datasets we use to evaluate our approach are relatively small… the performance of the bandit is inherently upper-bounded by the individual performances of the dynamic speculation algorithms used as arms."**
- https://arxiv.org/abs/2511.02017

### C22. NEGATIVE (paper): longer accepted prefixes cannot compensate for an expensive drafter
- arXiv 2607.12422, `Accepted Prefixes Are Not All You Need: A Negative Result on PEFT-Based Block-Diffusion Drafting`, verbatim: **"Thus, the drafter is parameter-efficient but not compute-efficient. Our results isolate a simple but important condition for successful speculative decoding: the drafter must be substantially cheaper to execute than the verifier. Longer accepted prefixes alone cannot compensate when draft computation remains verifier-scale."**
- https://arxiv.org/abs/2607.12422

### C23. NEGATIVE (canonical, 2023): origin of "larger batch ⇒ smaller speculation length"
- arXiv 2310.18813, `The Synergy of Speculative Decoding and Batching`, verbatim: **"a speculation length too large will deteriorate the performance"**; adaptive gives only an extra 9% latency reduction vs fixed under time-varying requests; hedged conclusion verbatim: **"Our evaluations show that our proposed method achieves equal or better performance than fixed speculation decoding schemes."**
- https://arxiv.org/abs/2310.18813

### C24. NEGATIVE (paper): adaptive γ over RL rollouts — only 1 of the pretrained drafters helped
- EfficientRollout, arXiv 2606.18967, verbatim: **"The other configurations are slower than No-SD for much of training because their block efficiency is too low to amortize draft, verification, and online-update overhead."**
- Verbatim: **"Under high-temperature sampling (T=1.0) and long reasoning generation, the pretrained drafters do not provide sufficient proposal quality for our math RL rollout distribution."** Pretrained auxiliary drafters reach only block efficiency τ=1.2–2.4 vs 3.6–3.9 for a quantized self-drafter.
- Verbatim: **"These results suggest that directly reusing public checkpoints or drafters pretrained with fixed-target LLM-serving recipes is insufficient to reliably obtain high τ in RL rollouts."**
- **Positive counterweight from the same paper**: adaptive γ gives "an overall 19.6% reduction in rollout-generation latency, compared with 13.5% and 11.8% for fixed γ=γ_low=5 and γ=γ_high=11".
- Gains vanish in the early **large-batch** regime, verbatim: **"enabling SD during early large-batch phases can reduce or eliminate its benefit"** — disabling SD for just the early **6–11%** of decoding steps beats always-on SD.
- https://arxiv.org/abs/2606.18967

### C25. NEGATIVE (paper): distributed (edge-cloud) speculative decoding is dominated by co-location when co-location is possible
- arXiv 2606.25091, verbatim: **"If the server can host both models, co-located SD has lower latency and communication than synchronous DSD, with the same per-output FLOPs and model-weight memory."** and **"at WAN RTTs, the cloud round trip remains too large for pipelined DSD to beat co-located SD."**
- https://arxiv.org/abs/2606.25091

### C26. NULL (methodological): no detectable safety divergence at temperature zero
- arXiv 2606.25097, verbatim: **"the tested temperature-zero vLLM stacks show no detectable safety divergence under TAIS. The largest absolute Cohen's h on matched target-only versus speculative refusal is 0.024, roughly an order of magnitude below the conventional trivial-effect floor"**. Included only as a null result. https://arxiv.org/abs/2606.25097

### C27. LIMITATION (paper, self-stated): Yggdrasil's optimal tree shape is single-request only
- Verbatim: **"The core results of Yggdrasil are derived under a latency-optimal setting where a single interactive request monopolizes all available GPU memory and compute… it is not applicable for the batched, throughput-oriented serving."** https://arxiv.org/abs/2512.23858

### C28. LIMITATION (paper, self-stated): Bastion evaluated at batch size 1 only
- Verbatim: **"Batch size constraints: Our evaluation targets batch size 1; serving regimes with larger batches alter the verification-cost profile and warrant dedicated treatment."** and **"Runtime profile calibration: Our default offline calibration assumes a stable runtime profile… deployments with pronounced runtime drift remain an open direction."** https://arxiv.org/abs/2605.29727

### C29. LIMITATION (paper, self-stated): Nightjar's switching costs come from an offline table
- Verbatim: **"A current limitation is that the switching-cost estimates are obtained from offline profiling and queried through a deployment-specific lookup table."** https://arxiv.org/abs/2512.22420

### C30. NEGATIVE (paper, self-stated): AdaEDL has no limitation or future-work section
- AdaEDL (arXiv 2410.18351) is a workshop paper whose abs page carries **no limitation or future-work statement**. Do not attribute a limitation quote to it. Its positive claim, verbatim: "We show that AdaEDL consistently outperforms static draft-length speculative decoding by 10%-57% as well as other training-free draft-stopping techniques by upto 10% in a variety of settings and datasets." and "we show that AdaEDL is more robust than these techniques and preserves performance in high-sampling-temperature scenarios." https://arxiv.org/abs/2410.18351

### C31. NEGATIVE (paper, verbatim): TALON on why fixed depth fails when acceptance drops
- Verbatim: **"Static methods enforce a fixed depth D, locking draft efficiency to a constant δ = D+1. Consequently, when the acceptance length τ drops in difficult contexts, the speedup R degrades significantly."** https://arxiv.org/abs/2601.07353

### C32. Dynamic pruning of EAGLE-3 draft trees — CLOSED AS NOT PLANNED
- Issue #41823 `[Feature]: Support Dynamic Pruning for Speculative Decoding Draft Trees in EAGLE-3`. https://github.com/vllm-project/vllm/issues/41823 — **Closed as not planned**, labels `feature request`, `stale`.
- Auto-close comment verbatim: **"This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"**
- The paired forum question (B19) never received a maintainer answer. https://discuss.vllm.ai/t/what-is-the-recommended-way-to-support-dynamic-pruning-for-speculative-decoding-draft-trees/2630

### C33. Per-request effective proposal lengths reference implementation — CLOSED UNMERGED
- PR #48204 `[Core][Spec Decode] Reference implementation: per-request effective proposal lengths (RFC #48202)`, LJX-xixi. https://github.com/vllm-project/vllm/pull/48204 — **Closed unmerged**; labels `dflash`, `mrv2`, `needs-rebase`, `qwen`, `speculative-decoding`, `v1`; Sprint - DFlash "Status: Backlog"; 1 participant; approving review never obtained. **Closing comment could not be retrieved** — not guessed.
- Its parent RFC #48202 is also closed with no maintainer comment (see B5/C15).

### C34. Packed Variable Length Speculative Decoding RFC — CLOSED, superseded
- RFC #47839, labels `RFC`, `dflash`; Sprint - DFlash "Status Done". https://github.com/vllm-project/vllm/issues/47839 — **Closed**. **Closing comment could not be retrieved** — not guessed.
- The design was effectively superseded by the merged #47808, whose thread states verbatim: **"Working on a prototype in MRV2 here: https://github.com/vllm-project/vllm/pull/47808 this prototype does not involve the scheduler; instead the scheduler always schedules `num_speculative_tokens + 1` and the model runner in-essence rejects tokens without even bothering to verify them."**

### C35. NEGATIVE: adaptive speculation is judged not worth it by its own docs, and it crashed in production
- **SGLang's own shipped docs** say, verbatim: **"If your workload is already stable and one static setting is well tuned, adaptive mode may not help much"**, and **"At high batch sizes, narrower ladders (e.g., [1, 2] or [1] ) often outperform wide ones"**. https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding
- **SGLang issue #30549** `[Bug] --speculative-adaptive crashes at startup: shared logits buffer sized for active draft_tokens, not max adaptive candidate` (Jul 8 2026) — **Closed** (label `inactive`, auto-closed). Verbatim: **"This issue has been automatically closed due to inactivity. Please feel free to reopen it if needed."**
  Body verbatim: **"Enabling --speculative-adaptive (default candidate config) crashes the server during startup on all TP ranks"** with `AssertionError: shared logits buffer holds 128 rows but caller needs 256`. Author root cause: adaptive **"pre-builds runtime states for every candidate step ( AdaptiveController.init_states ), and capturing the steps=7 state requests bs × 8 rows → assert."** and **"the combination appears untested."**
  https://github.com/sgl-project/sglang/issues/30549

### C36. NEGATIVE: the acceptance-target signal was argued to be the wrong control variable
- On PR #26504's thread, KookiesNKareem posted bench evidence that acceptance rate is nearly constant across load while the correct action flips sign, verbatim: acceptance was **"nearly flat across load (0.86-0.91 from batch 1 to 32)"** while the right action flipped from **"+23% (speculate at b=1) to -16% (disable at b=32)"**, and **"Neither setpoint works at both ends"**.
- Also verbatim: **"'off' isn't free... a spec-configured engine loses 15-18% vs a clean engine at b=32"** — i.e. the K=0 tier is not free, corroborating C11/C12/C16. https://github.com/vllm-project/vllm/pull/26504

### C37. LIMITATION (author self-stated, unresolved): host–device sync and un-tuned thresholds
- vLLM RFC #36657 (closed as not planned) declared its own two gaps, verbatim: **"Known limitation — GPU→CPU sync: the early-exit check calls .item() on a GPU scalar, forcing a host–device sync per draft step."** and **"Threshold is not auto-tuned: recommended starting range is 0.4–0.6 ; optimal value is model-pair and dataset dependent."** https://github.com/vllm-project/vllm/issues/36657
- This is the same synchronisation objection that killed PR #43522 (C2), and it recurs in B20.

### C38. NEGATIVE: the canonical per-sequence-K issue's own CUDA-graph blocker
- vLLM issue #17984 (`[Feature]: Per-sequence speculative decoding`, closed as not planned), verbatim: **"Ultimately, the current vLLM batch size processing method can only handle static SL that the K ( num_speculative_tokens ) set before inference."** and **"Both Batch Expansion and MQA Scorer face performance degradation on dynamic shape & CUDA Graphs. if K ( num_speculative_tokens ) changes each iteration or across sequences. CUDA graph can't hadling the dynamic shape/size."**
- The issue also contains the (now-unfulfilled) vLLM plan, verbatim: **"we will introduce per-sequence dynamic K in vLLM so that each request can adjust its speculative-token budget according to its own acceptance rate."**
- https://github.com/vllm-project/vllm/issues/17984

---

## D. HARDWARE-RULED-OUT for 1× RTX PRO 6000 Blackwell 96 GB (sm120), single node

| # | Item | Why it is ruled out | URL | READ BODY or TITLE ONLY |
|---|---|---|---|---|
| D1 | **vLLM Adaptive Verification (DSpark) — the flagship shipped adaptive-budget feature** | Requires `AttentionCGSupport.ALWAYS`; verbatim: **"FULL varlen decode graphs require AttentionCGSupport.ALWAYS , which the DSV4 sparse-MLA, sparse-SWA, and indexer backends report on SM100. Elsewhere adaptive verification is rejected at startup rather than falling back to PIECEWISE."** sm120 ≠ sm100. Reference measurement itself is **TP=8 on 8×B300 (SM100)**. | https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification | READ BODY |
| D2 | vLLM Adaptive Verification — additional exclusions | Verbatim: "Not supported with LoRA (the per-token LoRA mapping is built from CPU-side boundaries) or pipeline parallelism (cost curves and confidences exist only on the last rank)." Also `--enforce-eager` rejected at startup. | https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/ | READ BODY |
| D3 | vLLM Dynamic SD under data parallelism | Verbatim: "Not compatible with data parallelism ( --data-parallel-size > 1 ). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks." | https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ | READ BODY |
| D4 | vLLM Dynamic SD CUDA-graph modes | Verbatim: "Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature". Full-CG path is what crashes for MTP (#48494). | https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ | READ BODY |
| D5 | **NSA (Native Sparse Attention) on SM120 — declared an architectural dead end** | Verbatim: **"After extensive production testing on SM120 (RTX PRO 6000 Blackwell, 8×96GB, PCIe Gen5), NSA (Native Sparse Attention) cannot be made to work reliably for production serving on NVIDIA consumer/workstation GPUs. This is not a bug — it is an architectural incompatibility."** and **"On SM100 (datacenter Blackwell), dedicated FP8 block-scaled GEMM units handle this efficiently. On SM120, these units do not exist"**; **"after 7+ days of testing every available NSA+MTP Docker image on 8×RTX PRO 6000, none are production-viable. NSA on SM120 is a dead end."** Also: "MTP + TP>1 + concurrent>1 produces deadlocks". | https://github.com/vllm-project/vllm/issues/43235 | READ BODY |
| D6 | SGLang adaptive spec on 8×GPU (reference measurement) | PR #24055 speed tests: **"GLM-4.7-FP8 on 8×GPU (TP=8), EAGLE speculative decoding with topk=1"**. Not reproducible on one GPU at that model scale. | https://github.com/sgl-project/sglang/pull/24055 | READ BODY |
| D7 | SGLang adaptive spec + DP attention | Verbatim from source: `"enable_dp_attention=True is not supported (adaptive tier decisions are not synchronized across DP ranks)"`. Also unsupported: multi-layer EAGLE, two-batch overlap, PD-mux. | https://github.com/sgl-project/sglang/blob/c69844f0/python/sglang/srt/speculative/adaptive_spec_params.py | READ BODY |
| D8 | EcoSpec (MoE cost-aware SD) evaluation models | Evaluated on **DeepSeek-V3.1 (671B)**, Qwen3-235B-A22B, GPT-OSS-120B — none fit in 96 GB. | https://arxiv.org/abs/2607.12696 | READ BODY |
| D9 | Cascade / MoE utility-driven K evaluation models | Five popular MoEs incl. DeepSeek-V3 class; paper reports H100 results. | https://arxiv.org/abs/2506.20675 | READ BODY |
| D10 | Illusion paper's 70B data points | Llama3-70B "is executed on four GPUs" — verbatim: "the 70B model is executed on four GPUs, so even with small or medium batch sizes, the system is already compute-bound". Study used an **NVIDIA-donated DGX server**. | https://arxiv.org/abs/2601.11580 | READ BODY |
| D11 | UniVer / tree papers at scale | UniVer reports across tasks/models; SADDLE (HPCA'26) is a **PIM** result — requires processing-in-memory hardware. | https://arxiv.org/abs/2605.04543 · https://2026.hpca-conf.org/details/hpca-2026-main-conference/112/Adaptive-Draft-Sequence-Length-Enhancing-Speculative-Decoding-Throughput-on-PIM-Enab | READ BODY |
| D12 | Multi-node / cluster-scoped work | No multi-node adaptive-draft-length artifact was found. Nearest cluster-scoped item is WISP (distributed edge-cloud), arXiv 2601.11652, and "Speculation at a Distance" arXiv 2606.25091, which concludes verbatim: **"If the server can host both models, co-located SD has lower latency and communication than synchronous DSD"** — i.e. distribution is a net negative when co-location is possible. | https://arxiv.org/abs/2606.25091 | READ BODY |

**Fit-check data for exactly this hardware class (1× RTX PRO 6000 Blackwell 96 GB, sm120), READ BODY** — `lastloop-ai/vllm-blackwell-guide`, https://github.com/lastloop-ai/vllm-blackwell-guide:
- `CUDA graphs + flashinfer + MTP n=3 + FP8 KV` → **100 tok/s** (27B dense INT4); **170 tok/s** (35B-A3B MoE FP8).
- `CUDA graphs + flash_attn + MTP n=5` → **125 tok/s**, annotated verbatim **"+4% over n=3, marginal"**.
- Reported MTP metrics: "mean acceptance length 3.19, per-position acceptance 0.87/0.72/0.60, avg draft acceptance 73%".
- This is the only single-GPU sm120 speculation-depth datapoint found, and it shows **deepening K from 3 to 5 buys ~4%**.
- Long-context MTP failure on the same GPU class: vLLM issue #52756 (RTX PRO 6000 Blackwell sm_120, vLLM 0.27.1) shows drafts still scheduled past the drafter limit with **"Per-position acceptance rate: 0.000, 0.000"** and **"Avg Draft acceptance rate: 0.0%"**. https://github.com/vllm-project/vllm/issues/52756

### Additional hardware-fit notes (findings only)

| # | Item | Note | URL | R/T |
|---|---|---|---|---|
| D13 | SGLang adaptive spec startup crash "on all TP ranks" | `--speculative-adaptive` with default candidate config raised `AssertionError: shared logits buffer holds 128 rows but caller needs 256` at startup; author states **"the combination appears untested."** Auto-closed as inactive. | https://github.com/sgl-project/sglang/issues/30549 | READ BODY |
| D14 | Qwen3.5 hybrid linear attention caps MTP at 1 token | Community explanation of an architecture (not VRAM) limit: **"Qwen3.5 uses hybrid linear attention throughout. Its conv_states and recurrent_states do not have a sequence_length dimension, so they cannot be selectively accepted the way a traditional KV cache can... I think this may be the reason why only MTP-1 is supported."** | https://discuss.vllm.ai/t/qwen3-5-27b-fp8-speculative-decoding/2447 | READ BODY |
| D15 | Most published adaptive-spec evaluations at scale are ≥4 GPUs | Illusion uses 4 GPUs for 70B (D10); vLLM DSpark uses TP=8 on 8×B300 (D1); SGLang adaptive speed tests use 8×GPU TP=8 (D6); Cascade/EcoSpec evaluate 235B–671B MoEs (D8, D9); TRT-LLM adaptive work is unverified for single-GPU sm120 here (TITLE ONLY). | see D1, D6, D8, D9, D10 | mixed |


---

## Cross-cutting findings (findings only, no recommendations)

- **The "gains vanish at high batch" claim is now measured on a production engine, not just prototypes.** Illusion (2601.11580): EAGLE speedup on Llama3.1-8B/GSM8K falls **1.73× → 1.21×** from batch 1 → 128; for Llama3-70B/ShareGPT, batch 1 → 32 costs **14.0%** (1.96× → 1.72×).
- **Wide trees actively lose at batch.** Verbatim: "By batch size 64, the k = 21 tree falls below 1 × speedup on all workloads for both models, whereas the chain remains above 1 × throughout." Accepted length rose (2.25 → 2.92) while acceptance rate collapsed (0.415 → 0.095).
- **Verification dominates**: "the verification stage generally takes the largest execution time, ranging from 42% to 95% in all execution methods, which increases with the model size and batch size."
- **The engine's own adaptive feature has an unresolved tax.** Two independent operators (#49986, #49548) measure large regressions from DSD itself, including at K=0.
- **Two maintainer rejections of adaptive speculation length exist and are quotable** (C1, C2), both from benchislett, on different grounds (feature surface area vs synchronisation).
- **The one shipped vLLM adaptive-budget feature (A5/A6) sidesteps K selection entirely.** Verbatim from RFC #54749: "adaptive verification does not choose K. Drafting still happens at the scheduled length; it chooses how much of the already-drafted block to verify". The K decision remains a dense batch-size array index.
- **SGLang is the only engine found with a shipped EMA controller over `speculative_num_steps`** (A8/A9), and it is a simple `round(ema_accept_len) + 1` rule with hysteresis and a fixed candidate set `{1,3,7}` / `{1,2,5}` / `{1,2,6}` — not a bandit or RL method.
- **No bandit/RL controller has shipped in any engine.** Bandit work (A24–A27) is paper-only; the closest to an RL framing is the "learned verification payoff" line (Hybrid Verified Decoding, 2606.01019).
- **Thinking-vs-answer phase budgeting appears genuinely unclaimed.** The only phase-transition measurement found is n-gram acceptance decaying at the end of reasoning traces (2601.11580).
- **Red herrings in the lead list**: PRISM and HELIOS are not adaptive-speculation papers (see note under Section A). TurboSpec's canonical status is confirmed ("Closed-loop Speculation Control System for Optimizing LLM Serving Goodput"), but its mechanism is a goodput feedback controller, not a bandit/RL one.
- **Three artifacts the lead list asked for were located and read**: "Speculative Decoding: Performance or Illusion?" = **arXiv 2601.11580** (v2 18 Mar 2026); vLLM issue **#47014** = `[CPU][Bug] Dynamic Speculative Decoding crashes on CPU backend`; vLLM PR **#54748** = retitled `[Spec Decode] Make the dynamic SD schedule observable` (originally `[Spec Decode] Count scheduler steps by the K dynamic SD selected`).
- **Four engines have shipped *some* form of dynamic draft length**: vLLM (batch-size-keyed table, `num_speculative_tokens_per_batch_size`; plus per-step confidence-scheduled verification), SGLang (EMA controller over `speculative_num_steps` + per-BS state), TensorRT-LLM (dynamic draft length, one-model path — TITLE ONLY), and SGLang's per-request-K DSpark verify path. HuggingFace TGI and llama.cpp have **not**.
- **The strongest published counter-evidence to adaptive speculation is operational, not theoretical**: vLLM's own SGLang docs and its DSD issue tracker say adaptive mode has limited value when a static setting is well tuned (C35), and two operators independently measure that the K=0 tier is not free (C11, C12, C36).
- **The recurring architectural blocker is host–device synchronisation**: it is the stated reason PR #43522 was refused (C2), the self-declared known limitation of RFC #36657 (C37), and the objection to host-side verify-length planning (B20). The proposed ctx-axis route in SGLang PR #31716 works around it by computing `ctx_repr` CPU-side from `batch.reqs` rather than reading a GPU tensor.

---

## Search log (queries verbatim, by engine)

### web_search (this agent)
1. `Speculative Decoding Performance or Illusion MLSys 2026 Xiaoxuan Liu` · `adaptive speculative decoding dynamic draft length arXiv 2025` · `vLLM dynamic speculative decoding num_speculative_tokens auto-tuning issue` · `SpecDec++ adaptive speculation length acceptance rate prediction`
2. `vllm "Add Dynamic SD" pull 32374 num_speculative_tokens_per_batch_size merged` · `"DSpark" "adaptive_verification" vllm confidence scheduled verification`
3. `SGLang speculative-adaptive adaptive_spec_params merge PR adaptive speculative decoding num steps` · `PRISM adaptive speculation MLSys 2026 HELIOS speculative decoding` · `vLLM issue discussion adaptive speculative decoding num_speculative_tokens not planned wontfix`
4. `HELIOS speculative decoding MLSys 2026 adaptive speculation length` · `MLSys 2026 speculative decoding speculation length paper` · `"speculation length" adaptive 2026 arXiv bandit reinforcement learning draft budget`
5. `SpecKV adaptive controller speculation length gamma speculative decoding arXiv` · `"adaptive speculative decoding" vLLM discussion forum discuss.vllm.ai acceptance rate high batch` · `vLLM PR removed dynamic speculative decoding reverted auto-tune num_speculative_tokens closed`
6. `discuss.vllm.ai dynamic speculative decoding num_speculative_tokens acceptance rate` · `site:discuss.vllm.ai speculative decoding acceptance rate batch size` · `vLLM "num_speculative_tokens" tuning high batch no benefit forum`
7. `EcoSpec Cascade MoE speculative decoding requires 8 GPU H100 DeepSeek-V3 671B evaluation hardware` · `vLLM adaptive verification SM100 only sm120 RTX PRO 6000 Blackwell attention backend support` · `speculative decoding acceptance rate temperature thinking reasoning phase 2026 paper`
8. `"Acceptance Dynamics Across Cognitive Domains in Speculative Decoding" arXiv` · `UniVer Unified Speculative Decoding arXiv 2605.04543`
9. `arxiv "Acceptance Dynamics Across Cognitive Domains in Speculative Decoding" abstract`
10. `RTX PRO 6000 Blackwell sm120 speculative decoding EAGLE MTP vLLM issue not supported` · `speculative decoding gains vanish high batch negative result 2026 paper` · `vLLM "[Performance]" speculative decoding slower at high concurrency issue 2026`
11. `speculative decoding acceptance rate reasoning thinking phase vs answer phase 2026` · `adaptive speculation reasoning models thinking budget draft length 2026 arXiv`
12. `"HELIOS" "Adaptive Model And Early-Exit Selection" arXiv early exit LLM serving` · `AdaEDL adaptive draft length speculative decoding arXiv`

### web_search (academic sub-sweep, verbatim highlights)
`PRISM adaptive speculative decoding MLSys 2026` · `HELIOS adaptive speculation MLSys 2026` · `MLSys 2026 papers speculative decoding adaptive speculation length` · queries against `bandit speculative decoding`, `adaptive speculation length LLM serving`, `dynamic draft tree budget`, `acceptance prediction head speculative decoding`, `negative result speculative decoding throughput saturates` (0 results), `acceptance rate temperature speculative decoding analysis` (0 results)

### arXiv listing / API search
- `https://arxiv.org/search/?searchtype=all&query=Speculative+Decoding+Performance+or+Illusion&size=25` → resolved arXiv 2601.11580
- `https://arxiv.org/search/?searchtype=all&query=Acceptance+Dynamics+Across+Cognitive+Domains+in+Speculative+Decoding&size=10`
- `https://arxiv.org/search/?searchtype=all&query=Acceptance+Dynamics+Cognitive+Domains+Speculative&size=10`
- `http://export.arxiv.org/api/query?search_query=...` (multiple attempts — **rate-limited**, returned empty; abandoned in favour of the HTML search endpoint)

### GitHub (HTML pages, direct fetch via curl)
- vLLM PRs: 32374, 54748, 48944, 47808, 48692, 49163, 49164, 47012, 26504, 28693, 44885, 43522, 35301, 45953, 46399, 41481, 52436, 53367 (referenced)
- vLLM issues: 47014, 54749, 48494, 49986, 49548, 52258, 52756, 50708, 48202, 44506, 46295, 17984, 36657, 41821, 43235, 47277 (title only), 39790, 54691 (fetch failed)
- SGLang PRs: 24055, 24054, 31716; SGLang source: `python/sglang/srt/speculative/adaptive_spec_params.py` via raw.githubusercontent.com
- raw.githubusercontent.com: `lastloop-ai/vllm-blackwell-guide/main/README.md`

### Engine documentation / vendor sources
`https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/` · `.../adaptive_verification/` · `.../acceptance_metrics/` · `https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification` · `https://mlsys.org/virtual/2026/oral/3789` · `https://mlsys.org/virtual/2026/oral/3846` · `https://mlsys.org/virtual/2026/poster/10189` · `https://mlsys.org/media/mlsys-2026/Slides/3782.pdf` · `https://2026.hpca-conf.org/details/hpca-2026-main-conference/112/...`

### Coverage caveats
- `api.github.com` deliberately avoided (rate limit); `discuss.vllm.ai` was searched both via web_search and via its Discourse JSON API (`https://discuss.vllm.ai/search.json?q=…`) by the parallel engine-forum sweep, plus direct topic reads.
- arXiv API was rate-limited for the whole session; the `arxiv.org/search/` HTML endpoint was used instead and works.
- MLSys 2026 `papers.html` is JS-rendered and returned no paper list via curl; individual oral/poster pages were fetched instead, so the MLSys 2026 programme was **not** swept exhaustively.
- Proceedings-level sweeps of NeurIPS/ICML/MLSys 2026 are **not** exhaustive.
- **TITLE ONLY** items retained in this report: vLLM PRs #45953, #53367; vLLM issue #47277; SGLang PRs #21599, #25644, #23331; TRT-LLM PRs #10860, #8194, #7728; llama.cpp PR #26575; discuss.vllm.ai related-thread under B13; AdaEAGLE arXiv 2412.18910.

### Search log — engine-forum sweep (subagent; verbatim)

**web_search:** (1) `vllm dynamic speculative decoding num_speculative_tokens adaptive K discussion` (2) `sglang speculative_adaptive speculative_num_steps dynamic draft length issue` (3) `vllm github issue adaptive num_speculative_tokens auto-tune acceptance rate` (4) `speculative decoding dynamic speculation length not worth it maintainer` (5) `sglang pull request speculative-adaptive adaptive num_steps merged` (6) `TensorRT-LLM adaptive draft length speculative decoding issue dynamic num_draft_tokens` (7) `llama.cpp speculative decoding dynamic draft length adaptive n_draft issue` (8) `HuggingFace TGI speculative decoding adaptive num_speculative_tokens issue` (9) `sglang github issue adaptive speculative decoding problem "speculative-adaptive" not working` (10) `llama.cpp issue dynamic draft length speculative decoding closed wontfix maintainer` (11) `vllm RFC 41823 dynamic pruning draft tree speculative` (12) `tgi text-generation-inference issue adaptive speculation length closed not planned` (13) `vllm github pull closed unmerged "adaptive speculative" 2026` (14) `sglang issue adaptive speculative decoding not planned closed` (15) `"speculative" adaptive draft length tgi text-generation-inference github issue 2026` (16) `github issue "not planned" dynamic speculative decoding K auto-tune` (17) `llama.cpp github issue adaptive draft length speculative "n_draft" dynamic` (18) `tensorrt-llm github issue dynamic draft length speculative decoding closed not planned` (19) `sglang discussion adaptive speculative decoding tuning speculative_num_steps` (20) `vllm "speculative" "K" tuning issue 2026 "acceptance rate" dynamic draft tokens`

**discuss.vllm.ai Discourse JSON API** (`https://discuss.vllm.ai/search.json?q=`): (21) `dynamic+speculative+decoding` (22) `num_speculative_tokens` (23) `acceptance+rate` (24) `speculation+length` (25) `adaptive+K` (26) `EAGLE+num_steps` (27) `draft+acceptance+batch` (28) `speculative+num_steps`. Topic JSON reads: `https://discuss.vllm.ai/t/{2630,738,499,1096,2447}.json`

**GitHub search/list URLs:** (29) `https://github.com/vllm-project/vllm/discussions?discussions_q=speculative+decoding` (30) `https://github.com/sgl-project/sglang/issues?q=is%3Aissue+adaptive+speculative` (31) `https://github.com/sgl-project/sglang/issues?q=is%3Apr+speculative-adaptive` (32) `https://github.com/sgl-project/sglang/issues?q=is%3Aissue+speculative_num_steps` (33) `https://github.com/search?q=repo%3Asgl-project%2Fsglang+speculative_adaptive&type=issues` (34) `https://github.com/search?q=repo%3Asgl-project%2Fsglang+adaptive+num_steps&type=issues` (35) `https://github.com/search?q=repo%3Asgl-project%2Fsglang+is%3Aissue+speculative_num_steps+dynamic&type=issues` (36) `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+speculative+adaptive` (37) `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+num_speculative_tokens` (38) `https://github.com/vllm-project/vllm/issues?q=is%3Apr+speculative+adaptive+in%3Atitle`

### Search log — academic sweep (subagent; verbatim highlights)

`PRISM adaptive speculative decoding MLSys 2026` · `HELIOS adaptive speculation MLSys 2026` · `MLSys 2026 papers speculative decoding adaptive speculation length` · `bandit speculative decoding` · `adaptive speculation length LLM serving` · `dynamic draft tree budget` · `acceptance prediction head speculative decoding` · `negative result speculative decoding throughput saturates` (0 results) · `acceptance rate temperature speculative decoding analysis` (0 results). arXiv listing endpoint used: `https://arxiv.org/search/?searchtype=all&query=<url-encoded>`.

### Explicit non-findings (gaps, not evidence of absence)
- **HuggingFace TGI: no issue/PR found in this window about adaptive or dynamic draft length.** Its speculation docs cover static assisted decoding only.
- **No SGLang closed-unmerged / wontfix PR on adaptive speculation was found.** All three located adaptive artifacts (#21599, #25644, #23331) are merged.
- **No llama.cpp issue/PR explicitly rejecting adaptive draft length was found** — only missing C-API plumbing (#27089) and draft-cap patching (#26575).
- **Closing comments for vLLM #47839 and #48204 could not be retrieved** (curl HTML, embedded-JSON extraction and r.jina.ai all returned pages without them). Not guessed.


