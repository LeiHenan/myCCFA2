# Speculative Decoding — Gap Map: CLOSED / OPEN / ATTEMPTED-AND-ABANDONED / HARDWARE-RULED-OUT

**Sweep date:** 2026-09-13 · **Window:** prioritised 2025-06 → 2026-09 (older only where canonical)
**Rig assumed:** 1× RTX PRO 6000 Blackwell 96 GB, sm120, driver 580, single node; vLLM 0.29.0; SGLang 0.5.19; Qwen3-4B target + DFlash2 drafter + NGRAM.

**Provenance convention (non-negotiable, applied to every row):** `READ BODY` = the page/PDF/abstract/source/JSON payload was actually retrieved and its text read. `TITLE ONLY` = only a search-result title, snippet, or state flag was seen. Rows with no URL were dropped.

**Method note / honesty flag.** GitHub HTML issue pages do **not** embed comment bodies, and `api.github.com` was rate-limited to `0` for this IP for most of the sweep. Consequence, stated plainly: **no maintainer statement using the words "not planned" or "wontfix" about a speculative-decoding combination exists in the accessible record.** Every `not planned` closure found here is a **bare state change with zero comments**. Where a closing comment genuinely exists it is quoted; where it does not, the row says so and quotes only what was actually read. Nothing is paraphrased into a quote.

---

## A. CLOSED cells (someone shipped or published a working answer)

### A1. Draft-model drafting & draft training

| # | Question it answers | Who closed it | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A1.1 | Small LM as drafter for a larger target | vLLM | `--speculative-config '{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5,"method":"draft_model"}'` | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/draft_model.md | shipped (0.29.0) | READ BODY |
| A1.2 | Draft and target with **different vocabularies** | vLLM | Token-Level Intersection, `use_heterogeneous_vocab: true` | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/README.md | shipped (main); greedy-only | READ BODY |
| A1.3 | Target-agnostic drafter pre-training (one backbone, many targets) | Osprey | arXiv 2609.09338 — +16.1% Qwen3-8B, +21.2% Llama-3.3-70B, +22.7% MiniMax-M2.5 mean acceptance length | https://arxiv.org/abs/2609.09338 | published 2026-09-08 | READ BODY |
| A1.4 | Parallel (non-autoregressive) drafting | PARD | vLLM `"parallel_drafting": true`; `amd/PARD-Qwen3-0.6B`, k=12 | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/parallel_draft_model.md | shipped (0.29.0) | READ BODY |
| A1.5 | Train drafter against the *verification* process, not token imitation | VAT | arXiv 2608.30135 — up to +11.4% acceptance length, +8.7% wall-clock on EAGLE-3/DFlash | https://arxiv.org/abs/2608.30135 | published 2026-08-31 | READ BODY |
| A1.6 | Quantized **target** + quantized **drafter** | Competition report | arXiv 2607.04244 — Qwen3.5-4B on A10G, 6.978× avg speedup, 3rd place | https://arxiv.org/abs/2607.04244 | published 2026-07-05 | READ BODY |
| A1.7 | Single-GPU offline drafter training | SpecForge | *"**as low as 1 GPU**, as only need to accommodate the draft model"* (offline mode) | https://raw.githubusercontent.com/sgl-project/SpecForge/9b8873dfb03ed3bc10b105249f8ec6092e01b574/docs/basic_usage/training.md | shipped (docs) | READ BODY |
| A1.8 | Online draft co-training at long context / large scale | TapChannel + CP | arXiv 2609.07108 | https://arxiv.org/abs/2609.07108 | published 2026-09-07 | READ BODY |
| A1.9 | Native MTP head fine-tune entry point | vLLM Speculators | `speculators train` CLI | https://docs.vllm.ai/projects/speculators/en/latest/cli/train.html | shipped | TITLE ONLY |
| A1.10 | DFlash2 (local conv + candidate selector) upstreamed | vLLM | PR #52816, merged 2026-08-21 by WoosukKwon, SHA b389ac29 | https://github.com/vllm-project/vllm/pull/52816 | merged | TITLE ONLY (state read) |
| A1.11 | DFlash2 in SGLang | SGLang | PR #35371 — *"**3.43x over no-spec at batch 1 and about 24% over DFlash at concurrency 64**"* | https://github.com/sgl-project/sglang/releases/tag/v0.5.19 | shipped 0.5.19 | READ BODY |
| A1.12 | DFlash2 drafter quantization | community | W4A16 checkpoint with a measured calibration ablation | https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16 | published | READ BODY |

### A2. N-gram / prompt-lookup / retrieval drafting

| # | Question it answers | Who closed it | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A2.1 | Draft from prompt n-grams, no model | vLLM | `method:"ngram"`, `prompt_lookup_min`/`prompt_lookup_max` (default 5/5) | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/n_gram.md | shipped (0.29.0) | READ BODY |
| A2.2 | Suffix-tree drafting with frequency counts + adaptive depth | vLLM | `method:"suffix"`; `suffix_decoding_max_tree_depth`=24, `max_cached_requests`=10000, `max_spec_factor`=1.0, `min_token_prob`=0.1; needs `pip install arctic-inference` | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/suffix.md | shipped (0.29.0) | READ BODY |
| A2.3 | N-gram as a first-class SGLang algorithm | SGLang | `--speculative-algorithm NGRAM` | https://docs.sglang.io/docs/advanced_features/speculative_decoding | shipped (0.5.19) | READ BODY |
| A2.4 | Where n-gram/suffix pays | vLLM docs | *"Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts."* | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/suffix.md | shipped (docs claim) | READ BODY |
| A2.5 | N-gram is cheap in execution time | MLSys'26 | *"For n-gram, drafting time accounts for less than 2%"*; and it is the one method that stays above 1× at high batch | https://arxiv.org/abs/2601.11580 | published | READ BODY |
| A2.6 | N-gram × DP-attention (attempt) | SGLang | PR #28616 exists | https://github.com/sgl-project/sglang/pull/28616 | status unverified | TITLE ONLY |

### A3. EAGLE / MTP / Medusa-style heads

| # | Question it answers | Who closed it | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A3.1 | EAGLE / EAGLE-3 as supported methods | vLLM | `method:"eagle"` / `"eagle3"`, `draft_tensor_parallel_size` | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/eagle.md | shipped (0.29.0) | READ BODY |
| A3.2 | Native MTP, no separate draft model | vLLM | `method:"mtp"` (MiMo-7B; Gemma-4 assistant checkpoints → `Gemma4MTPModel`) | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/mtp.md | shipped (0.29.0) | READ BODY |
| A3.3 | Qwen3-Next MTP in vLLM | vLLM/Qwen | `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct | shipped | READ BODY |
| A3.4 | Qwen3-Next MTP in SGLang | SGLang/Qwen | `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` (NEXTN aliases EAGLE; no MTP algorithm value) | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct | shipped | READ BODY |
| A3.5 | EAGLE-3 head + FR-Spec token map | SGLang | `--speculative-algorithm EAGLE3`, `--speculative-token-map` | https://docs.sglang.io/docs/advanced_features/speculative_decoding | shipped (0.5.19) | READ BODY |
| A3.6 | UNO (target + trained LoRA adapter, no separate model) | SGLang | `--speculative-algorithm UNO --uno-lora-path ...` | https://docs.sglang.io/docs/advanced_features/speculative_decoding | shipped (0.5.19) | READ BODY |
| A3.7 | MTP head count vs acceptance, quantified | FastMTP | arXiv 2509.18362 — vanilla MTP 70%→10%→~0% at k=1/2/3; fine-tuned 80/56/36% | https://arxiv.org/abs/2509.18362 | published | READ BODY |
| A3.8 | Medusa (typical acceptance + simultaneous tree verification) | Medusa, ICML 2024 | arXiv 2401.10774 | https://arxiv.org/abs/2401.10774 | published | READ BODY |
| A3.9 | EAGLE feature-level draft+verify | EAGLE | arXiv 2401.15077 | https://arxiv.org/abs/2401.15077 | published | READ BODY |
| A3.10 | EAGLE-3 head shipped as weights | together/RedHat | `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/eagle.md | shipped (weights) | READ BODY |
| A3.11 | Hidden-state extraction required to train heads | vLLM | `extract_hidden_states` doc + PR #49811 in 0.29.0 | https://docs.vllm.ai/en/latest/features/speculative_decoding/extract_hidden_states/ | shipped (0.29.0) | READ BODY |
| A3.12 | Multi-adapter LoRA with EAGLE/NEXTN/DFLASH/DSPARK | SGLang | PR #34337 | https://github.com/sgl-project/sglang/releases/tag/v0.5.19 | shipped 0.5.19 | READ BODY |

### A4. Draft budget / speculation-length adaptation

| # | Question it answers | Who closed it | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A4.1 | **Static K per batch-size band** | vLLM | Dynamic Speculative Decoding: `num_speculative_tokens_per_batch_size: [[1,64,3],[65,128,1],[129,512,0]]` (PR #32374) | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/dynamic_speculative_decoding.md · https://github.com/vllm-project/vllm/pull/32374 | shipped (0.29.0) | READ BODY |
| A4.2 | **Per-(request,position) verification trimming** under a global budget | vLLM (DSpark) | Adaptive Verification: `enable_adaptive_verification: true` (PR #47808, merged 2026-08-12); blog 2026-08-14 | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/adaptive_verification.md · https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification | shipped (0.29.0), DSpark-only | READ BODY |
| A4.3 | **Runtime EMA step-count controller per BS tier** with pre-built CUDA graphs | SGLang | `--speculative-adaptive`; `target_steps = clamp(round(ema_accept_len) + 1, min, max)` | https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding · https://github.com/sgl-project/sglang/blob/c69844f0/python/sglang/srt/speculative/adaptive_spec_params.py | shipped (0.5.19) | READ BODY |
| A4.4 | Confidence-scheduled verification + semi-AR drafting | DSpark | arXiv 2607.05147 — deployed in DeepSeek-V4 serving; *"accelerates per-user generation speeds by 60 to 85 percent at matched throughput levels"* vs MTP-1 | https://arxiv.org/abs/2607.05147 | published 2026-07-06 | READ BODY |
| A4.5 | Adaptive verification depth pruning **for batched** SD | D-Cut | arXiv 2607.14647 — 1.26× → 1.65× at high concurrency; up to 3.0× on MoE | https://arxiv.org/abs/2607.14647 | published 2026-07-16 | READ BODY |
| A4.6 | Adaptive candidate length (canonical older ref) | SpecDec++ | arXiv 2405.19715 (COLM'25) — MDP → threshold policy + acceptance head | https://arxiv.org/abs/2405.19715 | published | READ BODY |
| A4.7 | Sequence-length threshold policy (canonical) | MagicDec | arXiv 2408.11049 | https://arxiv.org/abs/2408.11049 | published | READ BODY |
| A4.8 | Closed-loop goodput controller | TurboSpec | arXiv 2406.14066 | https://arxiv.org/abs/2406.14066 | published | READ BODY |
| A4.9 | Acceptance vs batch size, as a scaling law | Scaling Laws for SD | arXiv 2505.07858 — log-linear acceptance vs batch size | https://arxiv.org/abs/2505.07858 | published | READ BODY |
| A4.10 | Per-request utility-driven K for MoE | Cascade | arXiv 2506.20675 (MICRO'25 / MLSys'26) | https://arxiv.org/abs/2506.20675 | published | READ BODY |
| A4.11 | Bandit/control cluster for K selection | BanditSpec/UCBSpec/EXP3Spec, Not-a-Bandit, TapOut, Nightjar, SADDLE | arXiv (multiple) | https://arxiv.org/abs/2502.08468 · https://arxiv.org/abs/2512.22420 | published | READ BODY |
| A4.12 | Instance-adaptive policy for block-diffusion drafting | BlockPilot | arXiv 2606.31315 | https://arxiv.org/abs/2606.31315 | published | TITLE ONLY |
| A4.13 | Budget-aware block-diffusion drafting | Bastion | arXiv 2605.29727 | https://arxiv.org/abs/2605.29727 | published | TITLE ONLY |
| A4.14 | Dynamic draft length in TensorRT-LLM | NVIDIA | PRs #10860, #8194, #7728 | https://github.com/NVIDIA/TensorRT-LLM/pulls | merged | TITLE ONLY |
| A4.15 | Entropy lower bound for draft length | AdaEDL | arXiv 2412.18910 | https://arxiv.org/abs/2412.18910 | published | TITLE ONLY |
| A4.16 | K selection observability (proposed) | vLLM | PR #54748 — `vllm:spec_decode_scheduled_steps` labelled by selected K | https://github.com/vllm-project/vllm/pull/54748 | **OPEN** | READ BODY |

### A5. Verification strategy

| # | Question it answers | Who closed it | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A5.1 | Exact SD = same distribution as target | Leviathan et al., ICML 2023 Oral | arXiv 2211.17192 — *"an algorithm to sample from autoregressive models faster without any changes to the outputs ... and without changing the distribution."* | https://arxiv.org/abs/2211.17192 | published | READ BODY |
| A5.2 | Modified rejection sampling preserving the target distribution | Chen et al. (DeepMind) 2023 | arXiv 2302.01318 — *"a novel modified rejection sampling scheme which preserves the distribution of the target model within hardware numerics."* | https://arxiv.org/abs/2302.01318 | published | READ BODY |
| A5.3 | Multi-candidate verification via optimal transport | SpecTr, NeurIPS 2023 | arXiv 2310.15141 | https://arxiv.org/abs/2310.15141 | published | READ BODY |
| A5.4 | Typical acceptance + simultaneous tree verification | Medusa, ICML 2024 | arXiv 2401.10774 | https://arxiv.org/abs/2401.10774 | published | READ BODY |
| A5.5 | vLLM v1 verifier implements the 2211.17192 algorithm | vLLM (woosuk) | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` — docstring: *"The implementation strictly follows the algorithm described in https://arxiv.org/abs/2211.17192."* Accept test: `accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py | shipped | READ BODY |
| A5.6 | Typical acceptance as an alternate verifier in vLLM | simon-mo | PR #5131 | https://github.com/vllm-project/vllm/pull/5131 | merged | READ BODY |
| A5.7 | Real distribution bug in the verifier root-caused + fixed | WoosukKwon, merged 2026-08-28 | PR #54282 | https://github.com/vllm-project/vllm/pull/54282 | merged | READ BODY |
| A5.8 | SGLang tree-mask construction (3 modes) + ragged verify | SGLang | `eagle_utils.py`; `ragged_verify.py` (`SGLANG_RAGGED_VERIFY_MODE` = static/cap-accept/compact) | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/speculative/ragged_verify.py | shipped | READ BODY |
| A5.9 | Cross-TP seed semantics required by SD | WoosukKwon | PR #17929 (default seed 0 in V1) | https://github.com/vllm-project/vllm/pull/17929 | merged | READ BODY |
| A5.10 | sm120 EAGLE3 CUDA-graph illegal memory access fixed | hnyls2002 | SGLang PR #10892 (fixes #9888, #6309, #8296, #8336) — *"Reset self.positions in EAGLE CUDA graph runners when raw_bs and capture_bs differ."* | https://github.com/sgl-project/sglang/pull/10892 | merged | READ BODY |
| A5.11 | HF assisted-generation distribution bug fixed | Cyrilvallez | HF PR #48007 (closes #47932) | https://github.com/huggingface/transformers/pull/48007 | merged | READ BODY |
| A5.12 | HF: assistant must sample when target samples | gante | HF PR #33534 (reverts #30778) | https://github.com/huggingface/transformers/pull/33534 | merged | READ BODY |
| A5.13 | Verification is computationally free; the cost is runtime | 2026 | arXiv 2607.17283 — *"Across all five configurations, rejection-sampling arithmetic is under 1 ms per generated token — the mathematically delicate part of speculative decoding is computationally free."* | https://arxiv.org/abs/2607.17283 | published | READ BODY |
| A5.14 | Batched-verification correctness invariants (ragged tensors) | 2025 | arXiv 2510.22876 | https://arxiv.org/abs/2510.22876 | published | READ BODY |
| A5.15 | Verification allocation / which drafts merit verifying | 2026 | arXiv 2606.01019 | https://arxiv.org/abs/2606.01019 | published | READ BODY |
| A5.16 | Parallelising verification itself | Saguaro | arXiv 2603.03251 | https://arxiv.org/abs/2603.03251 | published | READ BODY |
| A5.17 | Quantized-KV made lossless by drafting on compressed KV, verifying on full KV | VeriCache | arXiv 2605.17613 | https://arxiv.org/abs/2605.17613 | published | READ BODY |
| A5.18 | Single-forward tree verification for diffusion drafters | DDTree | arXiv 2604.12989 | https://arxiv.org/abs/2604.12989 | published | READ BODY |
| A5.19 | OT verification extended to multi-draft block verification | SpecTr-GBV | arXiv 2604.25925 | https://arxiv.org/abs/2604.25925 | published | READ BODY |
| A5.20 | Verification-memory OOM fixed by chunking | youkaichao | vLLM PR #48630 | https://github.com/vllm-project/vllm/pull/48630 | merged | READ BODY |

### A6. Batching / quantization / caching + engine integration

| # | Question it answers | Who closed it | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| A6.1 | Full method list + qualitative high/low-QPS table | vLLM 0.29.0 | 12 spec-decode sub-docs | https://docs.vllm.ai/en/latest/features/speculative_decoding/ | shipped (0.29.0) | READ BODY |
| A6.2 | Full method list + constraint table | SGLang 0.5.19 | EAGLE-2/3, MTP, UNO, DFLASH, STANDALONE, NGRAM, FR-Spec | https://docs.sglang.io/docs/advanced_features/speculative_decoding | shipped (0.5.19) | READ BODY |
| A6.3 | Per-request acceptance metrics in API responses | vLLM | `--per-request-spec-decode-metrics summary\|detailed` (PR #48915); **404 at v0.28.0, 200 at v0.29.0** | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/acceptance_metrics.md | shipped (new in 0.29.0) | READ BODY |
| A6.4 | Mamba+EAGLE prefix-cache regression | vLLM | PRs #55760, #55861, merged 2026-09-08; in v0.29.0 — *"dense retention automatically restored for hybrid models using EAGLE/MTP"* | https://github.com/vllm-project/vllm/releases/tag/v0.29.0 | merged/shipped | READ BODY |
| A6.5 | Prefix-cache hit-rate drop from EAGLE/MTP explained | vLLM | PR #51769 + RFC #50438 | https://github.com/vllm-project/vllm/pull/51769 | merged (diagnosis) | READ BODY |
| A6.6 | Trailing-block-drop opt-out | vLLM | PR #53388, merged 2026-09-01 | https://github.com/vllm-project/vllm/pull/53388 | merged | READ BODY |
| A6.7 | EAGLE draft quant + KV quant compose | vLLM | PR #28435, merged 2025-11-17 | https://github.com/vllm-project/vllm/pull/28435 | merged | READ BODY |
| A6.8 | FP8 target + Eagle3 KV dtype | vLLM | PR #24505 | https://github.com/vllm-project/vllm/pull/24505 | merged | READ BODY |
| A6.9 | Gemma-4 FP8 MTP acceptance 0% → 60% | SGLang | PR #32440 | https://github.com/sgl-project/sglang/pull/32440 | merged 2026-07-26 | READ BODY |
| A6.10 | Mixed chunked prefill with spec enabled | SGLang | PR #36933 (generalises past the DFLASH/NGRAM doc restriction) | https://github.com/sgl-project/sglang/pull/36933 | merged | READ BODY |
| A6.11 | Grammar × spec decode made schedulable | SGLang | PR #12615 (overlap-scheduling guard); PR #30096 (removed the hard 400) | https://github.com/sgl-project/sglang/pull/12615 · https://github.com/sgl-project/sglang/pull/30096 | merged | READ BODY |
| A6.12 | DSD + full CUDA graphs | vLLM | PR #45953, merged 2026-07-04 | https://github.com/vllm-project/vllm/pull/45953 | merged | READ BODY |
| A6.13 | Chunked-prefill valid_mask bug | vLLM | PR #26231, merged 2025-10-06 | https://github.com/vllm-project/vllm/pull/26231 | merged | READ BODY |
| A6.14 | Release-level status | vLLM | v0.29.0, released **09 Sep 2026** by khluu, 621 commits since | https://github.com/vllm-project/vllm/releases/tag/v0.29.0 | released | READ BODY |
| A6.15 | Release-level status | SGLang | v0.5.19, released **05 Sep 2026** by Qiaolin-Yu, 629 commits, 786 PRs / 214 contributors | https://github.com/sgl-project/sglang/releases/tag/v0.5.19 | released | READ BODY |
| A6.16 | ML-SpecQD (quantized drafts) | 2025 | arXiv 2503.13565 — MXFP4 drafts, up to 2.72× | https://arxiv.org/abs/2503.13565 | published | READ BODY |
| A6.17 | Quantized self-spec verification | Quasar | arXiv 2603.01399 | https://arxiv.org/abs/2603.01399 | published | READ BODY |
| A6.18 | Hierarchical quantized KV self-speculation | QuantSpec | arXiv 2502.10424 | https://arxiv.org/abs/2502.10424 | published | READ BODY |

---

## B. OPEN cells (explicitly unsolved, with evidence that someone is still asking)

### B1. Is speculation worth it at batch size > 1?

**B1.1 — The K-decision has six competing signals and no contract.**
WHO: vLLM RFC #54749 (OPEN). URL: https://github.com/vllm-project/vllm/issues/54749 · READ BODY
VERBATIM: *"Six different signals are being proposed for this one decision right now."* and *"Batch size is what ships."* and *"there is no counter that would have told me so from inside vLLM."*
Missing: a stable interface for the K decision.

**B1.2 — K=0 is indistinguishable from "speculation is off" in metrics.**
WHO: PR #54748 (OPEN). URL: https://github.com/vllm-project/vllm/pull/54748 · READ BODY
VERBATIM: *"Steps the schedule sent to K=0 increment no counter, so 'the schedule stopped speculating on purpose' and 'speculation is not running' look identical in metrics."*

**B1.3 — Per-request proposal-length metadata has no contract.**
WHO: vLLM RFC #48202 (CLOSED with no maintainer answer). URL: https://github.com/vllm-project/vllm/issues/48202 · READ BODY
VERBATIM: *"The missing piece is not a policy but a contract: per-request proposal-length metadata and its semantics across the proposer, model runner, scheduler accounting, async scheduling, and metrics."*
Measured negative in the same RFC, VERBATIM: *"on Qwen3-4B on a single L20X up to batch 64, confidence early stopping did not improve wall clock over fixed full blocks at any threshold we tried"*.

**B1.4 — MTP is a net loss even with good acceptance.**
WHO: vLLM #47277 (OPEN, assigned). URL: https://github.com/vllm-project/vllm/issues/47277 · READ BODY
VERBATIM: *"MTP1 reaches around 82%-88% acceptance, and MTP2 can reach mean accepted length above 2.3. However, end-to-end throughput in graph mode is still below the best no-MTP vLLM baseline."* Numbers: no-MTP 2246.73 tok/s vs MTP1 1926.41 vs MTP2 1821.76. *"MTP1 is about 0.86x of the no-MTP graph baseline, not a speedup."*
Abandoned local attempts, VERBATIM: *"We also tried more aggressive local fast paths: a greedy rejection fast path for temperature=0, packed/allfast variants, faster attention metadata experiments, GPU token cache / active request gating variants. Some of these either regressed performance or produced non-clean outputs such as length:1 i…"*

**B1.5 — Acceptance length collapses under multi-request batching (DSpark).**
WHO: TensorRT-LLM #16767 (OPEN). URL: https://github.com/NVIDIA/TensorRT-LLM/issues/16767 · READ BODY
VERBATIM: *"DSpark speculative decoding... produces the expected accept length only at generation batch size 1. As soon as the disaggregated generation server batches more than one request, the measured accept length collapses toward ~1.0 (drafts are almost never accepted), so DSpark provides essentially no speedup under real serving load."*
Table: draft len 5 → AL 4.03 @ bs1 vs 1.52 @ bs16; draft len 7 → 4.51 vs 1.49. VERBATIM: *"Accept length is a per-request property and should be batch-invariant. MTP on the identical setup is ~2.85 at both batch 1 and batch 16."*

**B1.6 — Dynamic SD crashes/regresses at K=0 tiers.**
WHO: vLLM #48494, #49548 (both OPEN). URL: https://github.com/vllm-project/vllm/issues/48494 · https://github.com/vllm-project/vllm/issues/49548 · READ BODY
VERBATIM (#48494): *"The presence of the batch-size table is the trigger; K=0 tiers are not required."*
VERBATIM (#49548): *"catastrophic aggregate-throughput collapse under concurrency"* — 8 concurrent: ~232 tok/s static vs 24–157 tok/s dynamic, wall-clock 6.3 s → 40–60 s.

**B1.7 — Long-context speculation is net-negative and there is no per-sequence-length hook.**
WHO: vLLM #54691 / PR #54801 / #52258 / #47602 (OPEN). URL: https://github.com/vllm-project/vllm/issues/54691 · https://github.com/vllm-project/vllm/pull/54801 · READ BODY
VERBATIM (#54691): *"On hybrid GDN models (Qwen3.5-family), DFlash speculative decoding becomes a net slowdown at long context: single-stream 185k-context decode drops from ~71 tok/s (spec OFF) to ~16 tok/s (DT=4)."* and *"At 185k this is ~182 ms of FA kernel time per cycle (GPU fully occupied), so verification wins from speculation are completely eaten by the drafter's full-context scan."* and *"But nothing today lets a deployment say 'stop speculating beyond N context tokens'."*
VERBATIM (#52258): long-doc summarization *"collapses to 5–8% acceptance, making speculation net-negative"*; `max_model_len` is *"a no-op"*.
VERBATIM (#47602): *"a static global on/off leaves speedup on the table at short context while actively regressing long-context throughput"*.
Shipped attempt: PR #55412 `disable_spec_decode` in `SamplingParams` — https://github.com/vllm-project/vllm/pull/55412 (status as of last read: OPEN).
Same demand in SGLang #30263, VERBATIM: *"Speculative decoding is also not free under load: verification cost scales with batch_size × draft_length, so SD helps at low batch size but can reduce goodput at high batch size"* and the RFC's own explanation of why it is hard: *"The num_steps == 0 path is only consistent when the steps-0 runtime state is active."* / *"So per-request opt-out requires a runtime-state switch, not just an input-shape change."* — https://github.com/sgl-project/sglang/issues/30263

**B1.8 — Entropy-gated dynamic k, and context-length-dependent K.**
WHO: vLLM #54082 (OPEN, 2026-08-27 — the most recent asking in the window); vLLM PR #48944 (OPEN); SGLang PR #31716 (OPEN). URL: https://github.com/vllm-project/vllm/issues/54082 · READ BODY (status), TITLE ONLY (body)

**B1.9 — MoE adaptive depth never upstreamed.**
WHO: vLLM #44506 (OPEN, labelled stale), RFC #46295 (OPEN). URL: https://github.com/vllm-project/vllm/issues/44506 · READ BODY (state) / TITLE ONLY (body)

**B1.10 — The MLSys'26 oral's own future-work ask.**
WHO: Liu, Yu, Park, Stoica, Cheung — arXiv 2601.11580. URL: https://arxiv.org/abs/2601.11580 · READ BODY
VERBATIM: *"Taken together, these results point to a promising direction for future work: developing an accurate yet lightweight predictor capable of adapting to varying levels of acceptance behavior across workloads, requests and token positions."*
VERBATIM: *"We leave an analysis that measures overlap with respect to this evolving context for future work."*
VERBATIM: *"Finally, we do not account for the overhead of maintaining the KV cache for EAGLE's proposing head, which would otherwise require partial prefills if a request was paused and resumed."*
Oracle headroom, VERBATIM: *"the oracle setup achieves a speedup of approximately 2.75×, whereas the fixed proposed length configuration attains about 2.1× for the best proposed length of 5. Moreover, the gap generally widens as batch size increases"*.

**B1.11 — Thinking-vs-answer phase budgeting: genuinely empty lane.**
No paper found that budgets speculation separately for the reasoning phase vs the answer phase. Closest signal (READ BODY, arXiv 2601.11580): *"For n-gram, we additionally observe a decrease towards the last few token position intervals. This is likely because the model shifts from step-by-step reasoning ... to writing the final answer and stopping, which is less repetitive"*.

### B2. Is spec-decode output bit-identical / distribution-identical to non-speculative?

**B2.1 — Verify↔decode bitwise parity is undefined and undecided.**
WHO: vLLM RFC #54506 (OPEN, 2026-08-31). URL: https://github.com/vllm-project/vllm/issues/54506 · READ BODY
VERBATIM: *"the dominant cause is forward-pass arithmetic, not sampling. The target model's verification pass computes rows with M = k+1, plain decode computes the same rows with M = 1, and nothing in the current stack guarantees those two produce identical bits."*
VERBATIM: *"The invariant being violated: given the same checkpoint, the same input and the same runtime/load configuration, the target model's verify pass over rows [t..t+k] must reproduce, bit for bit, what plain M=1 decode would produce at each of those positions."*
VERBATIM ask: *"Should verify↔decode bitwise parity be an explicitly documented guarantee under a batch-invariant mode (with the cost that implies), or an explicit non-guarantee — so that users stop treating greedy-output changes under spec decode as a bug in their own configuration?"*
VERBATIM: *"If it becomes a guarantee, does it cover the compilation layer as well as kernels? Class B and Class C say that guaranteeing only kernels is not enough: fusion topology and per-engine execution form change results on their own."*
Measured symptom: acceptance ~100% → 72%, decode ~110 → 89 tok/s. Author later self-corrected two sub-claims (load strategy; token-index migration) but not the M-axis mechanism.
Related ask in the same thread, VERBATIM (yuvalluria, 2026-08-31): *"On scope: VLLM_BATCH_INVARIANT should mean what it says. A carve-out in docs is fine short-term, but the guarantee has to eventually cover all execution modes or users will file it as a bug every time."*

**B2.2 — n-gram speculation changes greedy output; died by inactivity, not resolution.**
WHO: vLLM #41758 — **auto-closed stale 2026-09-13 (the sweep date)**. URL: https://github.com/vllm-project/vllm/issues/41758 · READ BODY
VERBATIM stale bot 2026-08-10: *"This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days."*
VERBATIM close 2026-09-13: *"This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant."*
Reporter's narrowing, VERBATIM: *"In the instrumented rerun, scheduled_spec_decode_tokens was empty up to and including the decode row that emitted the first divergent generated token."*
Third-party repro, VERBATIM (mssucc): *"I tried to reproduce your situation, and actually I did not find the problem of ngram speculative decoding changing greedy output, but I found that when VLLM_BATCH_INVARIANT is not set, the first decoding result is inconsistent with subsequent results ... if VLLM_BATCH_INVARIANT=1 is set, this problem can be solved; the principle seems to be rounding error caused by chunked computation."*
No maintainer root-cause was ever posted.

**B2.3 — Greedy spec decode not token-identical (DFlash2 + thinking).**
WHO: vLLM #54928 (OPEN); SGLang #38009 (OPEN). URL: https://github.com/vllm-project/vllm/issues/54928 · https://github.com/sgl-project/sglang/issues/38009 · READ BODY
VERBATIM: *"Qwen3.8-27B with the Qwen3.8-27B-DFlash2 draft is not greedy-equivalent to the same BF16 target when Qwen thinking is enabled. This is reproducible with temperature=0, so it is a correctness issue rather than a normal sampling difference."* / *"With greedy decoding, speculative decoding should emit the same token sequence as target-only decoding."* (K=1 and `--enforce-eager` both affected.)

**B2.4 — DSpark forced-reject still not lossless; recurrent-state drift.**
WHO: SGLang #35150 (OPEN). URL: https://github.com/sgl-project/sglang/issues/35150 · READ BODY
VERBATIM: *"I can reproduce a deterministic correctness divergence between ordinary Base decode and DSpark TARGET_VERIFY even when every speculative draft token is deliberately rejected and only the target model's own argmax token is committed."* / *"The failure is cumulative rather than context-local."* / *"This suggests accumulated GDN recurrent-state drift somewhere in the TARGET_VERIFY / intermediate-state / commit path."*

**B2.5 — NPU hybrid-GDN MTP not lossless (closed by stale bot, no rebuttal).**
URL: https://github.com/sgl-project/sglang/issues/25587 · READ BODY
VERBATIM: *"At temperature=0.0, MTP speculative decoding output must be bitwise-identical to non-speculative decoding. This is guaranteed on NVIDIA and is the standard correctness criterion for speculative decoding."* / *"SiLU is non-linear and non-invertible"* / *"NPU acceptance rate 15–20% lower than H100"*.

**B2.6 — Batch-invariant speculative decoding.**
WHO: vLLM tracking issue #27433 (OPEN, 29 tasks). URL: https://github.com/vllm-project/vllm/issues/27433 · READ BODY
VERBATIM task item: *"Speculative decoding support (this might be hard)"* → #52522.
WHO is still asking — chenzhuofu, 2026-01-06, VERBATIM: *"I'm really curious about the status of batch-invariant support for spec decoding. Could you help explain more about why it is hard to support?"*
Maintainer yewentao256, 2026-01-06, VERBATIM: *"Hi @chenzhuofu, There are several things we need to do to support spec decoding, eg. support of TreeAttentionBackend attn backend, The accept/reject step, and torch.rand(...) RNG path. If you are interested in helping with this topic, feel free to have a sub issue discussing further."*
VERBATIM on hardware policy: *"Currently Hopper and Blackwell are fully supported and well tested."* and *"Sorry, older GPUs are not in our officialy support roadmap currently."*

**B2.7 — The BI spec-decode PR is narrow, taxes throughput, and omits MTP/GDN.**
WHO: vLLM PR #52522 (OPEN since 2026-08-16, 7 commits, needs-rebase). URL: https://github.com/vllm-project/vllm/pull/52522 · READ BODY
VERBATIM hard gate: *"VLLM_BATCH_INVARIANT only supports speculative decoding with Model Runner V2, EAGLE3/DFlash/DSpark, probabilistic drafting, standard rejection sampling, fixed speculative lengths, and adaptive verification disabled."*
VERBATIM mechanism: *"truncation could make the same position a draft token in one batch but a bonus token in another. This can definitely cause different tokens."*
VERBATIM self-correction (2026-09-02): *"I found that, in my original gsm8k evaluation, num_preemptions_total was zero, so recovery never worked. 0 mismatch mainly came from forcing batch-invariant requests to use the same Triton top-k/top-p implementation."*
Measured BI tax: EAGLE3 accuracy −0.30 pp, Req/s −25.03%, Output tok/s −24.64%.
Named gaps, VERBATIM: yuvalluria — *"One gap for the record: MTP/GDN (hybrid linear-attn/MoE) is not covered — the Mamba scan's h_C is also M-sensitive on recovery."*; andakai — *"not tested on mtp yet"*.

**B2.8 — BI mode hard-aborts on hybrid/GDN models.**
WHO: vLLM #42960 (OPEN). URL: https://github.com/vllm-project/vllm/issues/42960 · READ BODY
VERBATIM: *"RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN."* / *"This is a hard incompatibility — no fallback, no partial mode. It blocks reproducibility work for all Qwen3-Next / Qwen3.6-style models (and any other hybrid Mamba/GDN architecture)."*
Independent confirmation (WenyueD, 2026-09-01, Qwen3.8-27B + DFlash2 on A800), VERBATIM: *"Without batch invariance, target-only and DFlash2 reproducibly diverged at output token index 44: target-only selected token 3710 while DFlash2 selected 9930."* / *"The decisive batch-invariant test cannot currently run: even target-only initialization aborts with: RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN."*

**B2.9 — Sampling-parameter support is incomplete under spec decode.**
- vLLM source, VERBATIM: *"we can use top_p, top_k sampling for bonus tokens, while spec decode does not support these sampling strategies."*
- vLLM `sampling_params.py::_validate_spec_decode`, VERBATIM: *"The min_p and logit_bias sampling parameters are not yet supported with speculative decoding."* — https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/sampling_params.py · READ BODY
- vLLM PR #42802 (OPEN), VERBATIM: *"The consequence is that under speculative decoding, min_p is silently dropped on every verified target token — only the bonus token via the regular Sampler path honors it."* / *"Without min_p under spec decoding, Qwen3-Coder-Next 80B-A3B (and similar code-tuned models) hit repetition collapse on long completions"* — https://github.com/vllm-project/vllm/pull/42802 · READ BODY
- TRT-LLM refuses rather than silently drops, VERBATIM: *"min_p has no buffer there, so it would be silently dropped and the request would decode from a different distribution than the user asked for. Threading it through costs measurable throughput on the rejection path, so reject instead."* — https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/tensorrt_llm/_torch/speculative/spec_sampler_base.py · READ BODY

**B2.10 — Top-k tie-handling makes sampling batch-size dependent.**
WHO: vLLM #42259 (OPEN) + PR #50979 (OPEN). URL: https://github.com/vllm-project/vllm/issues/42259 · https://github.com/vllm-project/vllm/pull/50979 · READ BODY
VERBATIM: *"Because apply_top_k_top_p dispatches by batch size (logits.shape[0] >= 8 → Triton, else PyTorch), the same request can produce different sampled tokens depending on whether it is run alone or in a batch of >= 8."* / *"Ties in model logits are not rare: with bf16 weights (e.g. Qwen3-0.6B), ties within the top-8 occur at a large fraction of sampled positions."*
VERBATIM from #42259 scope: *"batch-invariance gaps across async scheduling, TP collectives, attention, convolution, and speculative decoding"*; *"MRV2 sampling/spec-decode kernels can emit an out-of-vocabulary ID when the final logits tiles contain only -inf"*.

**B2.11 — Top-p/top-k renorm nondeterminism can wedge TP ranks.**
WHO: SGLang PR #38565 (OPEN). URL: https://github.com/sgl-project/sglang/pull/38565 · READ BODY
VERBATIM: *"both of those are non-deterministic run to run on byte-identical input"* / *"When a coin lands inside the few-ULP gap between two ranks' target_probs, one rank alone accepts/rejects a draft token or picks a different bonus token. Its radix/KV state silently drifts from the other ranks ... and the ranks wedge in an NCCL collective."* Measured: 199/199 calls differ.

**B2.12 — vLLM's own documented non-guarantee.**
URL: https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/README.md · READ BODY
VERBATIM: *"While vLLM strives to ensure losslessness in speculative decoding, variations in generated outputs with and without speculative decoding can occur due to following factors: Floating-Point Precision... Batch Size and Numerical Stability: Changes in batch size may cause variations in logprobs and output probabilities"*. Also: *"vLLM does not currently guarantee stable token log probabilities (logprobs)."*

**B2.13 — No bounded-lossy greedy verification in HF.**
WHO: HF #48636 (OPEN). URL: https://github.com/huggingface/transformers/issues/48636 · READ BODY
VERBATIM: *"Greedy assisted decoding in Transformers uses strict argmax verification: the entire draft suffix is discarded at the first position where the draft token differs from the target argmax. This is lossless, but it rejects near-optimal draft tokens whose logit gap to the argmax is negligible, capping the practical speedup"* / *"Greedy decoding, however, still lacks a deterministic, explicitly-bounded relaxation mechanism."*

**B2.14 — Open paper-level question: which multi-path verification method is best?**
VERBATIM (arXiv 2602.16994): *"While prior work has proposed various verification algorithms for i.i.d rollouts, their relative performance under matched settings remains unclear."* / *"Therefore, an important open question remains: which multi-path verification method performs best, and under what conditions?"* — https://arxiv.org/abs/2602.16994 · READ BODY

### B3. Verification correctness bugs still open

| # | Bug | URL | Verbatim | Read |
|---|---|---|---|---|
| B3.1 | MTP + chunked prefill → word salad on **1× RTX PRO 6000 Blackwell sm_120** | https://github.com/vllm-project/vllm/issues/55894 | *"a request whose first decode step is scheduled while a long prompt is being chunk-prefilled comes back as word salad or an empty loop of special tokens for the rest of its life, from roughly its fifth token on. About 0.2–1% of requests on a mixed workload; up to 7% with a targeted trigger."* / *"That holds for token-KV attention; for backends with per-draft recurrent state it is a correctness bug."* Trigger table: `no MTP → 0/400`; `runner reorder_batch_threshold forced to 1 + k (monkeypatch) → 0/400` | READ BODY |
| B3.2 | MTP acceptance drops to exactly 0 mid-run, cross-model | https://github.com/vllm-project/vllm/issues/55357 | *"per-window MTP draft acceptance drops to exactly 0 (~2,000 drafted, 0 accepted), generation pins to the 1-token/step spec-decode floor (~66 tok/s vs 95–135 tok/s healthy at 50–85% acceptance on the same pod), and the model emits an endless repetition loop inside the thinking block"* / *"a second, architecturally unrelated model on the same image with the same MTP config — Gemma 4 26B (NVFP4, thinking on) — failed its garbage probe the same day with the identical signature."* | READ BODY |
| B3.3 | EAGLE verification wrong with sliding-window attention | https://github.com/sgl-project/sglang/pull/39287 | *"EAGLE verification can produce incorrect outputs with sliding-window attention in FlashInfer and Triton."* / *"Cross-backend validation also found a missing causal mask in TRT-LLM MHA's XQA verification call."* MT-Bench 6.1063 → 8.9313 at temp=0 | READ BODY |
| B3.4 | EAGLE verify commits argmax on ROCm, ignoring temperature/top_p | https://github.com/sgl-project/sglang/pull/37134 (re-filed; #31214 closed unmerged) | *"EAGLE spec-decode verify has been committing argmax on ROCm, ignoring [temperature and top_p]"* | READ BODY |
| B3.5 | Compact ragged verify appears unusable | https://github.com/sgl-project/sglang/issues/39173 | *"AssertionError: engram target-verify expects one equal block per request, got 42 tokens for 8 requests of 6"* / *"compact ragged verify currently appears unusable for this model."* | READ BODY |
| B3.6 | ngram verifier silently drops tool-call arguments | https://github.com/vllm-project/vllm/issues/56077 | *"tool_calls[0].function.arguments == \"{}\" (empty, argument silently dropped)"* | READ BODY |
| B3.7 | CPU spec-decode verifier reads uninitialized temperature/top_k/top_p | https://github.com/vllm-project/vllm/issues/55005 | *"on any CPU spec-decode run with a non-greedy request, logits.div_(temperature) and apply_top_k_top_p consume uninitialized new_empty memory — fresh zero pages give div_(0) → inf logits; arbitrary garbage gives the out-of-bounds crash previously observed downstream."* / *"This was reported before, and the fix everyone believed in is the buggy code"* | READ BODY |
| B3.8 | Spec decode corrupts returned logprobs (llama.cpp) | https://github.com/ggml-org/llama.cpp/issues/27972 | *"it runs after verification and never inspects weights, tensor layout, or spec type, so every --spec-type and every target/drafter"* [is affected] | READ BODY |
| B3.9 | Verification materializes an FP32 `[num_verification_tokens, vocab_size]` canvas | https://github.com/vllm-project/vllm/pull/53630 | *"the target rejection verifier currently materializes an FP32 processing canvas of shape [num_verification_tokens, vocab_size]. The production path bounds the transient allocation with a 1 GiB chunk, which in turn forces a cap on the total logits adaptive verification may schedule per step."* | READ BODY |
| B3.10 | k+1 verify routing in compressed-KV attention, tested only on Ampere | https://github.com/vllm-project/vllm/pull/40914 | *"⚠️ Tested ONLY on NVIDIA Ampere SM 8.6 (RTX A5000 primary, RTX 3090 cross-rig confirmation by @noonghunna). Hopper / Blackwell not yet tested."* — OPEN since 2026-04-26 | READ BODY |
| B3.11 | CUDA-graph support level depends on head geometry | https://github.com/vllm-project/vllm/issues/55581 | *"decides the CUDA-graph support level of a KV-cache group by dividing the target model's per-rank query-head count by that group's num_kv_heads"* | READ BODY |
| B3.12 | SGLang quantized draft gives ~0% acceptance, never root-caused | https://github.com/sgl-project/sglang/issues/39087 | *"This is a read-only report — I have not root-caused it and am not opening a PR."* / *"Decode is slower than running with no drafter at all, because it pays full verify cost for zero accepted tokens."* / ask: *"Warn (or refuse) when a DFLASH draft resolves to a quantized checkpoint"* | READ BODY |

### B4. Heads / drafters

**B4.1 — Quantized target silently zeroes EAGLE acceptance.**
WHO: vLLM #26402 (OPEN). URL: https://github.com/vllm-project/vllm/issues/26402 · READ BODY
VERBATIM: *"When using an EAGLE head with a compressed tensors quantized model the acceptance rate silently fails to zero and therefore the perf completely degrades."* / *"Hence vLLM thinks the EAGLE layers are quantized, but they aren't. Surprisingly this does not give an error, but the acceptance rate essentially drops to zero…"* / *"The bigger problem IMO is that the EagleProposer simply takes the VllmConfig from the target model. This is not robust whenever the draft model has some different configurations."*

**B4.2 — EAGLE-3 works at 32K but collapses at 262K.**
WHO: vLLM #37773 (OPEN). URL: https://github.com/vllm-project/vllm/issues/37773 · READ BODY
VERBATIM: *"The same model + EAGLE-3 head works perfectly at --max-model-len 32768, achieving 36.5% overall acceptance and +43% throughput improvement."* / *"Since they have completely different architectures and RoPE strategies, the bug is in vLLM's EAGLE-3 verification/coordination logic, not in the draft models."*

**B4.3 — MTP acceptance collapses beyond the trained context window.**
WHO: vLLM #37435 (OPEN). URL: https://github.com/vllm-project/vllm/issues/37435 · READ BODY
VERBATIM: *"the draft acceptance rate can collapse to ~0% while the target model still appears usable."* / *"beyond original context size, draft starts proposing nonsense / verifier rejects everything"*

**B4.4 — MTP × prefix-caching recurrent-state corruption.**
WHO: vLLM #43559 (OPEN); fix PR #48375 (OPEN since 2026-07-12, `mergeCommitSha: null`). URL: https://github.com/vllm-project/vllm/issues/43559 · https://github.com/vllm-project/vllm/pull/48375 · READ BODY
club-3090 re-verification on v0.29.0, VERBATIM: *"`vllm-pr48375-mamba-drop-eagle-block` (**still REQUIRED** — `drop_eagle_block` is STILL signature-only in v0.29.0's `MambaManager.find_longest_cache_hit`, so vllm#43559 is live)"*

**B4.5 — Unbounded `num_accepted` indices.**
WHO: vLLM PR #50021 (OPEN, `mergeable_state=blocked` since 2026-07-27). URL: https://github.com/vllm-project/vllm/pull/50021 · READ BODY
club-3090, VERBATIM: *"⚠️⚠️ **RE-VERIFIED ON v0.27.1 2026-08-16 — STILL UNBOUNDED, AND THE PATH MOVED. THE OLD CHECK NOW SILENTLY READS AS 'FIXED'.**"*

**B4.6 — DFlash fused-KV corrupts weight-quantized drafters.**
WHO: vLLM #51581 (OPEN). URL: https://github.com/vllm-project/vllm/issues/51581 · READ BODY
VERBATIM: *"Boot-success is NOT sufficient evidence here: a corrupt fused-KV path presents as a merely slow drafter, so **read accept-len**"*

**B4.7 — No supported way to serve an MTP draft layer quantized from a checkpoint.**
WHO: vLLM #49552 (OPEN). URL: https://github.com/vllm-project/vllm/issues/49552 · READ BODY
VERBATIM: *"There is currently no supported way to serve an MTP draft layer quantized from a checkpoint."*

**B4.8 — Adaptive verification is DSpark-only.**
URL: https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/adaptive_verification.md · READ BODY
VERBATIM: *"Adaptive verification needs per-position acceptance estimates, so today it is only supported for DSpark with a confidence head."*
Missing: a per-position acceptance estimator for EAGLE-3 / MTP / draft-model / n-gram.

**B4.9 — Medusa is shipped but undocumented.**
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/README.md · READ BODY
Zero `medusa` hits under `docs/features/speculative_decoding/` despite it being a first-class method; SGLang never mentions it.

**B4.10 — In-code limits on MTP depth and prefix caching.**
- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py — VERBATIM: *"Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate"*
- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next_mtp.py — VERBATIM: *"Qwen3NextMTP currently does not support 'all' prefix caching,"* · READ BODY

**B4.11 — No published head-training cost for Medusa / DFlash2 / DSpark.**
Missing: GPU-hour figures to size a single-GPU head-training plan. (EAGLE-3, DFlash and EAGLE-1 numbers exist — see D.)

### B5. Engine-level method gaps

**B5.1 — Heterogeneous-vocab drafting is greedy-only.**
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/README.md · READ BODY
VERBATIM: *"use_heterogeneous_vocab currently supports greedy draft sampling only. Probabilistic acceptance (temperature > 0 draft sampling) is not yet supported and will be added in a future release."*

**B5.2 — Per-request spec-decode metrics are experimental and n=1 only.**
URL: https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/acceptance_metrics.md · READ BODY
VERBATIM: *"metrics.speculative_decoding is experimental and its shape may change in a future release."* Also they *"are reported only for single-sequence requests and are null for n > 1."*
Note for measurement work: `num_draft_tokens` is defined as *"Total proposed draft tokens, after subtracting drafts invalidated by structured-output constraints."* — i.e. grammar rejections are folded into the denominator silently.

**B5.3 — Grammar/structured-output × spec decode remains a live fault line.**
- SGLang PR #12615 (merged): *"resolves an incompatibility issue that arises when using constrained decoding (specifically with grammars) in conjunction with speculative decoding… prevent overlap scheduling when both constrained decoding (using grammars) and speculative decoding are active"* — https://github.com/sgl-project/sglang/pull/12615 · READ BODY
- SGLang PR #30096 (merged) removed a hard error, VERBATIM: *"DFLASH speculative decoding does not support grammar-constrained decoding yet."* and *"Simply removing the guard would be unsafe — grammar requests would then run through an unconstrained verify and emit malformed output silently."* — https://github.com/sgl-project/sglang/pull/30096 · READ BODY
- vLLM PR #44927 — CLOSED UNMERGED, VERBATIM: *"This means a </think> token emitted as the main token is silently missed: reasoning_ended is never set to True, and grammar constraints (JSON mode, structured output) are permanently disabled for the rest of the request."* — https://github.com/vllm-project/vllm/pull/44927 · READ BODY

**B5.4 — PP + MTP unresolved.**
WHO: vLLM RFC #44697 (OPEN). URL: https://github.com/vllm-project/vllm/issues/44697 · READ BODY
VERBATIM: *"MTP speculative decoding doesn't work under pipeline parallelism (PP > 1). In some configs it crashes; in others it silently diverges from the no-spec greedy baseline."* / *"There are two earlier PP+MTP attempts (#39704, #38104), but both are untested and currently conflicting."*

**B5.5 — Runtime/per-request enable-disable: shipped late and still open in SGLang.**
vLLM #7569, #17984, SGLang #9319 all closed not-planned/stale; SGLang #30263 re-filed, VERBATIM: *"vLLM was asked for runtime API control over SD (#7569) and per-sequence SD control (#17984), both closed as not-planned, and SGLang's own #9319 (dynamic SD, including disable-under-load) was closed by the stale bot."* vLLM later shipped PR #55412 (OPEN as of last read). · https://github.com/sgl-project/sglang/issues/30263 · https://github.com/vllm-project/vllm/pull/55412 · READ BODY

**B5.6 — vLLM #54928 / SGLang #25587 / #35150 / #38009** — see B2.3–B2.5.

---

## C. ATTEMPTED-AND-ABANDONED cells

> Verbatim quotes are mandatory here and are supplied for every row. Where a maintainer comment could not be retrieved (GitHub API exhausted; HTML pages do not embed comment bodies), the row says so explicitly and quotes only what was actually read.

### C0. Structural finding about this section, stated up front

**Every `not planned` closure found in this sweep is a bare state change with zero comments.** Verified by reading each page: vLLM #53215, #46088, #39790, #47825, #52873, SGLang #36001, #4417 — no maintainer prose, no bot message, no closing comment. Search across `is:issue is:closed reason:"not planned"` and `label:wontfix` for spec-decode returned **no maintainer statement using the words "not planned" or "wontfix" anywhere in the accessible record.** The rows below therefore separate *quoted* abandonments from *state-only* abandonments. Quotes that do exist (author self-closures, design rejections, negative results) are reproduced exactly.

### C1. **The two explicit maintainer rejections of adaptive speculation length** (the strongest quoted rejections found)

**vLLM PR #44885 — adaptive verifier step-length (D-Cut-style)** — CLOSED UNMERGED
URL: https://github.com/vllm-project/vllm/pull/44885 · READ BODY
Maintainer **benchislett**, 2026-06-08, VERBATIM:
> **"As discussed here, the surface area of this feature is too significant to justify the marginal and hardware-specific gains #35301"**

**vLLM PR #43522 — dynamic verification for DFlash** — CLOSED UNMERGED
URL: https://github.com/vllm-project/vllm/pull/43522 · READ BODY
Maintainer **benchislett**, 2026-06-22, VERBATIM:
> **"I feel strongly that we must fundamentally avoid synchronization whenever possible. I do not feel comfortable supporting this feature at this time."**

Supporting pattern — a maintainer self-closed their **own** adaptive-spec PR to consolidate:
**vLLM PR #48692**, VERBATIM: *"Closing this PR, we will move forward and try to merge #47808."* (the DSpark confidence-scheduled verification path won over a general adaptive-length feature).

The rejected design's **own paper** states the same integration blocker, VERBATIM (arXiv 2607.14647):
> *"A practical limitation of D-cut lies in its system integration. Its per-step pruning produces a verification batch whose shape varies across steps, which does not fit cleanly into the cpu-gpu overlap (Spec-V2) and the full-plus-piecewise CUDA-graph capture of current inference engines, both of which assume a static per-step shape."*

### C2. Abandonments with **quoted** reasons

**C2.1 vLLM PR #47861 — MTP prefix-cache correctness for hybrid Mamba — closed by the AUTHOR DELETING THE HEAD REPO.**
URL: https://github.com/vllm-project/vllm/pull/47861 · READ BODY
PR body, VERBATIM: *"Avoid applying EAGLE/MTP cache peek-and-drop semantics to Mamba/GDN state cache groups, since recurrent state snapshots cannot be rewound like token KV blocks."*
Maintainer **ivanium**, 2026-08-05, VERBATIM: *"Sorry this went stale on conflicts rather than on review. Thanks for finding the root cause independently."* and *"I didn't carry the supports_eagle_cache_peek half because #46384 has since made the coordinator skip the EAGLE margin for MambaSpec, so that part is covered on main already."*
Outcome: half survived as PR #51113 (MERGED); sibling #45477 CLOSED.

**C2.2 vLLM PR #50808 — FLy speculative verification — CLOSED UNMERGED (lossy by design).**
URL: https://github.com/vllm-project/vllm/pull/50808 · READ BODY
VERBATIM: *"FLy is lossy by design and disabled by default: it does not preserve the target distribution, and nothing changes unless rejection_sample_method=\"fly\" is set."* / *"Standard rejection sampling stops at the first rejected draft token, so a single ambiguous position truncates the whole draft and wastes the remaining tokens the target has already verified."* / *"At such a position FLy defers to the draft token instead of rejecting, which lets the rest of the window be accepted."*

**C2.3 SGLang PR #33986 — sm120 DSpark verify patch WITHDRAWN BY ITS AUTHOR after it corrupted production output.**
URL: https://github.com/sgl-project/sglang/pull/33986 · READ BODY
Author **reSSL**, VERBATIM: *"**Withdrawing this PR.** I deployed this patch to production for about twelve hours and it intermittently corrupted generated text -- stray characters appearing in longer outputs. Rolled back."*
Collaborator **b8zhong**, VERBATIM: *"But we should just support it in Flashinfer. No?"*
Why it was needed, VERBATIM: *"Speculative verify submits batch * (gamma + 1) tokens, which is below 64 by construction, so that check can never pass and the server dies during CUDA graph capture."* — measured on **4× RTX PRO 6000 Blackwell (SM120)**.

**C2.4 SGLang PR #29538 — parallel DSpark verifier closed as superseded.**
URL: https://github.com/sgl-project/sglang/pull/29538 · READ BODY
**hnyls2002**, 2026-07-12, VERBATIM: *"Native DSpark support has now landed on main via #30261 ... so this parallel implementation is superseded and every touched file now conflicts. Closing; follow-up work is being tracked in #30344."*

**C2.5 SGLang PR #10657 — EAGLE-3 for Qwen3-Next, 9 months open, closed as superseded.**
URL: https://github.com/sgl-project/sglang/pull/10657 · READ BODY
VERBATIM: *"! EAGLE3 capture for qwen3-next is now native ( qwen3_next.py ), so I'm closing this as superseded. Please reopen if something's still missing."*

**C2.6 SGLang PR #20370 — quantization repack bug, closed unmerged, root cause never landed.**
URL: https://github.com/sgl-project/sglang/pull/20370 · READ BODY
VERBATIM: *"The crash came from the Marlin repack path during quantized weight processing, in_proj_a/in_proj_b (width 32) were intended to be ignored by quantization (see quant config), but inherited name mapping caused ignore-name mismatch, so these layers were quantized by mistake and hit Marlin."*

**C2.7 SGLang PR #20168 — grammar mask not enforced, author abandoned it.**
URL: https://github.com/sgl-project/sglang/pull/20168 · READ BODY
VERBATIM: *"This PR has been re-implemented in #20989. Please refer to that PR instead."*
Defect, VERBATIM: *"on Ascend NPU, apply_token_bitmask_inplace_triton can fail to enforce grammar masks for XGrammar-constrained decoding: tokens that are masked out (mask_allows=False) may still keep high logits and be selected, causing downstream Tokens not accepted errors in grammar.accept_token"*

**C2.8 vLLM PR #34163 — closed by its own author after 195 days.**
URL: https://github.com/vllm-project/vllm/pull/34163 · READ BODY
VERBATIM: *"Closing — no maintainer activity for 195 days and the repo has moved on. Happy to reopen if the fix is still wanted."* (CLOSED 2026-08-23, `mergeCommitSha: null`)

**C2.9 vLLM PR #40898 — DFlash sliding-window drafter, closed as superseded.**
URL: https://github.com/vllm-project/vllm/pull/40898 · READ BODY
VERBATIM: *"Follow-up to my backport comment above — this PR looks superseded, and the difference in approach is worth recording for anyone who lands here. Hybrid SWA/full DFlash drafters are already on main: #47914 ... plus the follow-up #48113."* (CLOSED 2026-07-14, `mergeCommitSha: null`)
Surviving code still declares limits, VERBATIM: *"sample_from_anchor=True is not supported for DFlash."* / *"%s draft attention (%s) does not support full CUDA graphs; running the draft eagerly."* — https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py · READ BODY

**C2.10 vLLM PR #26504 — per-sequence `DynamicProposer`.**
URL: https://github.com/vllm-project/vllm/pull/26504 · READ BODY
**hmellor**, VERBATIM: *"Closing as stale"* + *"Blocking as there are a lot of basic Python issues in the simple parts of this PR"*

**C2.11 vLLM PR #47012 — CPU dynamic-SD fix, closed by its author.**
URL: https://github.com/vllm-project/vllm/pull/47012 · READ BODY
VERBATIM: *"Closing this PR — the fixes here are no longer needed after #47162"*

**C2.12 vLLM PR #49163 / #49164 — DFlash physical-block shortening, self-closed next day.**
URL: https://github.com/vllm-project/vllm/pull/49163 · https://github.com/vllm-project/vllm/pull/49164 · READ BODY
Author, VERBATIM: *"shortening the physical draft block is not equivalent to computing the full trained block and truncating verification"*

**C2.13 vLLM PR #27434 — closed in favour of a duplicate.**
URL: https://github.com/vllm-project/vllm/pull/27434 · READ BODY
VERBATIM: *"Closing this in favor of it's duplicate #28435 which has a DCO sign-off and smoke tests."*

**C2.14 vLLM PR #48037 — spec-decode OOM profiling, superseded after maintainer doubt.**
URL: https://github.com/vllm-project/vllm/pull/48037 · READ BODY
**mgoin** (vLLM member), VERBATIM: *"This may be too conservative, not sure the 'right' approach here"* then *"Let's go with #48641"*. Shipped alternative: PR #48630.

**C2.15 vLLM PR #33729 — negative accepted-tokens metric crash — MERGED, with a maintainer admission that the metric definition is unresolved.**
URL: https://github.com/vllm-project/vllm/pull/33729 · READ BODY
**njhill**, VERBATIM: *"In some cases it's possible for no tokens to be generated in a step for a request which had spec decode tokens. It's actually not clear cut whether we should record 0 accepted tokens in this case or not record at all, it depends on the specific purpose / definition of acceptance rate."*

**C2.16 llama.cpp PR #28061 — don't re-verify replayed draft tokens — CLOSED UNMERGED.**
URL: https://github.com/ggml-org/llama.cpp/pull/28061 · READ BODY
VERBATIM: *"On backends where logits depend on batch shape or memory layout (Vulkan), the re-verification can reject a token the original verification accepted; the rejection restores the same checkpoint and replays again, and the slot loops on one position without emitting anything."* / *"the replay exists to rebuild state, not to re-decide."* / *"Not verified by me: I have not reproduced the livelock on unmodified master"*

**C2.17 SGLang PR #31214 / #32630 — ROCm verifier gaps.**
- #31214 CLOSED UNMERGED, VERBATIM: *"On ROCm/HIP, EAGLE speculative decoding always commits argmax (greedy) in verify. It ignores temperature and top_p"* / *"_is_hip is in that list because the sampling-verify ops (tree_speculative_sampling_target_only, top_k_renorm_prob, top_p_renorm_prob) are only imported under is_cuda()/is_musa(). So any EAGLE spec-decode run at temperature>0 on ROCm decodes greedy. On long reasoning output that turns into repetition loops. This is not model specific."* — https://github.com/sgl-project/sglang/pull/31214
- #32630 author-marked **"[AMD][Not-Merge]"**, VERBATIM: *"On ROCm both are None, so DSPARK/DFLASH speculative decoding dies with: TypeError: 'NoneType' object is not callable as soon as a request sets top_p < 1 or top_k > 1."* / *"tree_speculative_sampling_target_only has no cheap equivalent and stays None"* — https://github.com/sgl-project/sglang/pull/32630 · READ BODY

**C2.18 HF PR #47961 — return the true proposal q from candidate generators — CLOSED UNMERGED.**
URL: https://github.com/huggingface/transformers/pull/47961 · READ BODY
VERBATIM: *"Three candidate generators broke that contract, so with do_sample=True (and the default top_k=50) assisted decoding is not lossless as documented"* / *"With the mismatch, the acceptance ratio p_i/q_i is inflated by 1/Z (the raw mass kept by top-k/top-p), and on rejection the [p - q]+ residual is built from mass the drafter puts on tokens it can never propose."* Landing fix instead: PR #48007 (MERGED).

### C3. Abandonments that are **state-only** (no closing words exist to quote)

| # | Item | State | URL | Read |
|---|---|---|---|---|
| C3.1 | vLLM #53215 — *"[Performance]: MTP acceptance metrics fluctuate within a benchmark run for GLM-5.2-FP8 on 8×H200 with vLLM v0.24.0"* | Closed as **not planned**, **0 comments**, labels `glm`, `performance`, `quantization`. Reporter framed it as a question: *"We would like to understand whether this is expected because these metrics are calculated over short reporting windows with changing request/batch composition, or whether it indicates an issue with MTP, FP8 model quantization, FP8 KV cache, or scheduling."* Note the bench command uses `--dataset-name random`, which forces acceptance to be meaningless. | https://github.com/vllm-project/vllm/issues/53215 | READ BODY |
| C3.2 | vLLM #46088 — MTP + `--kv-cache-dtype auto` cross-sequence garbage (Gemma-4 W4A16) | Closed as **not planned**, **0 comments**. Reporter, VERBATIM: *"The exact same request that deterministically returns thoughtthought//thought… under --kv-cache-dtype auto returns the correct, coherent answer under --kv-cache-dtype fp8 — same weights, same greedy decode, only the KV dtype changed."* / *"Tokens bleed across sequences — e.g. a trivial request emits another concurrent request's tool-schema tokens (parameters:{properties:{) and the <\|"\|> STRING_DELIM sentinel."* No technical rebuttal recorded on the page. | https://github.com/vllm-project/vllm/issues/46088 | READ BODY |
| C3.3 | vLLM #39790 — spec decode trades TTFT for TPOT | Closed as **not planned** + labels `performance`, `stale`; **0 comments**. The `stale` **label** is the only why-signal. Reporter's un-rebutted numbers: *"Mean TPOT: ~17.11 ms (a 39.3% improvement)"* but *"Mean TTFT: ~158.44 ms (a 114% increase); P99 TTFT: ~1078.03 ms (a 331% increase)"*. | https://github.com/vllm-project/vllm/issues/39790 | READ BODY |
| C3.4 | vLLM #47825 — `mlp_speculator` documented but unrouted in V1 | Closed as **not planned**, no maintainer prose retrieved. VERBATIM body: *"`mlp_speculator` is still exposed as a speculative method ... but **nothing in `vllm/v1/` routes it**, so it crashes at engine init."* / *"`ValueError: Unknown speculative decoding method: mlp_speculator`"*. Filer's ask: *"Happy to send a fix: fail fast at config validation with a clear message (and fix the stale docs/example), or wire an MLP proposer into V1. Could a maintainer confirm which is preferred?"* | https://github.com/vllm-project/vllm/issues/47825 | READ BODY |
| C3.5 | vLLM #52873 — Qwen3-Next GDN + MTP: acceptance permanently dies | Closed as **not planned** — but the **filer closed it themselves** as not-a-defect, which is a real quoted reason. VERBATIM: *"**Update after full investigation — this is checkpoint-specific, not a vLLM defect. Closing.**"* / *"So the trigger is that checkpoint's weights on the MTP-proposer path — not the engine, and not AutoRound-INT4 as a class. The ngram result localizes it to the MTP proposer's own persistent state."* / residual ask: *"One honest note for maintainers: the collapse being permanent, engine-global, and surviving a fresh conversation is notable — if you consider hardening the MTP proposer against a marginal checkpoint worthwhile, I'm happy to share the repro; I'm just not filing that as a bug."* | https://github.com/vllm-project/vllm/issues/52873 | READ BODY |
| C3.6 | SGLang #36001 — DFLASH + `--kv-cache-dtype nvfp4` crash, **filed on this exact hardware + drafter** | Closed as **not planned**, **0 comments**, no labels, no assignee, no linked PR. See C4 below for the full body. | https://github.com/sgl-project/sglang/issues/36001 | READ BODY |
| C3.7 | SGLang #4417 — MTP + CUDA graph stuck at init on **2 H100 nodes** (`--tp 16 --nnodes 2`) | Closed as **not planned**, no prose retrieved. Hangs at *"Capture cuda graph begin."* | https://github.com/sgl-project/sglang/issues/4417 | READ BODY |
| C3.8 | vLLM #9995 — draft-model weights not counted in KV sizing | Closed as **not planned** + `stale`. VERBATIM: *"determine_num_available_blocks only considers the memory usage of the target model"* / *"it might cause cuda OOM"* | https://github.com/vllm-project/vllm/issues/9995 | READ BODY |
| C3.9 | vLLM PR #41823 — dynamic tree pruning | CLOSED as not planned; the paired forum thread (discuss.vllm.ai #2630) got only a bot reply, no maintainer answer | https://github.com/vllm-project/vllm/pull/41823 | READ BODY (state) |
| C3.10 | SGLang #7694 — cannot run grammar + MTP together | **closed as inactive**, never answered. VERBATIM: *"Howerve, it seems that i cannot open both at the same time in MTP mode？"* | https://github.com/sgl-project/sglang/issues/7694 | READ BODY |
| C3.11 | SGLang #20148 — eagle/eagle3 conflicts with xgrammar on NPU | **closed as inactive** 2026-05-09: *"This issue has been automatically closed due to inactivity."* Reporter promised *"I will provide a fixed version shortly."* (2026-03-09) and never did. | https://github.com/sgl-project/sglang/issues/20148 | READ BODY |
| C3.12 | SGLang #9888 — EAGLE3 IMA with FlashInfer | **closed by stale bot** 2025-12-03: *"This issue has been automatically closed due to inactivity."* Fridge003, VERBATIM: *"I guess the bug is related to padding for some uncaptured batch sizes..."* / *"Not sure whether it's caused on sglang side or flashinfer side."* Reporter: *"But isn't this a FlashInfer bug? Since FA3 and Triton backends work fine."* Later fixed by PR #10892 (MERGED). **Note: this was NOT a verification-math bug** — it was a CUDA illegal-memory-access in the EAGLE3 *draft* CUDA-graph runner from padding when `--parallel` is not in `--cuda-graph-bs`. | https://github.com/sgl-project/sglang/issues/9888 | READ BODY |
| C3.13 | vLLM #41559 — DFlash cannot compose with ANY KV quant | **Closed COMPLETED with zero comments.** VERBATIM: *"DFlash spec decode `cannot` compose with any of `fp8_e5m2`, `fp8_e4m3`, or `turboquant_4bit_nc` — it is locked to `bfloat16` KV cache."* / *"The spec decode throughput gains from DFlash do not justify halving the KV pool for long-context workloads."* Crash: `ValueError: KV cache dtype fp8_e5m2 is not supported for non-causal attention` | https://github.com/vllm-project/vllm/issues/41559 | READ BODY |
| C3.14 | vLLM #43996 — KV-cache block-boundary trim bug | OPEN but labelled `stale`, maintainer-authored (NickLucche), never fixed. VERBATIM: *"This is a correctness bug at block boundaries (prompt length is an exact multiple of block_size)."* | https://github.com/vllm-project/vllm/issues/43996 | READ BODY |
| C3.15 | vLLM PR #48204 — per-request proposal lengths reference impl | closed unmerged; closing comment NOT RETRIEVED | https://github.com/vllm-project/vllm/pull/48204 | READ BODY (state) |
| C3.16 | vLLM #17984, RFC #36657 — per-sequence SD control | both closed as not planned | https://github.com/vllm-project/vllm/issues/17984 · https://github.com/vllm-project/vllm/issues/36657 | READ BODY (state) |

### C4. SGLang #36001 in full — the row that matches this rig exactly

**SGLang #36001 — "[Bug] Speculative decoding (DFLASH) crashes with `--kv-cache-dtype nvfp4`"** — CLOSED, `stateReason: NOT_PLANNED`, **0 comments**
URL: https://github.com/sgl-project/sglang/issues/36001 · READ BODY (body); closing comment **does not exist**
Environment, VERBATIM: *"GPU: RTX PRO 6000 Blackwell Workstation Edition, 1x, 96GB"* · *"FlashInfer 0.6.17"* · *"Target checkpoint: `Qwen/Qwen3.8-27B-FP8`"* · *"Draft checkpoint: `incoai/Qwen3.8-27B-DFlash2`"* · *"Image: `lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2` (commit `5f55db3`)"*
Root cause, VERBATIM: *"`forward_batch_info.py`, in the padding branch used when `forward_mode.is_extend()` and `spec_info is not None` (draft-extend / target-verify), only sets `self.extend_num_tokens = num_tokens` — it never sets `self.extend_prefix_lens_cpu`, which stays at its dataclass default of `None`."*
Determinism, VERBATIM: *"So: any speculative-decoding draft-extend/verify step that ends up on the FlashInfer prefill/extend path with an NVFP4 KV cache will hit this deterministically."*
The compounding blocker, VERBATIM: *"There's a second, related gap: `--speculative-draft-attention-backend` is a single value applied to both prefill and decode for the draft... NVFP4 KV cache requires different backends for each... So there's currently no working `--speculative-draft-attention-backend` value for NVFP4 KV cache at all."*
Backend constraint quoted from source, VERBATIM: *"`trtllm_mha` is decode-only for native FP4 KV (`trtllm_mha_backend.py`: "TRTLLM MHA with native FP4 KV cache supports decode only; use a separate prefill backend such as flashinfer or triton."), and flashinfer is the only one that supports NVFP4 prefill/extend — which is exactly the path that hits the bug above."*
Isolation, VERBATIM: *"Dropping only `--kv-cache-dtype nvfp4` (everything else identical) boots and serves correctly, confirming the NVFP4 dequant path is the trigger."*

### C5. Single-rig A/Bs and community negative results (verbatim)

**C5.1 club-3090 — SGLang EAGLE-3 for Qwen3-Next PARKED.**
URL: https://raw.githubusercontent.com/noonghunna/club-3090/refs/tags/v0.10.2/docs/engines/SGLANG.md (branch is `master`; `main` returns HTTP 200 with an **empty body**) · READ BODY
Title VERBATIM: *"# SGLang — Qwen3-Next EAGLE-3 path PARKED; no shipped variant on this stack"*
> *"**Status (2026-05-21): PARKED.** Not currently a shipped variant on this stack for Qwen3-Next. Three independent findings drove the decision:"*
> *"**EAGLE-3 is sub-MTP for Qwen3-Next, even on Blackwell where it works.** Ex0bit's own published numbers on the PRISM-PRO-DQ model card report native MTP = **121 TPS (1.51×)** vs EAGLE-3 chain = **111 TPS (1.39×)**. The model family has a strong built-in MTP head; routing through an external drafter is structurally slower."*
> *"Three patch iterations (pre-import; sys.modules stub at engine init; per-process sys.modules stub at `sglang/__init__.py`) all failed — the walk re-fires during capture regardless of cache state."*
> *"The path is parked because EAGLE-3 < MTP for Qwen3-Next (point 1 above) is a structural finding, not a fixable bug."*
Companion, VERBATIM: *"⚠ PARKED 2026-05-21 — not a recommended path on this stack."* (models/qwen3.6-27b/sglang/README.md)

**C5.2 club-3090 — EAGLE-3 external drafter loses to MTP in a same-session A/B.**
URL: https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md · READ BODY
> *"Same-session **EAGLE3-external A/B** (self-converted `club-b9967` Q4_K_M draft, n=2, only draft+spec-type changed): **47.5 / 61.3 wall, accept 0.408** → −25% narr / −17% code — **MTP stays; no EAGLE3 compose.** Mechanism note: EAGLE3 accepts ~0.40 BOTH hidden-state-fed (vLLM, #662) and token-fed (here) → head-capability-limited, feed-robust; MTP is the reverse (0% hidden-state vs 0.74 token-fed → alignment-limited, capability-strong). A retrained/stronger head flips this — artifact kept on disk for the re-test."*

**C5.3 club-3090 — cross-model DFlash drafter ruled out same-day.**
URL: https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md · READ BODY
> *"**DFlash alternatives ruled out same-day:** stock-vLLM DFlash hard-blocked on hybrid `layer_types` (vllm#40898, `NotImplementedError` at model build); cross-model Anbeeld qwen3.6 DFlash-Q4_K_M drafter on the llama.cpp lane attaches (dims match) but loses to Tess's own MTP GGUF (51.0/64.5 vs 63.2/66.9; accept 52–58% vs 72–85% — **fine-tune shift costs ~20pp**)."*

**C5.4 club-3090 — the n-sweep knee, empirically located then abandoned.**
URL: https://github.com/noonghunna/club-3090/blob/master/docs/engines/VLLM.md and BENCHMARKS.md · READ BODY
> *"n=3 is the empirical sweet spot. n=4 nominally hits higher TPS on code but 4th-position acceptance collapses to ~21%. Don't push higher."*
> *"no-spec 70.1/70.0 → n=3 68.2/84.8 → **n=5 69.9/104.3 (knee: prose break-even, code +49%)** → n=6/8 REGRESS (83.8/84.5, tail accept 0.12–0.36)."*

**C5.5 club-3090 — explicit upstream negative result kept only as a re-test artifact.**
URL: https://github.com/noonghunna/club-3090/blob/master/models/qwen3.6-27b/vllm/patches/vllm-pr40914-k1-only/README.md (upstream PR #40914 STILL OPEN) · READ BODY
> *"## Status: negative result on Qwen3.6-27B"*
> *"With this overlay active, MTP acceptance stabilizes at AL=4.0 / ~100%, but outputs collapse into `!` floods and tool/multi-turn paths time out."*
> *"Skipping `mtp.*` drafter layers does not fix the corruption."*
> *"Keep this only as a re-test artifact."*
> *"Round 3 result: this did not fix the corruption. The target-side path is also wrong for this stack."*

**C5.6 club-3090 — engine consolidation, an engine slug retired.**
URL: https://github.com/noonghunna/club-3090/blob/master/docs/engines/IK_LLAMA.md · READ BODY
> *"`ik-llama/iq4ks-mtp` — the engine's last functional slug — was retired 2026-08-12, following `iq4ks-mtp-vision` and `iq4ks-two-stage` earlier the same day, when the single-card qwen surface consolidated onto vLLM."*

**C5.7 DFlash2 drafter quant — a theoretically tidier calibration measured worse.**
URL: https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16 · READ BODY
> *"A variant whose k/v Hessians also blended the context-KV precompute's input distribution — which is the theoretically tidier calibration — measured 7% **worse** greedy acceptance (3.12 vs 3.34 tokens per step, 118 vs 126 tok/s end to end) and is not what ships here."*

**C5.8 Silent MTP-head skip under quantization.**
URL: https://raw.githubusercontent.com/devnen/qwen3.6-windows-server/v1.3.3/docs/MTP_HEAD.md · READ BODY
> *"The loader silently skips the quantised head, MTP runs, and you get **0 % draft acceptance**, no speedup, no error message."* / *"**If it's near 0.0, your quant's MTP head got silently skipped.**"*

**C5.9 CoreML EAGLE-3 on-device: built, benched, abandoned.**
URL: https://raw.githubusercontent.com/john-rocky/CoreML-LLM/refs/heads/main/docs/EAGLE3_INTEGRATION_STATE.md · READ BODY
> *"Status: **Phase 2A + 2B done, Phase 3 benched, speculative currently slower than baseline.**"* / *"Phase 3 — iPhone 17 Pro bench | ⚠️ ran, **not faster than baseline 28.6 tok/s** (11–17 tok/s with fallback to T=1)"*

**C5.10 SpecJAX — functional but not faster.**
URL: https://raw.githubusercontent.com/tails-mpt/SpecJAX/refs/heads/main/README.md · READ BODY
> *"The sglang-jax EAGLE3 pipeline is functional (correct outputs, ~60–66% acceptance rates) but throughput gains are pending upstream optimization of the verify/tree-building path."*

**C5.11 SpecForge — training works, serving validation still manual.**
URL: https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md · READ BODY
> *"loading the result in a real speculative-decoding server and measuring acceptance remains a GPU-serving validation step."*
URL: https://github.com/sgl-project/SpecForge · READ BODY
> *"Unsupported combinations are rejected during config validation or run assembly instead of falling back to an older trainer."*

**C5.12 Community Blackwell parallelism matrix.**
URL: https://raw.githubusercontent.com/devnen/qwen3.6-windows-server/refs/tags/v1.3.9/docs/SPEC_DECODE_MATRIX.md · READ BODY
> *"Confirmed on a single RTX 5090 (sm_120) on 2026-05-05: TP=1 + MTP works"* / *"TP=2 + MTP | Works after the CPU-relay patch but ~7.5 tok/s, the CPU-relay allreduce dominates per-layer cost. Don't."*

**C5.13 Community sm_120 vs sm_100 kernel gap.**
URL: https://raw.githubusercontent.com/Infatoshi/dsv4-flash-2x-rtxpro6000s/main/patches/README.md · READ BODY
> *"sm_120 (RTX PRO 6000 Blackwell / GeForce Blackwell) is not sm_100 (B200/GB300). Three kernel families this model depends on simply don't exist for it: vendored DeepGEMM (zero sm_120 kernels), cutlass c3x fp8 block scaled-mm (rejects the arch), and TRTLLM MXFP4 MoE (capability-family-100 only)."* / *"The DSpark drafter uses top-k 256; flashinfer's sm_120 decode kernels are only built for top-k {128, 512, 1024} at page 64."* / patch 9: *"Without this, drafts are NaN and acceptance is exactly 0%."*

**C5.14 Community error-hint PR (a guard, not a fix).**
URL: https://github.com/vllm-project/vllm/pull/56204 · TITLE ONLY
Also: https://github.com/vllm-project/vllm/pull/53979 (OPEN), author added a `NotImplementedError` guard, VERBATIM: *"I'd rather block it than ship a mode I haven't put paired greedy numbers behind."* / *"block-diffusion speculative decoding and 4-bit KV are currently mutually exclusive on exactly the memory-bound cards that want both."*

**C5.15 A self-inflicted measurement artifact that mimicked an MTP bug exactly.**
URL: https://github.com/vllm-project/vllm/issues/52469 · READ BODY
Filer's root-cause comment, VERBATIM: *"## Root cause found: test-harness artifact, not a vLLM bug. Closing. After a full audit of every observation in this issue, I can now show the "corruption" was produced by **my own SSE extraction pipeline**, not by vLLM."*
> *"The regex `"content":"[^"]*"` terminates at the first `"` character — **including the `"` of an escaped quote (`\"`) inside the JSON string**."*
> *"**MTP off** → one token per SSE delta... **MTP on** → accepted speculative runs put several tokens in one delta... The regex truncates that delta to `The \` — whole words visibly vanish → "corrupted"."*
> *"So the artifact tracked the MTP toggle exactly, and turning MTP off "fixed" it every time. The intermittent/per-boot pattern I reported was also illusory."*

### C6. Removed-wholesale (features deleted, not deprecated)

| # | What was removed | Verbatim | URL | Read |
|---|---|---|---|---|
| C6.1 | vLLM V0 spec-decode stack deleted | *"This PR removes the spec decoding code in vLLM V0, which is almost superseded by vLLM V1."* (PR #21152, MERGED 2025-07-19) | https://github.com/vllm-project/vllm/pull/21152 | READ BODY |
| C6.2 | `SchedulerConfig.num_lookahead_slots` removed | *"SchedulerConfig.num_lookahead_slots has been replaced by SpeculativeConfig.num_speculative_tokens and isn't used anywhere, so let's remove it."* (PR #29000, MERGED Nov 2025) | https://github.com/vllm-project/vllm/pull/29000 | READ BODY |
| C6.3 | TRT-LLM two-model spec-decoding deployment mode removed | *"**[BREAKING CHANGE] Two-model speculative decoding removed.** The separate draft-engine path behind Eagle3, MTP-Eagle and Draft-Target has been removed; the one-model implementations (draft/drafter as a submodule) are now the only supported paths."* / *"EagleDecodingConfig.greedy_sampling and EagleDecodingConfig.posterior_threshold, which had no effect, are also removed."* (Release 1.2) | https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/release-notes.md | READ BODY |
| C6.4 | SGLang jump-forward decoding removed, never restored | *"Remove jump forward to simplify the code maintenance"* (merrymercy / Lianmin Zheng). zhyncs reply: *"Maybe Jump forward will be implemented using speculative decoding later on by @hnyls2002"* — never happened. Downstream: *"vLLM and SGLang no longer support jump forward decoding (compute_ff_tokens)"* and *"most of the time the benefits are very small, mostly just 1 or 2 occurences over a large schemas"*. Answerer yhay81: *"There is no documented model-quality or regex-correctness failure that caused the removal. ... I could not find a stronger public claim such as 'it was slower' or 'it produced incorrect regex output.'"* Discussion #32352 (Jul 2026) still asks why and is unanswered. | https://github.com/sgl-project/sglang/pull/4032 · https://github.com/sgl-project/sglang/discussions/32352 | READ BODY |
| C6.5 | vLLM legacy verifier path | `vllm/model_executor/layers/rejection_sampler.py` — REMOVED (main → 404; v0.8.5 → 200) | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/rejection_sampler.py | READ BODY |

### C7. Measured negative results for adaptive speculation (all verbatim)

| # | Finding | URL | Read |
|---|---|---|---|
| C7.1 | Every DSD arm *"pays a substantial throughput tax vs no-spec under production defaults"* — measured **−31% at ctx=400**, −21% for static K=3, plus a residual 6% eager-mode penalty even with graphs equalised | https://github.com/vllm-project/vllm/issues/49986 | READ BODY |
| C7.2 | DSD `[[1,4,2],[5,512,0]]` collapsed 8-concurrent throughput *"from ~232 tok/s aggregate ... to 24–157 tok/s"*, i.e. *"~1.5–10× worse than a plain non-spec run would be"* at K=0. H1: *"At K=0 the MTP drafter still runs a full draft-model forward every step."* | https://github.com/vllm-project/vllm/issues/49548 | READ BODY |
| C7.3 | DSD *"incur[s] a 12–25% throughput penalty vs no-spec ... even with an all-K=0 table producing near-zero draft tokens"* | https://github.com/vllm-project/vllm/issues/48494 | READ BODY |
| C7.4 | RFC #48202: *"confidence early stopping did not improve wall clock over fixed full blocks at any threshold we tried"* | https://github.com/vllm-project/vllm/issues/48202 | READ BODY |
| C7.5 | Cascade (2506.20675): *"speculation remains impractical in leading MoEs"*; *"every workload experiences slowdown for at least one K"*; *"current dynamic-K policies are infeasible for MoEs"* | https://arxiv.org/abs/2506.20675 | READ BODY |
| C7.6 | MemSpec (2608.10362): *"Exploration-based adaptation fails to improve throughput under memory constraints."* | https://arxiv.org/abs/2608.10362 | READ BODY |
| C7.7 | Nightjar (2512.22420): *"speculative decoding is not universally beneficial"* | https://arxiv.org/abs/2512.22420 | READ BODY |
| C7.8 | BanditSpec/TapOut: bandits *"do not always exceed the best baseline method"*, upper-bounded by the arm set | https://arxiv.org/abs/2502.08468 | READ BODY |
| C7.9 | EfficientRollout (2606.18967): most drafters were *"slower than No-SD"*; disabling SD for just the early 6–11% of large-batch steps beats always-on SD | https://arxiv.org/abs/2606.18967 | READ BODY |
| C7.10 | **SGLang's own docs concede it often does not pay**: *"If your workload is already stable and one static setting is well tuned, adaptive mode may not help much"* and *"At high batch sizes, narrower ladders (e.g., [1, 2] or [1] ) often outperform wide ones."* | https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding | READ BODY |

### C8. Papers whose conclusion is "this does not help"

| # | Claim, verbatim | URL | Read |
|---|---|---|---|
| C8.1 | *"Despite these advantages, PEFT-BD does not yield a practical speedup in our Qwen3-0.6B experiments."* / *"Thus, the drafter is parameter-efficient but not compute-efficient. Our results isolate a simple but important condition for successful speculative decoding: the drafter must be substantially cheaper to execute than the verifier."* | https://arxiv.org/abs/2607.12422 | READ BODY |
| C8.2 | *"the hybrid drafter increases HF-measured MAT only from 5.01 to 5.04 (+0.6%)… these numbers do not support a meaningful end-to-end speedup claim"* (labelled *"the central negative result of this study"*) / *"KV-Reuse improves long-range acceptance, although end-to-end speedups remain marginal under current training pipelines."* / *"long-range decay persists even in TTT-trained drafters."* | https://arxiv.org/abs/2604.26412 | READ BODY |
| C8.3 | *"Finetuning LLama 2 with multi-token prediction does not significantly improve performance. We tried to finetune LLama 2 with 4-token prediction but this did not yield significant improvements compared to the baseline."* | https://arxiv.org/abs/2404.19737 | READ BODY |
| C8.4 | *"The static method FR-Spec (Static 32k) fails to achieve tangible speedups (1.00×) because the latency reduction in projection is strictly negated by the drop in acceptance rate."* | https://arxiv.org/abs/2605.27390 | READ BODY |
| C8.5 | *"relaxation can require considerable capability evaluation unlike lossless speculative decoding, and many relaxed approaches rely on a drafter that is a good language model, making them unsuited for lightweight dedicated multi-token-prediction drafters."* | https://arxiv.org/abs/2607.08690 | READ BODY |
| C8.6 | *"We attribute this to an engineering limitation: SGLang's current support for the Gated DeltaNet architecture does not yet allow a fully efficient implementation of variable-length verification scheduling…"* | https://arxiv.org/abs/2607.19223 | READ BODY |
| C8.7 | *"the memory benefits from 4-bit weight quantization are diminished by the computational load from speculative decoding. Specifically, verifying a tree-style draft incurs significantly more time overhead than a single-token forward pass on 4-bit weight quantized models."* | https://arxiv.org/abs/2505.22179 | READ BODY |
| C8.8 | *"the acceptance length of even state-of-the-art speculators, like DFlash, EAGLE-3 and PARD degrade with generation length, reaching values close to 1 (i.e. no speedup) within just a few thousand output tokens"* | https://arxiv.org/abs/2605.09329 | READ BODY |
| C8.9 | *"conventional approaches fail to apply both simultaneously due to imbalanced compute requirements (between draft and target models), KV-cache inconsistencies, and communication overheads under small-batch tensor-parallelism."* | https://arxiv.org/abs/2506.11309 | READ BODY |
| C8.10 | *"while three of five configurations decelerate, either because the draft fails to out-speed a small target or because the quantized Metal backend executes 'parallel' verification serially"* / *"The guarantee is free; the speedup is a systems property."* | https://arxiv.org/abs/2607.17283 | READ BODY |
| C8.11 | *"For truncation-based methods, we identify a fundamental pitfall—performance can degrade significantly compared to the true truncation sampling baseline due to distributional distortion."* / *"The degradation is severe enough that typical acceptance falls below even the EAGLE-3 baseline on all four benchmarks and SpecCascade on two... The only thing tree verification buys in return is a slight block-efficiency gain, which does not offset the distortion."* | https://arxiv.org/abs/2607.26627 | READ BODY |
| C8.12 | *"In this work, we firstly present a systematic evaluation of verification strategies across model families, tasks, and sampling regimes, and find that Traversal Verification dominates consistently, with OT-based methods lagging far behind."* / *"Our neural selector allows OT-based methods like SpecInfer to outperform Traversal Verification for the first time"* | https://arxiv.org/abs/2602.16994 | READ BODY |
| C8.13 | *"Dynamic padding approaches reveal critical implementation errors: both DSD (Yan et al., 2025) and Meta's recent work (Tang et al., 2025) incorrectly sample bonus tokens from the draft model's distribution rather than the target model's, violating the fundamental correctness guarantee of speculative decoding."* | https://arxiv.org/abs/2510.22876 | READ BODY |
| C8.14 | *"Together, these observations reveal a central tension for MoE speculative decoding: a larger tree may improve draft coverage, yet simultaneously worsen the very bottleneck that determines end-to-end latency."* / *"These results confirm that adaptive truncation alone is insufficient; explicitly modeling target-side verification cost is crucial for efficient MoE speculative decoding."* | https://arxiv.org/abs/2605.00342 | READ BODY |
| C8.15 | *"this guarantee of semantic equivalence masks a severe operational vulnerability: draft-target alignment can be systematically attacked."* / *"On the GSM8K dataset, our attack increases the mean sample time by 62.3% while preserving the task quality."* | https://arxiv.org/abs/2607.21804 | READ BODY |
| C8.16 | *"Motivated by the curse of multilinguality, we hypothesize that speculative decoding is far less effective for low-resource languages due to the limited multilingual capacities of smaller models."* | https://arxiv.org/abs/2605.30580 | READ BODY |
| C8.17 | Verify cost grows non-linearly then plateaus at a kernel-path switch, VERBATIM numbers: verify cost per round on a 7B q4_k_m target *"grows nearly linearly — 51.0 ms for a 2-token pass, 74.7 ms for 3, 164.9 ms for 5, 218.5 ms for 7 — then drops to a flat 141–142 ms at batches of 9–13, the signature of a kernel-path switch"* | https://arxiv.org/abs/2607.17283 | READ BODY |
| C8.18 | *"UD-Q4_K_XL produces 0% MTP draft acceptance — blk.64.attn_*/ffn_* quantized below the threshold for MTP rejection sampling"* | https://huggingface.co/unsloth/Qwen3.6-27M-MTP-GGUF/discussions/32 → correct URL https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/discussions/32 | TITLE ONLY |

### C9. Multi-GPU draft parallelism — attempted repeatedly, none merged

**SGLang PR #17418 — "Add DP-Draft to run draft model with TP=1"** — OPEN since 2026-01-20
URL: https://github.com/sgl-project/sglang/pull/17418 · READ BODY
Its own benchmark shows both batch decay and a net loss:
> 1.87× @ bs1 → 1.80× @ bs2 → 1.83× @ bs4 → 1.78× @ bs8 → 1.58× @ bs16 → **1.20× @ bs32**; Kimi-K2: 3173 tok/s with draft vs **3466 tok/s without**; DeepSeek-R1 NEXTN 3441 vs 3195 (net win).
VERBATIM: *"Dense models are a better fit: For MoE drafts, EP=1 concentrates all experts on a single GPU and can lead to more fragmented kernel shapes, so throughput may not improve."*

**SGLang PR #23407 — "WIP: Add draft-DP mode"** — OPEN since 2026-04-21, +4.7% only
URL: https://github.com/sgl-project/sglang/pull/23407 · READ BODY
VERBATIM: *"running it with full TP/EP parallelism is wasteful — the communication cost dominates the compute savings."*

**vLLM RFC #44697 — MTP under PP** — OPEN; #39704, #38104 both untested+conflicting
URL: https://github.com/vllm-project/vllm/issues/44697 · READ BODY

---

## D. What is ruled out BY HARDWARE for a single-GPU researcher

Rig: 1× RTX PRO 6000 Blackwell 96 GB, **sm120**, driver 580, single node.

> **sm120 ≠ sm100.** `is_device_capability_family(100)` is FALSE on sm120 (120//10 = 12 ≠ 10). vLLM's `platforms/cuda.py` special-cases family(120) separately.

### D1. Kernel/family gaps on sm120

| # | Ruled out | Verbatim | URL | Read |
|---|---|---|---|---|
| D1.1 | **NVFP4 MoE** (FLASHINFER_CUTLASS / CUTEDSL / TRTLLM_MOE) | Filed on *"NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM12.0)"*: `ValueError: NvFp4 MoE backend 'FLASHINFER_CUTLASS' does not support the deployment configuration since kernel does not support current device.` / *"The NVFP4 MoE backend selection code only checks for SM9.0 (Hopper) and SM10.x family (data center Blackwell B100/B200), but not SM12.0."* Root cause: `is_device_capability_family(100)` | https://github.com/vllm-project/vllm/issues/33416 | READ BODY |
| D1.2 | **MXFP8 MoE native path** | *"This caused MXFP8 MoE to always fall back to MARLIN W8A16 on SM_120/SM_121 hardware… even though SM_12x implements the same tcgen05.mma MX tensor core instructions as SM_10x"* / *"Full enablement requires FlashInfer to ship flashinfer_trtllm_moe compiled for SM_12x targets"* — **PR OPEN since 2026-05-28** | https://github.com/vllm-project/vllm/pull/43911 | READ BODY |
| D1.3 | **NVFP4 KV cache** | *"nvfp4 KV crashes on consumer Blackwell (sm_120/121) on stock vLLM — `--kv-cache-dtype nvfp4` routes to the trtllm-gen FP4 FMHA, which has no build there"*; and SGLang #36001 shows DFLASH+NVFP4-KV has **no working draft attention backend at all** | https://github.com/sgl-project/sglang/issues/36001 · https://github.com/noonghunna/club-3090/blob/master/docs/DTYPE_MATRIX.md | READ BODY |
| D1.4 | **FP8 KV as anything but storage** | *"Neither FA3 nor the trtllm-gen FMHA builds for those arches"*; DFlash2 requires *"bf16 KV + FLASH_ATTN is mandatory"* (fp8 KV forces FlashInfer, ~40% slower) | https://github.com/noonghunna/club-3090/blob/master/docs/DTYPE_MATRIX.md | READ BODY |
| D1.5 | **W8A8 kernels** | reported *"does not build for sm ≥ 10.0"* | https://github.com/noonghunna/club-3090/blob/master/docs/FAQ.md | READ BODY |
| D1.6 | **NVFP4 CUTLASS build for SM120** | *"2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120, 96 GB VRAM each)"*; images *"do not include NVFP4 CUTLASS kernels for SM120"*; `nvfp4_kv_cache_kernels.cu` listed **Skipped** — PR OPEN | https://github.com/vllm-project/vllm/pull/41738 | READ BODY |
| D1.7 | **SM12x enablement for DeepSeek-V4** | 288 commits, unmerged; targets *"users who cannot use SM100-only TMEM / tcgen05 kernels"* | https://github.com/vllm-project/vllm/pull/41834 | READ BODY |
| D1.8 | **trtllm-gen decode kernel gated behind `is_sm100_supported()`** | *"4× NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0 (sm_120), 96 GB each"*; *"the working trtllm-gen decode kernel is gated behind is_sm100_supported(), even though it imports and runs fine on sm_120"* | https://github.com/sgl-project/sglang/issues/36701 | READ BODY |
| D1.9 | **DSpark on consumer Blackwell (SM120)** | *"DSPARK does not start on consumer Blackwell (SM120)."* — measured on *"4x RTX 6000D (SM120), TP4"* | https://github.com/sgl-project/sglang/pull/34717 | READ BODY |
| D1.10 | **Spec V2 (overlap scheduling + spec decoding) in FlashInfer attention** | *"Currently, only xqa MHA can support this for SM120."* / *"DSA (not currently supported)"* | https://github.com/sgl-project/sglang/issues/19637 | READ BODY |
| D1.11 | **DSpark verify on SM120** | *"decode-dsv4 has no topk=192 instantiation, so verify falls through to the prefill kernel's num_tokens > 64 assert"* | https://github.com/sgl-project/sglang/issues/33985 | READ BODY |
| D1.12 | **MXFP8 MoE grouped GEMM** | title asserts SM_120/SM_121 support needed | https://github.com/vllm-project/vllm/pull/43814 | TITLE ONLY |
| D1.13 | **NSA on SM120 declared an architectural dead end** | by an operator who tested 8× RTX PRO 6000 for 7+ days: *"This is not a bug — it is an architectural incompatibility."* | https://github.com/vllm-project/vllm/issues/43235 | READ BODY |
| D1.14 | **DFlash/DSpark + ANY KV quant** | *"DFlash spec decode `cannot` compose with any of `fp8_e5m2`, `fp8_e4m3`, or `turboquant_4bit_nc` — it is locked to `bfloat16` KV cache."* / *"block-diffusion speculative decoding and 4-bit KV are currently mutually exclusive on exactly the memory-bound cards that want both."* | https://github.com/vllm-project/vllm/issues/41559 · https://github.com/vllm-project/vllm/pull/53979 | READ BODY |

### D2. Parallelism / multi-node requirements

| # | Ruled out | Verbatim | URL | Read |
|---|---|---|---|---|
| D2.1 | **DSD + data parallelism** | *"Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. When DP is enabled, vLLM automatically disables `num_speculative_tokens_per_batch_size`"*; source `vllm/config/vllm.py:991` | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md | READ BODY |
| D2.2 | **MTP + PP>1** | *"In some configs it crashes; in others it silently diverges from the no-spec greedy baseline."* | https://github.com/vllm-project/vllm/issues/44697 | READ BODY |
| D2.3 | **spec decode + `--enable-dp-attention` (SGLang)** | NGRAM *"CUDA-only; no `--enable-dp-attention`"*; STANDALONE *"does not support `--enable-dp-attention`"*; DFLASH *"No `--enable-dp-attention`"*; UNO *"requires CUDA with FA3... tensor and pipeline parallel sizes of 1"* | https://docs.sglang.io/docs/advanced_features/speculative_decoding | READ BODY |
| D2.4 | **All draft-DP / draft-TP-1 schemes** | none merged; SGLang #17418 and #23407 both OPEN | https://github.com/sgl-project/sglang/pull/17418 · https://github.com/sgl-project/sglang/pull/23407 | READ BODY |
| D2.5 | **Adaptive verification + PP** | *"Not supported with LoRA... or pipeline parallelism (cost curves and confidences exist only on the last rank)."* | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/adaptive_verification.md | READ BODY |
| D2.6 | **Adaptive verification on sm120 — effectively unavailable** | vLLM AV requires *"FULL varlen decode graphs require AttentionCGSupport.ALWAYS, which the DSV4 sparse-MLA, sparse-SWA, and indexer backends report on SM100. Elsewhere adaptive verification is rejected at startup"*. Its reference measurement is TP=8 on 8×B300. | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/adaptive_verification.md | READ BODY |
| D2.7 | **spec decode + P/D disagg** | multi-node by construction; TRT-LLM #16767 shows accept-length collapse at batch>1; vLLM #43996 correctness bug unresolved+stale | https://github.com/NVIDIA/TensorRT-LLM/issues/16767 · https://github.com/vllm-project/vllm/issues/43996 | READ BODY |
| D2.8 | **SwiftSpec async SD** | *"serves Llama3-70B at 348 tokens/s on 8 Nvidia Hopper GPUs"* | https://arxiv.org/abs/2506.11309 | READ BODY |
| D2.9 | **Open draft co-training (CP + PP transport)** | *"For CP, we extend packed, load-balanced zigzag ring attention... For PP, TapChannel transports intermediate target features across stages"* — needs context-parallel and pipeline-parallel stages | https://arxiv.org/abs/2609.07108 | READ BODY |
| D2.10 | **Cascade / EcoSpec scale** | 235B–671B MoEs; MLSys'26 study used 4 GPUs for 70B and a donated DGX | https://arxiv.org/abs/2506.20675 | READ BODY |
| D2.11 | **Multi-node MTP bring-up** | SGLang #4417 required `--tp 16 --nnodes 2` and hung at *"Capture cuda graph begin."* | https://github.com/sgl-project/sglang/issues/4417 | READ BODY |
| D2.12 | **Batch-invariant validation at scale** | vLLM PR #26609 is titled *"[Feature] Batch Invariant: Deepseek-v3 Batch Invariant on 8xH100"*; #27229 is R1 at TP 8 | https://github.com/vllm-project/vllm/pull/26609 · https://github.com/vllm-project/vllm/issues/27229 | READ BODY |
| D2.13 | **DSpark verify critical path** | *"4xGB300 (attention TP4, MoE TP4 --ep-size 1, DSpark static verify block 5)"* | https://github.com/sgl-project/sglang/pull/39301 | READ BODY |
| D2.14 | **TRTLLM MLA target verification** | *"B300, TP8, full-architecture dummy Kimi K3, concurrency 1, MTP1 (verify M=2), FP8 KV, greedy sampling"* | https://github.com/sgl-project/sglang/issues/39107 | READ BODY |
| D2.15 | **GLM-5.2 NVFP4** | *"It now runs at 500+ tok/s/user on 8x B300, 450 on 4x GB300 (bs=1)"* | https://github.com/sgl-project/sglang/releases | READ BODY |
| D2.16 | **PayPal EAGLE3 study** | *"we benchmark EAGLE3 via vLLM against NVIDIA NIM on identical 2xH100 hardware"* | https://arxiv.org/abs/2604.19767 | READ BODY |

### D3. Training cost (single GPU cannot reach reference scale)

| # | Ruled out | Verbatim | URL | Read |
|---|---|---|---|---|
| D3.1 | **EAGLE-3 head training at reference scale** | *"trainable (within 1-2 days) and testable on **8x RTX 3090 GPUs**. So even the GPU poor can afford it."* | https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B | READ BODY |
| D3.2 | **Drafter training, modern reference scale** | *"Training a single drafter requires **128 GPU-hours** for Llama-3.1-8B-Instruct, **288 GPU-hours** for GPT-OSS-20B, and **320 GPU-hours** for Qwen3-30B-A3B-Instruct. All computations are performed on **NVIDIA H200 GPUs**."* | https://arxiv.org/abs/2605.10453 | READ BODY |
| D3.3 | **NeMo-Automodel DFlash training** | recipe is `torchrun --standalone --nproc_per_node=2` | https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/dflash.mdx | READ BODY |
| D3.4 | **Production EAGLE head training** | *"\| Production \| Llama 3.1 8B Instruct \| **8x A100 80 GB** \| ~2 h (1 epoch, 200k samples) \|"* | https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/eagle.mdx | READ BODY |
| D3.5 | **SpecForge two-GPU recipe** | *"The checked-in Qwen3.6-27B recipe owns a **two-GPU local stack**: one configured GPU runs the target capture server and another runs the trainer."* | https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md | READ BODY |
| D3.6 | **SpecForge serving evals at scale** | *"Qwen3.6-27B on **2 × A100** (TP2…)"; "Qwen3.5-397B-A17B on **8 × B200**"; "Kimi-K3 on **8 × B300**"*; and *"On our **8xH20 testbed**, a topology with **3 SGLang servers and 5 trainer workers** improves end-to-end training throughput by approximately **10%**"* | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | READ BODY |
| D3.7 | **70B EAGLE-3 speculator head training** | *"hardware: **4xA100**"* | https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3 | READ BODY |
| D3.8 | **Megatron ModelOpt speculative export** | `--tp_size 8` | https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/examples/post_training/modelopt/speculative.md | READ BODY |
| D3.9 | **Speculators response-regeneration tutorial** | *"**Time required:** ~10 mins on **2x H100 GPUs** (for 1K samples)"* | https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/response_regeneration.html | READ BODY |
| D3.10 | **Offline drafter data prep: GPU fits, disk does not** | *"Offline \| Only used during data preparation \| **Huge (e.g. ultrachat+sharegpt will need 12TB storage)** \| **as low as 1 GPU**"* | https://raw.githubusercontent.com/sgl-project/SpecForge/9b8873dfb03ed3bc10b105249f8ec6092e01b574/docs/basic_usage/training.md | READ BODY |
| D3.11 | **Qwen3.6 MTP official recipe** | uses `--tensor-parallel-size 8` | https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/raw/main/README.md | READ BODY |
| D3.12 | **DeepSeek-V3 / Kimi-K3 / 671B-class targets** | *"671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) module weights"* | https://huggingface.co/deepseek-ai/DeepSeek-V3 | READ BODY |
| D3.13 | **Meta MTP training compute** | *"In aggregate, training all models reported in the paper required around **500K GPU hours** … A100-80GB and H100."* | https://arxiv.org/abs/2404.19737 | READ BODY |
| D3.14 | **Qwen3-Next-80B MTP finetune** | ran on 4-GPU FSDP | https://huggingface.co/inference-optimization/Qwen3-Next-80B-A3B-Instruct-GSM8K-MTP-finetuned | READ BODY |
| D3.15 | **Draft-DP reference rig** | *"8×B200, DeepSeek-R1 MTP, FP8"* | https://github.com/sgl-project/sglang/pull/17418 | READ BODY |
| D3.16 | **Qwen3-Next EAGLE-3 on 24 GB** | *"Dual 3090 (TP=2) + AutoRound INT4 + EAGLE-3 — boots + serves coherent output"* / *"At 24 GB the target (~17 GB) + EAGLE-3 drafter (~3 GB) + Mamba state + KV cache leaves ~0-2 GB headroom."* — i.e. needed TP=2 | https://raw.githubusercontent.com/noonghunna/club-3090/refs/tags/v0.10.2/docs/engines/SGLANG.md | READ BODY |

### D4. Version traps that bite on a single GPU regardless of hardware

- **MRV2 is default in 0.29.0, but spec-decode features require it while other features force fallback to MRV1.** VERBATIM: *"Some features are not yet supported in MRV2… These include sequence parallelism, dual-batch overlap, elastic expert parallellism, custom logits processors and certain speculative decoding methods. For now, vLLM will still fall back to use MRV1 if any of these features are configured."* / *"we are considering Model Runner V1 deprecated and are targeting v0.32 for its removal. We do not intend to accept any more MRV1-specific improvements or optimizations."* — https://github.com/vllm-project/vllm/releases/tag/v0.29.0 · READ BODY
- **Full CUDA graphs with dynamic SD need MRV2.** VERBATIM: *"Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature"* — https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md · READ BODY
- **Adaptive verification rejects `--enforce-eager` at startup.** VERBATIM: *"Full cudagraphs are required: step costs are profiled from captured graphs, so `--enforce-eager` is rejected at startup."* — same AV doc · READ BODY
- **Adaptive verification + LoRA unsupported.** VERBATIM: *"Not supported with LoRA (the per-token LoRA mapping is built from CPU-side boundaries)"* · READ BODY
- **Draft exceeding the captured graph range asserts at runtime.** VERBATIM: `AssertionError: Shape: 2065 out of considered ranges: [(1, 2048)]` / *"The scheduler schedules up to max-num-batched-tokens tokens (e.g. 2048), but during spec decoding with draft models we pass one extra token per sequence."* — https://github.com/vllm-project/vllm/issues/32591 · READ BODY
- **DSD's own merged benchmark is BS1-only with prefix caching disabled** — every command in PR #45953 passes `--no-enable-prefix-caching` and reports TPOT at base/EAGLE3 only (`2.91 / 2.97 / 2.91` ms). DSD at concurrency on this class is essentially uncharacterized. — https://github.com/vllm-project/vllm/pull/45953 · READ BODY
- **vLLM's own DSD reference is 8×H200** (issue #53215 config uses `--tensor-parallel-size 8`), and Adaptive Verification's reference is TP=8 on 8×B300.

### D5. The single direct single-GPU sm120 datapoint found

| # | Finding | URL | Read |
|---|---|---|---|
| D5.1 | MTP n-sweep on a Blackwell single card: *"MTP n=3 → 100 tok/s (27B) / 170 tok/s (35B MoE); MTP **n=5 → 125 tok/s, annotated '+4% over n=3, marginal'**"*, per-position acceptance 0.87/0.72/0.60 | https://github.com/lastloop-ai/vllm-blackwell-guide | READ BODY (community, non-primary) |
| D5.2 | MTP + FP8 KV + chunked prefill bug was **filed and root-caused on exactly "1× NVIDIA RTX PRO 6000 Blackwell (sm_120)"** — so sm120 spec-decode work IS possible; it is the FP4/FP8 *datacenter* kernels that are gated out | https://github.com/vllm-project/vllm/issues/55894 | READ BODY |
| D5.3 | Batch-invariance validation has been reported on RTX 4090 (SM89), RTX 4080 (SM8.9), RTX 3060 (SM86), 4×A10G TP4 — and PR #52522's E2E test states *"The E2E test requires one CUDA GPU with at least 32 GB of memory"* | https://github.com/vllm-project/vllm/pull/52522 | READ BODY |

---

## E. Systems-integration interactions (structured output, TP/EP, paged attention, CUDA graphs, chunked prefill, prefix caching)

> ⚠️ **Provenance caveat for this section.** These rows came from a sub-sweep whose sandbox had `github.com` HTML blocked and `api.github.com` at 0/60. It retrieved issue/PR **bodies** via the GitHub *search* API (which returns body text) but could **not** retrieve **comment threads**. So **no maintainer comment is quoted here**; closure *reasons* for several `not_planned` issues are marked NOT VERIFIED. The vLLM source-file and docs quotes are the strongest part and are primary-source. Rows the lead agent independently verified are marked `[v]`.

### E1. 🔴 The strongest single new finding: rejection sampling is **provably not lossless under a grammar mask**

**arXiv 2605.07698 — "Future Validity is the Missing Statistic: From Impossibility to Φ-Estimation for Grammar-Faithful Speculative Decoding"** (Nie et al., 2026-05-08)
URL: https://arxiv.org/abs/2605.07698 · READ BODY
VERBATIM:
> *"any speculative decoder with local mask access, Leviathan rejection, and rollback soundness samples from the locally projected distribution mu^proj rather than the grammar-conditional distribution mu^star. This extends the GAD impossibility result to speculative decoding; on Dyck grammars with Qwen3-8B, the total-variation gap can reach 0.996."*
Scope caveat, VERBATIM: *"All fidelity claims are scoped to enumerable grammars and token tries."*

**This contradicts vLLM's own docs.** vLLM's "Lossless guarantees of Speculative Decoding" section claims the implementation is *"algorithmically validated to be lossless"* — but that claim is about the **unconstrained** rejection sampler. Under a grammar mask the guarantee does not hold in the grammar-conditional sense.
URL: https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/README.md · READ BODY `[v]`

### E2. Structured output × spec decode

| # | Item | Verbatim / finding | URL | Status | Read |
|---|---|---|---|---|---|
| E2.1 | Historically documented incompatible | PR body: *"Update the feature compatibility matrix to reflect that speculative decoding and structured output do not currently work together."* | https://github.com/vllm-project/vllm/pull/12373 | merged 2025-01-24 | READ BODY |
| E2.2 | Then made to work | *"The generated speculative draft tokens are validated according to the grammar to ensure they match the grammar constraints. This operation should not modify the matcher state."* | https://github.com/vllm-project/vllm/pull/14702 | merged 2025-04-30 | READ BODY |
| E2.3 | **One backend still refuses outright** | Source, VERBATIM: *"LM Format Enforcer backend does not support speculative tokens"* | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/structured_output/backend_lm_format_enforcer.py | shipped (hard incompatibility) | READ BODY `[v]` |
| E2.4 | **The mask can fail OPEN** | VERBATIM: *"which then writes `_full_mask` — every token in the vocabulary allowed — at rows that are real sampling positions."* / *"The failure is silent by construction: the mask fails open."* | https://github.com/vllm-project/vllm/issues/54437 | **OPEN** | READ BODY |
| E2.5 | Grammar-aware draft sampling proposed | VERBATIM: *"These tokens are guaranteed to be rejected by the verifier regardless of probability alignment, wasting entire speculative rounds."* | https://github.com/vllm-project/vllm/pull/47885 | OPEN | READ BODY |
| E2.6 | SGLang Outlines backend incompatible with ANY spec algorithm | *"OutlinesGrammar does not implement rollback(), which the speculative DFS verifier requires"*; *"Outlines uses a dense torch.bool mask, but spec_utils.traverse_tree() expects packed int32 bitmasks"*. Alternative #24889 **also closed-unmerged (2026-09-13)**. | https://github.com/sgl-project/sglang/pull/24499 | CLOSED UNMERGED | READ BODY |
| E2.7 | xgrammar rollback is O(N²) under spec decode | — | https://github.com/sgl-project/sglang/issues/38863 | OPEN | READ BODY |
| E2.8 | Merged fixes | #12615 (spec-v2 overlap-scheduling guard), #30096 (removed DFLASH-grammar 400) — both independently verified `[v]` | https://github.com/sgl-project/sglang/pull/12615 · https://github.com/sgl-project/sglang/pull/30096 | merged | READ BODY `[v]` |
| E2.9 | Closed-unmerged structured-output PRs | #44292, #48516, #44927, #49738 | https://github.com/vllm-project/vllm/pull/44292 · https://github.com/vllm-project/vllm/pull/48516 · https://github.com/vllm-project/vllm/pull/44927 · https://github.com/vllm-project/vllm/pull/49738 | closed unmerged | READ BODY |
| E2.10 | Still-open structured-output × spec issues | #49694, #52620, #49210, #47025, #40875, #52452, #52477, #43424, #40962 | https://github.com/vllm-project/vllm/issues/49694 | OPEN | READ BODY |

### E3. TP / EP / DP-attention

| # | Item | Verbatim | URL | Status | Read |
|---|---|---|---|---|---|
| E3.1 | Canonical "not combinable" statement | *"conventional approaches fail to apply both simultaneously due to imbalanced compute requirements (between draft and target models), KV-cache inconsistencies, and communication overheads under small-batch tensor-parallelism."* | https://arxiv.org/abs/2506.11309 | published | READ BODY `[v]` |
| E3.2 | **Elastic EP + draft model refused in source** | *"Elastic EP is not supported with draft model."* | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu/eplb_utils.py | shipped | READ BODY |
| E3.3 | TP comm cost of drafting, quantified in source | `use_local_argmax_reduction` *"Reduces communication from O(vocab_size) to O(2 * tp_size) per token."* | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu/spec_decode/ | shipped | READ BODY |
| E3.4 | **SGLang source rejections, verbatim** | *"Currently DFLASH speculative decoding does not support dp attention on non-NPU devices."* / *"Currently standalone speculative decoding does not support dp attention."* / *"Currently ngram speculative decoding does not support dp attention."* | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/speculative_hook.py | shipped | READ BODY |
| E3.5 | Vendor cards concede experimental scale | Kimi-K3: *"spec × EP × DP-attention is validated only at 8-GPU EP8 × DP2 (full GSM8K) — experimental at these scales."* DeepSeek-V4: *"they use DP Attention, which DSpark is incompatible with on current releases."* | https://github.com/sgl-project/sglang | shipped (docs) | READ BODY |
| E3.6 | **EAGLE3 TP=2 + draft TP=2 NCCL deadlock** | — | https://github.com/vllm-project/vllm/issues/44063 | OPEN | READ BODY |
| E3.7 | DP>1 + EP + DSpark assertion | `AssertionError` in `DPMetadata.make` | https://github.com/vllm-project/vllm/issues/56281 | OPEN | READ BODY |
| E3.8 | Draft/target attention backends can have **disjoint KV cache layouts** | — | https://github.com/vllm-project/vllm/issues/55312 | OPEN | READ BODY |
| E3.9 | SGLang PRs adding dp-attention to spec decode | #35898, #35321, #29506, #28616, #30916, #30642 | https://github.com/sgl-project/sglang/pull/35898 | OPEN | READ BODY |

### E4. Paged-attention backends

| # | Item | Verbatim | URL | Status | Read |
|---|---|---|---|---|---|
| E4.1 | **FlashInfer silently downgrades CUDA graphs, measured cost** | FlashInfer + spec decode *"silently downgrades cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE"*, and the reporter *"measured 47.5 → 55.2 tok/s (+16%) switching only `--attention-backend FLASHINFER` → `FLASH_ATTN`"* | https://github.com/vllm-project/vllm/issues/49547 | **OPEN** | READ BODY |
| E4.2 | **Cutlass-MLA produces incorrect output with spec decode** | *"The Cutlass-MLA backend produces incorrect output when using speculative decoding."* | https://github.com/vllm-project/vllm/pull/25984 | merged (Known issues) | READ BODY |
| E4.3 | AITER MLA cannot be used with Eagle3 | *"currently cannot be used with speculative decoding methods like Eagle3 due to a kernel_block_size conflict"* | https://github.com/vllm-project/vllm/pull/39616 | merged | READ BODY |
| E4.4 | MLA decode-only CUDAGraph capture | *"MLA only supports decode-only full CUDAGraph capture."* | https://github.com/vllm-project/vllm/issues/21505 | — | READ BODY |
| E4.5 | **SGLang source rejections, verbatim** | *"topk > 1 + page_size > 1 needs the two-pass cascade draft-decode ... flashmla / trtllm_mla can't express the per-branch tree, so reject."* / *"speculative_eagle_topk > 1 with page_size > 1 is only supported on ('flashinfer', 'fa3', 'triton')"* / *"Speculative decoding is currently not supported with Flex Attention backend"* | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/server_args.py | shipped | READ BODY |
| E4.6 | Samplers assume homogeneous batch behaviour | *"assume homogeneous sampling behavior across a batch, limiting support for dynamic serving workloads and preventing efficient CUDA Graph execution."* | https://arxiv.org/abs/2607.20475 | published | READ BODY |
| E4.7 | BF16-Q + fp8 KV + spec decode + SWA silent corruption on SM100 | — | https://github.com/vllm-project/vllm/issues/47899 · https://github.com/vllm-project/vllm/issues/47847 | OPEN | READ BODY |

### E5. **CUDA graphs are explicitly keyed on K** (answers the "graphs keyed on K" lead)

| # | Item | Verbatim | URL | Read |
|---|---|---|---|---|
| E5.1 | Spec decode is a *uniform decode batch* of width 1+K | *"speculative decode (max_query_len = 1+num_spec_tokens) as uniform decode batches"*; `AttentionCGSupport.UNIFORM_BATCH` docstring: *"this can be used for spec-decode i.e. 'decodes' are 1 + num_speculative_tokens"* | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/cuda_graphs.md | READ BODY `[v]` |
| E5.2 | **Backend support table — most backends cannot do uniform-batch graphs** | FlashInfer / AITER MLA / CUTLASS MLA / Mamba = `UNIFORM_SINGLE_TOKEN_DECODE`; *"only FlashAttention 3 supports it currently"*; and critically **"Unlisted backends are all declared as NEVER."** | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/cuda_graphs.md | READ BODY |
| E5.3 | Capture sizes are rounded to multiples of K+1 | `adjust_cudagraph_sizes_for_spec_decode` rounds every capture size to a multiple of *"num_speculative_tokens + 1"*; guard *"CUDAGraphMode.{name} is not supported with spec-decode for attention backend {min_cg_attn_backend}"* → falls back to PIECEWISE or NONE | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/compilation.py | READ BODY |
| E5.4 | **EAGLE is piecewise-graph only, by admission** | *"Eagle currently only supports PIECEWISE cudagraphs. ... NOTE(lucas): this is a hack, need to clean up."* | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_model_runner.py | READ BODY |
| E5.5 | Closed/merged/open graph PRs | CLOSED/UNMERGED #34880, #26937, #47513, #7090 · MERGED #28479, #54418, #46849 · OPEN #48494, #53520, #43924 (WIP cuda-graph rejection sampling), #43776, #43716, #55593 | https://github.com/vllm-project/vllm/pull/34880 | mixed | READ BODY |
| E5.6 | Closed NOT-PLANNED (reason NOT VERIFIED) | #49488 (persistent in-place drafting metadata), #27325 (*"the adaptation of cuda_graph_sizes causes the decoding process to fall back to eager"*) | https://github.com/vllm-project/vllm/issues/27325 | closed not planned | READ BODY (state) |

### E6. Chunked prefill — SGLang hard-disables, vLLM enables

| # | Item | Verbatim | URL | Read |
|---|---|---|---|---|
| E6.1 | **SGLang warning strings, verbatim** | *"Mixed chunked prefill is disabled: %s speculative decoding does not support it."* / *"Mixed chunked prefill is disabled because of using Frozen-KV MTP speculative decoding."* / *"Mixed chunked prefill is disabled for UNO speculative decoding."*; `supports_mixed_chunk()` docstring: *"ngram cannot join as is: its overlap relay skips output_tokens_buf, which the mixed input resolve reads."* — NGRAM unconditionally sets `enable_mixed_chunk=False` | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/server_args.py | READ BODY |
| E6.2 | SGLang v0.5.6 blanket statement | *"**Pipeline parallelism is not compatible with overlap schedule, speculative decoding, mixed chunked prefill.**"* (strings verified present in v0.4.9, v0.4.10.post2, v0.5.1–v0.5.7) | https://github.com/sgl-project/sglang | READ BODY |
| E6.3 | vLLM went the other way | PR #9291 implemented issue #5016 | https://github.com/vllm-project/vllm/pull/9291 | merged 2024-11-07 | READ BODY |
| E6.4 | **Mixed spec/plain decode batches read stale recurrent state** | *"generation continues from a state that silently lost a few tokens of history"* | https://github.com/vllm-project/vllm/issues/56531 | OPEN | READ BODY |
| E6.5 | MTP + chunked prefill + prompt_logprobs wrong | *"wildly wrong for a subset of requests"* | https://github.com/vllm-project/vllm/issues/53488 | OPEN | READ BODY |

### E7. Prefix caching — the structural conflict, in source and quantified

| # | Item | Verbatim | URL | Read |
|---|---|---|---|---|
| E7.1 | **The mechanism, in vLLM source** | *"If eagle is enabled, drop the last matched block to force recompute the last block to get the required hidden states for eagle drafting head."* / *"Eagle needs the tokens right before the generation point recomputed: drop one hash unit when fine-grained ..., else one cache block."* | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/single_type_kv_cache_manager.py | READ BODY |
| E7.2 | The opt-out flag, in config | *"Disable dropping the trailing prefix-cache block for EAGLE-like speculative methods. This is an experimental option for measuring the acceptance-rate impact of reusing that block."* | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py | READ BODY |
| E7.3 | Maintainer position: correct and intentional | *"That is correct and intentional — see the discussion in #38182 and the design doc attached to #17137."* / *"The drop itself is required for correctness… until that lands, the behaviour is here to stay and the only gap is that it is invisible. This is deliberately a UX-only change: no scheduling, allocation, or cache behaviour is touched."* | https://github.com/vllm-project/vllm/pull/51769 | READ BODY `[v]` |
| E7.4 | **Measured cost** | *"EAGLE/MTP prefix-cache last-block drop causes a **1,648-token recompute per hit** on one hybrid Qwen3.8 GDN layout — ~**30-40% batch throughput loss** on prefix-reusing workloads with speculative decoding enabled"* | https://github.com/vllm-project/vllm/issues/53670 | **OPEN** | READ BODY |
| E7.5 | **MTP can score literally zero prefix-cache hits** | with MTP *"the run scores 0 prefix-cache hits"*; *"Adding `\"disable_eagle_block_drop\": true` restores **22,176 hits out of 72,992 queried**"* | https://github.com/vllm-project/vllm/pull/55519 | OPEN | READ BODY |
| E7.6 | Quantified hit-rate drop | *"One Qwen3.6 report on vLLM 0.20.1 measured the hit rate dropping from 87.3% to 69.9%, with 1,056 fewer hit tokens per request—approximately one 1,072-token aligned block."* | https://github.com/vllm-project/vllm/issues/50438 | OPEN (RFC) | READ BODY |
| E7.7 | User-visible form of the same bug | *"Based on Qwen3.5-35B-A3B, why does enabling MTP speculative decoding actually reduce the prefix cache hit rate? ... Prefix cache hit rate: around 92% ... with `--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'`: Prefix cache hit rate: around 71%"* | https://github.com/vllm-project/vllm/issues/38182 | — | READ BODY |
| E7.8 | SGLang equivalent | *"EAGLE speculative decoding defeats radix prefix reuse for multi-turn traffic... no crash, silent 97%→40-53% reuse collapse"* — assigned to hzh0425 | https://github.com/sgl-project/sglang/issues/32459 | OPEN | READ BODY |
| E7.9 | Other source limits | *"Hybrid KV cache is not supported for eagle + chunked local attention."*; in `qwen3_next_mtp.py` / `qwen3_5_mtp.py` / `qwen4_exp` MTP: *"currently does not support 'all' prefix caching"* | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next_mtp.py | shipped | READ BODY `[v]` |
| E7.10 | **The general fix is admitted not to exist** | PR #33524's author, VERBATIM: *"for more complicated models with multiple attention groups, this PR does not fully address the EAGLE spiral block drop issue either. A general fix ... cannot directly cache the hit_blocks list returned by each attention type, because SWA attn and Mamba-style attn do not follow the downward-closed property"*; *"Fortunately, we don't have such complex models yet, so this is not a huge issue for now."* | https://github.com/vllm-project/vllm/issues/51771 | OPEN | READ BODY |
| E7.11 | Closed NOT_PLANNED, quoting upstream PR authors | PR #30877: *"Speculative decoding is temporarily disabled in this PR as there are still corner-case bugs when using with prefix-caching in align mode."*; PR #33726: *"Prefix Caching (old style, have not tested 'align' mode)"* | https://github.com/vllm-project/vllm/issues/41884 | closed not planned | READ BODY |
| E7.12 | SGLang PD-disagg decode radix + spec | *"Incompatible with --enable-hisparse, speculative decoding, and --disaggregation-transfer-backend fake."*; *"HiCache does not support Inkling MTP draft state yet."* | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/server_args.py | shipped | READ BODY |

### E8. 🔴 Documentation gap (material to any "what does the engine actually guarantee" question)

Grep-verified negative facts:
1. **vLLM's "Known Feature Incompatibility" section contains EXACTLY two items** — and mentions **none** of structured output, TP/EP, paged attention, CUDA graphs, chunked prefill, or prefix caching. VERBATIM: *"## Known Feature Incompatibility / 1. Pipeline parallelism is not composable with speculative decoding as of `vllm<=0.15.0` / 2. Speculative decoding with draft models is not supported in `vllm<=0.10.0`"* — https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/README.md · READ BODY `[v]`
2. **vLLM's APC design doc (`docs/design/prefix_caching.md`) contains ZERO occurrences of "EAGLE" / "speculative" / "spec"** — verified by grep.
3. Separately: `docs/design/cuda_graphs.md` **does** carry the K-keyed graph contract (E5.1/E5.2), so the CUDA-graph interaction is the one that *is* documented.

**Consequence:** every incompatibility in E2–E7 is undocumented in the official matrices. They are discoverable only by reading source or issue trackers.

---

## F. Draft-model drafting & training — additional rows, incl. the Qwen3-0.6B→Qwen3-4B answer

### F1. The exact "Qwen3-4B + small drafter" question was CLOSED, REJECTED, then RE-CLOSED, then MERGED

| # | Item | Verbatim / finding | URL | Status | Read |
|---|---|---|---|---|---|
| F1.1 | Standalone draft models were **removed** from vLLM | RFC #18571 (WoosukKwon) lists *"Draft model-based speculative decoding"* under the heading **"Features Temporarily Discontinued"** | https://github.com/vllm-project/vllm/issues/18571 | closed `stale` | READ BODY |
| F1.2 | Docs still carry the scar | *"Speculative decoding with draft models is not supported in `vllm<=0.10.0`"* | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/README.md | shipped (docs) | READ BODY `[v]` |
| F1.3 | **A user filed literally this pairing and was refused** | Issue title config: `vllm serve /Qwen/Qwen3-4B ... --speculative_config '{"model": "/Qwen/Qwen3-0___6B", "num_speculative_tokens": 5}'` → `NotImplementedError: Speculative decoding with draft model is not supported yet. Please consider using other speculative decoding methods such as ngram, medusa, eagle, or mtp.` (vLLM 0.13.0, RTX 4070) | https://github.com/vllm-project/vllm/issues/31883 | **closed `not planned`** + stale-closed 2026-05-30 | READ BODY |
| F1.4 | …and it was merged **anyway**, one month after the refusal | tomasruizt on #31883, 2026-01-29: *"This feature has been merged in https://github.com/vllm-project/vllm/pull/24322"* — then the stale bot closed the issue on 2026-05-30 with *"This issue has been automatically closed due to inactivity."* | https://github.com/vllm-project/vllm/pull/24322 | **MERGED** | READ BODY `[v]` |
| F1.5 | **The merging PR's benchmark rig was an RTX PRO 6000 96 GB — this rig's GPU class** | PR #24322 "feat: spec decode with draft models" (tomasruizt), RFC #18571 referenced by the same author in the paper's acknowledgements | https://github.com/vllm-project/vllm/pull/24322 | merged 2026-01-19 | READ BODY `[v]` |
| F1.6 | Single-GPU Qwen3-4B + 0.6B command, official | `--speculative-config '{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":12,"method":"draft_model","parallel_drafting":true}'` (PARD) | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/parallel_draft_model.md | shipped | READ BODY `[v]` |
| F1.7 | vLLM's own docs example uses **`Qwen/Qwen3-4B-Thinking-2507` + `Qwen/Qwen3-0.6B`** | `vllm serve Qwen/Qwen3-4B-Thinking-2507 ... --speculative-config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5, "method": "draft_model"}'` | https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/features/speculative_decoding/draft_model.md | shipped (0.29.0) | READ BODY `[v]` |

### F2. Vocabulary mismatch

| # | Item | URL | Status | Read |
|---|---|---|---|---|
| F2.1 | vLLM `universal_draft` / TLI merged 2026-07-02 | https://github.com/vllm-project/vllm/pull/38174 | merged | READ BODY |
| F2.2 | HF Transformers equivalent merged | https://github.com/huggingface/transformers/pull/35029 | merged | READ BODY |
| F2.3 | **SGLang TLI still open** | https://github.com/sgl-project/sglang/pull/22883 | **OPEN** | READ BODY |
| F2.4 | **SGLang silent corruption under vocab remap, fix PR sat unmerged ~2 months** | SGLang bug #24051 — *"every accepted draft token is silently corrupted"* — **auto-closed by the inactivity bot** while fix PR #23838 stayed unmerged; author: *"Could a maintainer please merge this when convenient?"* | https://github.com/sgl-project/sglang/issues/24051 · https://github.com/sgl-project/sglang/pull/23838 | closed (bot) / OPEN | READ BODY |

### F3. "The drafter is not the bottleneck" — the strongest repeated negative result

| # | Finding, verbatim | URL | Read |
|---|---|---|---|
| F3.1 | A **logits-distilled** draft with *"acceptance rate is very high"* still gave *"consistent performance drop ~30%"* | https://github.com/vllm-project/vllm/issues/15025 | READ BODY |
| F3.2 | *"MTP is ~21% slower than generating without speculation, **despite 100% draft accuracy**"* | https://github.com/ggml-org/llama.cpp/issues/23533 | READ BODY |
| F3.3 | Gemma4 MTP merge: 40 tok/s → ~4 tok/s | https://github.com/ggml-org/llama.cpp/issues/24266 | READ BODY |
| F3.4 | *"the drafter is parameter-efficient but not compute-efficient… Longer accepted prefixes alone cannot compensate when draft computation remains verifier-scale"* | https://arxiv.org/abs/2607.12422 | READ BODY `[v]` |

### F4. Draft-vocab pruning (FR-Spec): rejected on measurement in vLLM, still open in llama.cpp

**vLLM — CLOSED UNMERGED / NOT PLANNED, with a measurement-based rejection.** Reviewer, VERBATIM:
> *"the drafter forward pass is not nearly a huge bottleneck, so we don't expect fr-spec to speed things up that much"* — measured **1.52%** throughput effect.
Two PRs closed unmerged (#24343, #29334); issue #24506 not-planned. URL: https://github.com/vllm-project/vllm/pull/24343 · READ BODY

**llama.cpp — the same idea is an OPEN research issue with a large measured win.** Trimming the draft LM head to 32,768 tokens cut that kernel **−84.9%** (621 µs → 91 µs/call). URL: https://github.com/ggml-org/llama.cpp/issues/25187 · READ BODY
This is a clean case of the *same* optimization being rejected in one engine on measurement and actively pursued in another.

### F5. Online / on-the-fly draft training — the least reproducible area

| # | Finding, verbatim | URL | Read |
|---|---|---|---|
| F5.1 | *"98% acceptance rate during training… but the average acceptance rate was only 10%"* | https://github.com/sgl-project/SpecForge/issues/533 | READ BODY |
| F5.2 | Official 4.72× vs user 2.86× after 142k steps | https://github.com/sgl-project/SpecForge/issues/469 | READ BODY |
| F5.3 | Online training OOM | https://github.com/sgl-project/SpecForge/issues/521 | READ BODY |
| F5.4 | **Disaggregated training ~3.2× SLOWER than DP** — versus the vendor blog's *"+10%"* claim | https://github.com/sgl-project/SpecForge/issues/718 · https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | READ BODY |

### F6. Training data curation — a clear published answer

SpecForge v0.3, VERBATIM:
> *"regenerating dataset responses with the target model — greedily, in the reasoning mode that will be served — has been the single largest lever on final acceptance. We recommend target-generated data for every strategy."*
URL: https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ · READ BODY

### F7. Drafter training cost floor (relevant to D3 — some of it is cheap)

| # | Finding, verbatim | URL | Read |
|---|---|---|---|
| F7.1 | *"~2.5 h on a single 24 GB GPU, ~$3 of compute"* for a usable 0.5B drafter | https://huggingface.co/vexp-ai/horizon-draft-0.5b | READ BODY |
| F7.2 | Speculators CLI explicitly supports single-GPU training | https://raw.githubusercontent.com/vllm-project/speculators/main/docs/cli/train.md | READ BODY |
| F7.3 | Countervailing practitioner view: *"training will cost you couple of hundreds with uncertain benefits"* | https://forums.developer.nvidia.com/t/training-a-personal-draft-model-for-qwen/378611 | READ BODY |

This partially offsets D3: the *reference* numbers (128–320 H200-GPU-hours) are for frontier targets, but a usable small drafter is a single-24 GB-GPU job.

### F8. Additional quantized-head failures and abandonments

| # | Item | Verbatim / finding | URL | Status | Read |
|---|---|---|---|---|---|
| F8.1 | **Structural cause of the whole class** | *"The MTP head is **not part of the HF model graph** that the quantizer traces, so it is neither quantized nor listed in `ignore`. Its weights in the checkpoint are plain bf16 (typically re-attached from the base model after quantization)."* | https://github.com/sgl-project/sglang/issues/38574 | OPEN | READ BODY |
| F8.2 | NVFP4 draft → **0% acceptance**, fixed by disabling CUDA graphs | 0% default / 77.5% cudagraph-off / 92.3% eager | https://github.com/vllm-project/vllm/issues/54997 | closed | READ BODY |
| F8.3 | **Rejection with NO comment at all** | SGLang PR #38366 (EAGLE/NextN vs an NVFP4 target embedding) — closed unmerged by maintainer @Fridge003 **~2 h after opening, zero comments**. Body: *"EAGLE / NextN self-draft speculative decoding cannot start on a ModelOpt MIXED_PRECISION checkpoint whose token embedding is NVFP4, although the same checkpoint serves correctly without speculation."* | https://github.com/sgl-project/sglang/pull/38366 | **CLOSED UNMERGED, no comment** | READ BODY |
| F8.4 | llama.cpp MTP collapses at KV-slot boundaries — closed on a one-line template reply | Maintainer @am17an: *"Follow the issue template"*. Reporter's last words: *"KV slot boundary root cause (collapse at specific ctx-size values) remains open."* Data: a 256-token ctx delta flips MTP from 2.1× (AR 78%) to 1.04× (AR 0.6%). | https://github.com/ggml-org/llama.cpp/issues/23636 | **closed `not_planned`** | READ BODY |
| F8.5 | TRT-LLM: quantizing a model while keeping its EAGLE drafter is **officially not supported** | User: *"So to me it seems that there is currently no working solution for quantizing the model with an EAGLE drafter."* NVIDIA @hchings: *"the [Support Matrix] of the Eagle example README suggests that FP8/INT8 are not supported."* | https://github.com/NVIDIA/TensorRT-LLM/issues/3207 | closed | READ BODY |
| F8.6 | vLLM #41559 has a maintainer closing comment after all | @benchislett, 2026-08-11: *"Closing for now. We have support via FLASHINFER (CUTLASS) backend. If specific feature support is desired, please create fine-grained issues for each."* | https://github.com/vllm-project/vllm/issues/41559 | closed | READ BODY |
| F8.7 | Three vLLM Medusa/MTP issues stale-closed by bot boilerplate | #16477 (Medusa hangs when tp>1), #20813 (Specifying Medusa Choice Tree), #38339 (Step-3.5-Flash MTP acceptance: *"Current vLLM (main) \| 2.4%-4.6%"* vs *"v0.15.1 + Step-AI patch \| 97%-100%"* vs *"sglang \| ~50%"*) — all: *"This issue has been automatically closed due to inactivity..."* | https://github.com/vllm-project/vllm/issues/16477 · https://github.com/vllm-project/vllm/issues/20813 · https://github.com/vllm-project/vllm/issues/38339 | closed (bot) | READ BODY *(reported, not re-verified)* |
| F8.8 | An engineer's own admission of an untested interaction | vLLM #40831, @Sandermage on the Genesis v7.13 cycle: *"we did not test MTP at all in the v7.13 cycle."* / *"MTP × TurboQuant × cudagraph is a **separate bug class** that v7.13 does not cover."* / *"the depth of 'I don't know what I don't know' here is real."* | https://github.com/vllm-project/vllm/issues/40831 | — | READ BODY *(reported, not re-verified)* |

**Cross-engine pattern.** "Quantized target breaks the draft head" is reported independently in vLLM (#54997, #55250, #51581, #26402, #52873), SGLang (#36599, #38574, #36653, #39087, #38366) and TensorRT-LLM (#3207, declared unsupported outright). The dominant complaint is **silence**: acceptance collapses to ~0 with no error and no warning. Cause per F8.1: quantizers skip the head because it is outside the HF graph, and the engines then guess its precision.

### F9. Repository health (reported, not re-verified)

| # | Finding | URL |
|---|---|---|
| F9.1 | EAGLE last commit 2026-02-19 (maintained); SpecForge 2026-09-10 (maintained); **Medusa last commit 2024-04-18 (abandoned)** | https://github.com/FasterDecoding/Medusa |
| F9.2 | `FasterDecoding/hydra` and `FasterDecoding/self-speculative` both return **HTTP 404 — the repos do not exist**. Any search result implying otherwise is wrong. | https://github.com/FasterDecoding/hydra · https://github.com/FasterDecoding/self-speculative |
| F9.3 | EAGLE README still carries unchecked TODOs: `"- [ ] Support official EAGLE-3 for Qwen-3."` and `"- [ ] EAGLE-4."` | https://github.com/SafeAILab/EAGLE |

### F10. Additional hardware-ruled-out (draft training / eval scale)

SpecForge v0.3 8×H20 disaggregated · SpecForge paper 8×H200 · SpecBundle evals on 4×H200 / 8×B200 / 8×B300 · Ling-3.0-flash 4×Blackwell TP4 · RepSpec 8×A100 · PARD 8×MI250X · PRISM 12×A100 · *Performance or Illusion* 4×H100 for 70B (Llama3-70B + Llama3.2-1B = **133.7 GiB** static) · Kimi-K2.5 W4A8 8×MI325X (497 GB base) · EDA 4×H200.
**NOT ruled out** (single-GPU paths): vLLM #24322 (RTX PRO 6000 96 GB), Speculators single-GPU training, SGLang TLI single-GPU, llama.cpp #25187 (single RTX 5090), F7.1 (single 24 GB GPU, ~2.5 h).

---

## Corrections and red herrings (read this before trusting any lead list)

0. **METHOD CORRECTION (supersedes the provenance caveat at the top of this document).** GitHub issue/PR **comment bodies ARE retrievable** without the API: they live as `"body":"…"` inside `<script type="application/json" data-target="react-app.embeddedData">`. Two operational details matter: (a) curl must **KEEP** the proxy for `github.com` — unsetting it returns HTTP 000 on some paths; (b) the authoritative state is the badge `data-status="issueClosedNotPlanned" | "issueClosed" | "issueOpened" | "pullMerged" | "pullClosed"`. This is how the maintainer quotes in C1, C2, C24, F8.4 and F8.6 were obtained and re-verified. Rows that remain marked "closing comment NOT RETRIEVED" were genuinely not retrieved, not merely assumed missing.
1. **vLLM #43559 is CLOSED, not open** (badge `issueClosed`). Its fix PR **#48375 is still OPEN** (`mergeCommitSha: null`). The accurate and more interesting statement: **the issue was closed while its fix stayed unmerged** — which is why club-3090 still vendors the patch and reports it REQUIRED on v0.29.0.
2. **PRISM and HELIOS are NOT adaptive-speculation-length work.** PRISM is a draft-model *architecture* (MLSys'26 oral, https://mlsys.org/virtual/2026/oral/3789); HELIOS is multi-model *early-exit* selection (MLSys'26 oral, https://mlsys.org/virtual/2026/oral/3846, arXiv 2504.10724). A lead list that conflates them is wrong.
3. **TurboSpec (arXiv 2406.14066) is a closed-loop goodput controller, not a bandit** — and it already answered "speculation sometimes loses": VERBATIM *"TurboSpec automatically disables speculative decoding for large batches where performance benefits diminish"* / *"at a request rate greater than 16, proposing 3 tokens yields performance degradation"* / *"For large batch sizes, not speculate altogether can result in higher goodput."* Its own future work: *"We leave it as future work to develop an accurate and efficient predictor for accepted length."*
4. **vLLM PR #54748's title in circulation is wrong.** The actual title is *"[Spec Decode] Make the dynamic SD schedule observable"* (not "Count scheduler steps by the K dynamic SD selected"). It is **OPEN/unmerged**, and its companion **#54749 is an issue, not a PR**.
5. **ML-SpecQD's correct arXiv ID is 2503.13565** (2505.22148 is an unrelated paper).
6. **"HASS (arXiv 2408.15766)"** = *"Learning Harmonized Representations for Speculative Sampling"* (ICLR 2025) — **not** hardware-aware.
7. **`nvidia/Llama-3.1-70B-Instruct-Eagle3` does not exist** (404). **`FasterDecoding/hydra` and `FasterDecoding/self-speculative` also do not exist** (404). **DFlash2 is an official blog, not an arXiv paper** (https://inco.ai/blog/dflash2/, 2026-08-18); DFlash the paper is arXiv 2602.06036.
8. **SGLang #9888 is not a verification-math bug** — it is a CUDA illegal-memory-access in the EAGLE3 *draft* CUDA-graph runner from padding. Fixed by PR #10892.
9. **`web_search` served fabricated arXiv IDs during this sweep** (e.g. 2402.05099 is *Hydragen*, not *Hydra*; 2403.15304 is not *GLIDE*; 2406.04433/2402.13718 are not *Ouroboros*). Every arXiv ID in this map was verified by fetching `arxiv.org/abs/<id>` and reading back the title+authors, except rows explicitly marked TITLE ONLY.
10. **No maintainer statement using the words "not planned" or "wontfix" about a spec-decode combination was found.** The `not planned` closures are bare state changes. This is a finding about the record, not a gap in the search — and it is itself notable, because it means the *reasons* for those closures are unrecoverable.
11. **Self-contradiction inside vLLM's own docs:** the spec-decode page claims the implementation is *"algorithmically validated to be lossless"*, while arXiv 2605.07698 proves that under a grammar mask rejection sampling samples from the locally-projected distribution, not the grammar-conditional one (TV gap up to 0.996). The docs claim is scoped to the *unconstrained* sampler; the grammar case is a different guarantee.
12. **vLLM docs URL redirect:** `features/spec_decode/` 302-redirects to `features/speculative_decoding/`; content identical on `latest` and `v0.29.0`.
13. **A stale bot closed a resolved issue.** vLLM #31883 was refused (`not planned`), the fix merged as PR #24322, the merger said so in-thread, and the bot still closed it weeks later for inactivity.

---

## Search log

### Direct primary-source retrievals (this agent)
All via `curl` with proxy env vars unset, or via `web_fetch`.

**arXiv (each verified by reading title+authors+abstract from `arxiv.org/abs/<id>`):** 2502.19732, 2601.11580 (+ full HTML body + ar5iv), 2607.17283, 2607.05147, 2607.14647, 2606.24957, 2608.01651, 2608.30427, 2608.20743, 2608.28846, 2609.02897, 2609.09338, 2609.07108, 2608.30135, 2608.02989, 2607.04244, 2607.08690, 2605.07858, 2604.19767, 2605.09329, 2603.18016, 2512.22420, 2608.24004, 2604.26412, 2603.01399, 2502.10424, 2506.11309, 2503.13565, 2505.22179, 2505.17074, 2511.09844, 2509.18362, 2503.07807, 2607.12422, 2607.19223, 2607.21804, 2605.30580, 2605.00342, 2510.22876, 2607.26627, 2602.16994, 2606.20675, 2406.14066, 2405.19715, 2408.11049, 2404.19737, 2401.10774, 2401.15077, 2310.15141, 2302.01318, 2211.17192, 2606.18967, 2605.17613, 2603.03251, 2604.12989, 2606.01019, 2605.15051, 2604.25925, 2606.30265, 2601.05724, 2512.21911, 2602.16052, 2602.07223, 2608.03447, 2604.09603, 2606.18394, 2604.26469, 2607.27735, 2603.03333, 2609.06498, 2605.04263.
**arXiv search UI** (`arxiv.org/search/?searchtype=all&query=…&sortBy=submittedDate`): `speculative decoding` (200 results), `speculative decoding batch` (50), `speculative decoding prefix caching` (25), `speculative decoding tensor parallelism multi-GPU serving` (25), `draft model training speculative decoding distillation` (25), `acceptance length prediction speculative decoding adaptive` (25), `speculative decoding quantization`, `speculative decoding KV cache quantization acceptance`, `EAGLE quantized draft speculative decoding`, `FP8 KV cache speculative decoding acceptance rate`, `W4A16 speculative decoding draft model`, `speculative decoding limitations negative results does not help`.
**Blocked:** `export.arxiv.org/api/query` → HTTP 429 then 503 on every attempt. Later `arxiv.org/search` → HTTP 400 (rate-limited). `r.jina.ai` → HTTP 000. DuckDuckGo HTML → 0 parsed results (throttled); no findings attributed to it.

**vLLM docs** (raw markdown at `main` and `v0.29.0`): `docs/features/speculative_decoding/`: README, acceptance_metrics, adaptive_verification, draft_model, dynamic_speculative_decoding, eagle, extract_hidden_states, mlp, mtp, n_gram, parallel_draft_model, speculators, suffix. Version-diff probe: `acceptance_metrics.md` **404 at v0.28.0, 200 at v0.29.0**. Plus `docs/features/speculative_decoding.html` (nav), `docs/features/batch_invariance/`.
**SGLang docs:** `docs.sglang.io/docs/advanced_features/speculative_decoding` (+`.md`), `.../adaptive_speculative_decoding` (+`.md`); `docs.sglang.ai/...` (redirects).
**Source read:** `vllm/config/speculative.py`, `vllm/config/vllm.py:991`, `vllm/model_executor/models/qwen3_next_mtp.py`, `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`, `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`, `vllm/model_executor/layers/rejection_sampler.py` (404 on main), `vllm/sampling_params.py`, `sglang/.../speculative/eagle_utils.py`, `sglang/.../speculative/ragged_verify.py`, `sglang/.../speculative/adaptive_spec_params.py`, `sglang/kernels/ops/speculative/reject_sampling.py`, `tensorrt_llm/_torch/speculative/spec_sampler_base.py`, `llama.cpp/common/sampling.cpp`.

**GitHub issue/PR state + bodies** (via the `react-app.embeddedData` JSON payload): vllm issues 53215, 52873, 52469, 49488, 47825, 52107, 53178; sglang issues 36001, 4417; PRs 56552, 56375, 56080.
**GitHub HTML issue-search queries (verbatim):** vLLM — `is:issue speculative is:closed reason:"not planned"`, `+label:quantization`, `is:pr speculative is:closed is:unmerged`, `is:pr "spec dec" is:unmerged`, `is:pr eagle is:unmerged`, `is:pr mtp is:unmerged`, `is:pr dflash OR dspark`. SGLang — `is:issue speculative is:closed reason:"not planned"`, `is:pr speculative is:closed is:unmerged`.
**GitHub API:** `/rate_limit`, `/repos/{repo}/issues/{n}/comments`, `/repos/sgl-project/sglang/git/trees/main?recursive=1`, `/repos/{repo}/releases/tags/{tag}`. **Rate limit hit 0 repeatedly** — closing comments for several issues could not be retrieved.

### Web search queries run by this agent (verbatim)
1. `speculative decoding survey 2025 arXiv open problems`
2. `speculative decoding batch size acceptance rate degradation 2025`
3. `EAGLE-3 vLLM issue not planned wontfix`
4. `speculative decoding quantization draft model accuracy 2025 arXiv`
5. `vLLM speculative decoding issue "not planned" draft model batching`
6. `SGLang speculative decoding EAGLE3 issue closed wontfix`
7. `speculative decoding acceptance rate drops with batch size paper 2025`
8. `arXiv 2025 speculative decoding future work open challenge`
9. `SGLang 0.5.19 speculative decoding EAGLE3 NGRAM documentation`
10. `SGLang speculative decoding supported algorithms EAGLE NGRAM MTP draft model docs`
11. `SGLang issue speculative decoding not supported wontfix closed`
12. `"Speculative Decoding: Performance or Illusion" arXiv`

### Delegated sub-sweeps (6 parallel agents)
Draft-model drafting & draft training · N-gram/prompt-lookup drafting · EAGLE/MTP/Medusa heads · Draft budget/length adaptation & acceptance behaviour · Verification strategy · Batching/quantization/caching + engine integration.
Their own full search logs (≈300 additional verbatim queries across web_search, GitHub HTML issue search, arXiv search/abs/HTML, OpenAlex, and raw source retrieval) are preserved in their reports at:
- `spec_decode_heads_sweep.md` (heads sweep, 113 URLs)
- `specdec_sweep/REPORT.md` (draft-budget sweep, 55 closed + 23 open + 38 abandoned rows)
- `specdecode_sweep/specdecode_batching_quant_kvcache_sweep.md` (batching/quant/kv-cache sweep, 478 lines)

### Known gaps in this map
- **Closing-comment text does not exist** for the `not planned` closures (verified: zero comments on each). This is a finding about the record, not a gap in coverage. One exception was later retrieved: vLLM #41559 carries a maintainer closing comment (F8.6).
- **Still not transcribed despite the corrected method:** comment bodies for vLLM #40914, #48375, #50021, #43559 (the #43559 thread was partially retrieved — it is dominated by a third-party AI-generated "fix" comment, not maintainer prose).
- **Not swept:** vision-language / audio / diffusion-LLM speculative decoding beyond what appeared incidentally; ASR spec decode; non-NVIDIA backends (TPU, Gaudi, AMD) except where they surfaced as verifier bugs; security/privacy treated only via arXiv 2607.21804.
- **TITLE ONLY rows retained:** D1.12 (vLLM PR #43814); the mlsys.org slide PDF (subset fonts, no ToUnicode CMap); TechRxiv 10.36227/techrxiv.177101038.80960856 (HTTP 403 Cloudflare); vLLM #54526 + speculators #1065; vLLM #55250; SGLang #36653; Medusa #132; EAGLE #331; llama.cpp #26575; AdaEAGLE arXiv 2412.18910; the DEV Community distribution-shift post.
- **Unverified:** ML-SpecQD venue (abs page lists no journal ref); SGLang #39064 state; SGLang PR #28616 state; vLLM #54166 closed-unmerged status (second-hand from PR #56577's body); vLLM #16477/#20813/#38339 and #40831 quotes marked *(reported, not re-verified)*.
- **MLSys 2026 programme page is JS-rendered** — not swept exhaustively; only the specific orals/posters cited were read.
- **A dedicated quantization sub-sweep over-ran and was stopped.** Its material was rebuilt from primary sources; leads that could not be re-retrieved are flagged TITLE ONLY rather than dropped silently.

---

## Bottom line for this rig (no recommendations — just the boundary of what the record supports)

**Shipped and available on vLLM 0.29.0 / SGLang 0.5.19, single GPU, source-verified:** EAGLE/EAGLE-3, MTP (incl. Qwen3-Next), draft_model (with a documented `Qwen3-4B-Thinking-2507` + `Qwen3-0.6B` example), PARD parallel drafting, n-gram, suffix decoding, heterogeneous-vocab drafting (greedy only), per-request acceptance metrics, Dynamic SD (batch-size table), SGLang's EMA adaptive `num_steps`, and SGLang's DFlash2/UNO/STANDALONE.

**Documented as DSpark-only or SM100-only:** vLLM Adaptive Verification (requires `AttentionCGSupport.ALWAYS`, DSpark + confidence head, full CUDA graphs) — rejected at startup on sm120.

**The three structural blockers that recur across every sub-sweep:** (1) verification is the dominant cost and it scales with batch, so gains decay with concurrency; (2) EAGLE/MTP structurally drop the last matched prefix-cache block, which is *by design* and costs 30–40% throughput on prefix-reusing workloads; (3) quantizing either side silently zeroes acceptance, and the official incompatibility matrices document none of these interactions.
