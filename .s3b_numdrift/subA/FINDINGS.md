# subA — Numerical-determinism / reproducibility vocabulary for LLM forward-pass perturbations

Scratch dir: `/Users/leihenan/Desktop/myProject/.s3b_numdrift/subA/`
Sweep date: 2026-09-14. All retrieval done with `curl` (see environment rules). Every arXiv ID below was
**read back at its `arxiv.org/abs/` page** and the title I actually got is quoted.

**Quote-fidelity note.** Quotes were extracted programmatically from the saved HTML, then HTML tags stripped.
This means (a) inline emphasis/code tags leave a stray space before following punctuation (e.g. `token .`),
and (b) LaTeX math in arXiv HTML appears duplicated (e.g. `0.48 % 0.48\%`). Words and characters are otherwise
exact. Where I trimmed, I used `…`. Nothing is paraphrased inside quote marks.

---

## Artifact table

Legend — **Perturbation class**: INPUT (embeddings/prompt), WEIGHT (quantization/precision of weights),
NUMERICAL (reduction order, accumulation, batch shape, precision of activations). **Level**: DECISION
(argmax / emitted token) vs SOFT (perplexity, accuracy, TV/probability distance, logit delta).

| # | URL | State I verified by fetching | Verbatim quote (exact characters) | Class | Level |
|---|-----|------------------------------|-----------------------------------|-------|-------|
| 1 | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | **PUBLISHED**, live. `<title>` = `Defeating Nondeterminism in LLM Inference - Thinking Machines Lab`; byline "Horace He in collaboration with others at Thinking Machines", "Sep 10, 2025" | "Unfortunately, even defining what it means for LLM inference to be deterministic is difficult. Perhaps confusingly, the following statements are all simultaneously true: Some kernels on GPUs are nondeterministic . However, all the kernels used in a language model's forward pass are deterministic . Moreover, the forward pass of an LLM inference server (like vLLM) can also be claimed to be deterministic . Nevertheless, from the perspective of anybody using the inference server, the results are nondeterministic ." | NUMERICAL | both — DECISION via temperature-0 divergence counts |
| 1b | same | same | "If you compose some property under which the kernel is not invariant (i.e. batch-size) with nondeterminism of that property (i.e. the load the server is under), you get a nondeterministic system." | NUMERICAL | mechanism (no metric) |
| 1c | same | same | "We use Qwen/Qwen3-235B-A22B-Instruct-2507 and sample 1000 completions at temperature 0 with the prompt "Tell me about Richard Feynman" (non-thinking mode), generating 1000 tokens each. Surprisingly, we generate 80 unique completions, with the most common of these occuring 78 times." | NUMERICAL | **DECISION** |
| 1d | same | same | "Looking at where the completions differ, we see that the completions are actually identical for the first 102 tokens! The first instance of diverging completions occurs at the 103rd token." | NUMERICAL | **DECISION** (first-flip position) |
| 1e | same | same | `print((out1 - out2).abs().max()) # tensor(1669.2500, device='cuda:0')` — batch-size-1 vs batch-size-2048 matmul, first row | NUMERICAL | SOFT (raw logit delta; magnitude demo) |
| 2 | https://github.com/vllm-project/vllm/issues/42259 | **OPEN** (`"state":"OPEN"` in embedded JSON; `Open` badge). Title read back: `[RFC]: Logprobs/Logits Semantics and Determinism Across the vLLM Ecosystem · Issue #42259` | "The remaining open reports point to gaps in the semantics and determinism of returned logits/logprobs:" … "batch-size-dependent sampling when logits are tied" … "batch-invariance gaps across async scheduling, TP collectives, attention, convolution, and speculative decoding" | NUMERICAL | **DECISION** (tie-breaking changes the sampled token) |
| 2b | same | same | "— \| #50979 \| OPEN \| PyTorch top-k retains more than `k` tied logits while the Triton path keeps exactly `k`, making sampling batch-size dependent." (inner pipes escaped; this is one row of the RFC's bug table) | NUMERICAL | **DECISION** |
| 2c | same | same | Acceptance criteria: "MRV2 results are deterministic under the documented batch-invariance scope." | NUMERICAL | both |
| 3 | https://github.com/vllm-project/vllm/pull/51292 | **MERGED** (`"state":"MERGED"` + `Merged` badge). Title read back: `[Core] Disable fuse_allreduce_rms under VLLM_BATCH_INVARIANT (non-deterministic under TP) by tolleybot` | Diff (`.diff`, HTTP 200) contains the added code + comment: `# The fused all-reduce + RMSNorm path is not batch-invariant` / `if envs.VLLM_BATCH_INVARIANT:` / `return False` | NUMERICAL | mechanism only (no metric; it disables a kernel) |
| 4 | https://arxiv.org/abs/2506.09501 | **PUBLISHED but RETITLED.** ID is correct, **title is not what the task stated.** `abs/2506.09501v1` → `[2506.09501v1] Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning`; `abs/2506.09501` (v2) → `[2506.09501] Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference`. v2 dateline: `[Submitted on 11 Jun 2025 (v1), last revised 24 Oct 2025 (this version, v2)]`. Also **published at NeurIPS 2025** (proceedings page verified) | "under bfloat16 precision with greedy decoding, a reasoning model like DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size" | NUMERICAL | both — DECISION root cause, SOFT headline numbers |
| 4b | same | same | "Since we are using greedy decoding, the top-1 probability token differs between runs at the point of divergence. Figure 3 shows an example of token logits when two runs diverge, illustrating how numerical precision errors can flip the order of the top-1 and top-2 token probabilities." | NUMERICAL | **DECISION** |
| 4c | same | same | "We observe that for reasoning models, the token probability differences between the top two competing tokens are often minimal ." | NUMERICAL | **DECISION** (margin observation) |
| 4d | same | same | "However, such fluctuations do not always lead to a token flip, since a flip only occurs when the variation at a specific position exceeds the original top-1 and top-2 probability gap, which is inherently a stochastic event." | NUMERICAL | **DECISION** — and note it *explicitly declines* to give a probability model |
| 4e | same | same | "This increased variance arises because BF16's limited mantissa bits (7 bits compared to FP16's 10 bits) introduce larger errors in probability computations, increasing the likelihood of token flips when these fluctuations overlap with the small gap between top-1 and top-2 candidate tokens." | NUMERICAL (precision) | **DECISION** (qualitative likelihood only) |
| 5 | https://icml.cc/virtual/2026/poster/66224 | **PUBLISHED / accepted.** `<title>` = `ICML Poster Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch`; authors "Ziyang Zhang ⋅ Xinheng Ding ⋅ Jiayi Yuan ⋅ Rixin Liu ⋅ Huizi Mao ⋅ Jiarong Xing ⋅ Zirui Liu"; session "Tue, Jul 7, 2026 • 2:00 PM – 3:45 PM KST HALL A #2305". Links to OpenReview `5eZmlUyFpl` and repo `nanomaoli/llm_reproducibility` | "However, existing LLM serving frameworks can produce different outputs for identical inputs when tensor parallel (TP) size or batch size changes, even under greedy decoding. This arises from the non-associativity of floating-point arithmetic and inconsistent reduction orders across GPUs." | NUMERICAL | **DECISION** (+ SOFT "zero probability divergence") |
| 5b | https://arxiv.org/abs/2511.17826 | **PUBLISHED.** `<title>` = `[2511.17826] Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch`. Dateline `[Submitted on 21 Nov 2025 (v1), last revised 29 May 2026 (this version, v2)]` | "existing LLM serving frameworks exhibit non-deterministic behavior: identical inputs can yield different outputs when system configurations (e.g., tensor parallel (TP) size, batch size) vary, even under greedy decoding" … "Experiments confirm zero probability divergence and bit-wise reproducibility for deterministic inference across different TP sizes." | NUMERICAL | DECISION + SOFT |
| 5c | https://raw.githubusercontent.com/nanomaoli/llm_reproducibility/main/README.md | **PUBLISHED.** Confirms the venue chain: "[2026.04.30]: 🎉🎉🎉 Our paper on TBIK (Tree Based Invariant Kernels) has been accepted to ICML 2026!" and "[2025.09.25]: 🎉🎉🎉 Our paper has been selected for Oral Presentation for Neurips 2025." | "We propose TBIK, a TP-invariant matmul method that achieves determinism by strictly controlling the reduction order in matrix multiplications." | NUMERICAL | mechanism |
| 6 | https://docs.vllm.ai/en/v0.19.0/features/batch_invariance/ | **PUBLISHED** (HTTP 200). `<title>` = `Batch Invariance - vLLM` | "Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch." … "Disables certain optimizations that may introduce non-determinism (such as custom all-reduce operations in tensor parallel mode)" | NUMERICAL | **DECISION** ("output of a model") |
| 6b | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/usage/reproducibility.md | **PUBLISHED** (HTTP 200) | "vLLM does not guarantee the reproducibility of the results by default, for the sake of performance." … "Even with the above settings, vLLM only provides reproducibility when it runs on the same hardware and the same vLLM version." | NUMERICAL | both |
| 6c | **vLLM blog post on batch invariance — DOES NOT EXIST** | **NEGATIVE FINDING.** `https://blog.vllm.ai/sitemap.xml` (HTTP 200) lists **208** URLs; `grep -iE 'determin\|invarian\|batch\|reproduc'` returns **zero** matches. `https://blog.vllm.ai/archive.html` likewise. Newest posts are 2026-09. | — (no such artifact) | — | — |
| 6d | https://docs.sglang.io/advanced_features/deterministic_inference.html | **PUBLISHED** (HTTP 200). `<title>` = `Deterministic Inference - SGLang Documentation`. (Note: the older `sgl-project.github.io` URL is a 786-byte redirect stub to this host.) | "Even with temperature=0 , standard LLM inference can produce different outputs due to dynamic batching and varying reduction orders in GPU kernels." | NUMERICAL | **DECISION** |
| 6e | same as 6d | same | "The main source is varying batch sizes . Different batch sizes cause GPU kernels to split reduction operations differently, leading to different addition orders. Due to floating-point non-associativity ( (a + b) + c ≠ a + (b + c) ), this produces different results even for identical inputs." | NUMERICAL | mechanism |
| 7 | https://arxiv.org/abs/2605.30218 | **PUBLISHED.** `<title>` = `[2605.30218] MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference`. Dateline `[Submitted on 28 May 2026]`. Authors per full text: Kexin Chu (UConn), Yang Zhou (UC Davis), Wei Zhang (UConn) | **"The standard argmax perturbation bound ( Thinking Machines Lab, 2026 ; Yuan et al., 2025 ) supplies a practical starting point: if two batch-shaped logit vectors differ by at most ε pert in ℓ ∞ , their argmaxes can diverge only when the relevant margin is about 2 ε pert ."** | NUMERICAL | **DECISION — this is the single closest thing to a flip law that I found** (a sufficient-condition bound, not a probability model) |
| 7b | same | same | "Across five models, batch-induced token flips are sparse on the flip-rate benchmarks: on MATH500, Llama-3.1-8B flips on 0.48 % of synchronous decode steps, and all tested models stay within the 0.3 – 1.3 % range on MATH500, GSM8K, and HumanEval. K/V perturbations remain flat before flips, while low top-1/top-2 logit margins expose much of the flip risk." | NUMERICAL | **DECISION** (flip rate + margin correlation) |
| 7c | same | same | "We therefore inspect the logit vector ℓ over the vocabulary at token-divergence events and ask whether risky steps have a reference-free signature." (Table 4 defines `N(Δ) = \|{j : ℓ_j ≥ ℓ_(1) − Δ}\|`, mean ≈1 at stable steps, ≈2 at divergent steps) | NUMERICAL | **DECISION** — empirical margin distribution, **no fitted law** |
| 7d | same | same | "The difference is not sampling randomness. It is a numerical effect exposed by the serving shape." | NUMERICAL | framing |
| 8 | https://arxiv.org/abs/2604.13206 | **PUBLISHED.** `<title>` = `[2604.13206] Numerical Instability and Chaos: Quantifying the Unpredictability of Large Language Models`. Dateline `[Submitted on 14 Apr 2026]` | "we identify a chaotic "avalanche effect" in the early layers, where minor perturbations trigger binary outcomes: either rapid amplification or complete attenuation." … "1) a stable regime, where perturbations fall below an input-dependent threshold and vanish, resulting in constant outputs; 2) a chaotic regime, where rounding errors dominate and drive output divergence; and 3) a signal-dominated regime, where true input variations override numerical noise." | NUMERICAL (rounding) + INPUT (embedding sweep) | **DECISION** (regime/threshold, no probability) |
| 8b | same | same | "They matter most when the model is near a tie between the top candidate tokens (i.e., when the logit margin is small)." | INPUT/NUMERICAL | **DECISION** — qualitative |
| 9 | https://arxiv.org/abs/2601.17768 | **PUBLISHED.** `<title>` = `[2601.17768] LLM-42: Enabling Determinism in LLM Inference with Verified Speculation`. Dateline `[Submitted on 25 Jan 2026 (v1), last revised 30 Jan 2026 (this version, v2)]`. Authors: Raja Gond, Aditya K. Kamath, Ramachandran Ramjee, Ashish Panwar | "At the system level, this non-determinism arises from floating-point non-associativity combined with dynamic batching and GPU kernels whose reduction orders vary with batch size." … "Our key observation is that if a sequence is in a consistent state, the next emitted token is likely to be consistent even with dynamic batching." | NUMERICAL | **DECISION** (emitted-token consistency) |
| 10 | https://github.com/vllm-project/vllm/issues/50136 | **OPEN**. Title read back: `[Bug] should_custom_ar()'s size threshold makes all-reduce kernel selection batch-dependent, which is why custom all-reduce can't simply be re-enabled under VLLM_BATCH_INVARIANT · Issue #50136` | "Because the all-reduce tensor is `[num_tokens, hidden_size]`, that decision is a function of how many tokens are in the forward pass — i.e. of batch composition. The same request can take the custom kernel in one batch and NCCL in another, with different floating-point reduction orders." | NUMERICAL | mechanism (no metric) |
| 11 | https://github.com/vllm-project/vllm/issues/25404 | **CLOSED**. Title read back: `[Feature]: Kernel Dispatch Overrides (in pursuit of deterministic execution) · Issue #25404` | "I am working towards high-fidelity (+ bit-wise) numerical stability and would like overrides to be possible and easy to configure. For reference, please see the proof of concept from Horace here: #24583 and associated blog post: …" | NUMERICAL | mechanism |
| 12 | https://github.com/vllm-project/vllm/pull/24583 | **CLOSED** (not merged — a PoC). Title read back: `[Proof of Concept] Made vllm deterministic (tested for qwen3-8B) by Chillee` | (body not extractable via `"body":"` pattern; state read from embedded JSON) | NUMERICAL | — |
| 13 | https://raw.githubusercontent.com/thinking-machines-lab/batch_invariant_ops/main/README.md | **PUBLISHED** (HTTP 200). Companion release to the TML blog | "Without the upstream PR, we see that out of 1000 random length 100 completions we see 18 unique samples. After the upstream PR, there is only one unique sample." | NUMERICAL | **DECISION** (unique-sample count) |
| 14 | https://aclanthology.org/2025.naacl-long.211/ and https://arxiv.org/abs/2407.10457 | **PUBLISHED.** ACL Anthology `<title>` = `The Good, The Bad, and The Greedy: Evaluation of LLMs Should Not Ignore Non-Determinism - ACL Anthology`; arXiv `<title>` = `[2407.10457] The Good, The Bad, and The Greedy: Evaluation of LLMs Should Not Ignore Non-Determinism` | "Current evaluations of large language models (LLMs) often overlook non-determinism, typically focusing on a single output per example." | **not numerical** — this is *sampling* variance (greedy vs sampled), not floating-point | SOFT (accuracy) — context only, **not** a forward-pass-perturbation paper |
| 15 | https://github.com/sgl-project/sglang/pull/39319 | **CLOSED** (`"state":"CLOSED"`, `Closed` badge, no `mergedAt`) — i.e. closed **without** merge. Title read back: `[Bugfix] Preserve BF16 batch invariance with DeepGEMM 0.2 by Fridge003 · Pull Request #39319 · sgl-project/sglang` | (body not extractable via `"body":"` pattern) | NUMERICAL (BF16 kernel path) | not stated in retrievable text |
| 16 | https://github.com/sgl-project/sglang/issues/22949 | **OPEN**. Title read back: `Development Roadmap (2026 Q2) · Issue #22949 · sgl-project/sglang` | "Goals: converge on one stable SGLang rollout-engine API to cut per-framework drift on weight sync, sampling, and logprob semantics; upstream shared primitives (R3, FP8, deterministic inference, TIS/MIS) so all four frameworks benefit together." | NUMERICAL (named as a primitive) | neither — roadmap commitment |
| 17 | https://arxiv.org/abs/1902.02918 | **PUBLISHED** — *classical, NOT an LLM paper*; included because it occupies the law slot. `<title>` = `[1902.02918] Certified Adversarial Robustness via Randomized Smoothing`; dateline `[Submitted on 8 Feb 2019 (v1), last revised 15 Jun 2019 (this version, v2)]` | "We prove a tight robustness guarantee in $\ell_2$ norm for smoothing with Gaussian noise." | adversarial `ℓ2` — neither INPUT / WEIGHT / NUMERICAL in the LLM-serving sense | **DECISION** — certified class-stability radius (c-vii) |
| 18 | https://arxiv.org/abs/2607.27275 | **PUBLISHED.** `<title>` = `[2607.27275] Flat Score, Amplified Failures: How the Error Budget Masks Damage in Quantized LLM Agents`; dateline `[Submitted on 29 Jul 2026]` | "The score stays flat because the benchmark's ten-error budget absorbs the extra failures." | **WEIGHT** (4-bit post-training quantization) | **SOFT** (benchmark score, per-channel error rate) — explicitly **not** an argmax-flip paper; included only as a weight-perturbation datapoint where the soft metric masks damage |

---

## (c) Is there a *law* / *bound* / *probability model* for greedy token flip?

**Short answer: I found exactly one decision-level BOUND, and NO probability law.**

**(c-i) The one bound — MarginGate (arXiv 2605.30218), verbatim:**
> "The standard argmax perturbation bound ( Thinking Machines Lab, 2026 ; Yuan et al., 2025 ) supplies a practical starting point: if two batch-shaped logit vectors differ by at most ε pert in ℓ ∞ , their argmaxes can diverge only when the relevant margin is about 2 ε pert ."

This is a **sufficient condition for stability** (`margin > 2ε ⇒ argmax cannot flip`), i.e. a deterministic bound, **not** a probability model. It is the classical robustness/perturbation fact and is not new to LLMs.

**(c-ii) ⚠️ The attribution in that sentence does not survive checking.** MarginGate cites two sources; I fetched both and **neither states the bound**:
- "Thinking Machines Lab, 2026" resolves (per MarginGate's own reference list) to *`batch_invariant_ops`: Batch-invariant implementations of reduction operations*, https://github.com/thinking-machines-lab/batch_invariant_ops — I fetched the README: it is a code README with a batch-invariance unit test and the 18→1 unique-sample note. **It contains no argmax bound, no margin, no ε.**
- "Yuan et al., 2025" = arXiv 2506.09501 — I fetched the full HTML (`arxiv.org/html/2506.09501v2`) and grepped: **`margin` = 0 occurrences, `bound` = 0, `Theorem` = 0, `Proposition` = 0.**
- Likewise the TML *blog* the task pointed at: `margin` = 0, `flip` = 0, `argmax` = 0, `top-1` = 0; its single `bound` is the phrase "boundary conditions" in the FlashAttention discussion.
  - *Rigor note on that negative claim:* `grep -i margin` on the **raw** blog HTML returns **76** hits, which would look like a contradiction. I checked each: **75** are CSS (`margin:`, `margin-top`, `margin-right`) and the rest are the Tufte-style `class="margin-toggle"` sidenote toggle. After removing `<script>`/`<style>`/tags, **`margin` in prose = 0**. The zero is real, not an extraction artifact.

So: **the 2ε bound is real mathematics but is, on my retrieval, not actually stated in either LLM source it is attributed to.** Treat "the bound is established in the LLM literature" as unverified; treat the bound itself as classical numerical analysis (MarginGate also cites Goldberg 1991 and Higham 2002, which is the honest provenance).

**(c-iii) The explicit refusal to give a law** — 2506.09501 states the flip criterion in words and then declines to model it:
> "a flip only occurs when the variation at a specific position exceeds the original top-1 and top-2 probability gap, which is inherently a stochastic event."

**(c-iv) No formal machinery.** I grepped the full text of MarginGate (2605.30218) and the chaos paper (2604.13206) for `Theorem|Proposition|Lemma`: **0 hits in both.** These are empirical/measurement papers, not law-stating papers.

**(c-v) Closest empirical substitutes** (correlations, not laws):
- MarginGate Table 4: `N(Δ) = |{j : ℓ_j ≥ ℓ_(1) − Δ}|`, mean ≈ 1 at token-stable steps vs ≈ 2 at divergent steps (ratio 1.7×–3.0×), across 5 models and Δ ∈ {0.25, 0.5, 1.0, 2.0}.
- MarginGate: flip rates 0.3–1.3% of decode steps, 0.48% for Llama-3.1-8B on MATH500. **Terminology caution:** "the three flip-rate benchmarks" is *MarginGate's own* label for MATH500 / GSM8K / HumanEval — it is not a named external benchmark, so do not cite it as one. Their Table 1 caption reads "Token-level non-determinism rate across the batch-size sweep." and the text says "more than 98.7 % of synchronous decode steps are token-stable across the tested batch shapes".
- Chaos paper (2604.13206): three regimes with "an input-dependent threshold" between stable and chaotic.
- 2506.09501: top-1/top-2 probability gaps for reasoning models are "often minimal"; reports `Avg_Std@top1_prob` (a **SOFT** metric) and explicitly notes it does not always lead to a flip.

**Verdict: no prior work found that gives P(greedy token flips) as a function of (margin, perturbation size). The vocabulary in the literature is "flip rate", "logit margin", "top-1/top-2 margin", "near-tie", "first divergence", "batch invariance", "reduction order" — but the *law* slot is empty; the nearest occupant is a classical sufficient-condition bound repeated once, with an attribution I could not confirm.**

**(c-vi) Why the gap is real, and what the missing object looks like.** The 2ε statement is a *deterministic sufficient condition*, not a probability: it says `gap > 2ε ⇒ no flip` and is silent on `gap ≤ 2ε`, where flips become possible but are not certain. I checked the logic and it is sound — if `ℓ_(1) − ℓ_(j) > 2ε` for every `j ≠ 1`, then `ℓ'_(1) ≥ ℓ_(1) − ε > ℓ_(j) + ε ≥ ℓ'_(j)`, so the argmax is preserved.
To turn this into the *law* an occupancy check would want, a paper must supply **both** (i) a distribution for the per-step perturbation `ε_pert` induced by batch shape / reduction order, and (ii) a distribution for the margin `gap` at decode steps. On my retrieval **nobody does either, and nobody joins them**:
- for (i): MarginGate characterises K/V drift as "flat" at "a bounded pre-divergence baseline" and gives a slope statistic, verbatim: "A per-trial linear regression over the pre-divergence window gives a median deepest-layer K-slope within | 0.007 | |0.007| /token on all five models, indicating no measurable time-axis drift." (the doubled `| 0.007 | |0.007|` is the arXiv-HTML LaTeX-duplication artifact) — a bound, not a distribution; no paper fits a law to `ε_pert`.
- for (ii): the only assets I found are MarginGate's Table 4 counting statistic `N(Δ)` and 2506.09501's Figure 3 histogram, and the latter is **FP32-only and measured only at divergence points**, so it is not a usable prior over margins at arbitrary steps.
**The join of (i) and (ii) is the empty slot.**

**(c-vii) The law slot is not empty in the *classical* literature — only in the LLM literature.** Two mature, decision-level formulations already exist and are directly transferable; a paper claiming the flip law should engage them rather than re-derive:
- **Thurstone's discriminal process / probit (1927).** If each candidate's score is `u_i = µ_i + ϵ_i` with i.i.d. `ϵ ~ F(0, σ²)`, then the probability that the argmax switches from `i` to `j` is `F_diff(µ_j − µ_i)` — for Gaussian noise, `Φ(−gap/(σ√2))`. This *is* a closed-form decision-level flip law parameterised by the margin `gap` and perturbation scale `σ`. **No LLM paper I retrieved uses it.** (I verified the formulation via a Stanford course typescript, *Models of Discrete Choice*, Jonathan Wand, 2006, which states: "Assume that the overall utility of an item is the sum of the observed and unobserved components ui = µi + ϵi." — this is a teaching document, not the primary source; Thurstone 1927 itself is paywalled and I did **not** verify it directly. Treat the attribution as standard-textbook-level, not first-hand.)
- **Certified robustness via randomized smoothing (Cohen, Rosenfeld & Kolter).** Verified directly: https://arxiv.org/abs/1902.02918, `<title>` = `[1902.02918] Certified Adversarial Robustness via Randomized Smoothing`, dateline `[Submitted on 8 Feb 2019 (v1), last revised 15 Jun 2019 (this version, v2)]`. Quote: "We prove a tight robustness guarantee in $\ell_2$ norm for smoothing with Gaussian noise." This yields a **certified radius** inside which the predicted class provably cannot change — a decision-level bound of exactly the `margin vs perturbation` form, but for classifiers under adversarial `ℓ2` perturbation, not for LLM token generation under reduction-order noise.
- The 2ε bound that MarginGate states is the deterministic degenerate case of this family (`σ → 0`).

So the honest framing for an occupancy claim is: **the *form* of the law is classical and settled; what is missing is its instantiation for LLM serving numerics — i.e. a measured/derived distribution for `ε_pert` induced by batch shape and reduction order, calibrated against the empirical margin distribution at decode steps.**

---

## (d) Mechanism literature (the "why engines are nondeterministic" story)

The mechanism consensus, with sources I verified: **float non-associativity is necessary but not sufficient; the trigger is that *serving shape* (batch composition / token count) selects different reduction orders or different kernels at runtime.** Best statements:
- vLLM #50136: kernel *selection* (custom all-reduce vs NCCL) is a function of `num_tokens`, so the same request gets different reduction orders in different batches.
- vLLM PR #51292 (merged): a fused all-reduce + RMSNorm path is simply disabled under `VLLM_BATCH_INVARIANT`.
- vLLM RFC #42259: enumerates the remaining gaps (async scheduling, TP collectives, attention, convolution, speculative decoding, MoE combine).
- TML blog: the load→batch-size→per-request-result causal chain; also the important nuance that a plain `torch.mm` is bitwise run-to-run deterministic, so the naive "GPU concurrency" hypothesis is wrong.
- LLM-42 (2601.17768): frames it as dynamic batching + shape-varying reduction orders, and works around it by *scheduling* rather than kernel rewrites.
- TBIK (2511.17826): extends determinism from batch-size to **TP size** by forcing a consistent hierarchical tree reduction order.

---

## (i) Could NOT verify — and why

| Item | Why |
|------|-----|
| **OpenReview page for ICML 2026 poster 66224** (id `5eZmlUyFpl`, extracted from the ICML page) | **Blocked.** `openreview.net/forum?id=5eZmlUyFpl` → HTTP 200 but 4787-byte browser challenge, `<title>Verifying your browser \| OpenReview</title>`. `openreview.net/pdf?id=5eZmlUyFpl` → **HTTP 403**, `<title>Error 403 \| OpenReview</title>`. Consistent with the documented environment rule. I therefore verified artifact 5 via the ICML virtual page + arXiv 2511.17826 + the repo README instead. |
| **"Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference"** (OpenReview `QDOKyg7a5e`, surfaced by `web_search`) | **Unverified.** `openreview.net/forum?id=QDOKyg7a5e` → 4787-byte challenge; `openreview.net/pdf?id=QDOKyg7a5e` → **HTTP 403** (HTML, 12692 bytes). Mirror attempts failed: `papers.cool/venue/QDOKyg7a5e@OpenReview` and `.../QDOKyg7a5e` both → `Not Found`. Semantic Scholar graph API → **HTTP 429** on both attempts. I could not read back a title, so I do not cite it as an artifact. **The title as reported by search is not verified by me.** |
| **vLLM PR #51292 description body** | The page's embedded JSON yielded `"state":"MERGED"` and the title, but no `"body":"…","bodyHTML"` pair (0 hits). I verified the change itself from the authoritative `.diff` (HTTP 200) instead. |
| **Horace He / TML follow-up that states an argmax bound** | MarginGate cites "Thinking Machines Lab, 2026" for the bound, but its reference list resolves that to the `batch_invariant_ops` **GitHub repo**, whose README I fetched — no bound. I also enumerated **every** post on `thinkingmachines.ai/blog/` and `/news/` (HTTP 200, both): `defeating-nondeterminism-in-llm-inference` (Sep 10, 2025) is the **only** determinism post, and there is **no** 2026 TML post on the subject. Conclusion: no TML *post* states the 2ε bound. **The 2ε bound's true provenance remains unresolved — I could not verify any LLM source that states it.** |
| **vLLM blog post on batch invariance** (task artifact 6 suggested one by Vikram Sharma) | **Does not appear to exist.** `blog.vllm.ai/sitemap.xml` = 208 URLs, zero matching `determin\|invarian\|batch\|reproduc`. Reported as a negative finding, not as unverified. |
| **arXiv 2506.09501 titled "Give Me FP32 or Give Me Death?"** as a *current* title | The **task's stated pairing of ID↔title is stale**: that title is v1 only. Verified both titles directly. Flagging because it will mis-cite otherwise. |
| **Semantic Scholar API** | HTTP 429 ("Too Many Requests") on both queries, including after a 20 s backoff. Not used. |
| **SGLang determinism issues** | **Partially resolved.** My first extraction regex failed (GitHub nests the title in a child element); re-extracting on `data-testid="issue-pr-title-link"` worked and yielded two verified items — SGLang PR #39319 (`[Bugfix] Preserve BF16 batch invariance with DeepGEMM 0.2`, **CLOSED** without merge) and roadmap issue #22949 (**OPEN**, names "deterministic inference" as a primitive to upstream). See rows 15–16. No SGLang issue *dedicated to* nondeterminism was found by this query. |
| **llm-d determinism issues** | **Not found.** `github.com/llm-d/llm-d/issues?q=determinism+OR+deterministic+OR+nondeterminism&state=all` → HTTP 200 (312 KB) but **0** title matches for `determin\|reproduc\|invarian`. Web search surfaced only unrelated `llm-d-batch-gateway` retry-flake PRs #429/#447. |
| **MarginGate's Table 4 / Table 3 raw numbers beyond those quoted** | I read the tables from the arXiv HTML; some numeric cells are mangled by LaTeX duplication in HTML (e.g. `0.48 % 0.48\%`). I quote only cells whose value is unambiguous. |

---

## (ii) Exact search queries and URLs I ran

**`web_search` queries (verbatim):**
1. `argmax token flip probability logit margin perturbation LLM greedy decoding`
2. `logit margin distribution LLM top-1 stability numerical perturbation`
3. `greedy decoding first token divergence batch size floating point LLM serving`
4. `LLM-42 per-token verification deterministic inference`
5. `Greedy Decoding Is Not Precision-Invariant Cross-Precision Output Divergence openreview`
6. `flip-rate benchmark batch-induced token flips LLM`
7. `vLLM blog batch invariance deterministic kernels Vikram Sharma`
8. `site:blog.vllm.ai batch invariance`
9. `Horace He defeating nondeterminism follow-up batch invariant ops 2026`
10. `vLLM blog "batch invariance" deterministic inference post`
11. `site:blog.vllm.ai determinism`
12. `llm-d deterministic inference issue github`
13. `SGLang github issue nondeterminism batch size token flip`
14. `"The Good, the Bad, and the Greedy" Evaluation of LLMs should not ignore non-determinism NAACL aclanthology`

**Direct URLs fetched with `curl` (and what happened):**
- `https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/` → 200
- `https://arxiv.org/abs/2506.09501` → 200 (v2 title)
- `https://arxiv.org/abs/2506.09501v1` → 200 (v1 title — the discrepancy that matters)
- `https://arxiv.org/html/2506.09501v2` → 200
- `https://proceedings.neurips.cc/paper_files/paper/2025/hash/f80094a824ba5912d4a2de169c404a40-Abstract-Conference.html` → 200 (NeurIPS 2025 confirmation)
- `https://github.com/vllm-project/vllm/issues/42259` → 200 (OPEN)
- `https://github.com/vllm-project/vllm/pull/51292` → 200 (MERGED)
- `https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/51292.diff` → 200
- `https://github.com/vllm-project/vllm/issues/50136` → 200 (OPEN)
- `https://github.com/vllm-project/vllm/issues/50979` → 200 (OPEN; it is actually a **PR** titled `[Bugfix][Sampler] Keep exactly k tokens in top-k with tied logits`)
- `https://github.com/vllm-project/vllm/issues/25404` → 200 (CLOSED)
- `https://github.com/vllm-project/vllm/pull/24583` → 200 (CLOSED)
- `https://icml.cc/virtual/2026/poster/66224` → 200
- `https://arxiv.org/abs/2511.17826` → 200
- `https://raw.githubusercontent.com/nanomaoli/llm_reproducibility/main/README.md` → 200
- `https://arxiv.org/abs/2605.30218` + `https://arxiv.org/html/2605.30218v1` → 200
- `https://arxiv.org/abs/2604.13206` + `https://arxiv.org/html/2604.13206v1` → 200
- `https://arxiv.org/abs/2601.17768` → 200
- `https://raw.githubusercontent.com/thinking-machines-lab/batch_invariant_ops/main/README.md` → 200
- `https://docs.vllm.ai/en/v0.19.0/features/batch_invariance/` → 200
- `https://raw.githubusercontent.com/vllm-project/vllm/main/docs/usage/reproducibility.md` → 200
- `https://blog.vllm.ai/` → 200; `https://blog.vllm.ai/sitemap.xml` → 200 (208 URLs, no match); `https://blog.vllm.ai/archive.html` → 200
- `https://docs.sglang.io/advanced_features/deterministic_inference.html` → 200
- `https://sgl-project.github.io/advanced_features/deterministic_inference.html` → 200 but **786-byte redirect stub**
- `https://thinkingmachines.ai/blog/` → 200 and `https://thinkingmachines.ai/news/` → 200 (full post enumeration; only one determinism post, Sep 10 2025)
- `https://aclanthology.org/2025.naacl-long.211/` → 200 (correct paper)
- `https://aclanthology.org/2025.naacl-long.213/` → 200 but **WRONG PAPER** (PROMPTEVALS) — recorded as a caught error
- `https://arxiv.org/abs/2407.10457` → 200
- `https://arxiv.org/abs/1902.02918` → 200 (classical certified-robustness bound)
- `https://arxiv.org/abs/2607.27275` → 200 (quantized LLM agents, soft metric)
- `https://openreview.net/forum?id=5eZmlUyFpl` → 200 challenge; `/pdf?id=5eZmlUyFpl` → **403**
- `https://openreview.net/forum?id=QDOKyg7a5e` → 200 challenge; `/pdf?id=QDOKyg7a5e` → **403**
- `https://papers.cool/venue/QDOKyg7a5e@OpenReview` → 200 "Not Found"
- `https://api.semanticscholar.org/graph/v1/paper/search?...` → **429** (×2)
- `https://github.com/sgl-project/sglang/issues?q=determinism+OR+nondeterminism+OR+%22batch+invariant%22&state=all` → 200; first regex found nothing, re-extract on `data-testid="issue-pr-title-link"` yielded the titles
- `https://github.com/sgl-project/sglang/pull/39319` → 200 (**CLOSED**)
- `https://github.com/sgl-project/sglang/issues/22949` → 200 (OPEN, roadmap)
- `https://github.com/llm-d/llm-d/issues?q=determinism+OR+deterministic+OR+nondeterminism&state=all` → 200 (0 matches)

---

## Bonus observations worth passing upstream

1. **The ID↔title mismatch on 2506.09501 is a live citation hazard.** "Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning" is the **v1** title; the paper is now "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference" and is a **NeurIPS 2025** paper (the repo README says **Oral**). If a related-work section cites the old title, it is citing a superseded version.
2. **A small citation cluster exists and is worth treating as one group:** the same authors (Jiayi Yuan, Xinheng Ding, Zirui Liu, Xia Hu) appear on 2506.09501, on ICML 2026 poster 66224 / arXiv 2511.17826, and share the code repo `nanomaoli/llm_reproducibility`. MarginGate (2605.30218) then cites that cluster for the 2ε bound.
3. **The "decision vs soft metric" split is the real gap.** The field measures a lot of soft things (`Avg_Std@top1_prob`, accuracy std, probability divergence) and a few decision things (flip rate, unique-completion count, first-divergence index), but nobody has published the flip *probability* as a function of margin and perturbation size — which is precisely the occupancy slot.
