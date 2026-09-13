# Speculative Decoding — VERIFIER Implementation Sweep (2025-06 → 2026-09)

Scope: the VERIFIER only (rejection sampling / typical acceptance / tree verification), across vLLM,
SGLang, TensorRT-LLM, llama.cpp, HF transformers, and others. Every item carries a URL and a
READ BODY / TITLE ONLY marker. Quotes are verbatim.

---

## Section A — CLOSED / SHIPPED / IN-TREE

| # | Question it answers | Who closed it | Artifact | URL | Status | Evidence |
|---|---|---|---|---|---|---|
| A1 | What does vLLM's v1 verifier actually implement? | vLLM maintainers (woosuk) | `vllm/v1/sample/rejection_sampler.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/sample/rejection_sampler.py | shipped on `main` | READ BODY |
| A2 | Does vLLM still have the old `model_executor/layers/rejection_sampler.py`? | vLLM maintainers | legacy path | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/rejection_sampler.py (404) vs https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5/vllm/model_executor/layers/rejection_sampler.py (200, 400 lines) | REMOVED on `main`; present at v0.8.5 | READ BODY |
| A3 | Typical-acceptance as an alternative verifier | simon-mo (merged) | PR #5131 | https://github.com/vllm-project/vllm/pull/5131 | merged (50 commits) | READ BODY |
| A4 | Does vLLM document losslessness? With what caveats? | vLLM docs | docs "Lossless guarantees of Speculative Decoding" | https://docs.vllm.ai/en/latest/features/speculative_decoding/ | published | READ BODY |
| A5 | Which sampling params are refused under spec decode? | vLLM | `vllm/sampling_params.py::_validate_spec_decode` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/sampling_params.py | shipped | READ BODY |
| A6 | Is the verifier tested for distribution convergence / seed determinism? | vLLM | `tests/v1/sample/test_rejection_sampler.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/tests/v1/sample/test_rejection_sampler.py | shipped | READ BODY |
| A7 | Was there a real distribution bug in the verifier's noise stream? | WoosukKwon (merged) | PR #54282 | https://github.com/vllm-project/vllm/pull/54282 | merged | READ BODY |
| A8 | Seed / reproducibility semantics + spec decode | vLLM docs | `docs/usage/reproducibility.md` | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/usage/reproducibility.md | published | READ BODY |
| A9 | SGLang tree-mask construction modes | SGLang maintainers | `python/sglang/srt/speculative/eagle_utils.py` | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/speculative/eagle_utils.py | shipped | READ BODY |
| A10 | SGLang's chain verifier accept test (Triton) | SGLang maintainers | `python/sglang/kernels/ops/speculative/reject_sampling.py` | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/kernels/ops/speculative/reject_sampling.py | shipped | READ BODY |
| A11 | Ragged / variable-size tree verification in a batch | SGLang maintainers | `python/sglang/srt/speculative/ragged_verify.py` | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/speculative/ragged_verify.py | shipped | READ BODY |
| A12 | `speculative_sampling.py` location in SGLang today | SGLang maintainers | dir listing | https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/speculative | **file no longer exists** — replaced by `spec_utils.py`, `ragged_verify.py`, `eagle_utils.py` | READ BODY |
| A13 | HF fix to a broken `_speculative_sampling` | gante (merged) | PR #28508 | https://github.com/huggingface/transformers/pull/28508 | merged | READ BODY |
| A14 | Why HF forces the assistant to sample when the target samples | gante (merged) | PR #33534 | https://github.com/huggingface/transformers/pull/33534 | merged; **reverts #30778** | READ BODY |
| A15 | Original HF "speculative sampling breaks the distribution" report | closed by PR #33534 | Issue #32867 | https://github.com/huggingface/transformers/issues/32867 | closed (COMPLETED) | READ BODY |
| A16 | HF: drafter returns a q it did not sample from | closed (COMPLETED) | Issue #47932 | https://github.com/huggingface/transformers/issues/47932 | closed | READ BODY |
| A17 | TRT-LLM: verifier rejects sampling params it cannot honor | NVIDIA | `tensorrt_llm/_torch/speculative/spec_sampler_base.py` | https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/tensorrt_llm/_torch/speculative/spec_sampler_base.py | shipped | READ BODY |
| A18 | TRT-LLM removed the two-model spec-dec path (BREAKING) | zhaoyangwang-nvidia (merged) | PR #18721 | https://github.com/NVIDIA/TensorRT-LLM/pull/18721 | merged Sep 13 2026 | READ BODY |
| A19 | llama.cpp's verifier acceptance test | ggml-org | `common/sampling.cpp::common_sampler_sample_and_accept_n` | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/sampling.cpp | shipped | READ BODY |
| A20 | llama.cpp synthetic (benchmark-only) verification path | ggml-org | `tools/server/server-context.cpp` | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/server-context.cpp | shipped | READ BODY |
| A21 | SGLang docs on spec decode (tree/verify config) | SGLang docs | advanced_features/speculative_decoding | https://docs.sglang.ai/advanced_features/speculative_decoding.html | published | READ BODY |
| A22 | vLLM padding of resumed spec-decode requests (raggedness) | jeejeelee (merged) | PR #55126 | https://github.com/vllm-project/vllm/pull/55126 | merged Sep 4 2026 | READ BODY |
| A23 | vLLM verifier dtype / FP32 chunk canvas | (open PR, author only) | PR #53630 | https://github.com/vllm-project/vllm/pull/53630 | open | READ BODY |

### A1 — vLLM v1 `RejectionSampler`: what it implements (READ BODY)

Source: `vllm/v1/sample/rejection_sampler.py` (1124 lines), fetched from raw.githubusercontent.com.

Verbatim class docstring:
> "The implementation strictly follows the algorithm described in https://arxiv.org/abs/2211.17192."

Terminology it defines, verbatim: "accepted tokens: tokens that are accepted based on the relationship between the "raw" draft and target probabilities."; "recovered tokens: tokens that are sampled based on the adjusted probability distribution, which is derived from both the draft and target probabilities."; "bonus tokens: If all proposed tokens are accepted, the bonus token is added to the end of the sequence. The bonus token is only sampled from the target probabilities."

**Sampling-parameter caveat, verbatim (this is the answer to Q4 from the source itself):**
> "We pass in the bonus tokens instead of sampling them in the rejection sampler to allow for more flexibility in the sampling process. For example, we can use top_p, top_k sampling for bonus tokens, while spec decode does not support these sampling strategies."

Core accept test in `rejection_random_sample_kernel` (verbatim):
```
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```
preceded by the comment:
> "# NOTE(woosuk): While the draft probability should never be 0, we check it to avoid NaNs. If it happens to be 0, we reject."

Greedy path is a separate Triton kernel (`rejection_greedy_sample_kernel`); the padded-draft sentinel is documented verbatim as "# -1 is used for padded draft token ids that should be rejected."

Notable in-tree machinery:
- `MAX_SPEC_LEN = 128` with comment "Maximum number of speculative draft tokens allowed per request in a single step. This value is chosen to be large enough to handle typical use cases."
- A **third** verifier mode beyond rejection sampling: `spec_config.rejection_sample_method == "synthetic"`, using `synthetic_conditional_rates` and `unconditional_to_conditional_rates(...)`; the kernel branch is `accepted = uniform_prob < rate`.
- `apply_sampling_constraints()` applies temperature scaling + top-k/top-p to target logits, with verbatim comment "# NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask, which is slow for large vocab sizes. This may cause performance issues."
- FP64 uniform sampling with verbatim comment: "# NOTE(woosuk): We deliberately use float64 instead of float32 here because when using float32, there's a non-negligible chance that uniform_prob is sampled to be exact 0.0 as reported in https://github.com/pytorch/pytorch/issues/16706. Using float64 mitigates the issue."
- Reproducibility-motivated code comments (twice, in `generate_uniform_probs` and `sample_recovered_tokens`): "# Do not generate random numbers for requests with no draft tokens. This can be important for reproducibility."
- `rejection_greedy_sample_kernel` carries: "# FIXME(woosuk): Because is_greedy_ptr is not None at profiling run, re-compilation may happen during runtime when is_greedy_ptr is None."

### A2 — the legacy verifier path is gone (READ BODY)

`vllm/model_executor/layers/rejection_sampler.py` returns **404 on `main`**. The same path at tag
`v0.8.5` returns 200 (400 lines). `v0.10.0` already returns 404. So the pre-v1 `RejectionSampler`/
`rejection_sample` layer was deleted when v1 became the only path.

### A3 — typical acceptance merged as an alternative verifier (READ BODY)

PR #5131, "[Speculative Decoding 1/2] Add typical acceptance sampling as one of the sampling
techniques in the verifier", merged by **simon-mo** (50 commits). Author's description, verbatim:

> "In this PR we implement typical acceptance sampling as one of the alternate sampling techniques for the Speculative Decoding verifier. In this sampling we do the following"
> "For every draft token we compare its probability in the target probability distribution with the entropy of the target distribution and if it meets the acceptance criteria then we accept it. More details on the acceptance criteria can be found in section 3.3.1 in this paper ( https://arxiv.org/pdf/2401.10774 )"
> "If no draft tokens are accepted then we use greedy sampling to sample a token from the target distribution corresponding to k = 0. The paper also specifies sampling as the fallback option in case no draft tokens are selected"
> "If all the draft tokens are selected then we append the bonus token to the end. The Medusa paper does not talk about the bonus token optimization. However we accept the bonus token since it is drawn from the same target distribution that we are using for accepting the rest of the tokens."
> "Create a base class spec_decode_base_sampler.py to be used by both the rejection_sampler and the new typical_acceptance_sampler."

### A4 — vLLM's published losslessness claims and caveats (READ BODY)

From https://docs.vllm.ai/en/latest/features/speculative_decoding/ — verbatim:

> "Theoretical Losslessness - Speculative decoding sampling is theoretically lossless up to the precision limits of hardware numerics. Floating-point errors might cause slight variations in output distributions, as discussed in Accelerating Large Language Model Decoding with Speculative Sampling"
> "Algorithmic Losslessness - vLLM's implementation of speculative decoding is algorithmically validated to be lossless. Key validation tests include:"
> "Rejection Sampler Convergence : Ensures that samples from vLLM's rejection sampler align with the target distribution."
> "Greedy Sampling Equality : Confirms that greedy sampling with speculative decoding matches greedy sampling without it. This verifies that vLLM's speculative decoding framework, when integrated with the vLLM forward pass and the vLLM rejection sampler, provides a lossless guarantee."
> "vLLM Logprob Stability - vLLM does not currently guarantee stable token log probabilities (logprobs). This can result in different outputs for the same request across runs."

Explicitly-listed causes of divergence, verbatim:
> "While vLLM strives to ensure losslessness in speculative decoding, variations in generated outputs with and without speculative decoding can occur due to following factors:"
> "Floating-Point Precision : Differences in hardware numerical precision may lead to slight discrepancies in the output distribution."
> "Batch Size and Numerical Stability : Changes in batch size may cause variations in logprobs and output probabilities, potentially due to non-deterministic behavior in batched operations or numerical instability."

Documented incompatibilities, verbatim:
> "Pipeline parallelism is not composable with speculative decoding as of vllm<=0.15.0"
> "Speculative decoding with draft models is not supported in vllm<=0.10.0"
> "use_heterogeneous_vocab currently supports greedy draft sampling only. Probabilistic acceptance (temperature > 0 draft sampling) is not yet supported and will be added in a future release."

### A5 — vLLM hard-rejects min_p and logit_bias under spec decode (READ BODY)

From `vllm/sampling_params.py`, `_validate_spec_decode`, verbatim:
```python
# Some sampling parameters are not yet compatible with spec decoding.
if self.min_p > _SAMPLING_EPS or self.logit_bias:
    raise VLLMValidationError(
        "The min_p and logit_bias sampling parameters "
        "are not yet supported with speculative decoding."
    )
```

### A6 — vLLM verifier tests: distribution convergence + seed determinism (READ BODY)

`tests/v1/sample/test_rejection_sampler.py`:
- `test_rejection_sampling_approximates_target_distribution()` — docstring verbatim: "Verify rejection sampling approximates target distribution, despite sampling from a potentially distinct draft distribution." … "We expect that as we increase the number of samples, the distance between the observed distribution and the target distribution decreases."
- `test_deterministic_when_seeded(...)` — parametrized over `k` in [1,3,5], `vocab_size=1000`, `batch_size` in [1,4,8], `frac_seeded` in [0.0, 0.5], `n_rep=20`; asserts `torch.equal(results[j][i], results[0][i])` for seeded sequences. Note: it only asserts determinism for the *seeded* subset, and seeds are **per-request** `torch.Generator(...).manual_seed(i)`.

### A7 — a real distribution bug in the vLLM verifier, found and fixed (READ BODY)

PR #54282, "[Bugfix][Model Runner V2][Spec Decode] Decouple the draft's gumbel noise stream from the
target's", merged by **WoosukKwon**. Author's diagnosis, verbatim:

> "During target verification, the Gumbel noise used to re-sample a rejected draft token is drawn from the same Philox offset as the noise that produced that draft token (the same seed, pos), so byte-identical noise. Conditioned on the proposal winning the argmax, the other tokens' Gumbels are truncated below that max, most tightly for the tokens the draft ranked highest, but lost (high-q loser tokens generally have smaller noise, because otherwise they would have won), so the residual under-weights exactly those tokens and the output distribution is no longer the target's."
> "This affects draft_sample_method="probabilistic" only. Under the default greedy, draft_logits is None, the draft never calls gumbel_sample , and there is no shared noise vector."

Fix, verbatim: "Adds a salt (1 << 30) to the offset of the draft's Philox stream, decoupling the draft's Gumbel noise from the target's. Verification is a probability-ratio test against the cached draft distribution rather than a Gumbel coupling, so nothing depends on the two streams matching."

Also documents a second latent defect, verbatim: "the sampling position was previously derived from a buffer clamped at max_model_len, so near the context limit two consecutive draft steps could reuse one noise vector. It is now tracked unclamped."

### A8 — vLLM seed/reproducibility doc explicitly cites spec decoding (READ BODY)

`docs/usage/reproducibility.md`, verbatim:
> "It is impossible to un-specify a seed for V1 because different workers need to sample the same outputs for workflows such as speculative decoding. For more information, see: https://github.com/vllm-project/vllm/pull/17929"
> "Even with the above settings, vLLM only provides reproducibility when it runs on the same hardware and the same vLLM version."
> "In V1, the seed parameter defaults to 0 which sets the random state for each worker, so the results will remain consistent for each vLLM run even if temperature > 0."

### A9 — SGLang tree-mask modes (READ BODY)

`python/sglang/srt/speculative/eagle_utils.py`, `build_tree_kernel_efficient(...)`. It takes
`tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK` and `tree_mask_buf`, and branches over
`TreeMaskMode.QLEN_ONLY`, `TreeMaskMode.QLEN_ONLY_BITPACKING`, `TreeMaskMode.FULL_MASK` (else
`raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")`). Verbatim layout comments:

> "# e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)"
> "# where each row indicates the attending pattern of each draft token"
> "# if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)"
> "# position: where each token belongs to"
> "# e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7"
> "# then, positions = [7, 8, 8, 9]"

There is a per-step memset optimization with the verbatim justification: "Only the [0, seq_len) prefix columns depend on this fill; the kernel below writes every tree cell itself. Skip the (up to 100s of MB) per-step memset when nothing reads the mask."

Backends are dispatched per-platform: `torch.ops.npu.build_tree_kernel_efficient` (NPU),
`sgl_build_tree_kernel_triton` (XPU), `sgl_build_tree_kernel_efficient_cpu` (CPU),
`sgl_build_tree_kernel_efficient` (CUDA). `verify_tree_greedy_triton` and `verify_tree_greedy_func`
exist alongside.

**Verifier selection** (verbatim from the same file):
```
use_rejection_sampling = get_spec().speculative_use_rejection_sampling
sampling_fn = (
    chain_speculative_sampling_triton
    if use_rejection_sampling
    else tree_speculative_sampling_target_only
)
```
with the target probs renormalized by `top_k_renorm_prob` / `top_p_renorm_prob` before verification,
each guarded by `maybe_detect_nan(...)`.

### A10 — SGLang's Triton chain verifier accept test (READ BODY)

`python/sglang/kernels/ops/speculative/reject_sampling.py`, `speculative_sampling_classic_kernel`,
verbatim:
```
p = tl.load(TargetProbs + offset_prob)
q = tl.load(DraftProbs + offset_draft)
coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
if coin * q < p:
```
Residual on rejection is `relu(p - q)`; all-accepted branch samples pure target `p`. Verbatim
comment on the out-of-bounds concern:
> "# DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When all drafts are accepted cur_prob_row == num_steps is out of bounds for DraftProbs, but the all-accepted branch samples pure target p and never dereferences this pointer; on rejection cur_prob_row <= num_steps - 1."

and on the degenerate residual:
> "# Treat NaN q (degenerate draft rows) as 0: residual falls back to p."
> "# Pass 2: CDF. Degenerate residual (norm_sum == 0, i.e. p == q everywhere on rejection) leaves the cumsum at 0 <= target_u, so final_token falls back to VOCAB_SIZE - 1; acceptable since this case is numerically near-impossible."

### A11 — SGLang ragged verification (variable tree sizes in a batch) (READ BODY)

`python/sglang/srt/speculative/ragged_verify.py` defines verbatim:
```python
class RaggedVerifyMode(str, Enum):
    STATIC = "static"
    CAP_ACCEPT = "cap-accept"
    COMPACT = "compact"
```
read from `envs.SGLANG_RAGGED_VERIFY_MODE`, and validates the ragged layout with verbatim error
strings:
> "every request must verify the anchor (verify_len >= 1), got {…}"
> "capped layout has a row exceeding cap={…}"
> "total_verify_tokens {…} exceeds graph_num_tokens {…}"

The struct field `cap` is documented verbatim as: "Per-row upper bound (capped padded variant); rows never exceed it, so dense [bs, cap] consumers stay in bounds. None = full-coverage variant." `compute_ragged_extend_lengths`, `compute_uniform_extend_lengths`, `build_capture_verify_lens`, `compute_target_verify_graph_key`, and `ragged_verify_compact_enabled()` are the other entry points.

### A12 — SGLang's `speculative_sampling.py` no longer exists (READ BODY)

Directory listing of `python/sglang/srt/speculative` on `main` contains no `speculative_sampling.py`.
Relative to the requested paths, the tree now holds (among others): `spec_utils.py`, `eagle_utils.py`,
`eagle_info.py`, `spec_info.py`, `draft_utils.py`, **`ragged_verify.py`**, plus DFlash/DSpark layers
(`dflash_utils.py`, `dflash_info_v2.py`), UNO (`uno_tree.py`, `uno_validation.py`), and
`multi_layer_eagle_utils.py`. `spec_utils.py` itself carries the verifier-adjacent helpers, e.g.
`renorm_draft_probs(..., use_rejection_sampling, ...)` with verbatim docstring:

> "Plain softmax, except under rejection sampling where logits are temperature-scaled so the draft proposal q tracks the target sampling temperature (higher acceptance; correctness holds for any q)."
> "Returns (q, q(X), X). The verify's accept test coin*q(X) < p(X) is unbiased"

### A13/A14/A15/A16 — HF transformers verifier history (READ BODY)

**PR #28508 "Fix _speculative_sampling implementation"**, merged by **gante**. Verbatim:
> "Current implementation of _speculative_sampling accepts the draft model tokens all the time due to faulty test of the number of matches ( n_matches ). After fixing this issue I found and fixed several more issues in the implementation to reproduce the exact algorithm presented in the paper."

**Issue #32867 "Speculative sampling does not maintain probability distribution of main model"** — reporter observed that assisted decoding was effectively greedy. Verbatim:
> "However, the speculative model is always used greedily: … self . generation_config . do_sample = False . This is equivalent to setting the temperature to zero, so the output probability of the assistant model should always be 1 (for the selected token)."
> "Should theoretically be sampling but is not … Always outputs `public int get_current_time()`"

**PR #33534 "Generate: assistant should sample when the main model samples"**, merged by gante, closes
#32867. Verbatim maintainer rationale:
> "TL;DR in assisted generation, the assistant model must sample when the main model is sampling. Otherwise, mathematical properties in the corresponding code path do not hold (see speculative decoding paper)."
> "This reverts #30778 , where I forced the assistant model to always run greedy decoding for speed purposes (more matched candidate tokens = faster)."

**Issue #47932 "Speculative decoding: candidate generators return a q they did not sample from, breaking losslessness with do_sample=True"** (closed, stateReason COMPLETED). Verbatim:
> "Three of the block drafters return logits that are not the distribution their tokens were drawn from, so with do_sample=True the acceptance test is run against the wrong q ."
> "With a top-k warper the drafter samples from q(x)/Z ( Z = raw mass kept by the warper) but reports q(x) , so p_i/q_i is computed a factor 1/Z too large and draft tokens are accepted more often than the algorithm allows."
> "which uses them for the acceptance ratio … and for the [p - q]+ residual it samples from on rejection"

### A17 — TensorRT-LLM verifier refuses params it cannot honor (READ BODY)

`tensorrt_llm/_torch/speculative/spec_sampler_base.py::validate_request`, docstring verbatim:
> "Reject sampling parameters the one-model speculative path cannot honor."
> "The one-model sampling kernels take only temperature/top_k/top_p (see SpecMetadata.populate_sampling_params_for_one_model); min_p has no buffer there, so it would be silently dropped and the request would decode from a different distribution than the user asked for. Threading it through costs measurable throughput on the rejection path, so reject instead. This sampler also does not return context logits, generation logits, or log probabilities."

Raised messages, verbatim:
> "min_p is not supported with one-model speculative decoding. Drop min_p from the request, or disable speculative decoding."
> "repetition_penalty / presence_penalty / frequency_penalty require 'enable_penalty: true' in the speculative decoding config when using one-model speculative decoding."
> "repetition_penalty / presence_penalty / frequency_penalty are not supported with tree speculative decoding (eagle_choices / dynamic tree) yet. Drop the penalties, use a linear speculation mode, or disable speculative decoding."
> "The following output options are not supported with speculative decoding: {…}. Drop these options from the request, or disable speculative decoding."

Preceded by the verbatim comment on why tree penalties are refused: "Tree speculation lays each request's logits out as tree nodes, where a row's history is its root path rather than the rows before it. Applying the linear mapping there would let sibling branches penalize each other, so reject until tree-aware prefixes are implemented."

### A18 — TRT-LLM deleted the two-model verifier path (READ BODY)

PR #18721, "[None][refactor] BREAKING: Remove the two-model speculative decoding path and dead C++
spec-dec code", merged by **zhaoyangwang-nvidia** on Sep 13 2026. Verbatim:
> "This PR does two related things: it removes the two-model speculative decoding path , which had already been made unreachable and was marked for removal, and it clears out the dead code left on the C++ speculative-decoding side ."
> "An after-validator overwrote eagle3_one_model back to True , warning that 2-model "will be removed in a future release""
> "The API and ABI removals require downstream updates."

### A19/A20 — llama.cpp verifier mechanics (READ BODY)

`common/sampling.cpp`, `common_sampler_sample_and_accept_n(...)`, verbatim core:
```cpp
for (; i < draft.size(); i++) {
    const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
    common_sampler_accept(gsmpl, id, true);
    result.push_back(id);
    if (draft[i] != id) {
        break;
    }
}
```
i.e. the llama.cpp verifier is a token-equality match against the target's own sample, not a
probability-ratio test.

`tools/server/server-context.cpp` has a parallel **synthetic** verification path, verbatim comment:
> "// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target"
> "// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions"
with in-body comments:
> "// do not accept a drafted EOG token - it would end the generation early"
> "// synthetic draft tokens do not advance grammar or reasoning state"
> "// the last replay token is from the target and must advance both"

Server-side verification dispatch, verbatim:
```cpp
auto accepted = synth_probs.empty()
    ? common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft)
    : server_sample_and_accept_synth(...);
```

### A21 — SGLang docs (READ BODY)

From https://docs.sglang.ai/advanced_features/speculative_decoding.html:
> "—speculative-num-draft-tokens Maximum parallel verification capacity. Allows deeper tree evaluation but increases GPU memory usage."
> "In tree mode, both speculative acceptance thresholds must remain at 1.0, V must be at least B and at most 128, and V * K must not exceed 2048. SGLang validates the remaining tree-capacity and EAGLE parent-representation constraints at startup."
> "Ordinary overlap scheduling is supported. Tree mode does not yet support PDMux or the separate --enable-two-batch-overlap feature."
> "Optional: set SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True to force greedy verification."
> "UNO's K controls proposal-tree breadth and is unrelated to the request sampling parameter top_k ."
> "In SGLang's EAGLE-2 implementation, the draft tree is expanded for the configured steps and then reranked to select the top_num_draft_tokens final nodes as draft tokens."
> "Compared with EAGLE-style tree verification, DFLASH verifies a linear draft block and is configured around a block size / draft window."
> "Speculative decoding may increase GPU memory usage because the draft tree, CUDA graphs, and verification-related buffers consume additional VRAM."

### A22 — vLLM pads resumed spec-decode requests for uniform verifier shape (READ BODY)

PR #55126 (merged Sep 4 2026). Verbatim:
> "Keep resumed speculative-decode requests on the uniform verifier shape when a data-parallel rank has no already-running requests."
> "A single qlen=1 rank then prevents the DP group from using the full decode CUDA graph."
> "This change treats a request with resolved computed tokens as resumed and applies the existing reject-only -1 padding, provided no prefill was scheduled."

It cross-references other padding fixes: "[SpecDecode] Preserve reject-only residual verifier padding #54102 addresses Mamba alignment, rejection sentinels, adaptive verification, and verifier graph topology", "[Bugfix][Spec Decode] Don't pad a resumed decode request past max_model_len #53812", "[Bugfix][Core] Skip spec padding below prefill threshold #54028", "[Bugfix][Core][Spec Decode] Exclude scheduler padding from draft metrics #50518".

### A23 — vLLM verifier dtype/FP32 canvas (READ BODY)

PR #53630 (open). Verbatim:
> "Speculative decoding expands each request into multiple verification positions. Whenever any active logits processor is present, the target rejection verifier currently materializes an FP32 processing canvas of shape [num_verification_tokens, vocab_size] . The production path bounds the transient allocation with a 1 GiB chunk, which in turn forces a cap on the total logits adaptive verification may schedule per step."

---

## Section B — OPEN (who is still asking, and what is missing)

| # | Question it answers | Who is still asking | URL | What is missing |
|---|---|---|---|---|
| B1 | Greedy spec decode is not token-identical to target-only | forlayo (vLLM) | https://github.com/vllm-project/vllm/issues/54928 | Root cause + fix; open, labeled `bug`, `tool-calling` |
| B2 | DSpark "forced-reject" still drifts → not lossless | b24822530 (SGLang) | https://github.com/sgl-project/sglang/issues/35150 | GDN recurrent-state correctness in TARGET_VERIFY/commit |
| B3 | DFlash2 greedy diverges from target-only with thinking on | forlayo (SGLang) | https://github.com/sgl-project/sglang/issues/38009 | DFlash/hybrid state/thinking path isolation |
| B4 | CPU verifier consumes uninitialized temperature/top_k/top_p | mosafariuk (vLLM) | https://github.com/vllm-project/vllm/issues/55005 | Dtype-correct CPU `expand_kernel` shim (PR #55006 open) |
| B5 | min_p silently dropped on verified target tokens | camerono (vLLM) | https://github.com/vllm-project/vllm/pull/42802 | Merge; `_validate_spec_decode` still 400s the request |
| B6 | EAGLE + sliding-window attention verify is incorrect | whzzt (SGLang) | https://github.com/sgl-project/sglang/pull/39287 | Merge; tree mask + window bounds per node position |
| B7 | ROCm EAGLE verify silently commits argmax (ignores temp/top_p) | xiaobochen-amd (SGLang) | https://github.com/sgl-project/sglang/pull/37134 | Merge (re-filed after #31214 was closed unmerged) |
| B8 | top-p/top-k renorm is nondeterministic run-to-run → TP rank divergence | gilfordting (SGLang) | https://github.com/sgl-project/sglang/pull/38565 | Merge; deterministic AIR/radix kernels |
| B9 | Compact ragged verify (variable tree sizes) unusable | hushengkai (SGLang) | https://github.com/sgl-project/sglang/issues/39173 | Slot-count bug in ragged CUDA-graph capture |
| B10 | Spec decode corrupts returned logprobs | jvdneste (llama.cpp) | https://github.com/ggml-org/llama.cpp/issues/27972 | Server accept path emits p=1.0 for every token |
| B11 | Verifier in head dtype (memory/adaptive cap) | jyan-R (vLLM) | https://github.com/vllm-project/vllm/pull/53630 | Merge |
| B12 | Bounded-lossy greedy verification for HF | Kissmetothemoon (HF) | https://github.com/huggingface/transformers/issues/48636 | Feature request; strict argmax is the only lossless greedy mode |
| B13 | GLM-5.3 MTP draft acceptance ~1-2% | (vLLM) | https://github.com/vllm-project/vllm/issues/55605 | TITLE ONLY — not fetched |

### Verbatim quotes of the asking

**B1 — vLLM #54928** (READ BODY):
> "Qwen3.8-27B with the Qwen3.8-27B-DFlash2 draft is not greedy-equivalent to the same BF16 target when Qwen thinking is enabled. This is reproducible with temperature=0 , so it is a correctness issue rather than a normal sampling difference."
> "With greedy decoding, speculative decoding should emit the same token sequence as target-only decoding."

**B2 — SGLang #35150** (READ BODY):
> "I can reproduce a deterministic correctness divergence between ordinary Base decode and DSpark TARGET_VERIFY even when every speculative draft token is deliberately rejected and only the target model's own argmax token is committed."
> "The failure is cumulative rather than context-local:"
> "Base decode is deterministic across independent processes with a fixed seed."
> "Under gamma=1 forced rejection, DSpark initially matches Base but later diverges."
> "Using FP32 Mamba/SSM persistent state delays the divergence substantially, but does not eliminate it."
> "This suggests accumulated GDN recurrent-state drift somewhere in the TARGET_VERIFY / intermediate-state / commit path."

**B3 — SGLang #38009** (READ BODY):
> "For greedy decoding ( temperature=0 ), DFlash2 speculative decoding should produce exactly the same generated token sequence as target-only decoding for the same model, prompt, chat-template settings, and seed."
> "the DFlash2 result is exactly equal to the target-only result with thinking disabled. This makes the issue appear specific to the DFlash/hybrid state/thinking path rather than prompt formatting, sampling, or general model loading."

**B4 — vLLM #55005** (READ BODY):
> "So on any CPU spec-decode run with a non-greedy request, logits.div_(temperature) ( :545 ) and apply_top_k_top_p ( :565 ) consume uninitialized new_empty memory — fresh zero pages give div_(0) → inf logits; arbitrary garbage gives the out-of-bounds crash previously observed downstream. A second layer: even with copy-back, _ensure_int64(input_val) truncates floats — temperature 0.7 → 0 → rewritten to 1 by the replace_from=0 logic."
> "This was reported before, and the fix everyone believed in is the buggy code"

**B5 — vLLM PR #42802** (READ BODY):
> "RejectionSampler.apply_logits_processors only iterates sampling_metadata.logitsprocs.non_argmax_invariant , which excludes MinPLogitsProcessor (min_p is argmax-invariant). The consequence is that under speculative decoding, min_p is silently dropped on every verified target token — only the bonus token via the regular Sampler path honors it."
> "Without min_p under spec decoding, Qwen3-Coder-Next 80B-A3B (and similar code-tuned models) hit repetition collapse on long completions — the model emits result result result … until the cap."

**B6 — SGLang PR #39287** (READ BODY):
> "EAGLE verification can produce incorrect outputs with sliding-window attention in FlashInfer and Triton. FlashInfer clips the prefix but gathers from offset zero and retains the full-prefix mask layout. Triton's deterministic path also reads the full mask with a shortened row width, while its normal path applies the window using flattened query indices, which differ from tree depth for sibling nodes."
> "EAGLE additionally needs a matching tree mask and window bounds based on each node's position."
> "Cross-backend validation also found a missing causal mask in TRT-LLM MHA's XQA verification call. This is already addressed by the open PRs #32269 and #36038 , so it is outside the scope of this PR."
> "AITER may have similar issues, but is outside the scope of this PR."

Reported accuracy impact (temperature=0, TP=1, concurrency=16): Gemma3-27B AWQ (EAGLE3) / FlashInfer / MT-Bench "6.1063 → 8.9313"; LiveCodeBench v6 increment "21.14% → 25.71%".

**B7 — SGLang PR #37134** (READ BODY):
> "EAGLE spec-decode verify has been committing argmax on ROCm, ignoring" [temperature and top_p — body truncated at fetch]

**B8 — SGLang PR #38565** (READ BODY):
> "sgl_kernel.top_p_renorm_probs and sgl_kernel.top_k_renorm_probs forward to flashinfer's default kernels, and both of those are non-deterministic run to run on byte-identical input"
> "Every TP rank runs these kernels independently on the same logits. In DFlash/DSpark non-greedy verification ( build_dflash_verify_target_probs -> tree_speculative_sampling_target_only ) the output is compared against a uniform coin per draft position, and the bonus token is sampled from relu(q - p) over the same tensor. When a coin lands inside the few-ULP gap between two ranks' target_probs , one rank alone accepts/rejects a draft token or picks a different bonus token. Its radix/KV state silently drifts from the other ranks; a later turn on that conversation prefix-matches to different lengths per rank and the ranks wedge in an NCCL collective."

Measured, verbatim: "top_p_renorm_probs default (AIR) | 199 / 199" calls differ; "top_k_renorm_probs default (radix multi-CTA) | 199 / 199"; "sgl_kernel's compiled single-CTA top_k_renorm_probs | 0 / 199".

**B9 — SGLang #39173** (READ BODY):
> "AssertionError: engram target-verify expects one equal block per request, got 42 tokens for 8 requests of 6"
> "The test escape hatch SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE=1 avoids the capture assertion but then dies on the first request (second traceback below), so compact ragged verify currently appears unusable for this model."
> "Dense-layout consumers elsewhere handle ragged layouts through RaggedVerifyLayout.padded_to_bucket(padded_bs, cap) , but the Engram hasher reads the raw input_ids.shape[0] / req_pool_indices.shape[0] pair before any padding."

**B10 — llama.cpp #27972** (READ BODY):
> "The affected code is the generic speculative accept path in tools/server/server-context.cpp ; it runs after verification and never inspects weights, tensor layout, or spec type, so every --spec-type and every target/drafter" [...]
> "Acceptance was healthy in the repro below ( draft_n_accepted=14 of draft_n=24 ), i.e. the drafter is working normally."

**B11 — vLLM PR #53630** (READ BODY): see A23.

**B12 — HF #48636** (READ BODY):
> "Greedy assisted decoding in Transformers uses strict argmax verification: the entire draft suffix is discarded at the first position where the draft token differs from the target argmax. This is lossless, but it rejects near-optimal draft tokens whose logit gap to the argmax is negligible, capping the practical speedup of speculative decoding for greedy users."
> "PR #45979 ( assistant_ensemble_weight ) added a lossy verification option for the sampling path by mixing the target and draft distributions. Greedy decoding, however, still lacks a deterministic, explicitly-bounded relaxation mechanism."
> "The policy is fully deterministic (no RNG) and exactly reduces to strict greedy verification when B = 0 or m = 0 (all r_i must be 0, i.e. strict argmax prefix matching) — our gold-standard correctness check."

---

## Section C — ATTEMPTED AND ABANDONED (closed-unmerged / not planned / stale)

All of the following were verified as **closed without merge** (page shows "wants to merge" rather than
"merged N commits").

| Artifact | What it tried to do | Status | URL |
|---|---|---|---|
| vLLM PR #50808 | FLy: entropy-gated deferred (approximate) verification | closed-unmerged | https://github.com/vllm-project/vllm/pull/50808 |
| vLLM PR #44885 | Adaptive verifier step-length (D-Cut style) | closed-unmerged | https://github.com/vllm-project/vllm/pull/44885 |
| HF PR #47961 | Return the true proposal distribution q from candidate generators | closed-unmerged | https://github.com/huggingface/transformers/pull/47961 |
| llama.cpp PR #28061 | Stop re-verifying replayed draft tokens after checkpoint restore | closed-unmerged | https://github.com/ggml-org/llama.cpp/pull/28061 |
| SGLang PR #31214 | ROCm EAGLE verify greedy fix | closed-unmerged | https://github.com/sgl-project/sglang/pull/31214 |
| SGLang PR #32630 | "[AMD][Not-Merge]": torch fallbacks for top-k/top-p renorm | closed-unmerged (author-marked Not-Merge) | https://github.com/sgl-project/sglang/pull/32630 |
| SGLang issue #25587 | NPU Hybrid-GDN MTP not lossless | closed, label `inactive` (closed by github-actions bot) | https://github.com/sgl-project/sglang/issues/25587 |
| HF PR #30778 → reverted by #33534 | Forced the assistant to always decode greedily "for speed purposes" | reverted | https://github.com/huggingface/transformers/pull/33534 |

### Verbatim author/maintainer words

**vLLM PR #50808 — FLy verifier (closed-unmerged)** (READ BODY):
> "FLy is lossy by design and disabled by default : it does not preserve the target distribution, and nothing changes unless rejection_sample_method="fly" is set."
> "This PR adds FLy ( arXiv:2511.22972 ), an opt-in approximate verification policy for draft-model speculative decoding."
> "Standard rejection sampling stops at the first rejected draft token, so a single ambiguous position truncates the whole draft and wastes the remaining tokens the target has already verified."
> "At such a position FLy defers to the draft token instead of rejecting, which lets the rest of the window be accepted."
> "the target's top-3 entropy at i is at least fly_entropy_threshold — the target is genuinely uncertain here, not confidently disagreeing;"
> "This keeps the acceptance code paths, per-position accounting, and metrics" [...]
> "Greedy path ( apply_fly_greedy_acceptance_kernel ): overwrites the target argmax at position i with the draft token id."

Note: a second FLy PR exists at https://github.com/vllm-project/vllm/pull/53987 ("[Spec Decode][ROCm] Add FLy: entropy-gated deferred verification for draft-model speculative decoding") — READ BODY (existence + title), full body not read.

**vLLM PR #44885 — adaptive verifier step-length (closed-unmerged)** (READ BODY):
> "Add adaptive verifier step-length for parallel speculative decoding (DFlash, PARD / draft_model + parallel_drafting ). The drafter still proposes the full num_speculative_tokens every step, but the verifier dynamically shortens the next step's query_len based on drafter softmax probabilities and a profiled ITL cost table."
> "Core algorithm follows D-Cut: Adaptive Verification Depth Pruning for Speculative Decoding : hardware-profiled cost table + draft-confidence prefix-product scores + batch-wide global top-K allocation to maximize expected_accepted_tokens / verifier_ITL ."

**HF PR #47961 — the q-contract fix (closed-unmerged)** (READ BODY):
> "_speculative_sampling treats the returned candidate_logits as q , the proposal distribution the draft tokens were drawn from. Three candidate generators broke that contract, so with do_sample=True (and the default top_k=50 ) assisted decoding is not lossless as documented"
> "With the mismatch, the acceptance ratio p_i/q_i is inflated by 1/Z (the raw mass kept by top-k/top-p), and on rejection the [p - q]+ residual is built from mass the drafter puts on tokens it can never propose."
> "SinglePositionMultiTokenCandidateGenerator returns one-hot logits (point mass on the drafted token) describing its deterministic argmax proposal."

**llama.cpp PR #28061 — replay re-verification (closed-unmerged)** (READ BODY):
> "With full-checkpoint rollback, a partial draft acceptance restores the pre-round state and re-decodes the accepted tokens to rebuild it. That replay currently runs through the same verification as a fresh draft. On backends where logits depend on batch shape or memory layout (Vulkan), the re-verification can reject a token the original verification accepted; the rejection restores the same checkpoint and replays again, and the slot loops on one position without emitting anything."
> "This accepts the replayed tokens without re-verifying, and samples only the continuation from the final position. The replayed prefix was already accepted by the verification that triggered the restore — the replay exists to rebuild state, not to re-decide."
> "Not verified by me: I have not reproduced the livelock on unmodified master — see the issue for why. The original author reports the change is bit-identical on CPU (800-token greedy pair, 118 restore rounds), which matches the expectation that only batch-shape-sensitive backends are affected."

**SGLang PR #31214 — ROCm greedy verify (closed-unmerged)** (READ BODY):
> "On ROCm/HIP, EAGLE speculative decoding always commits argmax (greedy) in verify . It ignores temperature and top_p :"
> "if sampling_info . is_all_greedy or _is_cpu or _is_npu or _is_hip or _is_xpu : target_predict = torch . argmax (...) # greedy"
> "_is_hip is in that list because the sampling-verify ops ( tree_speculative_sampling_target_only , top_k_renorm_prob , top_p_renorm_prob ) are only imported under is_cuda()/is_musa() . So any EAGLE spec-decode run at temperature>0 on ROCm decodes greedy . On long reasoning output that turns into repetition loops. This is not model specific."

**SGLang PR #32630 — author-marked "[Not-Merge]" (closed-unmerged)** (READ BODY):
> "build_dflash_verify_target_probs() calls top_k_renorm_prob() and top_p_renorm_prob() unconditionally, but sgl_kernel only exports them under CUDA/MUSA. On ROCm both are None , so DSPARK/DFLASH speculative decoding dies with: TypeError: 'NoneType' object is not callable as soon as a request sets top_p < 1 or top_k > 1 ."
> "tree_speculative_sampling_target_only has no cheap equivalent and stays None"
> "One edge case worth noting: top_p <= 0 would mask every token and produce an all-zero row, so the most likely token is always kept."

**SGLang issue #25587 — NPU MTP not lossless, closed as `inactive`** (READ BODY):
> "When running Qwen3.5 (or any hybrid-GDN model) with MTP speculative decoding on Ascend NPU, the generated output diverges from non-speculative decoding at temperature=0.0 . This violates the lossless guarantee that is fundamental to any correct speculative decoding implementation."
> "At temperature=0.0 , MTP speculative decoding output must be bitwise-identical to non-speculative decoding. This is guaranteed on NVIDIA and is the standard correctness criterion for speculative decoding."
> "The current code calls conv_state_rollback to right-shift conv_states by (N - k) positions, intending to undo the effect of rejected tokens. This is mathematically incorrect"
> "SiLU is non-linear and non-invertible"
> "Right-shifting only moves existing (already-corrupted) values around; it does not restore what the window would have contained had only k tokens been processed. The result is a conv state that drifts further from ground truth on every verify call."
> "NPU acceptance rate 15–20% lower than H100"
Closing: the issue carries the label `inactive` and was closed by `github-actions` (bot). No maintainer technical rebuttal is present in the page body.

**HF PR #30778 — reverted** (evidenced from #33534, which I read):
> "This reverts #30778 , where I forced the assistant model to always run greedy decoding for speed purposes (more matched candidate tokens = faster)."

---

## Search log (every query, verbatim, with engine)

**web_search tool (4 queries in one call):**
1. `vLLM rejection_sampler.py implementation modified rejection sampling typical acceptance`
2. `vLLM speculative decoding lossless distribution preserving docs`
3. `SGLang speculative decoding lossless guarantee docs`
4. `speculative decoding seed reproducibility different output batch size issue vLLM`

**GitHub issue/PR search (github.com HTML), verbatim query strings:**
5. `https://github.com/vllm-project/vllm/issues?q=speculative+decoding+incorrect+output&state=all`
6. `https://github.com/vllm-project/vllm/issues?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative`
7. `https://github.com/vllm-project/vllm/issues?q=speculative+decoding+seed+reproducib&state=all`
8. `https://github.com/vllm-project/vllm/issues?q=%22spec+decode%22+different+output&state=all`
9. `https://github.com/vllm-project/vllm/issues?q=rejection+sampling+speculative&state=all`
10. `https://github.com/sgl-project/sglang/issues?q=speculative+lossless&state=all`
11. `https://github.com/sgl-project/sglang/issues?q=speculative+tree+mask&state=all`
12. `https://github.com/sgl-project/sglang/issues?q=speculative+incorrect+output&state=all`
13. `https://github.com/vllm-project/vllm/issues?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative+verifier`
14. `https://github.com/sgl-project/sglang/issues?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative`
15. `https://github.com/vllm-project/vllm/issues?q=typical+acceptance+sampler&state=all`
16. `https://github.com/vllm-project/vllm/issues?q=rejection+sampler+top_p+top_k&state=all`
17. `https://github.com/sgl-project/sglang/issues?q=speculative+sampling+top_p+top_k&state=all`
18. `https://github.com/vllm-project/vllm/issues?q=spec+decode+%22not+lossless%22+OR+%22distribution%22&state=all`
19. `https://github.com/NVIDIA/TensorRT-LLM/issues?q=speculative+decoding+incorrect+output&state=all`
20. `https://github.com/NVIDIA/TensorRT-LLM/issues?q=rejection+sampling+OR+typical+acceptance&state=all`
21. `https://github.com/ggml-org/llama.cpp/issues?q=speculative+decoding+different+output&state=all`
22. `https://github.com/ggml-org/llama.cpp/issues?q=speculative+seed+reproduc&state=all`
23. `https://github.com/huggingface/transformers/issues?q=assisted+generation+different+output&state=all`
24. `https://github.com/huggingface/transformers/issues?q=speculative+decoding+correctness&state=all`
25. `https://github.com/vllm-project/vllm/issues?q=rejection_sample_method+synthetic&state=all`
26. `https://github.com/vllm-project/vllm/issues?q=FLy+verification&state=all`
27. `https://github.com/vllm-project/vllm/issues?q=tree+mask+speculative+verify+ragged&state=all`
28. `https://github.com/sgl-project/sglang/issues?q=ragged+verify+padding&state=all`
29. `https://github.com/mlc-ai/mlc-llm/issues?q=speculative+decoding+correctness&state=all`
30. `https://github.com/turboderp-org/exllamav2/issues?q=speculative+decoding&state=all`
31. `https://github.com/vllm-project/vllm/issues?q=%22rejection+sampler%22+bug&state=all`
32. `https://github.com/vllm-project/vllm/issues?q=speculative+decoding+temperature+0+different+results&state=all`
33. `https://github.com/NVIDIA/TensorRT-LLM/issues?q=speculative+decoding+accuracy+regression&state=all`
34. `https://github.com/ggml-org/llama.cpp/issues?q=speculative+decoding+acceptance+verification&state=all`
35. `https://github.com/huggingface/transformers/issues?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative+sampling`
36. `https://github.com/vllm-project/vllm/issues?q=is%3Apr+is%3Aclosed+is%3Aunmerged+%22rejection+sampl%22`
37. `https://github.com/ggml-org/llama.cpp/issues?q=speculative+decoding+is%3Aclosed+label%3Awontfix`
38. `https://github.com/vllm-project/vllm/issues?q=speculative+label%3A%22won%27t+fix%22+OR+label%3A%22not+planned%22&state=all`
39. `https://github.com/sgl-project/sglang/issues?q=speculative+is%3Aclosed+is%3Aunmerged+verification`
40. `https://github.com/NVIDIA/nim-deploy/issues?q=speculative+decoding&state=all`
41. `https://github.com/friendliai/friendli-client/issues?q=speculative&state=all`
42. `https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Apr+is%3Aclosed+is%3Aunmerged+rejection+sampling`
43. `https://github.com/vllm-project/vllm/issues?q=accepts+token+reject+spec&state=all`
44. `https://github.com/vllm-project/vllm/issues?q=batch+invariance+speculative&state=all`
45. `https://github.com/sgl-project/sglang/issues?q=speculative+top_p+renorm+deterministic&state=all`
46. `https://github.com/vllm-project/vllm/issues?q=speculative+greedy+identical+non-speculative&state=all`

**GitHub code search:**
47. `https://github.com/search?q=repo%3Asgl-project%2Fsglang+tree_speculative_sampling_target_only&type=code`

**Direct raw.githubusercontent.com / docs fetches (source & docs reads, not searches):**
`vllm/v1/sample/rejection_sampler.py`, `vllm/model_executor/layers/rejection_sampler.py` (404 on main; 200 at v0.8.5), `vllm/sampling_params.py`, `vllm/v1/sample/ops/topk_topp_sampler.py`, `docs/usage/reproducibility.md`, `tests/v1/sample/test_rejection_sampler.py`, `sglang/.../speculative/{spec_utils.py,eagle_utils.py,eagle_info.py,ragged_verify.py,spec_info.py}`, `sglang/kernels/ops/speculative/{reject_sampling.py,spec_tree.py,ragged_verify_kernels.py}`, `NVIDIA/TensorRT-LLM/tensorrt_llm/_torch/speculative/spec_sampler_base.py`, `ggml-org/llama.cpp/common/speculative.cpp`, `ggml-org/llama.cpp/common/sampling.cpp`, `ggml-org/llama.cpp/tools/server/server-context.cpp`, `docs.vllm.ai/en/latest/features/speculative_decoding/`, `docs.sglang.ai/advanced_features/speculative_decoding.html`.

---

## Raw material saved in workspace

All fetched sources, docs and stripped issue/PR bodies are under
`/Users/leihenan/Desktop/myProject/spec_sweep/` (e.g. `vllm_v1_rejection_sampler.py`,
`vllm_sampling_params.py`, `sglang_eagle_utils.py`, `sglang_ragged_verify.py`,
`trt_spec_sampler_base.py`, `x_speculative.cpp`, `llama_server_ctx.cpp`, and `*.txt` bodies).
