# Prior-art sweep: speculative decoding × other engine features

Window prioritized: 2025-06 → 2026-09. Sweep date: 2026-09-13.
Evidence retrieved via `web_search`, `web_fetch`, `curl` to `raw.githubusercontent.com` / `arxiv.org` / `codeload.github.com`, and the GitHub **search** API (which returns full issue/PR bodies).

**Read-status legend.** `READ BODY` = I retrieved and read the page/file/issue-body text. `TITLE ONLY` = I saw only a search-result title/state/label line.
**Caveat on one source class.** GitHub per-issue *comments* (maintainer reply threads) could **not** be retrieved: `api.github.com` core quota was exhausted (0/60 remaining) and github.com HTML is unreachable from this network. Therefore **no maintainer comment text is quoted in this report**; every maintainer-attributed quote below comes from a PR/issue **body** (authored text), not from a comment thread. Issues whose closure reason lived only in a comment are marked NOT VERIFIED for the reason.

---

## Findings

### (a) Spec decode + STRUCTURED OUTPUT (grammar / xgrammar / outlines / JSON schema)

#### A. CLOSED

| # | Question it answers | Who closed it / who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| a1 | Was spec-decode + structured output **officially documented as incompatible**? | vLLM maintainers (rbryant) | PR #12373 `[Docs] Update spec decode + structured output in compat matrix` | https://github.com/vllm-project/vllm/pull/12373 | **merged 2025-01-24** | READ BODY |
| a2 | What shipped to make them work together? | vLLM (benchislett et al.) | PR #14702 `[V1][Feature] Enable Speculative Decoding with Structured Outputs` | https://github.com/vllm-project/vllm/pull/14702 | **merged 2025-04-30** | READ BODY |
| a3 | First documented failure mode | user report; fixed by #28298 | Issue #27210 `[Bug]: EAGLE Spec Decoding + Structured Outputs causes FSM Crash` | https://github.com/vllm-project/vllm/issues/27210 | **closed/completed 2025-11-08** | READ BODY |
| a4 | FSM crash under `json_object` + eagle3 | user report; closed completed | Issue #27969 `[Bug]: Structured output causes crash in speculative decoding` | https://github.com/vllm-project/vllm/issues/27969 | **closed/completed 2025-11-11** | READ BODY |
| a5 | Reasoning-boundary breakage: grammar silently disabled | user report; fixed by #44993/#44297 | Issue #43388 `json_object ... not enforced after Qwen thinking because reasoning end token is missed with async scheduling + spec decode` | https://github.com/vllm-project/vllm/issues/43388 | **closed/completed 2026-07-23** | READ BODY |
| a6 | `</think>` detection failure under MTP + reasoning | user report; fixed | Issue #34650 `Bug: Speculative Decoding (MTP) Causes </think> Detection Failure in Structured Output + Reasoning Mode` | https://github.com/vllm-project/vllm/issues/34650 | closed/completed (per search index) | TITLE ONLY |
| a7 | Strict tool calling + MTP/EAGLE rejects `</think>` → HTTP 500 | user report; fixed | Issue #44006 `[structured outputs] speculative decoding + VLLM_ENFORCE_STRICT_TOOL_CALLING=1 failed to advance FSM` | https://github.com/vllm-project/vllm/issues/44006 | closed/completed (per search index) | TITLE ONLY |
| a8 | Sec-level stalls on spec-decode + tool-call structured output | closed completed | Issue #49002 `Speculative Decoding + Structured Output（tool call）组合下，decode 阶段出现秒级卡顿` | https://github.com/vllm-project/vllm/issues/49002 | closed/completed | TITLE ONLY |
| a9 | Bitmask/FSM hardening at the reasoning boundary | vLLM (yuyue0225sc) | PR #44297 | https://github.com/vllm-project/vllm/pull/44297 | **merged 2026-07-04** | READ BODY |
| a10 | Bonus-token bitmask corrupted by padded `-1` drafts | vLLM | PR #44292 | https://github.com/vllm-project/vllm/pull/44292 | closed, **merged_at=None (closed-unmerged)** 2026-06-02 | READ BODY |
| a11 | Grammar advance across reasoning boundary | vLLM | PR #44993 | https://github.com/vllm-project/vllm/pull/44993 | **merged 2026-07-23** | READ BODY |
| a12 | Same-step grammar advance for all structured types | vLLM | PR #48516 | https://github.com/vllm-project/vllm/pull/48516 | closed, **merged_at=None (closed-unmerged)** 2026-07-23 | READ BODY |
| a13 | MTP spec-decode off-by-one in `should_advance()` | vLLM | PR #44927 | https://github.com/vllm-project/vllm/pull/44927 | closed, **merged_at=None (closed-unmerged)** 2026-08-22 | READ BODY |
| a14 | Async-grammar bitmask misalignment after draft trimming (ngram_gpu) | vLLM | PR #49738 | https://github.com/vllm-project/vllm/pull/49738 | closed, **merged_at=None (closed-unmerged)** 2026-08-23 | READ BODY |
| a15 | DSpark zero-draft-budget grammar crash | vLLM | PR #52436 | https://github.com/vllm-project/vllm/pull/52436 | **merged 2026-08-16** | READ BODY |
| a16 | Spurious FSM errors after speculative reasoning end | vLLM | PR #53046 | https://github.com/vllm-project/vllm/pull/53046 | **merged 2026-08-21** | READ BODY |
| a17 | Async scheduling + spec decode + structured outputs perf | vLLM | PR #29821 `[Perf] Async Scheduling + Speculative Decoding + Structured Outputs` | https://github.com/vllm-project/vllm/pull/29821 | **merged 2026-01-06** | READ BODY |
| a18 | MRV2 support for structured outputs + spec decode | vLLM | PR #32102, PR #33374 | https://github.com/vllm-project/vllm/pull/32102 · https://github.com/vllm-project/vllm/pull/33374 | **merged 2026-01-11 / 2026-02-02** | READ BODY |
| a19 | SGLang: outlines backend is **incompatible** with any speculative algorithm (crash: no `rollback()`, dense bool mask vs packed int32) | SGLang | Issue #24413 + PR #24499 | https://github.com/sgl-project/sglang/issues/24413 · https://github.com/sgl-project/sglang/pull/24499 | issue closed/completed 2026-07-05; PR closed, **merged_at=None** 2026-09-01 | READ BODY |
| a20 | SGLang: fix outlines grammar support in spec decode (alternative route) | SGLang | PR #24889 | https://github.com/sgl-project/sglang/pull/24889 | closed, **merged_at=None (closed-unmerged)** 2026-09-13 | READ BODY |
| a21 | SGLang DFLASH **rejected all grammar-constrained requests with HTTP 400** | SGLang | PR #30096 `[DFLASH] Support grammar-constrained decoding in speculative verify` | https://github.com/sgl-project/sglang/pull/30096 | **merged 2026-07-25** | READ BODY |
| a22 | SGLang: multi-layer EAGLE verify ignores grammar vocab mask | SGLang | Issue #31978 | https://github.com/sgl-project/sglang/issues/31978 | closed/completed 2026-07-21 | TITLE ONLY |
| a23 | SGLang: PD-disagg + EAGLE + grammar double-accept → FINISH_ABORT | SGLang | Issue #29110 | https://github.com/sgl-project/sglang/issues/29110 | closed/completed 2026-06-24 | TITLE ONLY |
| a24 | SGLang: overlap grammar FSM advance with verify (supersedes earlier attempts) | SGLang | PR #31488, PR #32110, PR #31738 | https://github.com/sgl-project/sglang/pull/31488 · https://github.com/sgl-project/sglang/pull/32110 · https://github.com/sgl-project/sglang/pull/31738 | **merged 2026-07-21 / 2026-07-23 / 2026-07-21** | READ BODY |
| a25 | SGLang UNO: grammar decoding explicitly unsupported | SGLang | `python/sglang/srt/speculative/uno_validation.py` | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/speculative/uno_validation.py | shipped in main | READ BODY |
| a26 | vLLM: `lm-format-enforcer` backend explicitly refuses spec tokens | vLLM | `vllm/v1/structured_output/backend_lm_format_enforcer.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/structured_output/backend_lm_format_enforcer.py | shipped in main | READ BODY |
| a27 | **Theory**: rejection sampling under grammar masks is NOT lossless w.r.t. the grammar-conditional distribution | Nie, Meng, Zou, Lin, Li, Zheng, Jang, Zhang | arXiv:2605.07698 | https://arxiv.org/abs/2605.07698 | preprint | READ BODY |
| a28 | Kernel-level: unified sampling + grammar masks + spec verification in one CUDA-graph-compatible kernel | Ponnusamy, Sahni, Wang, Dao | arXiv:2607.20475 SonicSampler | https://arxiv.org/abs/2607.20475 | preprint | READ BODY |

#### B. OPEN

| # | Question it answers | Who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| a29 | Grammar-aware **draft** sampling (avoid grammar-invalid drafts before verification) | vLLM contributor | PR #47885 `feat(spec_decode): Grammar-aware draft token sampling for structured outputs` | https://github.com/vllm-project/vllm/pull/47885 | **open**, created 2026-07-07 | READ BODY |
| a30 | Prototype "jump decoding" as a speculative method (`method="grammar"`) | vLLM contributor | PR #45400 `[Prototype][MRV2] Grammar spec dec (jump decoding) for structured outputs` | https://github.com/vllm-project/vllm/pull/45400 | **open** (draft), created 2026-06-12 | READ BODY |
| a31 | Grammar mask can **fail open** and emit unconstrained tokens (spec decode + async sched + PP) | user report (Peking Univ. LCPU) | Issue #54437 | https://github.com/vllm-project/vllm/issues/54437 | **open**, created 2026-08-30 | READ BODY |
| a32 | ngram_gpu + xgrammar + async scheduling: verifier accepts grammar-illegal drafts → HTTP 500 / silent truncation | user report | Issue #49694 | https://github.com/vllm-project/vllm/issues/49694 | **open** | READ BODY |
| a33 | ngram_gpu + xgrammar returns HTTP 500 under concurrent structured-output requests | user report | Issue #52620 | https://github.com/vllm-project/vllm/issues/52620 | **open**, created 2026-08-17 | TITLE ONLY |
| a34 | Engine-core livelock (100% CPU) with MTP spec decode + xgrammar; regression | user report | Issue #49210 | https://github.com/vllm-project/vllm/issues/49210 | **open**, created 2026-07-20 | TITLE ONLY |
| a35 | Crash with JSON structured output + spec decode on **guidance** backend | user report | Issue #47025 | https://github.com/vllm-project/vllm/issues/47025 | **open**, created 2026-06-29 | TITLE ONLY |
| a36 | ngram default `prompt_lookup_min=2` corrupts tool-call output with structured outputs | user report | Issue #40875 | https://github.com/vllm-project/vllm/issues/40875 | **open**, created 2026-04-25 | TITLE ONLY |
| a37 | Accepted blocks must be validated before commit (commit invariant) | vLLM contributor | PR #52452 | https://github.com/vllm-project/vllm/pull/52452 | **open** | READ BODY |
| a38 | Drive grammar masks from GPU logit counts (replacement for #52436) | vLLM contributor | PR #52477 | https://github.com/vllm-project/vllm/pull/52477 | **open** | READ BODY |
| a39 | Pre-commit grammar filter for boundary-step bonus tokens | vLLM contributor | PR #43424 | https://github.com/vllm-project/vllm/pull/43424 | **open** | READ BODY |
| a40 | Validate post-reasoning structured-output tokens in spec decode | vLLM contributor | PR #40962 | https://github.com/vllm-project/vllm/pull/40962 | **open** | READ BODY |
| a41 | SGLang xgrammar `rollback()` is O(N) list copy → O(N²) bitmask cost under spec decode | user report | Issue #38863 | https://github.com/sgl-project/sglang/issues/38863 | **open**, created 2026-09-10 | READ BODY |
| a42 | **Theory gap quantified**: TV gap up to 0.996 between local-mask speculative sampling and grammar-conditional target | authors of arXiv:2605.07698 | (same as a27) | https://arxiv.org/abs/2605.07698 | preprint | READ BODY |

#### C. ATTEMPTED-AND-ABANDONED / verbatim limitation quotes

- **vLLM, before the fix (2025-01-24), docs said the two do not work together** — PR #12373 body, verbatim: *"Update the feature compatibility matrix to reflect that speculative decoding and structured output do not currently work together."* (READ BODY, https://github.com/vllm-project/vllm/pull/12373)
- **vLLM `lm-format-enforcer` backend refuses outright** — source, verbatim: *"LM Format Enforcer backend does not support speculative tokens"* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/structured_output/backend_lm_format_enforcer.py, line ~136)
- **vLLM PR #14702 (the enabling PR), verbatim design**: *"The generated speculative draft tokens are validated according to the grammar to ensure they match the grammar constraints. This operation should not modify the matcher state."* … *"NOTE: This PR is now compatible with **both** xGrammar and Guidance backends in V1."* (READ BODY, https://github.com/vllm-project/vllm/pull/14702)
- **vLLM PR #47885 (still open), verbatim motivation**: *"When using speculative decoding with structured output constraints, the draft model frequently proposes tokens that violate grammar rules (JSON schema, regex, etc.). These tokens are **guaranteed** to be rejected by the verifier regardless of probability alignment, wasting entire speculative rounds."* (READ BODY, https://github.com/vllm-project/vllm/pull/47885)
- **vLLM Issue #54437 (open), verbatim failure mode**: *"which then writes `_full_mask` — **every token in the vocabulary allowed** — at rows that are real sampling positions."* … *"The failure is silent by construction: the mask fails **open**."* (READ BODY, https://github.com/vllm-project/vllm/issues/54437)
- **SGLang PR #30096 (merged), verbatim pre-fix behaviour**: *"Under `--speculative-algorithm DFLASH`, **any grammar-constrained request is rejected with HTTP 400**: > `DFLASH speculative decoding does not support grammar-constrained decoding yet.`"* (READ BODY, https://github.com/sgl-project/sglang/pull/30096)
- **SGLang PR #24499 (closed-unmerged), verbatim incompatibility**: *"SGLang currently accepts requests that combine speculative decoding with `--grammar-backend outlines`, but this combination crashes at runtime because: 1. `OutlinesGrammar` does not implement `rollback()`, which the speculative DFS verifier requires 2. Outlines uses a dense `torch.bool` mask, but `spec_utils.traverse_tree()` expects packed int32 bitmasks 3. The packed-bit membership check indexes `token_id // 32`, incompatible with Outlines' direct indexing"* (READ BODY, https://github.com/sgl-project/sglang/pull/24499)
- **SGLang UNO, source, verbatim**: *"UNO speculative decoding does not support grammar decoding."* (READ BODY, https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/speculative/uno_validation.py)
- **SGLang docs, verbatim**: *"Grammar decoding, returned logprobs or hidden states, sampling penalties, `min_p`, logit bias, custom logit processors, strict thinking, and deterministic inference are not yet supported."* (UNO section; READ BODY, https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/speculative_decoding.mdx)
- **THE KEY NEGATIVE RESULT (a27/a42)** — arXiv:2605.07698 abstract, verbatim: *"Grammar-constrained generation is often combined with local vocabulary masking and speculative decoding, but the resulting sampling law is not the grammar-conditional distribution users usually intend. We show that any speculative decoder with local mask access, Leviathan rejection, and rollback soundness samples from the locally projected distribution $\mu^{\mathrm{proj}}$ rather than the grammar-conditional distribution $\mu^\star$. This extends the GAD impossibility result to speculative decoding; on Dyck grammars with Qwen3-8B, the total-variation gap can reach 0.996."* … *"Because exact future validity is hard for general context-free grammars, we evaluate estimator hierarchies on tractable Dyck and finite JSON languages."* … *"All fidelity claims are scoped to enumerable grammars and token tries."* (READ BODY, https://arxiv.org/abs/2605.07698)
  - **Answer to "is rejection sampling lossless under grammar masks?"**: the paper's answer is **no** in the grammar-conditional sense — it is lossless only w.r.t. the *locally projected* distribution. Note the scope restriction in the last sentence.

---

### (b) Spec decode + TENSOR PARALLELISM / EXPERT PARALLELISM / DP ATTENTION

#### A. CLOSED

| # | Question it answers | Who closed it / who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| b1 | Spec decode + TP on the spec worker + chunked prefill did not work (H100 TP8) | user report | Issue #10276 | https://github.com/vllm-project/vllm/issues/10276 | **closed/completed** (2024; canonical) | READ BODY |
| b2 | `speculative_draft_tensor_parallel_size` could not be any value other than 1 | user report | Issue #10483 | https://github.com/vllm-project/vllm/issues/10483 | closed/completed | TITLE ONLY |
| b3 | Original feature request: draft model on a different TP size than target | vLLM community | Issue #4632; PRs #4933, #5414, #5856, #6050 | https://github.com/vllm-project/vllm/issues/4632 · https://github.com/vllm-project/vllm/pull/5856 | closed (2024) | TITLE ONLY |
| b4 | ROCm AITER MLA could not be used with EAGLE3 due to a `kernel_block_size` conflict | AMD/vLLM | PR #39616 | https://github.com/vllm-project/vllm/pull/39616 | **merged 2026-04-20** | READ BODY |
| b5 | DSpark draft silently picked a different attention backend than the pinned target (KV dtype rewritten to `fp8_ds_mla`) | vLLM | PR #52288 | https://github.com/vllm-project/vllm/pull/52288 | **merged 2026-08-15** | READ BODY |
| b6 | SGLang EAGLE/NextN + DP attention collective deadlock on draft sub-communication | user report | Issue #29144 | https://github.com/sgl-project/sglang/issues/29144 | closed/completed (label `inactive`) | TITLE ONLY |
| b7 | SGLang SPEC_V2 EAGLE accept length degraded with DP attention on GB300 | user report | Issue #16274 | https://github.com/sgl-project/sglang/issues/16274 | closed/completed | TITLE ONLY |
| b8 | SGLang DeepSeek R1/V3 error with dp-attention + NEXTN spec | user report | Issue #3943 | https://github.com/sgl-project/sglang/issues/3943 | closed/completed (2025-02) | TITLE ONLY |
| b9 | SGLang: EAGLE cuda graph + DP attention hanging (fixed) | SGLang | PR #2684 | https://github.com/sgl-project/sglang/pull/2684 | closed | TITLE ONLY |
| b10 | SGLang: fix speculative mixed batches with DP attention (GLM5) | SGLang | PR #37728 | https://github.com/sgl-project/sglang/pull/37728 | closed | TITLE ONLY |
| b11 | SGLang: spec decoding crashes with DP-Attention (fix) | SGLang | PR #33892 | https://github.com/sgl-project/sglang/pull/33892 | closed (CI labels) | TITLE ONLY |
| b12 | SGLang: NSA FA3 crash on padded speculative batches with DP attention | user report | Issue #24233 | https://github.com/sgl-project/sglang/issues/24233 | closed/completed 2026-05-01 | TITLE ONLY |
| b13 | SGLang: ImportError with GLM-5.2 + spec decoding (dp_attention API) | user report | Issue #30665 | https://github.com/sgl-project/sglang/issues/30665 | closed/completed 2026-07-09 | TITLE ONLY |
| b14 | **TP/DP divergence**: dynamic speculation K + DP → different K per rank → deadlock | vLLM, documented | `vllm/config/vllm.py` + docs | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/vllm.py · https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md | shipped in main; auto-disabled | READ BODY |

#### B. OPEN

| # | Question it answers | Who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| b15 | EAGLE3 + `tensor_parallel_size=2` + `draft_tensor_parallel_size=2` → NCCL timeout / collective deadlock | user report | Issue #44063 | https://github.com/vllm-project/vllm/issues/44063 | **open** (stale), created 2026-05-30 | TITLE ONLY |
| b16 | DP>1 + EP + DSpark spec decode fails to start (`AssertionError` in `DPMetadata.make()` during `profile_run`) | user report | Issue #56281 | https://github.com/vllm-project/vllm/issues/56281 | **open**, created 2026-09-10 | READ BODY |
| b17 | Draft and target can auto-select attention backends with **disjoint KV cache layouts** → hard init error | user report | Issue #55312 | https://github.com/vllm-project/vllm/issues/55312 | **open** | READ BODY |
| b18 | Elastic EP not supported with a draft model | vLLM source assertion | `vllm/v1/worker/gpu/eplb_utils.py`, `vllm/v1/worker/gpu_model_runner.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu/eplb_utils.py | shipped in main (assert) | READ BODY |
| b19 | SGLang: adaptive spec decoding under DP attention | SGLang contributor | PR #35898 | https://github.com/sgl-project/sglang/pull/35898 | **open**, created 2026-08-21 | TITLE ONLY |
| b20 | SGLang: dp attention support for DFLASH | SGLang contributor | PR #35321, PR #29506 | https://github.com/sgl-project/sglang/pull/35321 · https://github.com/sgl-project/sglang/pull/29506 | **open**, created 2026-08-18 / 2026-06-27 | TITLE ONLY |
| b21 | SGLang: NGRAM spec decoding with DP attention | SGLang contributor | PR #28616 | https://github.com/sgl-project/sglang/pull/28616 | **open**, created 2026-06-18 | TITLE ONLY |
| b22 | SGLang: DP-attention + spec decode crash in flashinfer MLA backend | SGLang contributor | PR #30916 | https://github.com/sgl-project/sglang/pull/30916 | **open** | TITLE ONLY |
| b23 | SGLang: `num_splits (b+1)` crash with DP-attention + spec decode (DSA) | SGLang contributor | PR #30642 | https://github.com/sgl-project/sglang/pull/30642 | **open** | TITLE ONLY |
| b24 | TensorRT-LLM: DSpark disaggregated gen-only deadlock under attention-DP | user report | Issue #17095 | https://github.com/NVIDIA/TensorRT-LLM/issues/17095 | **open**, created 2026-07-31 | TITLE ONLY |
| b25 | TensorRT-LLM: DSpark accept length collapses to ~1 at gen batch > 1 in disaggregated serving | user report | Issue #16767 | https://github.com/NVIDIA/TensorRT-LLM/issues/16767 | **open**, created 2026-07-23 | TITLE ONLY |

#### C. ATTEMPTED-AND-ABANDONED / verbatim limitation quotes

- **vLLM source, verbatim (Elastic EP + draft)**: *"Elastic EP is not supported with draft model."* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu/eplb_utils.py)
- **vLLM source, verbatim (draft TP size constraint)**: *"`speculative_draft_tensor_parallel_size=...` cannot be other value than 1 or target model tensor_parallel_size"* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py, ~line 1714)
- **vLLM source, verbatim (draft TP falls back to 1)**: *"%s cannot currently be run with tp>1; setting speculative_draft_tensor_parallel_size=1"* (READ BODY, same file, ~line 1702)
- **vLLM docs, verbatim (DP breaks dynamic K)**: *"Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. When DP is enabled, vLLM automatically disables `num_speculative_tokens_per_batch_size` and falls back to the static `num_speculative_tokens` value."* (READ BODY, https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md)
- **vLLM source, verbatim (TP communication cost of drafting)**: *"Use vocab-parallel local argmax instead of all-gathering full logits for draft token generation. Reduces communication from O(vocab_size) to O(2 * tp_size) per token."* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py, `use_local_argmax_reduction`)
- **vLLM source, verbatim (cudagraph shape must satisfy both spec-decode and TP)**: *"Can't determine cudagraph shapes that are both a multiple of {uniform_decode_query_len} (num_speculative_tokens + 1) required by spec-decode and {tensor_parallel_size} (tensor_parallel_size) required by sequence parallelism please adjust num_speculative_tokens or disable sequence parallelism"* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/compilation.py)
- **SGLang docs, verbatim (per-algorithm DP-attention bans)** — table rows for DFLASH / STANDALONE / NGRAM:
  - DFLASH: *"No `--enable-dp-attention`; `pp_size == 1`; disables overlap scheduler & mixed chunked prefill"*
  - STANDALONE: *"Does **not** support `--enable-dp-attention`"*
  - NGRAM: *"CUDA-only; no `--enable-dp-attention`; disables overlap scheduler & mixed chunked prefill"*
  (READ BODY, https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/speculative_decoding.mdx)
- **SGLang docs, verbatim**: *"Standalone speculative decoding currently **does not support** `--enable-dp-attention`."* and *"It currently **does not support** `--enable-dp-attention`."* (NGRAM)
- **SGLang source, verbatim (startup rejections)**:
  - *"Currently DFLASH speculative decoding does not support dp attention on non-NPU devices."*
  - *"Currently standalone speculative decoding does not support dp attention."* (with source comment `# TODO: support dp attention for standalone speculative decoding`)
  - *"Currently ngram speculative decoding does not support dp attention."* (with source comment `# TODO: support dp attention for ngram speculative decoding`)
  - *"DSpark with dp attention requires --enable-dp-lm-head."*
  - *"DSpark with dp attention does not support context parallel (attn_cp_size=...)."*
  (READ BODY, https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/speculative_hook.py)
- **SGLang source, verbatim (other parallelism interactions)**: *"DWDP does not support speculative decoding (MTP/draft workers)"* (https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/parallel_hook.py); *"Pipeline parallelism is not compatible with overlap schedule, speculative decoding"* (https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/validation_hook.py)
- **SGLang deployment card, verbatim (EP × DP × spec is experimental)**: *"**Spec Decode follows the Deploy knob.** Acceptance thins at large batch; spec × EP × DP-attention is validated only at 8-GPU EP8 × DP2 (full GSM8K) — experimental at these scales."* (READ BODY, https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx)
- **SGLang DeepSeek-V4 card, verbatim (DSpark incompatible with DP attention)**: *"The balanced and high-throughput recipes run target-only: they use DP Attention, which DSpark is incompatible with on current releases."* (READ BODY, https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)
- **SGLang deployment-config snippet, verbatim**: *"DFLASH speculative decoding does not support DP-Attention — the server rejects the combination at startup."* (READ BODY, https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/zai-org/glm-5.3.jsx)
- **SwiftSpec (arXiv:2506.11309) abstract, verbatim** — the canonical statement that spec decode and TP were not combined: *"Prior work on speculative decoding (which combines a small draft model with a larger target model) and tensor parallelism has each accelerated decoding. However, conventional approaches fail to apply both simultaneously due to imbalanced compute requirements (between draft and target models), KV-cache inconsistencies, and communication overheads under small-batch tensor-parallelism."* (READ BODY, https://arxiv.org/abs/2506.11309)
- **MoE/EP expert-scattering (arXiv:2607.12696) abstract, verbatim**: *"we observe that confidence-driven SD can introduce expert scattering: high-probability draft tokens may route to disjoint experts, increasing expert-weight memory traffic and reducing the speedup from speculation."* (READ BODY, https://arxiv.org/abs/2607.12696)
- **DraftExpert (arXiv:2607.24434) abstract, verbatim (expert-offload EP variant)**: *"verifying a multi-token block activates the union of target experts and is no longer close to one target step."* (READ BODY, https://arxiv.org/abs/2607.24434)

---

### (c) Spec decode + PAGED ATTENTION backends (FlashAttention / FlashInfer / Triton / MLA)

#### A. CLOSED

| # | Question it answers | Who closed it / who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| c1 | Enable FlashInfer for spec decoding (trtllm-gen decode kernel, prefill fallback) | vLLM | PR #25196 `[Spec Decode] Enable FlashInfer Spec Decoding` | https://github.com/vllm-project/vllm/pull/25196 | **merged 2025-09-24** | READ BODY |
| c2 | Enable efficient spec decode with FlashInfer-MLA (trtllm-gen spec-as-decode) | vLLM | PR #25984 | https://github.com/vllm-project/vllm/pull/25984 | **merged 2025-10-07** | READ BODY |
| c3 | cudagraph_support refactor to allow FULL graphs for spec decode with FlashInfer (TRTLLM-gen only) | vLLM | PR #28479 (revision of #26937) | https://github.com/vllm-project/vllm/pull/28479 | **merged 2025-11-12** | READ BODY |
| c4 | Improve error messages for LoRA + speculative and attention-backend conflicts | vLLM | PR #41523 | https://github.com/vllm-project/vllm/pull/41523 | closed, **merged_at=None** | READ BODY |
| c5 | AITER MLA + EAGLE3 kernel_block_size conflict on ROCm | AMD/vLLM | PR #39616 | https://github.com/vllm-project/vllm/pull/39616 | **merged 2026-04-20** | READ BODY |
| c6 | DFlash + FlashInfer + SWA + FP8 KV support | vLLM | PR #39995 | https://github.com/vllm-project/vllm/pull/39995 | closed, **merged_at=None** | READ BODY |
| c7 | GDN linear-attention backend asserted when a batch mixed plain decodes and spec decodes | user report | Issue #38196 | https://github.com/vllm-project/vllm/issues/38196 | **closed, merged_at=None** | READ BODY |
| c8 | GDN assertion failure with MTP spec decoding in CI | vLLM CI | Issue #34993 | https://github.com/vllm-project/vllm/issues/34993 | closed | READ BODY |
| c9 | ROCm AITER FA + EAGLE3 gave GSM8K ≈ 0.048 (broken accuracy) | user report | Issue #31625 | https://github.com/vllm-project/vllm/issues/31625 | closed | READ BODY |
| c10 | IMA (illegal memory access) with ngram spec decoding + flashinfer | user report | Issue #14765 | https://github.com/vllm-project/vllm/issues/14765 | closed | READ BODY |
| c11 | Qwen3-Next MTP spec decode fails with flashinfer | user report | Issue #25760 | https://github.com/vllm-project/vllm/issues/25760 | closed | READ BODY |
| c12 | SGLang: topk>1 with page_size>1 only supported on flashinfer/fa3/triton (flashmla & trtllm_mla cannot express the per-branch tree) | SGLang | `arg_groups/speculative_hook.py`; v0.5.6 also had a runtime `ValueError` | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/speculative_hook.py · https://raw.githubusercontent.com/sgl-project/sglang/v0.5.6/python/sglang/srt/server_args.py | shipped in main | READ BODY |
| c13 | SGLang: Flex Attention backend not supported with spec decoding | SGLang | `arg_groups/attention_hook.py` | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/attention_hook.py | shipped in main (assert) | READ BODY |
| c14 | SGLang: MiniCPM / Mamba-1 / hpc_ops / MiniMax-M3 backends refuse spec decoding | SGLang | backend sources | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/layers/attention/minicpm/backend.py · https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/layers/attention/attention_registry.py | shipped in main | READ BODY |

#### B. OPEN

| # | Question it answers | Who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| c15 | FlashInfer `get_cudagraph_support()` divides target head count by draft `num_kv_heads` → spec decode loses all CUDA graphs when head ratios differ | user report | Issue #55581 | https://github.com/vllm-project/vllm/issues/55581 | **open**, created 2026-09-06 | READ BODY |
| c16 | FlashInfer + spec-decode silently downgraded to PIECEWISE; measured −16% (−47.5 → 55.2 tok/s) | user report | Issue #49547 | https://github.com/vllm-project/vllm/issues/49547 | **open** | READ BODY |
| c17 | Doc fix to clarify the FlashInfer PIECEWISE downgrade warning | vLLM contributor | PR #49550 | https://github.com/vllm-project/vllm/pull/49550 | **open** | READ BODY |
| c18 | Full decode cudagraphs for spec decode on FlashInfer native path (needs flashinfer#3871) | vLLM contributor | PR #50885, PR #47979, PR #46638, PR #43200 | https://github.com/vllm-project/vllm/pull/50885 · https://github.com/vllm-project/vllm/pull/47979 | **open** | READ BODY |
| c19 | Reject unsafe FlashInfer BF16-Q + quantized KV + spec decode + SWA on SM100 (silent corruption) | vLLM contributor (upstream flashinfer#3864) | PR #47899 / Issue #47847 | https://github.com/vllm-project/vllm/pull/47899 · https://github.com/vllm-project/vllm/issues/47847 | **open** (guard not yet merged) | READ BODY |
| c20 | Drafter backend auto-selection picks FlashInfer on SM90 for a sliding-window drafter → `NotImplementedError` crash | user report | Issue #48495 | https://github.com/vllm-project/vllm/issues/48495 | **open** | READ BODY |
| c21 | DFlash SWA + FlashInfer mismatch on `next_n > 1` (SM120 sparse MLA) | vLLM contributor | PR #52499 | https://github.com/vllm-project/vllm/pull/52499 | **open** | READ BODY |
| c22 | Draft/target attention backends can have **disjoint KV cache layouts** | user report | Issue #55312 | https://github.com/vllm-project/vllm/issues/55312 | **open** | READ BODY |
| c23 | Pre-Hopper FlashInfer spec decode runs without CUDA graphs on the decode path | user report (framed as question/RFC) | Issue #54992 | https://github.com/vllm-project/vllm/issues/54992 | closed (2026-09-02), but content is an open design question | READ BODY |
| c24 | MLA + DeepSeek MTP full-cudagraph support request | user request | Issue #21505 | https://github.com/vllm-project/vllm/issues/21505 | closed | READ BODY |
| c25 | Paged-KV transactional caching under speculation (title only — publisher is Cloudflare-gated) | TechRxiv authors | TechRxiv 10.36227/techrxiv.177101038.80960856/v1 | https://www.techrxiv.org/doi/full/10.36227/techrxiv.177101038.80960856/v1 | preprint, **body NOT VERIFIED** (HTTP 403 via curl and web_fetch) | TITLE ONLY |
| c26 | Paged XQA kernel + speculative decoding | Microsoft/onnxruntime | PR #32340 | https://github.com/microsoft/onnxruntime/pull/32340 | unknown | TITLE ONLY |

#### C. ATTEMPTED-AND-ABANDONED / verbatim limitation quotes

- **vLLM Issue #49547, verbatim**: *"When speculative decoding is enabled (MTP/EAGLE, `num_speculative_tokens >= 1`) and the **FlashInfer** attention backend is selected on its native (non-trtllm-gen) decode path, vLLM silently downgrades `cudagraph_mode` from `FULL_AND_PIECEWISE` to `PIECEWISE`."* … *"measured **47.5 → 55.2 tok/s (+16%)** switching only `--attention-backend FLASHINFER → FLASH_ATTN`"* (READ BODY, https://github.com/vllm-project/vllm/issues/49547)
- **vLLM PR #49550, verbatim**: *"When speculative decoding is active and the resolved attention backend reports below `AttentionCGSupport.UNIFORM_BATCH`, `resolve_cudagraph_mode_and_sizes()` downgrades `cudagraph_mode` from `FULL_AND_PIECEWISE` to `PIECEWISE` (or `NONE`)."* (READ BODY, https://github.com/vllm-project/vllm/pull/49550)
- **vLLM PR #25984, verbatim "Known issues"**: *"The `Cutlass-MLA` backend produces incorrect output when using speculative decoding."* (READ BODY, https://github.com/vllm-project/vllm/pull/25984)
- **vLLM Issue #39616 (AITER MLA), verbatim**: *"AITER MLA is the fastest MLA attention backend on AMD MI300X/MI355X GPUs … However, it currently cannot be used with speculative decoding methods like Eagle3 due to a `kernel_block_size` conflict: - **AITER MLA** declares `get_supported_kernel_block_sizes() = [1]`"* (READ BODY, https://github.com/vllm-project/vllm/pull/39616)
- **vLLM Issue #21505, verbatim (MLA full-cudagraph restriction)**: *"This method builds the metadata for full cudagraph capture. Currently, only decode is supported for full cudagraphs with MLA. … MLA only supports decode-only full CUDAGraph capture."* (READ BODY, https://github.com/vllm-project/vllm/issues/21505)
- **vLLM Issue #47847, verbatim**: *"silently corrupted output — sporadic wrong tokens and short repetition loops in otherwise coherent generations"* for the combination FlashInfer + fp8 KV + `disable_flashinfer_q_quantization` + spec decode + SWA on SM100. (READ BODY, https://github.com/vllm-project/vllm/issues/47847)
- **vLLM Issue #55312, verbatim**: *"The KV cache layout intersection is only computed afterwards, per worker … and an empty intersection is a hard error at engine init. Nothing constrains the draft's choice to layouts the target can also satisfy."* (READ BODY, https://github.com/vllm-project/vllm/issues/55312)
- **SGLang source, verbatim (tree verification needs the right backend)**: *"topk > 1 + page_size > 1 needs the two-pass cascade draft-decode (shared prefix pass + per-branch expand pass with prefix-tail dup). Only these backends implement it; flashmla / trtllm_mla can't express the per-branch tree, so reject."* and *"speculative_eagle_topk > 1 with page_size > 1 is only supported on ('flashinfer', 'fa3', 'triton')"* (READ BODY, https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/speculative_hook.py)
- **SGLang source, verbatim**: *"Speculative decoding is currently not supported with Flex Attention backend"* (READ BODY, https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/attention_hook.py)
- **SGLang source, verbatim (older release)**: *"speculative_eagle_topk({...}) > 1 with page_size({...}) > 1 is unstable and produces incorrect results for paged attention backends. This combination is only supported for the 'flashinfer' backend."* (READ BODY, https://raw.githubusercontent.com/sgl-project/sglang/v0.5.6/python/sglang/srt/server_args.py)
- **SGLang sources, verbatim (backend refusals)**: *"MiniCPM backend does not support speculative decoding (target verify)"*; *"Mamba-1 (Falcon-Mamba) does not support speculative decoding yet"*; *"The hpc_ops attention backend does not support speculative ..."*; *"MiniMax-M3 MSA attention does not support speculative decoding under ..."* (READ BODY, files listed in c14)
- **LongSpec (arXiv:2502.17421) abstract, verbatim on tree-attention cost**: *"inefficiencies in tree attention mechanisms when managing long token sequences"*; its fix is *"an attention aggregation strategy that combines fast prefix computation with standard tree attention to enable efficient decoding."* (READ BODY, https://arxiv.org/abs/2502.17421)
- **SonicSampler (arXiv:2607.20475) abstract, verbatim on the CUDA-graph incompatibility of existing samplers**: *"existing implementations either accelerate only subsets of this pipeline, rely on multiple kernel launches, or assume homogeneous sampling behavior across a batch, limiting support for dynamic serving workloads and preventing efficient CUDA Graph execution."* Its result: kernels supporting *"grammar-constrained decoding, repetition, frequency and presence penalties, logit bias, temperature scaling, top-k / top-p / min-p filtering, and speculative verification - within a single batched kernel while remaining fully CUDA Graph-compatible."* (READ BODY, https://arxiv.org/abs/2607.20475)

---

### (d) Spec decode + CUDA GRAPHS

#### A. CLOSED

| # | Question it answers | Who closed it / who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| d1 | vLLM design doc for the new multi-mode CUDA Graph scheme (incl. spec-decode "uniform decode" definition) | vLLM | `docs/design/cuda_graphs.md` | https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md | shipped in main | READ BODY |
| d2 | Fix spec decode with cuda graph (old, canonical) | vLLM | PR #7090 | https://github.com/vllm-project/vllm/pull/7090 | closed, **merged_at=None** | TITLE ONLY |
| d3 | Enable Eagle drafter support for FULL CUDA Graph mode | vLLM | PR #34880 | https://github.com/vllm-project/vllm/pull/34880 | closed, **merged_at=None** | TITLE ONLY |
| d4 | Enable full CUDA graphs for spec decoding with FlashInfer (first attempt) | vLLM | PR #26937 | https://github.com/vllm-project/vllm/pull/26937 | closed, **merged_at=None** (superseded by merged #28479) | READ BODY |
| d5 | cudagraph_support refactor, per-backend per-kv-group | vLLM | PR #28479 | https://github.com/vllm-project/vllm/pull/28479 | **merged 2025-11-12** | READ BODY |
| d6 | Keep default CUDA graph sizes memory-safe for spec decode | vLLM | PR #54418 | https://github.com/vllm-project/vllm/pull/54418 | **merged 2026-08-30** | TITLE ONLY |
| d7 | Fuse AR speculator multi-step decodes back into one CUDA graph (MRV2) | vLLM | PR #46849 | https://github.com/vllm-project/vllm/pull/46849 | **merged 2026-08-11** | TITLE ONLY |
| d8 | ROCm gfx942 GPU memory access fault with MTP spec-decode + sparse-MLA decode under cuda graph | user report | Issue #47196 | https://github.com/vllm-project/vllm/issues/47196 | closed/completed 2026-06-30 | TITLE ONLY |
| d9 | Enable full CUDA graphs on padded DSpark | vLLM | PR #47513 | https://github.com/vllm-project/vllm/pull/47513 | closed, **merged_at=None** | TITLE ONLY |
| d10 | SGLang: keep speculative overshoot out of the radix cache key (touches graph keying) | SGLang | PR #35694 | https://github.com/sgl-project/sglang/pull/35694 | **open** | TITLE ONLY |

#### B. OPEN

| # | Question it answers | Who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| d11 | Padded drafter reuses CUDA-graph output buffers → `prompt_logprobs` corruption | vLLM contributor | PR #53520 | https://github.com/vllm-project/vllm/pull/53520 | **open** | TITLE ONLY |
| d12 | `num_speculative_tokens_per_batch_size` + MTP speculator fails full CUDA-graph decode capture | user report | Issue #48494 | https://github.com/vllm-project/vllm/issues/48494 | **open**, created 2026-07-13 | TITLE ONLY |
| d13 | CUDA graph rejection sampling (WIP, MRV2) | vLLM contributor | PR #43924 | https://github.com/vllm-project/vllm/pull/43924 | **open** WIP | TITLE ONLY |
| d14 | EAGLE prefill FULL CUDA graph drops mm_inputs for live multimodal batches | vLLM | Issue #43716 / PR #43776 | https://github.com/vllm-project/vllm/issues/43716 · https://github.com/vllm-project/vllm/pull/43776 | **open** | TITLE ONLY |
| d15 | FSDP/`query_start_loc.tolist()` crash: TurboQuant KV + spec-decode + chunked-prefill during CUDA graph capture | user report | Issue #40807 / PR #43747 | https://github.com/vllm-project/vllm/issues/40807 · https://github.com/vllm-project/vllm/pull/43747 | **open** | TITLE ONLY |
| d16 | Draft KV groups judged with the wrong model's config for CUDA-graph support (FlashInfer) | vLLM contributor | PR #55593 | https://github.com/vllm-project/vllm/pull/55593 | **open**, created 2026-09-06 | TITLE ONLY |
| d17 | FlashInfer native spec decode has no FULL decode graph (see c15–c18) | vLLM | Issue #49547, PR #50885 | https://github.com/vllm-project/vllm/issues/49547 | **open** | READ BODY |

#### C. ATTEMPTED-AND-ABANDONED / verbatim limitation quotes

- **vLLM `docs/design/cuda_graphs.md`, verbatim (graphs are keyed on K)**: *"In this document, we refer to pure decode (`max_query_len=1`) or speculative decode (`max_query_len =1+num_spec_tokens`) as **uniform decode** batches, and the opposite would be **non-uniform** batches"* … *"pure decode batches are uniform but may not be query length 1 (i.e. `num_tokens == num_reqs`), this occurs in the validation pass of spec-decode where "decode" batches will have a query length of `1+num_spec_tokens`."* (READ BODY)
- **vLLM `docs/design/cuda_graphs.md`, verbatim (backend capability enum)**: *"`UNIFORM_BATCH = 2` — CUDA Graphs supported for batches that only contain query lengths that are the same, this can be used for spec-decode i.e. "decodes" are 1 + num_speculative_tokens"*; *"Many attention backends also weren't ready for unified "full" CUDA Graphs capture (e.g., only FlashAttention 3 supports it currently) or only support CUDA Graphs for pure decode batches (e.g., Flashinfer, FlashMLA, and Mamba, etc.)."* (READ BODY)
- **vLLM `docs/design/cuda_graphs.md`, verbatim (backend table)**: `FlashInfer` → `UNIFORM_SINGLE_TOKEN_DECODE`; `AITER MLA` → `UNIFORM_SINGLE_TOKEN_DECODE`; `CUTLASS MLA` → `UNIFORM_SINGLE_TOKEN_DECODE`; `Mamba attention` → `UNIFORM_SINGLE_TOKEN_DECODE`; *"Unlisted backends are all declared as `NEVER`."* (READ BODY)
- **vLLM `vllm/config/compilation.py`, verbatim (spec-decode forces cudagraph size multiples of K+1)** — full function docstring/body quoted at https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/compilation.py: `adjust_cudagraph_sizes_for_spec_decode()` does `multiple_of = uniform_decode_query_len` and rounds every capture size up to a multiple of it; and the guard *"CUDAGraphMode.{name} is not supported with spec-decode for attention backend {min_cg_attn_backend} (support: {min_cg_support})"* followed by *"setting cudagraph_mode=PIECEWISE"* or `NONE`. (READ BODY)
- **vLLM `vllm/v1/worker/gpu_model_runner.py`, verbatim (drafter graph restriction)**: *"Eagle currently only supports PIECEWISE cudagraphs. Therefore only use cudagraphs if the main model uses PIECEWISE NOTE(lucas): this is a hack, need to clean up."* (READ BODY)
- **vLLM `vllm/v1/worker/gpu_model_runner.py`, verbatim (drafter vs padded metadata)**: *"Currently the drafter still only uses piecewise cudagraphs (and modifies the attention metadata in directly), and therefore does not want to use padded attention metadata."* (READ BODY)
- **vLLM Issue #49547, verbatim** — the cudagraph downgrade is *"emitted as a single `logger.warning` that names neither the performance cost nor the actionable alternative, so it is very easy to miss and to mistake for a benign informational line."* (READ BODY)
- **vLLM Issue #54992, verbatim**: *"On sm_8x, combining the FlashInfer backend with speculative decoding downgrades the model to `PIECEWISE` cudagraphs and leaves the draft model eager. FlashAttention on the same GPU keeps `FULL` graphs for both."* and the warning text quoted in that issue: *"CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend (support: AttentionCG..."* (READ BODY)
- **vLLM Issue #21505 (closed), verbatim**: *"MLA only supports decode-only full CUDAGraph capture."* — i.e. DeepSeek MTP multi-token verification could not be full-graph captured at that time. (READ BODY)
- **vLLM PR #50885 (open), verbatim** on why the graph key must change once q_len > 1: *"keys CUDA-graph decode wrappers by `(batch_size, q_len_per_req)`. This separation is required once native decode graphs accept multi-token queries, because FlashInfer freezes both values at plan time."* (READ BODY)
- **vLLM Issue #48494 (open, TITLE ONLY)**: *"[Bug][Spec Decode] num_speculative_tokens_per_batch_size + MTP speculator fails full CUDA graph decode capture (InputBatch.make_du..."* — the *dynamic-K-breaks-graph-capture* case; body NOT VERIFIED.
- **vLLM Issue #27325 (closed/not_planned, stale), TITLE ONLY**: *"[Bug]: When spec_tokens count is greater than 1, the adaptation of cuda_graph_sizes causes the decoding process to fall back to ea[ger]"* — https://github.com/vllm-project/vllm/issues/27325. **Closure reason NOT VERIFIED** (would have been in a comment thread; comment API was rate-limited).
- **vLLM Issue #49488 (closed/not_planned), TITLE ONLY**: *"[RFC] Persistent, in-place drafting attention metadata for full-CUDA-graph MTP/spec-decode"* — https://github.com/vllm-project/vllm/issues/49488. **Maintainer rejection wording NOT VERIFIED** (comment thread unavailable).
- **arXiv:2609.02897 (AdaptiveSpec) abstract, verbatim (dynamic K in production)**: *"A per-step tree policy adjusts the draft tree's depth, width, and node count directly from a fused signal of draft top-1 confidence and a rolling acceptance history … allowing the total draft count to vary rather than only be redistributed."* (READ BODY, https://arxiv.org/abs/2609.02897) — this is exactly the class of dynamic-K behaviour that conflicts with K-keyed graph capture.
- **arXiv:2607.06763 (Weaver) abstract, verbatim (rollback-free tree verification for a linear-attention model)**: *"To support fast verification for models with Gated Delta Net layers, we derive a rollback-free tree-verification algorithm and implement optimized CUDA kernels in SGLang."* (READ BODY, https://arxiv.org/abs/2607.06763)

---

### (e) Spec decode + CHUNKED PREFILL / mixed batching

#### A. CLOSED

| # | Question it answers | Who closed it / who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| e1 | Original feature request to combine chunked prefill with spec decode | vLLM community (cc @LiuXiaoxuanPKU @comaniac @rkooo567) | Issue #5016 | https://github.com/vllm-project/vllm/issues/5016 | **closed/completed** | READ BODY |
| e2 | The PR that implemented it | vLLM | PR #9291 | https://github.com/vllm-project/vllm/pull/9291 | **merged 2024-11-07** | READ BODY |
| e3 | IndexError on chunked prefill + EAGLE3 + TP8 (Llama-3.3-70B, 8×H200) | user report | Issue #20531 | https://github.com/vllm-project/vllm/issues/20531 | **closed/not_planned (stale)** 2026-07-06-era | READ BODY |
| e4 | Keep intermediate prefill chunks mamba-block-aligned with spec decode | vLLM | PR #45477 | https://github.com/vllm-project/vllm/pull/45477 | closed | TITLE ONLY |
| e5 | Keep mamba align prefill chunks block-aligned past `last_cache_position` | vLLM | PR #51113 | https://github.com/vllm-project/vllm/pull/51113 | unknown | TITLE ONLY |
| e6 | SGLang: enable target-only mixed chunked prefill for DSpark | SGLang | PR #35300 | https://github.com/sgl-project/sglang/pull/35300 | closed | TITLE ONLY |
| e7 | SGLang: **mixed chunked prefill is disabled** for EAGLE-family when the algorithm doesn't implement it | SGLang (long-standing, since ≤v0.4.9) | `server_args.py` (v0.4.9 → v0.5.7) and `arg_groups/speculative_hook.py` (main) | https://raw.githubusercontent.com/sgl-project/sglang/v0.5.6/python/sglang/srt/server_args.py · https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/speculative_hook.py | shipped | READ BODY |
| e8 | SGLang: NGRAM spec decoding **unconditionally** disables mixed chunked prefill | SGLang | same files | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/speculative_hook.py | shipped | READ BODY |
| e9 | GDN backend asserted on batches mixing plain decode and spec decode (a mixed-batch failure) | user report | Issue #38196 | https://github.com/vllm-project/vllm/issues/38196 | **closed, merged_at=None** | READ BODY |
| e10 | GDN assertion with MTP in CI (`num_decodes: 1, num_spec_decodes: 45`) | vLLM CI | Issue #34993 | https://github.com/vllm-project/vllm/issues/34993 | closed | READ BODY |

#### B. OPEN

| # | Question it answers | Who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| e11 | `prompt_logprobs` silently corrupted with MTP + Qwen3.5-family + chunked prefill (two builds, two checkpoints) | user report | Issue #53488 | https://github.com/vllm-project/vllm/issues/53488 | **open**, created 2026-08-23 | READ BODY |
| e12 | TurboQuant KV + spec-decode + chunked-prefill crashes CUDA graph capture | user report | Issue #40807 / PR #43747 | https://github.com/vllm-project/vllm/issues/40807 | **open** | TITLE ONLY |
| e13 | KpoolTailSpec excluded from generic slot-mapping kernel (chunked-prefill reads past block-table row) | vLLM contributor | PR #56469 (addresses #56380) | https://github.com/vllm-project/vllm/pull/56469 | **open** | READ BODY |
| e14 | Route every speculation-capable row through the speculative path (hybrid recurrent state) | vLLM contributor | PR #56531 | https://github.com/vllm-project/vllm/pull/56531 | **open** | READ BODY |

#### C. ATTEMPTED-AND-ABANDONED / verbatim limitation quotes

- **SGLang source (main), verbatim**:
  - *"Mixed chunked prefill is disabled: %s speculative decoding does not support it."*
  - *"Mixed chunked prefill is disabled because of using Frozen-KV MTP speculative decoding."*
  - *"Mixed chunked prefill is disabled for UNO speculative decoding."*
  - and the `supports_mixed_chunk()` docstring: *"Whether mixed chunk prefill may stay enabled with this algorithm. ngram cannot join as is: its overlap relay skips output_tokens_buf, which the mixed input resolve reads."*
  (READ BODY, https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/speculative_hook.py · https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/speculative/spec_info.py)
- **SGLang source (v0.5.6), verbatim**:
  - *"Mixed chunked prefill is disabled because of using eagle speculative decoding."*
  - *"The overlap scheduler and mixed chunked prefill are disabled because of using ngram speculative decoding."*
  - *"Overlap scheduler is disabled because of using eagle3 or standalone speculative decoding."*
  - *"Pipeline parallelism is not compatible with overlap schedule, speculative decoding, mixed chunked prefill."*
  (READ BODY, https://raw.githubusercontent.com/sgl-project/sglang/v0.5.6/python/sglang/srt/server_args.py)
- **SGLang docs (main), verbatim** — same rules carried into user docs: DFLASH/NGRAM rows say *"disables overlap scheduler & mixed chunked prefill"*; UNO says *"Mixed chunked prefill is disabled for UNO."*; NGRAM notes say *"It disables the overlap scheduler and mixed chunked prefill."* (READ BODY, https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/speculative_decoding.mdx)
- **Change history (verified by grepping tagged releases)**: the EAGLE and NGRAM strings are present in `server_args.py` at v0.4.9, v0.4.10.post2, v0.5.1–v0.5.7; the NGRAM/overlap wording first appears at v0.5.3. On `main` the file was refactored and the logic moved to `arg_groups/speculative_hook.py`. (READ BODY)
- **vLLM Issue #20531 (closed/not_planned), verbatim from the report**: *"IndexError: list index out of range on chunked prefill with speculative decoding"*, config `eagle3`, `num_spec_tokens=5`, `tensor_parallel_size=8`, and *"Prefix caching and chunked prefill are enabled by default V1 behavior. This issue also occurred in vLLM v0.8.5.post1 and has been hard to reproduce."* (READ BODY)
- **vLLM PR #56531 (open), verbatim on the mixed-batch state hazard**: *"The two layouts coincide only while `num_accepted_tokens == 1`, i.e. slot `0`. The moment a request accepts a draft token its live state sits at slot `k > 0`, and anything reading slot `0` gets a stale state - the one from before the accepted tokens were applied. Nothing detects this: the read is in bounds, the shapes match, and generation continues from a state that silently lost a few tokens of history."* (READ BODY)
- **vLLM Issue #53488 (open), verbatim**: *"the `prompt_logprobs` returned by `/v1/completions` are **wildly wrong for a subset of requests** — with no error, no warning, HTTP 200 — while the identical server **without** `--speculative-config` scores every request correctly."* (READ BODY)

---

### (f) Spec decode + PREFIX CACHING / radix cache / KV reuse

#### A. CLOSED

| # | Question it answers | Who closed it / who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| f1 | vLLM Automatic Prefix Caching **design doc** (the requested document) | vLLM | `docs/design/prefix_caching.md` | https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md | shipped in main | READ BODY |
| f2 | vLLM APC user doc + its spec-decode-specific section | vLLM | `docs/features/automatic_prefix_caching.md` | https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md | shipped in main | READ BODY |
| f3 | Early fix: prefix-caching logic for running requests without speculative tokens | vLLM | PR #19006 | https://github.com/vllm-project/vllm/pull/19006 | closed, **merged_at=None** | TITLE ONLY |
| f4 | Improve prefix caching logic in spec decoding | vLLM | PR #18668 | https://github.com/vllm-project/vllm/pull/18668 | **merged 2025-05-25** | TITLE ONLY |
| f5 | Long-standing question: "Can speculative decoding and prefix caching take effect simultaneously?" | user; closed **not_planned/stale** | Issue #22230 | https://github.com/vllm-project/vllm/issues/22230 | **closed/not_planned (stale) 2025-12-04** | READ BODY (body near-empty) |
| f6 | Earlier request for API control over spec decode + prefix caching | user; closed not_planned/stale | Issue #7569 | https://github.com/vllm-project/vllm/issues/7569 | **closed/not_planned (stale)** | TITLE ONLY |
| f7 | **EAGLE/MTP must drop the last matched prefix-cache block** (design constraint, in source) | vLLM | `vllm/v1/core/single_type_kv_cache_manager.py`, `vllm/config/speculative.py`, `vllm/v1/core/sched/scheduler.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/single_type_kv_cache_manager.py | shipped in main | READ BODY |
| f8 | ~20% accuracy drop with `--enable-prefix-caching` + MTP (Qwen3.6 35B-A3B) | user report; closed completed | Issue #43559 | https://github.com/vllm-project/vllm/issues/43559 | **closed/completed 2026-08-06** | READ BODY |
| f9 | Mamba prefix caching + MTP crashes at startup (NemotronH) | user report; closed completed | Issue #39809 | https://github.com/vllm-project/vllm/issues/39809 | **closed/completed** | TITLE ONLY |
| f10 | EngineCore crash in Mamba align-mode prefix caching with spec decoding | vLLM | PR #41898 | https://github.com/vllm-project/vllm/pull/41898 | closed, **merged_at=None** | TITLE ONLY |
| f11 | Mamba-align corner cases acknowledged by PR authors | quoted inside f12 | PR #30877, PR #33726 (as quoted in Issue #41884) | https://github.com/vllm-project/vllm/issues/41884 | — | READ BODY (via quote) |
| f12 | `IndexError` in `get_temporal_copy_spec` with DFlash spec decode + prefix caching | user report; closed **not_planned/stale** | Issue #41884 | https://github.com/vllm-project/vllm/issues/41884 | **closed/not_planned (stale) 2026-09-07** | READ BODY |
| f13 | XPU crash with spec decode + prefix caching in Mamba models (fix) | vLLM/Intel | PR #52186 | https://github.com/vllm-project/vllm/pull/52186 | closed, **merged_at=None** | TITLE ONLY |
| f14 | SGLang: EAGLE V2 crashes with NaN in logits when radix-cache prefix hit occurs (SM120) | user report | Issue #19796 | https://github.com/sgl-project/sglang/issues/19796 | closed/completed 2026-03-03 | TITLE ONLY |
| f15 | SGLang: PD-disagg **decode-side radix cache is incompatible with speculative decoding** | SGLang | `arg_groups/fields/disagg.py` | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/fields/disagg.py | shipped in main | READ BODY |
| f16 | SGLang: NGRAM/DFLASH radix-cache-key pollution fixed | SGLang | PR #22908 (AMD), PR #35694 | https://github.com/sgl-project/sglang/pull/22908 · https://github.com/sgl-project/sglang/pull/35694 | #22908 closed; #35694 **open** | TITLE ONLY |
| f17 | SGLang: HiCache draft state not supported for Inkling MTP | SGLang | `speculative/base_spec_worker.py` | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/speculative/base_spec_worker.py | shipped in main | READ BODY |
| f18 | vLLM: Mamba fine-grained prefix cache specifically *for* EAGLE/MTP on the Mamba group | vLLM | `docs/features/automatic_prefix_caching.md` | https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md | shipped in main | READ BODY |
| f19 | **Answered**: can a draft model reuse the target's KV cache? | Liu et al. | arXiv:2604.26412 KVShot | https://arxiv.org/abs/2604.26412 | preprint (v2 2026-05-09) | READ BODY |
| f20 | Long-context draft with constant-size draft KV + tree attention aggregation | Yang et al. | arXiv:2502.17421 LongSpec | https://arxiv.org/abs/2502.17421 | preprint (v4 2026-04-08) | READ BODY |

#### B. OPEN

| # | Question it answers | Who is asking | Artifact | URL | Status | Read |
|---|---|---|---|---|---|---|
| f21 | **[PD + SpecDec] prefix-cache trimming drops the wrong block when P has an extra lookahead block** | user report | Issue #43996 | https://github.com/vllm-project/vllm/issues/43996 | **open** | TITLE ONLY |
| f22 | GDN `mamba_get_block_table_tensor` gather index OOB with prefix caching + `num_speculative_tokens>0` | user report | Issue #42084 | https://github.com/vllm-project/vllm/issues/42084 | **open** (stale label) | TITLE ONLY |
| f23 | Mamba `align`-mode prefix caching + MTP on V1 (Qwen3-Next GDN) — feature work | vLLM contributor | PR #50172, PR #54637, PR #56466 | https://github.com/vllm-project/vllm/pull/50172 · https://github.com/vllm-project/vllm/pull/54637 · https://github.com/vllm-project/vllm/pull/56466 | **open** (54637 & 56466 WIP) | READ BODY (50172/54637/56466 summaries) |
| f24 | FlashInfer ReplaySSM prefix-cache lifecycle for STP/MTP | vLLM contributor | PR #55688 | https://github.com/vllm-project/vllm/pull/55688 | **open** | READ BODY |
| f25 | Preserve EAGLE SWA replay blocks for external APC (PD disaggregation) | vLLM contributor | PR #56449 | https://github.com/vllm-project/vllm/pull/56449 | **open** | READ BODY |
| f26 | **Measured**: EAGLE/MTP last-block drop costs a 1,648-token recompute per hit → ~30–40% batch throughput loss | user report | Issue #53670 | https://github.com/vllm-project/vllm/issues/53670 | **open** | READ BODY |
| f27 | EAGLE/MTP block drop + prefix caching untested for hybrid models with ≥3 attention groups | user report | Issue #51771 | https://github.com/vllm-project/vllm/issues/51771 | **open** | READ BODY |
| f28 | Misleading warning when `disable_eagle_block_drop` is on but the warning says reuse is disabled | user report + fix | Issue #55518 / PR #55519 | https://github.com/vllm-project/vllm/issues/55518 · https://github.com/vllm-project/vllm/pull/55519 | **both open** | READ BODY |
| f29 | SGLang: allow decode radix cache with speculative decoding (fix, not merged) | SGLang contributor | PR #32170 | https://github.com/sgl-project/sglang/pull/32170 | **open** | TITLE ONLY |
| f30 | SGLang: keep speculative overshoot out of the radix cache key | SGLang contributor | PR #35694 | https://github.com/sgl-project/sglang/pull/35694 | **open** | TITLE ONLY |

#### C. ATTEMPTED-AND-ABANDONED / verbatim limitation quotes

- **vLLM source, verbatim (the core incompatibility)** — `find_longest_cache_hit` docstring: *"If eagle is enabled, drop the last matched block to force recompute the last block to get the required hidden states for eagle drafting head. For multi-module MTP, this recompute also rewrites the dropped block's ..."* and inline: *"Eagle needs the tokens right before the generation point recomputed: drop one hash unit when fine-grained (the tail block's KV is append-only, so it still covers the reduced length), else one cache block."* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/single_type_kv_cache_manager.py)
- **vLLM source, verbatim (opt-out flag is experimental)**: *"Disable dropping the trailing prefix-cache block for EAGLE-like speculative methods. This is an experimental option for measuring the acceptance-rate impact of reusing that block. It does not disable the speculative drafter itself."* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py)
- **vLLM source, verbatim (runtime warning when the drop is off)**: *"EAGLE trailing prefix-cache block dropping is disabled. This is experimental and may affect speculative-token acceptance rates."* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py)
- **vLLM source, verbatim (hybrid KV + EAGLE + chunked local attention)**: *"Hybrid KV cache is not supported for eagle + chunked local attention."* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/single_type_kv_cache_manager.py)
- **vLLM model sources, verbatim (MTP + `mamba_cache_mode=all` prefix caching)**: *"Qwen3NextMTP currently does not support 'all' prefix caching, ..."*, *"Qwen3_5MTP currently does not support 'all' prefix caching, ..."*, *"Qwen4ExpMTP currently does not support 'all' prefix caching, ..."* (READ BODY, https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next_mtp.py · .../qwen3_5_mtp.py · https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/qwen4_exp/nvidia/mtp.py)
- **vLLM docs (APC), verbatim** — the fine-grained Mamba prefix cache exists specifically because of spec decode, and its conditions are: *"`--mamba-cache-mode align` / EAGLE/MTP speculative decoding on the Mamba group / `--prefix-match-unit` smaller than the Mamba block size / the model does not use multi-module MTP"* (READ BODY, https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md)
- **vLLM Issue #41884 (closed not_planned), verbatim quote of the *upstream PR authors'* acknowledgement (quoted in the issue body)**: PR #30877 — *"Speculative decoding is temporarily disabled in this PR as there are still corner-case bugs when using with prefix-caching in align mode."*; PR #33726 — *"Prefix Caching (old style, have not tested 'align' mode)"*. Repro context: *"Prefix cache hit rate was 68.6% at time of crash"*; workaround *"Pass `--no-enable-prefix-caching` (avoids the align-mode block copy logic)"*. (READ BODY, https://github.com/vllm-project/vllm/issues/41884)
- **vLLM Issue #53670 (open), verbatim**: *"EAGLE/MTP prefix-cache last-block drop causes a 1,648-token recompute per hit on one hybrid Qwen3.8 GDN layout — ~30-40% batch throughput loss on prefix-reusing workloads with speculative decoding enabled"*; config includes *"scheduler/hash block = 1648 tokens"*, *"Dual RTX 5090 TP=2, FP8 weights + FP8 KV, prefix caching on"*. (READ BODY)
- **vLLM Issue #51771 (open), verbatim quote of PR #33524's author, quoted in the issue body**: *"However, it is worth noting that for more complicated models with multiple attention groups, this PR does not fully address the EAGLE spiral block drop issue either. A general fix to this issue cannot directly cache the hit_blocks list returned by each attention type, because SWA attn and Mamba-style attn do not follow the downward-closed property (cache hit at token j does not indicate cache hit at i where i < j). So we need some more fundamental changes there."* … *"Fortunately, we don't have such complex models yet, so this is not a huge issue for now."* (READ BODY, https://github.com/vllm-project/vllm/issues/51771)
- **vLLM PR #55519 (open), verbatim measured regression and the flag that restores it**: *"with `{"method":"mtp","num_speculative_tokens":1}` the run scores 0 prefix-cache hits — consistent with the expected trailing-block drop (#38182), which on a 1056-token block consumes the whole of a shared prefix shorter than two blocks. Adding `"disable_eagle_block_drop": true` restores 22,176 hits out of 72,992 queried, the same as the no-speculation baseline, and 53,152 with `--prefix-match-unit 352`, again matching the no-speculation …"* (READ BODY, https://github.com/vllm-project/vllm/pull/55519)
- **vLLM PR #56449 (open), verbatim (EAGLE + external APC)** — describes the concrete failure: *"In the 64k shared-prefix Decode case, Prefill-side local APC was hot and Decode-side external prefix cache reported full hits, but Decode-side local APC did not grow at all."* and the cause: *"sliding-window groups may receive or keep only the local attention tail after external KV loading, while EAGLE local prefix-cache lookup needs one extra lookahead block"*. (READ BODY, https://github.com/vllm-project/vllm/pull/56449)
- **vLLM PR #55688 (open), verbatim (why prefix reuse is hard with ReplaySSM + MTP)**: *"FlashInfer ReplaySSM keeps its pending `x`, `dt`, and `B` history in separate ring buffers. Prefix reuse therefore requires more than copying canonical Mamba state: the live rings and their shared cursors must migrate with a block, and pending history must be materialized into a canonical checkpoint before publishing a reusable cache entry."* (READ BODY, https://github.com/vllm-project/vllm/pull/55688)
- **SGLang source, verbatim (PD-disagg decode radix cache)**: *"Enable radix cache on decode server (PD mode). Caches KV prefixes to avoid redundant transfers. Incompatible with --enable-hisparse, speculative decoding, and --disaggregation-transfer-backend fake."* (READ BODY, https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/fields/disagg.py · mirrored in https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/server_arguments.mdx)
- **SGLang source, verbatim (HiCache)**: *"HiCache does not support Inkling MTP draft state yet."* (READ BODY, https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/speculative/base_spec_worker.py)
- **SGLang docs, verbatim (mamba extra_buffer_lazy + spec)**: *"Compatible with speculative decoding (EAGLE/NGRAM/DSPARK/DFLASH); not supported under PD disaggregation."* and *"Must be divisible by page_size if set, and must be >= speculative_num_draft_tokens when using speculative decoding."* (READ BODY, https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/server_arguments.mdx)
- **KVShot (arXiv:2604.26412) abstract, verbatim negative result**: *"Extensive evaluations on Qwen3-8B show that KV-Reuse improves long-range acceptance, although end-to-end speedups remain marginal under current training pipelines."* … *"These findings suggest that realizing the full potential of KV-aware decoding requires moving beyond TTT toward block-wise training paradigms."* (READ BODY, https://arxiv.org/abs/2604.26412)
- **LongSpec (arXiv:2502.17421) abstract, verbatim motivation**: *"(1) the excessive memory demands posed by draft models due to large Key-Value (KV) cache; (2) performance degradation resulting from the mismatch between short-context training and long-context inference; and (3) inefficiencies in tree attention mechanisms when managing long token sequences."* (READ BODY)

---

### Cross-cutting: the vLLM docs "Known Feature Incompatibility" section (as requested)

Source: `docs/features/speculative_decoding/README.md` on `main`, retrieved verbatim. **The section contains exactly two items** and does **not** mention structured output, TP/EP, paged attention, CUDA graphs, chunked prefill, or prefix caching:

> ## Known Feature Incompatibility
>
> 1. Pipeline parallelism is not composable with speculative decoding as of `vllm<=0.15.0`
> 2. Speculative decoding with draft models is not supported in `vllm<=0.10.0`

https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md (READ BODY). Mirrored on the docs site: https://docs.vllm.ai/en/latest/features/speculative_decoding/ (TITLE ONLY — site not fetched).

The same README also carries the losslessness claims, verbatim: *"**Theoretical Losslessness** - Speculative decoding sampling is theoretically lossless up to the precision limits of hardware numerics."* … *"**Algorithmic Losslessness** - vLLM's implementation of speculative decoding is algorithmically validated to be lossless."* … and the caveat *"**Batch Size and Numerical Stability**: Changes in batch size may cause variations in logprobs and output probabilities"* (READ BODY).

### Cross-cutting: vLLM CUDA Graphs design doc (as requested)

https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md — retrieved and read in full; the spec-decode-relevant content is quoted in section (d) C above (`BatchDescriptor`, `AttentionCGSupport`, the `cudagraph_mode` enum, the backend support table, and `_check_and_update_cudagraph_mode` fallback policy). Key verbatim: *"`FULL_AND_PIECEWISE` — (default mode) full CUDA Graph for uniform decode, piecewise CUDA Graphs for others"*; *"we might resolve the incompatible CUDA Graphs mode by downgrading the mode to the best fit one."*

### Cross-cutting: vLLM Automatic Prefix Caching design doc (as requested)

https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md — retrieved and read. **Contains zero occurrences of "EAGLE", "speculative", or "spec", and no "drop" language.** The EAGLE block-drop semantics live only in source (`single_type_kv_cache_manager.py`) and in the user doc `docs/features/automatic_prefix_caching.md` (see f18). Verified by grep over the raw file. (READ BODY)

---

## Gaps / NOT VERIFIED

- **No maintainer comment text anywhere in this report.** GitHub core API quota was exhausted (60/60 used) and github.com HTML is network-blocked from this sandbox. Every closure *reason* that lived only in a comment thread is unverified: #22230 (not_planned), #49488 (not_planned), #27325 (not_planned), #41884 (not_planned), #20531 (not_planned), #7569 (not_planned). Their **states and state_reasons are verified** from the GitHub search API; their **verbatim maintainer wording is NOT VERIFIED**.
- **TechRxiv "Transactional KV Caching for Speculative Decoding under Paged KV Memory"** — TITLE ONLY; both curl and web_fetch returned HTTP 403 (Cloudflare).
- **`web_fetch` of github.com issue pages succeeds but truncates** the issue body behind GitHub's navigation chrome, so it was not usable as a body source.
- **arXiv API (`export.arxiv.org/api/query`) returned HTTP 429** throughout; literature discovery was done with `web_search` + per-paper `arxiv.org/abs/` fetches instead.
- Search-result snippets for vLLM PR #52288 (DSpark backend inheritance), PR #47885, and PR #44297 came from a third-party mirror (`app.semanticdiff.com`); the **contents were re-verified against the GitHub search API's own body text**, so they are reported as READ BODY.

---

## Search log (verbatim queries, engine)

### `web_search` (harness tool) — 10 calls, 40 queries

Call 1:
1. `vLLM "Known Feature Incompatibility" speculative decoding`
2. `vLLM speculative decoding structured output grammar incompatibility`
3. `SGLang speculative decoding chunked prefill overlap scheduler disable`
4. `vLLM cudagraph_mode piecewise speculative decoding`

Call 2:
5. `speculative decoding structured output grammar lossless rejection sampling 2026`
6. `grammar-constrained speculative decoding paper 2025 2026`
7. `speculative decoding tensor parallelism communication overhead all-reduce paper`
8. `speculative decoding chunked prefill mixed batching paper 2026`

Call 3:
9. `vLLM PR 47885 grammar-aware draft token sampling structured outputs spec_decode`
10. `SGLang enable-dp-attention speculative decoding incompatible EAGLE NGRAM DFLASH docs`
11. `speculative decoding prefix caching radix cache interaction paper 2026`
12. `speculative decoding CUDA graph capture dynamic speculation depth paper`

Call 4:
13. `speculative decoding FlashInfer FlashMLA paged attention backend compatibility issue`
14. `speculative decoding CUDA graph tree verification kernel paper 2026`
15. `EAGLE speculative decoding prefix caching KV reuse paper arxiv`
16. `speculative decoding MoE expert parallelism all-to-all overhead arxiv 2026`

Call 5:
17. `"speculative decoding" "tensor parallelism" communication overhead arxiv`
18. `speculative decoding chunked prefill mixed batch scheduling arxiv 2026`
19. `tree attention mask speculative decoding paged KV cache paper`
20. `vLLM Automatic Prefix Caching speculative decoding compatibility issue`

Call 6:
21. `arxiv "speculative decoding" "paged attention" verification kernel 2026`
22. `arxiv "speculative decoding" "chunked prefill" scheduling interference`
23. `arxiv speculative decoding multi-GPU tensor parallel draft model placement 2026`
24. `arxiv speculative decoding KV cache reuse prefix sharing acceptance`

Call 7:
25. `"Transactional KV Caching for Speculative Decoding under Paged KV Memory" abstract`
26. `speculative decoding expert parallelism MoE serving system paper 2026 arxiv`
27. `arxiv 2026 speculative decoding data parallel attention disaggregated serving`
28. `vLLM PR 45477 keep intermediate prefill chunks mamba-block-aligned spec decode`

### GitHub search API (`api.github.com/search/issues`) — `gh.py` / `gh2.py`

29. `repo:vllm-project/vllm speculative prefix caching in:title`
30. `repo:vllm-project/vllm speculative chunked prefill in:title`
31. `repo:vllm-project/vllm speculative structured output in:title`
32. `repo:vllm-project/vllm spec decode cuda graph in:title`
33. `repo:sgl-project/sglang speculative prefix caching in:title`
34. `repo:sgl-project/sglang speculative chunked prefill in:title`
35. `repo:sgl-project/sglang eagle grammar structured in:title`
36. `repo:sgl-project/sglang dp attention speculative in:title`
37. `repo:sgl-project/sglang radix cache speculative in:title`
38. `repo:vllm-project/vllm speculative tensor parallel in:title`
39. `repo:vllm-project/vllm speculative expert parallel in:title`
40. `repo:NVIDIA/TensorRT-LLM speculative decoding in:title`
41. `repo:vllm-project/vllm grammar draft token sampling in:title`
42. `repo:vllm-project/vllm structured output spec decode in:title`
43. `repo:vllm-project/vllm prefix caching spec decode`
44. `repo:sgl-project/sglang grammar speculative decoding in:title`
45. `repo:vllm-project/vllm eagle block drop prefix cache in:title`
46. `repo:vllm-project/vllm disable_eagle_block_drop in:title`
47. `repo:vllm-project/vllm speculative attention backend in:title`
48. `repo:vllm-project/vllm flashinfer spec decode in:title`
49. `repo:vllm-project/vllm speculative draft tensor parallel`
50. `repo:sgl-project/sglang speculative tensor parallel in:title`
51. `repo:sgl-project/sglang moe expert parallel speculative in:title`
52. `repo:NVIDIA/TensorRT-LLM eagle tensor parallel in:title`

### Direct URL retrievals (`curl`, `web_fetch`)

53. `raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/README.md`
54. `raw.githubusercontent.com/vllm-project/vllm/main/docs/design/cuda_graphs.md`
55. `raw.githubusercontent.com/vllm-project/vllm/main/docs/design/prefix_caching.md`
56. `raw.githubusercontent.com/vllm-project/vllm/main/docs/design/paged_attention.md` (HTTP 200; historical-only warning, no spec content)
57. `raw.githubusercontent.com/vllm-project/vllm/main/docs/features/{automatic_prefix_caching,structured_outputs}.md`
58. `raw.githubusercontent.com/vllm-project/vllm/main/docs/serving/{parallelism_scaling,expert_parallel_deployment}.md`
59. `raw.githubusercontent.com/vllm-project/vllm/main/{vllm/config/{vllm,compilation,speculative}.py, vllm/v1/core/sched/scheduler.py, vllm/v1/core/single_type_kv_cache_manager.py, vllm/v1/structured_output/backend_lm_format_enforcer.py, vllm/v1/worker/gpu/{eplb_utils,structured_outputs}.py, vllm/v1/sample/rejection_sampler.py, tests/v1/spec_decode/test_mtp_structured_output.py, tests/distributed/test_eplb_spec_decode.py}`
60. `raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/server_args.py` (refactored; 923 lines)
61. `raw.githubusercontent.com/sgl-project/sglang/v0.5.6/python/sglang/srt/server_args.py` (+ v0.4.9, v0.4.10.post2, v0.5.1–v0.5.5, v0.5.7)
62. `raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/{speculative_hook,attention_hook,parallel_hook,validation_hook,cuda_graph_hook}.py`
63. `raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/fields/{spec,schedule,disagg}.py`
64. `raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/speculative/{spec_info,uno_validation,base_spec_worker}.py`
65. `raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/managers/{scheduler,schedule_batch}.py`
66. `codeload.github.com/{vllm-project/vllm,sgl-project/sglang}/tar.gz/refs/heads/main` (full-tree greps, 124 MB / 131 MB)
67. `api.github.com/repos/{vllm-project/vllm,sgl-project/sglang}/git/trees/main?recursive=1`
68. `api.github.com/rate_limit` (x3)
69. `arxiv.org/abs/{2605.07698, 2607.20475, 2604.26412, 2607.12696, 2506.11309, 2502.17421, 2607.06763, 2609.02897, 2508.08192, 2607.24434}`
70. `techrxiv.org/doi/full/10.36227/techrxiv.177101038.80960856/v1` (HTTP 403) and `/doi/pdf/...` (HTTP 403)
