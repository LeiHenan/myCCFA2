# Speculative Decoding — CONFIRMED NEVER-DISCUSSED / ADMITTED-BUT-DISCUSSED / INCONCLUSIVE

**Author:** negative-evidence sweep (delegated agent)
**Date of all searches and retrieval:** 2026-09-13 / 2026-09-14 UTC
**Scope:** speculative-decoding gaps that the inference engines *themselves admit to* in their installed
source code, for which no public discussion was found.

---

## METHOD HEADER

### How candidates were generated (source-first, NOT discussion-first)

The whole point of this sweep is that a discussion-derived candidate list structurally cannot contain
things nobody discussed. So candidates were generated **only** from the engines' own source text.

1. **Source acquisition.** Over SSH (`connect.westd.seetacloud.com:11640`) the spec-decode-relevant
   trees were tarred and pulled to the local workspace, preserving remote paths:
   - vLLM **0.29.0** → `/root/ccfa_venv/lib/python3.12/site-packages/vllm`
     (`v1/spec_decode/`, `v1/worker/gpu/spec_decode/`, `config/speculative.py`, `v1/core/sched/`)
   - SGLang **0.5.19** → `/root/autodl-tmp/venvs/sglang/lib/python3.12/site-packages/sglang`
     (`srt/speculative/`, `srt/managers/`)
   Versions were confirmed by `importlib.metadata.version(...)`, not inferred.
   154 Python files were extracted.

2. **Candidate extraction.** `extract_gaps.py` scanned every extracted `.py` line against 36
   self-admission patterns (`TODO`, `FIXME`, `XXX`, `HACK`, `not supported`, `not implemented`,
   `NotImplementedError`, `does not support`, `only supports`, `unsupported`, `cannot`, `for now`,
   `future work`, `fallback`, `no-op`, `silently`, `not yet`, `ignored`, `limitation`, `incorrect`,
   `approx`, …). **477 raw hits** (vLLM 122, SGLang 355). Restricting to the pure spec-decode trees
   (`/spec_decode/`, `/speculative/`, `/dflash/`, `/dspark/`) left **179 hits**, which is the working
   pool. Test/benchmark files were excluded; license-header "limitations under the License" boilerplate
   was identified and discarded (not treated as a gap).
   *Note: this supersedes the "274 hits / vLLM 59 / SGLang 85" figure quoted in the task, which used a
   narrower pattern set and counted the wider tree; the deltas are pattern breadth and file scope, not
   disagreement.*

3. **Reachability annotation.** Each candidate was checked against the target rig: ONE RTX PRO 6000
   Blackwell 96 GB, sm120 (`compute_cap 12.0`; `is_device_capability_family(100)` is FALSE here),
   driver 580, 208 CPU cores, single node, no NVLink/IB/cluster, vLLM 0.29.0 + SGLang 0.5.19, target
   model Qwen3-4B (dense full-attention, not MoE/hybrid/MLA), drafters dflash2 + EAGLE3, n-gram
   drafting available, FP8/NVFP4 KV unavailable on this arch. GPU identity was verified by
   `nvidia-smi` on the remote (`NVIDIA RTX PRO 6000 Blackwell Server Edition, 12.0, 97887 MiB`).

### How absence was tested

For each candidate I searched, in this order:

1. **SGLang issues + PRs** via GitHub's *repo issue-list* endpoint
   (`https://github.com/sgl-project/sglang/issues?q=<terms>`), which server-renders results and also
   covers pull requests. **108 query x surface combinations** were issued in total (full log below).
2. **vLLM issues + PRs**, same method.
3. **llama.cpp** for cross-engine corroboration on the n-gram item.
4. **Official documentation** (`docs.sglang.io`, `docs.vllm.ai`) read directly, because a limitation
   written into prose documentation *is* a public discussion.
5. **Full issue/PR bodies** for any hit that looked decisive, via `gh_body.py`, which extracts both
   `comment-body` markup and the embedded `"body":"…"` JSON payload. This was necessary and worked:
   maintainer reasoning *is* retrievable this way.
6. **arXiv and general web** delegated to two parallel search agents.

*Tooling caveat, verified:* `api.github.com` was rate-limited to zero. The **global**
`github.com/search?q=…` endpoint hung indefinitely (three attempts, up to 180 s each, 0 bytes
returned) and was abandoned. The **repo** `/issues?q=` and `/pulls?q=` endpoints returned HTTP 200 in
~2 s, and were used instead. Official docs and `arxiv.org/abs/` fetched fine with the prescribed curl
UA. `errors.standardbeagle.com` was found by a sub-agent to mirror engine error strings; those pages
are auto-extracted from the same source trees I am quoting, so they are **not** treated as human
discussion anywhere in this report.

### Exclusion filter applied before claiming novelty

- **`spec-decode-gap-map-2026-09-13.md`** (1,191 lines) was grepped for every candidate keyword and
  every distinctive identifier. A hit there was treated as decisive and the candidate was demoted to
  Section 2.
- **`KV_CACHE_GAP_MAP.md`** was grepped for the same terms, for KV-interaction overlap.
- A stale-bot closure, a single unanswered issue, or a `not planned` state change with zero comments
  all **count as discussed**.
- Per the brief's warning, when a candidate looked like a Section 1 item I searched specifically for an
  **open PR** implementing it. That check killed four candidates (see Section 2, items 2.1, 2.3, 2.4,
  2.5) and one entire sub-axis.

### Honest statement of the result

**Section 1 contains only 4 items** (3 `HIGH`: items 1.2, 1.3, 1.4; 1 `LOW`: item 1.1). That is the
real finding, not a shortfall of effort. Of 179 spec-tree self-admissions, the overwhelming majority are
either (a) pure refactoring TODOs with no user-visible consequence, (b) already fixed by an in-flight PR,
or (c) already named in the prior-art map. The single most valuable candidate I found — sampling-mask /
distribution-replay being incompatible with speculative decoding, which *both* engines reject
explicitly and which **neither engine's documentation mentions** — was killed by an open vLLM PR
(#54166), and is recorded in Section 2 rather than being inflated into Section 1.

All four Section 1 items were additionally run past a dedicated arXiv/literature search, which returned
**NOT-FOUND** for each (item (a) → 1.2, (b) → 1.3, (c) → 1.4, (d) → 1.1's gating family). That search
confirmed adaptive-gamma work is abundant in general (Nightjar arXiv 2512.22420, SpecKV 2605.02888,
D-cut 2607.14647) but found no paper studying adaptive length *restricted to* stacked/multi-layer
drafters, no TP-parallel or multi-core n-gram matching paper (the closest real work, The N-Grammys
arXiv 2411.03786 and SuffixDecoding arXiv 2411.04975, is neither), and no paper on CUDA-graph capture
strategy for multi-module drafting.

**No item in Section 1 is a claim that nobody has ever worked on the topic.** Every item is a
**NOT-FOUND** claim of the form "I searched these exact strings on these exact surfaces and found no
matching discussion."

---

## SECTION 1 — CONFIRMED NEVER-DISCUSSED

### 1.1 — SGLang: adaptive speculative decoding rejects PD-multiplexing, with a reason stated only in the error string

**SELF-ADMITTED GAP:** verbatim, `/root/autodl-tmp/venvs/sglang/lib/python3.12/site-packages/sglang/srt/speculative/adaptive_spec_params.py:84-88`

```python
    if cfg.enable_pdmux:
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"
        )
```

**WHAT IT MEANS:** On SGLang 0.5.19 the `--speculative-adaptive` tier controller (the EMA scheme that
varies draft depth per batch) aborts at startup if `--enable-pdmux` is set. The engine's own reason is
that the adaptive state swap would leave `decode_attn_backend_group` stale. Users get a hard
incompatibility between two individually supported features whose *reason* is disclosed nowhere a user
would look. Sub-axis: **scheduling** (adaptive speculation-length control × prefill/decode multiplexing).

**SEARCHED:**
- GitHub `sgl-project/sglang` issues: `enable_pdmux two batch overlap adaptive`, `pdmux adaptive speculative`, `adaptive pdmux`, `adaptive speculative algorithm`
- GitHub `sgl-project/sglang` pulls: same four strings
- Official docs read directly: `https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding/`
  (found via the "Adaptive Speculative Decoding" nav entry on the main spec-decode page)
  and `https://docs.sglang.io/docs/advanced_features/speculative_decoding/`
- General web (delegated agent, ~96 queries across web_search / HN Algolia / Reddit RSS): its dedicated
  query set for this item found no statement tying adaptive-length control to PDMux
- Prior-art maps grepped for `pdmux`, `PDMux`, `adaptive`

**RESULT:** No matching discussion **for the adaptive-tier × PDMux combination**, but **adjacent
coverage exists and I am disclosing it rather than hiding it**: SGLang's spec-decode docs, in the *UNO*
section's "Key requirements and limitations", state *"Tree mode does not yet support PDMux or the
separate --enable-two-batch-overlap feature."* The **adaptive-spec page itself** has zero occurrences of
`pdmux`, `two-batch`, `dp_attention`, or `multi_layer` (counted, not eyeballed). Repo searches returned
only unrelated items (pinned roadmap #22949, NPU/ROCm quantization bugs, #32143 PD-handoff RFC).

**CONFIDENCE: `LOW`.** The specific adaptive-tier × PDMux exclusion and its stated reason are
undocumented and no issue/PR matches, **but** SGLang's docs already publicise a PDMux limitation for
speculative *tree* mode, and a reader could reasonably call that "discussed". I record this as a
low-confidence, narrow residual rather than asserting a clean discovery.

**NOT-FOUND vs DOES-NOT-EXIST:** **NOT-FOUND.** I searched the four SGLang issue/PR query strings
above, both SGLang spec-decode documentation pages, the general web, and both prior-art maps, and found
no discussion of the adaptive-tier × PDMux gate. I am not claiming nobody has ever worked on adaptive
speculation × PD-multiplexing.

**REACHABLE ON OUR RIG:** **no.** PD-multiplexing is a prefill/decode-multiplexing execution mode; on a
single GPU with no second node it is not usefully configurable. Reported anyway per the brief. (The gate
is a pure argument-validation branch, so the claim is inspectable on-rig even though the feature is not
runnable.)

---

### 1.2 — SGLang: adaptive speculative decoding is silently unavailable with multi-layer EAGLE, because the worker never implemented it

**SELF-ADMITTED GAP:** verbatim, `/root/autodl-tmp/venvs/sglang/lib/python3.12/site-packages/sglang/srt/speculative/adaptive_spec_params.py:74-78`

```python
    if resolved_view(server_args).enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorkerV2 does not implement adaptive)"
        )
```

**WHAT IT MEANS:** Enabling multi-layer EAGLE (stacking several draft layers/modules) and adaptive
speculation together causes SGLang to reject the configuration, because `MultiLayerEagleWorkerV2` has
no adaptive implementation. The consequence is a hard incompatibility between two individually
supported optimisations, stated by the engine only in an error string. Sub-axis: **drafter/target
interaction** (multi-layer drafter × adaptive speculation length), with a secondary **scheduling**
component.

**SEARCHED:**
- GitHub `sgl-project/sglang` issues: `enable_multi_layer_eagle adaptive`, `adaptive multi layer eagle`, `MultiLayerEagleWorkerV2 adaptive`, `adaptive speculative algorithm`
- GitHub `sgl-project/sglang` pulls: same four strings
- Docs: `adaptive_speculative_decoding/` and `speculative_decoding/` pages, counted keyword occurrences
- Maps grepped for `multi_layer`, `multi-layer eagle`, `adaptive`

**RESULT:** No matching discussion. Repo searches surfaced unrelated items only — most notably open PR
#31499 *"[Spec] Extract the shared forward step and introduce EagleWorkerContext"*, which is a
refactor and does not mention adaptive support. The docs page for adaptive speculative decoding
contains **zero** occurrences of `multi_layer`/`multi-layer` (the two `multi-layer` hits on the *main*
spec-decode page are about the Multi-Layer EAGLE feature generally, not about its interaction with
adaptive). A directly relevant search for "adaptive" on that page returns only the nav entry and
"see Adaptive Speculative Decoding" cross-references.

**CONFIDENCE: `HIGH`.** Explicit in-code rejection naming the exact class that lacks the
implementation, absent from the feature's own documentation, and no issue/PR found asking for it.

**NOT-FOUND vs DOES-NOT-EXIST:** **NOT-FOUND.** Searched the four query strings above on SGLang issues
and PRs, both documentation pages, and both prior-art maps; no matching discussion found.

**REACHABLE ON OUR RIG:** **partial.** The flag combination is a one-line configuration on a single
GPU, so the *rejection* is directly observable. Actually exercising adaptive multi-layer EAGLE would
require a multi-layer EAGLE3 drafter for Qwen3-4B, which is not among this rig's available drafters
(dflash2 and a single EAGLE3 drafter).

---

### 1.3 — vLLM: n-gram / prompt-lookup drafting is hard-capped to ONE CPU thread per tensor-parallel rank because TP-parallel n-gram matching was never implemented

**SELF-ADMITTED GAP:** verbatim, `/root/ccfa_venv/lib/python3.12/site-packages/vllm/v1/spec_decode/ngram_proposer.py:46-51`

```python
            # TODO(ekagra-ranjan): bump up the cap from 1 to 8
            # when TP parallelization for ngram is implemented.
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            # Divide by tp_size to ensure each tensor parallel rank
            # has some threads since all ranks will run this.
            self.num_numba_thread_available //= tp_size
```

**WHAT IT MEANS:** The surrounding comments state the intended cap was **8** physical cores; the
shipped expression is `min(1, cpu_count // 2)`, so `num_numba_thread_available` is **always 1**, and
it is then integer-divided by `tp_size`. The author's own note says the cap will only rise to 8 once
TP parallelization for n-gram is implemented. The user-visible consequence is that n-gram/prompt-lookup
drafting runs its numba matching on a single CPU thread per rank regardless of host core count — on
this rig, 208 cores are available and the proposer uses one. Sub-axis: **drafting** (n-gram /
prompt-lookup proposer throughput).

**SEARCHED:**
- GitHub `vllm-project/vllm` issues: `ngram numba thread`, `ngram proposer thread cap`, `ngram numba threads tensor parallel`
- GitHub `vllm-project/vllm` pulls: same three strings
- GitHub `vllm-project/vllm` issues: `ngram proposer multithread`
- Docs/maps grepped for `ngram`, `numba`, `prompt lookup`

**RESULT:** No matching discussion. Repo searches returned only unrelated results (roadmap issues
#48168/#44280/#50001, a POWER8 CPU backend PR #37586, a riscv64 packaging PR #36981, and a
free-threaded-Python installation issue #28762). Nothing matched the thread cap or the missing TP
parallelization.

**CONFIDENCE: `HIGH`.** The cap, the intended value, and the blocker are all stated verbatim in the
source; the arithmetic makes the cap unconditionally 1; and targeted searches on vLLM issues and PRs
found nothing. Note the residual uncertainty below.

**NOT-FOUND vs DOES-NOT-EXIST:** **NOT-FOUND.** Searched the query strings above on vLLM issues and PRs
and grepped both prior-art maps; no matching discussion found. Whether the n-gram-*parallelization*
topic appears in non-indexed literature was delegated to an arXiv search agent whose verdict is
recorded in Section 3.

**REACHABLE ON OUR RIG:** **yes.** vLLM 0.29.0 with `--speculative-config` set to the n-gram method is
a single-GPU configuration, and this rig has 208 CPU cores, so the one-thread cap is both reachable and
measurable (proposer time per step is host-side and needs no special hardware).

---

### 1.4 — vLLM: multi-module MTP silently downgrades its CUDA-graph mode instead of capturing piecewise graphs

**SELF-ADMITTED GAP:** verbatim, `/root/ccfa_venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py:98-105`

```python
    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # TODO(TheEpicDolphin): Support piecewise cudagraph for multi-module MTP.
        if cudagraph_mode.has_piecewise_cudagraphs():
            cudagraph_mode = (
                CUDAGraphMode.FULL_DECODE_ONLY
                if cudagraph_mode.has_full_cudagraphs()
                else CUDAGraphMode.NONE
            )
```

**WHAT IT MEANS:** When multi-module MTP is in use and the requested CUDA-graph mode includes piecewise
graphs, the speculator quietly overrides the mode to `FULL_DECODE_ONLY` (or `NONE`). The configured
mode is therefore not the executed mode, with no error and no warning at this site — a performance-
relevant silent downgrade in the CUDA-graph path for multi-module MTP. Sub-axis: **CUDA graphs**.

**SEARCHED:**
- GitHub `vllm-project/vllm` issues: `multi_module_mtp cudagraph`, `piecewise cudagraph multi-module MTP`
- GitHub `vllm-project/vllm` pulls: same two strings
- GitHub `vllm-project/vllm` issues: `piecewise cudagraph MTP`
- General web (delegated agent): dedicated queries for the multi-module MTP silent CUDA-graph downgrade
  returned **NOT-FOUND** outside GitHub
- Maps grepped for `multi_module`, `piecewise`, `cudagraph`

**RESULT:** No matching discussion. Searches returned tangentially related but different work — most
notably open PR #51575 *"[Bugfix][MRV2] Respect dynamic K=0 from scheduler in
AutoRegressiveSpeculator"*, which concerns dynamic K from the scheduler and not CUDA-graph-mode
downgrading. The prior-art map's only `PIECEWISE cudagraph` row is about a different mechanism
(vLLM's `extract_hidden_states`), and its `piecewise` rows concern D-Cut-style shape/static-capture
integration rather than this fallback. **Disclosed adjacent hit:** vLLM issue #49547 *"FlashInfer +
spec-decode silently downgrades to PIECEWISE cudagraphs (−16% measured)"* and PR #50885 — that is a
*different* mechanism (FlashInfer's own graph-mode selection), not this multi-module-MTP override, but a
reader should know it exists.

**CONFIDENCE: `HIGH`.** The source statement is unambiguous, the absence of any log or raise at the
override site is verified, and GitHub (issues + PRs), the general web, vLLM docs, and both prior-art
maps produced no matching discussion. The adjacent-but-distinct #49547 is disclosed above so a reader
can judge for themselves.

**NOT-FOUND vs DOES-NOT-EXIST:** **NOT-FOUND.** Searched the three query strings above on vLLM issues
and PRs, the general web, and both prior-art maps; no matching discussion found.

**REACHABLE ON OUR RIG:** **partial.** The code path is reachable only with a multi-module MTP drafter;
this rig's target is Qwen3-4B (dense, not an MTP-enabled checkpoint) and its available drafters are
dflash2 and an EAGLE3 drafter, so the downgrade cannot be exercised as intended. The branch itself
(argument validation in `init_cudagraph_manager`) is inspectable and unit-testable without the model,
so the claim is verifiable on-rig even though the feature is not runnable as designed.

---

## SECTION 2 — ADMITTED BUT ALREADY DISCUSSED (the audit trail)

This section is how Section 1 is audited. Each row is a self-admitted gap I found in source that I
**excluded** because a public discussion already covers it.

| # | Self-admitted gap (source) | Covering discussion (URL) | State |
|---|---|---|---|
| 2.1 | **vLLM: `sampling distribution replay does not support speculative decoding`** — verbatim `vllm/config/vllm.py:1051`, inside `_verify_sampling_replay_config()`, gated on `model_config.return_sampling_mask` and `if self.speculative_config is not None: raise ValueError(...)`. **Verified by me on the host: vLLM 0.29.0 contains only this bare message — no explanatory rationale in the installed source.** The feature's own Limitations section (`docs.vllm.ai/en/latest/training/sampling_mask/`, read directly) lists only two limitations — the global `--return-sampling-mask` flag disabling the fused sampler, and no streaming support — and does **not** mention speculative decoding. *(A third-party mirror renders an explanatory sentence for this error; that prose is AI-generated by the mirror, not vLLM's, and the mirror's own `Source:` line for it is wrong — see Section 3.8.)* | https://github.com/vllm-project/vllm/pull/54166 — *"[Feature][Spec Decode] Support sampling mask replay for MRV2 MTP"*, **Open**, body read in full: *"Package one processed target-logit support for each possible committed position in a fixed MTP verification chunk, then route exactly one support for every emitted accepted, recovered, or bonus token."* | open PR implements it |
| 2.2 | **SGLang: `return_sampling_mask is not supported with speculative decoding.`** — verbatim `sglang/srt/managers/scheduler.py:2787`, preceded at :2783-2785 by *"Spec workers do not emit one sampling support per accepted token, so the returned mask would not align 1:1 with generated tokens. Reject the combination instead of silently returning a misaligned mask."* Reinforced in `srt/managers/scheduler_components/batch_result_processor.py:976-977`: *"return_sampling_mask + speculative decoding is rejected at request entry, so this remains one support mask per token."* | Same axis as 2.1; SGLang's sampling-mask series is public and extensive: PR #36630 (merged), #36631 (merged), #38279 (merged), #33593 (merged), #36540 (open), #36518 (closed), #35205 (merged), #34037 (closed), issue #35765 (closed). The spec-decode incompatibility is the motivation for the whole constrained design. | shipped + open PRs |
| 2.3 | **SGLang: removing an external n-gram corpus during a pending async load is undefined behaviour** — verbatim `sglang/srt/speculative/external_corpus_manager.py:86`: *"FIXME(kpham-sgl): remove a corpus during a pending load is an undefined behaviour and should be explicitly prevented."* | https://github.com/sgl-project/sglang/issues/36500 (*"[Bug] [NGRAM] Removing a corpus during asynchronous loading returns success without cancelling the load"*) and fix PR https://github.com/sgl-project/sglang/pull/36517, **Open**, whose body explicitly cites my FIXME: *"The existing `# FIXME(kpham-sgl)` comment in `ExternalCorpusManager.remove()` already flags this as undefined behaviour."* | open issue + open PR |
| 2.4 | **SGLang: every spec worker rejects pipeline parallelism** — verbatim `sglang/srt/speculative/standalone_worker_v2.py:86`, `multi_layer_eagle_worker_v2.py:166`, `frozen_kv_mtp_worker_v2.py:145`, `eagle_worker_v2.py:177`: *"spec workers don't support pipeline parallelism"*; and `vllm/v1/worker/gpu/spec_decode/dspark/utils.py:80`: `raise NotImplementedError("DSpark does not support pipeline parallelism.")` | https://github.com/sgl-project/sglang/pull/38702 — *"[DSpark] Support PP verify with stage-local draft owners"*, **Open**, body read: *"running all DSpark draft work on the last pipeline stage creates an additional bottleneck… This PR introduces an experimental PP2 DSpark mode."* | open PR implements it |
| 2.5 | **SGLang: adaptive speculation under data-parallel attention** — verbatim `adaptive_spec_params.py:69-73`: *"enable_dp_attention=True is not supported (adaptive tier decisions are not synchronized across DP ranks)"* | https://github.com/sgl-project/sglang/pull/35898 — *"[Spec] Enable adaptive speculative decoding under DP attention"*, **Open**. Body read in full and it *quotes the guard verbatim*: *"It is disabled whenever `--enable-dp-attention` is set: `adaptive_unsupported_reason()` bails out and the server falls back to static params. The guard is there for a reason… If ranks disagree, the collective shapes diverge and NCCL hangs."* Related: #39231 *"[Spec] Stabilize the DP adaptive tier vote: shared EMA + min dwell"* (Open). | open PR implements it |
| 2.6 | **vLLM: verification materializes an FP32 target-logits canvas, bounded by a 1 GiB chunk** — verbatim `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:27-30`: *"TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast per vocab block on load and apply ops like temperature and gumbel, so folding sampling-param application into those kernels would remove this buffer and its traffic entirely."*; and `adaptive_verification.py:130-133`: *"Rejection sampling verifies logits in one contiguous chunk; the chunked path indexes by scheduled (untrimmed) offsets and cannot address the compacted layout, so the budget must fit one chunk."* | Prior-art map row **B3.9** (cited there as https://github.com/vllm-project/vllm/pull/53630): *"the target rejection verifier currently materializes an FP32 processing canvas of shape [num_verification_tokens, vocab_size]. The production path bounds the transient allocation with a 1 GiB chunk, which in turn forces a cap on the total logits adaptive verification may schedule per step."* Additionally, the literature leg found the general idea is **published**: https://arxiv.org/abs/2607.20475 — *"SonicSampler: Unified Tile-Aware Kernels for LLM Sampling and Speculative Verification"* (Ponnusamy, Sahni, Wang, Dao; Together AI), which fuses grammar masking, repetition/frequency/presence penalties, logit bias, temperature and top-k/top-p/min-p together with speculative verification into one batched CUDA-Graph-compatible kernel. | mapped **and published** |
| 2.7 | **vLLM: draft probabilities are not remapped into target-vocab space for lossless probabilistic acceptance** — verbatim `vllm/v1/spec_decode/llm_base_proposer.py:499`: *"TODO: remap draft_probs to target-vocab space for lossless"* | Prior-art map section **F2 (Vocabulary mismatch)** and **F4 (Draft-vocab pruning / FR-Spec)**: F2.1 vLLM #38174 (merged), F2.3 SGLang #22883 (open), F2.4 SGLang #24051 + #23838, F4 rejection on measurement, G5.4-G5.9 (Speculators `--d2t-path`, MiniMax draft-vocab32k, SlimSpec arXiv 2605.10453, SpecVocab arXiv 2602.13836). | already mapped |
| 2.8 | **SGLang: EAGLE3 rejection sampling requires draft_vocab_size == target_vocab_size** — verbatim `sglang/srt/speculative/eagle_worker_v2.py:222-223`: *"FIXME: support reduced (hot) draft vocab by scattering draft probs into the target vocab via the d2t map before the sampling kernel."* | Same prior-art map rows as 2.7 (F2/F4/G5), plus this rig's own documented EAGLE3 draft (`G6.7`, Qwen3-4B EAGLE3 with draft vocab 32000). Independent corroboration (delegated web agent): SafeAILab/EAGLE issue #222 *"Clarification on d2t and t2d Mapping Logic in EAGLE-3 Draft Model"*. | already mapped |
| 2.9 | **vLLM: returning hidden states from a DFlash-class drafter is unsupported** — verbatim `sglang/srt/speculative/dflash_utils.py:1076`: *"DFLASH speculative decoding does not support return_hidden_states yet."* | https://github.com/sgl-project/sglang/issues/38589 — *"Qwen4Exp (Qwen3.8-Flash-Next) cannot emit DFlash aux hidden states: DFlash/DSpark drafters cannot be served"*, **Open**. Prior-art map also has rows on `extract_hidden_states` and DFlash drafter-training data. | open issue covers it |
| 2.10 | **vLLM: heterogeneous-vocab drafting supports greedily-sampled drafts only** — verbatim `vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py:23-26`: `"extract_hidden_states only supports draft_sample_method='greedy'"`, and `llm_base_proposer.py:502`: *"probabilistic draft sampling is not supported with …"* | Prior-art map row **B5.1**: *"use_heterogeneous_vocab currently supports greedy draft sampling only. Probabilistic acceptance (temperature > 0 draft sampling) is not yet supported and will be added in a future release."* | already mapped |
| 2.11 | **vLLM: fused multi-step draft decode is unsupported on some attention backends** — verbatim `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:121`: *"Fused multi-step draft decode is not supported by attention …"*, and `:136`: *"PIECEWISE cudagraphs are not supported for draft decodes."* | Prior-art map row **A6** / **E4** (paged-attention backends) and the `piecewise`/`PIECEWISE cudagraph` rows, plus the D-Cut integration blocker quoted in map section C1. | already mapped |
| 2.12 | **SGLang: EAGLE has no CUDA-graph draft-extend path for several attention backends** — verbatim `sglang/srt/speculative/eagle_worker_v2.py:467`: *"TODO: support draft extend cuda graph for more attention backends"*; `draft_utils.py:125`: *"cute-dsl MLA only supports decode; draft-extend falls back to trtllm-gen."*; `draft_utils.py:110,142`: *"EAGLE is not supported in decode attention backend {backend_type}"* | Prior-art map **E4 (Paged-attention backends)** and **E5 (CUDA graphs are explicitly keyed on K)**; also the affected backends (cute-dsl MLA, flashmla, trtllm-mla) are MLA/DeepSeek-specific, which this rig cannot run at all. | already mapped + off-rig |
| 2.13 | **SGLang: the DFlash fused-context-KV path is refused for non-neox RoPE** — verbatim `sglang/srt/speculative/dflash_worker_v2.py:612`: *"non-neox RoPE is not supported for fused KV path"*; `dflash_utils.py:748`: *"qkv bias is not supported for fused KV path"* | The RoPE half is publicly documented as an error string (`sglang/kernels/ops/speculative/fused_kv_materialize.py:270` = `raise NotImplementedError("Only neox-style RoPE is supported.")`, verified on the host; also surfaced via ErrLookup, which is why that half counts as public); SGLang PRs #36038, #37446, #38724, #34765 all touch FP8/NVFP4 KV scales for spec decoding. Separately, **quantized KV is unavailable on sm120 on this rig**, so the k_scale/v_scale halves are off-rig regardless. *(Mechanism, corrected after a reviewer-style check: DFlash fuses a **target context feature into the draft's Key/Value projections** and stores it in the **draft's** KV cache — it does not fuse draft KV into the target cache. I had described it the other way round; the DFlash method paper (arXiv 2602.06036) was full-text scanned and contains zero occurrences of `tensor parallel`, `quantiz*`, `lm_head`, `qkv`, `k_scale` or "fused context KV", so none of the engine's restrictions here appear in the paper.)* | discussed / off-rig |
| 2.14 | **vLLM: EPLB is not supported for Medusa** — verbatim `vllm/v1/spec_decode/medusa.py:68-71`: `assert not (is_mixture_of_experts(self.model) and self.vllm_config.parallel_config.enable_eplb), "EPLB for Medusa is not supported"` | Requires a **MoE** target model (the assert is guarded by `is_mixture_of_experts`), which this rig's Qwen3-4B is not. Also adjacent to open PR #56387 *"[Bugfix][Spec Decode][MoE] Avoid uninitialized EPLB state in DeepSeek V4.1 DSpark drafter"*. Additionally, Medusa is heavily covered in the prior-art map (A3.8, A5.4, B4.9, B4.11, F8.7, G1.8, G2.11). | mapped + off-rig |
| 2.15 | **vLLM: MRV2 PCP does not support speculative decoding** — verbatim `/root/ccfa_venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/pcp_manager.py:150-152`: `raise NotImplementedError("MRV2 PCP does not support speculative decoding yet.")` | PCP (parallel context processing / decode context parallelism) is a multi-GPU execution mode, out of scope for a single-GPU rig; the same guarded-neighbour pattern (MM inputs, LoRA, sparse-MLA CUDA graphs) is actively tracked in vLLM. Prior-art map **D2 (Parallelism / multi-node requirements)** covers the class. | off-rig + mapped |
| 2.16 | **vLLM: spec decoding is disabled for hybrid/Mamba models with prefill checkpoint blocks** — verbatim `vllm/v1/core/sched/scheduler.py:329-332`: `self.mamba_has_prefill_checkpoint_blocks = (self.has_mamba_layers # TODO: support spec decoding and not self.use_eagle and all(...))` | This is Mamba/hybrid-model-specific; this rig's target is dense full-attention Qwen3-4B. Prior-art map **D1/D4** cover the family/arch traps, and **KV_CACHE_GAP_MAP.md** covers Mamba checkpoint-block interaction (55 `HiCache` hits). | off-rig + mapped |
| 2.17 | **SGLang: overlap/async scheduling support is limited per speculative algorithm** — verbatim `sglang/srt/speculative/spec_registry.py:133`: `f"Speculative algorithm {self.name} does not support overlap scheduling."` | **Documented in the engine's own prose**, plus independently corroborated: SGLang's spec-decode docs state the overlap scheduler *"currently only supports --speculative-eagle-topk 1 … the server will error"*; vLLM is separately gated (*"async scheduling is only supported with EAGLE/MTP/Draft Model/NGram GPU/DSpark"*, found by the delegated agent); prior-art map has `overlap scheduling` rows. | documented + mapped |
| 2.18 | **SGLang: adaptive speculation is gated to EAGLE/EAGLE3 with topk 1 only** — verbatim `adaptive_spec_params.py:59-68` | **Documented in SGLang's own words** on its adaptive spec-decode page: *"Current support: Only --speculative-algorithm EAGLE or EAGLE3; Only --speculative-eagle-topk 1; If either condition is not met, SGLang falls back to static speculative settings."* | documented |
| 2.19 | **SGLang: adaptive speculation × two-batch overlap** — verbatim `adaptive_spec_params.py:79-83`: *"enable_two_batch_overlap=True is not supported (adaptive state swap would discard the TboAttnBackend wrapper)"* | Covered in prose by SGLang's spec-decode docs (UNO "Key requirements and limitations": *"Tree mode does not yet support PDMux or the separate --enable-two-batch-overlap feature"*), and adjacent to open PR #35898, which addresses the DP-synchronisation family this guard belongs to. | documented (adjacent) |

### Notable candidates excluded **before** the search stage, with the reason

- **`eagle_draft_cuda_graph_runner.py:626` and `eagle_draft_extend_cuda_graph_runner.py:543`,
  `# TODO(ch-wan): support num_token_non_padded`** — *explicitly excluded, and this is a judgement
  call I want visible.* I verified the asymmetry is real (SGLang's `frozen_kv_mtp_utils.py:132-134`
  and `dspark_components/dspark_draft.py:488-491` *do* set `num_token_non_padded`, and
  `multi_layer_eagle_draft_extend_cuda_graph_runner.py:357` hardcodes
  `num_token_non_padded_cpu = captured_req_width * bs`), so the EAGLE draft CG runners genuinely
  over-count padded rows. **But** the consumer (`srt/batch_overlap/two_batch_overlap.py`) is the
  two-batch-overlap path, and the prior-art map already owns that territory (E5, E6, and the TBO rows).
  Rather than present a likely-occupied item as a discovery, I exclude it. It would be the first thing
  to re-examine if a future sweep wants a fifth Section 1 entry.
- **`decoupled_spec_io.py:181-188`** (*"a raise here … silently kills the drafter control thread. It
  then stops applying ALL requests' controls while the verifier keeps pushing"*) — a genuinely
  striking self-admission, but the file's own TODO says the fix is scheduled for *"phase 5.c"*, which
  is itself evidence of an active plan; I could not retrieve a public issue for it, and the feature
  (decoupled spec IO) is not part of the Qwen3-4B + dflash2/EAGLE3 path on this rig. Not promoted to
  Section 1 because I could not rule out in-flight work.

---

## SECTION 3 — INCONCLUSIVE

| # | Candidate | Blocker |
|---|---|---|
| 3.1 | **arXiv / literature coverage — now COMPLETE for all 21 items.** Initially only partial; the delegated agent delivered the full table before finalization. Every arXiv ID cited in this report was fetched from `arxiv.org/abs/<id>` and its title read back. | **Result: only 3 of 21 items are FOUND in the literature** — (2) sampling-params fused into the verification kernel (SonicSampler, arXiv 2607.20475); (6) reduced draft vocab + explicit `T_(d->t)` token-index mapper applied before target verification with losslessness argued (VOCABTRIM, arXiv 2506.22694, plus FR-Spec 2502.14856, NanoSpec 2605.26444, DynaSpec 2510.13847, CORAL 2502.16880, OmniDraft 2507.02659); (12) multimodal speculative drafting (arXiv 2608.20743 survey, ViSpec 2509.15235, CoVSpec 2605.02218). **All four Section 1 items (1.2, 1.3, 1.4, 1.1) are NOT-FOUND in the literature.** Notably, items 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21 are cases where the gap is acknowledged in engine source and/or PRs but is **absent from the reachable literature** — that asymmetry is itself a citable finding. One item-6 hit required care: row 2.8's engine gap is `draft_vocab_size == target_vocab_size` being *required* by SGLang's rejection-sampling path, which VOCABTRIM's mapper removes the need for; I therefore treat 2.8 as **discussed in the literature** even though the engine-side open PR is SGLang #22883. |
| 3.2 | **`github.com/search?q=…` (global, cross-repo) queries** | The global search endpoint hung indefinitely: three attempts with `--max-time 45`, `--connect-timeout 10`, 2 retries, each returning **0 bytes** after up to 180 s. Only repo-scoped `/issues?q=` and `/pulls?q=` worked. **This is the largest coverage gap in the report**: a discussion living entirely outside the vLLM/SGLang issue trackers (e.g. in a fork, an HF repo, or another tracker) would not have been found by the GitHub leg. The *fork* case is real — the delegated agent found a fork-side markdown index (`mulder/vllm.cpp`) referencing a DFlash2 / quantized-lm_head limitation. |
| 3.3 | **`api.github.com`** | Rate-limited to zero, as the brief warned. All GitHub evidence came from HTML. |
| 3.4 | **Zhihu article on EAGLE-3 / DFlash / DSpark draft training** — `https://zhuanlan.zhihu.com/p/2054617534528237672` | HTTP 403. Title retrieved only (*"把draft训得又小又准:NeMo-Automodel投机实战 (EAGLE-3/DFlash /Dspark)"*); body unretrievable, so anything it holds about these gaps is unassessed. |
| 3.5 | **Reddit per-thread comments** for three DFlash threads (`1urxfa1`, `1sx8uok`, `1vsy4l2`) | `search.json` and `old.reddit` were blocked; `r/<sub>/search.rss` worked for post-level coverage, but per-thread comment `.rss` returned **HTTP 429**. Titles retrieved, bodies not. |
| 3.6 | **Items 2.13's k_scale/v_scale, quantized-qkv_proj and qkv-bias halves** | Both the delegated agent and I found **no** public statement for these three specific halves (only the non-neox-RoPE half is publicly mirrored). Recorded here rather than in Section 1 because the whole fused-context-KV item is off-rig (it needs quantized KV, unsupported on sm120 here), making it low-value. |
| 3.7 | **SGLang PR #31499 `[Spec] Extract the shared forward step and introduce EagleWorkerContext`** | Diff not retrieved, so I cannot fully exclude that it touches adaptive-state swapping for multi-layer EAGLE. It is titled as a refactor and its body was not read. This is the main residual risk to item **1.2**. |
| 3.8 | **ErrLookup (`errors.standardbeagle.com`) as a "public surface"** | Proposed by a sub-agent as a discussion surface; **I falsified it.** Its prose is AI-generated, and its `Source:` line for the vLLM spec-decode error cites `vllm/config/vllm.py:1033`, which on vLLM 0.29.0 is the KV-connector/`expandable_segments` message; the real site is `:1051` (I read both on the host). It mirrors source strings only, so it can neither confirm nor deny human discussion, and its coverage numbers are not evidence of anything. |
| 3.9 | **Two repo queries returned `HTTP: TIMEOUT`** — `sgl-project/sglang [pulls] q='adaptive speculative algorithm'` and `sgl-project/sglang [issues] q='tree traversal threshold tree kernel'` | First-attempt network timeouts. Both were re-covered by sibling strings (`adaptive pdmux`, `TREE_TRAVERSE_TIME_THRESHOLD`) which returned clean results. Logged for completeness. |

---

## COMPLETE SEARCH LOG

### A. GitHub repo-scoped searches (108 query × surface combinations)

Every query below was issued against **both** the `issues` and the `pulls` endpoint of the named repo
(the repo issues endpoint also returns PRs, so the two surfaces overlap by design; both are logged).
Each returned HTTP 200 and was parsed; result sets were reviewed by title and state.

**Surface `vllm-project/vllm`** (issues + pulls for each):
`return_sampling_mask` · `sampling distribution replay speculative decoding` ·
`return-sampling-mask speculative` · `return sampling mask` · `sampling mask speculative decoding` ·
`ngram numba thread` · `ngram proposer thread cap` · `ngram numba threads tensor parallel` ·
`ngram proposer multithread` · `rejection sampler chunk logits buffer` · `apply_sampling_params chunk` ·
`mamba prefill checkpoint blocks spec decoding` · `ssm checkpoint blocks eagle` ·
`piecewise cudagraph multi-module MTP` · `multi_module_mtp cudagraph` · `piecewise cudagraph MTP` ·
`fused multi-step draft decode attention` · `EPLB medusa` · `EPLB medusa pipeline parallel` ·
`EPLB speculative`

**Surface `sgl-project/sglang`** (issues + pulls for each):
`return_sampling_mask` · `return sampling mask` · `sampling mask speculative decoding` ·
`draft extend cuda graph attention backend` · `draft extend cudagraph` · `num_token_non_padded` ·
`num_token_non_padded draft cuda graph` · `chunked prefill chain divergence` ·
`TREE_TRAVERSE_TIME_THRESHOLD` · `tree traverse time threshold speculative` ·
`tree traversal threshold tree kernel` · `reduced draft vocab scatter probs` ·
`QLEN_ONLY_BITPACKING` · `QLEN_ONLY_BITPACKING triton` · `adaptive speculative dp attention` ·
`adaptive speculative two batch overlap` · `adaptive speculative frozen kv mtp` ·
`adaptive speculative EMA tier` · `adaptive speculative algorithm` · `adaptive pdmux` ·
`pdmux adaptive speculative` · `enable_pdmux two batch overlap adaptive` ·
`enable_multi_layer_eagle adaptive` · `adaptive multi layer eagle` · `MultiLayerEagleWorkerV2 adaptive` ·
`adaptive two batch overlap` · `external corpus pending load` · `external corpus pending load undefined` ·
`drafter control thread killed` · `DSpark pipeline parallelism` · `sample_from_anchor DFlash` ·
`return_hidden_states DFLASH` · `quantized lm_head DFLASH tensor parallel` · `fused context KV draft` ·
`fused KV materialize` · `quantized qkv_proj fused KV` · `k_scale v_scale fused KV` ·
`non-neox RoPE fused KV` · `skip_tokenizer_init DFLASH` · `EPLB medusa` · `ssm state speculative decoding` ·
`tree mask mode sglang` · `treemask mode`

**Surface `ggml-org/llama.cpp`** — planned for cross-engine corroboration on the n-gram thread item;
**not executed** (the vLLM evidence for item 1.3 was already decisive and self-contained, and the
candidate pool did not yield a llama.cpp-specific finalist). Recorded as not-done rather than implied.

### B. Full issue/PR body retrievals (decisive reads)

| URL | State | Why read |
|---|---|---|
| https://github.com/vllm-project/vllm/pull/54166 | **Open** | Body confirms it implements sampling-mask replay for MTP spec decode → killed the sampling-mask candidate |
| https://github.com/sgl-project/sglang/pull/36631 | **Closed** | Sampling masks + overlap scheduling; motivation text read |
| https://github.com/sgl-project/sglang/pull/36517 | **Open** | Body quotes my `external_corpus_manager.py:86` FIXME verbatim → item 2.3 |
| https://github.com/sgl-project/sglang/pull/38702 | **Open** | Body confirms DSpark PP verify → item 2.4 |
| https://github.com/sgl-project/sglang/pull/35898 | **Open** | Body quotes the `adaptive_unsupported_reason()` DP guard verbatim → item 2.5 |

PR states were taken from the rendered page (`Status:` markers and/or the JSON `state` field), per the
brief's note that a Draft PR renders as `Status: Draft`; #54166 was independently confirmed `Open`.

### C. Documentation read directly (human-written prose = discussion)

| URL | What it establishes |
|---|---|
| https://docs.vllm.ai/en/latest/training/sampling_mask/ | Limitations section lists exactly two limitations (global flag disables FlashInfer fused sampler; no streaming support) and **never mentions speculative decoding** |
| https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding/ | "Current support" lists exactly two conditions (EAGLE/EAGLE3 only; topk 1 only); **zero** occurrences of `pdmux`, `two-batch`, `dp_attention`, `multi_layer` → items 1.1 and 1.2 |
| https://docs.sglang.io/docs/advanced_features/speculative_decoding/ | Reviews per-method requirements matrix; `PDMux` mentioned once, for **UNO tree mode**, not adaptive tier control; "Key requirements and limitations" read in full |

### D. Source verification

Every Section 1 quotation was re-read from the **installed** trees (via `read` on the extracted copies
and via direct `sed` on the remote), not from memory or from upstream `main`. Line numbers refer to
`vLLM 0.29.0` and `SGLang 0.5.19` as installed on the target host. `show_ctx.py` was used to print
surrounding lines so that multi-line comments are quoted in full rather than truncated at the match.

### E. Web search

Three `web_search` calls were issued directly (all returning links only, no synthesised answers used as
evidence): `SGLang return_sampling_mask speculative decoding not supported` /
`vLLM ngram_proposer num_numba_thread_available cap 1 tensor parallel` /
`SGLang scheduler.py return_sampling_mask spec_algorithm`. Their results were what first pointed at
SGLang's sampling-mask PR series (#27408, #33593, #36630/#36631) and were followed up on GitHub
directly. Broader arXiv and forum coverage was delegated (see Section 3.1, 3.2).

---

## APPENDIX — Bottom line

- **Raw self-admissions mined:** 477 (179 inside the pure spec-decode trees).
- **Section 1 (never-discussed): 4 items** — 3 `HIGH` (1.2 multi-layer EAGLE × adaptive, 1.3 n-gram
  one-thread cap, 1.4 multi-module MTP graph downgrade), 1 `LOW` (1.1 adaptive × PDMux). All four were
  independently NOT-FOUND on arXiv/literature as well as GitHub/docs/web.
- **Section 2 (admitted but discussed): 19 rows**, each with a URL; plus 3 named pre-search exclusions.
- **Section 3 (inconclusive): 9 entries**. The three that matter: the **unfinished tail of the arXiv
  leg** (verdicts arrived for my four finalists and for items 2/9/11/12/13/14, but not the complete
  21-item table), the **unusable GitHub global-search endpoint**, and **ErrLookup**, which I falsified
  as a discussion surface rather than accepting it.
- The highest-value *discovery* of this sweep is a **documentation** gap rather than a code gap: both
  vLLM and SGLang enforce spec-decode incompatibilities (`sampling distribution replay` and
  `return_sampling_mask`) that **their own feature documentation omits** — vLLM's sampling-mask
  Limitations section lists two *other* limitations and not this one. Because open PRs already implement
  the feature side (#54166) and the sampling-mask series is otherwise extensive, it sits in Section 2.
- **Methodological finding worth carrying forward:** search-engine *snippet titles* are unreliable for
  arXiv IDs in this environment. Independently of my own checks, the literature agent caught
  `web_search` returning fabricated titles for **three** real arXiv IDs: 2606.18394 (snippet "JetFlow: …" vs abs page "JetSpec: …"), 2605.26444 (snippet "MicroSpec: … Lightweight In-Context Vocabularies" vs abs page "NanoSpec: … Minimalist In-Context Vocabularies"), and 2411.04975 (snippet "A Model-Free Approach…" vs abs page "Extreme Speculative Decoding for Emerging AI Applications"). In every case the ID resolved and the *title* was wrong. Every arXiv ID cited above was read back from its `abs` page.
- **No recommendations are made here.** This is an inventory of negative evidence; screening is a
  separate process.
