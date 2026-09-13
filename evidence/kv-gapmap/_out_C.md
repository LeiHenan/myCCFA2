## C. ATTEMPTED-AND-ABANDONED cells (most valuable!)

### C.1 Closed-unmerged pull requests

#### C1. vLLM `[Core] Cache policy framework` — the `CachePolicy` eviction abstraction for V0, closed unmerged
- **WHO:** `ShawnD200` (author) in vllm-project/vllm; closed by `hmellor`, a vLLM Member.
- **URL:** https://github.com/vllm-project/vllm/pull/11928
- **Closure status:** PR closed unmerged by a named human maintainer (`hmellor`, vLLM Member), not a bot.
- **STATED REASON, VERBATIM:** Hi @ShawnD200 , I'm going to close this PR for the following reasons: It is for V0, for which we are not accepting new features (bugfixes only) It's 4 months stale You have the RFC to continue the discussion and make a new PR for V1 when the time comes! — `hmellor`, Member, May 6, 2025.
- **Date:** created 2025-01-10T11:03:04Z; closed 2025-05-06T09:27:09Z.
- **Evidence:** READ BODY

#### C2. vLLM `[Core] Support sparse KV cache framework` — H2O/SnapKV/Scissorhands/StreamingLLM eviction framework
- **WHO:** `chizhang118` (author) in vllm-project/vllm; closed by `hmellor`, a vLLM Member. This is the implementation PR for the still-open RFC #5751.
- **URL:** https://github.com/vllm-project/vllm/pull/5752
- **Closure status:** PR closed unmerged by a named human maintainer (`hmellor`); the RFC it implements (#5751) stays open.
- **STATED REASON, VERBATIM:** Closing as stale. If you plan to continue this work, feel free to re-open or make a new PR. — `hmellor`, Member, Feb 28, 2025. - Additional in-thread blocker, VERBATIM (from a reviewer, before closure): "Hi @chizhang118 , I'm very interested in your design doc to see the detail of sparse kv cache framework. but it's not available now, may you share a valid link? Thanks!" — the RFC's linked design doc was dead.
- **Date:** created 2024-06-21T20:31:34Z; closed 2025-02-28T13:47:28Z.
- **Evidence:** READ BODY

#### C3. SGLang `[Feature] Semantic eviction for radix cache` — agent-workload reset/summarisation pruning
- **WHO:** `rexxy-sasori` (author) in sgl-project/sglang; closed by `hnyls2002`, a SGLang Collaborator (human, not a bot).
- **URL:** https://github.com/sgl-project/sglang/pull/20088
- **Closure status:** PR closed unmerged by a human Collaborator (`hnyls2002`); the semantic-event path was never added to main.
- **STATED REASON, VERBATIM:** "Thanks @rexxy-sasori ! This is still a draft with no commits since 2026-03-08, and it carries local dev artifacts that cannot merge - a .gitmodules adding external/sglang pointing at a personal fork, plus docker/dev.Dockerfile . Several touched files have also moved since ( srt/metrics/collector.py -> srt/observability/metrics_collector.py , scheduler_output_processor_mixin.py -> srt/managers/scheduler_components/ , Sphinx docs/ -> Mintlify docs/docs/ ). semantic_event is still absent from main, so a rebased, artifact-free PR would be welcome. Please reopen if I've missed something."
- **Date:** created 2026-03-07, closed 2026-08-10 (the S13 duplicate row for the same PR had no separable closure)
- **Evidence:** READ BODY

#### C4. SGLang TurboQuant KV-cache PR — closed unmerged as one of three concurrent proposals
- **WHO:** YanGaev2 (author); closed by hnyls2002 (SGLang collaborator)
- **URL:** https://github.com/sgl-project/sglang/pull/22048
- **Closure status:** closed unmerged by a human collaborator, consolidating on #21419 (itself still open).
- **STATED REASON, VERBATIM:** "Thanks @YanGaev2 ! There are three concurrent TurboQuant proposals and we are consolidating on #21419 , which reuses the existing flashinfer_backend instead of adding a parallel flashinfer_tq backend to attention_registry.py , is roughly 1500 lines smaller, and ships an end-to-end accuracy eval ( python/sglang/test/simple_eval_niah.py ) rather than unit tests plus a single-request throughput number. This PR also targets two paths that no longer exist - python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py and the retired test/srt/ suite dir. The fused Triton encode/decode kernels and cuda-graph handling here are the strongest part - please consider contributing those on top of #21419 . Please reopen if I've missed something."
- **Date:** opened Apr 3, 2026; closed Aug 11, 2026 by hnyls2002
- **Evidence:** READ BODY

#### C5. vLLM Triton INT4 / INT2 per-token-head KV quantisation — superseded by a narrower PR; the INT2 half dropped
- **WHO:** JartX (author); closed by the author after review by tdoublep
- **URL:** https://github.com/vllm-project/vllm/pull/40633 | https://github.com/vllm-project/vllm/pull/39074
- **Closure status:** PR closed unmerged by its own author in favour of #40835 (which merged). The companion PR #39074 was also closed unmerged with only bot text on the page.
- **STATED REASON, VERBATIM:** "@tdoublep closed in favor of: #40835" — and, on the companion PR #39074 the only stated text is the `mergify` bot notice: "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @JartX . (posted by the `mergify` bot, May 23, 2026)"
- **Date:** opened Apr 22, 2026; closed May 25, 2026. Note the preceding automated comment in the same thread is from the `mergify` bot ("This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @JartX ."), but the actual closure is a human author decision with the stated reason above.
- **Evidence:** READ BODY

#### C6. vLLM TurboQuant KV cache with PolarQuant + QJL — closed unmerged as a duplicate design
- **WHO:** allaspectsdev (author); asked to close by gaby (vLLM maintainer)
- **URL:** https://github.com/vllm-project/vllm/pull/38662
- **Closure status:** closed unmerged by its author on a maintainer request; the QJL half did not survive into the shipped backend.
- **STATED REASON, VERBATIM:** "@allaspectsdev Please close this. See #38479"
- **Date:** opened Mar 31, 2026; "allaspectsdev closed this Apr 7, 2026". The review on this PR also flagged, verbatim, "missing scaling factors in the QJL bias correction", a stack buffer overflow in the polar decode path, and hardcoded 256-element buffers that "assume head_size is a power of 2".
- **Evidence:** READ BODY

#### C7. vLLM’s first INT8 KV-cache-quantisation PR — closed without landing; the follow-up split PRs never landed either
- **WHO:** AniZpZ (author); closed by simon-mo (vLLM collaborator)
- **URL:** https://github.com/vllm-project/vllm/pull/1112
- **Closure status:** closed unmerged by a maintainer (`simon-mo`); the maintainer’s reason addresses W8A8/INT8 support generally, not KV quantisation specifically.
- **STATED REASON, VERBATIM:** "Closing as we do have W8A8 and Int8 support nowadays. 🙏"
- **Date:** opened Sep 20, 2023; closed Oct 1, 2024 by simon-mo. Immediately after the close, a user asked the still-unanswered question, verbatim: "Hi, is there any plan to support int4 KVCacheQuant?" This is the historical root of OPEN-6.
- **Evidence:** READ BODY

#### C8. vLLM CPU KV “swap space” — never allocated in V1, parameter removed; the abandonment PR was picked up by another author
- **WHO:** mcelrath (reporter), vLLM; removal merged by DarkLight1337
- **URL:** https://github.com/vllm-project/vllm/issues/27984 | https://github.com/vllm-project/vllm/pull/27988 | https://github.com/vllm-project/vllm/pull/36216 | https://github.com/vllm-project/vllm/pull/35652
- **Closure status:** issue closed with the feature deleted: `--swap-space` and the `swap_space` field were removed by merged PR #36216. The original removal PR #27988 was closed unmerged and picked up by another author; the later per-layer `SwapConnector` revival (#35652) was closed unmerged by its author the day it opened, with no stated reason on the page.
- **STATED REASON, VERBATIM:** Since the deprecation of best_of sampling in #13997 (see also #13361 ), the swap_space parameter is useless and unused in the codebase. To make matters worse, it seems best_of was never properly implemented because num_cpu_blocks in EngineCore::_initialize_kv_caches() is hard coded to zero at vllm/v1/engine/core.py:252 and always has been according to the git history. and "This is probably why best_of had no usage -- it never worked and has always been disabled." — and, the replacement PR #36216 records the abandonment: The replacement PR records the abandonment verbatim: "Picks up the abandoned PR #27988 as suggested by @DarkLight1337 in #27984 ." The replacement PR body also states the technical reason verbatim: "In the V1 engine, num_cpu_blocks is hardcoded to 0 ( v1/engine/core.py:287 ), confirming swap space was never allocated or used."
- **Date:** #27984 opened 2025-11-03, closed 2026-03-07; #27988 opened 2025-11-03, closed 2026-03-07; #36216 merged 2026-03-07; #35652 opened and closed 2026-03-01
- **Evidence:** READ BODY

#### C9. vLLM `Kvserve fusion` (KV compression + Mooncake connector) — 33 commits, never merged
- **WHO:** VerrPower, vLLM PR #51648
- **URL:** https://github.com/vllm-project/vllm/pull/51648
- **Closure status:** PR closed unmerged by maintainer `hmellor`; the only machine-recorded reason on the page is the `mergify` rebase request — no human maintainer rationale was posted.
- **STATED REASON, VERBATIM:** "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @VerrPower ."
- **Date:** opened 2026-08-10; closed 2026-08-10
- **Evidence:** READ BODY

#### C10. vLLM `add KV cache tiering residency and lifecycle management` — self-closed as targeting the wrong repository
- **WHO:** JieYang2001 (intellistream), vLLM PR #48837
- **URL:** https://github.com/vllm-project/vllm/pull/48837
- **Closure status:** PR closed unmerged by its own author, who renamed the title to record the reason.
- **STATED REASON, VERBATIM:** The author retitled the PR and closed it the same day; the recorded title is exactly: "[Closed: wrong repository]feat: add KV cache tiering residency and lifecycle management"
- **Date:** opened 2026-07-16; closed 2026-07-16
- **Evidence:** READ BODY

#### C11. vLLM `Fail fast when CPU offload region exceeds available space` — closed because main already carried the fix
- **WHO:** Thibaultjaigu, vLLM PR #47073
- **URL:** https://github.com/vllm-project/vllm/pull/47073
- **Closure status:** PR closed unmerged by its own author as superseded by already-merged work (#50358, #48414, #52596).
- **STATED REASON, VERBATIM:** Closing: main already carries this fix. #50358 (merged 2026-08-05) added check_shm_free_space(self.total_size_bytes) in the creator path with the file cleanup on failure, which is exactly the shape @orozery asked for here, and the region has since been reworked further in #48414 and #52596 . Nothing left for this branch to add. Thanks for the review.
- **Date:** opened 2026-06-29; closed 2026-09-12
- **Evidence:** READ BODY

#### C12. vLLM `Bypass power-of-2 rounding in KV offload CPU pinned allocation` — folded into a different PR
- **WHO:** vLLM PR #48436, author procr1337 (verifier-corrected; `woosebastian` only cross-referenced the PR), closed by orozery
- **URL:** https://github.com/vllm-project/vllm/pull/48436
- **Closure status:** PR closed unmerged “in #50094” by maintainer `orozery`; no prose rationale was posted. (The evidence file names the author as `woosebastian`; the verifier corrected this to `procr1337`.)
- **STATED REASON, VERBATIM:** "orozery closed this in #50094"
- **Date:** opened 2026-07-12; closed 2026-07-29
- **Evidence:** READ BODY

#### C13. vLLM `SimpleCPUOffloadConnector` backlog bug report — closed by the fix PR, not as a duplicate
- **WHO:** vLLM issue #53498 (author greatyingzi), closed by maintainer ivanium
- **URL:** https://github.com/vllm-project/vllm/issues/53498
- **Closure status:** closed as completed by PR #53532 (`stateReason: COMPLETED`, `duplicateOf: null`) — the evidence file’s “closed as duplicate” framing is contradicted by the page’s own structured data.
- **STATED REASON, VERBATIM:** The page records the closure as a duplicate link to "#53532" with closing actor ivanium at 2026-09-02T03:23:11Z; the only prose is the reporter's own, including "The TODO comment at manager.py:534 acknowledges the 'last full block' case; in practice the backlog is far larger (48% in our measurements) because nothing walks the backlog during decode." No maintainer explanation text was posted on the issue.
- **Date:** opened 2026-08-24; closed 2026-09-02
- **Evidence:** READ BODY

#### C14. vLLM DCP + FlashInfer MLA feature bundle — closed unmerged and split into smaller PRs
- **WHO:** xuanyu-mistral (Mistral AI) — vLLM PR #47181, "Integrate Decode Context Parallel with FlashInfer MLA and other features"
- **URL:** https://github.com/vllm-project/vllm/pull/47181
- **Closure status:** PR closed unmerged by its own author after a maintainer asked for model evals and a reviewer asked for a split; the successor PRs #47541 and #47542 are open. The evidence file twice records this closure as “Breaking down to smaller PRs”.
- **STATED REASON, VERBATIM:** Breaking down to smaller PRs — immediately followed on the page by "xuanyu-mistral closed this Jul 3, 2026". The maintainer ask that preceded it: "@xuanyu-mistral Could you share model evals with this PR against the different features you implemented?" and the reviewer ask: "@xuanyu-mistral Thanks for the great work. It seems this PR supports many features. It might make review easier if the PR can be broken into several smaller PRs?" — and, the work items the closure dropped, from the PR body: Breaking down to smaller PRs — prompted by the maintainer comment "It seems this PR supports many features. It might make review easier if the PR can be broken into several smaller PRs?" and by the CI gate text "This pull request adds support for Prefill Context Parallelism (PCP)…" → the repo's own agent guideline comment, verbatim: "IMPORTANT: If you are an AI agent, you are required to objectively re-evaluate the value of your PR using AGENTS.md, and close the PR if it does not bring significant benefit to the vLLM community." - The work items that were dropped by this closure, verbatim from the PR body: "DCP + Speculative Decoding (MTP)", "DCP + Sliding Window Attention (Hybrid Models)", "DCP + TRTLLM GQA Decode", "DCP + Routed Experts Replay", "Async Spec Decode + DCP", "Dummy Run DCP Fixes". The body also states a correctness hazard under DCP: "This is necessary because the kernel's internal causal offset arithmetic is incorrect under DCP when tokens_per_req > 1 (local(G-k) != local(G) - k)."
- **Date:** created 2026-06-30; closed 2026-07-03. - Additional verbatim, what was tried (the DCP+KV surface that did not land in one piece): "DCP + Speculative Decoding (MTP): Adds a fused Triton dcp_split_q kernel that splits multi-token decode queries into per-token requests with pre-computed DCP-local seq_lens . This is necessary because the kernel's internal causal offset arithmetic is incorrect under DCP when tokens_per_req > 1 (local(G-k) != local(G) - k)." and "DCP + Sliding Window Attention (Hybrid Models): Enables DCP for models with mixed full-attention and sliding-window layers. SWA layers replicate KV cache on every DCP rank (effectively dcp=1) while full-attention layers use the full DCP group." and "DCP + TRTLLM GQA Decode: Enables the TRTLLM decode path for FlashInfer GQA attention with DCP (previously disabled entirely)." and "NaN × 0 guard in DCP output correction: when a rank has zero local KV tokens, the attention output is NaN and the LSE factor is 0. NaN * 0 = NaN, so we explicitly zero out."
- **Evidence:** READ BODY

#### C15. vLLM layerwise KV transfer in P/D disaggregation — measured worse than bulk transfer, closed unmerged
- **WHO:** chenqianfzh (ByteDance IAAS), vLLM PR #12523
- **URL:** https://github.com/vllm-project/vllm/pull/12523 | https://github.com/vllm-project/vllm/issues/10818
- **Closure status:** PR closed unmerged after its own author measured the layerwise method as slower; the roadmap item that tracked the same work was later closed `NOT_PLANNED`.
- **STATED REASON, VERBATIM:** "the layerwise method performs worse as it needs to retrieves layer_number tensors, so it requires more overhead. I will continue to work on the solution that the consumer starts to receive the KV cache once the producer completes a layer." — and, the roadmap entry that tracked it, verbatim: "- [ ] [Perf] layer-by-layer pipelining (#12523)"
- **Date:** PR created 2025-01-28, closed 2025-03-19; roadmap issue #10818 created 2024-12-02, closed not planned
- **Evidence:** READ BODY

#### C16. vLLM Extensible (growable) KV cache — 39-commit series closed unmerged
- **WHO:** `njhill` / `lwilkins` (Red Hat) — a 39-commit series
- **URL:** https://github.com/vllm-project/vllm/pull/50779
- **Closure status:** closed unmerged; the closure comment was retrieved by the verifier (“Replaced by …”) after the evidence file reported no justification text.
- **STATED REASON, VERBATIM:** "Replaced by https://github.com/vllm-project/vllm/pull/56492."
- **Date:** created 2026-08-02, closed 2026-09-11
- **Evidence:** READ BODY

#### C17. vLLM host-resident KV cache offloading for sparse MLA decode — closed unmerged in favour of a parallel implementation
- **WHO:** `LCAIZJ`
- **URL:** https://github.com/vllm-project/vllm/pull/51270
- **Closure status:** closed unmerged; the author’s own closure rationale was retrieved by the verifier after the evidence file reported no retrievable reason.
- **STATED REASON, VERBATIM:** "Since this PR and @MatthewBonanni's PR #51323 implement the same feature … I find PR #51323 to be a better fit for the first version to be merged. Therefore, I plan to close this PR and continue optimizing this feature based on PR #51323."
- **Date:** created 2026-08-06, closed 2026-08-11
- **Evidence:** READ BODY

#### C18. vLLM semantic checkpoints for recurrent-state prefix caching — `[Exp]` prototype closed unmerged
- **WHO:** `qianlihuang`
- **URL:** https://github.com/vllm-project/vllm/pull/49574
- **Closure status:** closed unmerged; no explicit closure comment exists on the page (the verifier confirmed this row’s “reason not retrieved” is defensible).
- **STATED REASON, VERBATIM:** No justification text retrievable. Commit subject of `[PATCH 1/5]`, verbatim: "\[Core\] Prototype semantic cache checkpoints" with body "Accept trusted frontend token boundaries and propagate them through the engine and scale-out request paths. Align recurrent prefill chunks so those boundaries can materialize under sparse retention." **No reason is asserted.**
- **Date:** created 2026-07-23, closed 2026-09-11
- **Evidence:** READ BODY

#### C19. vLLM rejecting prefix caching for pooling / encoder-only models — sibling PR closed unmerged
- **WHO:** `qsxustc` (closed) vs `HsukqiLee` (still open, #55159)
- **URL:** https://github.com/vllm-project/vllm/pull/55160
- **Closure status:** closed unmerged; no formal maintainer closure sentence exists, but a review comment on the page records the de-facto reason (the sibling PR #55159 targets the same function). The review comment is reproduced here exactly as the verification report renders it, including its ellipsis.
- **STATED REASON, VERBATIM:** "duplicates **#55159** … which targets the exact same function" — and, the commit subject of the closed PR itself: "\[Bugfix\] Reject unsupported prefix caching for pooling models"
- **Date:** created 2026-09-03, closed 2026-09-07
- **Evidence:** READ BODY

#### C20. TensorRT-LLM Blackwell-native FlashInfer context attention — closed as relocated out of the repository
- **WHO:** NVIDIA maintainers
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/issues/16860
- **Closure status:** closed as not planned — a relocation, not a technical rejection.
- **STATED REASON, VERBATIM:** "Moved to NVIDIA-dev/paragraf#140 so the work is tracked in the Paragraf repository."
- **Date:** created and closed 2026-07-25
- **Evidence:** READ BODY

#### C21. TensorRT-LLM closed-unmerged PR — a closure reason does exist on the page
- **WHO:** `trtllm-agent` (body author); closed by `yuxianq`
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/pull/18531
- **Closure status:** closed unmerged by `yuxianq`; the verifier retrieved the closure comment after the evidence file asserted no reason existed (#18544 is merged). Concerns FMHA routing rather than strictly KV cache.
- **STATED REASON, VERBATIM:** "Close since we already have a bugfix PR https://github.com/NVIDIA/TensorRT-LLM/pull/18544"
- **Date:** closed 2026-09-02
- **Evidence:** READ BODY

#### C22. vLLM unified `context_parallel_size` configuration/parallel-group design — superseded by the split PCP/DCP config
- **WHO:** qiruiyangmeta
- **URL:** https://github.com/vllm-project/vllm/pull/26057 | https://github.com/vllm-project/vllm/pull/26058
- **Closure status:** PRs closed unmerged by a human maintainer (`hmellor`) with a technical supersede reason — a named-maintainer verdict rather than bot text or a stale timer.
- **STATED REASON, VERBATIM:** This PR has been superseded by the Prefill Context Parallel (PCP) and Decode Context Parallel (DCP) implementations that have since been merged: [Feature] Prefill Context Parallel (PCP) basic support #28718 (PCP basic support) [Feature] Support Decode Context Parallel (DCP) for MLA #23734 (DCP for MLA) [DCP] Support Decode Context Parallel (DCP) for GQA with FlashAttention #24864 (DCP for GQA with FlashAttention) The codebase now uses separate prefill_context_parallel_size / decode_context_parallel_size config fields and get_pcp_group() / get_dcp_group() rather than the unified context_parallel_size / get_cp_group() approach proposed here. Thank you for the contribution! - The abandoned design's own sharding scheme, verbatim from the PR body: "Causal attention imposes a varying computational load for each token, as shown in the following figure. To ensure an even workload distribution, tokens should be partitioned across different context parallelism (CP) ranks. Specifically, the sequence is divided into 2 × cp_world_size chunks. Each CP rank i is assigned both the i-th chunk and the (2 × cp_world_size - i - 1)-th chunk."
- **Date:** #26057/#26058 created 2025-10-01, closed 2026-03-06
- **Evidence:** READ BODY

#### C23. vLLM MoRIIO multi-node TP16 prefill→decode KV dispatch — closed unmerged as superseded by three merged PRs
- **WHO:** chaeminlim-mb (+ edwinlim0919, MangoBoost)
- **URL:** https://github.com/vllm-project/vllm/pull/45228
- **Closure status:** PR closed unmerged; the closer comment is a human statement of supersession by #46116, #46115 and #47495.
- **STATED REASON, VERBATIM:** "The issue that this PR targets has been addressed by #46116, #46115, and #47495, all of which have been merged. This PR can be safely closed."
- **Date:** created 2026-06-11, final close ~2026-08-14 (#45228; companion #43063/#45230 quotes are sub-agent-retrieved)
- **Evidence:** READ BODY

#### C24. TensorRT-LLM `Reuse V2 decode KV cache across turns` — closed unmerged by a maintainer as a deferral
- **WHO:** erictsai-nv (PR author), closed by nvpohanh (Collaborator) on NVIDIA/TensorRT-LLM
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/pull/18711
- **Closure status:** PR closed unmerged by a human Collaborator (`nvpohanh`) — a direct, deliberate abandonment of cross-turn decode-KV reuse in the V2 cache manager, phrased as “for now”.
- **STATED REASON, VERBATIM:** "Closing this for now. I will ask @erictsai-nv to continue working on this when he is back."
- **Date:** last review activity 2026-09-04; closed 2026-09-08 (closedTime 2026-09-08T00:51:04Z)
- **Evidence:** READ BODY

#### C25. TensorRT-LLM inclusive host KV-cache tier (shadow reuse) — closed unmerged by its author after the reviewer never approved
- **WHO:** reasonsolo (PR author, closed it), reviewed by lowsfer (Member) on NVIDIA/TensorRT-LLM
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/pull/15828
- **Closure status:** PR closed unmerged by its author. NO CLOSURE COMMENT WAS POSTED — the quoted maintainer line is the last substantive comment on the design, not a closure statement.
- **STATED REASON, VERBATIM:** "Please hold on until bug reporter's response. It's a design choice. It uses more D2H bandwidth but D2H bandwidth is under-utilized anyway, and it improves cache hit rate because it avoids duplication and can hold more data."
- **Date:** opened 2026-07-02; closed 2026-07-31 (closedTime 2026-07-31T02:13:07Z)
- **Evidence:** READ BODY

#### C26. SGLang Router — honoring KV-event storage tiers in the cache-aware tree (closed as superseded, then re-split)
- **WHO:** Kangyan-Zhou (Collaborator) on SGLang
- **URL:** https://github.com/sgl-project/sglang/pull/38720
- **Closure status:** PR closed unmerged by its author as superseded; the same change was re-split into the stacked series #39108/#39109/#39110, all still open — the work moved rather than stopped.
- **STATED REASON, VERBATIM:** "Superseded — split into a stacked series and closed. The same change, byte-for-byte, now lands as #39108 (the tier-aware tree), #39109 (the sgl_router_kv_* /metrics series) and #39110 (the real-GPU e2e proof), all with their heads on sgl-project/sglang so they can chain base-to-head. Review there; this branch is gone."
- **Date:** opened 2026-09-09; closed 2026-09-11 (closedTime 2026-09-11T18:23:38Z)
- **Evidence:** READ BODY

#### C27. vLLM fine-grained prefix-cache hits for sliding-window groups — implementation closed unmerged twice by its own author
- **WHO:** vLLM contributor RichApple123 — PR #54319, then its revision PR #54397
- **URL:** https://github.com/vllm-project/vllm/pull/54397
- **Closure status:** both PRs (#54319 and its revision #54397) closed unmerged by their own author with no closure comment. The only retrievable statement of the blocking condition is an automated merge-conflict notice.
- **STATED REASON, VERBATIM:** "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @RichApple123 ." — and, the design intent that is now unmerged, from the PR body: This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @RichApple123 . — this is an automated GitHub notice, **not** a maintainer statement. The PR body itself records the design intent that is now unmerged: "Fixes #53786 by allowing `SlidingWindowManager` to participate safely in fine-grained hybrid prefix-cache lookup when its physical KV page is larger than the configured hash unit. Previously, one SWA group with a larger physical page disabled `enable_partial_hash_hits` for the entire hybrid coordinator." The predecessor #54319's body states: "This change only makes prefix endpoints hash-granular".
- **Date:** #54319 created 2026-08-29, closed 2026-08-29. #54397 created 2026-08-30, closed 2026-09-11.
- **Evidence:** READ BODY

### C.2 wontfix / not-planned / by-design / out-of-scope closures

#### C28. vLLM align-mode checkpoint retention visibility — author-closed as redundant, not a maintainer rejection
- **WHO:** vLLM issue #53595, author jschmied (a CONTRIBUTOR), self-closed
- **URL:** https://github.com/vllm-project/vllm/issues/53595
- **Closure status:** closed as not planned (`stateReason=NOT_PLANNED`) by the author himself; explicitly not a stale-bot closure and not a maintainer rejection. The report was folded into the fix PR #53596.
- **STATED REASON, VERBATIM:** "Closing: this was redundant with the PR that fixes it and with the PR that already describes the same invisibility." — and, the same closing comment, with the follow-up line: Closing: this was redundant with the PR that fixes it and with the PR that already describes the same invisibility. — followed by "- **#53596** carries the full report — cause, the asymmetry with the announced `mamba_cache_mode` default, and the measurement — and is where the fix is reviewed."
- **Date:** created 2026-08-24, closed 2026-08-24
- **Evidence:** READ BODY

#### C29. Sizing the SGLang hybrid Mamba state pool from explicit `max_running_requests` — marked not planned
- **WHO:** SGLang issue #33004 (feature request); maintainers marked it not planned; reporter is bash99 (Ben Rood)
- **URL:** https://github.com/sgl-project/sglang/issues/33004
- **Closure status:** closed as not planned by a maintainer (`huangzhilin-hzl`) on the day of filing. The maintainer’s own stated reason is NOT present in the retrieved payload; the only rationale text on the page is the reporter’s reaction.
- **STATED REASON, VERBATIM:** "I am disappointed that this issue was marked as 'not planned'."
- **Date:** created 2026-07-31, closed 2026-07-31
- **Evidence:** READ BODY

#### C30. Naive round-robin block allocation for sliding-window layers — deliberately designed out of the shipped hybrid KV manager
- **WHO:** vLLM maintainers, in the shipped hybrid KV cache design doc
- **URL:** https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/hybrid_kv_cache_manager.md
- **Closure status:** by design — considered and deliberately not adopted in the shipped vLLM hybrid KV cache design doc; a closed design decision, not a merged feature.
- **STATED REASON, VERBATIM:** "For sliding window attention layers, a naive implementation for memory allocation is to allocate `sliding_window_size` blocks and fill in the blocks in a round-robin way. But this naive implementation is not compatible with prefix caching so we didn't pick this design. In vLLM, we allocate different blocks for different tokens and free blocks that are outside the sliding window."
- **Date:** doc states it was written against commit `458e74` (the doc carries no date in the retrieved copy; the linked commit is the reference point)
- **Evidence:** READ BODY

#### C31. Cache-affinity-aware request ordering in the vLLM V1 scheduler (`CacheAffinityScheduler`) — dropped after the +7.4% result was shown to be a benchmark artefact
- **WHO:** `kliukovkin` (vLLM contributor), RFC #42185 / CacheAffinityScheduler plugin; also mirrored as S7-ABANDONED-1, S11-ABANDONED-1 and S13-ABANDONED-10
- **URL:** https://github.com/vllm-project/vllm/issues/42185
- **Closure status:** closed as not planned (`stateReason=NOT_PLANNED`). VERIFIER CORRECTION: the closer was the human RFC author `kliukovkin`, who posted a closing comment — NOT the stale bot and not a maintainer. The stale label/notice is genuinely present but is only a notice. The evidence file’s “the closure actor is the stale bot / no human reason exists” is false.
- **STATED REASON, VERBATIM:** **Decision:** Dropping PR — RAG improvement below threshold, no measurable benefit on correct async baseline. ... "RAG mean delta: **+0.2%** (noise). Decision threshold: ≥+5%." ... and the root cause of the earlier promising number: "### Why the original +7.4% RAG result was wrong — `CacheAffinityScheduler` initially extended synchronous `Scheduler` instead of `AsyncScheduler`." The author's own closing comment is also verbatim: "Closing this from my side: since filing, the ecosystem has moved in exactly this direction — prefix-cache-aware routing landed as a first-class concern in llm-d and AIBrix, which validates the premise better than further discussion here would." The automated message that also appears on the page is verbatim: "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!"
- **Date:** created 2026-05-09, closed 2026-08-18 (ClosedEvent 2026-08-18T00:44:01Z by kliukovkin)
- **Evidence:** READ BODY

#### C32. Sparse KV cache management framework RFC — closed “COMPLETED” while labelled stale, with only one of four tasks delivered
- **WHO:** `ShawnD200` (RFC author) in vllm-project/vllm; discussed by `heheda12345` (Collaborator).
- **URL:** https://github.com/vllm-project/vllm/issues/12254
- **Closure status:** `state=CLOSED`, `stateReason=COMPLETED`, labels include `RFC` and `stale`. No human ClosedEvent or explanatory comment was found; the only merged artifact (PR #12608) covers task (1) of four, and the CachePolicy interface for H2O/FastGen does not exist in the tree today. Treat “COMPLETED” as not substantiated for the eviction-policy content.
- **STATED REASON, VERBATIM:** the RFC's own goal statement now abandoned in practice: "CachePolicy is a general interface that manages KV cache allocations. It generalizes from basic cache behavior that always allocates new slots/blocks to append new tokens to the more sophisticated that dictates how cache space is used should there be new tokens, such as sliding window, H2O, FastGen, etc." - Thread positions showing the disagreement that stalled it, VERBATIM: `ShawnD200`: "For H2O you mentioned, I could be terribly wrong, but I don't see how it works without mapping, and you could have it either by explicit mapping or making block_size=1 so that block ids can be arranged to be a mapping as well. Prefix caching adds a layer of complexity, for \"full attention\" and \"sliding window\" it does not affect much, but for \"random\" selective eviction policy, either have block_size=1, or the block after token in it evicted can't be shared." `ShawnD200` again: "Then it comes down to: change block manager vs. attention module, this seems not a question, from single responsibility (and single place -- there is only one block manager but a few backends) point of view."
- **Date:** created 2025-01-21T08:46:39Z; issue last updated 2026-07-02T06:25:29Z.
- **Evidence:** READ BODY

#### C33. vLLM shipped TurboQuant backend drops QJL by design and needs a boundary-layer crutch for its aggressive presets
- **WHO:** vLLM (the TurboQuant backend authors, `vllm/model_executor/layers/quantization/turboquant/config.py`)
- **URL:** https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/quantization/turboquant/config.py
- **Closure status:** by design in shipped code — QJL is documented as deliberately omitted, and the aggressive presets are documented as requiring an empirical crutch (boundary-layer skipping).
- **STATED REASON, VERBATIM:** "QJL is intentionally omitted: community consensus (5+ independent groups) found it hurts attention quality by amplifying variance through softmax." — and, the second abandonment recorded in the same shipped file: "Empirically required for aggressive presets (k3v4_nc, 3bit_nc) — without it GSM8K drops ~30 points on Qwen3-4B."
- **Date:** code present in vLLM main as of retrieval (2026-09-13); the TurboQuant backend merged Apr 15, 2026 (PR #38479)
- **Evidence:** READ BODY

#### C34. QJL error correction (Algorithm 2) omitted from a second, independent TurboQuant implementation
- **WHO:** `Aaryan-Kapoor` and `veritatisquaesitoressumus`, in the ggml-org/llama.cpp discussion #20969 (verifier-corrected authorship; the evidence file attributed these words to TheTom)
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** omitted by design in an independent implementation — corroborates the vLLM TurboQuant omission. VERIFIER CORRECTION: these words were posted by `Aaryan-Kapoor` and `veritatisquaesitoressumus`, not by `TheTom`.
- **STATED REASON, VERBATIM:** QJL error correction (Algorithm 2) not implemented - Algorithm 1 alone is sufficient per the paper and, on the implementation plan, "The implementation follows Algorithm 1 (TurboQuant_mse) from the paper. Algorithm 2 (QJL error correction) is omitted as the paper shows MSE-optimal quantization alone is sufficient for KV cache compression without the extra bit cost."
- **Date:** posts dated Mar 2026 (thread started Mar 25, 2026)
- **Evidence:** READ BODY

#### C35. Direct GPU↔storage access (GPUDirect Storage / GPU-GPU) — explicitly excluded from vLLM’s multi-tier offloading design
- **WHO:** dannyharnik (vLLM), RFC #38260 "Multi-tier KV offloading via the vLLM offloading connector"
- **URL:** https://github.com/vllm-project/vllm/issues/38260
- **Closure status:** by design — an explicit “What we don’t intend to support” section in a maintainer-authored RFC; the shipped tiering implementation carries exactly this restriction.
- **STATED REASON, VERBATIM:** The RFC carries a section headed "What we don't intend to support", whose two bullets read verbatim: "Direct GPU access (neither GPU-storage or GPU-GPU communication)" and "Limited flexibility for variance in block size. While we allow vLLM block size to vary, CPU block size must be constant across all vLLM nodes (and a multiple of the underlying vLLM block size)." The shipped doc restates the first bullet verbatim: "Only the CPU primary tier has direct GPU access. Secondary tiers cannot read from or write to GPU memory; all GPU↔secondary transfers are staged through the CPU primary tier."
- **Date:** RFC opened 2026-03-26 (still Open as an RFC; the non-support decision is settled and codified in shipped code)
- **Evidence:** READ BODY

#### C36. MLA’s low-communication “absorb” form for training — hard-banned by Megatron-Core and measured to regress activation memory
- **WHO:** Megatron-Core (NVIDIA) as the abandoner; the LAGA authors as the measurers
- **URL:** https://arxiv.org/abs/2607.17644
- **Closure status:** by design — a hard assertion in shipped library code disables the technique with no documented reason; the accompanying paper explains why the ban is well-founded (20–34% activation-memory regression when ported to training).
- **STATED REASON, VERBATIM:** Open Megatron-Core's MLA implementation NVIDIA (2025) and the forward begins with an assertion that bans its own low-communication path from training: assert not (self.training and self.cache_mla_latents). The absorb reformulation — folding kv_b_proj into the query so attention runs against the compressed latent and only the small latent crosses the collective — is fully implemented in the class, but gated to is_decode_only() inference and hard-asserted out of training. The library ships no low-communication MLA training path. Additional verbatim: "MCore's designers deliberately restrict the absorb trick to inference: the class's prepare_for_absorption even concedes that it is \"not doing true absorption. We will add this support at a later time.\" One contribution of this paper (§3.2/4.3) is to explain why that restriction is well-founded — the absorbed form, ported to training, regresses activation memory by 20–34% (up to 9.2 GB at V3 scale)". Also verbatim from the paper's own limitations: "Why this is both comm-cheap and memory-neutral. ... Measured: Laga peak memory matches B1 within ≤0.5% (<27 MB) while B2 balloons by up to 9.2 GB (Table 2)."
- **Date:** submitted 20 Jul 2026 (v1).
- **Evidence:** READ BODY

#### C37. LMCache CacheBlend: blend store cannot be disabled and has no size cap
- **WHO:** `shawnguyen-tensormesh` (LMCache), issues #4665 and #4664 (the two topics were filed together by the same author)
- **URL:** https://github.com/LMCache/LMCache/issues/4665 | https://github.com/LMCache/LMCache/issues/4664
- **Closure status:** both issues closed as not planned (`stateReason: NOT_PLANNED`). VERIFIER CORRECTION: each page carries a comment “Closing here” plus a ClosedEvent whose actor is the issue author `shawnguyen-tensormesh`; the evidence file’s repeated claim that comment text does not exist and the actor is unidentifiable is false for all six CacheBlend issues.
- **STATED REASON, VERBATIM:** that gate was **not carried over** (MP connector built as a parallel implementation ignoring `kv_role=kv_consumer` / `skip_save`). Companion #4664: "Unbounded blend store fills the L2 disk and evicts the serving pods via node DiskPressure - no size cap".
- **Date:** both created 2026-08-20, closed 2026-08-21 by the author (“Closing here”)
- **Evidence:** READ BODY

#### C38. LMCache CacheBlend: no per-model correctness gate for the blend re-RoPE approximation
- **WHO:** `shawnguyen-tensormesh` (LMCache), issue #4666
- **URL:** https://github.com/LMCache/LMCache/issues/4666
- **Closure status:** closed as not planned (`stateReason: NOT_PLANNED`) by the issue author, whose only closure comment is “Closing here” (verifier-corrected: the evidence file claimed no comment text and no identifiable actor).
- **STATED REASON, VERBATIM:** There is **no output-level, per-model acceptance test** … "correctness depends on ... **no automated check that the blended output actually matches a full-recompute reference** before it is served."
- **Date:** opened Aug 20, 2026; closed Aug 21, 2026
- **Evidence:** READ BODY

#### C39. LMCache CacheBlend: missing inter-document padding for non-prefix reuse (~11% measured seam loss)
- **WHO:** shawnguyen-tensormesh (LMCache), opened 2026-08-20, closed 2026-08-21
- **URL:** https://github.com/LMCache/LMCache/issues/4667
- **Closure status:** closed as not planned by the issue author (comment: “Closing here”) — a measured seam tax was reported and not acted on.
- **STATED REASON, VERBATIM:** "Closing here"
- **Date:** opened Aug 20, 2026; closed Aug 21, 2026
- **Evidence:** READ BODY

#### C40. LMCache cross-pod (cross-replica) sibling KV reuse — works, but is transient; closed as not planned
- **WHO:** shawnguyen-tensormesh (LMCache), opened 2026-08-20, closed 2026-08-21
- **URL:** https://github.com/LMCache/LMCache/issues/4668
- **Closure status:** closed as not planned by the issue author (comment: “Closing here”). The capability itself was reported to exist, but as a burst rather than steady state.
- **STATED REASON, VERBATIM:** "Closing here"
- **Date:** opened Aug 20, 2026; closed Aug 21, 2026
- **Evidence:** READ BODY

#### C41. KV reuse across LoRA adapters — deliberately excluded by vLLM’s prefix-cache block-hash key
- **WHO:** vLLM project (by design, enforced in code; reinforced by security PR #54342)
- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py
- **Closure status:** not a closure — “by design”. vLLM folds the LoRA adapter name into the block hash, so two adapters can never hit each other’s cached blocks; an adapter change invalidates the chain from that point. Reinforced by security PR #54342 (TITLE ONLY).
- **STATED REASON, VERBATIM:** The function docstring and body, retrieved character-for-character from the raw source file: "Return LoRA name of the request if it is a LoRA request. Return empty list otherwise." with the body "if not request.lora_request: return []" / "return [request.lora_request.lora_name]". The keys are combined unconditionally: "lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)" ... "extra_keys: list[Any] = ( lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys )".
- **Date:** current `main` as of retrieval (2026-09-13)
- **Evidence:** READ BODY

#### C42. vLLM `kv_role='kv_both'` for NixlConnector — deprecated and scheduled for removal
- **WHO:** vLLM project (deprecation shipped in `main`; docs PR to stop recommending it)
- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py
- **Closure status:** shipped deprecation warning in code (“will be removed in a future release”); the recommended usage was removed from the docs (PR #55693, CLOSED). Not yet deleted.
- **STATED REASON, VERBATIM:** "Using kv_role='kv_both' with NixlConnector is deprecated and will be removed in a future release. Please set kv_role='kv_producer' for prefill instances and kv_role='kv_consumer' for decode instances."
- **Date:** current `main` as of retrieval (2026-09-13)
- **Evidence:** READ BODY

#### C43. vLLM uneven decode context parallelism for heterogeneous TP ranks — explicit maintainer scope rejection
- **WHO:** efschu (contributor) — vLLM PR #50736, "[Feature] Uneven decode context parallelism for heterogeneous TP ranks"; closed by hmellor (vLLM member)
- **URL:** https://github.com/vllm-project/vllm/pull/50736
- **Closure status:** closed unmerged by a human maintainer (`hmellor`) as out of scope in-tree — a clean maintainer “no”, not a stale bot.
- **STATED REASON, VERBATIM:** "Uneven TP sharding is not something I think we are interested in supporting in-tree in vLLM as it is quite niche and appears to need changes to every supported model. In situations where you have different sized GPUs you likely don't have fast interconnect to run TP efficiently anyway. You should probably be using PP instead, which can already be unevenly split across GPUs using VLLM_PP_LAYER_PARTITION"
- **Date:** closed Aug 3, 2026 (close event timestamp 2026-08-03T14:03:55Z).
- **Evidence:** READ BODY

#### C44. TurboQuant KV cache + long-context chunked prefill — a workspace ratio that cannot be tuned around; reporter-closed as duplicate
- **WHO:** vLLM user (issue author), issue #46767; closed by the reporter as a duplicate
- **URL:** https://github.com/vllm-project/vllm/issues/46767
- **Closure status:** closed as not planned (`stateReason: NOT_PLANNED`) by the reporter themselves as a duplicate — not a maintainer rejection. The live siblings #41565 and #43357 remain open.
- **STATED REASON, VERBATIM:** "Closing as a duplicate of #41565 (TurboQuant `_continuation_prefill` workspace under-sizing at long context). I posted the quantified ~1.43x scaling analysis (util-independent; need and cap both scale with --max-num-batched-tokens) as a comment there. Also see #43357."
- **Date:** created 2026-06-25; closed by the reporter
- **Evidence:** READ BODY

#### C45. Avoiding recomputation of the last-token hidden states for spec decoding — declined in-thread as not worth the complexity
- **WHO:** vLLM maintainer NickLucche, replying to RFC #31064 (author xinyu-intel)
- **URL:** https://github.com/vllm-project/vllm/issues/31064
- **Closure status:** closed as completed, but the requested generalisation was declined in-thread by a maintainer; the shipped artifact is a different, narrower debug connector.
- **STATED REASON, VERBATIM:** "I have an aversion to adding this complexity in order to avoid running a single forward pass for 1 token --- so just want to make sure I understand why the current impl is not working well"
- **Date:** created 2025-12-20T05:56:32Z; closed 2025-12-20 (same day)
- **Evidence:** READ BODY

#### C46. Dynamo KVBM disk→device onboarding (and KVBM itself) — closed as no longer supported
- **WHO:** `jthomson04` (Dynamo maintainer), `sachalmalick`, `pjdurden`, `bennorris123`
- **URL:** https://github.com/ai-dynamo/dynamo/issues/12750 | https://github.com/ai-dynamo/dynamo/pull/13184
- **Closure status:** closed as not planned by a human maintainer (`jthomson04`). A scope/support decision, not confirmation that the reported disk-onboarding performance problem is fixed. The companion PR #13184 was closed without merging; the verifier retrieved a fuller direct closure comment on the PR itself.
- **STATED REASON, VERBATIM:** "Closing as not planned because KVBM is no longer supported. This is a scope/support decision, not confirmation that the reported disk-onboarding performance problem is fixed; the corresponding PR #13184 will be closed without merging." — and, the PR #13184 closure comment itself (verifier-retrieved, replacing the cross-reference): "Closing without merge because KVBM is no longer supported. The corresponding issue #12750 has been closed as not planned; this is a support/scope decision, not a claim that the layout or performance problem was fixed."
- **Date:** #12750 created 2026-08-06, closed 2026-09-10; PR #13184 created 2026-08-13, closed 2026-09-10
- **Evidence:** READ BODY

#### C47. Dynamo KVBM lifecycle bug — second independent confirmation of the KVBM support removal
- **WHO:** `jthomson04` (CONTRIBUTOR)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/13285
- **Closure status:** closed as not planned by a human maintainer (`jthomson04`); a distinct closure text from #12750’s (different PR #13297, different bug), produced from the same template.
- **STATED REASON, VERBATIM:** "Closing as not planned because KVBM is no longer supported. This is a scope/support decision, not confirmation that the reported lifecycle bug is fixed; the corresponding PR #13297 will be closed without merging."
- **Date:** maintained comment 2026-09-10
- **Evidence:** READ BODY

#### C48. Dynamo KVBM GPU HBM hit-rate exposure — closed with an explicit apology and rationale
- **WHO:** `dagil-nvidia` (NVIDIA/Dynamo maintainer)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/5716
- **Closure status:** closed as not planned by a human maintainer (`dagil-nvidia`) — a deliberate “we would rather close than leave you hanging”.
- **STATED REASON, VERBATIM:** "The Dynamo project has been moving fast and unfortunately this one slipped through the cracks. We don't currently have someone actively working on it, and to be honest with you we'd rather close out than leave you hanging indefinitely."
- **Date:** created 2026-01-28, closed 2026-05-26
- **Evidence:** READ BODY

#### C49. Dynamo KVBM documentation — “feature is being sunset”; docs re-add explicitly not planned
- **WHO:** Ben Hamm (Dynamo), commit author
- **URL:** https://github.com/ai-dynamo/dynamo/commit/62d0cf34a53cbef0d067f8a9d59d2031a2d30935
- **Closure status:** documentation removal merged; KVBM deprecation announced in the catalog entry. Not yet deleted from the tree.
- **STATED REASON, VERBATIM:** Commit subject: "docs(recipes): drop KVBM references — feature is being sunset". Commit body: "Remove the agg-kvbm walkthrough pointer from the Qwen3-32B recipe and mark the catalog gap note accordingly. No docs re-add planned; source assets under recipes/qwen3-32b/vllm/agg-kvbm/ are untouched pending KVBM deprecation." The edited catalog entry reads, verbatim: "'agg-kvbm single-GPU walkthrough removed from the recipe entirely (2026-06-12): KVBM is being sunset, so no docs re-add is planned. Source assets at recipes/qwen3-32b/vllm/agg-kvbm/ remain untouched pending KVBM deprecation.'"
- **Date:** 2026-06-12
- **Evidence:** READ BODY

#### C50. Dynamo host-tier KV mirroring, offload and storage APIs — closed as superseded/deferred
- **WHO:** `hutm` (CONTRIBUTOR), GMS umbrella #10450 and children
- **URL:** https://github.com/ai-dynamo/dynamo/issues/10455
- **Closure status:** closed as not planned / superseded-deferred by a human contributor (`hutm`).
- **STATED REASON, VERBATIM:** "Closing this ticket as superseded/deferred, not completed."
- **Date:** closed 2026-07-13
- **Evidence:** READ BODY

#### C51. Dynamo GMS-managed KV umbrella retired — the SGLang, TRT-LLM, host/storage, router/P2P, TP=2 and Kubernetes scopes dropped from the MVP
- **WHO:** `hutm` (CONTRIBUTOR)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/10450
- **Closure status:** closed as not planned by a human contributor (`hutm`); all 18 still-open PRs from the umbrella were closed without deleting their branches.
- **STATED REASON, VERBATIM:** The SGLang, TensorRT-LLM, host/storage, router/P2P, TP=2, Kubernetes, and optimization scopes from this umbrella are not being carried into the MVP. The linked child tickets are being closed as superseded/deferred, not as completed. … "All 18 PRs that were still open from this umbrella have been closed without deleting their branches."
- **Date:** closed 2026-07-13
- **Evidence:** READ BODY

#### C52. llama.cpp DeepSeek-V4 prefix caching reprocesses the full prompt — closed not planned two days after the last comment
- **WHO:** `am17an` (CONTRIBUTOR); reporter `jaholmesuk`
- **URL:** https://github.com/ggml-org/llama.cpp/issues/25567
- **Closure status:** closed `not_planned` by a human contributor (`am17an`) — not a stale-bot closure (no stale label), though no explicit “won’t fix” sentence was retrieved.
- **STATED REASON, VERBATIM:** `am17an`: "The frozen anchor problem will still be there though. DSV4 can't do a partial seq_rm so it will restore to the nearest checkpoint instead and has to re-prefill everything after that. That's your +278 per request." Reporter's measurement: "So it reprocesses the full prompt on every request; no prefix reuse engaged over 20 passes up to ~7.7k tokens."
- **Date:** closed 2026-07-13
- **Evidence:** READ BODY

#### C53. llama.cpp canonical KV/long-context performance boundary — maintainers decline to invest in synthetic benchmarking
- **WHO:** `JohannesGaessler` (CONTRIBUTOR), `ngxson` (COLLABORATOR)
- **URL:** https://github.com/ggml-org/llama.cpp/issues/18722
- **Closure status:** closed `not_planned`. The retrieved page’s timeline was paginated and did not contain the ClosedEvent, so the closer cannot be named; the quoted maintainer statements are the reason on record.
- **STATED REASON, VERBATIM:** `JohannesGaessler`: "If at all possible we should measure performance using real models rather than synthetic benchmarks. I cannot speak for any of the other maintainers but I will not be investing any of my time into synthetic benchmarking." … "If you want to implement something other than what I'm describing you'll have to find someone else with review power." `ngxson`: "So, I don't see why we need to adding such test case in `llama-bench` for now."
- **Date:** created 2026-01-09, closed 2026-03-03
- **Evidence:** READ BODY

#### C54. SGLang HiCache host hits returned stale MoE sidecar data — closed not planned, with the fix pursued elsewhere
- **WHO:** `Cerdore`
- **URL:** https://github.com/sgl-project/sglang/issues/26975
- **Closure status:** closed `NOT_PLANNED` (2026-07-10) after a human root-cause comment. VERIFIER CORRECTION: the evidence file’s “guard PR #27067 shipped instead of the fix” conclusion is UNSUPPORTED — #27067 is itself the migration FIX (`fix(hicache): migrate captured indexer-topk on host-tier cache hits`), closed unmerged on 2026-06-05 as superseded by #27326. The verifier downgrades this row to READ BODY with the conclusion unsupported.
- **STATED REASON, VERBATIM:** the routed_experts capturer (state_capturer/) indexes its buffer by device KV-pool slot. HiCache load_back relocates tokens to new slots without migrating that sidecar data, so a host-tier hit returns stale rows from whatever previously occupied the new slot. MoE is skipped on cache hits, so there's no recomputation to fix it. Plain radix cache is unaffected (slots stay in place). Same structural hazard in enable_return_indexer_topk. … "Opened #27067 as a guard: hard error on the unsafe combo until the migration path lands. Auto-disabling either flag silently produces a different wrong result, so failing fast is the safer option."
- **Date:** created 2026-06-01, closed 2026-07-10
- **Evidence:** READ BODY (verifier downgrade: conclusion UNSUPPORTED)

#### C55. TensorRT-LLM reusing KV cache across different LoRAs — maintainer “not planned”
- **WHO:** filed by `ShuaiShao93`; closed by `karljang` (COLLABORATOR)
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/issues/11874
- **Closure status:** closed as not planned by a maintainer (`karljang`) — a direct maintainer “no” in that repository’s KV-reuse space. (The typo `apprach` and the double space are in the original.)
- **STATED REASON, VERBATIM:** Closing as not planned for now.  The aLoRA apprach is interesting and we'll keep it on the radar.\nThank you! (the typo `apprach` and the double space are in the original). Request verbatim: "Not sure if there are other options, but anything that supports reusing kv cache across different loras is useful for us."
- **Date:** created 2026-03-03, closed 2026-04-27. Labels: `KV-Cache Management`, `Lora/P-tuning`
- **Evidence:** READ BODY

#### C56. vLLM pipeline parallelism halves the usable KV block budget — answered by a maintainer as by-design
- **WHO:** reporter (issue body), answered by andoorve (vLLM maintainer)
- **URL:** https://github.com/vllm-project/vllm/issues/7039
- **Closure status:** closed as not planned by the stale bot AFTER a substantive maintainer answer; the maintainer’s “this is expected behaviour / the halving is fundamental” statement is the real human reason.
- **STATED REASON, VERBATIM:** This is kind of expected behavior based on what our implementation of PP aims to do. We report 1650 blocks because this is the total number of blocks available on your GPU. However, this gets divided into 2 KV cache sections to support multiple request streams at the same time. This is what allows us to have pipelining of request streams at the same time with load balancing of the two streams. This reporting could possibly be improved to reflect this. … "The situation you are talking about (very long prompt) is possible with some changes. You can submit a feature request for the same if there's a very clear use case for it. However, it wasn't a main focus until now since basically only 1 of those very long prompts could be resident in the cache at a time. This would mean essentially no pipelining, and you might want to see if tensor parallelism serves your use case better." The maintainer later restated it definitively: "So the halving is fundamental to the implementation here, whether you use 1 node or multiple the number of blocks per microbatch slot will be less compared to without pipeline parallelism."
- **Date:** created 2024-08-01, closed 2024-12-01
- **Evidence:** READ BODY

### C.3 Stale-closed (automated) — no human reason on record

#### C57. vLLM LoRA adapter reload can reuse stale prefix-cache blocks — stale-closed, never fixed
- **WHO:** vLLM issue #42125
- **URL:** https://github.com/vllm-project/vllm/issues/42125
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT — the closure reason is bot text, not a maintainer decision. A contributor offered to reproduce but no fix landed.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (preceded on the same page by "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!")
- **Date:** created 2026-05-09; closed in the 2026-09-12 stale wave (closure date from the GitHub search API)
- **Evidence:** READ BODY

#### C58. vLLM hybrid (DFlash speculative) + prefix caching IndexError in `get_temporal_copy_spec` — stale-closed
- **WHO:** vLLM issue #41884
- **URL:** https://github.com/vllm-project/vllm/issues/41884
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT — bot text only, no maintainer statement.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (also present: "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!")
- **Date:** created 2026-05-07; closed in the 2026-09-07 stale wave (closure date from the GitHub search API). Title verbatim: "[Bug][V1][Hybrid] IndexError in get_temporal_copy_spec during DFlash speculative decoding + prefix caching"
- **Evidence:** READ BODY

#### C59. vLLM `SimpleCPUOffloadConnector` + Hybrid KV Cache Manager block-pool exhaustion kills the engine — stale-closed
- **WHO:** vLLM issue #42085
- **URL:** https://github.com/vllm-project/vllm/issues/42085
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT. The bot’s closing comment was not present in the retrieved payload for this issue; the quoted label description is what the page carries. The failure is an engine-killing assertion.
- **STATED REASON, VERBATIM:** Over 90 days of inactivity (the `stale` label description as embedded in the retrieved page). No human maintainer reason was retrievable.
- **Date:** created 2026-05-08; closed in the 2026-09-12 stale wave (closure date from the GitHub search API)
- **Evidence:** READ BODY

#### C60. vLLM per-request prefix-cache-token telemetry on OTel spans — stale-closed
- **WHO:** vLLM issue #41788
- **URL:** https://github.com/vllm-project/vllm/issues/41788
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT — bot text only.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2026-05-06; closed in the 2026-09-08 stale wave (closure date from the GitHub search API). Title verbatim: "[Feature]: Add request-level OTel span attribute for cached prefix-cache input tokens"
- **Evidence:** READ BODY

#### C61. vLLM RFC to support KV cache compaction / token dropping (FastGen, DuoAttention, H2O, attention sink) — stale-closed before it was revisited
- **WHO:** `YaoJiayi` (RFC author) in vllm-project/vllm; discussed by `KuntaiDu`, `simon-mo`, `heheda12345`.
- **URL:** https://github.com/vllm-project/vllm/issues/10646
- **Closure status:** closed as `NOT_PLANNED` BY THE `github-actions` STALE BOT, not by a maintainer (ClosedEvent actor `github-actions`, labels `RFC` + `stale`). The human comments in the thread are the closest thing to a human reason, and they are not refusals.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (comment by `github-actions`, 2025-03-27T02:04:38Z), preceded on 2025-02-25T02:00:45Z by: "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!" - Human maintainer context, VERBATIM (these are the closest thing to a human reason, and they are not refusals): `simon-mo` (2024-11-26T21:45:49Z): "On the research side, @lynnliu030 is experimenting with token drop's impact on memory allocation. I think we can revisit this around EOY and discuss the exact API change." `YaoJiayi` (2024-11-26T22:43:39Z): "The key challenge for supporting more methods is the memory management side. For both attention sink and H2O, they maintain the same number of tokens across different heads and layers. If we want to support more advanced methods, we need a more flexible memory layout."
- **Date:** created 2024-11-25T23:42:43Z; closed 2025-03-27T02:04:38Z; issue last updated 2026-07-02T13:48:30Z.
- **Evidence:** READ BODY

#### C62. SGLang WLFU (weighted LFU) radix-cache eviction policy PR — stale-closed
- **WHO:** `tejas-goyal` (author) in sgl-project/sglang.
- **URL:** https://github.com/sgl-project/sglang/pull/23781
- **Closure status:** PR closed unmerged BY THE `github-actions` STALE BOT, not a human maintainer.
- **STATED REASON, VERBATIM:** "Thanks @tejas-goyal . Closing this because it has had no updates in 125 days."
- **Date:** created 2026-04-27T00:35:53Z; closed 2026-08-30T01:22:06Z.
- **Evidence:** READ BODY

#### C63. SGLang T-LRU (Tail-Optimized LRU) cache eviction policy PR — stale-closed
- **WHO:** `wenxinzhang0` (author) in sgl-project/sglang.
- **URL:** https://github.com/sgl-project/sglang/pull/21708
- **Closure status:** PR closed unmerged BY THE `github-actions` STALE BOT, not a human maintainer.
- **STATED REASON, VERBATIM:** Thanks @wenxinzhang0 . Closing this because it has had no updates in 136 days. - Additional detail from the PR description (its motivation, i.e. the workload-sensitive eviction criterion it wanted to add): "TEL-safe blocks : evicting them won't push the next turn's uncached token count above threshold ξ — these are evicted first."
- **Date:** created 2026-03-30T20:28:43Z; closed 2026-08-28T03:46:02Z.
- **Evidence:** READ BODY

#### C64. SGLang Marconi eviction PR — stale-closed while still a draft
- **WHO:** `qimcis` (author) in sgl-project/sglang.
- **URL:** https://github.com/sgl-project/sglang/pull/21862
- **Closure status:** PR closed unmerged BY THE `github-actions` STALE BOT, not a human maintainer; it was still a draft.
- **STATED REASON, VERBATIM:** "Thanks @qimcis . Closing this because it is still a draft and has not been updated in 146 days."
- **Date:** created 2026-04-01T17:40:52Z; closed 2026-08-26T01:32:50Z.
- **Evidence:** READ BODY

#### C65. SGLang fix for SLRU’s burst distortion and stale-hot-prefix stagnation (a shipped policy’s known failure mode) — stale-closed unfixed
- **WHO:** `HughLLiu` (author) in sgl-project/sglang.
- **URL:** https://github.com/sgl-project/sglang/pull/24075
- **Closure status:** PR closed unmerged BY THE `github-actions` STALE BOT, not a human maintainer. The shipped `slru` policy keeps the defect the PR described.
- **STATED REASON, VERBATIM:** Thanks @HughLLiu . Closing this because it has had no updates in 117 days. - The PR's own statement of the shipped `slru` policy's defect, VERBATIM: "Zombie Protected entries : once a prefix enters Protected, historical heat never decays, so stale prefixes can stay protected long after the workload has shifted." and "naive SLRU has burst and stale-hot-prefix failure modes". The docs for the shipped policy corroborate the trade-off from the other direction: "they hurt when prefix popularity shifts over time, because a prefix that earned a high hit count keeps its advantage after it stops being useful."
- **Date:** created 2026-04-29T16:29:18Z; closed 2026-09-02T01:20:09Z.
- **Evidence:** READ BODY

#### C66. vLLM phase-aware (think-phase vs answer-phase) KV cache quantisation — stale-closed despite a measured 58% KL reduction
- **WHO:** myProjectsRavi (author), vLLM issue #39416
- **URL:** https://github.com/vllm-project/vllm/issues/39416
- **Closure status:** closed as not planned BY THE STALE BOT (GitHub Actions), NOT by a maintainer — no human maintainer reason exists on the page. The author’s own measured result is in the body.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (from the GitHub Actions stale bot; the earlier bot comment was "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!") — and, the author’s own measured claim, from the same issue body as relayed by the S11 row: "I have published a paper with a closed-form theorem and empirical validation showing this approach cuts attention KL divergence by **58%** compared to uniform 3-bit quantization on DeepSeek-R1-Distill-1.5B, with no additional inference-time compute."
- **Date:** opened Apr 9, 2026; closed Aug 10, 2026 (state_reason `not_planned`). The author's own result, verbatim: "Theory-aligned (4/3) 4 3 0.00126 -58%" (KL divergence vs uniform 3-bit, DeepSeek-R1-Distill-Qwen-1.5B, n=50, GSM8K) — i.e. a measured 58% distortion reduction that was never picked up. A separate but related proposal is also sitting open and stale: "https://github.com/vllm-project/vllm/issues/38930" ("Entropy-adaptive per-head KV cache quantization: +8% quality over uniform at same compression", OPEN, labeled `unstale`).
- **Evidence:** READ BODY

#### C67. Dynamic Memory Compression (DMC) for vLLM — maintainer pushback on the retrofit-data cost, then stale-closed (two duplicate requests)
- **WHO:** tchaton (requester) / vLLM project; pushback from condy0919 (vLLM maintainer)
- **URL:** https://github.com/vllm-project/vllm/issues/3549 | https://github.com/vllm-project/vllm/issues/4728
- **Closure status:** closed as `not_planned`; the visible close action was performed by the automation bot, not a human. A human maintainer objection precedes it on the first issue; the second (duplicate) issue has no maintainer reason on the page at all.
- **STATED REASON, VERBATIM:** According to the paper,\n\n> In our experiments, we equip pre-existing LLMs—such as Llama 2 (Touvron et al., 2023) 7B, 13B, and 70B—with DMC by retrofitting them on a negligible percentage of the original pre-training data (~2% for 2× compression, and ~8% for 8× compression) and without adding any extra parameters to the original LLM.\n\nThe required data is large IMHO. — then the bot: "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" — and, the duplicate request #4728, where only bot text exists: This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! and then "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** #3549 created 2024-03-21, bot-closed 2025-01-02; #4728 created 2024-05-10, bot-closed 2024-11-27
- **Evidence:** READ BODY

#### C68. vLLM LoRA adapters on DeepSeek-V2 have no effect (later diagnosed as the MLA `kv_b_proj` path) — stale-closed undiagnosed, re-filed ~15 months later
- **WHO:** chenchen0611 (reporter) / vLLM project
- **URL:** https://github.com/vllm-project/vllm/issues/17155
- **Closure status:** stale-closed BY THE AUTOMATION BOT ONLY — no maintainer reason on the page. The later issue #48974 characterises it as unresolved. (Verifier note: the page for #17155 never mentions `kv_b_proj`, MLA or absorption; that mechanism comes from the later diagnosis.)
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! and then "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" The later diagnosis (#48974) states verbatim: "closest is #17155 (stale-closed, undiagnosed — consistent with this root cause)."
- **Date:** created 2025-04-25; stale-marked 2025-07-25; bot-closed 2025-08-24.
- **Evidence:** READ BODY

#### C69. Attention-sink support in MLA — requested, never implemented, stale-closed
- **WHO:** lhtin (reporter) / vLLM project
- **URL:** https://github.com/vllm-project/vllm/issues/42319
- **Closure status:** closed as `not_planned`; the only closure text is the automation bot’s — no human maintainer reason appears on the retrieved page.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! — the request body itself is verbatim: "Like normal attention, hope support sink in MLA attention." The actual close comment on the page reads: "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2026-05-11; stale-marked 2026-08-10; bot-closed 2026-09-10.
- **Evidence:** READ BODY

#### C70. vLLM FA4 MLA decode backend — requested, a volunteer offered to do it, never landed, stale-closed
- **WHO:** LucasWilkinson (reporter) / vLLM project
- **URL:** https://github.com/vllm-project/vllm/issues/35919
- **Closure status:** closed as `not_planned`; the closure text is automation-bot only — no maintainer reason on the retrieved page. The community volunteer’s offer got no maintainer response before the bot closed it. VERIFIER CORRECTION: the page says “should remain open”, not “should still be open”.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! and "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" The unanswered offer, verbatim: "Hi @LucasWilkinson , I see the sm100_mla kernels are already in csrc/attention/mla. Is the goal of this issue to hook these new kernels into the v1 backend logic? I have access to a B200 environment and would like to help with the integration and benchmarking."
- **Date:** created 2026-03-03; volunteer comment 2026-05-04; stale-marked 2026-08-05; bot-closed 2026-09-06.
- **Evidence:** READ BODY

#### C71. BFLA — a training-free sparse prefill attention backend offered for upstreaming to vLLM — stale-closed
- **WHO:** Alicewithrabbit (reference implementation at `github.com/Alicewithrabbit/BFLA`), RFC against vLLM 0.19.1
- **URL:** https://github.com/vllm-project/vllm/issues/42419
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT, NOT by a maintainer — the only closure text is bot text. This must not be read as a maintainer rejection.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! — followed by "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" The issue also carries the label text "RFC stale Over 90 days of inactivity".
- **Date:** created 2026-05-12T12:27:26Z; closed ~2026-09 (stale-bot closure; page shows no human close event).
- **Evidence:** READ BODY

#### C72. vLLM frequency- and cost-aware prefix-cache eviction RFC and its implementation PR — both stale-closed
- **WHO:** luolun (issue author), vLLM; implementation by Aminsed (PR #27539)
- **URL:** https://github.com/vllm-project/vllm/issues/23641 | https://github.com/vllm-project/vllm/pull/27539
- **Closure status:** issue closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT (`github-actions`), NOT by a maintainer; the linked implementation PR #27539 was closed by the same stale bot.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (issue #23641, posted by github-actions Bot at 2026-01-10T02:13:16Z, immediately before "github-actions Bot closed this"); for PR #27539: "This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you!"
- **Date:** issue created 2025-08-26, bot-closed 2026-01-10; PR #27539 bot-closed 2026-03-03
- **Evidence:** READ BODY

#### C73. vLLM workload-aware KV cache eviction policy PR — stale-closed unmerged
- **WHO:** Chasingdreams6 (JinboHan), vLLM
- **URL:** https://github.com/vllm-project/vllm/pull/22236
- **Closure status:** PR closed unmerged BY THE AUTOMATED STALE BOT (`github-actions`), NOT by a maintainer; labels included `stale` and `needs-rebase`.
- **STATED REASON, VERBATIM:** This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you! (posted by github-actions Bot, Feb 9, 2026, immediately before the close; the earlier mark-stale comment on Jan 9, 2026 from the same bot reads "This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this pull request should remain open. Thank you!")
- **Date:** created 2025-08-05, bot-closed 2026-02-09
- **Evidence:** READ BODY

#### C74. vLLM changing `kv_load_failure_policy` default from “recompute” to “fail” — maintainers agreed, then the stale bot closed it
- **WHO:** vLLM issue #32200 (RFC author, CC @sdavidbd @wseaton @tlrmchlsmth @hasB4K)
- **URL:** https://github.com/vllm-project/vllm/issues/32200
- **Closure status:** closed as `NOT_PLANNED`, and the only closure text is automated stale-bot text, not a maintainer decision.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! followed by "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2026-01-12T18:28:21Z; stale-bot comments dated 2026-05-21 and 2026-06-21
- **Evidence:** READ BODY

#### C75. vLLM NIXL connector silently disabling HMA — stale-closed; the duplicate/closure framing in the evidence file is unsupported
- **WHO:** vLLM issue #42024 (reporter), maintainer response citing chfeng-cs
- **URL:** https://github.com/vllm-project/vllm/issues/42024
- **Closure status:** VERIFIER CORRECTED: the issue was stale-closed by `github-actions` on 2026-09-12 (`duplicateOf: null`), NOT closed as a duplicate on 2026-08-08 (that date is the stale warning). The quoted comment is by contributor `chfeng-cs`, the author of the merged PR it cites — not a maintainer and not the closure reason. Downgraded to TITLE ONLY.
- **STATED REASON, VERBATIM:** "This duplicates #41847 which addresses the same issue. Note that a class-level supports_hma() check is insufficient for MultiConnector — the wrapper class implements SupportsHMA but effective support depends on all configured children. #41847 handles this via KVConnectorFactory.supports_hma_config() which recurses into child connectors."
- **Date:** issue reports vLLM 0.18.x; closure trace dated 2026-08-08
- **Evidence:** TITLE ONLY (verifier downgrade)

#### C76. vLLM NIXL telemetry throughput not accounting for transfer overlapping — stale-closed with no answer from the pinged maintainer
- **WHO:** vLLM issue #33170 (reporter), maintainer pinged (@NickLucche) and never replied
- **URL:** https://github.com/vllm-project/vllm/issues/33170
- **Closure status:** closed as `NOT_PLANNED`; the closure text is automated stale-bot text only — no human closure reason exists on the page.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2026-01-27T18:07:23Z; stale-bot activity 2026-05-06 and 2026-06-05
- **Evidence:** READ BODY

#### C77. vLLM P2P NCCL XpYd prefill worker crash — stale-closed with the fix still unlocated
- **WHO:** vLLM issue #25043 (reporter), with a second reporter confirming on the same page
- **URL:** https://github.com/vllm-project/vllm/issues/25043
- **Closure status:** closed as `NOT_PLANNED`; the closure text is automated stale-bot text only. VERIFIER CORRECTION: the evidence file’s date line was internally impossible; the real timeline is created 2025-09-17, human comments 2025-09-21 → 2025-10-15, stale label 2026-01-15, stale-closed 2026-02-15.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2025-09-17, stale 2026-01-15, closed 2026-02-15 (verifier-corrected)
- **Evidence:** READ BODY

#### C78. NVIDIA Dynamo heterogeneous-TP P/D in the SGLang path crashed on the recommended configuration — stale-closed
- **WHO:** NVIDIA Dynamo issue #5870 (reporter on H200 SXM, Dynamo 0.8.0)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/5870
- **Closure status:** closed as `NOT_PLANNED`; the closure text is automated inactivity text only.
- **STATED REASON, VERBATIM:** "This issue has been closed due to inactivity. If you believe this issue is still relevant, please feel free to reopen it with additional context or information."
- **Date:** created 2026-02-01T23:38:19Z; stale warning 2026-03-04, inactive closure 2026-03-09
- **Evidence:** READ BODY

#### C79. Dynamo KVBM silently provided no benefit unless an undocumented flag was set — stale-closed, negative result left in the body
- **WHO:** reporter (issue #7566); closed by the **stale bot**
- **URL:** https://github.com/ai-dynamo/dynamo/issues/7566
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE, not a maintainer decision. The negative result is in the issue body, not the closure.
- **STATED REASON, VERBATIM:** "All KVBM integration tests use --num-gpu-blocks-override (defaulting to 10 blocks in tests/kvbm_integration/common.py:530), which masks this issue. The KVBM guide ... does not mention --num-gpu-blocks-override as a requirement or recommendation, so **users following the documentation will silently get no KVBM benefit**."
- **Date:** stale-bot closed 2026-04-30
- **Evidence:** READ BODY

#### C80. Dynamo KVBM disk reads issued as many small 32 KB IOs instead of one contiguous 2 MB transfer — closed by an automation
- **WHO:** `ekaynar` (reporter); closed by `linear[bot]`, not a human
- **URL:** https://github.com/ai-dynamo/dynamo/issues/6101
- **Closure status:** `state_reason=COMPLETED` but closed by an automation (`linear[bot]`), not a human; the last human comment asking “Did you resolve this? Thanks” went unanswered.
- **STATED REASON, VERBATIM:** "when KVBM reads a full block from G3 (disk) -> G1 (HBM), NIXL issues many small 32 KB IOs instead of a single contiguous 2 MB transfer."
- **Date:** closed 2026-04-22
- **Evidence:** READ BODY

#### C81. Dynamo KVBM TTFT degradation with MTP — stale-closed, no human closure reason
- **WHO:** reporter (issue #6353); closed by the **stale bot**
- **URL:** https://github.com/ai-dynamo/dynamo/issues/6353
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE (ClosedEvent actor `github-actions`); the quoted fragment is the reported symptom, not a closure statement.
- **STATED REASON, VERBATIM:** "median TTFT increases significantly"
- **Date:** stale-bot closed 2026-04-23
- **Evidence:** READ BODY

#### C82. LMCache + Mooncake on a single node gave no benefit — stale-closed with the negative result unrebutted
- **WHO:** reporter; `xiaguan` (CONTRIBUTOR) responding
- **URL:** https://github.com/LMCache/LMCache/issues/1331
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE. The substantive negative result is the reporter’s and the maintainer’s, not the closure text.
- **STATED REASON, VERBATIM:** I tried both methods and found no benefit. What could be the reason? … "Experiment 2 found that the RDMA transmission protocol was slightly better than TCP, but not as good as Baseline." `xiaguan`: "If you're only using a single node, why not just use the local CPU backend?"
- **Date:** created 2025-08-13, bot-closed 2025-11-14
- **Evidence:** READ BODY

#### C83. vLLM selective KV cache offload (send part of the prompt to slower storage) — stale-closed after a maintainer redirect
- **WHO:** `ruocco` (author, llm-d/vLLM); `effi-ofer`, `michalmalka`
- **URL:** https://github.com/vllm-project/vllm/issues/39305
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE, with a substantive maintainer-side redirect earlier in the thread.
- **STATED REASON, VERBATIM:** `effi-ofer` (2026-04-12): "Once [multi-tier](https://github.com/vllm-project/vllm/issues/38260) is available, this can be used to offload the entire prompt but send a portion to slower storage (fs or object storage)." The author's own framing of the motivation is a negative observation about reuse: "Public traces like Mooncake or Alibaba [show](https://dl.acm.org/doi/pdf/10.1145/3773772) that 40-60% of the KV Cache is stored but never reused."
- **Date:** created 2026-04-08, bot-closed 2026-08-12
- **Evidence:** READ BODY

#### C84. vLLM topology-aware KV cache compression proposal — stale-closed with no maintainer reason
- **WHO:** `jianxinglee62-prog` (author); `jagmarques` (commenter)
- **URL:** https://github.com/vllm-project/vllm/issues/38725
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE. The only substantive technical reason on the page is a commenter’s, not a maintainer’s.
- **STATED REASON, VERBATIM:** "DeepSeek-V3's MLA compresses KV at the architecture level by using a shared latent vector. The residual KV cache is already smaller, so external compression may have diminishing returns. Worth benchmarking whether the gain is meaningful post-MLA vs. standard MHA models."
- **Date:** created 2026-04-01, bot-closed 2026-08-06
- **Evidence:** READ BODY

#### C85. vLLM KV-cache rollback / “step back” API in the engine core — maintainer rejection in-thread, then stale closure
- **WHO:** `robertgshaw2-redhat` (Red Hat, vLLM maintainer); proposer `alexbuiko-sketch`
- **URL:** https://github.com/vllm-project/vllm/issues/34698
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE after a maintainer rejection in-thread. The quoted maintainer sentence reproduces the page exactly, including its apparent typo.
- **STATED REASON, VERBATIM:** I think that adding an interface in the engine core to mutate existing sequences with a "step back" function is going to be accepted as it creates significant complexity into the core engine. However, I feel confident this feature could be implemented by leveraging vllm's prefix caching functionality. This is for example how beam search is implemented internally
- **Date:** created 2026-02-17, bot-closed 2026-09-06
- **Evidence:** READ BODY

#### C86. llama.cpp q4_0 symmetric KV cache performance regression — not reproduced; stale-closed
- **WHO:** `Zulf1qar` (reporter), `vibecodingengine` (independent test); closed by the **stale bot**
- **URL:** https://github.com/ggml-org/llama.cpp/issues/25422
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE.
- **STATED REASON, VERBATIM:** I tried to reproduce on RunPod RTX **5090** since RTX **5070** is not available there. I tested b9867, b9888, and b9890 with GLM-4.7-Flash-UD-Q4_K_XL.gguf, flash attention on, q4_0 K/V cache, --kv-unified, -c 98304, one server slot, and a fixed 15644-token synthetic prompt. I did not see a major regression on this setup. Reporter concedes upstream: "Ran few more tests on b9890 kept showing worse results, but I did just run b9924 and I got almost same result as b9867"
- **Date:** created 2026-07-07, bot-closed 2026-08-22
- **Evidence:** READ BODY

#### C87. llama.cpp KV cache in the RPC backend — the prompt cache does no useful work and its save path aborts the server; stale-closed
- **WHO:** `doornail` (reporter); closed by the **stale bot**
- **URL:** https://github.com/ggml-org/llama.cpp/issues/26128
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE.
- **STATED REASON, VERBATIM:** --cache-ram 0 fully disables the prompt cache and the crash goes away. … "The server prompt cache only earns anything when conversations outnumber slots; with -np 4 and ≤4 conversations nothing is ever evicted, so it does no useful work while its save path aborts the server."
- **Date:** bot-closed 2026-09-13
- **Evidence:** READ BODY

#### C88. llama.cpp cross-request caching gives no benefit on SWA models (always invalidated) — stale-closed
- **WHO:** reporter; closed by the **stale bot**
- **URL:** https://github.com/ggml-org/llama.cpp/issues/24587
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE.
- **STATED REASON, VERBATIM:** No actual benefit from cross-request caching (since it's always invalidated). Workaround: "Removing the `--ctx-cross-request-cache` flag".
- **Date:** bot-closed 2026-08-01
- **Evidence:** READ BODY

#### C89. TensorRT-LLM `enable_block_reuse=True` crash on Gemma 3 VSWA — stale-closed, never refuted
- **WHO:** filed by `stolorz`; closed by the **stale bot**
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/issues/13123
- **Closure status:** closed as not planned — AUTOMATED STALE-BOT CLOSURE, not a maintainer decision. A maintainer did post a genuine reproduction attempt (Gemma-3-1b on RTX PRO 6000 Blackwell) before the issue went stale, i.e. the bug timed out rather than being refuted.
- **STATED REASON, VERBATIM:** "Issue has not received an update in over 14 days. Adding stale label." — and, the stale-bot closing text on the same page: "This issue was closed because it has been 14 days without activity since it has been marked as stale."
- **Date:** created 2026-04-16, bot-closed 2026-06-11
- **Evidence:** READ BODY

#### C90. vLLM Prefill Context Parallel (PCP) for MLA models — full review cycle, then stale-closed
- **WHO:** FENP (author), reviewed by LucasWilkinson (vLLM maintainer)
- **URL:** https://github.com/vllm-project/vllm/pull/28988
- **Closure status:** closed as not planned by the STALE BOT (ClosedEvent actor `github-actions Bot`), not by a maintainer — the substantive human blockers are quoted below but no maintainer closing statement exists.
- **STATED REASON, VERBATIM:** This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this pull request should remain open. Thank you! — followed by: "This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you!" - Substantive human blockers recorded before the bot closure, verbatim: "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @FENP ." and from the automated review: "My review has identified a critical issue in the attention correction logic where DCP and PCP corrections are applied in the wrong order, which will lead to incorrect results." and from a maintainer: "I find it quite messy that _prepare_inputs modifies num_scheduled_tokens_np and scheduler_output internally. Its not very clear to the reader here thats whats happening and thats why this recomputation/re-assignment is required. I think we should try harder to keep PCP more isolated for now". - Its own hardware evidence, verbatim from the PR body's Test Plan: "vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --prefill-context-parallel-size 2" and "We evaluated the performance of PCP and TP on DeepSeek-R1 and Kimi-K2 using 4K-length inputs on the H20-3e." Note that even at `--tensor-parallel-size 1`, the configuration is `--prefill-context-parallel-size 2`, i.e. two ranks, therefore two GPUs. Benchmark rows in the same body are labelled "PCP1TP8", "PCP2TP4", "PCP4TP2", "PCP8TP1".
- **Date:** created 2025-11-19, last substantive human review activity before the stale timer, closed 2026-05-09
- **Evidence:** READ BODY

#### C91. vLLM Prefill Context Parallel for GQA FlashInfer (split-out follow-up, plus its predecessor) — stale-closed unmerged
- **WHO:** pisceskkk and LookAround0301 (authors), reviewed by LucasWilkinson
- **URL:** https://github.com/vllm-project/vllm/pull/28723 | https://github.com/vllm-project/vllm/pull/26864
- **Closure status:** #28723 closed unmerged by the STALE BOT (no “not planned” badge, no human reason); its predecessor #26864 — which carries the authors’ own negative GQA+PCP measurement — was closed unmerged earlier.
- **STATED REASON, VERBATIM:** This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you! (preceded by "This pull request has been automatically marked as stale because it has not had any activity within 90 days.") - The substantive human blocker recorded earlier in the thread, verbatim: "Left some comments on #28988 which I think similarly apply here". — and, the predecessor PR #26864, where the technique’s own measurement is the reason it could not be justified as a default: "but the performance decrease compare to the base. I run the pcp test as above command，and client send request 8K/1K ，2k/1k，1k/1k input/ouput，the test result as follows： the 1K/1K base data: the 1K/1K pcp data the 2K/1K base data: the 2K/1K PCP data Is there any error in my testing method? Why isn't PCP effective?"
- **Date:** #28723 created 2025-11-14, closed 2026-04-01; #26864 created 2025-10-15, closed 2025-11-20
- **Evidence:** READ BODY

#### C92. vLLM NCCL graph-mixing optimisation for context parallel (~1 ms TPOT win) — stale-closed unmerged
- **WHO:** FENP
- **URL:** https://github.com/vllm-project/vllm/pull/32106
- **Closure status:** PR closed unmerged by the STALE BOT; no maintainer closing comment exists — the only closure reason is bot text.
- **STATED REASON, VERBATIM:** This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you! (preceded by the stale-bot marking: "This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this pull request should remain open. Thank you!")
- **Date:** created 2026-01-11, closed 2026-05-22
- **Evidence:** READ BODY

#### C93. vLLM Decode Context Parallel failed to run on H200 — stale-closed, never resolved
- **WHO:** haic0 (reporter); pisceskkk responded
- **URL:** https://github.com/vllm-project/vllm/issues/27544
- **Closure status:** closed as `NOT_PLANNED` by the STALE BOT — no maintainer reason statement, only bot text.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! then "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2025-10-27, closed 2026-06-08
- **Evidence:** READ BODY

#### C94. vLLM DCP regression between v0.10.2 and v0.11.0 for DeepSeek-R1 — stale-closed, never root-caused
- **WHO:** Harshith-umesh
- **URL:** https://github.com/vllm-project/vllm/issues/26942
- **Closure status:** closed as `NOT_PLANNED` by the STALE BOT only — explicitly not a maintainer decision.
- **STATED REASON, VERBATIM:** This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you! then "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** created 2025-10-15, closed 2026-02-15
- **Evidence:** READ BODY

#### C95. vLLM FP8 KV cache + DCP for MLA, benchmarked on 8× and 16× RTX PRO 6000 — stale-closed unmerged
- **WHO:** grimulkan
- **URL:** https://github.com/vllm-project/vllm/pull/34795
- **Closure status:** closed as not planned by the STALE BOT; no human maintainer closing reason — explicitly bot-driven.
- **STATED REASON, VERBATIM:** This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this pull request should remain open. Thank you! then "This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you!"
- **Date:** created 2026-02-18, closed 2026-07-05
- **Evidence:** READ BODY

#### C96. SGLang context-parallel PRs — one produced incorrect long-context output, one was an unmaintained draft; both stale-closed
- **WHO:** SGLang contributor (PR #27276)
- **URL:** https://github.com/sgl-project/sglang/pull/27276 | https://github.com/sgl-project/sglang/pull/21865
- **Closure status:** both PRs closed unmerged by the STALE BOT; closure reason is bot text only in each case. Both rows are TITLE ONLY in the evidence.
- **STATED REASON, VERBATIM:** Closing this because it has had no updates in 99 days. — the technical content of the PR body was recorded as: "the sparse path is selected by a CP-blind gate and then runs with global, contiguous query geometry, which is wrong once round-robin CP strides each rank's tokens — producing incorrect long-context output". — and, the second PR (#21865), closed as an unmaintained draft: Closing this because it is still a draft and has not been updated in 146 days. — the PR body was recorded as noting "no CUDA graph support for CP".
- **Date:** closed 2026-09-12
- **Evidence:** TITLE ONLY

#### C97. vLLM T-LRU conversation-aware KV cache eviction RFC — stale-closed; NOT a maintainer decision
- **WHO:** wenxinzhang0 on vLLM (RFC issue #37823)
- **URL:** https://github.com/vllm-project/vllm/issues/37823
- **Closure status:** closed as `NOT_PLANNED` BY THE AUTOMATED STALE BOT — there is NO human maintainer reason on the page. Do not read this as a maintainer decision.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** opened Mar 22, 2026; stale-marked 2026-06-25; auto-closed 2026-07-27
- **Evidence:** READ BODY

#### C98. vLLM caching system-prompt token ids (to skip re-tokenisation) — stale-closed
- **WHO:** vLLM issue author (issue #29286 "[Performance]: cache system prompt token ids")
- **URL:** https://github.com/vllm-project/vllm/issues/29286
- **Closure status:** closed as `NOT_PLANNED` with labels `performance` and `stale`, BY THE AUTOMATED STALE BOT — no human maintainer reason on the page.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** opened 2025-11-24; auto-closed (NOT_PLANNED per page metadata)
- **Evidence:** READ BODY

#### C99. vLLM hybrid checkpoint ABI for non-KV prefix resume (RFC) — stale-closed
- **WHO:** CarlOskarRost on vLLM (RFC issue #40533)
- **URL:** https://github.com/vllm-project/vllm/issues/40533
- **Closure status:** closed as `NOT_PLANNED` with the `stale` label, BY THE AUTOMATED STALE BOT — no human maintainer reason on the page.
- **STATED REASON, VERBATIM:** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- **Date:** opened 2026-04-21; stale-marked 2026-07-23; auto-closed 2026-08-25
- **Evidence:** READ BODY

#### C100. vLLM potential use-after-free in the KV block allocator under eviction pressure — stale-closed while the candidate fix PR remains open
- **WHO:** Yunzez (reporter); closed by the stale automation. Fix PR #37164 remains OPEN.
- **URL:** https://github.com/vllm-project/vllm/issues/37076
- **Closure status:** closed as `NOT_PLANNED` by `github-actions` (stale automation), NOT by a maintainer. The candidate fix PR #37164 is still OPEN and unmerged.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (posted by `github-actions`, with the prior verbatim bot comment "This issue has been automatically marked as stale because it has not had any activity within 90 days.")
- **Date:** created 2026-03-14; stale 2026-06-15; closed 2026-07-15. The technical content that was abandoned, verbatim from flutist's comment on 2026-03-16: "When many requests sharing a common prefix are scheduled in the same step, cache_full_blocks inserts newly allocated blocks into the prefix-cache hash table before the GPU forward pass runs. A subsequent request in the same step can hit one of these blocks via get_cached_block, read its uninitialized GPU memory, and produce wrong output tokens — even at temperature=0."
- **Evidence:** READ BODY

#### C101. vLLM stats metric for `available_kv_cache_memory` — stale-closed unanswered
- **WHO:** MML-coder (requester); closed by the stale automation
- **URL:** https://github.com/vllm-project/vllm/issues/26850
- **Closure status:** closed as `NOT_PLANNED`, label `stale`, by `github-actions` (stale automation), NOT by a maintainer.
- **STATED REASON, VERBATIM:** This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you! (posted by `github-actions`; prior bot comment verbatim: "This issue has been automatically marked as stale because it has not had any activity within 90 days.")
- **Date:** created 2025-10-14; stale 2026-01-13; closed 2026-02-12. Contributing work referenced in the timeline includes commits titled verbatim "[Core] Add token-level KV cache metrics to V1 engine", but the issue itself was never resolved by a human.
- **Evidence:** READ BODY

### C.4 Reverted, withdrawn, or retracted by the author

#### C102. vLLM `--prefix-match-unit` documentation issue — withdrawn by its own author
- **WHO:** vLLM issue #54607, author sanjeevrg89 (self-closed)
- **URL:** https://github.com/vllm-project/vllm/issues/54607
- **Closure status:** closed as not planned (`stateReason=NOT_PLANNED`) by the author himself, not a maintainer and not a bot.
- **STATED REASON, VERBATIM:** "Closing this. The premise was wrong and the remaining justification did not survive checking either --prefix-match-unit exists, it does what I said could not be done, and it is documented …"
- **Date:** created 2026-08-31, closed 2026-09-01
- **Evidence:** READ BODY

#### C103. vLLM equal-block-sizes hybrid crash report — retracted by its author as a duplicate with a wrong premise
- **WHO:** vLLM issue #54199, author eak7824 (self-retracted)
- **URL:** https://github.com/vllm-project/vllm/issues/54199
- **Closure status:** closed as not planned (`stateReason=NOT_PLANNED`); the title itself reads “RETRACTED — duplicate of #53142 (my \"equal block sizes\" premise was wrong; see closing comment)”. Closed by the author.
- **STATED REASON, VERBATIM:** "**Retracting this report — the central premise is wrong, and #53142's mechanism explains our crash after all.**"
- **Date:** created 2026-08-28, closed 2026-08-29
- **Evidence:** READ BODY

#### C104. Attempted revert of the HMA GPU-side KV events feature — the revert PR itself was closed unmerged
- **WHO:** vllm-agent (automated agent account, head branch `auto-revert/pr-37688`); original feature by hickeyma
- **URL:** https://github.com/vllm-project/vllm/pull/39658
- **Closure status:** PR closed unmerged (`state=CLOSED`, `mergedTime=null`, `mergedBy=null`); the target feature PR #37688 remains MERGED. No human closure comment was retrievable — the only verbatim artifact is the commit message.
- **STATED REASON, VERBATIM:** "This reverts commit cc07dad789e9ae3f02ab6a4794800b8c1cc240c6."
- **Date:** created 2026-04-13, closed 2026-05-20
- **Evidence:** READ BODY

#### C105. A publicised “92.5% prompt-throughput collapse” and “memory paradox” for q4_0 KV — retracted by its own author
- **WHO:** dentity007, in the ggml-org/llama.cpp discussion #20969 (with attribution of the catch to u/audioen on r/LocalLLaMA)
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** self-retraction — the measurement that made q4_0 KV look catastrophic was wrong (RSS measurement, silent request failures); the corrected numbers reverse the conclusion.
- **STATED REASON, VERBATIM:** EDIT (April 1): The data below has been corrected. Original claims about a 92.5% prompt collapse and memory paradox were wrong (RSS measurement, silent request failures). See my correction reply below for accurate numbers. And the correction itself verbatim: "Correction to my post above: the benchmark data I shared was flawed. u/audioen on r/LocalLLaMA caught the methodology error and they were right." / "\"92.5% prompt throughput collapse at 64K\" -- Wrong. I measured throughput from requests that failed silently. Prompt throughput is identical across all cache types at all context lengths."
- **Date:** original post Mar 31, 2026; correction Apr 1, 2026
- **Evidence:** READ BODY

#### C106. SGLang MERGED REVERT of the `kv_cache_scheme` refactor for quantisation — it broke DeepSeek V3 FP4
- **WHO:** zhyncs (SGLang maintainer) — revert of PR #10132
- **URL:** https://github.com/sgl-project/sglang/pull/10935
- **Closure status:** MERGED REVERT — the revert landed; the refactor was undone.
- **STATED REASON, VERBATIM:** This reverts commit cd4da1f . and, under Motivation, "this breaks deepseek v3 fp4"
- **Date:** Sep 26, 2025
- **Evidence:** READ BODY

#### C107. Custom evictor implementations for the vLLM filesystem offload tier — withdrawn by the author as codebase pollution; the follow-up capacity-bound LRU attempt also closed unmerged
- **WHO:** varun-sundar-rabindranath (author, self-closed), vLLM PR #43725 "[FileSystemTierManager] Toy Evictors for FS Offloading"
- **URL:** https://github.com/vllm-project/vllm/pull/43725 | https://github.com/vllm-project/vllm/pull/52784
- **Closure status:** both PRs CLOSED unmerged by their own authors. The first has a verbatim reason; for the second (#52784) the page contains no closing comment — the last substantive content is Copilot’s automated review raising an eviction-never-triggers bug.
- **STATED REASON, VERBATIM:** Decided we dont want to pollute the main codebase. (sic — copied character-for-character, including the missing apostrophe) — and, the second attempt (#52784), whose evidence that the first was abandoned is recorded in its body: Decided we dont want to pollute the main codebase. (author varun-sundar-rabindranath, Jun 24, 2026, posted immediately before his own close of #43725; the PR was created 2026-05-27 and closed 2026-06-24). Evidence that the first attempt was abandoned, verbatim from PR #52784's body: "A previous attempt ( #43725 , "Toy Evictors for FS Offloading") was closed by the author without merging." For #52784 itself, the verbatim stated reason is: NO STATED REASON FOUND — the page contains no closing comment; the only closure evidence is the timeline line "XiaYiHann closed this Aug 19, 2026" on a PR created 2026-08-18, and the last substantive page content is Copilot's automated review raising an eviction-never-triggers bug ("When capacity_bytes is set, submit_store() only records _store_job_keys when KV events are enabled. That means get_finished_jobs() can't account for stored bytes or update the LRU (store_keys is None), so eviction never triggers and disk usage can still grow without bound.").
- **Date:** #43725 created 2026-05-27, closed 2026-06-24; #52784 created 2026-08-18, closed 2026-08-19
- **Evidence:** READ BODY

#### C108. Dynamo progressive decode-affinity yield under KV pressure — author closed it after finding the existing soft-affinity support gave the same gain
- **WHO:** anish-shanbhag (NVIDIA Dynamo), PR #14443, opened 2026-09-08, closed 2026-09-08; maintainer comment from ishandhanani
- **URL:** https://github.com/ai-dynamo/dynamo/pull/14443
- **Closure status:** PR closed unmerged by its author (human reason, not a bot): the same performance improvement was already available via soft affinity.
- **STATED REASON, VERBATIM:** Maintainer (ishandhanani): "We added a concept of hard/soft affinity in #13907 . Please leverage this and see if you can implement this PR as a custom policy" — Author (anish-shanbhag): "Thanks for flagging @ishandhanani , after some testing it looks like the same perf improvement can be achieved just by using soft affinity which is already supported, so am closing this PR"
- **Date:** opened Sep 8, 2026; closed Sep 8, 2026
- **Evidence:** READ BODY

#### C109. vLLM PoC “soft-pin recently-hit prefix-cache entries” — closed by the author as a workaround that would not fix the real bug
- **WHO:** manueldomke, vLLM
- **URL:** https://github.com/vllm-project/vllm/pull/42985
- **Closure status:** PR closed unmerged by the author (human, not a bot; not a maintainer decision), explicitly to avoid leaving a workaround in the review queue. (The S14 duplicate row for the same closure is merged here; its copy of the comment dropped the closing lines, the S9 copy is complete.)
- **STATED REASON, VERBATIM:** "Closing this too — same reasoning as #43302 (close comment). @stecasta's block-pool instrumentation on the #42948 thread (comment) shows the real bug is 131% pool overflow from the homogeneous lcm=256 physical block layout vs DSv4-Flash's {256, 64, 4, 8}-block-size KV groups. Under that overflow, some eviction every alloc cycle is mandatory; the recent-hit set in this PR can only delay the eviction of blocks that have already served a lookup hit, which by itself doesn't help the agentic-rotation case (per @stecasta's BS-sweep: 46% at BS=4, 21% at BS=8 — meaningful mitigation but not a fix). The real avenue is heterogeneous per-page-size pools so the bs=4 SWA indexer stops costing 64× its useful storage: #42082 (RFC) + #42374 (WIP). Closing this PR rather than leaving a workaround in the review queue. Thanks to everyone who looked at the earlier iterations. Refs: #42948, #42082, #42374."
- **Date:** created 2026-05-18, closed 2026-05-21
- **Evidence:** READ BODY

#### C110. vLLM popular-insert extension of the soft-pin — closed as superseded, and the superseding measurement later failed independent reproduction
- **WHO:** manueldomke, vLLM
- **URL:** https://github.com/vllm-project/vllm/pull/43191
- **Closure status:** PR closed unmerged by the author as superseded by #43302.
- **STATED REASON, VERBATIM:** "Superseded by #43302 . The protection-based approach in this PR can only mitigate the cascade collapse (max 46%/21% at BS=4/8 per stecasta's independent reproduction); the new PR closes it structurally via deliberate multi-storage with a pre-reserved pristine block pool, hitting 97%+ at the same workload. Closing."
- **Date:** created 2026-05-20, closed 2026-05-21
- **Evidence:** READ BODY

#### C111. vLLM ARC eviction for the GPU block pool, first attempt — closed by its author for a procedural reason and reopened as #27039
- **WHO:** albertoperdomo2, vLLM
- **URL:** https://github.com/vllm-project/vllm/pull/26921
- **Closure status:** CLOSED unmerged by the author (human) for a procedural reason, 7 minutes before reopening as #27039 (which merged).
- **STATED REASON, VERBATIM:** "I'll reopen the PR, messed up forcing the commit with sign off."
- **Date:** closed 2025-10-16; successor #27039 created 2025-10-16T16:21:38Z and merged 2025-11-12
- **Evidence:** READ BODY

#### C112. vLLM attention-sink-protected eviction policy request — withdrawn by its author after a collaborator’s clarification
- **WHO:** sahilmalik27 (issue author), vLLM
- **URL:** https://github.com/vllm-project/vllm/issues/36311
- **Closure status:** CLOSED (completed) by the issue author himself after a collaborator’s clarification — a withdrawn request, not a maintainer rejection.
- **STATED REASON, VERBATIM:** "Thanks for the quick clarification — you're right, and I should have read the block manager more carefully before opening this. vLLM only evicts free blocks, so sink tokens in active sequences are never at risk. The concern I was modeling (early token eviction corrupting attention distribution) applies to sliding-window KV cache modes like StreamingLLM, which vLLM doesn't implement. For standard PagedAttention, this is a non-issue. The underlying analysis (identifying sink tokens and spike channels) is still useful for quantization — knowing which hidden-state channels are outliers helps bitsandbytes/AWQ configs avoid clipping them. But that's a separate, narrower topic and doesn't belong in a KV eviction issue. Closing this. Apologies for the noise."
- **Date:** created and closed 2026-03-07
- **Evidence:** READ BODY

#### C113. Dynamo KVBM V2 transfer — merged revert, removing `BlockTransferHandlerV2` from the public API
- **WHO:** `jthomson04` (Dynamo)
- **URL:** https://github.com/ai-dynamo/dynamo/pull/5406
- **Closure status:** MERGED REVERT (`state=MERGED`, `mergedTime=2026-01-13T22:18:07Z`). The diff removes `BlockTransferHandlerV2` from the public API surface.
- **STATED REASON, VERBATIM:** Commit subject: "Revert \"feat: KVBM V2 transfer (#4068)\"" — commit body: "This reverts commit 827b8c3e511295a53351b44b6af596c85f189454." The diff removes `BlockTransferHandlerV2` from the public API surface: `-pub use transfer::{BlockTransferHandler, BlockTransferHandlerV1, BlockTransferHandlerV2};` `+pub use transfer::BlockTransferHandler;`
- **Date:** created 2026-01-13, merged 2026-01-13
- **Evidence:** READ BODY

#### C114. Dynamo KVBM documentation claimed tiers and knobs that do not exist — correction PRs opened to retract the claims
- **WHO:** Dynamo maintainers (open PRs #14012, #14011), and a counter-PR #13867
- **URL:** https://github.com/ai-dynamo/dynamo/pull/14012
- **Closure status:** documentation/claim retraction: the advertised capabilities were never implemented; the correction PRs #14012 and #14011 were open at retrieval (a counter-PR #13867 also exists).
- **STATED REASON, VERBATIM:** PR #14012 title "docs(kvbm): drop the object-storage tier that has no implementation", body: "Nothing in the runtime reads any of them." PR #14011 title "fix(kvbm): remove the transfer batch size knob that nothing reads", body: "Nothing reads it."
- **Date:** 2026 (open at time of retrieval)
- **Evidence:** READ BODY

#### C115. LMCache merged revert of its MP adapter shim
- **WHO:** LMCache
- **URL:** https://github.com/LMCache/LMCache/pull/3111
- **Closure status:** merged revert, 2026-04-23.
- **STATED REASON, VERBATIM:** "This reverts only the adapter shim (restores the pre-#3100 signature requiring parallel_strategy)."
- **Date:** 2026-04-23
- **Evidence:** READ BODY

#### C116. SGLang AMD DeepSeek-V4 unified-KV pool sizing — merged revert, then relitigated and relanded the next day with gating
- **WHO:** `Liangsheng Yin` (revert author, merged)
- **URL:** https://github.com/sgl-project/sglang/pull/38163
- **Closure status:** MERGED REVERT (`state=MERGED`, `mergedTime=2026-09-06T00:28:47Z`); the change was relanded the next day by PR #38192 with gating. No narrative reason was retrievable in the revert PR (PR comment bodies did not render for the evidence agent).
- **STATED REASON, VERBATIM:** Commit subject: "Revert \"[AMD][DSV4] Fix unified-KV pool sizing and SWA ring accounting (#30315)\"" — body: "This reverts commit 514b45fd3447a300c55a2532892a2c8539d3cd6d." No narrative reason was retrievable (PR comment bodies do not render).
- **Date:** created and merged 2026-09-06; reland merged 2026-09-07
- **Evidence:** READ BODY

#### C117. TensorRT-LLM merged revert of the FlashInfer→CuTeDSL MLA backend path about two weeks after it landed
- **WHO:** `pengbowang-nv`
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/pull/18653
- **Closure status:** MERGED REVERT (`state=MERGED`, `mergedTime=2026-09-08T08:29:00Z`), preserving the standalone CuTeDSL FMHA backend and later changes.
- **STATED REASON, VERBATIM:** "Remove the FlashInfer-to-CuTeDSL MLA backend path introduced by #17800 while preserving the standalone CuTeDSL FMHA backend and later DSA, Helix, combined-FMHA, and sysinfo changes."
- **Date:** created 2026-09-03, merged 2026-09-08 (reverting #17800, which had landed roughly two weeks earlier)
- **Evidence:** READ BODY

#### C118. SGLang zigzag prefill context parallelism — withdrawn (“temporarily unavailable”) for every current DeepSeek/GLM model
- **WHO:** SGLang project
- **URL:** https://docs.sglang.ai/advanced_features/dcp.html
- **Closure status:** feature disabled / withdrawn at the documentation level — a shipped technique pulled for the models that most need it, replaced by `interleave` with `--dp 1`. Not a PR closure. The quoted text lives in the SGLang cookbook pages; VERIFIER NOTE: the row’s headline DCP page does not contain the word “zigzag”, so the sourcing is imprecise.
- **STATED REASON, VERBATIM:** Zigzag prefill CP (`--cp-strategy zigzag`) is temporarily unavailable for DeepSeek V3.2, GLM-5, GLM-5.1, GLM-5.2, and GLM-5.3. Use `interleave` for these models and keep `--dp 1`; interleave DSA CP does not support `--dp` greater than 1. - Also verbatim from GLM-5.3 (`docs/cookbook/autoregressive/GLM/GLM-5.3.mdx` line 246): "Zigzag prefill CP (`--cp-strategy zigzag`) is temporarily unavailable for GLM-5.3. For prefill CP on CUDA, use `interleave` with `--dp 1` as shown below." - Also verbatim from GLM-5 (`docs/cookbook/autoregressive/GLM/GLM-5.mdx` line 97): "**Prefill CP on CUDA**: Zigzag (`--cp-strategy zigzag`) is temporarily unavailable for GLM-5. Use `--enable-prefill-cp --cp-strategy interleave` with `--dp 1`."
- **Date:** docs current in the retrieved SGLang tree (2026-09); line numbers in the evidence file are from a local checkout and are not reproduced here
- **Evidence:** READ BODY

#### C119. vLLM revert of “[Nixl][PD] DCP support for MLA models” after it broke DeepSeek-V4-Flash — the revert PR was itself closed unmerged in favour of a narrower fix
- **WHO:** vllm-agent (author of the revert PR), 1 commit
- **URL:** https://github.com/vllm-project/vllm/pull/54386
- **Closure status:** revert PR closed unmerged; the narrower fix #54400 was merged the same day. The substantive reason is a human/automated-agent technical statement in the PR body, not a stale timer.
- **STATED REASON, VERBATIM:** "The regression is one gated assignment. In SparseAttnIndexer.cp_kv_cache_interleave_size , the memoization is conditional: if isinstance ( get_forward_context (). attn_metadata , dict ): self . _cp_kv_cache_interleave_size = value return value When attn_metadata is not a dict , the value is never cached, so every subsequent forward re-enters get_current_vllm_config() . Outside a set_current_vllm_config() context that raises. Caching unconditionally on first access — or resolving the value eagerly in Worker.initialize_from_config , as the docstring says already happens — should fix this without a revert. If a maintainer lands that, please close this PR."
- **Date:** closed 2026-08-30
- **Evidence:** READ BODY

#### C120. vLLM ACE — Attention-Weighted Context Eviction for multi-turn agentic tool results — withdrawn by its own author after benchmarking
- **WHO:** KikoCisBot, PR author, on vLLM (PR #42645 "[RFC] feat(context): ACE — Attention-Weighted Context Eviction for multi-turn tool results")
- **URL:** https://github.com/vllm-project/vllm/pull/42645
- **Closure status:** PR closed UNMERGED by the author after a self-run benchmark; the author states the design lost to doing nothing. Not stale-bot, not maintainer-closed — an author withdrawal with measured data.
- **STATED REASON, VERBATIM:** "Withdrawing this: I measured it against doing nothing, and it lost Closing this rather than leaving it open, because I no longer believe the design in it."
- **Date:** created 2026-05-14 (first author comment 2026-05-14T14:12:48Z); closed 2026-08-10 (closedTime 2026-08-10T05:36:27Z)
- **Evidence:** READ BODY

#### C121. SGLang stripping the Anthropic billing header so Claude Code turns can share a prefix — closed unmerged by its author
- **WHO:** joninco, PR author, on SGLang
- **URL:** https://github.com/sgl-project/sglang/pull/21064
- **Closure status:** PR closed UNMERGED by the author (joninco closed this May 10, 2026); no maintainer merge, the fix did not land.
- **STATED REASON, VERBATIM:** "Anthropic SDK clients (e.g. Claude Code) include an `x-anthropic-billing-header` block in the system prompt that contains a per-request conversation context hash (`cch=...`). Because this block is the first system prompt entry and changes on every request, it produces a different token prefix each time, defeating RadixAttention prefix caching for the entire conversation."
- **Date:** created 2026-03-21, closed 2026-05-10
- **Evidence:** READ BODY

#### C122. vLLM auto-revert of “[Mamba] enable prefix cache by default” — the revert PR was closed unmerged in favour of the forward fix
- **WHO:** vllm-agent (CI failure analyzer bot-authored PR), closed by noooop
- **URL:** https://github.com/vllm-project/vllm/pull/51098
- **Closure status:** revert PR closed UNMERGED (`mergedTime` null; “noooop closed this Aug 5, 2026”). The underlying failures were real and are recorded verbatim.
- **STATED REASON, VERBATIM:** The PR removed the and not model_config.is_hybrid guard in _set_default_chunked_prefill_and_prefix_caching_args (vllm/engine/arg_utils.py), so prefix caching now defaults to on for hybrid/Mamba models, and vllm/model_executor/models/config.py now unconditionally defaults mamba_cache_mode to align" instead of "all". Two consequences: "align" mode asserts scheduler_config.enable_chunked_prefill. Any hybrid-model configuration that explicitly passes enable_chunked_prefill=False — previously valid, since prefix caching was off — now fails hard at VllmConfig validation."
- **Date:** opened 2026-08-05, closed 2026-08-05
- **Evidence:** READ BODY

#### C123. vLLM multi-storage replication for shared-prefix KV blocks — closed by its author after an independent test showed it was a regression
- **WHO:** manueldomke (vLLM contributor); independent test by stecasta
- **URL:** https://github.com/vllm-project/vllm/pull/43302
- **Closure status:** PR closed unmerged by the author. The failure was caught by an independent reproduction (`stecasta`), not by the author’s own benchmark: 0% hit rate and +10–13% wall time versus the no-cache baseline.
- **STATED REASON, VERBATIM:** Pre-reserving 4 GB of pristine blocks (the mechanism in this PR) under that constraint makes the regression strictly worse: effective pool drops ~10%, eviction pressure rises, even the chat-template prelude survivors get wiped. Hence the 0% hit rate. The mechanism does demonstrably help the one-shared-prefix-across-many-sessions shape my reproducer tested (chat-completion with large system prompt, RAG with shared retrieval context), and the 4-step #42948 issue-body reproducer hits 99.95% on A.3 with this PR. But those are narrower than the agentic-rotation case that drives the real production pain, and shipping a connector that worsens the more-common workload to land a narrower one isn't acceptable. and "Thanks to @stecasta for the careful re-test that prevented this from being merged as a regression."
- **Date:** created May 21, 2026; closed May 21, 2026. The independent test result, verbatim from stecasta in issue #42948: "| BS | N | Vanilla hit | This PR hit | This PR wall vs nocache baseline | ... | 4 | 8 | 1.0% | **0.0%** | +10% slower (348 s vs 317 s) | | 8 |16 | 0.3% | **0.0%** | +13% slower (581 s vs 514 s) |" and "Both prefix_cache_hit_rate and external_prefix_cache_hit_rate reported 0% throughout."
- **Evidence:** READ BODY

### C.5 Negative results: measured, and it did not pay off

#### C124. Elastic KV cache — reclaiming the prefill-activation reserve; built, measured, and reported as a negative result
- **WHO:** Sathishkumar Sivashanmugam (single-author paper)
- **URL:** https://arxiv.org/abs/2608.23658
- **Closure status:** published negative result — the mechanism was built and explicitly judged not to pay off (the penalty the premise requires is ~1% between chunk sizes of 8192 and 32768; simply lowering `max_num_batched_tokens` recovers more KV).
- **STATED REASON, VERBATIM:** "Having built the mechanism, we test the premise it rests on and report an honest negative result. It only pays off if a small prefill chunk size badly hurts prefill latency. In a controlled experiment injecting long prompts into a live decode load, that penalty is small (median time-to-first-token differs by about 1% between chunk sizes of 8192 and 32768 tokens), because prefill is compute bound and decode consumes only about one token per sequence per step. Simply lowering max_num_batched_tokens recovers more KV than the controller does, at nearly equal latency. The reserve also dilutes under tensor parallelism, from 16% of KV at TP1 to 2.7% at TP4. We state precisely when reclaiming the reserve could still help, and release the mechanism as a reusable userspace elastic-VMM allocator."
- **Date:** submitted 24 Aug 2026
- **Evidence:** READ BODY

#### C125. External/NVMe KV caching falls below break-even on a fast GPU — the authors’ own admission rule
- **WHO:** Joseph Kanichai, Tiziano De Matteis, Animesh Trivedi — “Building py-kvcache: A Performance Characterization of External KV Caching for vLLM with NVMe SSDs” (arXiv 2609.11744); also carried as S11-ABANDONED-44, S13-ABANDONED-18 and S5-ABANDONED-8
- **URL:** https://arxiv.org/abs/2609.11744 | https://arxiv.org/html/2609.11744v1
- **Closure status:** published negative/limiting result for the cross-tier direction on fast hardware: external KV caching should be treated as a per-setup admission decision, not a general win.
- **STATED REASON, VERBATIM:** Prefix caching can reduce the time to first token (TTFT) of long-context LLM requests by reusing previously computed key-value (KV) states, but for short prefixes or fast GPUs, recomputation can be faster than loading from an external cache. And the decisive measurement, verbatim: "Bailian trace replays improve TTFT on a weaker GPU, but on an H100 the average request falls below the break-even point and GPU memory alone retains enough prefixes. External KV caching should therefore be treated as a setup specific admission decision." — and, the same paper’s break-even and below-8k findings (the S5 row’s quotation had dropped the word “as”; corrected here): The same traces show a different outcome on the Snellius node. Their average request length is below the measured 6,203 token SSD break-even point for Qwen3 4B, on our setup, and the H100's larger GPU memory retains a large portion of the working set. As a result the TTFT of only using GPU prefix caching is identical to that of py-kvcache , while the native vLLM KV Offload implementation is 0.2 s higher. and "Below roughly 8k tokens the prefill is short enough that no storage device we measured can load the prompt faster than the GPU rebuilds it, and on a mid-range drive that threshold moves out to 32k. Therefore, external cache admission must depend on the model, GPU, SSD, and reusable prefix length rather than treating every hit as beneficial."
- **Date:** submitted 10 Sep 2026
- **Evidence:** READ BODY

#### C126. Exact composition of all cached recurrent states for hybrid prefix reuse — unnecessary and, on some architectures, harmful
- **WHO:** Yirui Liu, Ruoling Qi, Longwen Wang, Xuaner Wu, Jian Chen, Yuxin Jin, Jiawei Shao, Xuelong Li — "LinearKV"
- **URL:** https://arxiv.org/abs/2608.11231
- **Closure status:** published negative result against a specific competing technique (exact state composition, attributed to concurrent work HYPIC) inside the hybrid prefix-caching design space.
- **STATED REASON, VERBATIM:** "The algebraically principled alternative---composing all $K$ cached states into the exact full-prefix state, as concurrent work HYPIC does---is unnecessary and, on some architectures, even harmful. We compare the two across three hybrid models and three PIC selectors. On the two GDN models the two tie, both recovering most of full quality (up to $92\%$); on the Mamba-2 model, exact composition instead collapses under every selector---under EPIC, for instance, it recovers only $46.6\%$ of full quality, versus $86.8\%$ for a single cached block initializer. A single state initializer is also cheaper, cutting time-to-first-token to $0.46\times$ full prefill versus a further $5$--$17\%$ overhead for exact composition."
- **Date:** submitted 31 Jul 2026
- **Evidence:** READ BODY

#### C127. Repairing reused KV for non-prefix positions is often worse than not reusing at all (KVShareArena)
- **WHO:** Xi Shi, Qian Lou — "KVShareArena"
- **URL:** https://arxiv.org/abs/2609.10266
- **Closure status:** published negative result about the value of KV reuse outside the exact-prefix case, including a negative result for cache compression on agent reports and for trained adapters across checkpoints.
- **STATED REASON, VERBATIM:** We find that correcting positions, which needs no recomputation, is enough until a question needs several sources at once. There, only methods that pay, by re-encoding part of the cache or by training, recover half to two thirds of the gap; unrepaired caches can be worse than no cache. Cache-compression methods that are harmless on a single prompt fall significantly behind position correction on freshly written agent reports. And: "When a different checkpoint wrote the cache, training-free methods are barely affected, while an adapter trained on one checkpoint's caches loses quality."
- **Date:** submitted 9 Sep 2026
- **Evidence:** READ BODY

#### C128. Attention-score KV eviction — the selection signal contributes almost nothing (Random Attention)
- **WHO:** authors of "Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning" (arXiv 2609.03430), submitted 2026-09-03.
- **URL:** https://arxiv.org/abs/2609.03430
- **Closure status:** published negative result against the whole scored-selection family (H2O/SnapKV/Ada-KV/TriAttention-style scoring): random eviction within each head matches the strongest prior evictor while serving 32–43% higher throughput.
- **STATED REASON, VERBATIM:** We show that the selection signal contributes almost nothing. Random Attention keeps the prompt and evicts uniformly at random within each attention head, computing no score at all; across four models and six reasoning tasks it matches the strongest prior evictor while serving 32-43% higher throughput than it in vLLM deployment. Mechanism, VERBATIM: "1) the prompt is the fragile part of the cache, and most of the gap between selectors is just whether their selection signal happened to keep it; 2) the reasoning trace protects itself against eviction with redundancy at two levels, in the text (the model restates what it still needs as it works) and across attention heads (each keeps its own copy of the trace), so once the prompt is safe, a random draw retains enough copies of what the model still needs, and no score is required to pick them."
- **Date:** submitted 2026-09-03.
- **Evidence:** READ BODY

#### C129. “Protection Is (Nearly) All You Need” — structural protection dominates scoring; H2O/SnapKV/StreamingLLM/Ada-KV/QUEST collapse without it
- **WHO:** authors of arXiv 2605.18053 (38 pages, code and figure regeneration scripts released), submitted 2026-05-18.
- **URL:** https://arxiv.org/abs/2605.18053
- **Closure status:** published negative result: seven policies collapse to near-zero quality without structural protection, and with protection the simplified score-isolation variants are TOST-equivalent to LRU. Linear online credit estimation adds nothing (the paper’s own §8 “Negative Result”).
- **STATED REASON, VERBATIM:** Seven policies (LRU, H2O, SnapKV, StreamingLLM, Ada-KV, QUEST, Random) share a prompt-boundary vulnerability: without structural protection, they collapse to near-zero quality on six pure-transformer models (F1$\leq$0.064). and "With protection, simplified score-isolation variants are TOST-equivalent to LRU at $K{=}32$ ($\Delta{=}0.02$)". Overall conclusion, VERBATIM: "Overall: protection dominates; scoring differences are secondary once boundaries are guarded; per-head allocation gives a further modest gain." Its attention-mass measurement, VERBATIM: "the position-0 sink holds ${\sim}75\%$ of prefix mass, while other boundary tokens sit near ${\sim}0.41{\times}$ uniform expectation, so attention scorers retain the sink but still drop structurally critical tokens." — and, the paper’s explicit §8 negative result on learned credit estimation: Our negative result from the original online-credit experiment, bounded to a 4-feature linear estimator, suggests a cautionary lesson for linear online credit estimators in KV cache management. When the easy" structural fix accounts for the entire quality gap, there may be insufficient signal for simple learned components to exploit." Section heading, verbatim: "8 Negative Result: Linear Credit Estimation Adds Nothing". Second negative finding: "H2O-faithful's nominal advantage ( $p = 0.043$ ) does not survive Holm–Bonferroni correction." (approximate — see verification note)
- **Date:** submitted 2026-05-18.
- **Evidence:** READ BODY

#### C130. Rate–distortion survey — attention-magnitude/recency retention signals fail uniformly and irreversibly, and repeated agent compaction is almost never measured
- **WHO:** authors of "What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in LLMs and Agents" (arXiv 2607.08032), submitted 2026-07-09.
- **URL:** https://arxiv.org/abs/2607.08032
- **Closure status:** published negative characterisation across four research communities; the paper turns the observation into a benchmark proposal rather than a fix.
- **STATED REASON, VERBATIM:** Two patterns hold across the survey. At every layer the signal that decides what to keep is attention magnitude or recency, and it fails in the same way everywhere, by discarding, before the query is known and with no way to undo it, information the query later needs. - Second finding, VERBATIM (directly about this subsection's evaluation practice): "And while compression is measured carefully on single-turn long context, the repeated compaction that agents actually perform is almost never measured, and no benchmark holds one budget axis across all the layers at once."
- **Date:** submitted 2026-07-09.
- **Evidence:** READ BODY

#### C131. VarRate — SnapKV/Ada-KV-style token selection collapses 11–15 points under query-agnostic reuse
- **WHO:** authors of "VarRate: Training-Free Variable-Rate KV Cache Compression for Long-Context LLMs" (arXiv 2607.15498), submitted 2026-07-16.
- **URL:** https://arxiv.org/abs/2607.15498
- **Closure status:** published negative result on irreversible eviction, specifically naming SnapKV and Ada-KV; the paper abandons eviction in favour of rank allocation.
- **STATED REASON, VERBATIM:** Two leading training-free families are both structurally limited: token-selection methods (SnapKV, Ada-KV) score importance from an observation window and evict low-scoring tokens, but eviction is irreversible -- so when the importance signal degrades under query-agnostic reuse, accuracy collapses by 11-15 points; uniform low-rank coding keeps every token but spends equal rank everywhere, wasting budget. We observe that both failures share one cure: rank should be allocated, not evicted. Result, VERBATIM: "Because no token is dropped, it degrades by only 3.5-5.5 points where query-aware selection collapses."
- **Date:** submitted 2026-07-16.
- **Evidence:** READ BODY

#### C132. “More GPUs or a Smaller Cache?” — no cost-equivalence crossover found; extra GPUs are largely wasted spend below a model-size wall
- **WHO:** authors of arXiv 2608.23962, submitted 2026-08-25.
- **URL:** https://arxiv.org/abs/2608.23962
- **Closure status:** published negative result against the “add tensor parallelism instead of compressing the cache” answer to KV memory pressure (compression cheaper by 1.20×–2.00×; crossover at roughly 36B parameters on an 80 GB card).
- **STATED REASON, VERBATIM:** We place tensor-parallel configurations (degree 1 to 8) and KV-compressed configurations (16/8/4-bit, keep-ratios down to 0.25) on one costnormalised axis, cost per million tokens against latency, using a profiled simulator calibrated on A100, A40, and H100 hardware, and we go looking for the cost-equivalence crossover. We do not find one. Across two models (Llama-2 at 7B and 70B), three GPU types, and every level of memory relief we could construct, compression is cheaper by 1.20x to 2.00x. Boundary, VERBATIM: "A 7B model on an 80 GB device cannot exhaust its KV budget within its own context window, and the boundary that decides between the strategies is model size relative to device memory, at roughly 36B parameters for an 80 GB card. Below that wall, compression dominates and extra GPUs are largely wasted spend". Counter-finding, VERBATIM: "Tensor parallelism is the only lever that improves latency (compression makes per-token latency worse, by 8 to 93%, through batching contention)".
- **Date:** submitted 2026-08-25.
- **Evidence:** READ BODY

#### C133. VaSE — eviction often yields worse accuracy than selection-based sparse attention; a few large-magnitude value states cause catastrophic failure when evicted
- **WHO:** authors of "Value-Aware Stochastic KV Cache Eviction for Reasoning Models" (arXiv 2606.03928), submitted 2026-06-02.
- **URL:** https://arxiv.org/abs/2606.03928
- **Closure status:** published negative result on the eviction-vs-full-cache trade-off, plus a documented failure mode of magnitude-agnostic scoring.
- **STATED REASON, VERBATIM:** KV cache eviction methods reduce this cost by evicting unimportant key-value pairs from the cache, yet they often yield worse accuracy than selection-based sparse attention alternatives, which keep the full KV cache. Failure mode, VERBATIM: "First, a small fraction of value states have abnormally large magnitudes, and evicting them causes catastrophic failure where models enter repetitive reasoning loops." - Corroborating independent account of the same runaway-degeneration failure, VERBATIM (from arXiv 2608.15797): "Under aggressive budgets, this not only lowers accuracy but can also cause runaway degeneration, where the model produces incoherent or repetitive tokens until reaching the length limit."
- **Date:** submitted 2026-06-02.
- **Evidence:** READ BODY

#### C134. QJL residual correction / TurboQuantProd keys — rejected on independent measurement; MSE beats the paper’s recommended Prod path
- **WHO:** scos-lab (`turboquant` reference implementation README) and `Arclabs001` in the ggml-org/llama.cpp discussion #20969 (verifier-corrected authorship)
- **URL:** https://raw.githubusercontent.com/scos-lab/turboquant/refs/heads/main/README.md | https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** technique rejected after measurement: the reproduction concludes the paper’s recommended Prod path should not be used for attention, with the variance cost quantified. VERIFIER CORRECTION: the first quoted fragment in the S4-ABANDONED-14 row was posted by `Arclabs001`, not by `scos-lab`; only the second fragment is scos-lab’s.
- **STATED REASON, VERBATIM:** The paper recommends TurboQuantProd for Keys (unbiased inner product) and TurboQuantMSE for Values. Our experiments show **MSE for both is better** with the table "| GPT-2, b=4 | MSE (both) | Paper (Prod keys) | | PPL change | +1.1% | +6.5% |" and "**Why:** TurboQuantProd's QJL residual correction adds variance. Softmax attention amplifies variance more than bias. Low variance (MSE) beats unbiasedness (Prod) in practice." — and, the quantified variance finding in the same thread: QJL eliminates bias but explodes variance. And for attention, variance is worse than bias! with the preceding measured rows "-0.02%" and "+368.6%". And: "To clarify: the 300% PPL increase was specifically from using TurboQuantProd (the QJL residual correction method from Theorem 2 of the paper) for Key quantization at b=3 on GPT-2 (head_dim=64)."
- **Date:** README retrieved 2026-09-13; the same author's restatement in the llama.cpp thread is dated Mar 28, 2026
- **Evidence:** READ BODY

#### C135. vLLM’s own comprehensive evaluation concludes TurboQuant k8v4 is not worth using and the aggressive presets are unfit for production
- **WHO:** vLLM team (official vLLM blog, "A First Comprehensive Study of TurboQuant: Accuracy and Performance")
- **URL:** https://vllm.ai/blog/2026-05-11-turboquant
- **Closure status:** published negative result from the project that shipped the technique — the maintainers’ own recommendation is to prefer FP8 and avoid the aggressive presets.
- **STATED REASON, VERBATIM:** TurboQuant k8v4 does not provide any significant advantage over FP8. This TQ variant only provides modest KV-cache savings (2.4x vs 2x), which are not worth the consistent negative impact on throughput and latency metrics. And: "Avoid TurboQuant k3v4-nc and 3bit-nc without thorough validation. These aggressive variants can cause drastic accuracy drops that reach up to 20 points on challenging math and coding benchmarks. In addition to accuracy, their consistent performance degradation due to complex dequantization steps renders them unsuitable for production deployments." And the headline: "FP8 ( --kv-cache-dtype fp8 ) remains the best default for KV-cache quantization." Throughput verbatim: "All TurboQuant variants are strictly below BF16: on Qwen3-30B (Figure 9), ranging from 80% (k8v4) to 73% (3bit-nc)".
- **Date:** published May 11, 2026; benchmarked on vLLM 0.20.2 (commit 6ec9bbec3)
- **Evidence:** READ BODY

#### C136. turbo3/turbo4 KV cache types judged worse than plain q4_0 on perplexity over all of wiki.test.raw
- **WHO:** `ubergarm`, in the ggml-org/llama.cpp discussion #20969 (verifier-corrected authorship; the evidence file named the addressee)
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** community measurement reporting the new format loses to an existing simpler format — no adoption followed. VERIFIER CORRECTION: the post is by `ubergarm` (who opens with “@Dampfinchen”), not by Dampfinchen.
- **STATED REASON, VERBATIM:** "I tested and turbo3 and turbo4 look much worse than even q4_0 in my testing: ikawrakow/ik_llama.cpp#1509 (comment) I'm running llama-perplexity over entire wiki.test.raw . command and method shown in the linked details."
- **Date:** post dated Mar 27, 2026
- **Evidence:** READ BODY

#### C137. TurboQuant’s advertised sub-4-bit compression does not survive per-block metadata on a real 27B hybrid model
- **WHO:** a contributor running the TurboQuant × MTP correctness battery, in the ggml-org/llama.cpp discussion #20969
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** measured shortfall against the advertised ratio (2.75× at 256K instead of the ≈5× full-bit-rate ratio; ~6 bits/element once per-block norms are counted), plus a demonstration that MTP consumes the freed headroom.
- **STATED REASON, VERBATIM:** TurboQuant compresses KV by 2.75× at 256 K (16.00 → 5.81 GiB). The full-bit-rate ratio is ≈ 5×, but the per-block norms in turbo3 push effective bits/element to ≈ 6. And: "MTP roughly halves the practical ceiling on a 4090. Even with the smaller turbo3 KV (3.16 GiB at 136 K), TQ+MTP at 136 K is already at 24.02 GiB / 24.56 GiB total. The MTP draft branch's own buffers (compute, n_ubatch -scaled allocations) eat the headroom that TurboQuant freed up."
- **Date:** post in the thread (thread started Mar 25, 2026)
- **Evidence:** READ BODY

#### C138. turbo3 V-cache degrades generation-trajectory scores badly even when output-level smoke tests look clean
- **WHO:** sztlink (measurement) as reported and qualified by another contributor in the ggml-org/llama.cpp discussion #20969
- **URL:** https://github.com/ggml-org/llama.cpp/discussions/20969
- **Closure status:** measured decode-path divergence from the f16 reference that output-level smoke tests do not detect (the gap grows with model size and quantisation aggressiveness).
- **STATED REASON, VERBATIM:** @sztlink 's data on this very thread shows turbo3 V-cache degrades trajectory scores significantly on 27B (≈ 58) and dramatically on 32B (≈ 25) even when outputs look fine on smoke tests. Related in-thread finding, verbatim: "GTM says q8_0/turbo3 is near-excellent (88.83); trajectory says it has materially broken generation paths (FAIL). The gap grows with model size and quantization aggressiveness."
- **Date:** posts in the thread (started Mar 25, 2026)
- **Evidence:** READ BODY

#### C139. GPUDirect Storage (cuFile/KvikIO) prototype for KV cache — slower than every other implementation tested
- **WHO:** t348575 et al., py-kvcache (arXiv 2609.11744), §6.1 "Selecting the Storage Interface"
- **URL:** https://arxiv.org/html/2609.11744v1
- **Closure status:** negative result in a published paper — the authors built the GDS prototype and rejected it.
- **STATED REASON, VERBATIM:** "We separately tested whether GPUDirect Storage could remove the intermediate CPU transfer. We implemented a simple prototype using KvikIO, which provides Python bindings to cuFile and can access GPU buffers through GPUDirect Storage [ 28 ] . In our configuration, this path was slower than all our other implementations."
- **Date:** paper submitted 2026-09-10
- **Evidence:** READ BODY

#### C140. io_uring / xNVMe / SPDK userspace NVMe paths did not improve end-to-end TTFT
- **WHO:** t348575 et al., py-kvcache (arXiv 2609.11744), §5 "I/O engine and request size" and §6.1
- **URL:** https://arxiv.org/html/2609.11744v1
- **Closure status:** negative result in a published paper — the faster I/O engines improved synthetic microbenchmarks but not vLLM TTFT, so they were dropped from the final design.
- **STATED REASON, VERBATIM:** Correcting the backend improved storage microbenchmarks ( 8(a) ), but did not improve our vLLM TTFT. This result agreed with our prior characterization, since the workload consists of large, bandwidth limited reads and writes, so reducing overhead of I/O operations does not necessarily shorten the request's critical path. and "However, replacing the I/O path did not measurably improve TTFT in the LLM benchmark." and "High small-I/O IOPS and additional worker threads can improve a synthetic microbenchmark without improving end-to-end inference."
- **Date:** paper submitted 2026-09-10
- **Evidence:** READ BODY

#### C141. llm-d’s filesystem KV cache was deprecated mid-experiment and its staging allocation caused ~2.3× data amplification
- **WHO:** t348575 et al., py-kvcache (arXiv 2609.11744) §3.2/§5.1; llm-d project (`llm-d/llm-d-kv-cache`)
- **URL:** https://arxiv.org/html/2609.11744v1 | https://raw.githubusercontent.com/llm-d/llm-d-kv-cache/main/README.md
- **Closure status:** negative result plus upstream deprecation in a published paper; the standalone FS connector reached a final release and is now upstreamed into vLLM as the FS tier of the multi-tier offloading connector.
- **STATED REASON, VERBATIM:** However, this was deprecated while our experiments were ongoing, in favour of the secondary filesystem disk tier introduced directly into vLLM. and, on the measured defect, "The trace showed that llm-d copied and submitted data separately for each cache chunk. More importantly, its minimum staging-buffer allocation was larger than the KV data required by our test model. A request containing 4.4 GB of useful KV data consequently caused approximately 10 GB to be copied and read or written." — and, the llm-d README’s own sunset/upstream note: "**Now upstreamed into vLLM.** `llmd-fs-connector==0.23` (llm-d v0.8 / vLLM v0.23) is the **final release** — llmd-fs-backend is now the FS tier of vLLM's multi-tier offloading connector (`TieringOffloadingSpec`). All new features and support continue there"
- **Date:** paper submitted 2026-09-10
- **Evidence:** READ BODY

#### C142. SSD-backed KV cache restore is “no longer beneficial” on modern engines, and GPU Direct Storage does not rescue it (Tutti’s own measurements)
- **WHO:** Tutti authors (arXiv 2605.03375) — the S14 row for the same finding had no quote of its own and is merged here
- **URL:** https://arxiv.org/html/2605.03375v1 | https://arxiv.org/abs/2605.03375
- **Closure status:** negative results inside a published paper — SSD restore loses to DRAM (GPU bubbles >70%, ~80% with layer-wise transfers), GDS still relies on CPU intervention and leaves GPU bubble time above 70%; the paper reports a 98.3% hit-rate crossover below which the SSD tier is not beneficial.
- **STATED REASON, VERBATIM:** Moreover, as LLM engines continuously optimize inference computation, restoring KV cache from SSDs is no longer beneficial due to severe I/O bottleneck. Supporting measurements verbatim from the same section: "restoring KV cache from SSDs performs much worse than from DRAM, causing GPU bubbles to exceed 70% of total inference latency in all cases." and "Applying layer-wise transfers on SSDs (SSD-LW) further reduces I/O granularity and increases the number of operations, inflating end-to-end latency and pushing GPU bubble time to around 80% of total inference latency." — and, the same paper’s GDS verdict in §2.2: GDS ( Nvidia, 2024 ) removes CPU-GPU copies through peer-to-peer DMA, but still relies on CPU intervention to initiate each I/O, incurring substantial software overhead and limiting I/O parallelism ( DeepSpeedAI, 2025 ; Li et al., 2025 ) . Even with GDS, GPU bubble time remains high at above 70%, indicating that eliminating the CPU from the data path alone hardly alleviates the mismatch between paged KV layouts and SSD access patterns. The abstract states the same verdict verbatim: "rendering the low-parallelism CPU a severe bottleneck even with GPU Direct Storage (GDS), which still relies on CPU intervention to initiate each I/O and thus remains CPU-centric."
- **Date:** paper submitted 2026-05-05
- **Evidence:** READ BODY

#### C143. Multi-SSD KV offloading collapses to the baseline on a single SSD (Swarm)
- **WHO:** Swarm authors (arXiv 2603.17803), §8.4
- **URL:** https://arxiv.org/html/2603.17803v1 | https://arxiv.org/abs/2603.17803
- **Closure status:** the paper’s own scaling result — no benefit in the single-device configuration, which is exactly the single-node single-local-disk case.
- **STATED REASON, VERBATIM:** SSD Number. Figure 18 shows throughput as the number of SSDs increases from 1 to 8. With a single SSD, Swarm falls back to the baseline. As the number increases, its throughput scales steadily and consistently surpasses all baselines. The paper's own framing of the underlying limit is verbatim in the abstract: "Solid-state drives (SSDs) provide a cost-effective alternative, but naive SSD-based paging is fundamentally bandwidth-bound due to limited PCIe throughput and per-device bandwidth constraints."
- **Date:** paper submitted 2026-03-18
- **Evidence:** READ BODY

#### C144. KV offloading degrades accuracy on context-intensive tasks (Yandex) — low-rank projection of keys and unreliable landmarks
- **WHO:** Yandex authors, "KV Cache Offloading for Context-Intensive Tasks" (arXiv 2604.08426)
- **URL:** https://arxiv.org/abs/2604.08426
- **Closure status:** negative empirical result in a published paper (significant degradation on both Llama 3 and Qwen 3); the authors nonetheless argue the technique is salvageable with better compression and key selection.
- **STATED REASON, VERBATIM:** We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key reasons for poor accuracy: low-rank projection of keys and unreliable landmarks. Second bounded result verbatim: "Our observations suggest that some model-benchmark pairs require very large budgets (over 10%) to achieve near-lossless accuracy." The authors do NOT declare the technique dead — verbatim: "These are not conceptual problems with the offloading itself, but technical limitations that can be circumvented with better compression and key selection. Our 'view from trenches' shows that KV offloading is still a good candidate for production use across both easy and context-intensive tasks."
- **Date:** submitted 2026-04-09 (v5 2026-09-01)
- **Evidence:** READ BODY

#### C145. KV offload to host DRAM is memory-bound by construction on typical workloads — 99% of latency spent on transfers, GPUs at 28% of rated TDP
- **WHO:** authors of "Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading" (arXiv 2601.19910)
- **URL:** https://arxiv.org/abs/2601.19910
- **Closure status:** negative analytic + empirical result in a published paper: typical workloads exceed the critical cached-to-prefill token ratio by orders of magnitude.
- **STATED REASON, VERBATIM:** KV cache offloading enables long-context LLM inference by storing caches in CPU DRAM, but PCIe bandwidth limitations create severe bottlenecks. In this paper, we develops an analytical framework that derives κ_crit, the critical cached-to-prefill token ratio where execution becomes memory-bound and show typical workloads exceed this threshold by orders of magnitude. Empirical characterization reveals 99% of latency spent on transfers and serving offloaded requests results in GPU's consuming only 28% of their rated TDP, motivating our proposed optimizations for hardware interconnects, model architectures, and scheduling algorithms. Supporting figure verbatim (body §2.3, delegated retrieval): "HBM, tightly integrated on a silicon interposer, delivers TB/s, while CPU-GPU PCIe 5.0 provides only 64 GB/s—2% of HBM's bandwidth. Transferring a 50 GB KV cache takes 15 ms from HBM but 800 ms from CPU DRAM."
- **Date:** submitted 2025-12-16
- **Evidence:** READ BODY

#### C146. HERALD’s own measurement — host-memory KV fetch overtakes the forward pass at 8K context (3.12× slower; 7.61× at 32K)
- **WHO:** HERALD authors (arXiv 2606.21633), §2.3
- **URL:** https://arxiv.org/abs/2606.21633
- **Closure status:** the paper’s own negative characterisation of the PCIe tier, which is why it abandons per-step full-cache fetching.
- **STATED REASON, VERBATIM:** "However, the PCIe interconnect between host and device provides roughly two orders of magnitude lower bandwidth than HBM in GPUs, making the transfer cost a critical bottleneck. As illustrated in Figure 3 (b), at a batch size of B = 64, the per-step latency of fetching the full KV cache already surpasses the model forward latency starting from 8K context, where it is 3.12x slower. This gap widens drastically as the sequence extends, reaching 7.61x at 32K context. Even with a hypothetically perfect overlap of communication and computation, the per-step latency would still be limited by max(forward, fetch)."
- **Date:** submitted 2026-06-19 (v2 2026-08-03)
- **Evidence:** READ BODY

#### C147. Performing attention on the CPU in KV-offload systems is argued to be on the wrong side of the roofline (KVDrive)
- **WHO:** KVDrive authors (arXiv 2605.18071), §6.3
- **URL:** https://arxiv.org/abs/2605.18071
- **Closure status:** published rebuttal of a design family (“prior works advocate performing attention on the CPU in KV cache offloading systems”) — the paper argues that regime is bandwidth-roof-bound.
- **STATED REASON, VERBATIM:** "Prior works advocate performing attention on the CPU in KV cache offloading systems, arguing that CPU–GPU transfer overhead dominates GPU execution. We instead analyze why GPU-based attention is preferable in our system, using the roofline model. ... When the operational intensity falls below the threshold P, transferring data to the GPU yields no benefit, as performance is bounded by the CPU–GPU bandwidth roof. This is the regime assumed in prior studies." (approximate — see verification note)
- **Date:** submitted 2026-05-18
- **Evidence:** READ BODY

#### C148. SVD/low-rank compression of attention projections accelerates rank collapse on pretrained models (negative original finding)
- **WHO:** the "Phase Reversal" survey/experiment authors
- **URL:** https://arxiv.org/abs/2609.06341
- **Closure status:** paper reporting an explicitly negative original finding that undercuts the standard motivation for SVD-based KV/attention projection compression.
- **STATED REASON, VERBATIM:** we report an original finding that using SVD compression of attention projections actually has the opposite effect on the rank collapse of the network: while it strongly suppresses it at initialization, it accelerates on pretrained models (for GPT-2 124M, GPT-2 Medium 355M, and Pythia-160M) with minimal risk of object aliasing artifacts appearing (verified on all compression ratios) and is consistent across four rank estimation methods. The paper also states the problem framing verbatim: "Section VI identifies an open question the reviewed literature does not address: whether the compression methods of Sections III-B through III-D compound with, or counteract, the network's natural tendency toward rank collapse. ... The two regimes produce opposite answers."
- **Date:** submitted 6 Sep 2026 (v1).
- **Evidence:** READ BODY

#### C149. Pure low-rank (Tucker/SVD) KV backbones are insufficient on the value spectrum and degrade sharply past ~4×
- **WHO:** the JoLT authors, reporting their own ablation against the pure-low-rank alternative
- **URL:** https://arxiv.org/abs/2607.12550
- **Closure status:** paper finding that the low-rank-only variant does NOT pay off (it needs a quantised residual to be near-lossless), with downstream collapse documented (LLaMA GSM8K 18.5%/6.0%/1.5% at 4×/6×/8×).
- **STATED REASON, VERBATIM:** A pure low-rank backbone cannot reach near-lossless fidelity on its own, because truncation discards real energy, and on the flat value spectrum that energy is large. The residual recovers it. Additional verbatim: "Mistral degrades gracefully, losing roughly 4% perplexity per integer ratio step from 4× on, whereas LLaMA degrades sharply between 4× and 5×. We return to this split repeatedly, because it is the main caveat on an otherwise architecture-agnostic result." and "LLaMA GSM8K falls to 18.5%, 6.0%, and 1.5% at 4×, 6×, and 8× (Appendix F), the downstream face of its perplexity degradation". Also verbatim on the competing family: "A two-dimensional factorization must commit to one axis of redundancy and ignore the others, and a fixed-bit-width quantizer cannot reach the modest, intermediate compression ratios where quality is still free, because its smallest setting already overshoots them." (approximate — see verification note)
- **Date:** submitted 14 Jul 2026 (v1), revised 23 Aug 2026 (v2).
- **Evidence:** READ BODY

#### C150. Low-rank KV compression on the target model class — keys compress, values do not; Qwen3-4B is the sensitive case
- **WHO:** the KV-CoRE authors (benchmark paper, 5 domains / 16 languages)
- **URL:** https://arxiv.org/abs/2602.05929
- **Closure status:** benchmark finding that limits low-rank KV compression and specifically flags Qwen3-4B as substantially more sensitive than LLaMA-2-7B; the authors conclude future compression must be layer-aware.
- **STATED REASON, VERBATIM:** Keys are consistently more compressible than values. and "We evaluate the performance degradation of compressed models using both perplexity (PPL) and GPT-score. Figure 3 presents PPL heatmaps of Qwen3-4B and LLaMA-2-7B on the Alpaca dataset across a grid of KV-cache compression ratios ... We can see that LLaMA-2-7B remains relatively stable, showing only modest PPL increases even under aggressive compression, whereas Qwen3-4B is more sensitive, exhibiting substantial degradation." Also verbatim on why uniform compression fails: "This heterogeneity indicates that future KV-cache compression should be layer-aware: applying a uniform compression ratio risks overly degrading high-rank layers while missing opportunities for more aggressive reduction in lower-rank ones."
- **Date:** submitted 5 Feb 2026 (v1), revised 7 Feb 2026 (v2).
- **Evidence:** READ BODY

#### C151. Post-training SVD of projection weights for KV savings — achievable compression is bounded, and compressing Q and K together is catastrophic
- **WHO:** the "Thin Keys, Full Values" authors
- **URL:** https://arxiv.org/abs/2603.04427
- **Closure status:** paper concluding the retraining-free variant does NOT pay off beyond moderate ratios (K-only SVD at rank 192 degrades by 26%), while the obvious stronger form destroys attention patterns.
- **STATED REASON, VERBATIM:** Table 5 reveals a striking asymmetry. Compressing both W_Q and W_K is catastrophic—at rank 192, errors in Q and K compound through the softmax, producing a cross-term Δ_Q Δ_K^⊤ that destroys attention patterns. However, compressing K alone is far more forgiving: rank 384 (d_model/2) incurs only 2.0% degradation. and "Post-training SVD of W_K alone provides a viable, retraining-free path to moderate KV cache savings. However, the achievable compression is limited: K-only SVD at d_model/4 (rank 192) degrades by 26%, whereas training from scratch with d_select=d_model/4 costs only 4.3% on WikiText-103."
- **Date:** v1 submitted 16 Feb 2026; announced March 2026; the 28 March 2026 revision is v4, not v2 (verifier correction).
- **Evidence:** READ BODY

#### C152. MLA’s shipped “MQA-absorb-only” decoding design does not pay off off-H100 (hardware coupling)
- **WHO:** the GQLA authors, analysing the deployed MLA design
- **URL:** https://arxiv.org/abs/2605.15250
- **Closure status:** paper concluding that the shipped MLA decode path is a regression relative to its own goals on non-H100 hardware — three coupled drawbacks: hardware coupling to H100, loss of head-axis tensor parallelism, and zero MTP gain on commodity inference GPUs.
- **STATED REASON, VERBATIM:** We identify three coupled hardware drawbacks of MLA's MQA-absorb-only design: hardware coupling to H100, loss of head-axis tensor parallelism, and zero MTP gain on commodity inference GPUs. and "On the NVIDIA H100, whose BF16 roofline (Williams et al., 2009) ridges around 295 FLOPs/byte, the absorbed MQA path with the canonical configuration ... and single-token decoding lands its arithmetic intensity at ≈242 FLOPs/byte, just below the ridge. This perfect H100 fit, however, is the only operating point MLA exposes."
- **Date:** v1 submitted 14 May 2026; the 21 July 2026 revision is v3, not v2 (verifier correction).
- **Evidence:** READ BODY

#### C153. FP8 KV cache and NVFP4 KV cache PRs on SM120 — measured slower, or unmerged with critical bugs
- **WHO:** reported by piweb-labs in vLLM RFC #43235 (the abandoned artifacts are #41402 and #41404, which were not retrieved directly — the quote below is the third-party statement on the page that WAS retrieved)
- **URL:** https://github.com/vllm-project/vllm/issues/43235
- **Closure status:** VERIFIER CORRECTION: the abandonment premise is REFUTED — #41402 is CLOSED with `stateReason: COMPLETED` (not “closed without resolution”) and #41404 is a pull request, not an issue. The quoted third-party text is on an OPEN RFC page (#43235). The verifier also found the evidence file’s second quote misattributed: it comes from a ROCm/gfx950 (MI355X) report about `rope_kvcache` slowdowns, whose surrounding text says the opposite (“It was not a deadlock … a factor of 8 is indistinguishable from a hang from the outside”). What survives is the measured SM120 negative result quoted below.
- **STATED REASON, VERBATIM:** "MTP + TP>1 + concurrent>1 produces deadlocks (see #41402, #41404 — both closed without resolution)." — and, the same RFC page, on the FP8/NVFP4 paths (the evidence file’s third quotation, from issue #55687, is a different hardware stack and is dropped here per the verification report): "FP8 KV Cache is slower than BF16 on SM120 because there are no native FP8 block-scaled compute units. NVFP4 KV Cache PRs (#21601 in SGLang) remain unmerged with critical bugs."
- **Date:** created 2026-05-20; stale-marked 2026-08-19.
- **Evidence:** READ BODY (verifier correction applied)

#### C154. LMCache CacheBlend — non-prefix reuse decays back to the cold baseline under sustained novel RAG traffic
- **WHO:** `shawnguyen-tensormesh` (author and closer), LMCache issue #4663 (also filed as S7-ABANDONED-2)
- **URL:** https://github.com/LMCache/LMCache/issues/4663
- **Closure status:** closed as not planned by the issue author (comment “Closing here”, verifier-corrected) after the measured decay was reported: once L1 saturates, LRU evicts the warm document chunks and the speedup decays to ~1.0–1.1× within ~1–3 batches (~30–90 queries).
- **STATED REASON, VERBATIM:** Once L1 saturates, LRU evicts the **warm document chunks** (the only thing the matcher can actually reuse), so the speedup **decays from a post-warm burst down to the cold baseline within a few dozen queries.** Nothing is functionally wrong; the store/lookup **asymmetry** is the defect. Production impact: any benchmark that re-warms before each measurement reports the burst (e.g. 4–6×) and completely misses that steady-state throughput under real sustained novel traffic falls back to ~1×. Also: "**Verdict:** **CONFIRMED** - reproduced live and grounded in source." and "Sustained uniform-novel: decays to **~1.0–1.1×** (cold floor) within ~1–3 batches (~30–90 queries)."
- **Date:** created 2026-08-20, closed 2026-08-21
- **Evidence:** READ BODY

#### C155. DCPP (dynamic chunk resizing for chunked-prefill pipeline parallelism) — measured to LOSE at 512K
- **WHO:** Yan Shi, Xiaochao Wang et al. (Huawei Technologies / Shanghai Jiao Tong University), the VPP paper's own negative result
- **URL:** https://arxiv.org/abs/2608.26523
- **Closure status:** paper concludes the technique does not pay off at long context: +10.5% compute, +36.1% exposed communication, 20.47 s residual bubble, a 4.6% TTFT regression and 4.4% throughput loss at 512K (it wins only in the 64K–256K band).
- **STATED REASON, VERBATIM:** Existing approaches mitigate this imbalance through dynamic chunk resizing (Dynamic CPP, DCPP), but our measurements show that this trades scheduling overhead for load balancing, which becomes unfavorable on long sequences. and, with the numbers: "At 512K, where the optimal chunk-size budget is 24K, profiler traces show that CPP requires 22 invocations, whereas DCPP requires 97, or 4.4 × 4.4\times as many." and "However, these savings are outweighed by the cost of finer-grained execution. Computation time increases by 10.5% (+26.48 s) as smaller chunks reduce operator efficiency, while the increased number of chunk boundaries raises exposed communication by 36.1% (+6.01 s). Despite dynamic balancing, 20.47 s of bubble remains, and DCPP ultimately incurs a 14.32 s (4.6%) TTFT regression over CPP." and "DCPP outperforms CPP on 64K–256K sequences, delivering consistent 1.6–3.6% improvements in both throughput and TTFT. However, its advantage progressively diminishes with sequence length and reverses at 512K, resulting in a 4.4% throughput loss and a 4.6% TTFT increase." and the generalisation: "These observations yield a critical insight: DCPP trades scheduling overhead for load-balancing gains. As sequence length grows, however, DCPP requires progressively more invocations, and the resulting fragmentation overhead eventually outweighs the benefits of bubble reduction."
- **Date:** submitted 27 Aug 2026.
- **Evidence:** READ BODY

#### C156. Dropping a KV row and observing no accuracy loss as validation of an eviction policy — shown to be an unsound inference
- **WHO:** Zefeng Cai, Zerui Cai (Independent Researchers) — "Compute Globally, Materialize Locally: The Memory Contract of Sparse Event-KV"
- **URL:** https://arxiv.org/abs/2607.23693
- **Closure status:** published negative result that undercuts the standard validation methodology of long-context KV eviction / episodic-memory designs; the paper also shows that YaRN position extension widens rather than closes the null.
- **STATED REASON, VERBATIM:** Long-horizon agents increasingly reuse their KV cache as memory: a serving system keeps a subset of cached entries and drops the rest. Eviction and episodic-memory schemes therefore rest on a premise rarely tested directly, that a retained event is still informative once the observations that produced it are gone. and the conclusion, verbatim: "One consequence follows for anyone who evicts. An ablation that drops a source event and observes no accuracy loss has not shown that the source was unnecessary; it may have retained a row that already carried the answer." and "For anyone who evicts the corollary is that dropping a source event and observing no accuracy loss does not show the source was unnecessary." - Second, separate negative inside the same paper (position-extension makes the null WORSE, not better): "16 16 of 20 20 conversations exceed Qwen3's native window, where restoring range with YaRN widens the null rather than closing it (both arms re-run together on one stack: LoCoMo − .048 -.048 [ − .103 , + .009 ] [-.103,+.009] without scaling, − .121 -.121 [ − .181 , − .068 ] [-.181,-.068] with it; paired per-conversation change − .074 -.074 , p = .02 p{=}.02 )."
- **Date:** arXiv:2607.23693v1 [cs.AI] 26 Jul 2026.
- **Evidence:** READ BODY

#### C157. Final-answer accuracy as the metric for KV compression at long context — an asymmetric diagnostic; the “answer–evidence gap”
- **WHO:** Mengting Ai, Jingrui He, Yue Guo (University of Illinois Urbana-Champaign) — "Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression"
- **URL:** https://arxiv.org/abs/2608.01631
- **Closure status:** published negative result whose own framing undercuts accuracy-only evaluation of long-context KV compression, including token-eviction families (H2O/SnapKV-class).
- **STATED REASON, VERBATIM:** KV cache compression is commonly evaluated by final-answer accuracy, implicitly assuming that preserving the answer also preserves the reasoning that supports it. We test this assumption for large reasoning models and show that it can fail: under compression, correct answers and the validity of their visible supporting rationales can be preserved at different rates. and "Across tasks, token-eviction methods can preserve competitive final-answer accuracy while substantially degrading chain support or perturbation faithfulness. We call this the answer-evidence gap." and "A coverage-preserving quantization control is substantially less affected, suggesting that the failure is tied less to KV memory reduction itself than to losing access to parts of the reasoning trace." and "we show that the final–answer accuracy is an asymmetric diagnostic for compressed reasoning. While collapsed accuracy reveals compression damage, preserved accuracy can hide evidence loss, creating an illusion of competence where correct answer lacks supported evidence." - Task-dependence, verbatim (which is what makes it usable as a scoping result): "Our results show that the answer–evidence gap is strongly task-dependent: on AIME, GPQA-Diamond, and MedCalc, compressed models can maintain competitive accuracy while reasoning validity or faithfulness substantially degrades. In contrast, on retrieval tasks such as RULER QA, where answers depend more directly on retaining exact supporting evidence, compression damage is more visible through accuracy collapse."
- **Date:** arXiv:2608.01631v1 [cs.CL] 03 Aug 2026.
- **Evidence:** READ BODY

#### C158. Fixed-parameter eviction policies under workload mismatch — degrade by up to 2.7× (SAECache’s own comparison)
- **WHO:** SAECache authors (arXiv 2605.18825), self-reported negative result about the fixed-parameter class
- **URL:** https://arxiv.org/abs/2605.18825
- **Closure status:** published paper reporting that the fixed-parameter alternative degrades badly — a negative result about a whole class of KV eviction policies, stated by the authors.
- **STATED REASON, VERBATIM:** "Through extensive evaluation across heterogeneous workloads, we demonstrate that SAECache achieves 1.4x-2.7x TTFT improvement over production-style baselines, while fixed-parameter alternatives can degrade by up to 2.7x under workload mismatch -- a failure mode our adaptive approach avoids entirely."
- **Date:** submitted 12 May 2026
- **Evidence:** READ BODY

#### C159. Dynamo KVBM 0.7.0 regression — TTFT 1.27 s → 8.23 s; worked around by the reporter, never fixed
- **WHO:** reporter (issue #4774)
- **URL:** https://github.com/ai-dynamo/dynamo/issues/4774
- **Closure status:** reported regression, worked around by the reporter (server-side flag change), not fixed. VERIFIER CORRECTION: the issue was created 2025-12-05, not 2025-12-17.
- **STATED REASON, VERBATIM:** In 0.6.1 I was getting a TTFT of 1.27 seconds. Now I'm seeing a TTFT of 8.23 seconds. … "I tried --distributed-executor-backend=mp and I saw no improvement"
- **Date:** reported 2025-12-05 (verifier-corrected; no 2025-12-17 activity exists on the page)
- **Evidence:** READ BODY

#### C160. LMCache CPU offload measured as pure overhead next to vLLM’s own prefix cache
- **WHO:** `KuntaiDu` (vLLM & LMCache committer), `sammshen` (LMCache contributor)
- **URL:** https://github.com/LMCache/LMCache/issues/1109
- **Closure status:** closed `COMPLETED` — materially this is an explained-away negative result, not a fix: only 3 tokens were loaded from LMCache because vLLM’s prefix cache absorbed the hits, so the measurement was of LMCache overhead alone.
- **STATED REASON, VERBATIM:** `KuntaiDu` (2025-07-21T17:32:30Z): "Currently CPU memory size needs to be larger than GPU KV cache size in order for LMCache to show perf improvement." `sammshen` (2025-07-21T23:34:50Z): "The key thing to notice is the logs that shows that we are only loading 3 tokens from lmcache because vllm prefix caching is receiving cache hit" … "This means we are only measuring lmcache overhead and no benefits." … "lmcache will not excel here because vllm has a ton of space for prefix caching."
- **Date:** created 2025-07-21, closed 2025-07-29
- **Evidence:** READ BODY

#### C161. LMCache’s reported 25× speedup was attributable to vLLM, not to LMCache CPU offload
- **WHO:** reporter; `FeiGSSS` (LMCache maintainer) responding
- **URL:** https://github.com/LMCache/LMCache/issues/2334
- **Closure status:** `NOT_PLANNED`, closed by the AUTOMATED STALE BOT. The negative result is in the issue body and the maintainer reply, not in the closure.
- **STATED REASON, VERBATIM:** Body: "Performance comparison between vLLM standalone and vLLM with LMCache CPU Offload shows minimal difference in measured TTFT improvements, despite the same 25x speedup being reported for both configurations." … "The 25x improvement seen in both cases appears to be coming from vLLM's internal KV cache mechanism rather than LMCache's CPU offload". `FeiGSSS`: "vLLM has its own CPU KVCache offload, which is slightly slower than LMCache (as reported in the paper). I think your results is expectable."
- **Date:** created 2025-12-31, closed 2026-04-06
- **Evidence:** READ BODY

#### C162. LMCache token-dropping / compression accuracy is within run-to-run noise
- **WHO:** LMCache (PR #4500 body)
- **URL:** https://github.com/LMCache/LMCache/pull/4500
- **Closure status:** the PR merged, but its own documented measurement (accuracy carries ~1–2/30 run-to-run noise from decode non-determinism) undercuts the accuracy claim.
- **STATED REASON, VERBATIM:** "Accuracy on this workload carries ~1–2/30 run-to-run noise from decode non-determinism"
- **Date:** 2026
- **Evidence:** READ BODY

#### C163. vLLM FP8 KV-cache — the measured failure modes before the fixes, and the residual “when to avoid FP8” list
- **WHO:** vLLM (AWS + Red Hat AI authors of the blog)
- **URL:** https://vllm.ai/blog/2026-04-22-fp8-kvcache
- **Closure status:** published negative result, partially remediated; residual limits are documented in the blog’s own “When to Avoid FP8 KV-Cache” section (short contexts, head_dim 256 prefill, 91%→13% NIAH at 128k on Hopper before the fix, gpt-oss-20b break-even beyond 700k tokens).
- **STATED REASON, VERBATIM:** On Hopper GPUs, the FP8 Flash Attention 3 kernel suffered from accumulation precision loss at long contexts. On a 128k needle-in-a-haystack task, FP8 accuracy dropped from 91% (BF16 baseline) to just 13% — a regression traced to imprecise FP32 accumulation in the Tensor Cores. "The FP8 ITL slope for models with sliding-window attention layers (e.g., gpt-oss-20b) was nearly identical to BF16 (96% of BF16 slope), meaning users gained almost no decoding speedup despite halving memory. The break-even point exceeded 700k tokens — well beyond most practical context lengths." Residual: "for head dimensions larger than 128, the prefill performance remains behind BF16." And the explicit avoidance list: "FP8 KV-cache quantization is not always the right choice. Consider staying with BF16 if: Your contexts are short (< ~7k tokens): FP8 has a small constant overhead (the intercept gap), so at short contexts BF16 may be slightly faster for ITL. Your model uses head_dim = 256 and prefill latency matters: The two-level accumulation overhead increases TTFT by ~1.6x at long contexts." Quantified in their Table 3: `gpt-oss-20b before (v0.10.2) FP8 741,565 96%` break-even tokens and slope. (approximate — see verification note)
- **Date:** published 2026-04-22
- **Evidence:** READ BODY

#### C164. Seven curated KV-compression mechanisms all rejected under a pre-registered protocol at matched mean cache
- **WHO:** arXiv 2605.14292
- **URL:** https://arxiv.org/abs/2605.14292
- **Closure status:** published negative result (pre-registered protocol, held-out confirmation) on long-form mathematical reasoning with two distilled-reasoning models at budgets b ∈ {64, 128}.
- **STATED REASON, VERBATIM:** "We study seven mechanisms across these five families under matched mean cache on long-form mathematical reasoning (MATH-500~\cite{hendrycks2021math}) with two distilled-reasoning models (Qwen-7B and Llama-8B variants of DeepSeek-R1-Distill~\cite{deepseek2025r1}) at budgets $b \in \{64, 128\}$. All seven were rejected."
- **Date:** submitted 2026-05-14, revised 2026-05-17
- **Evidence:** READ BODY

#### C165. Mixed-precision KV bit allocation can be worse than uniform quantisation — “distortion model mismatch”
- **WHO:** arXiv 2605.06675 (RateQuant)
- **URL:** https://arxiv.org/abs/2605.06675
- **Closure status:** published negative result about the popular technique; the paper’s own contribution is the fix. Applying one quantiser’s distortion model to another inverts the allocation order.
- **STATED REASON, VERBATIM:** "We show, however, that such mixed-precision allocation has a hidden pitfall: each quantizer follows a different distortion curve D(b)=alpha*beta^{-b}, and the decay rate beta varies from 3.6 to 5.3 across quantizer designs. Applying one quantizer's distortion model to another inverts the allocation order and makes performance worse than uniform quantization. We call this failure mode distortion model mismatch"
- **Date:** 2026-05
- **Evidence:** READ BODY

#### C166. Per-layer FP8 KV-quantisation error attribution is non-diagnostic — errors survive on cross-layer cancellation
- **WHO:** arXiv 2607.28699 (WitCert)
- **URL:** https://arxiv.org/abs/2607.28699
- **Closure status:** published negative result: in a 28-layer sweep no single layer’s pollution alone loses anything (0/28).
- **STATED REASON, VERBATIM:** aggressive schemes survive on cross-layer error cancellation, not per-step fidelity -- in a 28-layer sweep, no single layer's pollution alone loses anything (0/28) … "raw-cast fp8 from 22.8 back to 79.7 on hard RULER tasks"
- **Date:** 2026-07
- **Evidence:** READ BODY

#### C167. Per-layer eviction routing has null results on short inputs (MATH-500, LongBench TREC)
- **WHO:** arXiv 2604.17695 (MoE-nD)
- **URL:** https://arxiv.org/abs/2604.17695
- **Closure status:** published negative results, self-labelled “null results”: the solver picks keep=1.0 on most layers for short inputs, cleanly characterising when per-layer routing has headroom to help.
- **STATED REASON, VERBATIM:** "Two null results -- MATH-500 and LongBench's TREC -- share a principled cause (short inputs, solver picks keep=1.0 on most layers), cleanly characterizing when per-layer eviction routing has headroom to help."
- **Date:** 2026-04
- **Evidence:** READ BODY

#### C168. Sliding window is order-optimal among suffix-only cache policies — a negative result bounding a whole policy class
- **WHO:** arXiv 2605.25085
- **URL:** https://arxiv.org/abs/2605.25085
- **Closure status:** published negative result bounding the suffix-only policy class; whether recurrent or propagating cache summaries can beat the scaling is left open by the authors.
- **STATED REASON, VERBATIM:** our main result characterizes the per-token memory requirement of suffix-only cache policies: a sliding-window scheme attains distortion eps with window w = O(eps^{-1/alpha}), and -- under an additional two-sided Bayes-risk condition -- a converse shows w = Omega(eps^{-1/alpha}) is necessary within this policy class … "Whether recurrent or propagating cache summaries can beat this scaling is left open." (approximate — see verification note)
- **Date:** 2026-05
- **Evidence:** READ BODY

#### C169. The standard chunked-mean fast-weight write is silently wrong — C²−C spurious cross-token outer products per chunk
- **WHO:** arXiv 2605.22884 (Tensor Cache)
- **URL:** https://arxiv.org/abs/2605.22884
- **Closure status:** published negative result about a widely used approximation.
- **STATED REASON, VERBATIM:** "the common chunked-mean training shortcut A<-lambda A+eta(kbar ⊗ vbar) silently introduces C^2-C spurious cross-token outer products per chunk" (approximate — see verification note)
- **Date:** 2026-05
- **Evidence:** READ BODY

#### C170. Prefetching cannot hide CPU-offload transfers — PCIe-bandwidth bound, not prediction-quality bound (WiSP)
- **WHO:** arXiv 2606.21868 (WiSP)
- **URL:** https://arxiv.org/abs/2606.21868
- **Closure status:** published negative result explicitly about a “natural next step”: speculative transfers compete with demand transfers instead of hiding them. (Scope caveat from the evidence: the mechanism concerns experts, not KV.)
- **STATED REASON, VERBATIM:** "A natural next step is to predict future experts and prefetch them. We find that this does not help in single-stream decode: the bottleneck is PCIe bandwidth, not prediction quality, so speculative transfers compete with demand transfers instead of hiding them."
- **Date:** submitted 2026-06-20, revised 2026-08-30
- **Evidence:** READ BODY

#### C171. Three attempts to beat LRU for cross-request KV prefix caching all failed on 68,266 real Claude Code requests plus 23,608 Mooncake requests
- **WHO:** `gauravapiscean`, independent reproduction study
- **URL:** https://github.com/gauravapiscean/agentic-kv-cache | https://raw.githubusercontent.com/gauravapiscean/agentic-kv-cache/main/README.md
- **Closure status:** published null result on real traces; the leaf restriction both major engines implement bought 0.02pp on this workload. The author also documents a harness trap (an offline Belady oracle losing to LRU).
- **STATED REASON, VERBATIM:** I replayed **68,266 requests from 393 real Claude Code sessions** and **23,608 Mooncake requests** through a prefix-cache simulator, tried to beat the production baseline three different ways, and failed. Section heading, verbatim: "## 4. Three ways to beat LRU, three failures". Result verbatim: "Monotone negative. Every component made it worse, and the one I was most confident in — coherent eviction — was the worst." And: "**LRU-leaf is a stronger baseline than the literature treats it as.** I couldn't beat it with three independent mechanisms on real traces." A directly shipped-engine-relevant finding, verbatim: "**Incidental finding:** flat block LRU and radix-leaf-restricted LRU differ by **0.02pp** on this workload. The leaf restriction both major engines implement buys essentially nothing here." Also documents a harness trap: "In my first run, Belady — an *offline oracle* — lost to LRU. That's not a result, that's a broken harness, and it's worth publishing because I expect it to be common."
- **Date:** 2026 (traces dated 062126; the study is undated in the README but references 2026 datasets)
- **Evidence:** READ BODY

#### C172. vLLM’s own docs state that disaggregated prefill DOES NOT improve throughput
- **WHO:** vLLM project (the docs page's own warning)
- **URL:** https://docs.vllm.ai/en/latest/features/disagg_prefill.html
- **Closure status:** shipped feature with an explicit negative scope statement in the project’s own documentation — the technique is published and supported, but the headline benefit is disclaimed.
- **STATED REASON, VERBATIM:** "!!! note Disaggregated prefill DOES NOT improve throughput."
- **Date:** doc current in retrieved tree (vLLM main)
- **Evidence:** READ BODY

#### C173. “Beyond the Buzz: A Pragmatic Take on Inference Disaggregation” — disaggregation is not a universal solution
- **WHO:** arXiv 2506.05508 (authors' own conclusion)
- **URL:** https://arxiv.org/abs/2506.05508 | https://arxiv.org/html/2506.05508v1
- **Closure status:** published negative/limiting result — the paper’s headline finding undercuts multi-node disaggregation for a large part of the design space.
- **STATED REASON, VERBATIM:** "As demonstrated in Figure 1, disaggregation is not a universal solution. In the following sections, we examine the performance benefits of disaggregated inference serving across a broad design space."
- **Date:** arXiv 2506.05508, retrieved 2026-09
- **Evidence:** READ BODY

#### C174. Prefix caching for small reasoning models — measured to make 7B serving WORSE
- **WHO:** Qi Li et al. (HKUST-GZ), paper "Reasoning Language Model Inference Serving Unveiled: An Empirical Study"
- **URL:** https://arxiv.org/abs/2510.18672
- **Closure status:** published empirical study whose own experiments conclude the technique does not pay off in the 7B regime (increased latency).
- **STATED REASON, VERBATIM:** "However, for 7B models, prefix caching negatively impacts efficiency, leading to increased latency."
- **Date:** submitted 2025-10-21
- **Evidence:** READ BODY

#### C175. Immediate online KV compaction for agents hurts performance; delaying compaction recovers much of the gap
- **WHO:** authors of "Practical Online KV Cache Compaction for LLM Agents: An Empirical Study"
- **URL:** https://arxiv.org/abs/2608.00902
- **Closure status:** published empirical study whose headline result is that the naive (immediate) form of the technique hurts — the negative finding is the paper’s main contribution.
- **STATED REASON, VERBATIM:** "Experiments on BrowseComp-Plus and WideSearch show that immediate compaction often hurts performance, whereas delaying compaction to use the agent's future queries recovers much of the gap. Moreover, TE is often more robust than AM under imperfect proxies."
- **Date:** submitted 2026-08-02
- **Evidence:** READ BODY

#### C176. Learned importance scoring for KV compression does not beat random selection
- **WHO:** Brady Steele — “On the Limits of Learned Importance Scoring for KV Cache Compression” (the lead-author name was not independently confirmed by the verifier; the title and abstract quotes were)
- **URL:** https://arxiv.org/abs/2601.14279
- **Closure status:** published negative result — the paper’s own headline is that the learned scorer fails across 5 seeds, 4 retention levels and 3 tasks.
- **STATED REASON, VERBATIM:** "Despite architectural sophistication (multi-horizon lookahead, cross-attention), SIP does not outperform simple baselines, including random selection, across 5 seeds, 4 retention levels, and 3 tasks."
- **Date:** submitted 2026-01-13
- **Evidence:** READ BODY

#### C177. Nexus — the off-anchor KV splice boundary, and a drift gate that does not predict drift
- **WHO:** Mustafa Arslan, "Nexus: Depth-Adaptive KV-Cache Splicing and Retrieval-Decoupled Tool Routing for Agentic LLMs on Unified Memory"
- **URL:** https://arxiv.org/abs/2608.20397
- **Closure status:** published paper containing two explicit self-declared negative results that bound the technique; the KV-splice lever is the bounded part.
- **STATED REASON, VERBATIM:** "Two negative results bound the design: the off-anchor RoPE fidelity boundary, and the failure of a reference-free drift gate to predict drift (Spearman rho = 0.193)."
- **Date:** submitted 2026-07-01
- **Evidence:** READ BODY

#### C178. Permanent eviction for reasoning chain-of-thought is catastrophic (accuracy 0–2.5%), and the reproduced SOTA eviction method lands at 0–32%
- **WHO:** Aojie Yuan, Tianqi Shen, Dajun Zhang, "Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy for LLM Reasoning"
- **URL:** https://arxiv.org/abs/2605.09490
- **Closure status:** published negative result about the dominant approach, plus a failed reproduction of the then-SOTA eviction method.
- **STATED REASON, VERBATIM:** "The dominant response -- permanently evicting low-importance tokens -- is catastrophic for reasoning: accuracy collapses to 0-2.5% when half the cache is removed."
- **Date:** submitted 2026-05-10
- **Evidence:** READ BODY

#### C179. Query-aware KV-compression evaluation — the pay-off does not survive a matched-budget audit; SnapKV loses to a trivial baseline
- **WHO:** authors of "How Query Visibility Changes KV-Cache Compression Rankings: A Matched-Budget Audit" (arXiv 2607.11942)
- **URL:** https://arxiv.org/abs/2607.11942
- **Closure status:** published negative result: under the query-agnostic protocol only KeyDiff consistently beats a best-of-3 trivial baseline, and the most widely deployed method (SnapKV) loses to “keep the start and the recent window” on average (−0.066).
- **STATED REASON, VERBATIM:** under the agnostic protocol, of the five audited methods that share a common attention backend, only KeyDiff beats a best-of-3 trivial baseline consistently (31 of 36 cells), and the most widely deployed method, SnapKV, loses to keep the start and the recent window" on average (-0.066)." and "The per-method drop between the two protocols is ordered consistently with how visible the question is to each method's scoring signal, legible in its source code: from Delta=+0.198 for SnapKV (the question sits inside its 64-token observation window) down to Delta=+0.011 for KeyDiff (its score contains no query term at all)."
- **Date:** submitted 2026-07-11.
- **Evidence:** READ BODY

### C.6 Papers whose own limitations undercut their technique

#### C180. AVMP — the paper frames its own production impact as an unmeasured hypothesis
- **WHO:** AVMP authors, arXiv 2605.22416 (the quote is in §6.3 “Production hypothesis”, not in a limitations section — verifier correction to the evidence file’s label)
- **URL:** https://arxiv.org/abs/2605.22416
- **Closure status:** paper whose own text undercuts the technique — its central production claim is explicitly labelled a hypothesis to be tested, not a measured outcome.
- **STATED REASON, VERBATIM:** "We frame the production impact of AVMP as a hypothesis to be tested, not a measured outcome."
- **Date:** submitted 21 May 2026
- **Evidence:** READ BODY

#### C181. QAQ needs attention scores that do not exist at quantisation time — worked around with an assumption rather than solved
- **WHO:** Shichen Dong, Wen Cheng, Jiayu Qin, Wei Wang — the QAQ paper’s own text (arXiv:2403.04643). Verifier correction: the evidence file’s shorthand “Cheng et al.” follows the submitting author, not the first author.
- **URL:** https://arxiv.org/html/2403.04643v2
- **Closure status:** self-declared limitation inside the paper: the method’s required attention scores are not available at quantisation time, so it invokes the persistence-of-importance assumption. No 2025–2026 revision and no implementation in either engine tree.
- **STATED REASON, VERBATIM:** "Our quantization method necessitates the attention scores that quantify the degree to which future tokens attend to preceding tokens. However, these attention scores are not available at the time of quantization. To address this limitation, we invoke the persistence of importance theory, which posits that the attention scores ( i.e. , importance) of each token remain relatively constant ( i.e. , persistence) throughout the process of token generation."
- **Date:** v1 Mar 7, 2024; v2 Apr 12, 2024. No 2025-2026 revision, and QAQ has no implementation in either engine tree (see OPEN-12).
- **Evidence:** READ BODY

#### C182. Monolithic single-cartridge KV for a whole document collection — collapses to near-chance when composed, and the replacement does not cover mid-conversation insertion
- **WHO:** Eyuboglu et al. (2025), the original Cartridges design — falsified by the CAS paper (arXiv:2606.04557), and by CAS's own limitations section
- **URL:** https://arxiv.org/abs/2606.04557
- **Closure status:** published paper concluding the earlier “cartridges” technique does not pay off at collection scale, while its own limitations section concedes that mid-conversation cartridge loading would invalidate the previously computed KV, that compression tolerance varies by document type (100× TechQA vs ≤2× FinQA), and that retrieval quality is bounded by raw-text matching.
- **STATED REASON, VERBATIM:** A critical limitation of this approach is that cartridges are monolithic and non-compositional: encoding an entire collection into a single KV block does not scale, and naively mixing cartridges trained in isolation collapses performance to near chance. and "However, their approach relies on a single monolithic cartridge per collection, which does not scale to real-world document collections: loading all documents into one cache bloats the prefix with irrelevant tokens, while information-dense content (e.g., financial tables, clinical notes, technical specifications) cannot be highly compressed without loss." - Inverted-negative from the CAS paper's OWN limitations section (its technique does not cover mid-conversation cache insertion at all): "In multi-turn scenarios, however, a cartridge may need to be loaded mid-conversation—for example, when a user references a new document after several turns of dialogue. Prepending the cartridge at that point would invalidate the previously computed KV entries, requiring a full recomputation of the conversation history." And the compression-tolerance bound: "Compression tolerance varies with document complexity ( 100 × 100\times for TechQA, ≤ 2 × {\leq}2\times for FinQA), reaching within 5% of uncompressed baselines. Finally, we show that cartridges can be combined with retrieval, matching RAG performance while using 3–4 × \times fewer tokens, though for information-dense tasks such as FinQA, text RAG maintains an edge." And on retrieval quality: "This is a pragmatic choice, but it means retrieval quality is bounded by how well the query matches the raw document text rather than the compressed KV representation."
- **Date:** submitted 3 Jun 2026.
- **Evidence:** READ BODY

#### C183. Region-aware eviction decays — MemDecay’s own abstract names attention-score normalisation as the main limitation
- **WHO:** MemDecay authors (arXiv 2607.10582)
- **URL:** https://arxiv.org/abs/2607.10582
- **Closure status:** published paper whose own abstract undercuts part of the technique: accumulated-attention retention performs better on unpinned content, and the main limitation is identified as attention-score normalisation.
- **STATED REASON, VERBATIM:** "Accumulated-attention retention performs better on unpinned content, however, and ablations identify attention-score normalization as the main limitation of the current formulation. These results establish semantic prompt structure as a robust signal for KV-cache management while clarifying how it should be combined with attention-based importance."
- **Date:** submitted 12 Jul 2026
- **Evidence:** READ BODY

#### C184. CacheTTL’s TTL cost model assumes the tool-call distribution is stable — the paper’s own Appendix D concedes it may produce suboptimal TTLs under distribution shift
- **WHO:** Continuum/CacheTTL authors (arXiv 2511.02230), "Appendix D Limitations and Future Work"
- **URL:** https://arxiv.org/html/2511.02230v4
- **Closure status:** published paper’s own limitations section scoping the TTL derivation — a scoped failure mode, not a retraction.
- **STATED REASON, VERBATIM:** Sensitivity of the TTL Cost Model: CacheTTL relies on a cost–benefit model that combines empirical tool-call CDFs, memory-usage estimates, and a memoryfulness" factor to derive optimal TTL values. While this design is principled, it assumes that tool-call distributions and workload characteristics are sufficiently stable for historical samples to be predictive. In highly volatile or adversarial workloads, such as agents whose tool latencies abruptly shift due to back-end contention or external API variability, the model may produce suboptimal TTLs, temporarily degrading scheduling efficiency. ... We leave handling sudden distribution shifts in agent as future work."
- **Date:** v4 HTML retrieved; v1 submitted 4 Nov 2025
- **Evidence:** READ BODY

#### C185. Star Attention — distributed attention is SLOWER than ordinary single-machine inference below 32K, by the paper’s own Appendix/Table 6
- **WHO:** NVIDIA (Acharya, Jia, Ginsburg), the paper's own Appendix C.1 / Table 6
- **URL:** https://arxiv.org/abs/2411.17116 | https://arxiv.org/html/2411.17116v1
- **Closure status:** published result that undercuts the technique’s own applicability range: for sequence lengths below 32K vanilla inference is faster, and the technique also permits accuracy loss by design (1.6–4.9% Multi-NIAH, 0.9–6.8% QA declines).
- **STATED REASON, VERBATIM:** For sequence lengths below 32K, vanilla inference is faster than the distributed attention mechanisms, primarily due to the GPU communication overhead incurred in the distributed setups. However, in long context scenarios i.e. on sequence lengths exceeding 32K tokens, Star Attention begins to demonstrate clear performance advantages. - The paper's own Table 6 numbers make the loss concrete for Llama3.1-8B-Instruct: `16K → Vanilla 7s, Ring 10s, Star 9s`; `32K → Vanilla 10s, Ring 12s, Star 10s`; `64K → Vanilla 18s, Ring 22s, Star 12s`; `128K → Vanilla OOM, Ring 53s, Star 20s`. - The technique also permits accuracy loss by design, verbatim: "The choice of block size is dependent on the user on how much accuracy can be traded for improved speed." and the reported quality gap, verbatim: "Notably, Star Attention achieves scores nearly identical to global attention in Single-NIAH tasks. However, in more complex tasks such as Multi-NIAH and QA, it shows slight decline in performance, with reductions ranging from 1.6% to 4.9% in Multi-NIAH and 0.9% to 6.8% in QA tasks."
- **Date:** arXiv 2411.17116v1, 2024-11-26
- **Evidence:** READ BODY

