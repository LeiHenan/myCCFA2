
### Title-vs-claim notes (all OK, but the descriptive labels are not always the artefact's title)

These are not URL failures, but a reader following the "Artifact" column would not find a paper by the name S13 gives it:

| S13 row | S13 gives the artifact as | Actual title on the page | Verdict |
|---|---|---|---|
| CLOSED-5 | "AgentKVShift" | `AgentKVShift: Efficient KV Cache Reuse for Agentic Memory Systems` | OK (short name) |
| CLOSED-6 | "SideQuest" | `SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning` | OK |
| CLOSED-9 | "GrowPage" | `GrowPage: On-Demand KV Budgeting for Efficient LLM Reasoning Serving` | OK |
| CLOSED-10 | "**SAECache**" | `Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches` — "SAECache" is only the in-paper system name, not the title | OK but mislabelled as the artifact name |
| CLOSED-12 | "Zipage" | `Zipage: Maintain High Request Concurrency for LLM Reasoning through Compressed PagedAttention` | OK |
| HW-1 | "**CacheScout**" | `Learning Agent Execution for KV-Cache Management in Agentic Serving` — "CacheScout" is the in-paper system name | OK but mislabelled |
| HW-5 | "Multi-turn LLM conversations under LRU …" | `Multi-Turn LLM Conversations under the Least-Recently-Used Policy: Mean-Field Asymptotics and Hit Ratio Approximation` | OK (paraphrase, not the title) |
| HW-6 | "psRL" | `psRL: Efficient Training for Agentic AI via Training-Time Prefix Sharing` | OK |
| OPEN-6 | "#38000 '[Mamba] Thin cached states by coverage instead of evicting the LRU tail'" | title matches exactly | OK |
| ABANDONED-3 | "Reasoning Language Model Inference Serving Unveiled: An Empirical Study" | title matches exactly (lead author is Qi Li; page confirms HKUST-GZ affiliation) | OK |
| ABANDONED-14 | "Brady Steele, 'On the Limits of Learned Importance Scoring for KV Cache Compression'" | title matches exactly | OK (author name not independently checked on the page) |
| ABANDONED-16 | "Momchil Hardalov, Gonzalo Iglesias, Adrià de Gispert, 'Cartridges at Scale…'" | title matches exactly | OK |
| ABANDONED-17 | "Aojie Yuan, Tianqi Shen, Dajun Zhang, 'Not All Thoughts Need HBM…'" | title matches exactly | OK |
| ABANDONED-18 | "py-kvcache authors (AtLarge group, VU Amsterdam)" | title matches exactly | OK |
| OPEN-8 | "NVIDIA / TensorRT-LLM — open PR #16252 '[#16251][feat] KV cache manager v1: add disk (L3) tier with per-request retention TTL'" | title matches exactly (author `nafis271`) | OK |
| ABANDONED-12 | "'[feat] Inclusive host KV-cache tier (shadow reuse) for KVCacheManager V2'" | actual title carries a prefix: `[TRTLLM-13617][feat] Inclusive host KV-cache tier (shadow reuse) for KVCacheManager V2` | OK (prefix dropped) |
| ABANDONED-6 | "'PR closed UNMERGED (joninco closed this May 10, 2026)'" — the row's title text is a description, not a title | actual title: `[Anthropic API] Strip billing header to fix prefix caching` | OK |

## Verbatim quote check (ABANDONED rows)

Every ABANDONED row carries a claimed verbatim closure/result quote. Each was searched in the cited page after normalisation (case, whitespace, unicode quotes/dashes, markdown code-span backticks, PDF hyphen spacing). 97 individual quoted strings were extracted from the ABANDONED section; the substantive ones are reported per row below.

| Row | Status | Notes |
|---|---|---|
| ABANDONED-1 (vLLM #42645, ACE) | **FOUND EXACTLY** | Closure quote `"Withdrawing this: I measured it against doing nothing, and it lost Closing this rather than leaving it open, because I no longer believe the design in it."` is verbatim (the source itself runs the two sentences together). Benchmark numbers quote, `#36311 … ref_cnt == 0` lesson, and "scoreboard, not a benchmark" quote all verbatim. `closedTime=2026-08-10T05:36:27Z`, `state=CLOSED` both match. |
| ABANDONED-2 (TRT-LLM #18711) | **FOUND EXACTLY** | Closure quote `"Closing this for now. I will ask @erictsai-nv to continue working on this when he is back."` verbatim. All four body bullets and the beam-search review comment verbatim (only `_reuse_token_source()` / `request.max_beam_num_tokens` lose underscores to code-span rendering). `closedTime=2026-09-08T00:51:04Z` matches. |
| ABANDONED-3 (arXiv 2510.18672) | **FOUND EXACTLY** | `"However, for 7B models, prefix caching negatively impacts efficiency, leading to increased latency."` verbatim in the full text. Section "5.3 Is Prefix Caching Useful for Contributing Efficient RLLM Serving" verbatim. Hardware list verbatim (Appendix G.2). `"vLLM version 0.8.1 and SGLang version 0.4.6.post1"` is genuine — it appears as `"We use vLLM^16 version 0.8.1 and SGLang^34 version 0.4.6.post1"` (Appendix G.3); the footnote markers are what break a naive substring search. `"Table 14: Results of RLLM-7B without Prefix Cache"` verbatim. Submitted 2025-10-21 confirmed. |
| ABANDONED-4 (arXiv 2608.23658) | **FOUND EXACTLY** | Allocation/negative-result quote, TP-dilution quote, and `"works with CUDA graphs and prefix caching"` all verbatim. Testbed quote matches modulo code-span spaces (`gpu_memory_utilization=0.9`, `max_model_len=32768`). |
| ABANDONED-5 (arXiv 2608.00902) | **FOUND EXACTLY** | All three quotes verbatim (abstract). Submitted 2026-08-02 confirmed. |
| ABANDONED-6 (SGLang #21064) | **FOUND EXACTLY** | The `x-anthropic-billing-header` explanation, the `~4,800 → ~20,800+` measurement, and the regression-test sentence are all verbatim on the PR page. Note: the S13 row heading ("strip the Anthropic billing header so Claude Code turns can share a prefix") is the agent's description; the PR's own title is `[Anthropic API] Strip billing header to fix prefix caching`. |
| ABANDONED-7 (vLLM #51098) | **FOUND EXACTLY** | The revert-rationale quote (both consequences), the HybridKVCacheCoordinator sentence, and the `AssertionError: … block_sizes=[544, 544, 544, 544, 181], hash_block_size=98464 …` text are all verbatim. `closedTime=2026-08-05T05:44:33Z` matches. |
| ABANDONED-8 (vLLM #37823, T-LRU) | **FOUND EXACTLY** | Stale-bot closure text verbatim; T-LRU abstract-style quote and the "conversation-length blind" quote verbatim; the arXiv 2510.15152 link inside the issue is real and resolves to `Tail-Optimized Caching for LLM Inference` (fetched, HTTP 200) — so the agent's "not itself an evidence source" caveat is honest and correct. `createdAt=2026-03-22`, stale `2026-06-25`, closed `2026-07-27` all match. |
| ABANDONED-9 (vLLM #29286) | **FOUND EXACTLY** | Stale-bot closure text, the H20 tokenizer cost quote, and both proposed-implementation sentences verbatim. Source itself uses an ellipsis in the same place ("… separate the system prompt from other prompts, we can use condition…"), so the elision is the author's, not the agent's. `createdAt=2025-11-24`, `stateReason=NOT_PLANNED` match. |
| ABANDONED-10 (vLLM #42185) | **FOUND APPROXIMATELY** | All substantive quotes verbatim (the FCFS/cache-affinity RFC body, the "sglang's RadixAttention … 10–30% throughput uplift" citation, the rejected-radix-tree note, "Software engineer at Meta…"). **One defect**: the "STATED REASON, VERBATIM" is given as `"This issue has been automatically closed due to inactivity…"`, but that exact string does **not** occur on #42185 (0 occurrences). What the page actually contains is the *pre-closure* stale notice: `"This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days…"` — which is exactly the string S13 uses for its "Additional retrieved detail" line in ABANDONED-10. The closure state itself (`state=CLOSED`, `stateReason=NOT_PLANNED`) is confirmed, so the claim "closed by stale bot with no human reason" stands; only the quoted closure wording is misattributed between the two vLLM stale-bot templates. |
| ABANDONED-11 (vLLM #40533) | **FOUND EXACTLY** | Stale-bot closure text verbatim; hybrid non-KV-state paragraph and the `web_search` tool-call corruption quote verbatim. `createdAt=2026-04-21`, stale `2026-07-23`, closed `2026-08-25` all match. |
| ABANDONED-12 (TRT-LLM #15828) | **FOUND EXACTLY** | The maintainer comment `"Please hold on until bug reporter's response. It's a design choice…"` is verbatim, and S13 correctly labels it as the last substantive maintainer comment rather than a closure reason. `closedTime=2026-07-31T02:13:07Z` matches. |
| ABANDONED-13 (SGLang #38720) | **FOUND APPROXIMATELY** | The superseded-and-split quote and the `2.4% host-hit / 1-in-61` quote are verbatim. **One defect**: the quote `"The router's in-process radix tree applies every BlockRemoved as a full removal... So a pod loses ownership of a prefix…"` joins two non-adjacent sentences across an elided 138-character passage (`: for every node carrying any hash in block_hashes, drop worker from that node's worker set. -- src/policies/kv_events/tree.rs, HashTree::remove`). The words on both sides are exact, but the ellipsis silently deletes a code reference and the agent's rendering changes the source's colon into a full stop. It reads as a seamless quotation while actually being a splice. |
| ABANDONED-14 (arXiv 2601.14279) | **FOUND EXACTLY** | The `"Despite architectural sophistication (multi-horizon lookahead, cross-attention), SIP does not outperform simple baselines, including random selection, across 5 seeds, 4 retention levels, and 3 tasks."` and the three-point key-findings list are verbatim (abstract). Submitted 2026-01-13 confirmed. |
| ABANDONED-15 (arXiv 2608.20397) | **FOUND EXACTLY** | Both negative results, the RoPE/off-anchor paragraph, and the Apple M4 Max testbed quote are verbatim. |
| ABANDONED-16 (arXiv 2606.04557) | **FOUND EXACTLY** | The monolithic/non-compositional quote, the budget-manager quote, and the 2–6 points residual quote are verbatim. |
| ABANDONED-17 (arXiv 2605.09490) | **FOUND EXACTLY** | The 0–2.5% collapse quote, the R-KV reproduction quote, and the 3%-eviction/GSM8K/MATH-500 quote are verbatim. |
| ABANDONED-18 (arXiv 2609.11744) | **FOUND EXACTLY** | Abstract break-even quote and the `"Their average request length is below the measured 6,203 token SSD break-even point for Qwen3 4B…"` quote verbatim. The `"External caching is not automatically beneficial…"` quote is verbatim up to a trailing `…` which is the agent's own truncation (the source continues `"…and a CPU↔GPU transfer, not to mention scheduling, synchronization, etc."`). |
| ABANDONED-19 (SGLang #20088) | **NO VERBATIM QUOTE EXISTS (as S13 itself discloses)** | S13 honestly declares this row has no retrievable closure statement. Confirmed: page shows `rexxy-sasor wants to merge 17 commits into sgl-project:main`, `state=CLOSED`, `closedTime=2026-08-10T03:55:49Z`, no closure comment. The one thing the row does quote, `"rexxy-sasori wants to merge 17 commits into sgl-project:main"`, has a typo: the login is **rexxy-sasor** (S13 also spells it `rexxy-sasori` in the WHO field). |

### ABANDONED quote tally

- Clean verbatim closures/results: **16 of 19** rows (ABANDONED-1…9, 11, 12, 14…18).
- Verbatim but with an undisclosed splice: **1 row** (ABANDONED-13).
- Quoted string not present on the cited page: **1 row** (ABANDONED-10, stale-bot template swapped).
- Row with no observable quote to check: **1 row** (ABANDONED-19, disclosed by S13 itself).
- In quote-string terms: **62 FOUND EXACTLY**, **7 FOUND EXACTLY (differ only in code-span/underscore/unicode rendering)**, **2 FOUND APPROXIMATELY (splices)**, **1 NOT FOUND on the cited page** (the ABANDONED-10 closure sentence — the page carries the other stale-bot template).

