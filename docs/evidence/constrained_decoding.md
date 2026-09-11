# Subfield screen: Structured / constrained decoding runtime

**Date of screen:** 2026-09-11
**Screener stance:** adversarial; default kill.
**Primary sources used:** local checkouts `c2prior/vllm_src` (commit `07b7553`, 2026-09-11) and `c2prior/sglang_src` (commit `ad7f57c`, 2026-09-11); arXiv HTML/ar5iv full texts; GitHub issue/PR HTML with browser UA (embedded `"state"` field); `huggingface.co/papers/<id>` mirrors.

**Evidence-labelling convention used throughout.** "CONFIRMED" = I fetched the page and saw both the title and the sentence I quote. "Unresolved" = I could not obtain the body. Per session standing rule, HTTP status codes are evidence in neither direction.

---

## T1 — Density

### Key groups and their 2026 output (all CONFIRMED unless marked)

| Group | Work | Date | Confirmed via |
|---|---|---|---|
| SJTU Zhiyuan + CMU + NVIDIA (Linzhang Li, Yixin Dong, Guanjie Wang, Ziyi Xu, Alexander Jiang, Tianqi Chen) | **XGrammar 2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs**, arXiv `2601.04426` (v3); also ACM CAIS 2026, doi `10.1145/3786335.3813124` | 2026-01 (v1) | ar5iv full text |
| Saarland (Michael Sullivan, Alexander Koller) | **Accelerating Constrained Decoding with Token Space Compression** (CFGzip), arXiv `2605.29986` | 2026-05-28 | arXiv HTML full text |
| NTU et al. (Wenhua Nie, Zijie Meng, Kun Zou, Zheng Lin, Ziwei Li, Haoran Zheng, Jyh-Shing Roger Jang, Hao Zhang) | **Future Validity is the Missing Statistic: From Impossibility to Φ-Estimation for Grammar-Faithful Speculative Decoding**, arXiv `2605.07698` | 2026-05 | ar5iv full text |
| PKU + SJTU + Tsinghua (Yongmin Li, Yihong Dong, Jia Li, Ge Li) | **Efficient Grammar-Constrained Decoding via Parser Stack Classification** (PSC), arXiv `2608.03065` | 2026-08-04 | arXiv HTML full text |
| Amazon (Xingzi Xu, Karim Bouyarmane) | **Trie Automata for Constrained Decoding over Large Finite Sets**, arXiv `2608.12574` | 2026-08-12 | arXiv HTML full text |
| UCAS Hangzhou (Linze Wu, Xinrui Chen) | **Parser States Already Know: Structure-Conditioned KV Persistence for Structured Generation** (PASK), arXiv `2608.28276` | 2026-08-28 | arXiv HTML full text |
| (authors Hantao Hua, Jiming Su, hao tang, Yiping Yao, Feng Zhu) | **Gram2Token: Enabling Run-time GPU-Native Grammar-Constrained Decoding for LLMs**, ICML 2026 poster | ICML 2026 (poster Jul 8) | icml.cc poster page |
| (Avinash Reddy, Thayne Walker, Jaime Ide, Amrit Singh Bedi) | **The Hidden Cost of Structured Generation in LLMs: Draft-Conditioned Constrained Decoding** (DCCD), ICML 2026 poster | ICML 2026 (poster Jul 8) | icml.cc poster page |

Foundational works resolved by title this session (mirror page fetched, abstract body not read): **XGrammar** `2411.15100` — "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models"; **Outlines** `2307.09702` — "Efficient Guided Generation for Large Language Models"; **JSONSchemaBench** `2501.10868` — "Generating Structured Outputs from Language Models: Benchmark and Studies" (note: several search indices render this as "A Rigorous Benchmark of…" — that longer title is **not** the arXiv title; do not cite it that way).

### Rate estimate

I did **not** run an exhaustive count, and I will not pretend to. What I can defend: **eight substantial, distinct systems/algorithm papers in the first eight months of 2026** (listed above), i.e. **≥1 per month at the level that matters**, with three arriving in a single four-week window (2026-08-04, 08-12, 08-28). Beyond these, the tail is much larger: searching the engine trackers alone returns 120 vLLM issues/PRs with "grammar" in the title, 90 with "structured output", 101 with "guided decoding", 130 in SGLang with "grammar", 62 in TensorRT-LLM with "guided". SGLang alone has a **seven-PR `[Spec]` programme** on grammar × speculative decoding (#32110, #32353, #32380, #32393, #32409 closed; #28986, #30155 open).

**Venues:** ICML 2026 (at least two posters), ACM CAIS 2026 (XGrammar 2), arXiv cs.AI/cs.SE/cs.LG. Notably absent from the confirmed set: **no MLSys/OSDI/SOSP/NSDI/ASPLOS/EuroSys paper on this topic surfaced in any of my probes.** The systems-venue home for this work appears to be ICML/CAIS + arXiv, not the venues this researcher is targeting.

**Current SOTA systems:** XGrammar 2 (the reference engine, integrated into both SGLang and vLLM; "XGrammar 2 is an open-source project and has been widely adopted by industry products and open-source inference engines"), llguidance (Microsoft), Outlines (.txt). Research-prototype SOTA now ahead of or level with these on their own metrics: CFGzip, PSC, Trie Automata, Gram2Token.

---

## T2 — The runtime's maturity, and where the frontier sits

### What actually ships (from the Sept-11-2026 checkouts)

**vLLM (`07b7553`).** `vllm/v1/structured_output/` carries four backends: `backend_xgrammar.py`, `backend_guidance.py`, `backend_outlines.py`, `backend_lm_format_enforcer.py`. The pipeline is: **fill the bitmask on CPU** (`grammar_bitmask()` in `vllm/v1/structured_output/__init__.py`), with a thread-pool fast path gated by `self.fill_bitmask_parallel_threshold = 128` and `fill_bitmask_parallel_batch_size = 16`; then **async-copy to GPU** and apply with a Triton kernel (`vllm/v1/worker/gpu/structured_outputs.py`, `StructuredOutputsWorker.apply_grammar_bitmask`, using a dedicated `copy_stream` and `PIN_MEMORY`). Speculative decoding is integrated: the bitmask tensor is allocated as `max_batch_size * (1 + max_num_spec_tokens)` rows ("one for each speculative position, and one more for the bonus token"). **vLLM ships no jump-forward decoding at all** — see the verbatim TODOs in T3/GAP A.

**SGLang (`ad7f57c`).** `python/sglang/srt/constrained/` has `xgrammar_backend.py`, `llguidance_backend.py`, `outlines_backend.py`, `reasoner_grammar_backend.py`, `grammar_manager.py`; plus Triton kernels `kernels/ops/grammar/token_filter_ops.py` and `bitmask_ops.py` (the latter adapted from XGrammar's `apply_token_bitmask_inplace_triton.py`). Spec-decode overlap shipped: PR **#31488 MERGED**, authored by core maintainer merrymercy. **Jump-forward is dead code** — see T3/GAP A.

**TensorRT-LLM.** `guided_decoding` is a first-class feature (62 tracker items with "guided" in the title); PR #15863 (closed) adds "[None][feat] Add low-latency host task dispatch mode for guided decoding". Open correctness items #18891 ("Guided decoding restarts its grammar after KV-cache recomputation") and #18892 ("EAGLE3 dynamic-tree decoding emits tokens outside the requested grammar").

### Measured overhead — the numbers that decide this screen

These four numbers, taken together, are the whole story:

1. **XGrammar 2 end-to-end:** "compared to XGrammar, XGrammar 2 has about a **7x speedup** over the end-to-end latency. Besides, the gap between the result of XGrammar 2 and the result without constraints is **no more than 6%**." Setup: Nvidia RTX 5090 + Xeon 8470Q, SGLang v0.5.3.post3. (`2601.04426`)
2. **PSC end-to-end:** "end-to-end LLM throughput with PSC **approaches that of unconstrained decoding**", after "up to **700 times** speedup in mask computation on complex programming language grammars, and up to **30 times** speedup for schema-conformant JSON". (`2608.03065`, 2026-08-04)
3. **Trie Automata per-step cost:** "**7× faster per-step valid-token computation (0.65 µs vs. 5.8 µs)** compared to XGrammar". (`2608.12574`)
4. **llguidance per-token overhead:** "llguidance has about **250 µs per-token overhead** with OpenAI Harmony Response Format and a **more than 1000 µs per-token overhead** with Llama's Tool Calling Format". (`2601.04426`) — llguidance is the laggard, and it is one backend of four in vLLM.

**Where the residual cost is, per the literature:** the *simple* case (JSON schema, tool calling) is solved — XGrammar 2 and PSC both report ≈unconstrained throughput. What is **not** solved is *complex* CFGs. CFGzip states it plainly: "The heavily-optimized XGrammar2 can quickly compute token masks for simple CFGs such as JSON schema grammars, resulting in negligible overhead. However, as we demonstrate empirically in Section 4, **more complex CFGs still present a challenge to XGrammar2: we record a ~2-10x slowdown over the base LLM with CFGs such as C++ and a modified variant of Python.**" (`2605.29986`) That is the one live gap — and CFGzip itself claims 7.5× on it.

### Frontier vs engines: verdict on direction

**The literature frontier is AHEAD of the shipped engines on every axis I probed**, and in two cases the papers are explicitly selling engine-integration wins the engines have not taken:

- **Mask computation:** literature ahead (Gram2Token, PSC, CFGzip, Trie Automata). Gram2Token's ICML abstract is a direct claim of engine-level superiority: "Across four model families under **schema-diverse continuous batching**, Gram2Token achieves a geometric-mean throughput improvement of **1.38×** over the strongest baseline, with a maximum speedup of **1.85×**."
- **Batch-serving integration path:** literature ahead, and this is *pure* engine lag. Trie Automata: "Because precomputed masks enable a stateless serving path that bypasses the guided decoding pipeline, this advantage compounds in batch serving: end-to-end vLLM throughput reaches **219 req/s vs. XGrammar's 7.5 req/s at batch size 256 (29×)**." The paper itself concedes the decomposition: "This 29× combines the algorithmic speedup with **integration-path savings that only precomputed masks make possible**." A 29× that is mostly pipeline bypass is not a research gap; it is a vLLM PR.
- **Constraint × KV/memory:** literature ahead. PASK: "**up to 2.2× higher throughput and 3.3× lower TPOT**, while using 0.53× the peak GPU memory of Full KV." (`2608.28276`)
- **Constraint × spec-decode:** level-to-ahead. Engines shipped it (vLLM #14702, SGLang #31488) *and* the literature already has the theory (`2605.07698`).

This is the **mirror image of the failure mode this researcher already burned 11 topics on**, but with the sign flipped: here the literature is ahead, and the engines are closing the distance on a visible, well-instrumented front. Either way, the researcher lands on the wrong side of the spread.

---

## T3 — Three candidate gaps, each tested on both sides

Method note: for each gap I ran (i) a literature probe and (ii) a tracker-crowding probe per the updated protocol. A gap counts **only if both sides are clear**: literature genuinely silent **AND** no crowded engine programme racing it. **All three fail. I report them as killed, with the evidence that kills them.**

### GAP A — Jump-forward / forced-continuation multi-token emission in the vLLM scheduler

**(a) Category:** scheduling / runtime. ✓
**(b) Explicit statement, verbatim:**

From the local primary source, `c2prior/vllm_src/vllm/v1/core/sched/scheduler.py:574` (commit `07b7553`, 2026-09-11; mirrors https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py):

> "At each step, the scheduler tries to assign tokens to the requests so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`. This is general enough to cover chunked prefills, prefix caching, speculative decoding, and the **"jump decoding" optimization in the future**."

And `vllm/v1/structured_output/backend_guidance.py:173-177`:

> "# TODO - Add jump decoding support in the future:
> #   self.ll_matcher.compute_ff_bytes() - this should always work
> #   self.ll_matcher.compute_ff_tokens() - this only works for
> #     "canonical" tokenizers"

**(c) Not a flag:** correct, but for a reason that kills it. Implementing it requires scheduler-level changes to token-budget accounting, preemption safety, and stop-condition enforcement. vLLM PR #37105 ("[V1] Fix jump-forward decoding correctness and add tests (follow-up to #36142)", **CLOSED**, not merged) documents both: "**Fix pending_ff_tokens cleared globally each step** — Previously, all pending ff_tokens were cleared every schedule step, even for requests not scheduled in that step (preempted or skipped due to token budget). Those requests would lose their deterministic tokens permanently, corrupting continuation state." and "**Fix missing stop checks after appending ff_tokens** — ff_tokens were appended without calling check_stop, allowing requests to bypass EOS/stop-token/max-token termination."

**(d) Verifiable on 1–2 GPUs:** yes (xgrammar backend exposes `find_jump_forward_string`, referenced at `vllm/v1/structured_output/backend_xgrammar.py:146`).

**Experiment that would be run:** implement jump-forward over `xgrammar`, benchmark JSON tool-calling and SQL grammars, batch 1–64, one RTX 5090, measure TTFT + output tok/s vs unconstrained and vs the shipped path.

**Arithmetic — and why it fails.** The saving is *not* the model forward. Jumped tokens still have to be forwarded through the model to populate KV (as a mini-prefill); only the per-step sampling + mask construction is skipped. Using the field's own measured mask cost, **5.8 µs/token for XGrammar** (`2608.12574`), against a decode step on a single consumer card: even generously, an 8B model on an RTX 5090 at batch 1 runs ~15–20 ms/step. **5.8 µs / 15 ms ≈ 0.04%.** Even at batch 64, where the mask build is ~64 × 5.8 µs ≈ 0.37 ms against a step of tens of ms, the ceiling is **~1%**. Jump-forward only pays on *long forced spans* (fixed JSON scaffolds, SQL keywords, enum values), where it converts N sequential steps into one prefill — but XGrammar 2 already reports the tool-calling/JSON path at **≤6% off unconstrained**, which is exactly where those forced spans live. **Plausible speedup: ~1–2% on real workloads.** Below any top-venue bar.

**⚑ Engine-tracker crowding check (updated protocol): CROWDED — FAILS.**
Four competing vLLM proposals, **three simultaneously open/draft**, all confirmed by fetching each page:

| # | State | Created | Title |
|---|---|---|---|
| [#15490](https://github.com/vllm-project/vllm/pull/15490) | **DRAFT** | 2025-03-25 | [V1][Experimental] Jump-forward decoding (aarnphm) |
| [#36142](https://github.com/vllm-project/vllm/pull/36142) | **OPEN** | 2026-03-05 | [Feature]: Initial Implementation Jump Decoding Guidance (FredericOdermatt) |
| [#45400](https://github.com/vllm-project/vllm/pull/45400) | **DRAFT** | 2026-06-12 | [Prototype][MRV2] Grammar spec dec (jump decoding) for structured outputs (mgoin) |
| [#37105](https://github.com/vllm-project/vllm/pull/37105) | **CLOSED** | 2026-03-15 | [V1] Fix jump-forward decoding correctness and add tests |

Two of these are **competing designs on the same decision**, stated as such in #45400's own body: "Prototype of jump-forward decoding for structured outputs, framed as a speculative method (`method="grammar"`) **instead of direct token injection like #36142**." That is the parent's RFC #54749 pattern exactly: multiple named proposals, one shipped answer.

**Worse — the field has already run this experiment and removed the feature.** SGLang PR **#4032, MERGED**, by core maintainer merrymercy, titled "Misc clean up; **Remove the support of jump forward**", body verbatim:

> "**Remove jump forward to simplify the code maintenance**"

I verified the consequence in the 2026-09-11 SGLang checkout: `try_jump_forward` / `jump_forward_str_state` / `jump_and_retokenize` still exist as methods on the grammar backends, but grepping `python/sglang/srt/managers/` and `grammar_manager.py` for any of them returns **nothing** — no scheduler call site. The only "callers" are delegating wrappers in `reasoner_grammar_backend.py`, which are themselves uncalled.

**Literature side: not silent.** The mechanism is fully documented — the LMSYS compressed-FSM blog (2024-02-05) and llguidance's `docs/fast_forward.md`, both cited *inside the engines' own code* (`outlines_jump_forward.py` header cites the blog; the vLLM TODO links `fast_forward.md`). vLLM #15490's body independently re-derives the hard part ("this will inadvertently affect the model outputs. This phenomena is often known as **Coalescence** in structured generations… one can implement a retokenization strategy such that it will ensure least amount of interference") and states the practical ceiling: "**the case for single-token bitmask are relatively rare, meaning in most cases we won't be able to utilize the jump mechanism here.**"

**⇒ GAP A: KILLED.** Literature present, tracker crowded with ≥2 competing open designs, and one flagship engine shipped then deleted the feature.

---

### GAP B — GPU-native grammar execution: eliminate the CPU↔GPU round trip and the CPU mask build

**(a) Category:** kernels / runtime. ✓
**(b) Explicit statement, verbatim.** SGLang PR [#31488](https://github.com/sgl-project/sglang/pull/31488) (MERGED, merrymercy), body:

> "When speculative decoding runs together with grammar (constrained decoding), grammar-constrained spec decode needs a **CPU round trip every verify step**: copy the draft tokens out (D2H), advance the grammar FSM over the previous step's committed tokens, build the vocab bitmask, and copy it back (H2D). **Today these run around the GPU instead of under it, and the scheduler also forces cross-batch overlap off, so every grammar decode step has large GPU bubbles**"

The PR's own diagram labels the bubbles explicitly (`####### idle #######`, `## idle ##` bracketing the target verify).

**(c) Not a flag:** ✓ — eliminating the round trip requires relocating FSM state and transition tables into GPU memory and writing new kernels, not toggling a config.
**(d) 1–2 GPUs:** ✓.

**Experiment:** move FSM state + transition tables to GPU, compute bitmasks device-side, remove D2H/H2D; benchmark `schema-diverse continuous batching` at batch 64–512 against shipped XGrammar path.

**Arithmetic.** The bitmask is `batch × ⌈V/32⌉` int32. For V = 262 144 (Gemma-3-era vocab): 8192 int32 = **32 KiB per row**. At batch 256 with 1 spec token, vLLM allocates `256 × 2 = 512` rows = **16 MiB per decode step** to move H2D. At a realistic achievable PCIe Gen4 x16 rate of ~25 GB/s, that is **≈0.64 ms/step**, plus the D2H of draft tokens. A 1-GPU 8B decode step at batch 256 is roughly 40–60 ms → **~1–1.5%**. At batch 1024 it reaches ~4–6%. **Plausible speedup: low single digits at realistic batch sizes.** Note also that the engines have already blunted this: vLLM parallelizes the CPU fill above batch 128 through a thread pool, and #31488 already overlapped the round trip rather than removing it.

**⚑ Tracker crowding: CROWDED — FAILS.** ≥2 competing open proposals on the same decision:
- vLLM [#52477](https://github.com/vllm-project/vllm/pull/52477) **OPEN** — "[Bugfix][Spec Decode][Structured Output] Drive grammar masks from GPU logit counts"
- vLLM [#55931](https://github.com/vllm-project/vllm/pull/55931) **OPEN** — "[BugFix][Core] Make the structured-output grammar poll non-blocking"
- TensorRT-LLM #15863 (closed) — "[None][feat] Add **low-latency host task dispatch mode** for guided decoding"
- SGLang #31488 **MERGED**, plus the `[Spec]` series #32380 ("Derive NGRAM grammar tree links on the host instead of reading back `retrive_next_token`"), #32393, #32409.

**Literature side: saturated.**
- **Gram2Token**, ICML 2026 — "**Gram2Token, a GPU-native framework** that preprocesses deterministic byte-level grammar execution into token-level transitions before inference… reducing run-time enforcement to category lookup, masking, and state update rather than parser-style byte traversal." 1.38× geomean, 1.85× max. Code: `github.com/Paradozile/Gram2Token`.
- **PSC** `2608.03065` — up to 700× mask computation, throughput ≈ unconstrained.
- **CFGzip** `2605.29986` — "more than **20x** reduction in overhead over the current SoTA grammar engine", "up to **7.5x** reduction in total constrained generation time".
- **Trie Automata** `2608.12574` — 7× per-step, 29× e2e in vLLM.

The idea "compute the mask on the GPU / stop doing it byte-by-byte on the CPU" is the single most-occupied square in this subfield in 2026.

**⇒ GAP B: KILLED.** Literature saturated, tracker crowded, ceiling ~1–6%.

---

### GAP C — Parser state as a first-class scheduling/memory signal (schema reuse + grammar-keyed KV persistence)

**(a) Category:** memory management / scheduling. ✓
**(b) Explicit statement, verbatim.** XGrammar 2 (`2601.04426`), on why schema-set dynamism defeats pre-caching:

> "The dynamic structures pose new challenges to the current structured generation engines like XGrammar, which relies on pre-generated caches to accelerate token mask generation. Because a new combination of tags and schema may be introduced as we change the reasoning strategy and tool calling patterns, **it is harder to pre-cache all possible structures ahead of time**."

Reinforced by Gram2Token: "Break-even and grammar-complexity analyses show that **this overhead is amortized by grammar reuse, longer outputs, and larger batches**." — i.e. the entire field's economics rest on schema reuse, but nobody *schedules for* it.

**(c) Not a flag:** ✓ — a real mechanism (cross-request mask deduplication keyed on matcher state; grammar-state-keyed KV persistence).
**(d) 1–2 GPUs:** ✓.

**Experiment:** deduplicate bitmask construction across requests at identical matcher states (content-hash the FSM state, compute once, broadcast); measure at batch 64–256 with R requests sharing one schema. Separately: attach grammar FSM state to KV blocks so a prefix-cache hit restores parser state directly.

**Arithmetic.** With mask cost ≈ **5.8 µs/request/step** (XGrammar, measured in `2608.12574`), at batch 64 the aggregate mask cost is ~0.37 ms against a step of ~15–40 ms → **~1–2.5% of step time**. Eliminating 100% of it buys ≤2.5%. To exceed a 10% end-to-end gain you need mask cost ≥ 1.5 ms/step, which requires either a very complex CFG — precisely the C++/Python regime where **CFGzip already claims 7.5×** — or a very large batch, where the cost is already parallelized. **Plausible speedup: ≤2.5% in the easy regime; the hard regime is taken.**

**⚑ Tracker crowding: CROWDED — FAILS.**
- vLLM RFC [#39848](https://github.com/vllm-project/vllm/issues/39848) **OPEN** (2026-04-15) — "[RFC]: Unifying Tool Calling via **Region-Scoped Guided Decoding, Tool-Aware Grammars**, and Related Parsers"
- SGLang [#38392](https://github.com/sgl-project/sglang/pull/38392) **OPEN** — "perf: **compile grammars and apply vocab mask only on the entry TP rank**"
- vLLM [#54090](https://github.com/vllm-project/vllm/pull/54090) **OPEN** — "Add compilation timeout for JSON schema and grammar in xgrammar" (pairs with open issue #54003)
- SGLang #22196 (closed) — "[Feature] Add env-configurable bounded grammar cache"
- TensorRT-LLM [#18891](https://github.com/NVIDIA/TensorRT-LLM/issues/18891) **OPEN** (2026-09-08) — "[Bug]: **Guided decoding restarts its grammar after KV-cache recomputation**"

**Literature side: occupied.**
- **PASK** `2608.28276` does exactly the KV half: "To our knowledge, **no prior KV policy uses token-level parser transitions from constrained decoding to control generated-KV persistence** across Transformer layers" → 2.2× throughput, 3.3× lower TPOT, 0.53× peak memory.
- XGrammar 2 ships the reuse half: hierarchical FSM hashing with "perfect cache hit" / "partial cache hit" reuse of token-mask caches across grammars.
- Trie Automata has a dedicated "**Mixed schemas**" experiment (§5.1.3), "**Prefix Sharing Analysis**" (Appendix H.7), and "**Constraint-Aware Dispatch Beyond Finite Sets**" (Appendix K).

**⇒ GAP C: KILLED.** Both halves of the mechanism were claimed in August 2026, and the tracker has ≥2 competing open proposals.

---

### Summary table (T3)

| Gap | Lit. silent? | Tracker clear? | Ceiling | Verdict |
|---|---|---|---|---|
| A. Jump-forward in vLLM scheduler | **No** (blog + llguidance docs, cited in-engine) | **No** — #15490 OPEN, #36142 OPEN, #45400 DRAFT; #4032 MERGED removal in SGLang | ~1–2% | **KILLED** |
| B. GPU-native grammar execution | **No** (Gram2Token ICML'26, PSC, CFGzip, Trie Automata) | **No** — #52477 OPEN, #55931 OPEN, #15863, #31488 MERGED | ~1–6% | **KILLED** |
| C. Parser state → scheduling/KV | **No** (PASK 2608.28276; XGrammar 2 cross-grammar cache) | **No** — #39848 OPEN, #38392 OPEN, #54090 OPEN, TML #18891 OPEN | ≤2.5% | **KILLED** |

**No gap survived both sides of the test.** I did not find a fourth candidate in the time available that was not immediately absorbed by one of the above or by the impossibility result below.

---

## T4 — The kill argument

**The single strongest reason: the subfield has already spent its speedup budget, and the field's own numbers say so.**

Constrained decoding became a research target because mask computation was expensive. In 2026 that expense is gone, by the field's own measurements, in three independent papers using three different mechanisms:

- XGrammar 2: "the gap between the result of XGrammar 2 and the result **without constraints is no more than 6%**."
- PSC: "end-to-end LLM throughput with PSC **approaches that of unconstrained decoding**."
- Trie Automata: per-step mask cost **0.65 µs** — against decode steps measured in milliseconds.

**When the ceiling is 6%, there is no room for a speedup paper at MLSys.** A reviewer's first question is "what is the headroom?", and the honest answer here is "≤6% end-to-end, and ≤2.5% for any single mechanism I can isolate." Worse, the 6% is *the total*, so every candidate mechanism must be a fraction of it. My own arithmetic for all three gaps lands at **1–6%**, and that arithmetic uses the field's own published constants — it is not pessimistic, it is the field's numbers.

**Three secondary reasons, each independently sufficient:**

1. **Every remaining mechanism is simultaneously paper-occupied and engine-raced.** This subfield has the worst of both worlds the parent's protocol now checks for. On mask computation: four 2026 papers *and* a seven-PR SGLang `[Spec]` programme. On jump-forward: five vLLM tracker items *and* a MERGED SGLang removal. There is no square where one side is clear, let alone both.

2. **The field has already run the flagship experiment and rejected it.** SGLang — the group that published the compressed-FSM jump-forward blog — merged PR #4032 with the body "Remove jump forward to simplify the code maintenance", and I verified in the Sept-11-2026 checkout that no scheduler call site remains. vLLM's jump-forward correctness follow-up (#37105) is CLOSED, not merged, with documented corruption of continuation state under preemption and token-budget pressure. When the inventors of a mechanism delete it and the main alternative engine cannot merge it past correctness review, a 6–12-month paper proposing it is not a contribution; it is a regression report.

3. **What is genuinely open is not a speedup.** The one place the literature says the problem is unsolved is distributional: `2605.07698` proves that local masking plus Leviathan rejection "samples from the locally projected distribution μ^proj rather than the grammar-conditional distribution μ^⋆", with total-variation gaps up to **0.996**, and that exact correction is **#P-hard** ("Computing Φ_t(y) exactly is #P-hard for general context-free grammars, even under a unigram base language model"). The researcher's constraints explicitly exclude impossibility results — and the fix here makes decoding *slower*, not faster. The open problem and the acceptable problem are disjoint sets.

**Subfield fit is also wrong.** The researcher's #1 strength is scheduling/memory management/runtime. This subfield's surviving hard problems are parser theory (Earley, PDA/FST classification, syntactic congruence — occupied by PKU/SJTU/Tsinghua and Saarland) and kernel-level mask mechanics (occupied by Amazon and an ICML group). Scheduling here is thin: the "scheduler" content is jump-forward token accounting, which is a bug-fix surface, not a research surface.

**A note on venue.** Across every probe, the systems venues this researcher targets — MLSys, OSDI, SOSP, NSDI, ASPLOS, EuroSys — produced **no confirmed paper in this subfield**. The work goes to ICML and ACM CAIS and arXiv. Choosing this subfield means competing on a venue axis the researcher did not select, against groups who publish there natively.

**The honest bottom line:** this is not a subfield where a gap is hiding. It is a subfield that has been *converged on* — five independent 2026 papers driving the same overhead toward zero, with two engines racing to absorb them. Entering now means arriving after the result.

---

## T5 — VERDICT

**EXHAUSTED.** The literature and the engines are racing to close the same already-nearly-closed overhead (≤6% end-to-end by the field's own measure), every candidate mechanism is both paper-occupied in 2026 and the subject of ≥2 competing open engine proposals, and the only genuinely open problem is an impossibility result — which the researcher's constraints forbid and which would not be a speedup anyway.

---

## Citation hygiene

**CONFIRMED — page fetched, title and quoted sentence both seen:**

| Work | ID / URL |
|---|---|
| XGrammar 2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs | arXiv `2601.04426` (also doi `10.1145/3786335.3813124`, ACM CAIS 2026) |
| Accelerating Constrained Decoding with Token Space Compression (CFGzip) | arXiv `2605.29986` |
| Future Validity is the Missing Statistic (Φ / impossibility) | arXiv `2605.07698` |
| Efficient Grammar-Constrained Decoding via Parser Stack Classification (PSC) | arXiv `2608.03065` |
| Trie Automata for Constrained Decoding over Large Finite Sets | arXiv `2608.12574` |
| Parser States Already Know: Structure-Conditioned KV Persistence (PASK) | arXiv `2608.28276` |
| Gram2Token: Enabling Run-time GPU-Native Grammar-Constrained Decoding for LLMs | ICML 2026 poster, `icml.cc/virtual/2026/poster/62392` |
| The Hidden Cost of Structured Generation in LLMs: Draft-Conditioned Constrained Decoding | ICML 2026 poster, `icml.cc/virtual/2026/poster/62339` |
| XGrammar: Flexible and Efficient Structured Generation Engine for LLMs | arXiv `2411.15100` — **title confirmed via HF mirror; abstract body not read** |
| Efficient Guided Generation for Large Language Models (Outlines) | arXiv `2307.09702` — **title confirmed via HF mirror; abstract body not read** |
| Generating Structured Outputs from Language Models: Benchmark and Studies (JSONSchemaBench) | arXiv `2501.10868` — **title confirmed via HF mirror; abstract body not read** |
| vLLM `#15490` DRAFT / `#36142` OPEN / `#45400` DRAFT / `#37105` CLOSED / `#52477` OPEN / `#55931` OPEN / `#54090` OPEN / `#54003` OPEN / `#54453` OPEN / `#52620` OPEN / `#39848` OPEN / `#14702` MERGED | github.com/vllm-project/vllm |
| SGLang `#31488` MERGED / `#4032` MERGED / `#38863` OPEN / `#31711` OPEN / `#38392` OPEN / `#30155` OPEN / `#28986` OPEN / `[Spec]` series `#32110 #32353 #32380 #32393 #32409` CLOSED | github.com/sgl-project/sglang |
| TensorRT-LLM `#18891` OPEN / `#18892` OPEN / `#15863` closed | github.com/NVIDIA/TensorRT-LLM |

**UNRESOLVED — do not cite without your own verification:**
- **GAD (Grammar-Aligned Decoding)** — cited as `[15]` inside `2605.07698`; I did not resolve its own ID or page.
- **Beurer-Kellner et al. 2024 ("Coalescence")** — cited inside PASK and XGrammar 2 bibliographies. My guessed ID `2403.06988` returned **404** on the HF mirror; per the standing rule that is "unresolved", **not** "nonexistent". I have no confirmed ID.
- **GRID: Grammar-Railed Decoding for Enterprise SQL Generation**, `2607.11951` — title appeared in search results only; I never fetched its own page.
- **vLLM RFC `#32142`** ("[RFC]: Support function calling using `structural_tag`") — title from a search snippet only; PR body not fetched.
- **vLLM `#47885`** — asserted in local session notes as a structured-output × spec-decode PR; I did not verify it exists. **Treat as unverified.**
- **XGrammar `2411.15100`, Outlines `2307.09702`, JSONSchemaBench `2501.10868`** — titles confirmed, but I did not read their abstracts, so I quote no numbers from them.

**Rejected attributions found during this screen (report these as traps):**
- Search engines render JSONSchemaBench as "**A Rigorous Benchmark** of Structured Outputs for Language Models". The arXiv title is "**Generating Structured Outputs from Language Models: Benchmark and Studies**" (`2501.10868`). Do not cite the longer form.
- A search result attributed `2608.03065` to "**Formatron (Sun et al.)**". The fetched paper is by **Yongmin Li, Yihong Dong, Jia Li, Ge Li** (PKU/SJTU/Tsinghua) and is titled "Efficient Grammar-Constrained Decoding via Parser Stack Classification". Formatron is a different, earlier work. Do not merge them.
- The `2608.12574` arXiv HTML carries an inline critique of its own headline: "End-to-end throughput is measured only on vLLM, and the headline **29×** (Table 3) is a **vLLM-specific figure**…". The paper's own body supports the caveat ("integration-path savings that only precomputed masks make possible"). Cite the 29× only with that qualifier attached.

**Retraction check performed:** I re-read the body of every feed and issue page from which I took a headline number, and found no author retraction in-thread. The only self-caveat found is the `2608.12574` reviewer note recorded above; it narrows the claim but does not retract it.
