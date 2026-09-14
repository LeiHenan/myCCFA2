# S-B separability test: is the "does speculation pay" surface non-separable across (batch, context)?

**File written:** 2026-09-15. **Task:** zero-GPU analysis gating GPU spend on the S-B direction (matched-control decomposition of the pay / does-not-pay surface).
**Method:** every number and quote below was read by me out of a page I fetched with `curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "<Chrome/126 UA>"`. Issue/PR bodies and comments were parsed from the embedded JSON payloads (`<script type="application/json" data-target="react-app.embeddedData">` → `payload.structured_data.articleBody`, and the sibling `"body":"…"` / `<div class="markdown-body">` payloads), never from search snippets. `api.github.com` was not used. All six arXiv IDs cited were read back at `arxiv.org/abs/<id>` (title + date confirmed, §5.7).
**Prior record read first:** `notes/REVERIFY_adaptive_k_2026-09-15.md`. Its four quoted fragments from #54749 and the "ctx 900/1900/4000 → 1.29×/1.30×/1.36× at c=256 with per-arm stdev ≤ 1.72 %" claim all reproduce against the fetched pages. Two things it did **not** contain, and which change the picture, are in §1 (S4/S5 correction block) and §1 (S9 retraction).

---

## §1 The extracted data table

Every row is a surface; every surface carries the exact quote. **"Pay" is never inferred here — §2 assigns signs; §1 only records what was measured.**

### S1 — `#54691`, break-even acceptance surface, 3 ctx × 5 batch = **15 measured cells** (the crux)

*Source:* `https://github.com/vllm-project/vllm/issues/54691`, 3rd of 7 comments, author `hongboshi1234`. The issue has exactly **7 comments** (`"userInteractionCount": 7` in the page's structured data) and I extracted **8 bodies** = issue body + 7 comments, so this surface is complete as served.

Verbatim method: *"Method: run the same config twice, once normally and once with rejection_sample_method \"synthetic\" and synthetic_acceptance_rates all zeros. Every draft is rejected, so exactly one token comes out per step, but the full draft and verify still run. The ratio of the two per-token latencies is then the acceptance you would need just to break even. No acceptance measurement needed, and it isolates cost from data."*
Verbatim config: *"vLLM 0.28.1, TP8 on B300, DCP-8, k=4, max-num-seqs 64, prefix caching on. Both halves on one node in one session, only the drafter differing. Numbers are break-even acceptance:"*
Verbatim table (reproduced identically a second time in the 7th comment):

| context | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|
| 2k   | 1.47 | 1.42 | 1.55 | 1.63 | 1.74 |
| 32k  | 1.91 | 2.09 | 2.82 | 3.77 | 4.94 |
| 131k | 3.51 | 4.53 | 6.79 | 10.20 | 14.59 |

**What these numbers are:** a *threshold*, not a throughput. **What they are not:** a pay/no-pay label. No achievable accept length is measured at any of the 15 cells. This is the single most important fact in this note.

### S2 — `#54691`, the four cells the S-B claim rests on: **inferred, not measured**

*Source:* `#54749` body, update dated 2026-09-13, author `seongyun1104`. Verbatim:

> *"For any achievable accept length in (1.91, 3.51), the resulting pays / does-not-pay region cannot be produced by a context rule and a batch rule consulted separately: (2k, B=8) pays, so neither rule fires there; (32k, B=8) does not pay, so the context rule must fire at 32k; but then (32k, B=1) would be switched off, and it pays."*

The same argument, longer, is in `#54691` 7th comment: *"For any `a` in (1.91, 3.51) — an interval that contains the 3.23 ceiling you cite — the 2k row pays at every batch size, the 131k row pays at none, and the 32k row splits partway along."*
And: *"Suppose it could: let `C` be the set of contexts the context rule switches off and `R` the set of batch sizes the batch rule switches off, with a cell switched off when either fires. `(2k, B=8)` pays, so `2k ∉ C` and `B=8 ∉ R`. `(32k, B=8)` does not pay, so one of the two must fire, and since `B=8 ∉ R` it has to be `32k ∈ C`. But then `(32k, B=1)` is switched off too, and `(32k, B=1)` pays. Contradiction. The boundary is a function of the pair, not of either coordinate."*

The only direct number offered for `a` is: *"For reference the highest long-context accept length I know of on this drafter is 3.23 at ~95k"* (`#54691`, 3rd comment). **No `a` is measured at 2k, 32k or 131k, and no `a` is measured at B=1 or B=8.**

### S3 — `#54691`, measured per-token cost split at **B=1**, 3 ctx × 2 terms = **6 measured cells**

*Source:* `#54691`, 4th comment, `hongboshi1234`. *"Splitting the per-token cost at batch 1 into target and draft:"*

| context | target step | draft step (overhead / k) |
|---|---|---|
| 2k   | 10.49 ms | 1.24 ms |
| 32k  | 10.72 ms | 2.43 ms |
| 131k | 11.51 ms | 7.21 ms |

*"The target barely moves: +9.7% across a 64x increase in context."* / *"The draft step grows 481% over the same range, and at 131k one draft step costs 63% of a full target forward even though the drafter is far smaller."*
These six numbers are internally consistent with S1 at k=4: `1 + 4·(draft/target)` = 1.473 / 1.907 / 3.506 vs the reported 1.47 / 1.91 / 3.51.

### S4 — `#54691`, one measured throughput pair, both coordinates known (B=8, 131k)

*Source:* `#54691`, 3rd comment. Verbatim: *"Concretely at 131k batch 8: 235.6 ms per token with the drafter against 23.1 ms without, about 10x."*
Also: *"Worth ruling out the obvious explanation: this is not CUDA graph fallback. Verify width is max_num_seqs x (1+k) = 320 against a largest captured graph of 512, and the receipt confirms we stayed inside the graph. So it is real step cost."*

### S5 — `#54691`, single-stream 185k + short-context, **B=1**, 4 measured throughput cells + 2 caveat cells

*Source:* `#54691` issue body, author `wannagoxuf`. *"### Measurements (RTX 4090 x4 TP4, vLLM 0.28.0, Qwen3.5-family hybrid GDN target + DFlash2 drafter, single stream)"*

| config | 185k decode | short-context decode |
|---|---|---|
| SPEC_OFF | ~71 tok/s | - |
| DFlash DT2 | 12.1 tok/s | - |
| DFlash DT4 | 16.0 tok/s | ~190 tok/s |
| DFlash DT8 | - | 218 tok/s |

*"DT scaling makes it *worse* at 185k (DT2 < DT4 < ...), which is the opposite of short context. CUDA graphs are not the cause (eager 14.7 vs graph 15.0 tok/s at 185k). Numbers are medians of repeated runs."*
**The short-context cells have no SPEC_OFF baseline ("-")**, so they carry no readable sign. **The short-context value is not stated.**

### S6 — H100 NVL / Gemma-4-31B, the only surface with a measured no-spec baseline on the ctx axis: 4 ctx × 4 arms = **16 measured throughputs, all at ONE batch value**

*Source:* `#54691`, 2nd comment, `seongyun1104` (a verbatim reproduction of the PR #48944 decomposition table). *"H100 NVL, `gemma-4-31B-it-qat-FP8` + its assistant drafter, `prefix_repetition` at concurrency 256, position-balanced 2-trial. A′ is a batch-only schedule `[[1,64,3],[65,128,1],[129,512,0]]`; C′ is the same schedule with a context axis added. Output tok/s:"*

| ctx | no_spec | static K=3 | A′ (batch-only) | C′ (batch × ctx) | C′/A′ |
|----:|----:|----:|----:|----:|:--:|
| 400 | 2711.7 | 2139.8 | 1875.6 | 1890.7 | 1.01× |
| 900 | 1987.1 | 1838.9 | 1453.5 | 1874.6 | **1.29×** |
| 1900 | 1815.2 | 1822.4 | 1416.6 | 1848.2 | **1.30×** |
| 4000 | 1535.9 | 1692.8 | 1232.8 | 1680.4 | **1.36×** |

The same A′/C′ pair is also in the `#48944` PR body (*"Primary contrast — C′ (6-cell 2D schedule) vs A′ (3-item batch-only), same DSD-mode cost:"*), with *"Per-cell stdev <2%, order-bias max 1.72% (signal 29-36× larger)."*
`#54749`'s own evidence section gives only the batch-only and (bs,ctx) columns (1875.6/1890.7 · 1453.5/1874.6 · 1416.6/1848.2 · 1232.8/1680.4).
**Concurrency is 256 at every ctx point; the batch axis is not swept at all.** The batch-only schedule's three tiers are `[1,64,3],[65,128,1],[129,512,0]`, so c=256 exercises only the third tier.

### S7 — `#48627`, short-context batch crossover: 3 concurrency points × 2 arms = **6 measured throughputs, context NOT STATED**

*Source:* `https://github.com/vllm-project/vllm/issues/48627` issue body (RFC, 2026-07-14). *"**1. Short-context crossover (the motivation for today's batch table) — reproduced:**"*

| client concurrency | K=0 (tok/s) | best K (tok/s) | gain |
|---|---|---|---|
| 30 | 1,400 | 2,500 (K=3) | 1.79× |
| 60 | 2,200 | 3,000 (K=3) | 1.36× |
| 128 | 3,413 | 3,413 | 1.00× (converged) |

Stack: *"Gemma-4-31B (FP8, hybrid sliding+global attention) with its MTP drafter on a single H100 NVL 94GB, vLLM 0.23/0.24"*. **The context length for these three cells is not given anywhere in the section — I could not read it**, so none of them can be placed on the (batch, ctx) grid.

### S8 — `#48627`, dose-response table: **explicitly withdrawn by its own author ("Do not cite them")**

*Source:* `#48627` issue body §Motivation.2. Verbatim correction block, dated 2026-08-31:

> **[Correction, 2026-08-31 — the ratios in this table are superseded. Do not cite them.]** … the cells were single-run without a per-launch cache wipe or order reversal, and footnote [1] already records the 4k cell drifting 1.28 → 1.32 → 1.36 across runs. Under stricter methodology (position-balanced 2 trials, autotune cache wiped per launch, cold-start burn, 3 warm-up runs discarded + 3 measured per cell, per-arm stdev at most 1.72 %) the *direction* reproduces but the *level* attenuates: speculation against no-speculation goes 0.79× / 0.93× / 1.00× / 1.10× at ctx 400 / 900 / 1900 / 4000, i.e. speculation loses at short context here and crosses break-even near ctx 1900.

The withdrawn table (recorded here only because it is the table the prior note cited, and because the correction's replacement numbers are computed from S6) was ctx ~460/970/1,990/4,096 → K=0 3,107/2,320/2,110/1,768 tok/s and K=3 3,640/2,897/2,915/2,397 tok/s (gains 1.17×/1.25×/1.38×/1.36×), with TPOT pairs 81.0/62.9, 109.3/77.8, 119.9/78.0, 137.8/96.9 ms, plus *"(client concurrency 256 fixed; at concurrency 192 the ctx≈2k cell gives 1.45×.)"*.
**The replacement figures (0.79/0.93/1.00/1.10) are exactly `static K=3 ÷ no_spec` from S6** — 2139.8/2711.7=0.789, 1838.9/1987.1=0.925, 1822.4/1815.2=1.004, 1692.8/1535.9=1.102 — so they are a re-expression of S6, not an independent measurement.
Also in `#48627` body, the direction's claim in its own words: *"**Consequence: optimal K is a function of `(batch, ctx)` and is not separable.**"*

### S9 — `#49986`, the retraction that bears directly on S6/S7's method: context-dependence of the DSD tax **does not reproduce**

*Source:* `https://github.com/vllm-project/vllm/issues/49986`, 11th of 15 comments (author of the S6 measurement, `seongyun1104`, 2026-08-24). Verbatim:

| ctx | concurrency | no_spec TPOT | dsd K=0 TPOT | tax |
|---:|---:|---:|---:|---:|
| 400 | 2 | 14.374 ms | 15.317 ms | **+6.56 %** |
| 38 000 | 2 | 15.945 ms | 16.975 ms | **+6.46 %** |
| 4 000 | 26 | 27.218 ms | 28.945 ms | +6.34 % |
| 16 000 | 6 | 17.713 ms | 18.546 ms | +4.70 % |
| 400 | 189 | 77.164 ms | 93.820 ms | **+21.58 %** |

> *"**Context, at fixed batch (C=2): +6.56 % → +6.46 %, i.e. −0.10 pp across a 95× context range.** Flat, and what movement there is has the wrong sign."*
> *"**Batch, at fixed context (400): +6.56 % → +21.58 %, +15.02 pp across a 90× batch range.**"*
> *"**The KV pool ties them together**: with the drafter loaded, 189 concurrent requests fit at ctx 400 but only 2 at ctx 38 000. Any sweep that lets concurrency follow context — which is what a production-shaped sweep does — folds a +15 pp batch effect into the context reading and reports a context result that is not there. The fixed-batch column is what isolates context."*
> *"@Suppressor72 measured −8.5 % (short) → −35 % (32 k) on dual RTX 5090 / MRV2, a 4.1× change in the fraction. **This stack does not reproduce that shape: 1.0×.**"*

### S10 — batch axis at short context, dual RTX 5090 / Qwen3.6-35B-A3B MoE / MTP k=2: **12 measured throughputs, ONE context (short prompt, "~50 input tokens")**

*Source:* `#49986`, 2nd comment, `Suppressor72`. *"dual RTX 5090 SM120, TP=2, Qwen3.6-35B-A3B-Uncensored-FP8 MoE, MTP k=2"*

| Config | Single-stream | 4-session | 8-session |
|---|---|---|---|
| No-spec baseline | 210 t/s | 450 t/s | 694 t/s |
| DSD V1 (PIECEWISE + H1 + H3) | 152 t/s (−28%) | 203 t/s (−55%) | 527 t/s (−24%) |
| DSD MRV2 + #49652 (H2 fixed) | 285 t/s (+36% vs no-spec!) | 195 t/s (−57%) | 308 t/s (−56%) |
| DSD MRV2 + #49652 + H3 fix (#51466) | 289 t/s (+38%) | 190 t/s (−58%) | 412 t/s (−41%) |

Same comment: *"**H3: K threshold thrashing** | ±164 t/s variance (84% of mean) at 4-session | Fixed in #51466"*.

### S11 — `#49986`, dual-model HTTP campaign: 2 models × 2 contexts × 3 arms = **12 measured throughputs**

*Source:* `#49986`, 5th comment, `Suppressor72`. *"These are small-sample results (3-5 rounds). They should be treated as point estimates, not precise decompositions."* Columns are C1 no-spec / C2 always-K=0 / C3 normal DSD.

Short prompt (~50 input tokens, 256 output, 5 rounds), 8-session:
| Model | C1 (t/s) | C2 (t/s) | C3 (t/s) | C2 vs C1 | C3 vs C2 | C3 vs C1 |
|---|---|---|---|---|---|---|
| 27B dense | 434.6 | 400.4 | 397.5 | -7.9% | -0.7% | -8.5% |
| 35B MoE | 975.7 | 891.1 | 792.9 | -8.7% | -11.0% | -18.7% |

32k input context (~32k tokens, 512 output, 3 rounds), 8-session:
| Model | C1 (t/s) | C2 (t/s) | C3 (t/s) | C2 vs C1 | C3 vs C2 | C3 vs C1 |
|---|---|---|---|---|---|---|
| 27B dense | 336.2 | 218.1 | 217.3 | -35.1% | -0.4% | -35.4% |
| 35B MoE | 747.8 | 584.0 | 567.7 | -21.9% | -2.8% | -24.1% |

*"1. **The gap is strongly context-length dependent.** The 27B dense total gap grows from -8.5% (short prompt) to -35.4% (32k context) at 8-session."* — **and then retracted on the other stack in S9.** Both are measured; they disagree. Batch is fixed at 8 sessions within each row, so this surface has one batch value per column and two contexts.

### S12 — fixed-batch K sweep: **2 cells, no throughputs read**

*Sources:* `#48627` 9th comment (`seongyun1104`) and `#49986` 3rd comment, identical wording: *"our fixed-batch context sweep (K∈{0..7}, 4k vs 32k) landed WEAK but directionally shows optimal K rising with context (K\*(4k)=3, K\*(32k)=5), but the forcing-loss margin stayed below our pre-registered 15% bar"*. Two cells: `K*(4k)=3`, `K*(32k)=5`; the underlying throughputs are in an external repo I did not fetch.

### S13 — cells I could NOT read (explicit list)

1. **`#54691` DCP-degree pair** — *"DCP-8: 134 ms inter-token latency / DCP-4: 33.1 ms"* (4th comment). Batch and context are not stated for either number → unplaceable on the grid.
2. **`#48627` short-context crossover ctx** — S7's context length (3 cells).
3. **The achievable accept length `a` at every one of the 15 S1 cells.** Not published anywhere I read.
4. **`#48944`'s decomposition comment** (`https://github.com/vllm-project/vllm/pull/48944#issuecomment-5091663057`) — the PR body calls it *"Full methodology + per-cell data in decomposition comment"*. GitHub serves this PR's conversation from a client-side proto route (`pullRequestsConversationsRoute.Timeline`); the comment is **not** in the served HTML (2 `markdown-body` divs only: PR body + a bot notice), there is no `"body":"` payload, the `.atom` endpoint returned HTML rather than a feed, and a third-party reader returned HTTP 403. **I did not read it.** Its Table 1 is reproduced verbatim in S6, which I did read, so I treat S6 as the readable form of it — but any *additional* cells in that comment are unknown to me.
5. **`#54749` has zero comments** — `"userInteractionCount": 0` in the page's structured data, and exactly 1 `"body":"` payload (the issue body) in the fetched HTML. So "full bodies and all comments" for #54749 = the body plus its two self-updates (2026-09-07, 2026-09-13), all of which I read in full (15,949 characters).
6. **Whether `#48944`/`#48627` held the *scheduled* batch constant across their ctx sweep.** The stacks are stated as `prefix_repetition` at client concurrency 256; I found no statement in any page I read about the achieved batch at ctx 4000 vs ctx 400, and S9 documents on the *same rig* that the KV pool makes concurrency and context inseparable (189 fit at ctx 400, 2 at ctx 38 000). **I could not verify that c=256 was achievable at ctx 4000.**
7. Throughputs behind S12 (external `depthchart` repo), `#49986`'s later HTTP-campaign tables beyond the two rows quoted, and `#47277` (fetched, 10 bodies, not mined — it is about MTP/CUDA-graph overhead, not a (batch, ctx) pay surface).

### arXiv IDs cited (all read back at `arxiv.org/abs/<id>`, title + date confirmed)

| ID | Title read back | Date |
|---|---|---|
| 2608.08721 | *LibraSpec: Dynamic Diffusion-Based Speculative Decoding via Marginal-Gain-Driven Optimization* | 2026/08/09 |
| 2607.12696 | *Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts* | 2026/07/14 |
| 2607.27735 | *A Sparse Glimpse of the Whole: Train-Free Self-Speculative Decoding* | 2026/07/30 |
| 2606.01019 | *Hybrid Verified Decoding: Learning to Allocate Verification in Speculative Decoding* | 2026/05/31 |
| 2408.11049 | *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding* | 2024/08/20 |
| 2406.14066 | *TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput* | 2024/06/20 |

(2607.27735's abstract contains *"extending the speculation horizon can reduce rather than improve speedup when the marginal acceptance probability falls below the…"*, which matches the sentence `#54749` attributes to SparseSpec-L.)

---

## §2 Pay/no-pay definition and the resulting sign matrix

### 2.1 Two definitions are used in the sources; they are not the same

**Definition A (threshold definition — what the S-B claim actually uses).** A cell *pays* iff the achievable expected accept length `a` at that cell exceeds the cell's break-even acceptance ratio.
Source: `#54691` 3rd comment, *"The ratio of the two per-token latencies is then the acceptance you would need just to break even."* and *"At 2k the drafter pays everywhere. At 32k it stops paying past about batch 2."* Sign rule: **pay ⟺ a > (break-even ratio)**.

**Definition B (throughput definition — the only one backed by measured pay/no-pay labels).** A cell *pays* iff the speculative arm's output tok/s (or 1/TPOT) exceeds the same-stack, same-workload no-spec (or baseline-arm) figure.
Source: `#48627` body, *"speculation against no-speculation goes 0.79× / 0.93× / 1.00× / 1.10× at ctx 400 / 900 / 1900 / 4000, i.e. speculation loses at short context here and crosses break-even near ctx 1900."* Sign rule: **pay ⟺ ratio > 1** (or the stated +% > 0).

The two definitions give **opposite** answers on S6 at ctx 900 and 1900: Definition B says static K=3 does not pay at 900 (0.93×) and only ties at 1900 (1.00×); the S-B narrative ("context is what tells you when to turn it back on") is about C′ vs A′ (1.29×/1.30×), a **schedule-vs-schedule** contrast at which *neither* arm beats no-spec at ctx 900 (C′ = 0.94× no_spec) and only C′ does at ctx 1900 (1.02×). **I flag this explicitly: "pays" in `#54749` §1 means "the (bs,ctx) schedule beats the batch-only schedule", not "speculation beats no speculation".** The `#48627` sentence quoted above uses the second meaning for the same numbers.

### 2.2 Sign matrix under Definition A

Only `a` is free. Enumerating every `a`-interval between the 15 distinct break-even values (exact rational arithmetic), the sign matrix takes 16 distinct forms. The form that makes the S-B claim's own four cells true is `a ∈ (2.82, 3.51)`:

```
              B=1    B=2    B=4    B=8    B=16      basis
ctx 2k        PAY    PAY    PAY    PAY    PAY       1.47 1.42 1.55 1.63 1.74  all < a
ctx 32k       PAY    PAY    PAY    nopay  nopay     1.91 2.09 2.82 | 3.77 4.94
ctx 131k      nopay  nopay  nopay  nopay  nopay     3.51 4.53 6.79 10.20 14.59 all > a
```
For `a ∈ (2.09, 2.82)` the 32k split moves to B=2|B=4; for `a ∈ (1.91, 2.09)` it moves to B=1|B=2. The claim's three load-bearing cells — (2k,B=8) PAY, (32k,B=8) nopay, (32k,B=1) PAY — are true for **all** `a ∈ (1.91, 3.51)`, as the source states.
**Every one of these 15 signs is an inference from an unmeasured scalar.**

### 2.3 Sign matrix under Definition B (measured)

**B-row only — S6, H100 NVL / gemma-4-31B-it-qat-FP8, c=256:**

```
arm              ctx=400        ctx=900        ctx=1900       ctx=4000
static K=3       0.79  nopay    0.93  nopay    1.00  tie      1.10  PAY
A' batch-only    0.69  nopay    0.73  nopay    0.78  nopay    0.80  nopay
C' (bs,ctx)      0.70  nopay    0.94  nopay    1.02  PAY      1.09  PAY
```
Ratios computed by me from the four S6 throughput columns; they reproduce the source's own 0.79/0.93/1.00/1.10 verbatim.

**Batch row at short ctx — S10, dual RTX 5090 / Qwen3.6-35B MoE, 1 / 4 / 8 sessions:**

```
config                          1 sess        4 sess        8 sess
DSD MRV2 + #49652               +36%  PAY     -57%  nopay   -56%  nopay
DSD MRV2 + #49652 + #51466      +38%  PAY     -58%  nopay   -41%  nopay
DSD V1 (PIECEWISE)              -28%  nopay   -55%  nopay   -24%  nopay
```

**Single measured cells with both coordinates readable:**

```
(B=1, ctx=185k)   nopay   71 tok/s no-spec vs 12.1 (DT2) / 16.0 (DT4) tok/s   S5
(B=8, ctx=131k)   nopay   235.6 ms vs 23.1 ms per token                        S4
(B=2, ctx=38k)    nopay   tax +6.46% vs no_spec (K=0 arm)                      S9
(B=2, ctx=400)    nopay   tax +6.56% vs no_spec (K=0 arm)                      S9
(B=189, ctx=400)  nopay   tax +21.58% vs no_spec (K=0 arm)                     S9
(B=26, ctx=4k)    nopay   tax +6.34%  vs no_spec (K=0 arm)                     S9
(B=6, ctx=16k)    nopay   tax +4.70%  vs no_spec (K=0 arm)                     S9
```
For the four S9 rows "pay" should be read as "the *K=0 arm* is not free", not "speculation loses": the arm drafts nothing.

### 2.4 What the grid actually contains

| | value |
|---|---|
| cells with a measured **threshold**, both coordinates readable | **15** (S1) |
| distinct (batch, ctx) coordinates with a measured **pay/no-pay label** | **7** — (1,185k), (8,131k), (2,38k), (2,400), (6,16k), (26,4k), (189,400) |
| of those 7, coordinates where the contrast is *speculation vs no speculation* rather than a narrower one | **2** — (1,185k) and (8,131k); the other five are "the K=0 arm is not free" (§2.3 caveat) or "schedule vs schedule" |
| further measured labels with only **one** coordinate readable (S10/S11: 3 concurrency values, short prompt, ctx not stated) | **6** (MRV2 arms × {1,4,8} sessions) + 6 (27B dense / 35B MoE × 2 ctx × 8 sessions, S11) |
| cells with a label on a grid where **more than one batch value and more than one ctx value** were measured on the *same* stack+workload | **0** |

---

## §3 Separability test

Data used: the 15 measured break-even thresholds (S1). The labels under test are the Definition-A signs (§2.2), since those are the surface the S-B claim is about. All arithmetic is exact rational (`fractions.Fraction`) or IEEE double for the regressions; scripts are reproducible from the numbers printed here.

### 3.1 The models tried

| # | class | form | free parameters | identifiable parameters |
|---|---|---|---|---|
| M1 | "two rules consulted separately" | `off = {B ≥ b0} ∪ {ctx ≥ c0}` (finite-sample version of the union argument) | 2 | 2 |
| M1′ | AND variant | `off = {B ≥ b0} ∩ {ctx ≥ c0}` | 2 | 2 |
| M2 | **separable score, power law** | `pay ⟺ β0 + β1·log B + β2·log ctx > 0` — i.e. `pay = f(B) + g(ctx) > 0` with `f = β1·log B`, `g = β2·log ctx` | 3 | **2** (scale-invariant: `log B + (β2/β1)·log ctx < −β0/β1`) |
| M3 | separable score, raw features | `pay ⟺ β0 + β1·B + β2·ctx > 0` (equivalent to fitting logistic regression on `[1, B, ctx]`) | 3 | 2 |
| M4 | 2-corner staircase | `off = {ctx ≥ c0} ∪ {B ≥ b1 ∧ ctx ≥ c1}` | 3 | 3 |
| M5 | **free-form separable** | `pay ⟺ f(B) + g(ctx) > 0` for arbitrary reals `f` on 5 values, `g` on 3 | 8 | **7** (shift-invariant) |
| M6 | the stated cost form, free-form | `pay ⟺ a > 1 + k·M(B)/F(ctx)` | 8 | 7 |
| M7 | the stated cost form, affine | `pay ⟺ a > 1 + k·(m0+m1·B)/(f0+f1·ctx)` | 4 | **3** (scale-invariant) |

**M5/M6 feasibility is exact, not numeric.** `f(B)+g(ctx)>0` on a 5×3 grid is feasible iff the constraint graph has no strict edge inside a strongly connected component (pay cell = strict edge `t_ctx → f_B`, nopay cell = non-strict edge `f_B → t_ctx`). Implemented with Tarjan SCC over 8 nodes and 15 edges. **M2/M3 separability is exact too** (dense angle sweep, 200 000 directions, plus all hull-direction candidates — the first version of this test used only pay−nopay difference vectors, which is an *incomplete* candidate set and produced a false negative that I corrected; the corrected result is what is reported).

### 3.2 Result: the sign pattern, by `a`-interval

`FIT` = reproduces all 15 signs; `-` = fails. Exact.

| `a` interval | M1 (2p) | M2 (2p) | M4 (3p) | M5 (6p) |
|---|---|---|---|---|
| 0 < a < 1.42 | FIT | FIT | FIT | FIT |
| 1.42 < a < 1.47 | – | – | – | FIT |
| 1.47 < a < 1.55 | FIT | FIT | FIT | FIT |
| 1.55 < a < 1.63 | FIT | FIT | FIT | FIT |
| 1.63 < a < 1.74 | FIT | FIT | FIT | FIT |
| 1.74 < a < 1.91 | – | **FIT** | FIT | FIT |
| **1.91 < a < 2.09** | – | **FIT** | FIT | FIT |
| **2.09 < a < 2.82** | – | **FIT** | FIT | FIT |
| **2.82 < a < 3.51** | – | **FIT** | FIT | FIT |
| 3.51 < a < 3.77 | – | **FIT** | – | FIT |
| 3.77 < a < 4.53 | – | **FIT** | – | FIT |
| 4.53 < a < 4.94 | – | **FIT** | – | FIT |
| 4.94 < a < 6.79 | – | **FIT** | – | FIT |
| 6.79 < a < 10.20 | – | **FIT** | – | FIT |
| 10.20 < a < 14.59 | – | **FIT** | – | FIT |
| a > 14.59 | – | **FIT** | FIT | FIT |

Fitted witnesses (logistic regression on `[1, log B, log ctx]`, hand-rolled gradient ascent, 400 000 iterations, both **15/15 accuracy**):

| sub-interval | fitted `w = [β0, β1, β2]` | normalised rule |
|---|---|---|
| `a ∈ (2.82, 3.51)` | `[115.4938, −6.4328, −10.0812]` | **`log B + 1.568·log ctx < 17.96`** — i.e. `B · ctx^1.568 < 6.3 × 10^7` |
| `a ∈ (1.91, 2.82)` | `[97.8872, −8.6367, −8.6050]` | **`log B + 0.996·log ctx < 11.334`** — i.e. `B · ctx^0.996 < 8.3 × 10^4` |

Two numbers each; I verified cell-by-cell that each reproduces all 15 signs. Under M3 (raw `[1, B, ctx]`) the same fit reaches only **8/15** for `a ∈ (1.91, 2.82)` and **7/15** for `a ∈ (2.82, 3.51)`.

**Reading.** The S-B argument is valid only for M1 (the 2-parameter "a context rule and a batch rule consulted separately, with a cell switched off when either fires"). It is **false for the general separable-score class**, and the failure of M1 is not evidence of non-separability — M1 is a *union-of-two-axis-aligned-rules* hypothesis, whereas the separable class is a half-plane in the two transformed coordinates, and the observed matrix is a monotone staircase that any such half-plane with the right exponent cuts correctly. A power law in batch and context (2 effective parameters) reproduces the entire 15-cell pattern on **every** `a` interval that matters, including all three sub-intervals of the source's own `(1.91, 3.51)`.

### 3.3 Result: the magnitude / cost-form question — where a real interaction does exist

The sign test above is weak by construction (one threshold cuts one monotone-ish matrix). The S-B claim also asserts something stronger in `#54749` §2: that a stated cost model `K* = argmax_K E_accept(K) / (F(ctx) + K·M(bs))` fails, and that *"That gap closes under recalibration — the per-context coefficient has to be roughly an order of magnitude larger relative to the per-batch one — **not under a change of functional form**."*

I tested that sentence directly, on the 15 measured thresholds.

**(a) Exact refutation of the ordering — no fitting, no free parameters.** In a separable form `be(B,ctx) = 1 + k·M(B)/F(ctx)`, the sign of `M(B=1) − M(B=2)` is a single fixed quantity. The data contain **2 exact 2×2 reversals**:

| rows | columns | difference in col 1 | difference in col 2 |
|---|---|---|---|
| B=1 vs B=2 | 2k vs 32k | 1.47 − 1.42 = **+0.05** | 1.91 − 2.09 = **−0.18** |
| B=1 vs B=2 | 2k vs 131k | 1.47 − 1.42 = **+0.05** | 3.51 − 4.53 = **−1.02** |

No separable form of **any** functional form — not `f+g`, not `f·g`, not the ratio — can reproduce the *ordering* of the 15 measured thresholds. That part of the claim is exactly right. Note the size of what carries it: the entire contradiction rests on `be(2k,B=1)=1.47 > be(2k,B=2)=1.42`, a **0.05 absolute / 3.4 % relative** difference, in a row that otherwise spans 1.42 → 1.74 (22 %) across B=1 → B=16.

**(b) Affine cost form M7, fitted by exact least squares** (smallest eigenvector of `AᵀA`, Jacobi rotations, 4×4):
`F(C) = 1 − 6.45854e−06·C`, `M(B) = 0.971263 + 0.0543824·B`.
**R² = 0.7841** (SSE = 42.83 on `be−1`), with systematic residuals: the whole 2k row is over-predicted by **+119 % to +160 %**, the 131k/B=1 cell by **+165 %**, while 32k/B=8..16 and 131k/B=16 are under-predicted by −36 % to −41 %. Pairwise ordering violations: **10/105** (Kendall τ = 0.810). Sign pattern from the fitted model: **1 to 6 of the 15 cells mislabelled**, depending on `a` (6 at a=2, 2 at a=2.5, 1 at a=3 and at a=3.3).

**(c) Best *possible* separable fit (M6, free-form M and F = 6 effective parameters), on `log(be−1)` by two-way decomposition:** **R² = 0.9692**. But the residual is not noise — it is a systematic interaction, and the form's identifying restriction is violated by a factor of 3.4:

| implied quantity | B=1 | B=2 | B=4 | B=8 | B=16 | spread |
|---|---|---|---|---|---|---|
| `F(2k)/F(32k)` | 1.936 | 2.595 | 3.309 | 4.397 | 5.324 | **2.75×** |
| `F(2k)/F(131k)` | 5.340 | 8.405 | 10.527 | 14.603 | 18.365 | **3.44×** |

A separable `M(B)/F(ctx)` requires each of those rows to be constant. They are not, and they are monotone in B. Residual factors `exp(resid)` run 1.495 / 1.166 / 0.997 / 0.813 / 0.707 across B in the 2k row.

**(d) The measured decomposition refutes the cost form outright, with no fitting at all.** S3 gives the target and draft terms directly at B=1:

| ctx | measured `F` (target step) | measured `M` (draft step) |
|---|---|---|
| 2k | 10.49 ms | 1.24 ms |
| 32k | 10.72 ms | 2.43 ms |
| 131k | 11.51 ms | 7.21 ms |

`F(ctx) + K·M(batch)` requires `M` to be a function of batch alone. At **fixed batch B=1**, the measured marginal draft cost grows **5.8×** (1.24 → 7.21 ms) across context. Separately, the B=1 row of the threshold surface implies `F(2k)/F(131k) = 5.340`, whereas the measured target step *grows* by 11.51/10.49 = **1.097×**. A 5.34× required *decrease* against a measured 1.097× *increase* is not a coefficient-calibration error; **no choice of coefficients in `F(ctx) + K·M(bs)` can produce that row.** The cost model's error is a missing `K·M(batch, ctx)` interaction term, i.e. a change of functional form — the opposite of what `#54749` §2 concludes.

### 3.4 Answer to the question as asked

* **Is the sign pattern reproducible by a separable form?** **Yes** — `pay ⟺ B·ctx^1.568 < 6.3e7`, **2 effective free parameters**, 15/15 cells, for every `a` in the operating interval and in fact for every `a > 1.47` except a 0.05-wide window at `(1.42, 1.47)`.
* **How many parameters does each candidate need?** M1 (union of two 1-D rules) 2 → **fails**; M3 (raw-feature half-plane) 2 → **fails** (7–8/15); M2 (log-feature half-plane) 2 → **fits**; M4 (2-corner staircase) 3 → fits for `a < 3.51`, fails above; M5 (free-form `f(B)+g(ctx)`) 6 → fits for **every** `a`.
* **Does the *stated cost form* reproduce the surface?** **No** — refuted exactly (2 reversals), refuted on magnitudes (implied `F`-ratio spread 3.44×), and refuted by the source's own measured decomposition (`M` depends on ctx at fixed batch, 5.8×).

---

## §4 Verdict

# `SEPARABLE FORM FITS ⇒ KILL`

Scope, stated precisely so the parent agent can act on it:

1. **The published pay/no-pay pattern — the entire non-separability argument in `#54691`/`#54749` — is reproduced exactly by a separable form with 2 effective free parameters** (`pay ⟺ B·ctx^1.568 < 6.3e7` at `a ∈ (2.82, 3.51)`; `pay ⟺ B·ctx^0.996 < 8.3e4` at `a ∈ (1.91, 2.82)`), on all 15 cells, for every achievable accept length in the operating interval `(1.91, 3.51)` and for every `a` outside the single 0.05-wide window `(1.42, 1.47)`. The direction's own proof is valid **only** against the 2-parameter class "a batch rule OR a context rule, consulted separately". That class is not the separable class the research question is about, and its failure is not evidence of non-separability.
2. **Both surfaces with a measured pay/no-pay sign have exactly one batch value** (S6: 4 ctx × 1 concurrency; S10/S11: 3 concurrency × 1 ctx). Neither can constrain separability across batch and context at all — a single row (or column) is separable by construction. The 15-cell grid that *could* constrain it carries **thresholds, not labels**, and every sign in it is an inference from an unmeasured scalar `a`.
3. **What survives is a narrower and different claim, and it is real:** the *break-even threshold surface* has an exact interaction (2 two-in-two reversals) that no separable form can reproduce, and the stated cost model `F(ctx) + K·M(batch)` is refuted by the source's own measured decomposition at fixed batch (draft cost 1.24 → 7.21 ms, 5.8×, while the target step moves 1.097×). But that is a **cost-model specification error** — "the functional form is wrong", the opposite of `#54749` §2's conclusion — not a claim that the pay/no-pay region is non-separable. If the parent wants to keep anything alive, it must be re-scoped to **"price the draft term's context dependence"**, which is a calibration/mechanism question, and it is *not* licensed by the existing data either: the only surface that shows the interaction has a self-admitted measurement-method caveat (`#54691` 5th comment: *"My measurement method is at fault. I get the cost by forcing all drafts to be rejected … If rejection does meaningful work, the numbers above are worst-case rather than representative."*) and its author has said he is running the control.
4. **Additional reason to discount the S-B magnitude claim:** the same author's later fixed-batch diagnostic on the same rig (S9) **failed to reproduce** context-dependence of the speculative tax (flat to −0.10 pp across a 95× context range, batch instead carrying +15.02 pp), and explicitly warns that *"Any sweep that lets concurrency follow context … folds a +15 pp batch effect into the context reading and reports a context result that is not there."* I could not verify that S6's c=256 was achieved at ctx 4000 (§1 S13.6).

**Recommendation to the parent: do not spend GPU time on the S-B non-separability direction as scoped.** If a measurement is wanted anyway, the cheapest decisive one is in the box below.

### Cheapest measurement that would overturn this verdict

**Measure the achievable accept length `a` — not the break-even — at the four cells of `#54749`'s own unrun 2×2, on `hongboshi1234`'s stack: (2k,B=1), (2k,B=8), (32k,B=1), (32k,B=8).**

* Why exactly these four: they are the cells the contradiction is built from, and they are the cells where every sign is currently an inference. One run of the existing config with `rejection_sample_method` back to normal, reading the accept length, gives four measured labels instead of four assumed ones.
* Cost: 4 configs, one node, no new hardware, no new code — the harness already exists.
* Falsifier for the KILL: **if `a` at (32k,B=1) comes out ≤ 1.91**, the contradiction evaporates (`(32k,B=1)` stops paying, `32k ∈ C` becomes consistent, and even M1 fits). **If `a` at (32k,B=1) > 1.91 and `a` at (32k,B=8) < 3.77**, the four labels stand — and the verdict stays KILL, because the 2-parameter separable score already reproduces the resulting matrix.
* Equivalent-value alternative if that rig is unavailable: **re-measure the two cells `be(2k,B=1)` and `be(2k,B=2)` with error bars.** The entire exact refutation of separability rests on the 0.05 difference between 1.47 and 1.42, and the source's own method caveat leaves it open. Two cells; if the difference reverses or vanishes, the strongest surviving argument for non-separability disappears with it.

---

## §5 What I could not verify

1. **I did not read `#48944`'s decomposition comment** (`#issuecomment-5091663057`), which the PR body names as the container of *"Full methodology + per-cell data"*. GitHub serves that PR's conversation through a client-side proto route; the comment is absent from the served HTML, no `"body":"` payload exists, `.atom` returned HTML, and a third-party reader returned 403. Its Table 1 appears verbatim in `#54691`'s 2nd comment, which I did read (S6). Any additional cells it contains are unknown to me.
2. **I did not open `#48944`, `#54801`, `#48627`-linked PR bodies for data beyond what is quoted**: `#48944`'s PR body I read in full (it is in the served HTML); `#54801`'s PR body I fetched (`pr54801.html`, 380 132 bytes) but it contains no pay/no-pay measurement, so I did not mine it.
3. **The `depthchart` external artifacts** referenced throughout (`EVIDENCE_LEDGER.md`, `POLICY_INTERFACE.md`, `RESULTS.md`, `ctx_uplift/RESULTS.md`, `ctx_tax_mechanism/RESULTS.md`, `tax_attribution/PREREGISTRATION.md`) were **not fetched**. Everything I report about them is a quote of a GitHub comment that cites them.
4. **Whether `#48944`/`#48627` held the achieved batch constant across the ctx sweep.** I did not find a statement about it, and S9 documents on the same rig that concurrency and context are coupled by the KV pool. This is a live confound on the one surface with a measured no-spec baseline.
5. **`a` at all 15 cells of S1**: unmeasured in every source I read. My sign matrices are therefore conditional on a stated scalar, and I say so wherever they appear.
6. **`#54691`'s proposed mechanism does not stand.** Its 5th comment retracts the DCP story and leaves the magnitude of the draft overhead open (*"So I do not have an explanation for the magnitude, only for the shape"*), and the `#54749` author states independently *"I checked their proposed *mechanism* and it does not hold."* The surface is therefore unexplained by its own author, and its calibration control (`synthetic_acceptance_rates` all 1.0) was **still running** as of the last comment I read. I did not find any later comment reporting its result.
7. **arXiv IDs**: all six verified by reading `citation_title` / `citation_date` back from `arxiv.org/abs/<id>` (§1 table). No ID or title in this note came from a search snippet.
8. **`#47277`** was fetched (10 bodies) but not mined — it concerns MTP/CUDA-graph per-step overhead, not a (batch, ctx) pay surface.
9. **No GPU was used and no number was invented.** Every figure in §1 is a transcription of a quote; every figure in §2–§4 is either that transcription or arithmetic I ran and printed, with the method named at the point of use. Where a cell could not be read I have written "I could not read it" rather than inferring it. Absence claims are phrased as "I did not find it".
