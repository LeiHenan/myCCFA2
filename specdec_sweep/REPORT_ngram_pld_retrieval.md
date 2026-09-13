# Speculative Decoding — N-GRAM / PROMPT-LOOKUP / RETRIEVAL-BASED DRAFTING
Prior-art / gap-mapping sweep. Compiled 2026-09-13. Window prioritised: 2025-06 → 2026-09.

Legend: **RB** = READ BODY (page/PDF/API content retrieved and read). **TO** = TITLE ONLY (search-result title/snippet only).

> Note: a sibling sweep wrote `specdec_sweep/REPORT.md` on a different sub-topic (draft-budget / speculation-length adaptation). This file is the n-gram/PLD/retrieval-drafting sweep and is deliberately a separate path.

---

## A. CLOSED (someone shipped or published a working answer)

| # | Question it answers | Who closed it | Artifact (paper/PR/flag) | URL | Status | READ |
|---|---|---|---|---|---|---|
| A1 | Can n-gram (prompt-lookup) drafting run inside vLLM **V1** with no draft model? | LiuXiaoxuanPKU (vLLM) | PR #12193 "[V1][Spec Decode] Ngram Spec Decode"; merge commit `80f63a3` | https://github.com/vllm-project/vllm/pull/12193 | merged | RB |
| A2 | What is the user-facing vLLM knob for n-gram / prompt-lookup drafting? | vLLM | Docs "N-Gram Speculation" (`--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'`), page dated Aug 11 2026 | https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/ | shipped | RB |
| A3 | Which vLLM spec method to pick; what are the n-gram keys/defaults? | vLLM | Docs "Speculation Methods" — `prompt_lookup_max` / `prompt_lookup_min` (both default 5); N-gram rated "Low to medium gain / Medium gain" | https://docs.vllm.ai/en/latest/features/speculative_decoding/ | shipped | RB |
| A4 | Can **suffix-tree** drafting be a first-class vLLM method? | aurickq (Snowflake) + vLLM | PR #25784 "[Spec Decode] Integrate Suffix Decoding from Arctic Inference"; config keys `suffix_decoding_max_tree_depth` etc. | https://github.com/vllm-project/vllm/pull/25784 | merged/shipped | RB |
| A5 | vLLM suffix-decoding implementation surface | vLLM | API doc `vllm.v1.spec_decode.suffix_decoding` (wraps `arctic_inference.suffix_decoding.SuffixDecodingCache`) | https://docs.vllm.ai/en/v0.15.0/api/vllm/v1/spec_decode/suffix_decoding/ | shipped | RB |
| A6 | Can n-gram drafting be put on the **GPU** and made async-scheduler compatible? | vLLM contributors | PR #29184 "NGram GPU Implementation compatible with Async Scheduler" (`method: "ngram_gpu"`, H20 benchmarks) | https://github.com/vllm-project/vllm/pull/29184 | merged | RB |
| A7 | Can speculative decoding be **auto-disabled when the batch/queue grows**? | vLLM (V0) | PR #4592 `--speculative-disable-queue-size N`; merge commit `f942efb` | https://github.com/vllm-project/vllm/pull/4592 | merged | RB |
| A8 | Does SGLang have an n-gram (model-free) speculative algorithm? | SGLang | Docs "Ngram Speculative Decoding": `--speculative-algorithm NGRAM`; PR #11010 renamed lookahead→ngram (merge `24f7cb1`) | https://docs.sglang.io/docs/advanced_features/speculative_decoding ; https://github.com/sgl-project/sglang/pull/11010 | merged/shipped | RB |
| A9 | SGLang n-gram parameters and defaults | SGLang | `--speculative-ngram-max-trie-depth` (18), `--speculative-ngram-capacity` (10,000,000), `--speculative-ngram-match-type` BFS/PROB, `--speculative-ngram-{min,max}-bfs-breadth` | https://docs.sglang.io/docs/advanced_features/speculative_decoding | shipped | RB |
| A10 | Can an **external corpus / suffix automaton (SAM)** be a draft source in SGLang? | SGLang (kpham-sgl et al.) | PRs #21425 (load corpus + build SAM), #22203 (multi-SAM HTTP API), #22294, #22471, #22538 (dynamic alloc), #22737 (per-request trie) | https://github.com/sgl-project/sglang/issues/21052 | merged | RB |
| A11 | SGLang n-gram + **DP attention** support | whycoming | PR #28616 "feat(spec): support NGRAM speculative decoding with DP attention" — removes `ValueError("Currently ngram speculative decoding does not support dp attention.")`; validated 2×H20, tp2/dp2 | https://github.com/sgl-project/sglang/pull/28616 | merged | RB |
| A12 | Does **llama.cpp** have n-gram / prompt-lookup drafting? | ggml-org | `docs/speculative.md`: five shipped variants — `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`; `--spec-default` "(enables ngram-mod)" | https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md | shipped | RB |
| A13 | Self-speculative (no draft model) decoding from token history in llama.cpp | srogmann | PR #18471 "Add self-speculative decoding (no draft model required)" — `--spec-self`; gpt-oss-120b 181.24 → 445.75 tok/s, "draft acceptance rate = 0.76827" | https://github.com/ggml-org/llama.cpp/pull/18471 | merged | RB |
| A14 | Suffix-tree lookup over a *tree* of sequences in llama.cpp `lookup` example | JohannesGaessler | PR #8648 "lookup: Use tree of sequences instead of single sequence" | https://github.com/ggml-org/llama.cpp/pull/8648 | merged | RB |
| A15 | Can N-gram spec decoding be **auto-enabled** by the serving stack? | NVIDIA TensorRT-LLM | Blog "N-Gram Speculative Decoding in TensorRT LLM" → `spec_decode_algo=AUTO`; page generated by commit `4e69c14`, last updated Apr 08 2026 | https://github.com/NVIDIA/TensorRT-LLM/blob/a4349b2cd9a521af85335a517722f90fc14d4f15/blogs/tech_blog/blog07_NGram_performance_Analysis_And_Auto_Enablement.html | shipped | RB |
| A16 | Original retrieval-based (non-n-gram) draft-token source | He, Zhong, Cai, Lee, He | REST, arXiv 2311.08252, NAACL 2024 | https://arxiv.org/abs/2311.08252 | published | RB |
| A17 | Suffix-tree drafting at agentic scale | Oliaro, Jia, Campos, Qiao (Snowflake/CMU) | SuffixDecoding, arXiv 2411.04975v3 (07 Oct 2025); NeurIPS 2025 **Spotlight** | https://arxiv.org/abs/2411.04975 | published | RB |
| A18 | Suffix automaton as the exact-longest-suffix matcher | Hu et al. | SAM Decoding, ACL 2025 Long (pp. 12187–12204) | https://aclanthology.org/2025.acl-long.595/ | published | RB |
| A19 | Recycling decoding-time candidate tokens instead of using a corpus | Luo et al. | Token Recycling, ACL 2025 Long (pp. 6816–6831), Outstanding Paper | https://aclanthology.org/2025.acl-long.338/ | published | RB |
| A20 | Dense (semantic) retrieval instead of exact-match retrieval for drafting | Gritta, Xue, Lampouras | DReSD, Findings of ACL 2025 (pp. 19822–19832) | https://aclanthology.org/2025.findings-acl.1017/ | published | RB |
| A21 | Unifying retrieved exact patterns + verification logits into one draft tree | Zhang et al. | RACER, arXiv 2604.14885, Findings of ACL 2026 | https://arxiv.org/abs/2604.14885 | published | RB |
| A22 | RACER as a shipping-engine algorithm | SGLang | PR #39211 adds RACER as a training-free speculative algorithm reusing TARGET_VERIFY logits | https://github.com/sgl-project/sglang/pull/39211 | open PR (code exists) | RB |
| A23 | Optimal **tree shape** when n-gram/PLD matches and logit drafts have different acceptance | Jin, Nguyen, Inoue | Goose, arXiv 2604.02047v2 (10 Aug 2026) | https://arxiv.org/abs/2604.02047 | published | RB |
| A24 | Model-free parallel decoding with no draft model and no datastore | Fu, Bailis, Stoica, Zhang | Lookahead Decoding, arXiv 2402.02057 | https://arxiv.org/abs/2402.02057 | published | RB |
| A25 | The canonical PLD artifact (repo) | Saxena (apoorvumang) | `prompt-lookup-decoding` repo (`prompt_lookup_num_tokens` in HF `generate()`) | https://github.com/apoorvumang/prompt-lookup-decoding | shipped | RB (repo) |
| A26 | A pure-Rust longest-suffix n-gram proposer beyond llama.cpp's 4-token window | danielwinterw / Mesh-LLM | PR #1037 "suffix N-gram draft proposer (prompt-lookup decoding)"; merge commit `88f8b95` (Jul 22 2026) | https://github.com/Mesh-LLM/mesh-llm/pull/1037 | merged | RB |
| A27 | SGLang suffix decoding (Arctic Inference) user surface | adityakamat24 | PR #13553 "[Feature] Add suffix decoding speculative algorithm" (`--speculative-algorithm SUFFIX`) | https://github.com/sgl-project/sglang/pull/13553 | closed PR; doc present in branch | RB |
| A28 | Adaptive (semantic + lexical) model-free PLD drafting | Runheng Liu et al. | AdaPLD, arXiv 2606.05742v2 (14 Jun 2026) | https://arxiv.org/abs/2606.05742 | published (preprint) | RB |
| A29 | LRU n-gram cache tables as the *only* draft source | Zhiyao Ma et al. | Cacheback Decoding, arXiv 2511.21699; EMNLP 2025 main, pp. 31067–31072 | https://arxiv.org/abs/2511.21699 | published (EMNLP 2025) | RB |
| A30 | Stochastic adaptive n-gram drafting for **reasoning** trajectories | STAND, arXiv 2506.04708 | https://arxiv.org/abs/2506.04708 | published (v1 Jun 2025; latest rev. 21 May 2026) | RB |
| A31 | Learning-free **batched** n-gram speculation from weights + context | Stewart et al. (SIERRA) | The N-Grammys, arXiv 2411.03786 (ENLSP-IV @ NeurIPS 2024) | https://arxiv.org/abs/2411.03786 | published | RB |
| A32 | First benchmark of SD for LLM **test-time scaling** incl. n-gram methods | arXiv 2509.04474 (Aug 2025) | https://arxiv.org/abs/2509.04474 | published | RB |
| A33 | Semantic-embedding retrieval + soft-gated verification for RSD | Chen et al. | SENSE, arXiv 2606.00021 (14 Apr 2026) | https://arxiv.org/abs/2606.00021 | published (preprint) | RB |
| A34 | Adaptive triggering + relaxed verification for retrieval-enhanced SD | Fang et al. | ReSpec, arXiv 2511.01282 (3 Nov 2025) | https://arxiv.org/abs/2511.01282 | published (preprint) | RB |
| A35 | Using the CoT itself as the draft, plus a suffix cache, for reasoning models | Valluri et al. | SSR: Self-Speculation for Reasoning Models, arXiv 2608.20359 (17 Jun 2026) | https://arxiv.org/abs/2608.20359 | published (preprint) | RB |
| A36 | Survey taxonomy placing "simple n-gram prediction" at one end of the draft spectrum | Hu et al. | "Speculative Decoding and Beyond: An In-Depth Survey of Techniques", arXiv 2502.19732v4 (8 Oct 2025) | https://arxiv.org/abs/2502.19732 | published | RB |
| A37 | n-gram drafting across 11 languages incl. low-resource | Paudel, Ginn et al. | "Speculative Decoding and the Curse of Multilinguality", arXiv 2605.30580 (4 Aug 2026) | https://arxiv.org/abs/2605.30580 | published (preprint) | RB |

---

## B. OPEN (explicitly unsolved, with evidence someone is still asking)

**B1. External corpus / long-input scale for the SGLang n-gram trie is unfinished.**
WHO: kpham-sgl, SGLang maintainers (labels `roadmap`, `speculative-decoding`).
URL: https://github.com/sgl-project/sglang/issues/21052 (opened Mar 20, 2026; still **Open**)
What is missing — verbatim from the issue's own "Limitations" section:
> "No external corpus support. The trie is only populated from the current decoding session's output tokens. Ngram speculative decoding works best with a large reference corpus, but there is currently no mechanism to load one."
> "Insert path does not scale to long inputs. It builds trie paths for almost every suffix, leading to near-O(n²) memory growth; when capacity is full, eviction can only remove leaf nodes and may fail."
Still-open work items in the same issue (verbatim): "**Expand SAM match length (currently still cap by max_trie_depth)**", "SAM for user request's input", "User request take in corpus_ids to match", "Transport SAM across scheduler in multi-scheduler system (disagg, DP, etc)".

**B2. Expanding the SAM match past `max_trie_depth` is still an open PR.**
WHO: the PR author, who leaves the default off for lack of data.
URL: https://github.com/sgl-project/sglang/pull/36978 (Open)
Verbatim: "Add `--speculative-ngram-max-sam-match-depth`. Default it to `--speculative-ngram-max-trie-depth` for backward compatibility."
Sibling PR #37071 was **closed**, and its body says: "This is a constructed worst case, not a claim about any real workload — I do not have numbers on a real corpus yet, which is also why the default is off." (https://github.com/sgl-project/sglang/pull/37071)

**B3. Cross-request / global n-gram cache in vLLM is proposed but unmerged.**
WHO: hzeng2000 (vLLM PR author); reviewers still "Awaiting requested review".
URL: https://github.com/vllm-project/vllm/pull/44597 (Open; labelled `needs-rebase`; created 2026-06-05, updated 2026-08-23; API confirms `merged: false`)
Verbatim (PR body): "The existing ngram proposer only scans the current request context. With `prompt_lookup_cache_scope=\"global\"`, the proposer records prompt ngram continuations in a bounded process-local cache and can reuse them across requests. The default remains `local`."
GitHub's banner on the PR: "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @hzeng2000 ."

**B4. Dynamic external n-gram corpora (SGLang) is an open PR.**
URL: https://github.com/sgl-project/sglang/pull/35102 — "[Spec][Ngram] feat: support dynamic external ngram corpora" (Open)
Related open PR: https://github.com/sgl-project/sglang/pull/37879 — "[Spec][Ngram]: Add global Trie/SAM proposal allocation"

**B5. Suffix decoding in llama.cpp — online tree only; no global cross-request corpus tree.**
WHO: PR #26283 author.
URL: https://github.com/ggml-org/llama.cpp/pull/26283 (Open, Jul 29 2026)
Verbatim: "Currently, we build only the online tree only and no global corpus tree. vLLM optionally keeps the global cross-request cache (up to 10k past requests). We implement the per-request/prompt tree only."

**B6. Combining n-gram with a model-based drafter in one vLLM serving session is unsupported.**
WHO: AbdulrahmanHashem.
URL: https://github.com/vllm-project/vllm/issues/46977 (Open, Jun 29 2026)
Verbatim: "Currently `--speculative-config` accepts a single method. It would be very useful to allow combining multiple speculative decoding methods in a single serving session — specifically, pairing a primary spec method (mtp, draft_model, EAGLE3, DFlash, etc.) with ngram as a secondary layer."
> "ngram is effectively free (zero inference cost, no VRAM overhead) and excels precisely in those repetitive ranges, but contributes nothing to base generation speed on its own."

**B7. Pipelining two drafters (MTP → ngram-mod) in llama.cpp was requested and closed as not planned.**
WHO: ElSnacko (requester).
URL: https://github.com/ggml-org/llama.cpp/issues/23184 (Closed **as not planned**; label `stale`)
Verbatim: "When combining `--spec-type draft-mtp,ngram-mod`, the two speculative decoding strategies run independently and generate separate draft token streams. They do not share context, ngram-mod does not see the tokens predicted by the MTP heads, so it cannot extend them. This wastes verification time and provides no benefit over draft-mtp alone."
> "Adding ngram-mod independently on top provides no speedup, only verification overhead."
No maintainer comment is visible on the page; it carries only the `stale` label.

**B8. n-gram GPU speculator in vLLM ModelRunner V2 is still an open PR.**
URL: https://github.com/vllm-project/vllm/pull/40704 (Open; labels include `ready`, `mrv2`; created 2026-04-23, updated 2026-09-12)
Verbatim: "Added a new NGram GPU speculator. The main feature is a new implementation at: `vllm/v1/worker/gpu/spec_decode/ngram/speculator.py`".

**B9. SGLang NGRAM validation gaps are open bugs with reproductions.**
- https://github.com/sgl-project/sglang/issues/36495 (Open) — "With both values set to `4`, the first ordinary generation request aborts the scheduler in the C++ NGRAM trie: `munmap_chunk(): invalid pointer`". Verbatim: "Neither the official parameter documentation nor the CLI help states that capacity must be greater than maximum trie depth, and the server accepts the combination at startup." Fix PR: https://github.com/sgl-project/sglang/pull/36506
- https://github.com/sgl-project/sglang/issues/36352 (Open) — NGRAM + `--attention-backend torch_native` accepted then crashes at warmup. Verbatim: "The bug being reported is therefore primarily the missing argument validation: an unsupported combination is accepted and fails later with an opaque scheduler traceback instead of a clear startup error."
- https://github.com/sgl-project/sglang/issues/38129 (Open) — "Reusing an NGRAM request ID can return a stale draft after insertion". Fix PR: https://github.com/sgl-project/sglang/pull/38130
- https://github.com/sgl-project/sglang/issues/36500 (Open) — removing a corpus during async loading returns success without cancelling the load.

**B10. n-gram + structured output / tool calls in vLLM is an open, reproducible corruption class.**
WHO: Sandermage.
URL: https://github.com/vllm-project/vllm/issues/40875 (Open)
Verbatim: "tool-call output is corrupted in ~50% of requests even on a stack with all known related fixes applied… Config-only workaround (no code changes): set `prompt_lookup_min=8`. Achieves 100% clean tool-call rate (n=30 single-query, 96% n=25 multi-query) on the same hardware/model where default `prompt_lookup_min=2` gave ~50%."
Other open n-gram correctness bugs: #52620 (ngram_gpu + xgrammar HTTP 500 under concurrency), #56077 (ngram corrupts qwen3_coder tool-call parser), #39273 (ngram corrupts output on hybrid GDN/Qwen3.5), #42533 (ngram_gpu proposes past `max_model_len` budget), #49918 (prefill len 1 + `num_speculative_tokens` misclassified).
https://github.com/vllm-project/vllm/issues/52620 · https://github.com/vllm-project/vllm/issues/56077 · https://github.com/vllm-project/vllm/issues/39273 · https://github.com/vllm-project/vllm/issues/42533 · https://github.com/vllm-project/vllm/issues/49918

**B11. Batched-serving speculation control is named as open future work by the suffix-n-gram authors themselves.**
WHO: Oliaro, Jia, Campos, Qiao (SuffixDecoding).
URL: https://arxiv.org/html/2411.04975v3 (Appendix C "Batch-level Speculation Control")
Verbatim: "For batched serving scenarios, optimizing the speculation per request in a batch is crucial for many practical online deployments. While SuffixDecoding focuses on what tokens to speculate, it is also compatible with existing works that explore how much to speculate per request… These are interesting and important directions for future work."

**B12. Batched serving + sampling-based verification remain future work for the retrieval+logit tree method.**
WHO: Jin, Nguyen, Inoue (Goose).
URL: https://arxiv.org/html/2604.02047v2 (Limitations)
Verbatim: "The adjacency table stays small (< 7 MB), and our analysis assumes greedy decoding; production-grade batched serving and sampling-based verification (Leviathan et al., 2023) remain future work."
> "A batch-size sweep (Section D.7) shows Goose leading at small batch and batched AR overtaking as batch size grows (crossover ≈ 4–16), the general speculation-vs-batching trade-off."

**B13. Unsupported backend/feature combinations for SGLang NGRAM are stated as hard limits, not roadmap items.**
URL: https://docs.sglang.io/docs/advanced_features/speculative_decoding
Verbatim table entry: "NGRAM | Ngram cache from previous tokens | No | `--speculative-algorithm NGRAM` CUDA-only; no `--enable-dp-attention`; disables overlap scheduler & mixed chunked prefill"
> "No extra model available : Use NGRAM ( `--speculative-algorithm NGRAM` , CUDA-only)."
> "Ngram speculative decoding only supports CUDA ."

**B14. vLLM's own docs put n-gram and suffix drafting at the bottom of the method matrix.**
URL: https://docs.vllm.ai/en/latest/features/speculative_decoding/
Verbatim: "vLLM supports a variety of methods of speculative decoding. Model-based methods such as EAGLE, MTP, draft models, PARD and MLP provide the best latency reduction, while simpler methods such as n-gram and suffix decoding provide modest speedups without increasing workload during peak traffic."

**B15. n-gram drafting for reasoning / test-time scaling is explicitly framed as an open integration problem.**
WHO: authors of the test-time-scaling SD benchmark.
URL: https://arxiv.org/abs/2509.04474
Verbatim (abstract, final sentence): "**We hope this benchmark spurs further research on speculative decoding for test-time scaling**, enabling faster and more practical reasoning in LLMs through better handling of repetitive and diverse reasoning paths."

**B16. Multilingual drafting remains open: distillation is reported not to generalize.**
URL: https://arxiv.org/abs/2605.30580
Verbatim (abstract): "We find, though, that **distillation generalizes poorly across tasks in the same language, and we argue that assembling a task-agnostic, fully representative dataset is infeasible for low-resource languages.**"

---

## C. ATTEMPTED-AND-ABANDONED (highest value; verbatim quotes)

**C1. llama.cpp `lookup` — the author of the n-gram lookup-tree PR concluded further investment is not worthwhile.**
URL: https://github.com/ggml-org/llama.cpp/pull/8648 (PR merged, but see the author's conclusion)
Verbatim (JohannesGaessler, PR body):
> "Since the current trend for models is going towards larger vocabulary sizes I think that it is **not worthwhile to invest more work into n-gram-based lookup decoding** unless the latency increase from an increase in batch size were to become extremely small."
Supporting data in the same body (verbatim):
> "Due to its much larger vocabulary size the number of input tokens needed for good n-gram drafts is much higher for Gemma 2 than it is for Mistral. With Mistral you get a speedup even with a cold dynamic lookup cache but **with Gemma 2 you actually get a performance regression** because as of right now CUDA graphs are only supported for a batch size of 1…"
> "it takes ~50 previous runs on an RTX 4090 to sufficiently populate the dynamic lookup cache in order to break even. After ~100 previous runs the speedup is ~10%."

**C2. Ornith 1.5 35B-A3B — n-gram speculation explicitly closed as a negative.**
URL: https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/ornith-15-b70/notes/2026-08-23-ornith35b-ngram-speculation-negative.md (commit `de1bd81`)
Verbatim:
> "Status: **CLOSED NEGATIVE — keep the public recipe target-only**"
> "Both complete runs passed all freshness/finality gates. The n-gram server reported only 22 accepted tokens from 336 generated draft tokens across the seven requests where it logged a proposal (6.55% reported acceptance). The 48-token verification blocks therefore cost far more than they saved."
> "A bounded shorter profile (`N=4`, `M=8`) was also attempted. It reported the first response generation at 79.86 tok/s, then failed to finalize the HTTP stream/task or begin request two while remaining GPU-active. The process was terminated after 3m44s. That arm is a server-hang negative, not performance evidence."
> "Do not enable generic n-gram speculation in the Ornith guide. A future assisted lane needs either a genuinely fast compatible draft model or a workload-specific static corpus with measured acceptance."
Machine-readable run data (median 113.000242 → 96.423720 tok/s, **−14.67 %**): https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/ornith-15-b70/data/2026-08-23-ornith35b-ngram-speculation-summary.json
Parent commit wording: "**No-model n-gram speculation — CLOSED NEGATIVE:** default `ngram-simple` accepted only 22/336 reported draft tokens and reduced the fresh-suite median from `113.000` to `96.424 tok/s` (-14.67%)." — https://github.com/steveseguin/b70-optimization-lab/commit/de1bd812fd460a44bcbcc31c48f82cc7cf28bf32

**C3. llama.cpp ngram-mod gets stuck in a verification loop; the mitigation PR was closed unmerged and the root cause is still unknown.**
URL: https://github.com/ggml-org/llama.cpp/pull/25819 — "server : add stuck-loop escape for ngram-mod (WIP)" (state **Closed**, unmerged)
Verbatim (author):
> "❗ This is WiP and a mitigation PR, not an actual fix."
> "When ngram-mod speculative decoding fails verification, spec_draft is set to the accepted tokens (including the correction token) and the checkpoint is restored. On the next iteration the draft is reused instead of regenerated and sometimes fails again, creating a loop with non-deterministic end condition."
> "This PRs adds logs to diagnose the problem, and a 'band-aid' mitigation - it detects loop, and breaks out of it. **However, the root cause of the loop is not yet clear.**"
Runtime log quoted in the PR: "STUCK speculative loop: 4 consecutive checkpoint restores with no progress (matched 20/21 draft tokens), diverging at draft index 20: draft 63 ('`', logit 24.2188) vs sampled 7561 ('`,', logit 24.2500). Applying mitigation to force progress".

**C4. SGLang "ngram fallback to regular decode after threshold batch size" — closed unmerged.**
URL: https://github.com/sgl-project/sglang/pull/13376 (opened Nov 16, 2025 by yubofredwang; **closed Mar 5, 2026**; 28 commits; never merged)
Verbatim (author's statement of the problem):
> "Speculative Decode often works well for relatively smaller batch sizes. When the batch size is large, we want to run regular decode."
> "From the benchmark, at large batch size, we can reduce 50% of the latency."
On the EAGLE-3 arm of the same idea (verbatim): "because of the required draft extend step after each decode, the latency increases by a lot even after merging them into a single CUDA graph." No maintainer rejection comment is visible on the page.

**C5. vLLM "Add support for sampling penalties to v1 ngram speculative decoding" — auto-stale, then closed by its own author, unmerged.**
URL: https://github.com/vllm-project/vllm/pull/18441 (GitHub API: `state=closed`, `merged=false`, `closed_at 2025-11-21T04:28:23Z`)
Verbatim (github-actions bot, 2025-11-20):
> "This pull request has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this pull request should remain open. Thank you!"
Author (pooyadavoodi) then "closed this" on Nov 21, 2025. Labels at close: `speculative-decoding`, `needs-rebase`, `stale`, `v1`.

**C6. vLLM n-gram greedy output is not baseline-identical — closed as not planned / stale.**
URL: https://github.com/vllm-project/vllm/issues/41758 — "Closed as **not planned**"; labels `bug`, `stale`
Verbatim (reporter SAKETH11111):
> "I found a deterministic output difference between a no-speculative vLLM server and the same image/config with ngram speculative decoding enabled."
> "repeated serving prompt: 20/20 baseline-vs-ngram comparisons mismatched"
> "code-like control prompt with the same servers/params: 20/20 comparisons matched"

**C7. vLLM n-gram does not work under V1 in 0.8.3/0.8.4 — closed as not planned / stale.**
URL: https://github.com/vllm-project/vllm/issues/16883 — "Closed as **not planned**", label `stale`

**C8. vLLM "Evaluate multiple ngram speculations" (RAG-motivated) — closed as not planned / stale.**
URL: https://github.com/vllm-project/vllm/issues/6785 — "Closed as **not planned**"; labels `feature request`, `stale`
Verbatim (requester chenglu66):
> "During the ngram-spec-decode stage, I've always had a question: In RAG, there isn't just one document relevant to the answer; why don't we first let the large model generate 3 tokens, and then take all possible results in the N-gram?"

**C9. vLLM n-gram + Qwen3 tool calls truncated — closed as not planned / stale.**
URL: https://github.com/vllm-project/vllm/issues/21307 — "Closed as **not planned**"; labels `bug`, `stale`
Verbatim: "tool calls are often truncated, we managed to patch this using very hacky method of adding placeholder field into the tool call and partially parse json result".

**C10. vLLM n-gram was Pareto-worse than no speculation in a user's own benchmark; issue closed with no answer.**
URL: https://github.com/vllm-project/vllm/issues/16258 (Closed; vLLM 0.7.3, 2×L4, V0 engine)
Verbatim (reporter dtransposed, Apr 8 2025):
> "I have seen that regardless of the configuration, the inference of the sample model n-gram model is **Pareto worse than the inference without the n-gram model**, pretty much regardless of the contents of the speculative_config ."
> "Our n-gram model inference has: has an acceptance rate of 70% (good) but the efficiency 40% (quite low judging from the code) much lower prompt processing throughput vs. vanilla. 4x lower generation throughput."
> "This isolated script shows that for speculative decoding, the generation toks/s is roughly 3x slower than the vanilla use case"
> "However, I could not make my experiments fly, despite significant effort."

**C11. vLLM CI: n-gram speculative decoding changed output on 46/100 prompts — below the project's own 66 % correctness bar.**
URL: https://github.com/vllm-project/vllm/issues/35168 (Closed; CI-triage-filed, opened Feb 24 2026)
Verbatim from the captured assertion:
> "The `test_ngram_and_suffix_correctness` test with ngram method fails because the speculative decoding output matches the reference output only 54 out of 100 times (54%), which is below the required 66% accuracy threshold."
> "`AssertionError: assert 54 >= 66`"

**C12. HF `transformers`: prompt-lookup decoding produced different generations than standard decoding under greedy/float32 — closed without a stated fix.**
URL: https://github.com/huggingface/transformers/issues/30448 (Closed; label `Generation`)
Verbatim (reporter shwetha0312):
> "PLD is generating inconsistent outputs compared to standard decoding with the above settings. I tried with a different model as well (llama2), the results are still inconsistent."
> "However this is not the case when I use an older version of transformers (for the same settings)… transformers version: 4.37.1"

**C13. SuffixDecoding's own evaluation: model-free drafting loses on non-repetitive workloads.**
URL: https://arxiv.org/html/2411.04975v3 (§4)
Verbatim:
> "On non-agentic workloads such as Spec-Bench (which includes open-ended single-turn tasks and 8 MT-Bench categories), **SuffixDecoding alone is outperformed by EAGLE-2/3 and Token Recycling, as expected for less repetitive scenarios.**"
> "We include Spec-Bench to stress-test SuffixDecoding's limitations and evaluate the hybrid fallback mechanism."
On PLD specifically (verbatim): "existing model-free approaches, such as prompt-lookup decoding (PLD) (Saxena, 2023), achieve low overhead and rapid token generation, but typically lack adaptivity. These methods speculate a fixed number of tokens irrespective of acceptance likelihood, leading to wasted computational resources on verifying long and improbable draft sequences."

**C14. History-only NGRAM collapses on open-ended chat while a logit-augmented drafter does not.**
URL: https://github.com/sgl-project/sglang/pull/39211 (RACER in SGLang; open PR, author-reported numbers)
Verbatim: "This is not a chat-serving benchmark, but it highlights a useful difference between the draft sources: **history-only NGRAM becomes much less effective when reusable context patterns are sparse, while RACER can still obtain candidates from target logits.**"
Raw numbers on ShareGPT (`K=32`, trie depth 10): Qwen2.5-7B/3090 — Vanilla 49.24, NGRAM 71.38 (1.45×), RACER 95.00 (1.93×); Qwen3.8-27B/A800 — Vanilla 27.70, **NGRAM 28.15 (1.02×)**, RACER 41.03 (1.48×).
Repo announcing the integration: https://github.com/hkr04/RACER

**C15. Sparse retrieval (REST) is documented as the weaker paradigm inside its own subfield.**
URL: https://aclanthology.org/2025.findings-acl.1017/ (DReSD, Findings of ACL 2025)
Verbatim (abstract):
> "Sparse retrieval (He et al., 2023, REST), which operates on the surface form of strings, is currently the dominant paradigm due to its simplicity and scalability. However, **its effectiveness is limited due to the usage of short contexts and exact string matching.**"
> "DReSD achieves (on average) 87% higher acceptance rates, 65% longer accepted tokens and 19% faster generation speeds compared to sparse retrieval (REST)."

**C16. Retrieval-based drafting "fails completely" when no exact match exists — stated as a family-level limitation.**
URL: https://en.papernotes.org/ACL2026/llm_efficiency/racer_retrieval-augmented_contextual_rapid_speculative_decoding/ (paper note for RACER)
Verbatim: "**Limitations of Prior Work**: Existing training-free methods suffer from two types of issues: (1) Retrieval-based methods (e.g., PLD, REST) rely on exact token matching and **fail completely when matching continuations do not exist in the context**".
Primary-source corroboration (RACER abstract): "retrieval-based drafts break when no exact match exists, while logits-based drafts lack structural guidance." — https://arxiv.org/abs/2604.14885
RACER's own stated limitations (same note): "Evaluation is limited to batch size 1 and greedy decoding; large-batch and sampling scenarios remain to be verified."

**C17. Cross-engine: existing n-gram proposers in llama.cpp are capped at a 4-token match, so drafts "stay short and get rejected" on long agent transcripts.**
URL: https://github.com/Mesh-LLM/mesh-llm/pull/1037
Verbatim (author danielwinterw):
> "The existing simple / cache proposers wrap llama.cpp's N-gram proposer, **capped at a 4-token match window (NGRAM_CACHE_MAX_NGRAM)**. In long agent transcripts a 4-token match is ambiguous: it occurs in many places with different continuations, so **drafts stay short and get rejected.**"
> "**Acceptance percentage alone does not rank a proposer.**"

**C18. SGLang's original "support ngram as speculative-model" request was closed as inactive before the feature later landed.**
URL: https://github.com/sgl-project/sglang/issues/5365 ("Closed", label `inactive`)
Verbatim (requester huakyouin): "I'm wondering if SGLang backend could support using ngram as a speculative model, similar to how vLLM does it… This is very useful when we don't want to load another full model just for drafting."

**C19. llama.cpp ngram-mod combined with MTP is reported to crash with CUDA OOM (linked from B7).**
URL: https://github.com/ggml-org/llama.cpp/issues/23154 — referenced as "Eval bug: CUDA ERROR crash when using MTP ngram-mod". Not independently read — **TO**.

**C20. vLLM `ngram_gpu` async acceptance rate does not match the CPU path — fix PR still blocked on rebase.**
URL: https://github.com/vllm-project/vllm/pull/44056 — "fix(ngram): match async ngram_gpu acceptance rate to CPU" (Open). GitHub banner: "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @shiyangyang2001-lgtm ."

**C21. Lexically-anchored reuse (i.e. PLD/n-gram retrieval) has limited recall under surface-form variation, and deterministic span copying is brittle.**
URL: https://arxiv.org/abs/2606.05742 (AdaPLD, v2 14 Jun 2026)
Verbatim (abstract): "We identify two limitations of existing reuse-based methods: **lexically anchored retrieval has limited recall under surface-form variation, and deterministic span copying can be brittle when the retrieved context does not uniquely determine the continuation.**"

**C22. Retrieval-based SD is "brittle to surface-level variations" because of "rigid lexical dependencies".**
URL: https://arxiv.org/abs/2606.00021 (SENSE, 14 Apr 2026)
Verbatim (abstract): "While Retrieval-based Speculative Decoding (RSD) is favored for its plug-and-play versatility, **its potential is impeded by rigid lexical dependencies, rendering both retrieval and verification brittle to surface-level variations.**"

**C23. Retrieval-enhanced drafters like SAM-Decoding "often trigger unnecessary retrievals" under heuristic switching.**
URL: https://arxiv.org/abs/2511.01282 (ReSpec, 3 Nov 2025)
Verbatim (abstract): "While model-based methods like EAGLE-2 are accurate but costly, **retrieval-enhanced methods like SAM-Decoding rely on heuristic switching strategies that often trigger unnecessary retrievals.**"

**C24. N-gram drafting hits a top-k, not top-1, ceiling — quantified.**
URL: https://arxiv.org/abs/2411.03786 (The N-Grammys)
Verbatim (abstract): "**While the predicted next token of the base model is rarely the top prediction of these simple strategies, we observe that it is often within their top-k predictions for small k.**"

**C25. Draft-target alignment can be adversarially collapsed, which is an operational vulnerability specific to speculative-decoding acceptance.**
URL: https://arxiv.org/abs/2607.21804 (ADSD, 23 Jul 2026)
Verbatim (abstract): "this guarantee of semantic equivalence masks a severe operational vulnerability: **draft-target alignment can be systematically attacked.**… On the GSM8K dataset, our attack increases the mean sample time by **62.3%** while preserving the task quality."

**C26. Model-only (non-n-gram) speculative decoding is "far less effective" outside English — the n-gram drafter is the fallback of last resort.**
URL: https://arxiv.org/abs/2605.30580 (4 Aug 2026)
Verbatim (abstract): "Motivated by the curse of multilinguality, we hypothesize that speculative decoding is far less effective for low-resource languages due to the limited multilingual capacities of smaller models. We test eleven languages under a standard speculative decoding setup and find strong evidence for our hypothesis."
> "Finally, we propose **weaker n-gram models as draft models; these provide moderate speed-ups due to their minuscule inference cost.**"

**C27. n-gram methods are the only category that reliably exploits repetition in test-time scaling — stated as an unexploited gap for the other categories.**
URL: https://arxiv.org/abs/2509.04474 (Aug 2025)
Verbatim (abstract): "Extensive experiments reveal that **simple n-gram-based methods effectively capture repetitive patterns, demonstrating unique potential in accelerating test-time scaling.** This phenomenon demonstrates the value of integrating n-gram-based methods with model-based or training-based approaches to balance acceleration for both repetitive and diverse reasoning in test-time scaling."

---

## D. Hardware-ruled-out for a single-GPU researcher (1× RTX PRO 6000 Blackwell 96 GB, sm120, single node)

| # | Artifact / claim | URL | Why it is out of reach | READ |
|---|---|---|---|---|
| D1 | SGLang NGRAM docs: "NGRAM … CUDA-only; **no `--enable-dp-attention`**; disables overlap scheduler & mixed chunked prefill" | https://docs.sglang.io/docs/advanced_features/speculative_decoding | DP attention is the multi-GPU data-parallel path and is disabled; overlap scheduler + mixed chunked prefill are also disabled, a throughput penalty with no single-GPU workaround. | RB |
| D2 | SGLang PR #28616 (NGRAM + DP attention) — "Validated on **2× H20**… e2e GSM8K test (**tp2/dp2**)" | https://github.com/sgl-project/sglang/pull/28616 | The only validated topology for NGRAM+DP is 2 GPUs (TP2×DP2). | RB |
| D3 | SGLang roadmap item (still unchecked): "Transport SAM across scheduler in **multi-scheduler system (disagg, DP, etc)**" | https://github.com/sgl-project/sglang/issues/21052 | Requires multiple scheduler processes / GPUs. | RB |
| D4 | SuffixDecoding SWE-Bench end-to-end: "vLLM is deployed on **4 H100 GPUs configured with 4-way tensor parallelism** and prefix caching enabled" | https://arxiv.org/html/2411.04975v3 (Fig. 6 caption) | The headline end-to-end agent result (up to 4.5×) is a 4-GPU deployment. | RB |
| D5 | SuffixDecoding cluster: "**single p5.48xlarge AWS instance equipped with 8 × NVIDIA H100 80G GPUs and 2TB of main memory**" | https://arxiv.org/html/2411.04975v3 (§4 and App. A) | 8×H100 / 2 TB host RAM; the cache-sizing math also assumes "144GB of CPU memory per A100 GPU". | RB |
| D6 | TensorRT-LLM N-Gram study: "Hardware: **8 × B200 GPUs (Blackwell)** … Tensor Parallel: 8" | https://github.com/NVIDIA/TensorRT-LLM/blob/a4349b2cd9a521af85335a517722f90fc14d4f15/blogs/tech_blog/blog07_NGram_performance_Analysis_And_Auto_Enablement.html | Every published N-gram speedup in that blog (96.13 % @bs1, 63.99 % @bs4, 33.06 % @bs32) is TP=8 on 8×B200. | RB |
| D7 | vLLM n-gram TP>1 work: `torchrun --nproc-per-node=2`; "in TP > 1 setting, it would need to spawn TP*X threads" | https://github.com/vllm-project/vllm/pull/26056 | The multi-threaded n-gram lookup optimization is only exercised at TP≥2. | RB |
| D8 | vLLM n-gram corruption bug environment: "Hardware: **2× NVIDIA RTX A5000 (Ampere SM 8.6), TP=2**" | https://github.com/vllm-project/vllm/issues/40875 | The `prompt_lookup_min=8` workaround was validated on a 2-GPU TP=2 stack. | RB |
| D9 | vLLM n-gram tool-truncation bug: `--tensor-parallel-size 4` on Qwen3-235B-A22B-GPTQ-Int4, `--max-model-len 131072` | https://github.com/vllm-project/vllm/issues/21307 | Model + TP4 exceeds a single 96 GB card at that context. | RB |
| D10 | Lookahead decoding headline scaling: "**4x with strong scaling on multiple GPUs** in code completion tasks" | https://arxiv.org/abs/2402.02057 | The 4× figure is explicitly multi-GPU strong scaling; the single-accelerator claim in the same abstract is only "up to 1.8x on MT-bench". | RB |
| D11 | vLLM n-gram GDN-corruption bug environment: "GPU: NVIDIA **GH200 480GB**", `--max-model-len 131072 --max-num-batched-tokens 131072` | https://github.com/vllm-project/vllm/issues/39273 | Reproducer needs 480 GB of memory and a 128 K-token single batch. | RB |
| D12 | vLLM n-gram-under-V1 failure report: "GPU 0..3: NVIDIA L40" (4 GPUs) | https://github.com/vllm-project/vllm/issues/16883 | 4-GPU TP deployment. | RB |
| D13 | Suffix decoding requires an out-of-tree CUDA dependency: `pip install arctic-inference==0.1.1`; docs' Limitations block: "**CUDA Only**: Suffix decoding currently requires CUDA devices", "**No DP Attention**: Data parallel attention is not yet supported" | https://github.com/sgl-project/sglang/pull/13553 and https://raw.githubusercontent.com/adityakamat24/sglang/404ea0ddb829585c5a5794b3945fe43df3ada70b/docs/advanced_features/suffix_decoding.md | No source I read states sm120 / RTX PRO 6000 support for the Arctic Inference suffix kernels, and the documented Limitations are CUDA-only + no DP attention. **This is an absence of evidence, not a documented sm120 block.** | RB |

**No artifact read in this sweep states a >96 GB VRAM requirement specifically for n-gram / prompt-lookup / suffix drafting.** The binding constraints found are: multi-GPU **topology** restrictions (D1–D3, D7), **evaluation** hardware being multi-GPU (D4–D6, D8), and the LLM sizes used (D9, D11). Every single-GPU n-gram/suffix result encountered ran on one 1× A100/H100/H20/4090/RTX 3090-class card — e.g. "Tested with Qwen3-0.6B on one NVIDIA RTX 4090" (https://github.com/sgl-project/sglang/pull/36978) and "single H100 GPU with batch size 1" (https://arxiv.org/html/2411.04975v3).

---

## Search log (verbatim)

### web_search queries
1. `vLLM speculative-config ngram prompt_lookup_max prompt_lookup_min documentation`
2. `SGLang speculative-algorithm NGRAM documentation`
3. `Prompt Lookup Decoding GitHub apoorvumang`
4. `REST Retrieval-Based Speculative Decoding arXiv 2311.08252`
5. `suffix decoding SpecDecoding arXiv 2411.04975`
6. `lookahead decoding negative result 2025 does not help`
7. `n-gram speculative decoding acceptance rate poor non-copy tasks 2025`
8. `token recycling speculative decoding 2025`
9. `vLLM pull request add ngram speculative decoding method`
10. `vLLM ngram_gpu speculative decoding PR 2025`
11. `SGLang NGRAM speculative algorithm source code`
12. `TensorRT-LLM lookahead decoding ngram`
13. `vLLM ngram speculative decoding documentation num_speculative_tokens prompt_lookup_max`
14. `SGLang speculative decoding NGRAM documentation 2026`
15. `arXiv 2026 n-gram speculative decoding negative result not worth it`
16. `prompt lookup decoding fails reasoning thinking phase speculative`
17. `llama.cpp --spec-type ngram-simple prompt lookup decoding PR`
18. `vLLM suffix decoding speculative method 2026`
19. `HuggingFace transformers prompt_lookup_num_tokens assisted generation docs`
20. `vLLM issue prompt_lookup_min tool call corruption structured output`
21. `llama.cpp speculative decoding spec-type ngram-simple lookup docs`
22. `SuffixDecoding MLSys 2025 arctic inference suffix decoding`
23. `RAPID retrieval speculative decoding 2025`
24. `"SAM decoding" suffix automaton speculative decoding paper`
25. `llama.cpp "spec-type" ngram-simple issue negative regression`
26. `speculative decoding n-gram reasoning thinking tokens acceptance low 2026 paper`
27. `prompt lookup decoding RAG long context retrieval drafting 2026 paper`
28. `n-gram speculative decoding batching batch size degrades 2026`
29. `SGLang speculative-algorithm SUFFIX arctic-inference pull request`
30. `vLLM pull request suffix decoding arctic inference merged`
31. `SuffixDecoding arctic inference vLLM production Snowflake blog`
32. `2026 paper "n-gram" speculative decoding "does not" help reasoning tasks acceptance`
33. `speculative decoding multi-GPU tensor parallel n-gram requirement 2025`
34. `Jacobi decoding parallel decoding limitations 2025 paper negative`
35. `n-gram speculative decoding RAG retrieval augmented generation acceptance rate 2026`
36. `github vllm ngram speculative "not planned" closed wontfix`
37. `github sglang ngram issue "not planned" closed`
38. `llama.cpp ngram speculative decoding issue regression closed wontfix`
39. `prompt lookup decoding issue closed "not planned" huggingface transformers`
40. `speculative decoding n-gram requires tensor parallel multi-GPU limitation issue`
41. `SGLang NGRAM DP attention not supported multi-GPU issue`
42. `"lookahead decoding" deprecated removed 2025 vLLM`
43. `vLLM prompt lookup decoding removed deprecated method`
44. `suffix decoding suffix tree speculative decoding multi-GPU 8xA100 evaluation only`
45. `EAGLE-3 speculative decoding 70B requires multiple GPUs 8xH100`
46. `ngram speculative decoding single GPU only limitation documentation 2026`
47. `vLLM ngram speculative decoding sm120 Blackwell RTX PRO 6000 issue`
48. `arctic-inference suffix decoding Blackwell sm120 support`
49. `speculative decoding ngram GPU kernel sm90 sm100 only arch support issue`
50. `RTX PRO 6000 Blackwell speculative decoding ngram vLLM`

### GitHub search API (api.github.com/search/issues)
51. `repo:vllm-project/vllm ngram speculative in:title` → 25 results
52. `repo:sgl-project/sglang ngram in:title` → 86 results
53. `repo:huggingface/transformers prompt_lookup in:title` → 0 results
54. `repo:NVIDIA/TensorRT-LLM lookahead in:title` → 14 results

### GitHub API object reads
55. `https://api.github.com/repos/vllm-project/vllm/pulls/44597`
56. `https://api.github.com/repos/vllm-project/vllm/pulls/18441`
57. `https://api.github.com/repos/vllm-project/vllm/pulls/40704`
58. `https://api.github.com/rate_limit` (×3)
59. `https://api.github.com/repos/steveseguin/b70-optimization-lab/contents/experiments/ornith-15-b70/notes`

### Direct page fetches (curl with proxies unset)
60. https://arxiv.org/abs/2311.08252
61. https://arxiv.org/abs/2411.04975
62. https://arxiv.org/html/2411.04975v3
63. https://arxiv.org/abs/2402.02057
64. https://arxiv.org/abs/2211.17148
65. https://arxiv.org/abs/2506.04708
66. https://arxiv.org/abs/2502.02022
67. https://arxiv.org/abs/2402.12374
68. https://arxiv.org/abs/2604.02047
69. https://arxiv.org/html/2604.02047v2
70. https://arxiv.org/abs/2604.14885
71. https://github.com/vllm-project/vllm/pull/12193
72. https://github.com/vllm-project/vllm/pull/44597
73. https://github.com/vllm-project/vllm/pull/44056
74. https://github.com/vllm-project/vllm/pull/40704
75. https://github.com/vllm-project/vllm/pull/18441
76. https://github.com/vllm-project/vllm/pull/4237
77. https://github.com/vllm-project/vllm/pull/4592
78. https://github.com/vllm-project/vllm/pull/25784
79. https://github.com/vllm-project/vllm/pull/26056
80. https://github.com/vllm-project/vllm/pull/29184
81. https://github.com/sgl-project/sglang/pull/11010
82. https://github.com/sgl-project/sglang/pull/13376
83. https://github.com/sgl-project/sglang/pull/13553
84. https://github.com/sgl-project/sglang/pull/27585
85. https://github.com/sgl-project/sglang/pull/28616
86. https://github.com/sgl-project/sglang/pull/35102
87. https://github.com/sgl-project/sglang/pull/36978
88. https://github.com/sgl-project/sglang/pull/37071
89. https://github.com/sgl-project/sglang/pull/37879
90. https://github.com/sgl-project/sglang/pull/39211
91. https://github.com/sgl-project/sglang/issues/21052
92. https://github.com/sgl-project/sglang/issues/5365
93. https://github.com/sgl-project/sglang/issues/36352
94. https://github.com/sgl-project/sglang/issues/36495
95. https://github.com/sgl-project/sglang/issues/36500
96. https://github.com/sgl-project/sglang/issues/38129
97. https://github.com/vllm-project/vllm/issues/41758
98. https://github.com/vllm-project/vllm/issues/35168
99. https://github.com/vllm-project/vllm/issues/16258
100. https://github.com/vllm-project/vllm/issues/6785
101. https://github.com/vllm-project/vllm/issues/46977
102. https://github.com/vllm-project/vllm/issues/40875
103. https://github.com/vllm-project/vllm/issues/52620
104. https://github.com/vllm-project/vllm/issues/56077
105. https://github.com/vllm-project/vllm/issues/39273
106. https://github.com/vllm-project/vllm/issues/42533
107. https://github.com/vllm-project/vllm/issues/49918
108. https://github.com/vllm-project/vllm/issues/30012
109. https://github.com/vllm-project/vllm/issues/24888
110. https://github.com/vllm-project/vllm/issues/16883
111. https://github.com/vllm-project/vllm/issues/21307
112. https://github.com/ggml-org/llama.cpp/pull/8648
113. https://github.com/ggml-org/llama.cpp/pull/18471
114. https://github.com/ggml-org/llama.cpp/pull/26283
115. https://github.com/ggml-org/llama.cpp/pull/25819
116. https://github.com/ggml-org/llama.cpp/pull/5479
117. https://github.com/ggml-org/llama.cpp/pull/19164
118. https://github.com/ggml-org/llama.cpp/issues/23184
119. https://github.com/Mesh-LLM/mesh-llm/pull/1037
120. https://github.com/huggingface/transformers/issues/30448
121. https://github.com/apoorvumang/prompt-lookup-decoding
122. https://github.com/Luowaterbi/TokenRecycling
123. https://github.com/steveseguin/b70-optimization-lab/commit/de1bd812fd460a44bcbcc31c48f82cc7cf28bf32
124. https://aclanthology.org/2025.acl-long.338/
125. https://aclanthology.org/2025.acl-long.595/
126. https://aclanthology.org/2025.findings-acl.1017/
127. https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/
128. https://docs.vllm.ai/en/latest/features/speculative_decoding/
129. https://docs.vllm.ai/en/v0.15.0/api/vllm/v1/spec_decode/suffix_decoding/
130. https://docs.sglang.io/docs/advanced_features/speculative_decoding
131. https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md
132. https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/a4349b2cd9a521af85335a517722f90fc14d4f15/blogs/tech_blog/blog07_NGram_performance_Analysis_And_Auto_Enablement.html
133. https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/spec_decode/ngram_proposer.py
134. https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py
135. https://raw.githubusercontent.com/adityakamat24/sglang/404ea0ddb829585c5a5794b3945fe43df3ada70b/docs/advanced_features/suffix_decoding.md
136. https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/ornith-15-b70/notes/2026-08-23-ornith35b-ngram-speculation-negative.md
137. https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/ornith-15-b70/data/2026-08-23-ornith35b-ngram-speculation-summary.json
138. https://raw.githubusercontent.com/hkr04/RACER/main/README.md
139. https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/
140. https://en.papernotes.org/ACL2026/llm_efficiency/racer_retrieval-augmented_contextual_rapid_speculative_decoding/
141. https://www.cs.cmu.edu/~csd-phd-blog/2025/suffix-decoding/ — surfaced by search, not fetched (**TO**)

### Blocked / failed (recorded honestly)
142. `https://huggingface.co/blog/assisted-generation` — curl exit 28 (timeout, twice). Not read ⇒ PLD's canonical HF blog is **TITLE ONLY** in this sweep. The repo https://github.com/apoorvumang/prompt-lookup-decoding *was* read.
143. `http://export.arxiv.org/api/query?...` — HTTP 429 on every attempt (rate-limited); the arXiv-listing leg was delegated to a subagent instead. arXiv `/abs/` and `/html/` fetches worked normally.
144. `https://r.jina.ai/...` proxy — HTTP 000, unusable.
145. `https://www.semanticscholar.org/paper/Speculative-Decoding-and-the-Curse-of-Paudel-Ginn/...` — HTTP 202, no readable body ⇒ **TITLE ONLY**.
146. github.com HTML intermittently returned HTTP 000 / timed out for ~15 minutes mid-sweep; all such URLs were retried successfully (except #142).

### arXiv search-UI queries (HTML, `https://arxiv.org/search/?searchtype=all&query=...&size=50&order=-announced_date_first`) — added in the second pass
147. `https://arxiv.org/search/?searchtype=all&query=n-gram+speculative+decoding&start=0&size=50&order=-announced_date_first`
148. `https://arxiv.org/search/?searchtype=all&query=prompt+lookup+decoding&start=0&size=50&order=-announced_date_first`
149. `https://arxiv.org/search/?searchtype=all&query=retrieval-based+speculative+decoding&start=0&size=50&order=-announced_date_first`
150. `https://arxiv.org/search/?searchtype=all&query=suffix+decoding+speculative&start=0&size=50&order=-announced_date_first`
151. `https://arxiv.org/search/?searchtype=all&query=lookahead+decoding&start=0&size=50&order=-announced_date_first`

### Additional direct fetches from the second pass
152. https://arxiv.org/abs/2605.30580 — Speculative Decoding and the Curse of Multilinguality
153. https://arxiv.org/abs/2509.04474 — Scaling Up, Speeding Up (SD benchmark for test-time scaling)
154. https://arxiv.org/abs/2506.04708 — STAND: Accelerated Test-Time Scaling with Model-Free Speculative Sampling
155. https://arxiv.org/abs/2411.03786 — The N-Grammys
156. https://arxiv.org/abs/2511.21699 — Cacheback
157. https://arxiv.org/abs/2606.05742 — AdaPLD
158. https://arxiv.org/abs/2607.21804 — Adversarial Prompts for Acceptance Collapse in Speculative Decoding
159. https://arxiv.org/abs/2511.01282 — When, What, and How: Rethinking Retrieval-Enhanced Speculative Decoding (ReSpec)
160. https://arxiv.org/abs/2606.00021 — SENSE
161. https://arxiv.org/abs/2608.20359 — SSR: Self-Speculation for Reasoning Models
162. https://arxiv.org/abs/2502.19732 — Speculative Decoding and Beyond: An In-Depth Survey of Techniques
