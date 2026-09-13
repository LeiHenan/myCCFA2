## Empty subsections (explicitly: nothing found, and the queries tried)

This map's rule is that a subsection with no genuine evidence says so rather than being padded. Below is, per subsection, exactly what came back empty and the queries that were tried. The full unabridged declarations, including every non-promoted lead URL, remain in the per-subsection evidence files (`evidence/kv-gapmap/S1.md` … `S14.md`, each with a `## EMPTY SUBSECTIONS` section).


### 1. paged/block KV management (`S1.md`)

No class came back empty. All four classes have at least four evidenced rows. Two caveats about evidence depth, recorded honestly rather than padded:

1. **Closed-unmerged PR rows were dropped for want of a stated reason.** I identified many closed-unmerged vLLM PRs in this subsection via the HTML issue-search endpoint (`is:pr is:unmerged label:kv-cache-manager`, and `is:pr is:unmerged "page size"` / `"block size"`). Examples retrieved as titles only: #46841 "[Bugfix][Core] Fix request-bound KV cache sizing" (CLOSED unmerged), #4762 "[Core][Bugfix]: fix prefix caching for blockv2" (CLOSED unmerged), #54949 "[XPU] env to control block size alignment for mamba" (CLOSED unmerged), #54735 "[Bugfix][DCP] Align SimpleCPU offload hybrid geometry" (CLOSED unmerged), #55622, #55134, #56320, #56223, #56222, #56221, #56205. For #46841 I retrieved the PR page and its rendered comment bodies and found **only** bot/AI text ("You have reached your Codex usage limits for code reviews." and "This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @lesj0610 ."), not a human closure rationale. For #4762 I retrieved the PR body but no closure comment. Per hard rule 3 I did not emit ABANDONED rows for these, because I could not retrieve a verbatim stated reason. The blocker is structural: GitHub lazy-loads PR conversation bodies, and the REST API (`/repos/vllm-project/vllm/issues/{n}/comments`) returned HTTP 403 "API rate limit exceeded for 50.7.158.236" on every attempt across the whole session.
2. **#53178 and #42571 also lack a retrievable closure reason.** Both are `stateReason=NOT_PLANNED` in the page metadata, but the fetched payloads contain no maintainer comment explaining the closure (for #53178 only the RFC body is present; for #42571 only the environment block). They are omitted from ABANDONED for the same reason. #42571 ("[Bug]: KV Block double free when using eager SimpleCPUOffloading + Sliding window attention", created 2026-05-13, closed 2026-09-12) sits in the same 2026-09-12 bulk-closure batch as ABANDONED-5 and is likely also a stale-bot closure, but the page does not show the bot comment, so I am not asserting that.

Queries tried for the classes where I searched hardest:
- ABANDONED: `is:issue is:closed reason:"not planned" kv cache repo:vllm-project/vllm`; `is:issue is:closed reason:"not planned" kv cache repo:sgl-project/sglang`; `is:pr is:unmerged label:kv-cache-manager repo:vllm-project/vllm`; `is:pr is:unmerged "page size" repo:vllm-project/vllm`; `is:pr is:unmerged "block size" repo:vllm-project/vllm`; `is:pr is:unmerged "memory pool" repo:sgl-project/sglang`; `vLLM "wontfix" OR "not planned" KV cache block size allocator issue closed`; `vLLM PR "closed" unmerged "KV cache" block manager revert`; `llama.cpp KV cache defragmentation "not needed" removed unified cache`; `llama.cpp KV cache defragmentation removed defrag_thold deprecated`
- HW_RULED_OUT: `state+space+model+KV+cache+memory+management+serving`; `hybrid+attention+SSM+KV+cache+allocator+page`; `virtual+memory+KV+cache+GPU+LLM+inference`; plus direct `arxiv.org/html/<id>v1` setup-section extraction for 2506.15155, 2605.22416, 2606.06256, 2509.06261.


### 2. prefix caching (`S2.md`)

No class came back empty. Coverage caveats and items that could **not** be promoted to a numbered row (per the verbatim-quote rule) are listed here.

**Closed-unmerged PRs found but NOT emitted as ABANDONED rows (no retrievable verbatim closure reason):**
- https://github.com/vllm-project/vllm/pull/54609 — "WIP: Enable prefix caching for FI ReplaySSM" by askliar. State CLOSED, `merged_at` absent. Page payload contains only the timeline event "askliar closed this Sep 7, 2026" and no comment; the PR body returned by the GitHub search API is empty. Topic: prefix caching for the FlashInfer ReplaySSM path in hybrid Mamba models.
- https://github.com/vllm-project/vllm/pull/54953 — "WIP: Unify MTP/STP/Prefix Caching for ReplaySSM" by askliar. Same author, same closure date (2026-09-07), same absence of any stated reason. Topic: unifying MTP/STP/prefix-caching lifecycles for ReplaySSM.
- https://github.com/vllm-project/vllm/pull/54319 — "[Prefix Cache] Support fine-grained SWA hits", closed unmerged by the author on the day it was opened with no comment at all.
- https://github.com/vllm-project/vllm/pull/51640 — "[Security] Include cache_salt in HF3FS external cache keys", closed 2026-08-10 by the author on the day it was opened, `merged_at` absent, no closure comment. Topic: salting × cross-instance external KV cache keys (the exact question in OPEN-7).
- https://github.com/vllm-project/vllm/pull/52971 — "[Kimi K3][Pref] Reuse internal checkpoint blocks for partial prefix caching", closed unmerged 2026-08-22. The only reviewer statement retrievable is a scope request, not a rejection: "Thanks for building on #52789 . Since this PR currently overlaps with its checkpoint infrastructure, please keep it in draft until #52789 lands, then rebase it on top and remove the duplicated pieces." Its functionality was subsequently carried by #53614 (CLOSED-9).

**arXiv-search coverage limit:** `https://arxiv.org/search/?searchtype=all&query=<q>` returned HTTP 200 for the first query in this session and HTTP 429 for every subsequent one, even with 18–25 s spacing and 4 retries per query. Only one listing page (`prefix caching KV cache`, 50 results) was captured. The `export.arxiv.org/api/query` Atom endpoint returned HTTP 429 on every attempt. All other paper evidence below was obtained from individual `arxiv.org/abs/<id>` and `arxiv.org/html/<id>v1` retrievals, which were reliable. As a result the paper-level sweep is *not* exhaustive and is biased toward papers named in the one listing page that succeeded.

---


### 3. eviction/reuse policy (`S3.md`)

None of the four classes is empty. Two sub-topics inside the subsection produced no independently-evidenced item and are recorded here rather than fabricated:

1. **RazorAttention-specific production/abandonment evidence.** I retrieved the canonical paper (https://arxiv.org/abs/2407.15891, READ BODY, "we propose RazorAttention, a training-free KV cache compression algorithm, which maintains a full cache for these crucial retrieval heads and discards the remote tokens in non-retrieval heads") but found no merged engine support, no closed-unmerged PR, and no issue about it that I could retrieve. Its retrieval-head premise is answered only indirectly by RazorAttention-adjacent work (the `DuoAttentionPress` in kvpress, "split heads into retrieval heads (no compression) and streaming heads (StreamingLLM approach)"). No CLOSED/OPEN/ABANDONED/HW row is emitted for RazorAttention specifically.
2. **Scissorhands-specific artifact.** Scissorhands appears only inside other artifacts' text (the vLLM RFC #5751 list of eviction methods; the vLLM issue #55463 motivation list) and inside `kvpress`'s history. I did not retrieve a Scissorhands paper page or any Scissorhands-specific engine artifact, so no row is emitted.

Queries that returned nothing usable for these two: `RazorAttention retrieval head KV cache eviction 2026 follow-up`, `Scissorhands PyramidKV limitations negative results`, `Scissorhands KV cache eviction implementation vLLM`.

---


### 4. KV quantization (`S4.md`)

No class came back empty. All four classes have at least five genuinely evidenced rows.

Two targeted sub-topics returned NO directly-retrieved evidence and are therefore NOT represented as rows (rather than being fabricated):

1. **`kv-cache-dtype` calibration-free INT4 per-token-head accuracy on Qwen3-4B specifically.** No page was retrieved that reports GSM8K/logprob deltas for `int4_per_token_head` on Qwen3-4B; the shipped accuracy test uses `meta-llama/Llama-3.2-1B-Instruct` at `MAX_MODEL_LEN = 1024`, `max_tokens = 4`, `tensor_parallel_size = 1`, `backend = "TRITON_ATTN"` (read locally at `/tmp/kvsrc/vllm-main/tests/models/quantization/test_per_token_kv_cache.py`). Queries attempted are listed below.
2. **A revert of a KV-quantisation change in vLLM itself** (as opposed to SGLang's ABANDONED-11). Queries attempted did not surface one, and no such PR page was retrieved, so no row was emitted.

---


### 5. offloading & tiered storage (`S5.md`)

None of the four classes came back empty. Caveats on completeness and on items deliberately NOT emitted as rows:

- **RDMA KV stores as such**: no dedicated RDMA-KV-store artifact was emitted as its own row. The RDMA material I actually retrieved is Mooncake's Messenger (cited in HW-1), vLLM's NIXL-based P2P tier (cited in CLOSED-4 and CLOSED-13), and the 2026 papers that argue *against* whole-prefix RDMA fetch (only abstract-level retrieval; not emitted as rows).
- **InfiniGen** (verified: arXiv 2406.19707, "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management", submitted 28 Jun 2024): its setup is a **single** RTX A6000 48GB with 96GB DDR4-2666 (verbatim: "We run the experiments on a system equipped with an NVIDIA RTX A6000 GPU [ 44 ] with 48GB of memory and an Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory. PCIe 3.0 × 16 interconnects the CPU and GPU."). It is therefore **NOT** HW-ruled-out, and it is a KV-selection/prefetch technique rather than a storage-tier technique, so it is not a clean CLOSED row for this subsection either. Left out of the tables deliberately rather than misclassified.
- **CacheGen** (verified: arXiv 2310.07240) is **NOT** HW-ruled-out — its setup is one NVIDIA A40 GPU server with 384GB host memory ("Hardware settings: We use an NVIDIA A40 GPU server to benchmark our results. The server is equipped with 384GB of memory and two Intel(R) Xeon(R) Gold 6130 CPUs"). Emitted as CLOSED-10 only.
- **Pensieve** (verified: arXiv 2312.05516) is **NOT** HW-ruled-out — evaluated on one NVIDIA A100 PCIe 80GB (Microsoft Azure NC24ads_A100_v4). Emitted as CLOSED-11 only. Note: "Pensieve: A Tiered, Fully-Automated, and Cost-Effective Virtual Cluster..." is a *different, unrelated* Pensieve.
- **FlexGen** (verified: arXiv 2303.06865, "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU") is explicitly single-GPU, so it is not HW-ruled-out; not emitted as a row because I did not retrieve a body quote tying it to a currently-open question.
- **HCache** (verified: arXiv 2410.05004, "Fast State Restoration in LLM Serving with HCache"): its headline testbed is a **partially** ruled-out configuration — verbatim "4 × A100-40G SXM4 connected via NVLink. The host has 2 × AMD EPYC 7642 CPUs, 256G DDR4 memory, and 4 × Samsung PM9A3 4TB enterprise SSDs." — but the paper also states verbatim "For Llama2-7B/13B, we use a single A100 GPU to serve them." Because it does **not require** more than one GPU, it is deliberately NOT emitted as an HW_RULED_OUT row.
- **CXL-SpecKV** (arXiv 2512.11920): I did not retrieve its body, so its hardware is unverified. Not emitted — a TITLE ONLY row asserting a hardware requirement would be a guess.
- **TieredKV / SpecKV / DeepSpeed ZeRO-Inference**: no verified ID or retrieved setup section. No evidence, no row.
- **Klotski** (arXiv 2502.06888): verified to be an MoE inference paper, not KV-cache offloading. Off-topic; excluded.
- **NVMe/SSD tiers, GPUDirect Storage, host-memory bandwidth bottleneck**: all three are covered — GPUDirect Storage by ABANDONED-4 (KvikIO prototype slower than everything) and ABANDONED-14 (Tutti: GDS stays CPU-centric); host-memory bandwidth by ABANDONED-18 and ABANDONED-19; NVMe/SSD tiers by CLOSED-4, CLOSED-12, CLOSED-21, CLOSED-22, ABANDONED-5, ABANDONED-15, ABANDONED-16.
- **Closed-unmerged with NO stated reason (deliberately NOT emitted as an ABANDONED row)**: vLLM PR #52784 "[Feature][KV Offload] Add capacity-bound LRU eviction to FileSystemTierManager" (https://github.com/vllm-project/vllm/pull/52784) is recorded as closed unmerged on 2026-08-19 by its author XiaYiHann, but the page carries no closing comment and therefore no retrievable stated reason. Under the no-fabricated-quote rule it is reported here rather than as a row.
- **Automated-stale closures (bot text only — NOT presented as maintainer decisions)**: three KV-eviction items in the offload/prefix-cache area were closed by an inactivity bot with no human rationale, so they are recorded here rather than as ABANDONED rows. All three carry the same bot text; verbatim examples: "This pull request has been automatically closed due to inactivity. Please feel free to reopen if you intend to continue working on it. Thank you!" (vLLM PRs #22236 "Workload-Aware KVCache Eviction Policy" https://github.com/vllm-project/vllm/pull/22236 and #27539 "[Core] Prefix cache: frequency- and cost-aware eviction (opt-in)" https://github.com/vllm-project/vllm/pull/27539) and "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!" (vLLM issue #23641 "[RFC]: Frequency and Cost Aware Eviction Policy for Prefix Caching" https://github.com/vllm-project/vllm/issues/23641). This evidence came from delegated retrieval, not from pages I fetched myself.
- **Item deliberately not classified**: vLLM PR #26921 "[V1][KV Cache] Add ARC (Adaptive Replacement Cache) eviction policy" was closed and then superseded — the recorded author comment is "I'll reopen the PR, messed up forcing the commit with sign off." (https://github.com/vllm-project/vllm/pull/26921), and the replacement PR #27039 "Implement ARC KV cache eviction policy" was merged. This is an author slip, not an abandoned technique, so it gets no row. Delegated retrieval; I did not fetch these two pages myself.
- **Automated-stale closures**: the only stale-bot closure I retrieved in this subsection is vLLM #39766 ("stale — Over 90 days of inactivity", closed as not planned). It is emitted as a row only in the QUERIES/notes trail, not as an ABANDONED row, because the closure reason is bot text and not a maintainer decision. Every other ABANDONED row above carries either a human/author statement or a paper's own published conclusion.


### 6. compression/low-rank (`S6.md`)

None of the four classes came back empty. Specifically:
- CLOSED: 16 rows, all with retrieved sources.
- OPEN: 9 rows, all with retrieved sources.
- ABANDONED: 12 rows. **Caveat on quality, not quantity:** five of the twelve (ABANDONED-2, -9, -10, -12 and the bot half of -1) rest on stale-bot closure text rather than a human maintainer decision, and are labelled as such. Only ABANDONED-1 contains a genuine human maintainer objection ("The required data is large IMHO."), and ABANDONED-3..-8 are paper-level negative/limiting results rather than engine-level rejections. I found **no** human-maintainer "wontfix"/"not planned"/explicit-rejection artifact for: cross-layer KV merging (MiniCache-style) in vLLM or SGLang, latent/low-rank KV (Palu-style) in vLLM or SGLang, learned KV codebooks, or MLA upcycling (X-EcoMLA/TransMLA/MHA2MLA-style). The `label:wontfix` search on vllm-project/vllm returned zero results, so a "wontfix label" artifact may not exist in that repo at all.
- HW_RULED_OUT: 8 rows.

Searches that came back empty and are therefore reported as genuinely-no-result rather than omitted:
- `repo:vllm-project/vllm label:wontfix` → 0 results (GitHub search HTML, one attempt)
- arXiv full-text search `MiniCache` → 0 parsed results on the attempt that hit an arXiv 429; the record was instead confirmed by direct abstract fetch of a specific ID
- arXiv full-text search `You Only Cache Once` → 0 parsed results (arXiv 429 on that attempt); confirmed by direct abstract fetch
- arXiv full-text search `Dynamic Memory Compression retrofitting` → 0 parsed results (arXiv 429); confirmed by direct abstract fetch
- GitHub REST search API was unusable for this entire session: every call returned `{"message":"API rate limit exceeded for 50.7.158.236. ..."}` (HTTP 403), so all GitHub discovery was done through the HTML search page and direct issue/PR page fetches.

---


### 7. cross-request & cross-replica sharing (`S7.md`)

- **KV multicast / broadcast (one-to-many KV dissemination)**: no shipped artifact, merged PR, open issue, or paper found that implements KV cache multicast or broadcast to multiple consumers. The closest retrieved item was vLLM PR title "[Bugfix][CUDA] Fix the FlashInfer's fused all-reduce issue where NVLink multicast isn't available" (https://github.com/vllm-project/vllm/pull/55973), which is NVLink multicast for all-reduce, i.e. tensor-parallel communication, not KV dissemination — not emitted as a row. No ABANDONED row either: I found no closed-unmerged/wontfix artifact with a retrievable verbatim closure reason on this topic, so per the hard rules I emit nothing rather than a row without a quote. Queries run for this: `all:"KV cache" AND all:"multicast"`, `all:"KV cache" AND all:"broadcast"` (arXiv API — both rate-limited, see concerns), `kv cache broadcast OR multicast` (GitHub HTML search, vllm-project/vllm), `multicast OR broadcast kv cache` (GitHub HTML search, sgl-project/sglang), `cross-engine OR cross engine OR multicast` (GitHub HTML search, LMCache/LMCache).
- **A merged cross-*vendor* engine KV exchange (e.g. a shipped vLLM↔TensorRT-LLM KV format bridge)**: the only retrieved cross-engine artifact is LMCache's (vLLM and SGLang, via the LMCache layer), emitted as CLOSED-10. The TensorRT-LLM HTML search `kv cache sharing OR reuse across instances` returned no parsed result rows within the retrieval window, so nothing is emitted for TRT-LLM. Queries run: `cross-engine OR "cross engine" kv` (vllm-project/vllm), `cross-engine OR cross engine OR multicast` (LMCache/LMCache), `kv cache reuse across requests OR instances` (NVIDIA/TensorRT-LLM — returned zero parsed rows).
- **KV deduplication as a first-class cross-replica feature** (distinct from prefix-cache key determinism and from LMCache's fingerprint dedup): no standalone shipped "KV dedup service" artifact was found. Two partial answers are emitted instead: CLOSED-5 (identical content → identical keys cross-process) and CLOSED-9 (`--enable-dedup-content`).


### 8. long context (`S8.md`)

No class came back genuinely empty. Coverage notes and thin spots, stated honestly:

- **ABANDONED for *ring attention KV specifically***: no closed/unmerged PR or paper-level retraction of a ring-attention-for-KV proposal was found in vLLM, SGLang, or arXiv within the window. What exists instead is vLLM's design doc declaring the partial-Q/partial-KV ring path "under active development" (HW-3) — that is an OPEN status, not an abandonment, so no ABANDONED row was emitted for it. I did not find a maintainer rejection of ring attention to quote, and I will not invent one.
- **CLOSED for "RAG with many documents" at the *engine* level**: the only shipping artifacts I could verify are prefix caching (`--enable-prefix-caching`, `--prefix-match-unit`) and the research-side CAS/CoinRAG line. I found no merged vLLM/SGLang PR that specifically targets many-document RAG KV (e.g. per-document cache selection inside the engine), so that sub-question has no CLOSED row of its own; it is represented by CLOSED-3 (system-prompt bloat) and CLOSED-8 (CAS).
- **CLOSED for "attention sinks for long context" inside a *dense GQA long-context* backend on sm120**: sink support exists per backend (CLOSED-6) but the in-tree docs explicitly exclude sinks from the Blackwell head_size=256 FA4 kernel (`does not support logit soft capping, attention sinks, mm_prefix/R-SWA masking, DCP, or windowed encoder attention`), so on this target hardware the sink path is backend-conditional rather than universally available. Counted as a caveat on CLOSED-6, not as a separate row.
- **Thin: exact PR number for the `--enable-mamba-fine-grained-prefix-cache` merge.** CLOSED-3 is evidenced from the shipped docs page and the in-tree config source rather than from the merge PR, because I did not retrieve the merge PR page.
- **arXiv API sweeps returned no data**: `export.arxiv.org/api/query` answered HTTP 429 (rate-limited) throughout this session, and `arxiv.org/list/...` returned HTTP 404 for the 2606/2609 listing URLs I tried. All paper-level findings in this file therefore come from `arxiv.org/abs/<id>` and `arxiv.org/html/<id>v1`, which worked reliably.

---


### 9. KV lifetime/TTL economics (`S9.md`)

None of the four classes came back empty. Two honesty caveats rather than empty classes:

1. **No maintainer-authored "wontfix" / "not planned" rejection of a KV-lifetime proposal was found.** The only `NOT_PLANNED` stateReason in the ABANDONED set (#23641) and the two other stale closures (#22236, #27539) are all `github-actions` STALE BOT text, explicitly labelled as such. Every human closure of a KV-eviction PR found was an *author self-close*. Queries run for this: `repo:vllm-project/vllm label:wontfix kv` (returned total 0), `repo:vllm-project/vllm eviction policy in:title`, `repo:vllm-project/vllm "TTL" kv in:title,body`, plus reading the full label lists of #26921, #27039, #42985, #43191, #43725, #49114, #41383, #54327, #36311 and #52784 (no `wontfix`/`not planned` label on any of them).
2. **One ABANDONED topic has no retrievable stated reason and is therefore folded, not rowed.** PR #52784 (capacity-bound LRU eviction for the FS tier, author self-close 2026-08-19, one day after opening) is documented inside ABANDONED-3 with the explicit marker "NO STATED REASON FOUND". It is not given its own row because no verbatim reason exists on the page. Similarly, PR #35652 (SwapConnector) is folded into ABANDONED-2 with only its timeline line quoted, because it carries no closing comment.

Also not represented as rows, for the same reason: several 2026 HiCache PRs in SGLang (e.g. #39297, #39283, #39269, #39267, all created 2026-09-13 and still OPEN) were seen only as list entries in a search result and were not opened, so they are omitted rather than cited at TITLE ONLY.


### 10. disaggregated prefill/decode (`S10.md`)

No class is empty. Coverage notes and the honest weak spots:

- **SGLang is thin on purpose.** `/tmp/kvsrc/sglang-main` is a partial checkout (only `benchmark/`, `docker/`, and docs fragments present; no `python/sglang` tree), so no SGLang source-level verbatim grep was possible. SGLang evidence here is limited to its documentation page (CLOSED-7 URL) and to `GitHub` search results, and the SGLang-specific ABANDONED row (ABANDONED-7) is actually a Dynamo-side issue about the SGLang backend.
- **LMCache layerwise was dropped from ABANDONED.** LMCache PR #4221 ("feat(sglang): add LMCachePDConnector for PD disaggregation over NIXL") was closed unmerged and **by its own author** (`skaulintel closed this Jul 23, 2026`) only ~21 minutes after opening, with no comment on the page. I could retrieve the closure *event* but not a *stated reason*, so per the hard rules it is not emitted as an ABANDONED row. Its PR body does record a real layerwise-vs-PD incompatibility: "Add LMCachePDConnector, extending the non-layerwise LMCacheConnector because only engine.store (not store_layer) forwards transfer_spec to the storage manager / PDBackend." Related open LMCache layerwise correctness bugs exist (#4921, #4132, #4133) but are storage-tier layerwise issues rather than P/D-transfer ones.
- **No "smaller-model, single-GPU P/D" paper was found.** Every P/D or KV-transfer systems paper retrieved for this subsection assumed ≥2 accelerators. The RTX PRO 6000-specific search queries below returned nothing on-topic, so I did not manufacture a row.
- **The `Prefill-as-a-Service` paper note** surfaced through a third-party aggregator (a GitHub paper-notes issue) was not used as a source; the arXiv abstract page was retrieved directly instead and is what is cited.


### 11. negative results (`S11.md`)

- **TensorRT-LLM: no longer empty, but still thin.** My own repo-scoped search `repo:NVIDIA/TensorRT-LLM is:closed reason:"not planned" "kv cache"` returned HTTP 403 (shared-IP search rate limit) on every attempt. The delegated TensorRT-LLM investigation later supplied exactly **one** unambiguous maintainer "we will not do this" KV-cache decision (ABANDONED-55, KV reuse across LoRAs), one merged revert (ABANDONED-56), one stale-bot close (ABANDONED-57) and one relocation (ABANDONED-58). It also found **no** verified case where TensorRT-LLM maintainers concluded that KV quantization (FP8/INT4/INT8), eviction or offloading does not pay off — in that repository the FP8/NVFP4 KV work is shipping in the opposite direction. Two TensorRT-LLM-adjacent items were visible only as titles inside the *Dynamo* tracker (`#4387` "[FEATURE]: Layerwise KV Cache Transfer for Disaggregated TensorRT-LLM Serving" and `#729` "[FEATURE]: How to manage offload percentage", both closed `not_planned`); neither page was retrieved, so under the verbatim rule they are **not** emitted as rows.
- **HW_RULED_OUT items with a retrieved setup sentence for *papers*: EMPTY.** All seven HW rows are GitHub issues, not papers. The delegated arXiv investigation retrieved **only abstract pages, never full-text HTML**, and therefore could not certify any multi-GPU/NVLink/InfiniBand requirement from an experimental-setup section. Every hardware string it did retrieve was single-GPU and hence *in* scope (e.g. arXiv 2606.21868: "On a real 24 GiB RTX 3090, WiSP achieves up to 2.0x the decode throughput of static offload at the same memory budget when the model does not fit."; arXiv 2605.06675: "The entire calibration takes 1.6 s on a single GPU and adds zero overhead at inference time."). HW-6 and HW-7 are TITLE-ONLY leads with no evidenced requirement.
- **HuggingFace TGI ABANDONED items in the 2025-06 → 2026-09 window: EMPTY.** The delegated investigation reports: `"kv cache" "not planned"` → total_count 0; `"kv cache" offload` → 1 result (issue #991, closed not_planned but on 2024-03-27, **outside the window**); the 72 hits on `"prefix caching"` were "almost entirely log-line noise ('Disabling prefix caching because of VLM model')". Two scope-limit statements were seen as search-API text-match fragments on **open** issues (#3333: "Note that prefix caching is not supported for VLMs. The hardware also needs to support flashinfer or flash-attention2, so prefix caching is not supported on GPUs with a CUDA capability that is lower than 8.0."; #3110: "NotImplementedError: Vlm do not work with prefix caching yet") — but these are **support-scope limits, not abandoned optimizations**, and were not full-page reads. The TGI closed-unmerged PR enumeration failed with HTTP 403 on both queries, so this gap is real and not a true negative.
- **Papers that FAILED TO REPRODUCE a claimed KV-cache speedup: EMPTY** for the paper literature. The only reproduction-style negative result found is non-paper (ABANDONED-46, the agentic-kv-cache trace study). One paper is *adjacent* — arXiv 2503.24000 identifies "missing pieces in their performance measurement, which could hinder their adoption in practice" — but it does not frame itself as a failed reproduction.
- **A maintainer statement of the exact form "offloading gave no speedup because PCIe bandwidth bound" for LMCache or vLLM: EMPTY.** The delegated LMCache/Dynamo investigation states plainly: "I found no PR or issue stating *'offloading gave no speedup because PCIe bandwidth bound'*, *'compression hurts accuracy'*, *'we reverted the async loader'*, or *'not worth it for single node'* in those words." The closest genuine analogues are ABANDONED-17, -18, -22 (offload measures as pure overhead vs vLLM's own prefix cache) and ABANDONED-45 (PCIe-bound prefetching, but for MoE experts).

---


### 12. hardware-ruled-out (`S12.md`)

No class is empty for S12. All four classes (CLOSED, OPEN, ABANDONED, HW_RULED_OUT) have multiple evidenced rows. Three specific *sub-questions* did come back thin or empty, and are recorded here rather than fabricated:

1. **"Multi-node KV technique abandoned by a *paper's* own conclusion, with a stated reason in the paper's limitations section"** — I did not find a paper that explicitly abandons a multi-GPU KV technique in its own limitations section. What I found instead are (a) PR-level abandonments with maintainer/stale-bot reasons, and (b) papers whose results undercut their own technique (ABANDONED-13 Beyond the Buzz, ABANDONED-14 Star Attention, ABANDONED-12 the vLLM disagg-prefill doc note) plus MemServe's own admission that the multi-node transport was never implemented (HW-14). The Star Attention, Beyond-the-Buzz and MemServe rows are the closest genuine matches.
2. **"SGLang-side ABANDONED PR/issue with a maintainer's *stated wontfix* reason"** — I still could not retrieve one, even after a dedicated sub-agent sweep. What exists instead is: (a) three SGLang PRs closed unmerged with reasons that are **entirely bot text** or **entirely absent** — ABANDONED-18 (`#27276`, stale bot: "Closing this because it has had no updates in 99 days."), ABANDONED-19 (`#21865`, stale bot: "Closing this because it is still a draft and has not been updated in 146 days."), ABANDONED-20 (`#25846`, author self-close with no comment and an empty template); and (b) documentation-level retraction (ABANDONED-11, `--cp-strategy zigzag` "temporarily unavailable"). **No SGLang maintainer statement declaring a multi-GPU KV technique infeasible was found.** My own direct GitHub search API attempts were blocked by HTTP 403 (secondary rate limit) in every case; the sub-agent reported the same 403 on its SGLang `disaggregation in:title` queries. This area is explicitly **under-sampled**.
3. **"A hardware-ruled-out KV technique specific to *this* GPU generation (sm120 / RTX PRO 6000 Blackwell)"** — nothing I retrieved is gated on sm120 specifically. All hardware gates I found are of a different kind: GPU *count* (TP/DCP/PCP rank counts), interconnect class (NVLink/MNNVL/InfiniBand/RDMA), or model scale. The one sm120-adjacent artifact is vLLM PR `#34795`'s author benchmark table, which reports DCP on 8× and 16× "RTX 6000 Pro" — i.e. sm120 appears in that data only as a *multi-GPU* configuration (`TP 8 DCP 8` / `TP 16 DCP 16`), never at TP=1.

Queries that returned nothing usable for these gaps, verbatim as executed:
- `https://api.github.com/search/issues?q=repo:sgl-project/sglang+context+parallel+is:issue+is:closed&per_page=40&sort=created&order=desc` → HTTP 403
- `https://api.github.com/search/issues?q=repo%3Avllm-project%2Fvllm+disagg+in%3Atitle+type%3Apr+is%3Aunmerged&per_page=40&sort=created&order=desc` → HTTP 403
- `https://api.github.com/rate_limit` → confirmed `"core": {"limit": 60, "remaining": 0}`
- Per the research sub-agent's report (declared as un-retrieved rather than filled): SGLang `disaggregation in:title` (×2), vLLM `label:wontfix`, vLLM `nixl in:title ... is:unmerged` — all HTTP 403; and `repo:vllm-project/vllm label:not-planned` returns `total_count 0` because that label does not exist in vLLM.
- `decode context parallel` as an exact phrase in the arXiv API returned 0 usable results (per sub-agent report); the workable term is "context parallelism" + KV cache decoding.

---


### 13. single-GPU agentic/long-context (`S13.md`)

No class came back empty. Coverage notes and shortfalls, stated honestly:

- **No ABANDONED row exists for "KV reuse across turns with changing system prompts" that is a clean maintainer rejection.** The best evidence found is SGLang PR #21064 (ABANDONED-6, closed unmerged by its author, with a measured 4.8k → 20.8k prefix-hit improvement left on the table) and vLLM issue #29286 (ABANDONED-9, stale-bot closed). Neither is a maintainer saying "we will not do this". Searches for a maintainer-stated rejection on this specific topic returned nothing retrievable.
- **No ABANDONED row for "shared-prefix KV deduplication within one engine" as a distinct feature.** The closest material is block-hash dedup, which is simply how vLLM/SGLang work by default (CLOSED-2, CLOSED-1), plus the block-size divisibility defect surfaced in ABANDONED-7. No one appears to have proposed and abandoned a separate within-engine dedup feature under that name in the retrieved corpus.
- **ABANDONED-19 (SGLang PR #20088) lacks a verbatim closure reason.** The page shows `Closed` with 17 unmerged commits but posts no closure comment that could be retrieved. It is flagged inline rather than silently passed off as a reasoned abandonment.
- **TRT-LLM PR #15828 (ABANDONED-12) likewise has no closure comment**; the verbatim quote supplied is the last substantive maintainer comment and is explicitly labelled as such, not as a closure reason.
- **SGLang PR #33315 ("[srt] Batchable context forwards over cached prefix KV", closed unmerged 2026-08-26) was DROPPED** from ABANDONED because no verbatim closure reason could be retrieved from the page. It is recorded here so the lead is not lost: https://github.com/sgl-project/sglang/pull/33315 (retrieved, READ BODY of the PR chrome; the author closed it with no comment).
- **vLLM #36311 ("[Feature Request] Pluggable KV cache eviction policy with attention sink protection")** was retrieved and is cited inside ABANDONED-1 as the source of the ref_cnt==0 lesson, but is not emitted as its own row because its closure state and reason were not independently confirmed on the page beyond the quoting of it by the ACE author.

---


### 14. correctness/observability (`S14.md`)

None. All four classes have at least one evidenced entry.

One sub-topic was thin and should be flagged rather than padded: I found no *retrieved, verbatim-quoted* artifact that closes the loop on "KV eviction changes the measured quality of a model" as a first-class, shipped **metric** (as opposed to a published audit). CLOSED-15/CLOSED-16 and ABANDONED-10/ABANDONED-11 are papers; the shipped engine metrics in CLOSED-4/CLOSED-5 measure *cache* behaviour (hit rate, block lifetime/reuse), not output-quality deltas attributable to eviction. I did not find a merged PR that adds an output-quality-vs-eviction metric to vLLM or SGLang, and I am not asserting one exists.
