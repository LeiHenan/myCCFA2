# KV-cache self-admitted gaps with **no matching public discussion**

**Compiled:** 2026-09-14 · **Author:** candidate-hunting subagent · **Deliverable:** negative evidence only — no recommendations, no priorities.

Target rig (decides reachability, never whether an item is reported): **one** RTX PRO 6000 Blackwell 96 GB, **sm120 / compute capability 12.0**, driver 580, 208 CPU cores, single node, no NVLink / no InfiniBand / no cluster; **vLLM 0.29.0** (`/root/ccfa_venv/lib/python3.12/site-packages/vllm`), **SGLang 0.5.19** (`/root/autodl-tmp/venvs/sglang/lib/python3.12/site-packages/sglang`), target model **Qwen3-4B** (dense, full attention, not MoE / not hybrid Mamba-GDN / not MLA), drafters dflash2 and an EAGLE3 Qwen3-4B drafter. CPU / host-memory / local-disk KV tiers are in reach.

---

## 0. Method header — how candidates were generated, and how absence was tested

### 0.1 Candidate generation (source-first, never discussion-first)

Discussions were deliberately **not** used as the seed. The seed is the installed engine source trees on the rig, reached over SSH (`ssh -p 11640 root@connect.westd.seetacloud.com`).

1. **Existing repo tool, re-run and re-verified.** `probes/p17-open-cells/scan_engine_todos.py --selftest` was copied to the rig and run against both trees. Selftest passed (`selftest ✔ 模式匹配/主题分类/计数（3 条命中，kv 类 1 条）`). Full-scan totals reproduced the previously reported numbers exactly:

   | tree | spec | **kv** | sched | quant | other | total lines |
   |---|---|---|---|---|---|---|
   | vLLM 0.29.0 | 59 | **58** | 101 | 223 | 3511 | 3952 |
   | SGLang 0.5.19 | 85 | **72** | 151 | 214 | 4119 | 4641 |

   ```
   /root/ccfa_venv/bin/python /tmp/scan_engine_todos.py --root <vllm>   --tag vllm   --out /tmp/vllm_todos.md
   /root/ccfa_venv/bin/python /tmp/scan_engine_todos.py --root <sglang> --tag sglang --out /tmp/sglang_todos.md
   ```

2. **Recall repair — the bundled scanner has a blind spot.** Its topic classifier is *first-match-wins* over the dict `{spec, kv, sched, quant}`, so a KV-cache line that also contains the substring `draft`, `propos`, `chunked`, `quant`, `fp8`, … is silently filed under `spec`/`sched`/`quant` and never reaches the 58/72 KV buckets. A second extractor was written for this run — a scratch file, **not committed to the repo** — that reuses the bundled tool's 13 trigger patterns unchanged and adds tiered classification:
   * **Tier A** = the canonical 58 + 72 KV-topic hits (reproduces the bundled tool).
   * **Tier B** = the **hit line itself** matches one of 41 KV keywords (`kv_cache`, `block_table`, `prefix_cache`, `radix`, `hicache`, `offload`, `evict`, `swap`, `kv_cache_dtype`, `page_size`, `paged_attention`, `sliding_window`, `chunked_prefill`, `num_gpu_blocks`, …) even though the line was filed under another topic.

   Tier B added **120 vLLM + 84 SGLang = 204** further KV-relevant self-admitted-gap lines. All 130 Tier-A hits are reproduced in this document's triage; several Section 1 and Section 2 items came only from Tier B.

3. **Triage.** ~65 distinct `file:line` sites were pulled from the rig with ±10 lines of real context (a scratch `ctx.py`) and read. Sites were dropped when the statement was (a) platform-specific to hardware we do not have (SM90/SM100/SM8.0/ROCm/AITER/HIP/XPU/MLX/NPU), (b) model-family-specific to something other than a dense full-attention model (MLA, MoE, hybrid Mamba/GDN, DeepSeek-V4, MiniMax, Kimi), (c) about speculative decoding rather than KV (owned by another researcher), or (d) a defensive `assert` on a code path that cannot be selected. 34 distinct candidates survived into search (14 of them added only after the alternate-phrasing sweep).

### 0.2 Absence testing — the exact procedure

For every surviving candidate the verbatim string — and, where the string was too generic to be a fingerprint, the identifier names appearing on the same line — was searched on **four surfaces**:

| surface | mechanism | why this mechanism |
|---|---|---|
| GitHub issues **and** PRs, `vllm-project/vllm` | `GET https://github.com/vllm-project/vllm/issues?q=<query>` | `github.com/search` returned **HTTP 429** on every attempt; the repo-scoped issues list is server-rendered, returns 200, and lists **issues and PRs together** |
| GitHub issues **and** PRs, `sgl-project/sglang` | `GET https://github.com/sgl-project/sglang/issues?q=<query>` | same |
| arXiv | `GET https://arxiv.org/search/?searchtype=all&query=<q>` (HTML UI) | `export.arxiv.org/api/query` returned **HTTP 429 / `Rate exceeded.`** on every call; the HTML UI works |
| open web | the `web_search` tool, results followed by `curl` on the specific page | `web_fetch` was unreliable; every `web_search` hit that mattered was re-fetched directly with `curl -sL --retry 4 -A "Mozilla/5.0 … Chrome/126.0"` and read |

GitHub result pages were parsed from the embedded React payload by brace-matching each node that carries `"__isIssueOrPullRequest"`, so **titles, numbers and kinds come from the page itself**, and PR/issue **bodies were read** from the `comment-body` / `markdown-body` blocks when a hit mattered. Spot-checks confirmed body text is present and greppable for both issues and PRs — e.g. the vLLM PR page carries the words *"Resolves the FIXME: num_stored_blocks can be stale and omit evicted blocks in the middle of the request"*, which is exactly the string needed to exclude a candidate (see §2.3). A prior note in this project that "PR comment bodies don't render" is **false** for this environment.

**104 GitHub query executions** (65 + 25 + 14 primary / alternate-phrasing / focused-verification, each run against **both** repos ⇒ 208 page fetches, 0 hard fetch failures) and **12 arXiv queries**, plus ad-hoc open-web searches. The complete list is in §4.

### 0.3 Prior-art exclusion, run **before** any claim was written

`KV_CACHE_GAP_MAP.md` (3,540 lines, 560 URLs) and `spec-decode-gap-map-2026-09-13.md` (1,191 lines) were loaded whole and every candidate's fingerprint keywords were matched against both.

A keyword that appears **only** inside the maps' own "Search log" section (e.g. `simple_kv_offload`, `rope_kvcache`, `vllm/v1/kv_offload/`) is the map's *grep command*, not a map row, and was not treated as prior discussion; a keyword that appears in a **row** (e.g. `cache_salt`, `evict_host`, `kv cache events`, `hfp3fs`/`hf3fs`, `fp4 kv`, `sm120`) was treated as prior discussion and the item moved to §2 or dropped. The map section list (`A1–A57`, `B1–B116`, `C.1–C.6`, `D.1–D.5`) was read for every collision.

### 0.4 Honest size statement

**Section 1 has 9 items.** Two are `HIGH`, six `MEDIUM`, one `LOW`. Two of the nine are unreachable on this rig. This is a small Section 1 on purpose: the source-first method is high-yield for *finding* self-admitted gaps but the same method shows that most of them are either (a) already named in a merged PR body by the engineer who wrote the restriction, or (b) on a hardware/model axis this rig does not have. **The single most important result of this run is negative about the method itself:** of the candidates that reached search and were resolved, **24 were removed because a public artefact already states them** (§2) against **9** that survived (§1) — the case that broke the most candidates was the merged feature PR whose own *Modifications* section lists the limitations it deliberately left in (see §2.1, §2.2). Claims below are therefore deliberately **NOT-FOUND** claims, never DOES-NOT-EXIST claims.

**Every statement marked `SELF-ADMITTED GAP` is a verbatim line retrieved from the installed tree on the rig, with `file:line`.** Nothing in §1 is an inference about what ought to exist.

---

## Section 1 — CONFIRMED NEVER-DISCUSSED

### 1.1 SGLang — the FA4 attention backend refuses to write the KV cache in place

- **SELF-ADMITTED GAP:** `kernels/ops/attention/flash_attention_v4_sm120.py:242`
  ```python
  if k is not None or v is not None:
      raise NotImplementedError("FA4 does not support updating KV cache in-place.")
  ```
  The identical guard exists in the non-SM12x FA4 wrapper at `kernels/ops/attention/flash_attention_v4.py:264`.
- **WHAT IT MEANS:** On the `fa4` attention backend, the attention kernel cannot be handed K/V and asked to also store them, so every layer must issue a **separate KV-store** (`token_to_kv_pool.set_kv_buffer(...)` at `flashattention_backend.py:1330` and `:1857`) on top of the attention kernel. The user-visible consequence is one extra kernel launch per layer per forward on the FA4 path — the same *class* of overhead the engine already documents for FP8 KV (SGLang issue #30815) but on a different cause. Axis: **attention / KV lifecycle write path**.
- **SEARCHED:**
  - GitHub `vllm-project/vllm` + `sgl-project/sglang` issues+PRs: `FA4 "does not support updating KV cache in-place"` → 0 + 0; `"updating KV cache in-place"` → 0 + 0; `"updating KV cache in-place" OR "FA4 does not support updating"` → 0 + 0; `FA4 KV cache update in place sm120` → 3 + 4 (all unrelated: `#51581`, `#52181`, `#20547`, `#23139`, `#30360`, `#30780`); `flash_attention_v4_sm120` → 0 + 1 (`sglang#32991`, the *feature* PR that added the kernels).
  - arXiv: `FA4 flash attention paged KV cache sm120`, `SGLang FA4 KV cache in-place update`.
  - Open web: `SGLang FA4 flash attention updating KV cache in-place NotImplementedError`, `SGLang sm120 attention backend paged KV support Blackwell`.
- **RESULT:** Nothing matching. The only SGLang artefact in the neighbourhood is `sgl-project/sglang#32991` *"feat(attention): add architecture-owned SM12x FA4 kernels"* (MERGED), which introduces the files and does not discuss the in-place restriction. The arXiv queries returned no paper tying an FA4-class kernel's inability to fuse the KV store to a serving-level cost.
- **CONFIDENCE:** `HIGH` — the string is a complete, distinctive sentence; four independent phrasings across both repos returned zero matching nodes, and the same sentence occurs twice in the tree, so it is deliberate API surface rather than a typo.
- **NOT-FOUND vs DOES-NOT-EXIST:** I searched four GitHub phrasings on both engines' issue+PR lists, two arXiv queries and two open-web queries and found **no matching discussion**. I am **not** claiming nobody has ever worked on fusing the FA4 KV store.
- **REACHABLE ON OUR RIG:** **yes** — `fa4` is a registered attention backend (`srt/layers/attention/attention_registry.py:241 @register_attention_backend("fa4")`), the SM12x specialisation is wired at `srt/layers/attention/flashattention_backend.py:286–293`, and our GPU is sm120.

---

### 1.2 SGLang — an MXFP8 KV cache is excluded from the prefix-valid commit path

- **SELF-ADMITTED GAP:** `srt/mem_cache/memory_pool.py:3601`
  ```python
  def set_kv_buffer_prefix_valid(self, *args, **kwargs):
      raise NotImplementedError(
          "prefix-valid commit is unsupported for MXFP8 KV cache "
          "(it does not carry the scale buffers)."
      )
  ```
- **WHAT IT MEANS:** With `--kv-cache-dtype mxfp8`, the pool cannot mark a freshly written span as prefix-valid, i.e. it cannot participate in the commit step that lets a partially-written prefix become visible to the radix/prefix cache. The engine's own parenthetical names the mechanism (the scale buffers are not carried through this entry point), so the consequence is that MXFP8 users lose the prefix-valid commit rather than getting a wrong answer. Axis: **prefix-reuse / quantization**.
- **SEARCHED:**
  - GitHub, both repos: `"prefix-valid commit"` → 0 + 0; `"prefix-valid" OR "set_kv_buffer_prefix_valid"` → 1 + 5 (vLLM `#52627` unrelated; SGLang `#25371`, `#30815`, `#31652`, `#35718`, `#38289` — none about prefix-validity, the nearest is `#35718` *"Support mxfp8 KV cache in PD transfer"*, which is transfer, not commit); `mxfp8 KV cache prefix valid commit` → 11 + 1 (no match).
  - arXiv: `quantized KV cache CPU offloading`, `post-capture KV cache backing virtual memory reservation`.
  - Open web: `SGLang "prefix-valid" set_kv_buffer_prefix_valid MXFP8`, `SGLang "post-capture" memory pool KV cache backing`.
- **RESULT:** Nothing matching. `set_kv_buffer_prefix_valid` appears in public only as part of larger pool refactors; the MXFP8 exclusion is not raised anywhere I could reach.
- **CONFIDENCE:** `MEDIUM` — the phrase is distinctive and the searches were clean, but three of the docstring's neighbours (`_create_buffers`, `set_kv_buffer`, `move_kv_cache`) are hot refactor targets (SGLang `#25371` RFC *"mem_cache pool / allocator restructure"* is open), so the entry point could be raised incidentally without the exclusion being named.
- **NOT-FOUND vs DOES-NOT-EXIST:** I searched three GitHub phrasings on both engines plus arXiv and open web and found **no matching discussion** of MXFP8 KV cache and prefix-valid commit. I am not claiming the restriction has never been considered internally.
- **REACHABLE ON OUR RIG:** **yes (partial)** — `mxfp8` is an accepted `--kv-cache-dtype` value with **no architecture gate** in `srt/mem_cache/kv_cache_dtype.py:58` (it resolves to `torch.float8_e4m3fn`), and sm120 is inside the SM100/SM120 support envelope SGLang documents for FP8-family KV. The restriction fires only when the prefix-valid commit is actually exercised.

---

### 1.3 SGLang — an MXFP8 KV cache cannot use the HND layout

- **SELF-ADMITTED GAP:** `srt/mem_cache/memory_pool.py:3295`
  ```python
  if self.use_hnd:
      # Buffers are NHD; the inherited HND move_kv_cache branch
      # would silently relocate wrong bytes.
      raise ValueError(
          "MXFP8 KV cache does not support SGLANG_USE_HND_KVCACHE."
      )
  ```
- **WHAT IT MEANS:** `SGLANG_USE_HND_KVCACHE` (head-major layout, chosen for kernel-locality reasons) is hard-incompatible with MXFP8 KV cache, so the two optimisations cannot be combined — a user must give one up. The in-source comment states the failure mode that the guard is preventing: silently relocating the wrong bytes. Axis: **KV layout / quantization**.
- **SEARCHED:**
  - GitHub, both repos: `MXFP8 "SGLANG_USE_HND_KVCACHE"` → 0 + 0; `"MXFP8" HND` → 0 + 1 (`sglang#34916`, unrelated logging PR); `SGLANG_USE_HND_KVCACHE` → 0 + 3 (`#28713` MiniMax-M3 mem-cache split, `#32452` HND CPU offload, `#36710` Apple-Silicon runner — none about MXFP8).
  - arXiv: `quantized KV cache CPU offloading`.
  - Open web: `SGLang MXFP8 HND KV cache layout`.
- **RESULT:** Nothing matching. `SGLANG_USE_HND_KVCACHE` is discussed in public only for MiniMax-M3, CPU offload and the MLX runner. The MXFP8 combination is not raised.
- **CONFIDENCE:** `MEDIUM` — the two identifiers are exact and the combined query is a clean zero, but both knobs are niche enough that the combination may simply never have been attempted publicly, which is a weaker signal than an actively-searched-for gap.
- **NOT-FOUND vs DOES-NOT-EXIST:** I searched three GitHub phrasings on both engines plus open web and found **no matching discussion** of MXFP8 KV cache with the HND layout. I am not claiming nobody has hit it.
- **REACHABLE ON OUR RIG:** **yes** — `SGLANG_USE_HND_KVCACHE` is a runtime environment switch and MXFP8 KV has no architecture gate on sm120; single GPU, no cluster needed.

---

### 1.4 SGLang — FP4 / NVFP4 KV-cache methods have no plain-attention dequant read path

- **SELF-ADMITTED GAP:** `srt/layers/quantization/fp4_kv_cache_quant_method.py:277`
  ```python
  """Dequantize one packed FP4 KV tensor for plain attention reads."""
  raise NotImplementedError(
      f"KV cache method {self.name!r} does not support plain KV dequant reads."
  )
  ```
  with the inverse guard on the unquantized side at `:322`
  ```python
  raise NotImplementedError(
      "Unquantized KV cache does not support FP4 KV dequantization."
  )
  ```
- **WHAT IT MEANS:** An FP4/NVFP4 KV cache can only be consumed by attention backends that were taught to read packed FP4 directly. Any backend or kernel that wants an ordinary dequantized K/V tensor — the generic path most backends share — cannot get one, so FP4 KV cache selection is coupled to a specific backend set rather than being a transparent dtype change. Axis: **quantization / attention-backend compatibility**.
- **SEARCHED:**
  - GitHub, both repos: `"plain KV dequant reads"` → 0 + 0; `"plain KV dequant reads" OR "does not support plain KV dequant"` → 0 + 0; `fp4 KV cache "dequantize_kv_tensor" OR "plain KV dequant"` → 0 + 0.
  - arXiv: `quantized KV cache CPU offloading`.
  - Open web: `SGLang FP4 KV cache plain dequant reads attention backend`, `SGLang NVFP4 KV cache dequantize plain reads`.
- **RESULT:** Nothing matching on any surface — four query formulations, all zero on both repos.
- **CONFIDENCE:** `MEDIUM` — zero on four formulations is strong for absence, but this is an `@abstractmethod`-adjacent declaration, so a reader could argue it is an interface statement rather than a defect; the `HIGH` label is withheld for that reason, not for search weakness.
- **NOT-FOUND vs DOES-NOT-EXIST:** I searched four phrasings across both engines' issue+PR lists plus arXiv and open web and found **no matching discussion**. I am not claiming the FP4 dequant path has never been considered.
- **REACHABLE ON OUR RIG:** **yes** — the tree's own gate is `"NVFP4 KV cache quantize requires SM100/SM120 or SM90 fallback GPU"` (`srt/layers/quantization/kvfp4_tensor.py:183`), which sm120 satisfies; `--kv-cache-dtype nvfp4|fp4_mx_block16` is accepted in `kv_cache_dtype.py`.

---

### 1.5 SGLang — CPU offloading is not implemented for the unified memory pool

- **SELF-ADMITTED GAP:** `srt/mem_cache/unified_memory_pool.py:632`
  ```python
  def get_cpu_copy(self, indices, mamba_indices=None):
      raise NotImplementedError(
          "CPU offloading is unsupported under the unified layout."
      )

  def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
      raise NotImplementedError(
          "CPU offloading is unsupported under the unified layout."
      )
  ```
- **WHAT IT MEANS:** Selecting the unified device pool removes the host-memory KV tier entirely: `get_cpu_copy` / `load_cpu_copy` are the two functions the HiCache/offload controller calls, and both are hard `NotImplementedError`. A user who wants the unified layout for its memory-accounting or hybrid-pool benefits cannot have a host KV tier at the same time. Axis: **offload**.
- **SEARCHED:**
  - GitHub, both repos: `"CPU offloading is unsupported under the unified"` → 0 + 0; `unified memory pool CPU offload unsupported` → 18 + 15 (all loose-keyword noise: vLLM `#2021`, `#11450`, `#11715`…; SGLang `#7704`, `#10062`, `#13055`, `#23602`, `#23882`, `#32657` — none about this restriction); `"CPU offloading is unsupported"` → 0 + 1 (`sglang#35888` *"Support CPU offload for mxfp8 KV cache"*, a different pool family and a **merged fix**, not a statement of this limit).
  - arXiv: `CPU offload KV cache page-major layout`, `quantized KV cache CPU offloading`.
  - Open web: `SGLang "unified memory pool" "CPU offloading is unsupported"`, `SGLang unified memory pool CPU offloading unsupported`.
- **RESULT:** Nothing matching. The only public SGLang work in this region is `sglang#29678` *"feat(mem_cache): unified memory pool for hybrid Mamba / SWA models"* (OPEN); its body was retrieved and it does **not** mention CPU offloading — it lists other limitations (PD disaggregation, speculative decoding, prefill CUDA-graph backend).
- **CONFIDENCE:** `HIGH` — exact-phrase zero on both repos, and the one plausible covering artefact (`#29678`) was fetched and read and does not state this limitation.
- **NOT-FOUND vs DOES-NOT-EXIST:** I searched three GitHub phrasings on both engines, two arXiv queries and two open-web queries, and read the body of the only candidate PR, and found **no matching discussion**. I am not claiming the unified pool was never intended to gain host offload.
- **REACHABLE ON OUR RIG:** **no** — the unified memory pool in 0.5.19 is built for hybrid Mamba / SWA models; Qwen3-4B is dense full attention, so the pool family is not selectable. Reported because the prompt requires out-of-reach items to be reported, and because the restriction is machine-independent.

---

### 1.6 SGLang — allocators that are not the paged allocator have no CPU copy path at all

- **SELF-ADMITTED GAP:** `srt/mem_cache/allocator/base.py:127`
  ```python
  def get_cpu_copy(self, indices, mamba_indices=None):
      # FIXME: reuse the get_cpu_copy after paged allocator is implemented
      raise NotImplementedError()

  def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
      # FIXME: reuse the load_cpu_copy after paged allocator is implemented
      raise NotImplementedError()
  ```
  (the sibling methods on the same class raise with a message: `alloc_extend is only for paged allocator`, `alloc_decode is only for paged allocator`, `base.py:135–139`.)
- **WHAT IT MEANS:** The base allocator — inherited by every non-paged allocator family — raises a **bare** `NotImplementedError()` with no message for both directions of the host KV copy. The user-visible consequence is that any offload attempt on such an allocator fails with a message-less traceback rather than a diagnosable error, and the FIXME states the intended fix is gated on "after paged allocator is implemented", i.e. it is deferred, not designed away. Axis: **offload / block-pool management**.
- **SEARCHED:**
  - GitHub, both repos: `"after paged allocator is implemented"` → 0 + 0; `"alloc_extend is only for paged allocator" OR "after paged allocator is implemented"` → 0 + 0; `"paged allocator" get_cpu_copy` → 0 + 6 (SGLang `#7409`, `#9990`, `#25090`, `#37146`, `#38559`, `#38840` — all MLA/PD/DCP/metrics work, none about the FIXME).
  - arXiv: `CPU offload KV cache page-major layout`.
  - Open web: `SGLang allocator get_cpu_copy paged allocator FIXME`.
- **RESULT:** Nothing matching. No public artefact quotes either FIXME string.
- **CONFIDENCE:** `MEDIUM` — both `file:line` and strings are verbatim and the exact-phrase searches are clean, but the FIXME's natural home is the allocator-restructure RFC `sgl-project/sglang#25371`, whose **body could not be retrieved** (the page returns no `comment-body` block for this artefact — see §3.1), so a residual possibility of overlap remains and is disclosed rather than guessed away.
- **NOT-FOUND vs DOES-NOT-EXIST:** I searched three GitHub phrasings on both engines plus open web and found **no matching discussion**, with the `#25371` body unretrieved and disclosed. I am not claiming the deferral was never agreed.
- **REACHABLE ON OUR RIG:** **partial** — the base allocator is reached by non-paged allocator families (SWA / pure-SWA variants, `srt/mem_cache/allocator/swa.py`); those need an SWA-class or hybrid model rather than a dense full-attention one, so a *configuration* on this rig can reach the class but Qwen3-4B alone does not select it.

---

### 1.7 vLLM — the extract-hidden-states proposer refuses a quantized KV cache outright

- **SELF-ADMITTED GAP:** `model_executor/models/extract_hidden_states.py:196`
  ```python
  if is_quantized_kv_cache(kv_cache_dtype):
      raise NotImplementedError("Quantized KV cache not supported")
  ```
- **WHAT IT MEANS:** The `extract_hidden_states` speculative path — the supported way to harvest auxiliary hidden states with vLLM — cannot be combined with **any** quantized KV cache. `--kv-cache-dtype fp8` (or any other quantized value) plus this proposer is an immediate hard failure, so the two memory-saving choices are mutually exclusive for exactly the workflow that uses this proposer to produce EAGLE3-style drafter data. Axis: **quantization / KV lifecycle**.
- **SEARCHED:**
  - GitHub, both repos: `"Quantized KV cache not supported"` → 0 + 0; `"Quantized KV cache not supported" OR extract_hidden_states` → 25 + 0 (the extract-hidden-states area is *very* active — `#46399`, `#46426`, `#46788`, `#46973`, `#48124`, `#49301`, `#49562`, `#49811`, `#50815`, `#50894`, `#51016`, `#51328` — but **every** hit is about layer selection, MRv2 support, staging memory, TP page-size scaling, per-group `slot_mapping` or fixed-K proposers; **none** mentions a quantized KV cache).
  - arXiv: `quantized KV cache CPU offloading`.
  - Open web: `vLLM extract_hidden_states quantized KV cache not supported`.
- **RESULT:** Only unrelated hits — the proposer's *other* defects are extensively filed, which makes the absence of the quantization one conspicuous rather than accidental.
- **CONFIDENCE:** `MEDIUM` — a clean exact-phrase zero against a **crowded** neighbour set is good evidence, but the crowdedness also means a matching report could exist under wording I did not guess.
- **NOT-FOUND vs DOES-NOT-EXIST:** I searched two GitHub phrasings on both engines, arXiv, and the open web, and found **no matching discussion** of the quantized-KV restriction in this proposer. I am not claiming it has never been reported.
- **REACHABLE ON OUR RIG:** **yes** — `extract_hidden_states` is a first-class speculative method (`config/speculative.py:1126 elif self.method == "extract_hidden_states": self.model = "extract_hidden_states"`), it is a single-GPU path, and the rig already has an EAGLE3 Qwen3-4B drafter, which is exactly the workflow this proposer serves.

---

### 1.8 SGLang — the HiCache host tier implements only the "backup-only" eviction policy

- **SELF-ADMITTED GAP:** `srt/managers/cache_controller.py:960`
  ```python
  def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
      if not backup_only:
          raise ValueError("Other eviction policies are not supported yet.")
  ```
- **WHAT IT MEANS:** The host-tier eviction entry point carries a policy selector, and every value of that selector except the default is unimplemented. SGLang ships a *radix-tree* eviction-policy surface (`--radix-eviction-policy`, lru/lfu/slru/priority) but the **host KV tier** has exactly one behaviour, so the tier cannot be tuned the way the tree can. Axis: **eviction / offload**.
- **SEARCHED:**
  - GitHub, both repos: `"Other eviction policies are not supported"` → 0 + 0; `"evict_host" OR "Other eviction policies"` → 1 + 25 (the 25 are the active HiCache host-eviction work: `#27424`, `#27562`, `#27631`, `#28027`, `#28047`, `#28429`, `#28430`, `#28468`, `#28507`, `#28508`, `#28614`, …); `HiCache host eviction policy backup_only` → 0 + 0; `"backup_only"` → 2 + 7 (SGLang `#8404`, `#22542`, `#26886`, `#33656`, `#33862`, `#37122`, `#38452` — host-mirror reclamation and L3-stub work, none about the policy selector).
  - arXiv: `prefix cache eviction policy host tier HiCache`.
  - Open web: `vLLM "Other eviction policies" host offload backup_only`, `SGLang HiCache host eviction policy backup_only`.
- **RESULT:** Nothing matching the unimplemented branch. **Disclosed counter-evidence:** SGLang's host-eviction code is under heavy public repair (`#27562`, `#28468`, `#33862`, `#38452`), and `backup_only` is mentioned incidentally in seven public artefacts. The *specific* statement "no eviction policy other than backup-only exists" is not raised in any of them.
- **CONFIDENCE:** `LOW` — the verbatim quote and the two exact-phrase zeros are solid, but the surrounding axis is crowded enough that the honest reading is "this specific branch was not raised", not "this area is untouched". A reader should treat this item as the weakest of the nine.
- **NOT-FOUND vs DOES-NOT-EXIST:** I searched four GitHub phrasings on both engines plus arXiv and open web and found **no matching discussion of the unimplemented non-backup-only branch**, while finding substantial adjacent host-eviction work. I am not claiming host-eviction policy is unstudied.
- **REACHABLE ON OUR RIG:** **yes** — a single-node HiCache deployment with a local host tier exercises `evict_host` directly (host memory is not a cluster resource; 208 cores and ample RAM are available).

---

### 1.9 SGLang — an MXFP8 KV cache cannot carry the DCP KV mask

- **SELF-ADMITTED GAP:** `srt/mem_cache/memory_pool.py:3405`
  ```python
  raise NotImplementedError("MXFP8 KV cache does not support DCP KV masks.")
  ```
- **WHAT IT MEANS:** The third MXFP8 restriction in the same class (alongside §1.2 and §1.3): the decode-context-parallel masking path — which tells a rank which KV slots it owns — is unavailable for MXFP8 KV cache, so the two cannot be combined. Axis: **quantization / attention**. This one sits in the same MXFP8 cluster as §1.2–§1.3 and is reported separately only because its blocker is a different mechanism (parallelism bookkeeping, not scale buffers or layout).
- **SEARCHED:**
  - GitHub, both repos: `MXFP8 "dcp_kv_mask" OR "DCP KV masks"` → 0 + 11 (SGLang `#14194`, `#18167`, `#25090`, `#29140`, `#29678`, `#31568`, `#32796`, `#34355`, `#34432` — all generic DCP bring-up, none mentioning MXFP8).
  - arXiv: `quantized KV cache CPU offloading`.
  - Open web: `SGLang MXFP8 DCP KV mask`.
- **RESULT:** Nothing matching — eleven DCP artefacts, zero MXFP8 context.
- **CONFIDENCE:** `MEDIUM` — clean combined zero, but the DCP axis is broad enough that this may simply be an untested combination.
- **NOT-FOUND vs DOES-NOT-EXIST:** I searched one combined exact-identifier GitHub query on both engines plus open web and found **no matching discussion**. I am not claiming it was never hit.
- **REACHABLE ON OUR RIG:** **no** — DCP (decode context parallelism) requires more than one GPU, and this rig has exactly one, no NVLink and no cluster.

---

## Section 2 — ADMITTED BUT ALREADY DISCUSSED (the excluded candidates)

Each row: verbatim source line → the public artefact that already covers it.

### 2.1 SGLang page-major KV layout — **the merged feature PR names both limitations itself**

- `srt/mem_cache/memory_pool.py:3241` — `raise NotImplementedError("CPU offloading is unsupported under the page-major layout (TODO: split token ids into page/slot for the 4-D index).")`
- `srt/mem_cache/kv_cache_configurator.py:1232` — `assert (not enable_page_major), "page-major KV layout is not supported with fp4 KV cache"`
- `srt/mem_cache/memory_pool.py:3234` — `raise NotImplementedError("page-major layout has no per-layer contiguous regions; KV transfer / disaggregation is unsupported (TODO: expose the single _raw buffer with a page-aware transfer scheme).")`
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/pull/29533 (MERGED) — *"feat(mem_cache): page-major (layer-major within a page) KV/state layout"*. The body was retrieved and states both, verbatim: *"Layout-incompatible inherited methods (contiguous-buf-infos, CPU offload, prefix-commit) raise NotImplementedError instead of silently mis-indexing the 4-D strided views."* and *"`--enable-page-major-kv-layout` is not yet supported with fp4 KV cache (asserted) or the speculative-decode target-verify path."* The same PR also carries `TODO(ch-wan)` for the GDN prefill `.contiguous()` gather/scatter under the envelope.
- **Why it is here and not in §1:** this was my strongest candidate until the PR body was read. It is the cleanest illustration of why the source-first method must end in a discussion check.

### 2.2 SGLang HND KV layout × CPU offload
- `srt/mem_cache/memory_pool.py:2283` — `assert not self.use_hnd, ("CPU KV offload indexes by slot (NHD); HND KV cache (SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet.")`
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/pull/32452 (OPEN) — *"[MHA] Support HND KV cache layout in CPU offload"*. Body retrieved: *"This PR adds CPU offloading support for HND KV cache layout while preserving existing NHD behavior"* and *"Explicitly reject the unsupported combination of HND layout and quantized KV cache."*

### 2.3 vLLM SimpleCPUOffload — stale `num_stored_blocks` under CPU eviction
- `v1/simple_kv_offload/manager.py:588` — `# FIXME (yifan): handle CPU cache eviction, where num_stored_blocks can be stale and omit evicted blocks in the middle of the request.`
- **ALREADY DISCUSSED AT:** https://github.com/vllm-project/vllm/pull/47235 — *"[Bugfix][KVOffload] Restore evicted CPU blocks and fix cursor advancement"*. Body retrieved verbatim: *"Resolves the FIXME: num_stored_blocks can be stale and omit evicted blocks in the middle of the request"*. (A companion control-branch PR, https://github.com/vllm-project/vllm/pull/47234, documents the behaviour.) Note the FIXME is still present in the installed 0.29.0 tree, so the installed snapshot predates the fix.

### 2.4 vLLM FA4 CuTe paged KV on SM12.0 — the arch the rig actually has
- `vllm_flash_attn/cute/interface.py:1186` and `third_party/tml_fa4/interface.py:1242` — `assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"`
- **ALREADY DISCUSSED AT:** https://github.com/vllm-project/vllm/issues/51405 (OPEN) — *"[Bug]: Inkling (InklingForConditionalGeneration) fails on SM120 — tml_fa4 attention lacks paged-KV support"*; plus https://github.com/vllm-project/vllm/pull/51560 (OPEN) *"Fail fast on Inkling's unsupported GPU architectures"* and https://github.com/vllm-project/vllm/pull/49681 (CLOSED) *"Inkling sm_12x (GB10) boot + memory-safety fixes"*. Also overlaps map row **B55**.

### 2.5 vLLM — `--kv-cache-dtype` cannot be inferred from the checkpoint (mgoin TODO)
- `model_executor/layers/attention/attention.py:192` — `# TODO (mgoin): kv cache dtype should be specified in the FP8 checkpoint config and become the "auto" behavior`
- **ALREADY DISCUSSED AT:** https://github.com/vllm-project/vllm/pull/55134 — *"[Bugfix][Model] DeepseekV4: resolve kv_cache_dtype=\"auto\" to fp8_ds_mla"*; https://github.com/sgl-project/sglang/pull/35455 — *"[Quant] Load compressed-tensors kv_cache_scheme scales"*; overlap with map row **B37** and https://github.com/vllm-project/vllm/issues/51751.

### 2.6 vLLM — KV cache sharing is not supported for MRv2
- `config/cache.py:229` — `NOTE: KV cache sharing is not supported for MRv2 (v2 model runner).`
- **ALREADY DISCUSSED AT:** https://github.com/vllm-project/vllm/pull/35045 — *"[Model Runner V2] Support sharing kv cache layers"* (MERGED); also `"mrv2"` appears in `spec-decode-gap-map-2026-09-13.md`.

### 2.7 vLLM — all `kv_cache_groups` are assumed to have the same number of blocks
- `v1/core/kv_cache_manager.py:46` — `kv_cache_groups have the same number of blocks, which is true for now but will be broken if we want to give different block_size to different kv_cache_groups in the future.`
- **ALREADY DISCUSSED AT:** https://github.com/vllm-project/vllm/pull/34373 *"[Feature] Enable uniform KV cache allocation for multi-group HMA models"*; https://github.com/vllm-project/vllm/pull/39031 *"[Core] Per-group BlockPool for hybrid Mamba/attention models"*; https://github.com/vllm-project/vllm/issues/42966. Overlaps map rows **B1**, **B2**, **B80**.

### 2.8 vLLM — prefix-cache stats not conditional on `log_stats`
- `v1/core/kv_cache_manager.py:148` — `# FIXME: make prefix cache stats conditional on log_stats. We still need this comment because when the log stats is enabled there are still potential configs we could expose in the future.`
- **ALREADY DISCUSSED AT:** vLLM commit `d9737ca` *"[V1][Misc] stop update prefix cache stats when logs_stats is disabled"* (https://github.com/vllm-project/vllm/commit/d9737ca1c66662c7ed4e3047df74452d323c240e); plus https://github.com/vllm-project/vllm/pull/48860 and https://github.com/vllm-project/vllm/pull/48668.

### 2.9 vLLM — `SlidingWindowManager` cannot serve fine-grained (partial) hits
- `v1/core/single_type_kv_cache_manager.py:930` — `"SlidingWindowManager does not support fine-grained (partial) cache hits"`
- **ALREADY DISCUSSED AT:** https://github.com/vllm-project/vllm/issues/53786 — *"[Feature][KV cache] Support fine-grained prefix hits for sliding-window groups"*; https://github.com/vllm-project/vllm/pull/54319 and https://github.com/vllm-project/vllm/pull/54397 *"[Prefix Cache] Support fine-grained SWA hits"*; https://github.com/vllm-project/vllm/pull/54661. Overlaps map row **B13**.

### 2.10 vLLM — `runner_kv_caches` and multiple attention layers per decoder block
- `v1/worker/utils.py:610` — `# TODO - analyze where runner_kv_caches is used and the right way to ensure it properly reflects multiple attention layers in the same decoder block.`
- **ALREADY DISCUSSED AT:** https://github.com/vllm-project/vllm/pull/21088 — *"[v1] Add Whisper model support (encoder-decoder)"* (MERGED; the same decoder block hosting both cross- and self-attention is the `# One typical case is encoder-decoder model, e.g., bart.` comment two lines above the TODO). Also https://github.com/vllm-project/vllm/pull/20900 and https://github.com/vllm-project/vllm/pull/14098.

### 2.11 vLLM — `store_threshold` unavailable on the tiering spec
- `v1/kv_offload/tiering/spec.py:290` — `raise ValueError("store_threshold is not supported for TieringOffloadingSpec")`
- **ALREADY DISCUSSED AT:** the `store_threshold` axis has a dense public trail: https://github.com/vllm-project/vllm/pull/40020 (multi-tier framework), https://github.com/vllm-project/vllm/pull/35342, https://github.com/vllm-project/vllm/pull/41727, https://github.com/vllm-project/vllm/pull/52022, https://github.com/vllm-project/vllm/pull/52227, https://github.com/vllm-project/vllm/pull/54759, https://github.com/vllm-project/vllm/pull/51787. Overlaps map rows **A52**, **A53**.

### 2.12 vLLM — fused FP8 output combined with paged KV is refused
- `vllm_flash_attn/cute/interface.py:564` — `assert page_table is None, "fused FP8 output + paged KV not supported yet"`
- **ALREADY DISCUSSED AT:** https://github.com/vllm-project/vllm/issues/29920 (OPEN) — *"[Feature]: Add support for fused fp8 output to FlashAttention 3"*; https://github.com/vllm-project/vllm/pull/43050 (MERGED) — *"feat: MLA prefill enable FA4 fp8 output"*; https://github.com/vllm-project/vllm/issues/33097. The exact sentence is unraised (0 hits) but the capability request is public and open, so this is not unoccupied ground. Related: `interface.py:1091` `assert qv is None, "fused FP8 output + MLA (qv) not supported yet"`.

### 2.13 SGLang — post-capture KV backing is refused for a quantized KV cache
- `srt/mem_cache/memory_pool.py:1986` — `if self.post_capture_active: raise NotImplementedError("Post-capture KV backing is not supported for quantized KV cache.")`
- **ALREADY DISCUSSED AT:** the post-capture KV-sizing feature has an open, active public trail: https://github.com/sgl-project/sglang/pull/30157 *"Size KV pool after CUDA graph capture (opt-in)"*, https://github.com/sgl-project/sglang/pull/33427, https://github.com/sgl-project/sglang/pull/33445, https://github.com/sgl-project/sglang/pull/33852, https://github.com/sgl-project/sglang/pull/34053. Additional reason for exclusion: on the dense MHA path the configurator already gates the feature off for quantized pools (`kv_cache_configurator.py:1771` passes `post_capture_active=self.post_capture_kv_active and quant_method is None`), so this error is not reachable from Qwen3-4B.

### 2.14 SGLang — HiCache draft pools require `UnifiedRadixCache`
- `srt/mem_cache/kv_cache_builder.py:102` — `raise NotImplementedError("HiCache draft pools require UnifiedRadixCache.")`
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/pull/30393 (MERGED) — *"[HiCache] Support packed and sidecar draft caches for MTP/EAGLE/DSpark"*; plus https://github.com/sgl-project/sglang/pull/35875, https://github.com/sgl-project/sglang/pull/37424, https://github.com/sgl-project/sglang/pull/35221, https://github.com/sgl-project/sglang/pull/35789.

### 2.15 SGLang — radix cache and deterministic inference
- `srt/arg_groups/attention_hook.py:598` — `f"Currently radix cache is not compatible with {attention_backend} attention backend for deterministic inference. It will be supported in the future."`
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/issues/10278 and map row **B114** (which quotes the same checklist, including the unticked `FlashInfer Support` and `Making Prefill with Radix Cache has the same output as Prefill without Radix cache` items).

### 2.16 SGLang — CUDA graph is not disabled when HiCache loading is triggered
- `srt/managers/scheduler.py:3781` — `# todo (zhiqiang): disable cuda graph execution if hicache loading triggered`
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/issues/38300 *"TP2 hang with HiCache, breakable prefill CUDA graphs, and FlashInfer MNNVL on B300"*; https://github.com/sgl-project/sglang/pull/38354 *"[cuda-graph] Do not let an explicit prefill backend override hard incompatibilities"*; https://github.com/sgl-project/sglang/issues/38448; https://github.com/sgl-project/sglang/pull/38463; and the out-of-tree commit *"Do not disable piecewise CUDA graph for hierarchical cache"* (DarkraiHL/sglang `126d60c`) which shows the opposite decision is actively argued.

### 2.17 SGLang — the C++ radix tree ignores `cache_salt` and emits no KV events
- `srt/mem_cache/radix_cache_cpp.py:43` — `"cache_salt is not supported by the experimental C++ radix tree"`
- `srt/mem_cache/radix_cache_cpp.py:57` — `"HiRadixCache does not support kv cache events yet"`
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/pull/10317 *"Refactors radix cache for extra key support"*; https://github.com/sgl-project/sglang/issues/24568 *"KV-cache event block hashes ignore RadixKey.extra_key/cache_salt"*; https://github.com/sgl-project/sglang/pull/27735; https://github.com/sgl-project/sglang/pull/30827; and for events: https://github.com/sgl-project/sglang/issues/29709 (RFC *"KV Cache Events for HiCache L3 Storage Backends"*), https://github.com/sgl-project/sglang/pull/29923, https://github.com/sgl-project/sglang/pull/38486, https://github.com/sgl-project/sglang/pull/11178, https://github.com/sgl-project/sglang/pull/35164. Also map rows **A13**, **A14**, **B4**, **B17**.

### 2.18 SGLang — Rust TreeCore gaps (session radix cache, custom components, C128)
- `srt/mem_cache/rust_tree_core/adapter.py:308` — `# TODO(Jialin): Port session-reference-aware TreeCore support from #29173.` / `raise ValueError("--enable-session-radix-cache is not supported by the Rust TreeCore")`; `:314` — `# TODO(Jialin): Port custom component registration from #25754 and C128 support from #33676.`
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/pull/32710 *"[Radix Cache] Add Rust TreeCore backend with shared parity tests"*, https://github.com/sgl-project/sglang/pull/29901, https://github.com/sgl-project/sglang/pull/30145, https://github.com/sgl-project/sglang/pull/29074, https://github.com/sgl-project/sglang/pull/37290, https://github.com/sgl-project/sglang/issues/20415; and the session-radix side https://github.com/sgl-project/sglang/pull/29173, https://github.com/sgl-project/sglang/pull/27058, https://github.com/sgl-project/sglang/pull/29436, https://github.com/sgl-project/sglang/pull/29099. Map rows **A21**, **B77**.

### 2.19 SGLang — "save kv cache" unsupported for the fallback rotary embeddings
- `srt/layers/rotary_embedding/base.py:422` — `assert (fused_set_kv_buffer_arg is None), "save kv cache is not supported for fallback_rotary_embedding."`; `srt/layers/rotary_embedding/mrope.py:184` — same for `MRotaryEmbedding`.
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/pull/13078 — *"Fix fused_set_kv_buffer_arg is not supported for native implementation"*; https://github.com/sgl-project/sglang/pull/11468 — *"Remove fused_set_kv_buffer_arg assert for torch.compile compatibility"*; also the whole fused-rope+KV-write series https://github.com/sgl-project/sglang/pull/9014, https://github.com/sgl-project/sglang/pull/9077, https://github.com/sgl-project/sglang/pull/10749, https://github.com/sgl-project/sglang/pull/10945.

### 2.20 SGLang — `--prefill-only-disable-kv-cache` × prefill-CP / HiSparse / no-op pool
- `srt/arg_groups/kv_cache_hook.py:455` — `"--prefill-only-disable-kv-cache is incompatible with --enable-prefill-cp: the prefill-CP path stages K/V through the paged cache, which the no-op pool does not support."`; `:449` for `--attn-cp-size > 1`; `:464` for `--enable-hisparse`.
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/pull/23675 (the feature), https://github.com/sgl-project/sglang/pull/26593, https://github.com/sgl-project/sglang/pull/27159, https://github.com/sgl-project/sglang/pull/28580, https://github.com/sgl-project/sglang/pull/27343. The backend-dependent half of this validation is itself public.

### 2.21 SGLang — HiCache `buffer_only` host-memory mode restrictions
- `srt/arg_groups/hicache_hook.py:198` — `"--hicache-host-memory-mode buffer_only does not support --hicache-write-policy write_back; use write_through or write_through_selective."`; `:204` — `"--hicache-host-memory-mode buffer_only is not supported on decode instances: …"`; `srt/mem_cache/buffer_mode/pipeline.py:173` — same.
- **ALREADY DISCUSSED AT:** the mode has six or more open/merged public artefacts: https://github.com/sgl-project/sglang/pull/20535 *"[HiCache] Add L2 prefetch-buffer-only memory mode"*, https://github.com/sgl-project/sglang/pull/36341, https://github.com/sgl-project/sglang/pull/36345, https://github.com/sgl-project/sglang/pull/37424, https://github.com/sgl-project/sglang/pull/37464, https://github.com/sgl-project/sglang/pull/16909.

### 2.22 SGLang — `PureSWATokenToKVPoolAllocator` requires `page_size == 1`
- `srt/mem_cache/allocator/swa.py:513` — `raise NotImplementedError("PureSWATokenToKVPoolAllocator does not support page_size > 1.")` (three times: `alloc_extend`, `alloc_decode`, `alloc_extend_swa_tail`)
- **ALREADY DISCUSSED AT:** the allocator/pool layout is the subject of an open RFC and a long stack: https://github.com/sgl-project/sglang/issues/25371 *"[RFC][Refactor] mem_cache pool / allocator restructure"*, https://github.com/sgl-project/sglang/pull/28494, https://github.com/sgl-project/sglang/pull/29678, https://github.com/sgl-project/sglang/pull/32709, https://github.com/sgl-project/sglang/pull/36637; plus the sibling TODO `# TODO: support page_size > 1 for swa spec` at `srt/layers/attention/flashattention_backend.py:3344` and `srt/layers/attention/xpu_backend.py:1292`.

### 2.23 SGLang — `fa_skip_kv_cache` (embedding mode) refuses FP8 KV descaling
- `srt/layers/attention/flashattention_backend.py:1515` — `assert k_descale is None and v_descale is None, ("fa_skip_kv_cache uses raw K/V tensors, FP8 KV cache descaling is not supported in this mode")`
- **ALREADY DISCUSSED AT:** https://github.com/sgl-project/sglang/pull/21971 *"perf: skip KV cache in FA backend for embedding mode"*, https://github.com/sgl-project/sglang/pull/24097 *"Restrict fa_skip_kv_cache to non-MLA backends"*, https://github.com/sgl-project/sglang/pull/28580 (which gates the fast path behind the public flag), https://github.com/sgl-project/sglang/pull/27343.

### 2.24 vLLM — `prefix caching + batch invariance is currently not supported for FLASHINFER and TRITON_MLA`
- `model_executor/layers/attention/attention.py:363`
- **ALREADY DISCUSSED AT:** map row **B108** (*"Prefix caching is still not supported in vLLM's batch-invariant mode"*, citing vLLM #27433 and #46592) and map row **B110**.

### 2.25 Hardware/platform-scoped items excluded by the rig spec
| source line | why excluded |
|---|---|
| `vllm_flash_attn/cute/interface.py:675` — `# The SM90 fp8-KV-dequant producer is TMA-only (no cp.async fallback) … assert page_size == tile_n` | SM90-only path; rig is sm120 |
| `vllm_flash_attn/cute/interface.py:1042`, `third_party/tml_fa4/interface.py:1116` — `assert page_table is None, "paged KV not supported on SM 8.0"` | SM80-only path; same family as §2.4, which is discussed |
| `vllm_flash_attn/cute/flash_fwd_sm100.py:740`, `third_party/tml_fa4/flash_fwd_sm100.py:1275` — `raise NotImplementedError("Block sparsity + paged KV not supported on SM100")` | SM100; rig is sm120 |
| `v1/attention/backends/triton_attn.py:489`, `:497`, `:724` — FP8 KV needs SM89+; bf16 KV needs SM80+; quantized KV unsupported for encoder attention | rig is sm120 (SM89+ satisfied) and Qwen3-4B is decoder-only. Encoder-only prefix caching is separately map row **B89** |
| `v1/attention/ops/prefix_prefill.py:831`, `v1/attention/ops/chunked_prefill_paged_decode.py:362` | verified unreachable on CUDA: `prefix_prefill.context_attention_fwd` is imported only by `chunked_prefill_paged_decode.py:17`, which is imported only by `v1/attention/backends/rocm_attn.py:30` |
| `engine/arg_utils.py:2826` — `"Prefix caching is not supported for RISC-V CPUs"` | RISC-V; see https://github.com/vllm-project/vllm/pull/22112 |
| `srt/mem_cache/memory_pool_host.py:435`, `:503`; `srt/mem_cache/pool_host/mla.py:395`, `:413`; `srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:453`, `:1793`; `srt/mem_cache/kv_cache_builder.py:272`, `:277` | DeepSeek-V4 / MLA / SWA-compress model families outside Qwen3-4B |
| `srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py:57` — `# Todo: Implementing data eviction logic after HiCache supports prefix information pass-through` | hf3fs is a distributed storage service; map row **D78** rules the tier out |
| `srt/mem_cache/storage/flexkv/flexkv_connector.py:151`; `srt/mem_cache/memory_pool.py:2272` | disaggregated / distributed KV store |
| `srt/layers/attention/flashmla_backend.py:445`, `flashinfer_backend.py:1335`, `trtllm_mha_backend.py:302`, `srt/layers/attention/dsv4/*`, `srt/layers/quantization/kvfp4_tensor.py:183` (SM90 branch) | MLA / cross-attention / DeepSeek-V4 / SM90 |
| `v1/attention/backends/mla/*`, `models/qwen4_exp/*`, `models/deepseek_v4/*`, `models/kimi_k3/*`, `third_party/fmha_sm100/*` | MLA / MoE / non-Qwen3 model families |

---

## Section 3 — INCONCLUSIVE (search blocked; absence **not** claimed)

### 3.1 SGLang `sgl-project/sglang` issue/PR **#25371** body unretrievable
- **BLOCKER:** the page for https://github.com/sgl-project/sglang/issues/25371 (RFC *"[Refactor] mem_cache pool / allocator restructure"*) returns **no `comment-body` / `markdown-body` block** to my extractor, and `api.github.com` is rate-limited to zero in this environment. I therefore could not read the RFC that is the most likely origin of the `allocator/base.py` FIXME in §1.6.
- **CONSEQUENCE:** §1.6 is reported at `MEDIUM` confidence with this overlap explicitly disclosed. Absence is claimed only for the two verbatim FIXME strings, which returned zero on every surface.

### 3.2 Cross-repository GitHub search is unavailable
- **BLOCKER:** global `https://github.com/search?q=…&type=issues` returned **HTTP 429** on every attempt (three attempts, ~0.7 s each, before and after the repo-scoped searches). Repo-scoped search (`/<owner>/<repo>/issues?q=`) works, so all GitHub evidence is confined to `vllm-project/vllm` and `sgl-project/sglang`.
- **CONSEQUENCE:** I did **not** search LMCache, TensorRT-LLM, llama.cpp, NVIDIA Dynamo, Hugging Face forums, Reddit, X, or any blog for these candidates. Items in §1 could in principle be discussed in those venues; I make no claim about them. Where an open-web search happened to surface a non-vLLM/SGLang artefact it is cited (§2), but that was opportunistic, not systematic.

### 3.3 The arXiv API is unusable; arXiv coverage rests on a single query phrasing per topic
- **BLOCKER:** `http://export.arxiv.org/api/query?…` returned **HTTP 429 / `Rate exceeded.`** on every call (four attempts across the session). All arXiv evidence therefore comes from the `arxiv.org/search/?searchtype=all` HTML UI.
- **MITIGATION:** the HTML UI was re-run with a corrected result-count parser, and **11 of the 12 queries returned an explicit `total=0`**; the twelfth (`quantized KV cache CPU offloading`) returned 6 results whose titles do not touch any §1 item. One earlier pass had one query fail outright (curl exhausted its retries); that query succeeded on the re-run.
- **RESIDUAL LIMITATION (why this is still listed as a blocker):** each topic was probed with **one** query string, and arXiv's `all` field is a metadata/full-text search, not a substitute for a venue-complete literature sweep. A paper that describes one of the §1 behaviours under different vocabulary would not appear. So the arXiv result supporting §1 is properly stated as *"this exact query returned zero results on arXiv"*, not *"no paper covers this"*.

### 3.4 `web_fetch` tool unreliability
- **BLOCKER:** the `web_fetch` tool timed out on the first `github.com/search` attempt. `curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "Mozilla/5.0 … Chrome/126.0 Safari/537.36"` worked, and all page reads reported here used it.
- **CONSEQUENCE:** none for the evidence that *was* collected; noted because it bounds what "searched the web" means in this document.

### 3.5 Items searched but **not** classifiable into §1 — absence not claimed
- `srt/kv_canary/radix_cache_walker.py:39`, `:140` — `raise NotImplementedError(f"walk_radix_cache_for_canary does not support {cache_type.__name__}")`. The exact-phrase search returned 0, but `kv_canary` itself has **25 SGLang hits** (`#31709`, `#32829`, `#33520`, `#33656`, `#38596`, …) and the walker is actively extended, so I could not separate "unsupported tree types" from "not yet extended". **Not claimed.**
- `srt/mem_cache/sparsity/core/sparse_coordinator.py:41` — `# TODO: Add more trackers for hierarchical KVCache management`. The phrase search returned 0, but `sparse_coordinator` has **16 SGLang hits** and the sparse+HiCache framework is under active construction (`#16086`, `#16984`, `#22865`). **Not claimed.**
- `srt/layers/attention/flashattention_backend.py:1517` (FP8 KV descaling × `fa_skip_kv_cache`) — the exact assertion is unraised, but the mode itself is public (§2.23) and it targets embedding models rather than Qwen3-4B. **Not claimed as unoccupied.**
- `vllm_flash_attn/cute/flash_fwd_sm100.py:227`, `vllm_flash_attn/cute/flash_fwd_sm90.py:77` and the identical `third_party/tml_fa4/flash_fwd_sm100.py:266`, `third_party/tml_fa4/flash_fwd_sm90.py:66` — `"Paged KV does not support irregular head dim"`. Exact-phrase search returned **0 on both repos**, but Qwen3-4B's head dimension is tile-regular, so the restriction is not reachable on this rig. Absence not claimed.
- `v1/core/single_type_kv_cache_manager.py:945` — `# TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to optimize the time complexity from O(max_num_blocks) to …`. A performance TODO on the sliding-window hit scan; the surrounding capability is public (§2.9) and SWA models are outside Qwen3-4B. **Not claimed.**

---

## 4. Complete search log

### 4.1 Source extraction (rig, over SSH)

```
scp probes/p17-open-cells/scan_engine_todos.py  → /tmp/scan_engine_todos.py   (rig)
/root/ccfa_venv/bin/python /tmp/scan_engine_todos.py --selftest
/root/ccfa_venv/bin/python /tmp/scan_engine_todos.py --root /root/ccfa_venv/lib/python3.12/site-packages/vllm            --tag vllm   --out /tmp/vllm_todos.md
/root/ccfa_venv/bin/python /tmp/scan_engine_todos.py --root /root/autodl-tmp/venvs/sglang/lib/python3.12/site-packages/sglang --tag sglang --out /tmp/sglang_todos.md
/root/ccfa_venv/bin/python /tmp/kv_gap_extract2.py  --root <vllm>   --tag vllm   --out /tmp/vllm_kv2.json     # Tier A 58 / Tier B 120
/root/ccfa_venv/bin/python /tmp/kv_gap_extract2.py  --root <sglang> --tag sglang --out /tmp/sglang_kv2.json   # Tier A 72 / Tier B  84
/root/ccfa_venv/bin/python /tmp/ctx.py < pairs1.txt   # 20 vLLM file:line sites, ±10 lines
/root/ccfa_venv/bin/python /tmp/ctx.py < pairs2.txt   # 28 SGLang file:line sites, ±10 lines
/root/ccfa_venv/bin/python /tmp/ctx.py < pairs3.txt   # 17 Tier-B file:line sites, ±10 lines
grep -rn "def do_kv_cache_update" /root/ccfa_venv/.../vllm            # 16 impls
grep -rn "chunked_prefill_paged_decode|ops.prefix_prefill" <vllm>     # proves ROCm-only reachability
grep -rn "enable_page_major_kv_layout|mxfp8|evict_host(|post_capture" <sglang>
```
Installed-tree identity verified on the rig: `vllm 0.29.0 /root/ccfa_venv/lib/python3.12/site-packages/vllm/__init__.py`; `sglang 0.5.19`.

### 4.2 Prior-art exclusion (local)

```
python3 - <<'PY'   # 34 candidate fingerprints × KV_CACHE_GAP_MAP.md (3,540 lines) + spec-decode-gap-map-2026-09-13.md (1,191 lines)
  PRESENT-in-row:  sm120, simple_kv_offload, fp4 kv, cuda graph, evict_host, cache_salt, kv cache events, hf3fs/3fs, rope_kvcache
  PRESENT(spec map only): sm12.0, mrv2, mxfp8
  ABSENT from both: paged KV not supported, fused fp8 output + paged kv, store_threshold tiering, prefix cache stats,
                    runner_kv_caches, HND kvcache cpu offload, page-major layout cpu offloading, unified layout cpu offloading,
                    post-capture kv backing, prefix-valid commit, hicache draft pools, retraction policy radix, buffer_only,
                    prefill-only-disable-kv-cache, session radix cache rust, chunk cache prefix caching, pure swa allocator,
                    save kv cache fallback rotary, sparse coordinator, get_cpu_copy paged allocator fixme, kv_canary,
                    fa_skip_kv_cache descaling, kv_cache_dtype auto fp8 prefill, kv cache groups same num blocks
PY
```

### 4.3 GitHub — issues **and** PRs, per-repo (the log below is machine-generated from the run records; `vllm` = `vllm-project/vllm`, `sglang` = `sgl-project/sglang`; hit counts are result nodes parsed from the page, including PRs)

<!-- BEGIN generated github log -->
#### Batch 1 — primary candidate sweep (65 queries)

| # | query | `vllm-project/vllm` | `sgl-project/sglang` |
|---|---|---|---|
| 1 | `"Paged KV not supported on SM 12.0"` | 3 hit(s) | 1 hit(s) |
| 2 | `"fused FP8 output" AND paged` | 3 hit(s) | 0 hit(s) |
| 3 | `"fused FP8 output"` | 5 hit(s) | 0 hit(s) |
| 4 | `"store_threshold"` | 12 hit(s) | 0 hit(s) |
| 5 | `simple_kv_offload` | 25 hit(s) | 0 hit(s) |
| 6 | `SimpleCPUOffloadConnector eviction` | 17 hit(s) | 0 hit(s) |
| 7 | `num_stored_blocks` | 5 hit(s) | 0 hit(s) |
| 8 | `"prefix cache stats"` | 25 hit(s) | 0 hit(s) |
| 9 | `runner_kv_caches` | 5 hit(s) | 0 hit(s) |
| 10 | `"KV cache sharing"` | 25 hit(s) | 19 hit(s) |
| 11 | `"kv_sharing_fast_prefill"` | 25 hit(s) | 2 hit(s) |
| 12 | `"QK Norm" AND RoPE AND KVCache` | 25 hit(s) | 25 hit(s) |
| 13 | `rope_kvcache_fusion` | 3 hit(s) | 0 hit(s) |
| 14 | `"kv_cache_dtype" auto fp8 prefill unsupported` | 25 hit(s) | 25 hit(s) |
| 15 | `"different block_size" kv_cache_group` | 25 hit(s) | 1 hit(s) |
| 16 | `"number of blocks" kv_cache_groups` | 25 hit(s) | 1 hit(s) |
| 17 | `unify_kv_cache_spec_page_size fallback` | 20 hit(s) | 0 hit(s) |
| 18 | `prefix caching RISC-V` | 13 hit(s) | 0 hit(s) |
| 19 | `SlidingWindowManager "fine-grained"` | 14 hit(s) | 0 hit(s) |
| 20 | `triton attention "FP8 KV cache is not supported"` | 5 hit(s) | 0 hit(s) |
| 21 | `prefix_prefill fp8 "Unsupported FP8"` | 3 hit(s) | 0 hit(s) |
| 22 | `KVCacheCoordinatorNoPrefixCache` | 7 hit(s) | 0 hit(s) |
| 23 | `FA4 "does not support updating KV cache in-place"` | 0 hit(s) | 0 hit(s) |
| 24 | `flash_attention_v4_sm120` | 0 hit(s) | 1 hit(s) |
| 25 | `"Post-capture KV backing"` | 0 hit(s) | 0 hit(s) |
| 26 | `"post-capture" quantized` | 4 hit(s) | 5 hit(s) |
| 27 | `"prefix-valid commit"` | 0 hit(s) | 0 hit(s) |
| 28 | `set_kv_buffer_prefix_valid` | 0 hit(s) | 4 hit(s) |
| 29 | `"page-major" "CPU offloading"` | 0 hit(s) | 0 hit(s) |
| 30 | `"page-major" fp4` | 0 hit(s) | 21 hit(s) |
| 31 | `"CPU offloading is unsupported"` | 0 hit(s) | 1 hit(s) |
| 32 | `HND "CPU offload"` | 6 hit(s) | 1 hit(s) |
| 33 | `SGLANG_USE_HND_KVCACHE` | 0 hit(s) | 3 hit(s) |
| 34 | `"HiCache draft pools"` | 0 hit(s) | 1 hit(s) |
| 35 | `HiCacheDraftPlan OR "draft sidecar"` | 0 hit(s) | 9 hit(s) |
| 36 | `hicache "cuda graph"` | 3 hit(s) | 25 hit(s) |
| 37 | `"Other eviction policies are not supported"` | 0 hit(s) | 0 hit(s) |
| 38 | `evict_host backup_only` | 0 hit(s) | 0 hit(s) |
| 39 | `"retraction policy" radix` | 0 hit(s) | 4 hit(s) |
| 40 | `"cache_salt" radix tree` | 3 hit(s) | 24 hit(s) |
| 41 | `HiRadixCache "kv cache events"` | 0 hit(s) | 13 hit(s) |
| 42 | `"buffer_only"` | 25 hit(s) | 25 hit(s) |
| 43 | `"prefill-only-disable-kv-cache"` | 0 hit(s) | 25 hit(s) |
| 44 | `"no-op pool"` | 0 hit(s) | 2 hit(s) |
| 45 | `"enable-session-radix-cache"` | 0 hit(s) | 21 hit(s) |
| 46 | `Rust TreeCore radix` | 0 hit(s) | 19 hit(s) |
| 47 | `ChunkCache prefix caching` | 0 hit(s) | 6 hit(s) |
| 48 | `PureSWATokenToKVPoolAllocator` | 0 hit(s) | 11 hit(s) |
| 49 | `"save kv cache" rotary` | 0 hit(s) | 15 hit(s) |
| 50 | `fused_set_kv_buffer_arg` | 0 hit(s) | 25 hit(s) |
| 51 | `mini_3fs metadata eviction` | 0 hit(s) | 0 hit(s) |
| 52 | `HiCache "data eviction" prefix information` | 0 hit(s) | 0 hit(s) |
| 53 | `"Unsupported KV cache type for decode offload"` | 0 hit(s) | 5 hit(s) |
| 54 | `walk_radix_cache_for_canary` | 0 hit(s) | 0 hit(s) |
| 55 | `fa_skip_kv_cache` | 0 hit(s) | 11 hit(s) |
| 56 | `sparse_coordinator hierarchical KVCache` | 0 hit(s) | 0 hit(s) |
| 57 | `MXFP8 KV cache` | 25 hit(s) | 25 hit(s) |
| 58 | `"MXFP8" HND` | 7 hit(s) | 1 hit(s) |
| 59 | `"paged allocator" get_cpu_copy` | 0 hit(s) | 6 hit(s) |
| 60 | `"hicache-host-memory-mode"` | 0 hit(s) | 11 hit(s) |
| 61 | `hicache write_back "buffer_only"` | 0 hit(s) | 19 hit(s) |
| 62 | `hicache-size not supported` | 2 hit(s) | 25 hit(s) |
| 63 | `"disaggregation-decode-enable-radix-cache"` | 0 hit(s) | 25 hit(s) |
| 64 | `"does not support" "flashinfer attention backend"` | 14 hit(s) | 5 hit(s) |
| 65 | `"plain KV dequant reads"` | 0 hit(s) | 0 hit(s) |

#### Batch 2 — alternate-phrasing sweep (25 queries)

| # | query | `vllm-project/vllm` | `sgl-project/sglang` |
|---|---|---|---|
| 66 | `"page_major_kv_layout" OR "page-major KV layout"` | 0 hit(s) | 12 hit(s) |
| 67 | `page-major layout offload unsupported` | 3 hit(s) | 6 hit(s) |
| 68 | `HiCache host eviction policy backup_only` | 0 hit(s) | 0 hit(s) |
| 69 | `"backup_only"` | 2 hit(s) | 7 hit(s) |
| 70 | `"updating KV cache in-place"` | 0 hit(s) | 0 hit(s) |
| 71 | `FA4 KV cache update in place sm120` | 3 hit(s) | 4 hit(s) |
| 72 | `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` | 0 hit(s) | 3 hit(s) |
| 73 | `"Post-capture KV"` | 0 hit(s) | 25 hit(s) |
| 74 | `"prefix_valid" OR "set_kv_buffer_prefix_valid"` | 1 hit(s) | 5 hit(s) |
| 75 | `mxfp8 KV cache prefix valid commit` | 11 hit(s) | 1 hit(s) |
| 76 | `fp4 KV cache "dequantize_kv_tensor" OR "plain KV dequant"` | 0 hit(s) | 0 hit(s) |
| 77 | `kv_canary OR "KV canary"` | 0 hit(s) | 25 hit(s) |
| 78 | `"sparse_coordinator" OR "SparseCoordinator"` | 0 hit(s) | 16 hit(s) |
| 79 | `"store_threshold is not supported"` | 0 hit(s) | 0 hit(s) |
| 80 | `"fused FP8 output + paged KV"` | 0 hit(s) | 0 hit(s) |
| 81 | `"Paged KV does not support irregular head dim"` | 0 hit(s) | 0 hit(s) |
| 82 | `"Quantized KV cache not supported"` | 0 hit(s) | 0 hit(s) |
| 83 | `embedding mode "FP8 KV cache descaling"` | 0 hit(s) | 0 hit(s) |
| 84 | `"after paged allocator is implemented"` | 0 hit(s) | 0 hit(s) |
| 85 | `"--hicache-size" not supported` | 1 hit(s) | 25 hit(s) |
| 86 | `unified memory pool CPU offload unsupported` | 18 hit(s) | 15 hit(s) |
| 87 | `MXFP8 "SGLANG_USE_HND_KVCACHE"` | 0 hit(s) | 0 hit(s) |
| 88 | `MXFP8 "dcp_kv_mask" OR "DCP KV masks"` | 0 hit(s) | 11 hit(s) |
| 89 | `page-major "no per-layer contiguous regions"` | 0 hit(s) | 0 hit(s) |
| 90 | `"no-op pool" OR "noop pool" prefill context parallel` | 0 hit(s) | 4 hit(s) |

#### Batch 3 — focused NOT-FOUND verification sweep (14 queries)

| # | query | `vllm-project/vllm` | `sgl-project/sglang` |
|---|---|---|---|
| 91 | `"plain KV dequant reads" OR "does not support plain KV dequant"` | 0 hit(s) | 0 hit(s) |
| 92 | `"prefix-valid" OR "set_kv_buffer_prefix_valid"` | 1 hit(s) | 5 hit(s) |
| 93 | `"evict_host" OR "Other eviction policies"` | 1 hit(s) | 25 hit(s) |
| 94 | `"page-major" "CPU offloading" OR "page-major layout"` | 0 hit(s) | 13 hit(s) |
| 95 | `"updating KV cache in-place" OR "FA4 does not support updating"` | 0 hit(s) | 0 hit(s) |
| 96 | `walk_radix_cache_for_canary OR kv_canary` | 0 hit(s) | 25 hit(s) |
| 97 | `"Quantized KV cache not supported" OR extract_hidden_states` | 25 hit(s) | 0 hit(s) |
| 98 | `"kv cache dtype" checkpoint config auto` | 25 hit(s) | 25 hit(s) |
| 99 | `"page-major KV layout is not supported"` | 0 hit(s) | 0 hit(s) |
| 100 | `"Post-capture KV backing" OR "post-capture"` | 18 hit(s) | 25 hit(s) |
| 101 | `"CPU offloading is unsupported under the unified"` | 0 hit(s) | 0 hit(s) |
| 102 | `"alloc_extend is only for paged allocator" OR "after paged allocator is implemented"` | 0 hit(s) | 0 hit(s) |
| 103 | `MXFP8 "SGLANG_USE_HND_KVCACHE"` | 0 hit(s) | 0 hit(s) |
| 104 | `"PureSWATokenToKVPoolAllocator"` | 0 hit(s) | 11 hit(s) |

**104 query executions · 208 page fetches · 0 hard fetch failures · 1,269 result nodes parsed.**
<!-- END generated github log -->

### 4.4 GitHub — direct page reads (bodies fetched, not just titles)

| URL | why read | outcome |
|---|---|---|
| https://github.com/vllm-project/vllm/pull/47235 | does it cover the `simple_kv_offload` FIXME? | **yes** — body: *"Resolves the FIXME: num_stored_blocks can be stale and omit evicted blocks in the middle of the request"* |
| https://github.com/sgl-project/sglang/pull/29533 | does the page-major PR state its own limits? | **yes** — *"Layout-incompatible inherited methods (contiguous-buf-infos, CPU offload, prefix-commit) raise NotImplementedError"*; *"`--enable-page-major-kv-layout` is not yet supported with fp4 KV cache (asserted)"* |
| https://github.com/sgl-project/sglang/pull/32452 | HND × CPU offload | **yes** — it is the fix, and it explicitly rejects HND × quantized KV |
| https://github.com/sgl-project/sglang/pull/29678 | does the unified-pool PR mention CPU offload? | **no** — body retrieved (5,293 chars), limitations listed are PD disagg / spec decode / prefill CUDA graph, not CPU offload |
| https://github.com/sgl-project/sglang/issues/25371 | does the allocator RFC state the FIXME? | **body unretrievable** → §3.1 |

### 4.5 arXiv

<!-- BEGIN generated arxiv log -->
#### arXiv (`https://arxiv.org/search/?searchtype=all&query=…`, HTML UI; the `export.arxiv.org` API returned HTTP 429 on every attempt)

| # | query | result |
|---|---|---|
| 1 | `KV cache SM120 Blackwell paged attention` | `total=0` |
| 2 | `FA4 flash attention paged KV cache sm120` | `total=0` |
| 3 | `CPU offload KV cache page-major layout` | `total=0` (first attempt FETCH_FAILED; succeeded on re-run) |
| 4 | `quantized KV cache CPU offloading` | `total=6` — arXiv:2607.07144 *Fractal KV-Cache Archives*; arXiv:2602.07721 *ParisKV*; arXiv:2507.19823 *HCAttention*; arXiv:2505.19586 *TailorKV*; arXiv:2503.16163 *SpeCache*; arXiv:2502.12665 *A²ATS*. **None relevant to any §1 item.** |
| 5 | `post-capture KV cache backing virtual memory reservation` | `total=0` |
| 6 | `prefix cache eviction policy host tier HiCache` | `total=0` |
| 7 | `KV cache events radix tree prefix cache observability` | `total=0` |
| 8 | `cache salt prefix cache isolation radix tree` | `total=0` |
| 9 | `sliding window allocator page size KV cache` | `total=0` |
| 10 | `radix cache retraction policy preemption KV` | `total=0` |
| 11 | `vLLM paged KV cache SM120 unsupported attention` | `total=0` |
| 12 | `SGLang FA4 KV cache in-place update` | `total=0` |

11 of 12 queries returned an explicit zero count. The residual limitation is vocabulary and coverage, not the fetch path — see §3.3.
<!-- END generated arxiv log -->

### 4.6 Open web (`web_search` tool) — 18 query executions

Grouped by intent; each was followed by a direct `curl` read of any artefact that mattered.

- SM120 / FA4 / paged KV: `"Paged KV not supported on SM 12.0" vllm tml_fa4`; `vLLM SM120 paged attention not supported RTX PRO 6000`; `SGLang sm120 attention backend paged KV support Blackwell`; `vLLM flash attention FA4 CuTe sm120 paged KV assertion`
- page-major layout / FP4: `SGLang "page-major" KV layout fp4 KV cache not supported`; `SGLang "page-major" KV layout fp4`; `SGLang "unified memory pool" "CPU offloading is unsupported"`
- post-capture: `vLLM "post-capture" KV cache backing virtual memory reservation`; `SGLang "post-capture" memory pool KV cache backing`
- SimpleCPUOffload eviction: `vLLM simple_kv_offload CPU cache eviction stale num_stored_blocks`; `vLLM "Other eviction policies" host offload backup_only`; `vLLM SimpleCPUOffloadScheduler store_threshold tiering`
- HiCache: `SGLang hieracical cache piecewise CUDA graph disable HiCache loading`; `SGLang "HiCache" draft pool SIDECAR unified radix cache requirement`
- FA4 in-place: `SGLang FA4 flash attention updating KV cache in-place NotImplementedError`
- radix/prefix: `SGLang "cache_salt" "C++ radix tree" experimental`; `SGLang radix cache "retraction policy" improve decode preemption`
- misc: `vLLM prefix_prefill "kv_cache_dtype='auto'" FP8 KV Cache prefill kernel`; `vLLM "QK Norm" "RoPE" "KVCache fusion" unsupported dtype`; `vLLM "KV cache sharing" MRv2 model runner not supported`; `vLLM "runner_kv_caches" multiple attention layers same decoder block`; `SGLang "save kv cache" fallback_rotary_embedding fused set_kv_buffer`; `SGLang "prefill-only-disable-kv-cache" no-op pool`; `vLLM "number of blocks" kv_cache_groups different block_size future`; `vLLM prefix cache stats conditional log_stats FIXME`; `SGLang PureSWA allocator page_size greater than 1 SWA spec decode`; `SGLang chunk cache prefix caching unsupported decode instance`; `SGLang "prefix-valid" set_kv_buffer_prefix_valid MXFP8`; `SGLang "unified memory pool" "CPU offloading is unsupported"`

---

## 5. Tally

| stage | count |
|---|---|
| self-admitted-gap lines scanned (Tier A KV-topic) | 130 (58 vLLM + 72 SGLang) |
| additional KV-relevant lines recovered by the recall repair (Tier B) | 204 (120 + 84) |
| distinct `file:line` sites pulled with ±10 lines of context and read | 65 |
| distinct candidates that reached search | 34 |
| **Section 1 — no matching discussion found** | **9** (2 `HIGH`, 6 `MEDIUM`, 1 `LOW`; reachability: 5 yes, 2 partial, 2 no) |
| Section 2 — admitted but already discussed | 25 groups (20 fully URL-matched, 5 hardware/model-scoped) |
| Section 3 — inconclusive / blocked | 5 entries (1 unretrievable body, 3 surface-level blockers, 1 set of 5 unclassifiable candidates) |
| GitHub query executions / page fetches / hard failures | 104 / 208 / 0 |
| arXiv queries | 12 — 11 returned an explicit `total=0`; 1 returned 6 unrelated results |
| open-web query executions | 18 |

**No file other than this one was created or modified in the repository.** Scratch extractors and run records lived in `/tmp/kvneg/` on the analysis host and in `/tmp/` on the rig.
