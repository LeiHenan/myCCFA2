# TGI (huggingface/text-generation-inference) — Determinism / Batch-Invariance Prior-Art Sweep

**Verdict: TGI has NOTHING for determinism / reproducibility / batch-invariance** — zero env vars, zero CLI
flags, zero docs pages, zero issues or PRs on the topic.

It *does* contain the exact mechanism that batch-invariance work exists to defeat (batch-size-dependent
attention-kernel selection) plus the enabling conditions (TF32 enabled, CUDA-graph batch buckets).

**Source basis:** full `main` tarball downloaded and grepped.
HEAD = `b4adbf2f6e2e` (2026-03-21T11:34:22Z), verified via `commits/main.atom`; matches tarball mtime.
Repo is in **maintenance mode** since #3345 (2025-12-18).
Raw fetches: `/tmp/eng/tgi/` · extracted tree: `/tmp/eng/tgi/text-generation-inference-main/`

## 1) Evidence table

| # | Engine | Artifact | URL | Status | READ BODY / TITLE ONLY | Path/section | One-line relevance |
|---|---|---|---|---|---|---|---|
| 1 | TGI | repo-wide term sweep | https://github.com/huggingface/text-generation-inference (main@b4adbf2f6e2e) | present-in-source | READ BODY | whole 11MB tree, `*.py/*.rs/*.md/*.mdx/*.toml/*.yaml` | Only 5 hits for `deterministic\|reproducib\|batch_invar` in ENTIRE repo; none is a determinism feature |
| 2 | TGI | TF32 enabled globally | https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/models/__init__.py#L66 | present-in-source | READ BODY | `models/__init__.py:66,69` | `allow_tf32=True` on matmul and cuDNN — low-precision reductions, no determinism guard |
| 3 | TGI | **BATCH-SIZE-DEPENDENT KERNEL SELECTION** | https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/attention/cuda.py#L132 | present-in-source | READ BODY | `cuda.py:132-134` | `use_v1 = max_s <= 8192 and (max_num_partitions == 1 or num_seqs * num_heads > 512)` — kernel switches on BATCH SIZE (`num_seqs`) |
| 4 | TGI | PagedAttention V2 split-K + reduction | https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/attention/cuda.py#L15 | present-in-source | READ BODY | `cuda.py:15,64,137-170` | `_PARTITION_SIZE=512`; `max_num_partitions=(max_s+511)//512`; V2 allocates `tmp_output/exp_sums/max_logits` then reduces → split-K reduction order varies |
| 5 | TGI | flash-decoding split caveat | https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/attention/cuda.py#L93 | present-in-source | READ BODY | `cuda.py:88-97` | Comment: "Number of splits is not correctly handled by the current path … window_right is set to 0 and the split logic is never applied." |
| 6 | TGI | DEFAULT attention backend = flashdecoding | https://github.com/huggingface/text-generation-inference/blob/main/launcher/src/main.rs#L146 | present-in-source | READ BODY | `launcher/src/main.rs:146-151` | `fallback_attention` = `"paged"` if no CUDA or major<8, else `"flashdecoding"` → flashdecoding is default on Ampere+ |
| 7 | TGI | CUDA-graph batch buckets | https://github.com/huggingface/text-generation-inference/blob/main/launcher/src/main.rs#L2181 | present-in-source | READ BODY | `launcher/src/main.rs:2181` | default `cuda_graphs = [1,2,4,8,16,32]`; batches padded to fixed buckets set by concurrent load → per-request numerics vary with load |
| 8 | TGI | `seed` = sampling-only, no invariant guarantee | https://github.com/huggingface/text-generation-inference/blob/main/router/src/lib.rs#L410 | present-in-source | READ BODY | `router/src/lib.rs:410-418`; `docs/openapi.json` → `GenerateParameters.seed` | Doc string is literally "Random sampling seed." — no cross-batch/cross-kernel claim; NO server-level seed flag exists |
| 9 | TGI | seed default = random per request | https://github.com/huggingface/text-generation-inference/blob/main/router/src/validation.rs#L296 | present-in-source | READ BODY | `router/src/validation.rs:296-305` | `// If seed is None, assign a random one` via `thread_rng().gen()` |
| 10 | TGI | per-request `torch.Generator` | https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/utils/tokens.py#L507 | present-in-source | READ BODY | `tokens.py:507-517, 530-540` | Sampling = `torch.Generator(device)` + `manual_seed(seed)` PER REQUEST (`HeterogeneousSampling` loops per row) → sampling RNG is batch-independent; divergence must originate in forward-pass logits |
| 11 | TGI | "reproducible" philox seed (triton fallback) | https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/attention/flash_attn_triton.py#L754 | present-in-source | READ BODY | `flash_attn_triton.py:754-756` | `philox_seed=0x1BF52` hardcoded "so we get reproducible results for testing"; reachable only via ROCm fallback (`rocm.py:323`) and MPT (`mpt_modeling.py:270`) |
| 12 | TGI | `deterministic` hardcoded False | https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/models/custom_modeling/qwen2_vl.py#L189 | present-in-source | READ BODY | `qwen2_vl.py:189`; `qwen2_5_vl.py:519` | The ONLY literal "deterministic" flag in the repo: `flash_attn_2_cuda.varlen_fwd(..., False, # deterministic, None)` — hardcoded off, vision encoder only |
| 13 | TGI docs | **NO determinism/reproducibility page** | https://github.com/huggingface/text-generation-inference/blob/main/docs/source/_toctree.yml | present-in-source | READ BODY | `docs/source/_toctree.yml` (full); `docs/source/*` listing | Full toctree has no determinism/reproducibility section; no such `.md`/`.mdx` file exists in `docs/` |
| 14 | TGI docs | **NO seed/determinism CLI flag** | https://github.com/huggingface/text-generation-inference/blob/main/docs/source/reference/launcher.md | present-in-source | READ BODY | `docs/source/reference/launcher.md`; `launcher/src/main.rs` flag list | `grep seed\|determin\|reproduc` in launcher.md = 0 matches; enumerated 45 launcher flags contain NO `--seed` / `--deterministic` |
| 15 | TGI | PR: flashdecoding replaces paged as default | https://github.com/huggingface/text-generation-inference/pull/1940 | merged | TITLE ONLY | PR title | "[Major Change][Undecided yet] Move to FlashDecoding instead of PagedAttention kernel." — merged |
| 16 | TGI | PR: introduce sampling seeding | https://github.com/huggingface/text-generation-inference/pull/37 | merged | TITLE ONLY | PR title | "feat: Support sampling seeding" — origin of per-request seed; sampling-scope only |
| 17 | TGI | PR: fix seeding on gpu | https://github.com/huggingface/text-generation-inference/pull/42 | merged | TITLE ONLY | PR title | "fix(server): fix seeding on gpu" — seed correctness was a per-request sampling bug, not a determinism feature |
| 18 | TGI | PR: fix seeding with multiple shards | https://github.com/huggingface/text-generation-inference/pull/44 | merged | READ BODY (diff) | `pr_44.diff`: `proto/generate.proto` seed optional→required; `router/src/validation.rs` | Adds "if seed is None, assign a random one"; confirms seed is per-request sampling state only |
| 19 | TGI | PR "Fix seeded output." | https://github.com/huggingface/text-generation-inference/pull/1949 | merged | READ BODY (diff) | `pr_1949.diff`: `Cargo.toml`/`Cargo.lock`/`poetry.lock`/`requirements*` + `__snapshots__/test_flash_llama_simple.json` | Diff is ONLY version bumps + regenerated snapshots: the SAME test with the SAME seed produced DIFFERENT generated text after a dependency bump → seeded output not stable |
| 20 | TGI | Issue: same seed + prefix caching → different responses | https://github.com/huggingface/text-generation-inference/issues/2670 | open | READ BODY (+comments) | Issue body + comments by `sam-ulrich1`, `claudioMontanari` | "exact same params and input render different results with seed set" across machines; author: "After disabling prefix caching I seem to be getting the same response" — seed does NOT guarantee identity |
| 21 | TGI | Issue: kernel choice changes output | https://github.com/huggingface/text-generation-inference/issues/1262 | closed | READ BODY | "Inconsistent generation with flash attention disabled" | Same seed `123456`, FA on → coherent text; FA off → "nonsense, repeated output" — numerics are kernel-path dependent |
| 22 | TGI | Issue: long prompt → different responses | https://github.com/huggingface/text-generation-inference/issues/1159 | closed | READ BODY | "too long prompt could get different response from the same model" | Same request, prompt >6000, multiple requests differ even at `temperature=0.001` |
| 23 | TGI | Issue: no global seed / cannot match HF | https://github.com/huggingface/text-generation-inference/issues/1041 | closed | READ BODY | "Is there a way to set global seed?" | User asks for `transformers.enable_full_determinism` equivalent; no such global determinism knob in TGI |
| 24 | TGI | Issue: seeds ignored | https://github.com/huggingface/text-generation-inference/issues/254 | closed | READ BODY | "Seed not working" | `seed=0/42/64` all returned identical text |
| 25 | TGI | Issue: seed not working | https://github.com/huggingface/text-generation-inference/issues/326 | closed | READ BODY | "seed not working" | bloom-560m, seed had no effect on output |
| 26 | TGI | Issue: temperature-1.0 seed bug | https://github.com/huggingface/text-generation-inference/issues/687 | closed | TITLE ONLY | title | "At a temperature of 1.0, the seeds will not work properly" |
| 27 | TGI | PR: seed 1 in all tests | https://github.com/huggingface/text-generation-inference/pull/1591 | closed | TITLE ONLY | title | "fix: prefer seed 1 in all tests" — seeds used only for test snapshot stability |
| 28 | TGI | Issue: endpoint path changes results | https://github.com/huggingface/text-generation-inference/issues/2747 | open | READ BODY | "Different inference results and speed between /generate and OpenAI endpoint" | 2.4.0 + `PREFIX_CACHING=true`: same model differs across API paths |
| 29 | TGI | Issue: endpoint path changes results | https://github.com/huggingface/text-generation-inference/issues/3203 | open | READ BODY | "Different result between /chat/completions and /generate endpoint" | 3.2.3: different generation across endpoints |
| 30 | TGI | Search: NO matching issues/PRs | https://api.github.com/search/issues?q=repo:huggingface/text-generation-inference+deterministic+in:title | total_count=0 | READ BODY (API JSON) | — | Zero issues/PRs with "deterministic" in title |
| 31 | TGI | Search: NO matching issues/PRs | https://api.github.com/search/issues?q=repo:huggingface/text-generation-inference+reproducib | total_count=0 | READ BODY (API JSON) | — | Zero issues/PRs mentioning "reproducib*" anywhere in body |
| 32 | TGI | Search: NO batch-invariance concept | https://api.github.com/search/issues?q=repo:huggingface/text-generation-inference+%22batch+invarian%22 | total_count=0 | READ BODY (API JSON) | — | Zero results — no `VLLM_BATCH_INVARIANT` analogue anywhere |

### Key negative results (explicit)

- **NO `VLLM_BATCH_INVARIANT` analogue.**
- **NO `--seed` launcher flag** (45 enumerated launcher flags).
- **NO determinism env var** in the full enumerated env-var set: `ATTENTION`, `DISABLE_CUSTOM_KERNELS`,
  `DISABLE_SGMV`, `PREFIX_CACHING`, `CUDA_GRAPHS`, `VLLM_CONTIGUOUS_PA`, `VLLM_GRAPH_PROMPT_RATIO`,
  `PYTORCH_TUNABLEOP_*`, `MAX_BATCH_*`, `SDP_ON_BF16`, `TGI_GRAPH_RESERVED_MEM`, …
- **NO** `torch.use_deterministic_algorithms`, `cudnn.deterministic`, or `CUBLAS_WORKSPACE_CONFIG` anywhere.
- The only "invariance" hits in the tree are unrelated: softmax translation-invariance comment
  (`bloom_modeling.py:102`) and radix-tree allocation invariants (`backends/v3/src/radix.rs:919+`).

## 2) Queries run

**Local greps** (on `/tmp/eng/tgi/text-generation-inference-main`, 11MB, full tree):

- `grep -rniE "deterministic|reproducib|batch_invar" --include=*.py --include=*.rs --include=*.md --include=*.mdx --include=*.toml --include=*.yaml --include=*.yml .` → 5 hits
- `grep -rniE "\bseed\b" --include=*.py --include=*.rs server router launcher backends`
- `grep -rniE "split[-_ ]?k|num_splits|numsplit"` (all C/CUDA/Python/Rust/MD) → 0 hits
- `grep -rniE "batch_invar|batch-invar|invariance|invarian"` (all files) → only unrelated hits
- `grep -rniE "flashdecoding|flash-decoding|FLASH_DECODING"`
- `grep -rniE "VLLM_[A-Z_]+" -o` → confirms no `VLLM_BATCH_INVARIANT`
- `grep -rniE "use_deterministic_algorithms|cudnn\.deterministic|CUBLAS_WORKSPACE_CONFIG|torch\.backends\.cudnn|allow_tf32|float32_matmul_precision"` → only `allow_tf32=True` (2 lines)
- Enumerated all `os.environ[...]` names in `server/ launcher/ router/ backends/`
- `grep -rniE "seed|determin|reproduc" docs/`; read `docs/source/_toctree.yml` fully; grep on `docs/source/reference/launcher.md`
- `grep -oE '"--[a-z0-9-]+"' launcher/src/main.rs` → 45 flags, no `--seed`
- `grep -rniE "cuda_graph|graph_bucket|CUDA_GRAPH"`
- Direct reads: `cuda.py:15,55-170,215-300`; `models/__init__.py:58-70`; `tokens.py:24-135,500-560`;
  `router/src/lib.rs:405-422`; `router/src/validation.rs:296-305`; `launcher/src/main.rs:130-175,1118-1130,2165-2190`;
  `flashinfer.py:155-200`; `qwen2_vl.py:160-205`; `qwen2_5_vl.py:495-535`; `flash_attn_triton.py:735-790`

**Network:**

- `GET https://codeload.github.com/huggingface/text-generation-inference/tar.gz/refs/heads/main` → `/tmp/eng/tgi/tgi.tar.gz` (3,154,971 bytes)
- `GET https://github.com/huggingface/text-generation-inference/commits/main.atom` → HEAD `b4adbf2f6e2e` 2026-03-21T11:34:22Z
- GitHub search API (`repo:huggingface/text-generation-inference`):
  `deterministic+in:title` (0) · `reproducib` (0) · `non-deterministic` (3) · `"batch invarian"` (0) ·
  `"same seed"` (8) · `flashdecoding` (18) · `seed+in:title` (10) · `"batch size"+different output` (56) ·
  `is:merged+flashdecoding+in:title` (3) · `is:merged+seed+in:title` (4)
- GitHub REST core API: `/issues/2670`, `/issues/1041` (bodies)
- HTML fetch: `/issues/1262`, `/1159`, `/254`, `/326`, `/2747`, `/3203`, `/2670`; `/pull/1940`, `/1949`, `/37`, `/44`
- Raw diffs: `/pull/1949.diff` (932 lines), `/pull/44.diff` (445 lines)
- `web_search`: "text-generation-inference TGI deterministic reproducible batch invariance";
  "TGI seed different results batch size huggingface text-generation-inference"

## 3) Items that could NOT be opened

- **COULD NOT OPEN:** `https://api.github.com/repos/huggingface/text-generation-inference/issues/1262` , `/1159` , `/254` , `/687`
  — HTTP 403 `API rate limit exceeded for 202.120.234.156` (shared unauthenticated IP).
  Worked around via `github.com` HTML pages — those bodies **were** read.
- **COULD NOT OPEN:** `https://api.github.com/repos/huggingface/text-generation-inference/commits?sha=main`
  — HTTP 403 rate limit. Worked around with `commits/main.atom`.
- `https://github.com/huggingface/text-generation-inference/pull/1940` — HTML fetched (531 KB) but the PR body
  did not extract from `embeddedData`; title, state and merged-status obtained, body **not** read → recorded as TITLE ONLY.
- Third-party, **NOT primary, NOT fetched** (TITLE ONLY from search results, not used as evidence):
  `https://theneuralbase.com/tgi/learn/intermediate/seed-for-reproducibility/` — unofficial tutorial site,
  not TGI documentation.
