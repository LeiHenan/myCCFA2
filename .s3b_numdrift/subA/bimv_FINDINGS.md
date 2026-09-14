# bimv_FINDINGS.md — Prior-art sweep: batch invariance / determinism in LLM inference engines

Scope: official vLLM docs + blog, NVIDIA docs/blog, SGLang + llm-d issues, PyTorch reproducibility docs.
All fetches performed with `curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "<Chrome UA>"` per project rules.
`api.github.com` not used (rate-limited to zero). GitHub issue bodies extracted from embedded JSON (`"body":"…"`) in the fetched HTML.

**Encoding note (applies to all GitHub/forum quotes below):** GitHub's embedded JSON gave UTF-8 bytes that `unicode_escape`-decoded into mojibake for non-ASCII punctuation. Where an em-dash (U+2014) or curly quote appeared as `â`/`â` in the raw extraction, I have restored the intended character. All ASCII characters are byte-exact. Any other alteration is flagged inline.

---

## 1. Verified sources

| # | Exact URL | HTTP | Title / heading actually read back | VERBATIM quote | Classification | DECISION or SOFT? |
|---|-----------|------|-----------------------------------|----------------|----------------|-------------------|
| 1 | `https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/batch_invariance.md` | **200** (5741 bytes) | H1: `# Batch Invariance` | "Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch." | NUMERICAL (batch-composition-dependent reduction order) | **DECISION** — asserts model *output* determinism (token output), not a soft metric |
| 2 | `https://docs.vllm.ai/en/latest/features/batch_invariance.html` | **200** (670726 B; 302→ `/en/latest/features/batch_invariance/`) | `<title>`: `Batch Invariance - vLLM`; H1: `Batch Invariance¶` | "Batch invariance is currently in beta. Some features are still under active development." | NUMERICAL | SOFT-adjacent (doc framing) — see row 1 for the decision-level claim |
| 3 | `https://docs.vllm.ai/en/stable/features/batch_invariance.html` | **200** (628863 B; 302→ `/en/stable/features/batch_invariance/`) | `<title>`: `Batch Invariance - vLLM`; H1: `Batch Invariance¶` | same document text as row 2 at this build | NUMERICAL | same as row 2 |
| 4 | `https://raw.githubusercontent.com/vllm-project/vllm/main/docs/usage/reproducibility.md` | **200** (2019 bytes) | H1: `# Reproducibility` | "vLLM does not guarantee the reproducibility of the results by default, for the sake of performance." | NUMERICAL (scheduling nondeterminism) | **DECISION** — the doc's stated remedy is per-output reproducibility; see extra quotes §2.4 |
| 5 | `https://docs.vllm.ai/en/latest/configuration/env_vars.html` | **200** (1036239 B) | page auto-generated from vLLM source (`VLLM_BATCH_INVARIANT` entry) | "# Enable batch-invariant mode: deterministic results regardless of batch composition. Requires NVIDIA GPU with compute capability >= 9.0." | NUMERICAL | SOFT — a config comment; no argmax claim |
| 6 | `https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference` | **200** (83018 B) | `<title>`: `No More Train-Inference Mismatch: Bitwise Consistent On-Policy Reinforcement Learning with vLLM and TorchTitan \| vLLM Blog`; H1 identical | "We demonstrate an open-source bitwise consistent on-policy RL run with TorchTitan as the training engine and vLLM as the inference engine." | NUMERICAL (bitwise kernel equivalence, train vs infer) | **SOFT** — bitwise logit/numerics matching + RL reward curves; no token-flip claim |
| 7 | `https://github.com/vllm-project/vllm/issues/27433` | **200** (361584 B) | `[Feature]: Batch Invariant Feature and Performance Optimization · Issue #27433 · vllm-project/vllm` | "We have basically support Batch Invariant based on https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/" | NUMERICAL | SOFT — tracking issue, feature checklist |
| 8 | `https://github.com/vllm-project/vllm/issues/56370` | **200** (405088 B) | `[Bug]: Batch invariance is broken when sequence parallelism / async TP is enabled (`VLLM_BATCH_INVARIANT=1` + `pass_config.enable_sp`) · Issue #56370` | "the same request produces different logprobs — and with greedy sampling different tokens — depending on how many other requests are in the batch." | NUMERICAL (SP / all-reduce + fused GEMM-comm reordering) | **DECISION** — explicit greedy token flips; measured table row: "prompts whose greedy output changed 17 / 64" |
| 9 | `https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/` | **200** (96892 B) | `<title>`: `Defeating Nondeterminism in LLM Inference - Thinking Machines Lab`; H1: `Defeating Nondeterminism in LLM Inference` | "Confusingly, although this can lead to nondeterministic kernels, concurrency (and atomic adds) end up being completely uninvolved in LLM inference nondeterminism!" | NUMERICAL (batch-size-dependent reduction order) | **DECISION** — post is explicitly about greedy/temperature-0 token stability |
| 10 | `https://docs.nvidia.com/nim/large-language-models/1.12.0/deterministic-mode.html` | **200** (24689 B) | `<title>`: `Deterministic Generation Mode in NVIDIA NIM for LLMs — NVIDIA NIM for Large Language Models (LLMs)`; H1: `Deterministic Generation Mode in NVIDIA NIM for LLMs#` | "NVIDIA NIM for LLMs supports deterministic generation mode, which ensures consistent text generation across multiple inference runs and requests." | NUMERICAL | **DECISION** — "generated text remains consistent … particularly important for batched requests" |
| 11 | `https://developer.nvidia.com/blog/author/vmailthody/` | **200** (173927 B) | `<title>`: `Author: Vikram Sharma Mailthody \| NVIDIA Technical Blog` | (no batch-invariance post present — see §3.1) | — | — |
| 12 | `https://forums.developer.nvidia.com/t/why-nemotron-3-nvfp4-models-are-not-deterministic-using-vllm/372685` (and `.json`, both 200) | **200** (204710 B / 36052 B) | `Why nemotron 3 NVFP4 models are not deterministic using vLLM?` | "I cannot run nemotron 3 NVFP4 models using vLLM on a single DGX Spark in a determinstic way. I run the same prompt two consecutive time and the outputs are different. The client request configuration has temperature at 0 and a fixed seed." — post 1, user `sertitto`, launching with `-e VLLM_BATCH_INVARIANT=1` | **WEIGHT** (NVFP4 W4A4 quantization) + NUMERICAL | **DECISION** — temperature 0, fixed seed, differing outputs. **User forum, NOT official NVIDIA doc** |
| 13 | `https://docs.pytorch.org/docs/2.14/notes/randomness.html` | **200** (200810 B) | `<title>`: `Reproducibility — PyTorch 2.14 documentation`; H1: `Reproducibility#` | "Bitwise matching numerics across different SDPA backends are not guaranteed, even for the same inputs and dtype. Each backend performs floating-point accumulation in a different order, and because floating-point addition is not associative, the results will differ between backends." | NUMERICAL (reduction/accumulation order) | **SOFT** — bitwise logit/attention numerics; no argmax claim |
| 14 | `https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html` | **200** (257169 B) | `torch.use_deterministic_algorithms` API reference | "Reduction numerics are very sensitive to triton configs. In deterministic mode, Inductor will use some heuristics to pick the most promising configs rather than do autotuning." | NUMERICAL (reduction order) | **SOFT** — numeric-stability guidance |
| 15 | `https://github.com/sgl-project/sglang/issues/36291` | **200** (280476 B) | `--enable-deterministic-inference accepted and reported enabled, but has no effect on an NVFP4 W4A4 + hybrid-Mamba checkpoint with a DFlash2 drafter (GB10 / sm_121) · Issue #36291` | "On a **Qwen3.8-27B NVFP4 (W4A4) checkpoint with hybrid Mamba layers and a DFlash2 drafter**, the same flag is accepted, appears as `enable_deterministic_inference=True` in the resolved `ServerArgs`, and **temperature-0 outputs remain non-reproducible**." | **WEIGHT** (NVFP4 W4A4) + NUMERICAL | **DECISION** — temperature-0 non-reproducibility, silent coverage gap |
| 16 | `https://github.com/sgl-project/sglang/issues/39235` | **200** (268411 B) | `[Bug] DeepSeek-V4 SM120 decode pads q to 64 heads for an SM90 constraint; removing the pad changes greedy output despite the kernel being bit-identical · Issue #39235` | "What I would like a second opinion on is that removing it **changes greedy output**, which should be impossible if the padded heads are inert." | NUMERICAL (padding changes attention math) | **DECISION** — greedy output change despite bit-identical kernel |

**Agreement with parent's independently verified items:** rows 4, 6-area and the "no vLLM batch-invariance blog post" conclusion agree with everything the parent reported. I found no contrary evidence. See §3.1.

---

## 2. Supplementary verbatim material (same documents, additional passages)

### 2.1 vLLM `batch_invariance.md` — hardware claim and mechanism
> "Batch invariance is supported on the following platforms:"
> "- NVIDIA GPUs with compute capability 8.0 or higher."
> "- Intel XPUs with Triton support."

> "When batch invariance is enabled, vLLM:"
> "1. Uses deterministic kernel implementations for attention and other operations"
> "2. Ensures consistent numerical behavior across different batch sizes"
> "3. Disables certain optimizations that may introduce non-determinism (such as custom all-reduce operations in tensor parallel mode)"

> "Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility."

> "Batch invariance is currently in beta. Some features are still under active development."

**Internal contradiction worth flagging:** this feature doc says **compute capability 8.0 or higher**; the auto-generated env-var reference (row 5) says **"Requires NVIDIA GPU with compute capability >= 9.0."** Both are official vLLM docs at `main`/`latest`. Not reconciled anywhere I could retrieve.

### 2.2 vLLM `reproducibility.md` — the scheduling-vs-batch-invariance distinction
> "vLLM does not guarantee the reproducibility of the results by default, for the sake of performance. To achieve reproducible results:"
> "- In offline mode, you can either set `VLLM_ENABLE_V1_MULTIPROCESSING=0` which makes scheduling deterministic, or enable [batch invariance](../features/batch_invariance.md) to make the outputs insensitive to scheduling."
> "- In online mode, you can only enable [batch invariance](../features/batch_invariance.md)."

> "Even with the above settings, vLLM only provides reproducibility when it runs on the same hardware and the same vLLM version."

### 2.3 vLLM blog post — authorship check and cost figure
Authors as printed on the post: **Bram Wasti, Wentao Ye, Teja Rao, Michael Goin, Paul Zhang, Tianyu Liu, Natalia Gimelshein, Woosuk Kwon, Kaichao You, Zhuohan Li**. **No "Vikram Sharma" among them.**
> "Our current results show that the bitwise RL run is 2.4x slower than the non-bitwise case."
> "vLLM heavily leverages torch.compile and is able to maintain batch-invariance with it - but to maintain cross-framework compatibility would require a change to the trained version of the model."
> "We leveraged the forward pass kernels from vLLM's recent batch invariance work and wrote simple backward passes for these ops."

### 2.4 vLLM issue #56370 — the decision-level measurement
> "`VLLM_BATCH_INVARIANT=1` is supposed to make outputs independent of the batch composition. With tensor parallelism it holds as long as sequence parallelism is **off**, but as soon as `pass_config.enable_sp=True` is set (or `fuse_gemm_comms=True`, which implies it), the same request produces different logprobs — and with greedy sampling different tokens — depending on how many other requests are in the batch."
> "Nothing in the config guards against this combination: `enable_sp` / `fuse_gemm_comms` are never checked against `VLLM_BATCH_INVARIANT`"
> "`docs/features/batch_invariance.md` also does not mention SP / async TP."

Measured table (as printed in the issue body; TP=4, `enable_sp=True`, `sp_min_token_num=64`): **"tokens with different logprob 1522 / 1536"**, **"prompts whose greedy output changed 17 / 64"**, **"max abs logprob diff 2.98"**. The TP=4 / `enable_sp=False` baseline row reports **"0 / 1536"** and **"0 / 64"**.

### 2.5 Thinking Machines — the strongest determinism framing
> "What might be more surprising is that even when we adjust the temperature down to 0 This means that the LLM always chooses the highest probability token, which is called greedy sampling. (thus making the sampling theoretically deterministic), LLM APIs are still not deterministic in practice"
> "But why aren't LLM inference engines deterministic? One common hypothesis is that some combination of floating-point non-associativity and concurrent execution leads to nondeterminism based on which concurrent core finishes first."
> "Typically a GPU launches a program concurrently across many 'cores' (i.e. SMs). … The atomic add is 'nondeterministic' — the order in which the results accumulate is purely dependent on which core finishes first."
> "However, the forward pass of an LLM involves no operations that require atomic adds. Thus, the forward pass in an LLM is in fact 'run-to-run deterministic.'"
> "Nevertheless, from the perspective of anybody using the inference server, the results are nondeterministic."

### 2.6 NVIDIA NIM deterministic mode — full operative text
> "Deterministic mode is only supported on TRT-LLM buildable profiles. To enable deterministic generation, set the following environment variable:"
> "NIM_FORCE_DETERMINISTIC = 1"
> "FP8 Profiles : H100 GPUs with NVIDIA NVLink. A100 and H100-NVL are not supported."
> "The generated text remains consistent across multiple inference runs, particularly important for batched requests."
> "There may be a slight degradation in performance metrics, including latency and throughput, due to the additional constraints required to ensure determinism."

**Note:** this is **TensorRT-LLM via NIM**, and the env var is `NIM_FORCE_DETERMINISTIC`, **not** `VLLM_BATCH_INVARIANT`.

### 2.7 PyTorch reproducibility — remaining quotes
> "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds."
> "Deterministic operations are often slower than nondeterministic operations, so single-run performance may decrease for your model. However, determinism may save time in development by facilitating experimentation, debugging, and regression testing."
> "The backward pass uses non-deterministic atomic operations by default." (SDPA `SDPBackend.FLASH_ATTENTION` row)
> "For example, running the nondeterministic CUDA implementation of torch.Tensor.index_add_() will throw an error:"

### 2.8 `torch.use_deterministic_algorithms` — overlap caveat
> "This flag does not detect or prevent nondeterministic behavior caused by calling an inplace operation on a tensor with an internal memory overlap or by giving such a tensor as the out argument for an operation. In these cases, multiple writes of different data may target a single memory location, and the order of writes is not guaranteed."

### 2.9 SGLang #36291 — the silent-gap framing (decision-relevant)
> "On a GB10 (sm_121, aarch64), `--enable-deterministic-inference` **works correctly on a dense BF16 model** in the same container image, on the same GPU: identical concurrent requests collapse from 8 distinct completions to 1."
> "The flag therefore appears not to cover one or more of those paths, and it fails **silently** — the server reports the feature as on."
> "The failure mode is not 'deterministic mode is broken'. It is **a coverage gap that presents identically to success**. That is worse than an error, because the natural check — confirm the flag was applied — passes."

### 2.10 SGLang / vLLM item titles surfaced by repo-scoped issue search (titles verified in listings; bodies NOT fetched)
- `https://github.com/sgl-project/sglang/pull/38160` — `[Feature] Support BF16 and batch-invariant inference with DeepEP v2`
- `https://github.com/sgl-project/sglang/pull/38176` — `keeping router GEMM in fp32 for deterministic inference (DeepSeek V3/V4)`
- `https://github.com/sgl-project/sglang/pull/39319` — `[Bugfix] Preserve BF16 batch invariance with DeepGEMM 0.2`
- `https://github.com/sgl-project/sglang/issues/36174` — `[Feature] Support deterministic inference for DeepSeek-V4`
- `https://github.com/vllm-project/vllm/pull/55958` — `[Bugfix][Determinism] Refuse batch-invariant mode on unsupported platforms`
- `https://github.com/vllm-project/vllm/pull/56528` — `[Test][Determinism] Cover VLM batch invariance in default execution mode`
- `https://github.com/vllm-project/vllm/pull/56377` — `[Bugfix] Disable sequence parallelism / async TP under batch invariance and add a TP regression test`
- `https://github.com/vllm-project/vllm/pull/56244` — `[Model] Mamba2: Apply the gate inside the scan in batch-invariant mode`

---

## 3. Could NOT verify — and exactly why

### 3.1 A vLLM blog post by "Vikram Sharma" about batch invariance — **DOES NOT EXIST as described**
- `https://blog.vllm.ai/` → **200**, but `%{url_effective}` = `https://vllm.ai/blog` — it is the *same* page (identical 452583 bytes). There is no separate `blog.vllm.ai` site.
- `https://vllm.ai/blog` → **200**, 452583 bytes. I enumerated **all 130 post links** in the HTML. Grep counts: `invariance` = **0**, `determinis` = **0**, `Vikram` = **0**, `ikram` = **0**, `eproducib` = 2 (both in an unrelated DeepSeek-GB300 benchmark blurb).
- `https://vllm.ai/blog/rss.xml` → **200**, 37633 bytes, 50 `<item>` elements (feed is truncated to recent 50). Zero `<title>` matches for `batch|determin|invari|reproduc|bitwise`.
- `https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts` → **200**, 504611 bytes. Enumerated **134** `_posts/*.md` filenames. Exactly **one** matches `batch|determin|invari|reproduc|bitwise`: `_posts/2025-11-10-bitwise-consistent-train-inference.md` — whose author list (see §2.3) does **not** include Vikram Sharma.
- `https://developer.nvidia.com/blog/author/vmailthody/` → **200**, `<title>` = `Author: Vikram Sharma Mailthody | NVIDIA Technical Blog`. **Vikram Sharma Mailthody is a real NVIDIA author**, but the article links on that page are: *Restore LLM Inference Capacity in Seconds with Shadow Engine Recovery in NVIDIA Dynamo*; *DynoSim: Simulating the Pareto Frontier*; *NVIDIA Dynamo Snapshot: Fast Startup for Inference Workloads on Kubernetes*; *NVIDIA Dynamo Adds GPU Autoscaling, Kubernetes Automation, and Networking Optimizations*. **None** is about batch invariance, and none is a vLLM blog post.
- **Conclusion:** the premise "(2) a vLLM blog post by Vikram Sharma about batch invariance" is **false**. The name likely conflates **Vikram Sharma Mailthody** (NVIDIA Dynamo blogs) with the **vLLM batch-invariance work** (Bram Wasti / Wentao Ye / vLLM team) — the latter being documented, not blogged, on vLLM's side. Do **not** cite a "Vikram Sharma vLLM batch invariance blog post" — no such URL exists.

### 3.2 Fetches that failed outright
| URL | Result |
|-----|--------|
| `https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference.md` | **HTTP 404** (17883 bytes error page) |
| `https://vllm.ai/blog-assets/markdown/2025-11-10-bitwise-consistent-train-inference.md` | **HTTP 404** (29511 bytes error page) |
| `https://github.com/vllm-project/vllm/pull/55958` | **timed out twice at 60 s**; no HTTP code obtained. Title only, from the repo-scoped issue-search listing. |
| `https://forums.developer.nvidia.com/t/why-nemotron-3-nvfp4-models-are-not-deterministic-using-vllm/372685` (HTML, first attempt) | **timed out at 60 s**; succeeded on retry with `--max-time 40`. The Discourse `.json` endpoint (same thread, +`.json`) returned **200** and is the source of §2's forum quotes. |

There is no "View Markdown Source" raw asset for vLLM blog posts at either guessed path; the post's own "View Markdown Source" link points to `https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2025-11-10-bitwise-consistent-train-inference.md`, which I verified exists by filename in the `_posts` tree listing but did **not** fetch.

### 3.3 Not retrieved / not attempted — listed as UNVERIFIED
- **No official NVIDIA developer-blog post on "batch invariance" or on the `VLLM_BATCH_INVARIANT` env var was found.** The only *official NVIDIA* determinism doc I could retrieve is the NIM/TensorRT-LLM page (row 10, `NIM_FORCE_DETERMINISTIC`). NVIDIA's `developer.nvidia.com/blog/?s=batch+invariance` → **200** (150732 B) but returned **zero** parseable article links (the search page is JS-driven). **Treat "NVIDIA has an official batch-invariance blog" as unverified/absent.**
- `https://deepwiki.com/vllm-project/vllm/4.6-determinism-and-batch-invariance` — appeared in `web_search` results; **not fetched, not verified, and not an official doc**. Do not cite.
- `https://docs.sglang.io/advanced_features/deterministic_inference.html` — parent reports **200 / REAL**; I did **not** fetch it myself. Not independently confirmed by me.
- `https://docs.vllm.ai/en/v0.19.0/features/batch_invariance/` — parent reports **200 / REAL**; I confirmed the equivalent at `/en/latest/` and `/en/stable/` (rows 2–3), not the `v0.19.0` pin.
- **`api.github.com`** — deliberately not used (project rule: rate-limited to zero). All GitHub data came from HTML/embedded JSON.
- **`openreview.net`** (forum = 4787-byte browser challenge; pdf = HTTP 403) and **`export.arxiv.org`** (429) — parent-verified blocked; **I did not attempt them, and this sweep cites no arXiv IDs or OpenReview material at all.**
- **llm-d**: searched `https://github.com/llm-d/llm-d/issues?q=determinism&state=all` → **200**, **zero** results; `q=nondeterministic` → **200**, zero; `q=batch+invariance` → **200**, zero; `q=reproducib` → **200**, zero. **Sanity-checked the parser** with `q=inference` → **200** and 11+ real hits, so these negatives are genuine, not a scraping artifact. **llm-d has no determinism issues in its tracker.**
- SGLang issue/Pull bodies for #36174, #38160, #38176, #39319 and vLLM PRs #55958, #56244, #56377, #56528 are **title-only** (from search listings); their bodies were not read.

### 3.4 Weakness disclosure on the forum evidence (row 12)
The NVIDIA Developer Forum thread is **user-generated**, not NVIDIA official documentation, and it contains at least one technically incorrect community reply:
> "But most importantly floating point calculation by the gpu will always produce slightly different results for the same input values." — post 4, user `mangosq`

That claim is **contradicted by the Thinking Machines post** (§2.5: *"running the same matrix multiplication on the same data repeatedly will always provide bitwise equal results"*) and by §2.4's `enable_sp=False` baseline of `0 / 1536`. Cite the thread only as evidence that a **user on NVFP4 + `VLLM_BATCH_INVARIANT=1` + temperature 0 observes nondeterminism**, not as an authoritative mechanism claim.

---

## 4. Exact search queries and URLs tried

### 4.1 `web_search` queries (discovery only — every cited primitive was re-fetched by curl and title-checked)
1. `vLLM blog post batch invariance Vikram Sharma`
2. `vllm.ai blog batch invariant kernels determinism`
3. `"batch invariance" vLLM NVIDIA developer blog deterministic inference`
4. `VLLM_BATCH_INVARIANT environment variable deterministic NVIDIA`
5. `"Vikram Sharma" batch invariance vLLM blog post`
6. `"Vikram Sharma" deterministic LLM inference kernel`
7. `NVIDIA developer blog deterministic inference LLM batch invariance repeatable`
8. `developer.nvidia.com blog reproducible deterministic LLM inference kernels`

### 4.2 Exact URLs fetched (with status)
**vLLM docs / source**
- `https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/batch_invariance.md` → **200**
- `https://docs.vllm.ai/en/latest/features/batch_invariance.html` → **200** (effective `/en/latest/features/batch_invariance/`)
- `https://docs.vllm.ai/en/stable/features/batch_invariance.html` → **200** (effective `/en/stable/features/batch_invariance/`)
- `https://raw.githubusercontent.com/vllm-project/vllm/main/docs/usage/reproducibility.md` → **200**
- `https://docs.vllm.ai/en/latest/configuration/env_vars.html` → **200**

**vLLM blog**
- `https://blog.vllm.ai/` → **200** (effective `https://vllm.ai/blog`)
- `https://vllm.ai/blog` → **200**
- `https://vllm.ai/blog/rss.xml` → **200**
- `https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference` → **200**
- `https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference.md` → **404**
- `https://vllm.ai/blog-assets/markdown/2025-11-10-bitwise-consistent-train-inference.md` → **404**
- `https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts` → **200**
- `https://github.com/vllm-project/vllm-project.github.io/issues?q=invariance&state=all` → **200**

**GitHub issues / PRs**
- `https://github.com/vllm-project/vllm/issues?q=batch+invariance&state=all` → **200**
- `https://github.com/vllm-project/vllm/issues/25404` → **200** (kernel dispatch overrides; located via search, not analysed in this sweep)
- `https://github.com/vllm-project/vllm/issues/27433` → **200**
- `https://github.com/vllm-project/vllm/issues/56370` → **200**
- `https://github.com/vllm-project/vllm/pull/55958` → **timeout ×2, no HTTP code**
- `https://github.com/sgl-project/sglang/issues?q=determinism&state=all` → **200**
- `https://github.com/sgl-project/sglang/issues?q=nondeterministic&state=all` → **200**
- `https://github.com/sgl-project/sglang/issues?q=deterministic&state=all` → **200**
- `https://github.com/sgl-project/sglang/issues?q=batch+invariance&state=all` → **200**
- `https://github.com/sgl-project/sglang/issues/39235` → **200**
- `https://github.com/sgl-project/sglang/issues/37451` → **200** (CI failure tracker; not determinism-relevant, discarded)
- `https://github.com/sgl-project/sglang/issues/36291` → **200**
- `https://github.com/llm-d/llm-d/issues?q=determinism&state=all` → **200**, 0 results
- `https://github.com/llm-d/llm-d/issues?q=nondeterministic&state=all` → **200**, 0 results
- `https://github.com/llm-d/llm-d/issues?q=batch+invariance&state=all` → **200**, 0 results
- `https://github.com/llm-d/llm-d/issues?q=reproducib&state=all` → **200**, 0 results
- `https://github.com/llm-d/llm-d/issues?q=inference&state=all` → **200** (parser sanity check, real hits)

**NVIDIA**
- `https://docs.nvidia.com/nim/large-language-models/1.12.0/deterministic-mode.html` → **200**
- `https://developer.nvidia.com/blog/author/vmailthody/` → **200**
- `https://developer.nvidia.com/blog/?s=batch+invariance` → **200**, 0 parseable article links

**PyTorch**
- `https://docs.pytorch.org/docs/stable/notes/randomness.html` → **200** but **1350-byte client-side redirect stub**, `<title>Redirecting&hellip;</title>`, target `../../2.14/notes/randomness.html`
- `https://pytorch.org/docs/stable/notes/randomness.html` → **200**, also the redirect stub
- `https://docs.pytorch.org/docs/2.14/notes/randomness.html` → **200** (canonical target)
- `https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html` → **200**

**Other**
- `https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/` → **200** (discovered as an outbound citation from the vLLM blog post and from vLLM issue #27433)
- `https://forums.developer.nvidia.com/t/why-nemotron-3-nvfp4-models-are-not-deterministic-using-vllm/372685` → **timeout ×1, then 200**
- `https://forums.developer.nvidia.com/t/372685.json` → **200**

---

## 5. Single strongest verbatim quote about determinism (decision-level)

From **vLLM issue #56370** (`HTTP 200`, title read back: *"[Bug]: Batch invariance is broken when sequence parallelism / async TP is enabled (`VLLM_BATCH_INVARIANT=1` + `pass_config.enable_sp`)"*):

> "With tensor parallelism it holds as long as sequence parallelism is **off**, but as soon as `pass_config.enable_sp=True` is set (or `fuse_gemm_comms=True`, which implies it), the same request produces different logprobs — and with greedy sampling different tokens — depending on how many other requests are in the batch."

This is the strongest because it is (a) a first-party vLLM bug report against the **official `VLLM_BATCH_INVARIANT=1` flag**, (b) explicitly **decision-level** — "different tokens" under greedy sampling, not a perplexity or logit-distance proxy, and (c) quantified in the same body (`"prompts whose greedy output changed 17 / 64"` versus a clean `"0 / 64"` baseline without SP).

For a **documentation-level** claim rather than a bug report, the strongest is the vLLM feature doc itself:

> "Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch."

…paired with its own caveat in the same document:

> "Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility."
