# Speculative-decoding drafting heads (EAGLE / HASS / Medusa / Hydra / MTP / DFlash2) — prior-art & gap map

Sweep date **2026-09-13**. Window prioritised 2025-06 → 2026-09; older work only where canonical.

**Legend** — `READ BODY` = I retrieved and read the page/abstract/body text. `TITLE ONLY` = I saw only a title or snippet (fetch failed, was rate-limited, or the page rendered without its body). No row appears without a URL.

**Environment facts verified this session:** `arxiv.org/abs/` and `arxiv.org/html/` work via `curl` (proxies unset). `export.arxiv.org` API = HTTP 429 "Rate exceeded" (unusable). `raw.githubusercontent.com` and `huggingface.co` are blocked to `curl` (HTTP 000) but work via `web_fetch`. `github.com` HTML works via `curl`; issue/PR **body and comment text is extractable** from the embedded HTML for open/closed threads, but GitHub's REST API is rate-limited to HTTP 403. `web_search` returned **fabricated arXiv IDs** — every ID below was independently probed against `arxiv.org/abs/` and the verified title is reported.

---

## A. CLOSED (someone shipped or published a working answer)

| # | Question it answers | Who closed it | Artifact | URL | Status | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|---|
| A1 | Full official method registry for speculative decoding in vLLM | vLLM | `SpeculativeMethod` literal: `ngram, medusa, mlp_speculator, draft_model, suffix, custom_class, eagle, eagle3, extract_hidden_states, <MTP types>, dflash, dspark, ngram_gpu` | https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py | Shipped | READ BODY |
| A2 | Exact vLLM config for `method:"eagle3"` | vLLM | `{"model":"RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3","draft_tensor_parallel_size":2,"num_speculative_tokens":2,"method":"eagle3"}` | https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle.html | Shipped | READ BODY |
| A3 | Exact vLLM config for `method:"mtp"` | vLLM | `{"method":"mtp","num_speculative_tokens":1}` | https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp.html | Shipped | READ BODY |
| A4 | Medusa is a first-class vLLM method (but undocumented) | vLLM | `"medusa"` in registry + `MedusaModel` + `MedusaProposer`; **no docs page exists** | https://github.com/vllm-project/vllm/tree/main/docs/features/speculative_decoding | Shipped, undocumented | READ BODY |
| A5 | SGLang algorithm flag values | SGLang | `--speculative-algorithm`: `UNO, DFLASH, EAGLE, EAGLE3, STANDALONE, NGRAM, NEXTN` (`NEXTN` = alias of `EAGLE`) | https://docs.sglang.io/advanced_features/speculative_decoding.html | Shipped | READ BODY |
| A6 | SGLang escape hatch for BF16 drafter on a quantized target | SGLang | `--speculative-draft-model-quantization unquant` (default = "Same as target") | https://docs.sglang.io/advanced_features/speculative_decoding.html | Shipped | READ BODY |
| A7 | **Qwen3-Next MTP in vLLM** | Qwen + vLLM | `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct | Shipped | READ BODY |
| A8 | **Qwen3-Next MTP in SGLang** | Qwen + SGLang | `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct | Shipped | READ BODY |
| A9 | DFlash2 merged upstream into vLLM | vLLM (@SubSir) | PR #52816 "[Spec Decode] DFlash2: local convolution + candidate selector" — **MERGED 2026-08-21T05:27:22Z by WoosukKwon**, merge SHA `b389ac29465b33f9e9c534df221ea3c129e9793f` | https://github.com/vllm-project/vllm/pull/52816 | Merged | TITLE ONLY (state/timestamps read from page JSON; body not rendered) |
| A10 | DFlash2 external drafter end-to-end recipe | Inco AI / z-lab | vLLM `{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}`; SGLang `--speculative-algorithm DFLASH --speculative-num-draft-tokens 8` | https://inco.ai/blog/dflash2/ | Shipped 2026-08-18 | READ BODY |
| A11 | DFlash2 design + measured gain | Inco AI | "+20% more output from every verification pass, for around 1% added cycle latency" | https://inco.ai/blog/dflash2/ | Published | READ BODY |
| A12 | EAGLE-3 canonical drafter + trainer | SafeAILab / Yuhui Li | `yuhuili/EAGLE3-LLAma3.1-Instruct-8B`; "trained using SpecForge or NVIDIA TensorRT Model-Optimizer" (per Ex0bit card) | https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B | Shipped | READ BODY |
| A13 | EAGLE-3 drafter for Qwen3.6-27B | Ex0bit | `Ex0bit/Qwen3.6-27B-PRISM-EAGLE3` — "a small (~0.6 B trainable) draft head"; `compressed/` = 1.1 GB "recommended for serving — fastest" | https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3 | Shipped | READ BODY |
| A14 | DFlash2 drafter requantised to W4A16 | syvai | 1.92 B / 3.85 GB bf16 → 1.19 GB W4A16 | https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16 | Shipped | READ BODY |
| A15 | Native MTP-head fine-tuning without loading the verifier | vLLM project | `MTPConverter` → finetune → stitch; MTP head "~100M–400M params" | https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp.html | Shipped | READ BODY |
| A16 | Single-GPU head training is a supported entry point | vLLM project | "Supports single-GPU and multi-GPU distributed training." / "**Single GPU:** drop the `torchrun` wrapper and call `speculators train` directly" | https://docs.vllm.ai/projects/speculators/en/latest/cli/train.html | Shipped | READ BODY |
| A17 | Offline EAGLE-3 training fits in **1 GPU** | SGLang / SpecForge | "Offline | … | **as low as 1 GPU**, as only need to accommodate the draft model" | https://raw.githubusercontent.com/sgl-project/SpecForge/9b8873dfb03ed3bc10b105249f8ec6092e01b574/docs/basic_usage/training.md | Shipped | READ BODY |
| A18 | Head-training frameworks and which heads they cover | SGLang / vLLM / NVIDIA | SpecForge: EAGLE3, EAGLE3.1, P-EAGLE, DFlash, DFlash2, Domino, DSpark. Speculators: EAGLE-3, P-EAGLE, DFlash, DFlash2, DSpark, MTP finetune. NeMo AutoModel: `train_eagle1/2/3`, `TrainDFlash/DFlash2/Domino/JetSpecRecipe` | https://github.com/sgl-project/SpecForge · https://github.com/vllm-project/speculators | Shipped | READ BODY |
| A19 | Qwen3.6 MTP is officially recommended on 8-GPU TP | Qwen | `vllm serve Qwen/Qwen3.6-27B --tensor-parallel-size 8 … --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`; SGLang `--tp-size 8 … --speculative-algo NEXTN` | https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/raw/main/README.md | Shipped | READ BODY |
| A20 | MTP head count vs acceptance was formally quantified | FastMTP authors | vanilla MTP acceptance 70 % → 10 % → ~0 % at k=1/2/3; fine-tuned 80/56/36 % | https://arxiv.org/abs/2509.18362 | Published 2025-09-16 | READ BODY |
| A21 | EAGLE-3/hidden-state drafting theory | EAGLE-1/2/3 | 3.05–4.26× (EAGLE-2); up to 6.5× (EAGLE-3) | https://arxiv.org/abs/2401.15077 · https://arxiv.org/abs/2406.16858 · https://arxiv.org/abs/2503.01840 | Published | READ BODY |
| A22 | HASS = "Learning Harmonized Representations for Speculative Sampling" (correct title for 2408.15766) | HASS authors | ICLR 2025; 2.81–4.05× | https://arxiv.org/abs/2408.15766 | Published | READ BODY |
| A23 | Medusa / Hydra canonical designs | FasterDecoding / Together | Medusa 2.2–3.6×; Hydra sequential draft heads, up to 1.31× over Medusa | https://arxiv.org/abs/2401.10774 · https://arxiv.org/abs/2402.05109 | Published | READ BODY / READ ABSTRACT |
| A24 | DeepSeek-V3 MTP module size and role | DeepSeek-AI | "671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights" | https://huggingface.co/deepseek-ai/DeepSeek-V3 | Shipped | READ BODY |
| A25 | EAGLE-3 long-context drafter alternative | OWL authors | LSTM drafter gives ~5× higher acceptance length than EAGLE-3 on long context | https://arxiv.org/abs/2510.07535 | Published 2025-10-08 | READ BODY |
| A26 | SGLang EAGLE-3 capture for Qwen3-Next became native | SGLang | `qwen3_next.py` now contains the capture path (basis for closing PR #10657 as superseded) | https://github.com/sgl-project/sglang/pull/10657 | Closed as superseded 2026-06-10 | READ BODY (title + embedded state + closing comment) |
| A27 | Prefix-cache default regression for Mamba+EAGLE fixed | vLLM (@ZJY0516) | PR #55760 "[Core] Default prefix_cache_retention_interval to dense for Mamba + EAGLE" MERGED 2026-09-08; PR #55861 "[Bugfix][Core] Apply dense prefix cache default to hybrid models" MERGED 2026-09-08 | https://github.com/vllm-project/vllm/pull/55760 · https://github.com/vllm-project/vllm/pull/55861 | Merged | TITLE ONLY (state/timestamps from page JSON) |

---

## B. OPEN (explicitly unsolved, with evidence someone is still asking)

**B1 — Does an EAGLE/MTP head survive a quantized target without silent acceptance loss?**
- **Who is asking:** vLLM issue reporter; the Nvidia/community quant ecosystem.
- **URL:** https://github.com/vllm-project/vllm/issues/26402
- **What's missing:** a guard that detects the misconfiguration at load time instead of silently serving at 0 % acceptance.
- **VERBATIM:** *"When using an EAGLE head with a compressed tensors quantized model the acceptance rate silently fails to zero and therefore the perf completely degrades."* … *"EAGLE layers are registered as further layers in the target model. But in a compressed tensors checkpoint those are not part of the ignore list. Hence vLLM thinks the EAGLE layers are quantized, but they aren't. Surprisingly this does not give an error, but the acceptance rate essentially drops to zero…"* … *"The bigger problem IMO is that the EagleProposer simply takes the VllmConfig from the target model. This is not robust whenever the draft model has some different configurations."* — READ BODY.

**B2 — Why does EAGLE-3 acceptance collapse to 0 % at 262K but work at 32K?**
- **Who is asking:** vLLM issue #37773.
- **URL:** https://github.com/vllm-project/vllm/issues/37773
- **What's missing:** root cause in vLLM's EAGLE-3 verification/coordination logic — two architecturally unrelated heads fail identically.
- **VERBATIM:** *"EAGLE-3 speculative decoding acceptance rate progressively collapses to 0% during generation when `--max-model-len 262144`."* … *"The same model + EAGLE-3 head works perfectly at `--max-model-len 32768`, achieving 36.5% overall acceptance and +43% throughput improvement."* … *"Since they have completely different architectures and RoPE strategies, the bug is in vLLM's EAGLE-3 verification/coordination logic, not in the draft models."* — READ BODY.

**B3 — Native MTP loses its RoPE/YaRN config beyond the trained window**
- **Who is asking:** vLLM issue #37435.
- **URL:** https://github.com/vllm-project/vllm/issues/37435
- **What's missing:** the draft model's config must inherit the target's `--hf-overrides`; it currently does not.
- **VERBATIM:** *"the draft acceptance rate can collapse to ~0% while the target model still appears usable."* … *"beyond original context size, draft starts proposing nonsense / verifier rejects everything"* … *"I believe speculative/MTP draft config is not inheriting target-model `--hf-overrides`, which can break long-context speculation when RoPE/YaRN scaling is injected at runtime rather than baked into the checkpoint config."* — READ BODY.

**B4 — MTP × prefix-caching recurrent-state corruption on Qwen3-Next hybrids**
- **Who is asking:** vLLM issue #43559 (reporter reproduced ~20 % accuracy drop on 35B-A3B); fix PR #48375 by @potto007 is **still OPEN** (created 2026-07-12, 2 commits, `mergeCommitSha: null`).
- **URLs:** https://github.com/vllm-project/vllm/issues/43559 · https://github.com/vllm-project/vllm/pull/48375
- **What's missing:** `MambaManager.find_longest_cache_hit` receives `drop_eagle_block` and silently ignores it. club-3090 re-verified on v0.29.0 that the parameter is *still signature-only*: *"`vllm-pr48375-mamba-drop-eagle-block` (**still REQUIRED** — `drop_eagle_block` is STILL signature-only in v0.29.0's `MambaManager.find_longest_cache_hit`, so vllm#43559 is live)"* — READ BODY (club-3090 `docs/UPSTREAM.md`).
- **Title verbatim:** *"[Bug]: Accuracy drops ~20% when `--enable-prefix-caching` is used together with MTP speculative decoding (Qwen3.6 35B-A3B)"*.

**B5 — Unbounded `num_accepted` indices crash the GDN spec-decode path**
- **Who is asking:** vLLM PR #50021 (@amittell), **OPEN** since 2026-07-27, `mergeable_state=blocked`.
- **URL:** https://github.com/vllm-project/vllm/pull/50021
- **What's missing:** the fix is not merged and not inherited by any pin. club-3090's independent re-verification: the unbounded load is *still* present on v0.27.1 at `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:106`, and the old file path now yields a false negative. VERBATIM: *"⚠️⚠️ **RE-VERIFIED ON v0.27.1 2026-08-16 — STILL UNBOUNDED, AND THE PATH MOVED. THE OLD CHECK NOW SILENTLY READS AS 'FIXED'.**"* — READ BODY (club-3090 `docs/UPSTREAM.md`).
- **Title verbatim:** *"[Bugfix] Bound accepted-token state lookups in GDN/KDA spec decode"*.

**B6 — DFlash fused-KV projection silently corrupts weight-quantized drafters**
- **Who is asking:** vLLM issue #51581 (OPEN since 2026-08-09).
- **URL:** https://github.com/vllm-project/vllm/issues/51581
- **What's missing:** `qwen3_dflash.py` has zero quantization awareness; a corrupt fused-KV path presents as "merely a slow drafter", so boot success is not evidence.
- **VERBATIM (title):** *"[Bug][Spec Decode]: DFlash fused-KV projection calls F.linear on a sliced qkv_proj weight — breaks (and can silently corrupt) any weight-quantized drafter"*. club-3090 adds: *"Boot-success is NOT sufficient evidence here: a corrupt fused-KV path presents as a merely slow drafter, so **read accept-len**"* — READ BODY.

**B7 — Qwen3-Next MTP cannot use 'all' prefix caching**
- **Who is asking:** vLLM source itself (in-tree limitation comment).
- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py
- **VERBATIM:** *"Qwen3NextMTP currently does not support 'all' prefix caching,"* — READ BODY.

**B8 — MTP depth is warned against in vLLM's own config code**
- **Who is asking:** vLLM maintainers (runtime warning shipped in `vllm/config/speculative.py`).
- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py
- **VERBATIM:** *"Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate"* — READ BODY.

**B9 — Adaptive verification is restricted to a single method**
- **Who is asking:** vLLM docs.
- **URL:** https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification.html
- **VERBATIM:** *"Adaptive verification needs per-position acceptance estimates, so today it is only supported for DSpark with a **confidence head**."* — READ BODY.

**B10 — Medusa has no official documentation anywhere**
- **Who is asking:** (documentation gap; zero "medusa" hits under `vllm/docs/`).
- **URL:** https://docs.vllm.ai/en/latest/features/speculative_decoding.html
- **What's missing:** no flag schema, no supported-model list, no limitation statement. SGLang's spec-decoding page never mentions Medusa. — READ BODY.

**B11 — `mlp_speculator` is documented but unusable on vLLM V1**
- **Who is asking:** vLLM issue #47825 (@fenghourun), CLOSED AS NOT PLANNED 2026-07-07.
- **URL:** https://github.com/vllm-project/vllm/issues/47825
- **VERBATIM:** *"Net: mlp_speculator is documented/exemplified but unusable on V1, with a confusing error deep in init. Happy to send a fix: fail fast at config validation with a clear message (and fix the stale docs/example), or wire an MLP proposer into V1. Could a maintainer confirm which is preferred?"* — READ BODY. **Title:** *"[Bug]: mlp_speculator speculative method is unrouted in V1 and crashes at engine init (ValueError: Unknown speculative decoding method)"*.

**B12 — MTP × TurboQuant × CUDA-graph still degenerate on Qwen3-Next**
- **Who is asking:** vLLM issue #40880 (OPEN; filed 2026-04-25 by noonghunna).
- **URL:** https://github.com/vllm-project/vllm/issues/40880
- **What's missing:** the ngram-scoped fixes do not cover the MTP proposer path. VERBATIM: *"the MTP forward path goes through a different proposer (`EagleProposer` configured with `method="mtp"`) that the ngram-scoped fixes don't cover."* Also quotes a maintainer: *"we did not test MTP at all in the v7.13 cycle... your data shows that assumption is wrong."* — READ BODY.

**B13 — DFlash2 on llama.cpp and Ollama is still on unmerged PR branches**
- **URL:** https://inco.ai/blog/dflash2/
- **VERBATIM:** *"git fetch origin pull/27342/head:pr-27342"* (llama.cpp) and *"git fetch origin pull/17865/head:dflash2"* (Ollama). — READ BODY.

**B14 — EAGLE-3 on llama.cpp needs unmerged WIP patches**
- **URL:** https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ
- **VERBATIM:** *"EAGLE-3 chain (needs the WIP PR #18039 patches + the RS-rollback fix — a one-shot llama.cpp patch script is documented alongside the drafter)"* — READ BODY.

**B15 — SpecForge admits acceptance is unvalidated on a real server**
- **URL:** https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md
- **VERBATIM:** *"loading the result in a real speculative-decoding server and measuring acceptance remains a GPU-serving validation step."* — READ BODY.

**B16 — No published head-training cost for Medusa/DFlash2/DSpark**
- **URLs:** https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3 · https://github.com/sgl-project/SpecForge
- **What's missing:** Medusa's card gives inference scope only; SpecForge lists DFlash2/DSpark as trainable with **no GPU count and no wall-clock**. — READ BODY.

**B17 — EAGLE-1/2/3 have no Limitations or Future Work section at all**
- **URLs:** https://arxiv.org/abs/2401.15077 · https://arxiv.org/abs/2406.16858 · https://arxiv.org/abs/2503.01840
- **Finding (negative):** the strings "future work"/"future" do not appear in the EAGLE-1/2/3 arXiv HTML bodies; no section titled "Limitations" exists in any of the three. Reported as absent rather than inferred. — READ BODY.

---

## C. ATTEMPTED-AND-ABANDONED

**C1 — SGLang + EAGLE-3 on Qwen3-Next: built, booted, then PARKED**
- **Who tried:** club-3090 (`noonghunna/club-3090`). Two vendored patches, dual-3090 TP=2, reached "boot + coherent output" 2026-05-20.
- **URLs:** https://github.com/noonghunna/club-3090/blob/master/docs/engines/SGLANG.md · https://github.com/noonghunna/club-3090/blob/master/models/qwen3.6-27b/sglang/README.md · tag copy: https://raw.githubusercontent.com/noonghunna/club-3090/refs/tags/v0.10.2/docs/engines/SGLANG.md
- **VERBATIM:** *"**Status (2026-05-21): PARKED.** Not currently a shipped variant on this stack for Qwen3-Next. Three independent findings drove the decision:"* … *"**EAGLE-3 is sub-MTP for Qwen3-Next, even on Blackwell where it works.** Ex0bit's own published numbers on the PRISM-PRO-DQ model card report native MTP = **121 TPS (1.51×)** vs EAGLE-3 chain = **111 TPS (1.39×)**. The model family has a strong built-in MTP head; routing through an external drafter is structurally slower."* … *"Three patch iterations (pre-import; sys.modules stub at engine init; per-process sys.modules stub at `sglang/__init__.py`) all failed — the walk re-fires during capture regardless of cache state."* … *"The path is parked because EAGLE-3 < MTP for Qwen3-Next (point 1 above) is a structural finding, not a fixable bug."* — READ BODY. **Note:** the doc title in the current `master` tree is literally *"# SGLang — Qwen3-Next EAGLE-3 path PARKED; no shipped variant on this stack"*. Confirmed identical at tag `v0.10.2`. **Sibling docs checked:** `docs/engines/VLLM.md` (MTP is the shipped path), `docs/engines/LLAMA_CPP.md`, `docs/engines/IK_LLAMA.md` (single-card MTP), `docs/UPSTREAM.md` (the bug ledger).

**C2 — EAGLE-3 external drafter loses to MTP on the *same* llama.cpp rig (head-capability-limited)**
- **Who tried:** club-3090, same-session single-variable A/B.
- **URL:** https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md (the `mtp.yml` b9967 re-bench row)
- **VERBATIM:** *"Same-session **EAGLE3-external A/B** (self-converted `club-b9967` Q4_K_M draft, n=2, only draft+spec-type changed): **47.5 / 61.3 wall, accept 0.408** → −25% narr / −17% code — **MTP stays; no EAGLE3 compose.** Mechanism note: EAGLE3 accepts ~0.40 BOTH hidden-state-fed (vLLM, #662) and token-fed (here) → head-capability-limited, feed-robust; MTP is the reverse (0% hidden-state vs 0.74 token-fed → alignment-limited, capability-strong). A retrained/stronger head flips this — artifact kept on disk for the re-test."* — READ BODY.

**C3 — vLLM PR #40914 (TurboQuant K+1 spec-verify): explicit negative result, kept only as a re-test artifact**
- **Who tried:** club-3090, vendored 2026-05-11.
- **URL:** https://github.com/noonghunna/club-3090/blob/master/models/qwen3.6-27b/vllm/patches/vllm-pr40914-k1-only/README.md · upstream https://github.com/vllm-project/vllm/pull/40914 (**still OPEN**, created 2026-04-26)
- **VERBATIM:** *"## Status: negative result on Qwen3.6-27B"* … *"With this overlay active, MTP acceptance stabilizes at AL=4.0 / ~100%, but outputs collapse into `!` floods and tool/multi-turn paths time out."* … *"Skipping `mtp.*` drafter layers does not fix the corruption."* … *"Keep this only as a re-test artifact. Do not mount it in shipping composes unless upstream changes the PR into a true P67-equivalent multi-query TurboQuant spec-decode fix."* … *"Round 3 result: this did not fix the corruption. The target-side path is also wrong for this stack."* — READ BODY.

**C4 — MTP acceptance collapse: closed as NOT PLANNED, blamed on the checkpoint export**
- **Who tried / who closed:** filed by @noonghunna, 2026-08-19; **CLOSED AS NOT PLANNED**.
- **URL:** https://github.com/vllm-project/vllm/issues/52873
- **Title VERBATIM:** *"[Bug]: Qwen3-Next GDN + MTP: crossing sequence position 32768 permanently kills draft acceptance engine-wide (0% until restart)"*. Page state verbatim: *"Closed as not planned"*. — READ BODY (state + title + body).
- **The filer's own conclusion (verbatim, from their repo ledger):** *"🟢 **Closed by us — NOT a vLLM bug (checkpoint-specific)**, 2026-08-20"* … *"the permanent 0% collapse is specific to the `Avuja/Qwen3.8-27B-int4-AutoRound` weights on the MTP-proposer path"* — https://github.com/noonghunna/club-3090/blob/master/docs/UPSTREAM.md — READ BODY.
- **Corroborating row (verbatim):** *"Avuja's built-in MTP drafter permanently collapsed to 0% acceptance at ~14.5k cumulative gen tokens; Frozenlock (same auto-round 4/g128/sym recipe, different export, quantized MTP head) runs the drafter clean past 22–24k… Checkpoint-specific, NOT a vLLM bug (FP8 + ngram both immune; vllm#52873 closed not_planned)."* — https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md — READ BODY.

**C5 — vLLM PR #34163 (MLPSpeculator fix): stale-closed by its own author after 195 days**
- **URL:** https://github.com/vllm-project/vllm/pull/34163 — **CLOSED unmerged 2026-08-23T21:44:24Z**, created 2026-02-09, 1 commit, `mergeCommitSha: null`.
- **VERBATIM (author @Mr-Neutr0n):** *"Closing — no maintainer activity for 195 days and the repo has moved on. Happy to reopen if the fix is still wanted."* — READ BODY.

**C6 — vLLM PR #40898 (DFlash sliding-window drafter): closed as SUPERSEDED, with the divergence recorded**
- **URL:** https://github.com/vllm-project/vllm/pull/40898 — **CLOSED unmerged 2026-07-14T17:35:47Z**, created 2026-04-26, 10 commits, `mergeCommitSha: null`.
- **VERBATIM (@jschmied):** *"Follow-up to my backport comment above — this PR looks superseded, and the difference in approach is worth recording for anyone who lands here. Hybrid SWA/full DFlash drafters are already on main: #47914 ("[Spec Decode] Support hybrid (SWA + full attention) DFlash drafters", 0d12618), plus the follow-up #48113."* — READ BODY.
- club-3090's own negative finding for the same blocker: *"stock-vLLM DFlash hard-blocked on hybrid `layer_types` (vllm#40898, `NotImplementedError` at model build)"* — READ BODY.

**C7 — SGLang PR #10657 (EAGLE-3 for Qwen3-Next): 9 months open, then closed as superseded**
- **URL:** https://github.com/sgl-project/sglang/pull/10657 — created **2025-09-19T09:18:13Z**, **CLOSED 2026-06-10T08:13:39Z**, `mergedBy: null`, `mergeCommitSha: null`, 3 commits, head branch `support-qwen3_next-eagle3`, author @AnnaYue.
- **VERBATIM (closing comment):** *"! EAGLE3 capture for qwen3-next is now native ( qwen3_next.py ), so I'm closing this as superseded. Please reopen if something's still missing."* — READ BODY.

**C8 — SGLang PR #20370 (Qwen3.5 AWQ Marlin repack): closed unmerged after 5 months, root cause never landed upstream**
- **URL:** https://github.com/sgl-project/sglang/pull/20370 — created 2026-03-11, **CLOSED 2026-08-10T03:54:21Z**, 1 commit, `mergeCommitSha: null`.
- **VERBATIM (author's own diagnosis):** *"The crash came from the Marlin repack path during quantized weight processing, in_proj_a/in_proj_b (width 32) were intended to be ignored by quantization (see quant config), but inherited name mapping caused ignore-name mismatch, so these layers were quantized by mistake and hit Marlin."* — READ BODY.
- club-3090 still carries its own local patch for this and lists the upstream merge as a drop trigger: *"SGLang upstream merges the AutoRound name-mapper fix (track `sgl-project/sglang#19406` + `#20370`) | We can drop our `patch_sglang_autoround_fused_bf16.py` vendor."* — READ BODY.

**C9 — vLLM PR #52216 shipped a regression that zeroed EAGLE/MTP Mamba cache hits**
- **URL:** https://github.com/vllm-project/vllm/pull/52216 — "Promote `prefix_cache_retention_interval` to an argument and change the default to 0", **MERGED 2026-08-17T13:20:50Z by @tlrmchlsmth** — then reverted-in-effect by #55760/#55861 on 2026-09-08.
- **club-3090 VERBATIM:** *"⛔⛔ **REGRESSION SHIPPED IN v0.28.0; FIX SHIPPED IN v0.29.0. v0.28.0 IS SKIPPED (maintainer, 2026-09-10).**"* … *"with sparse retention only the latest replay boundary keeps a Mamba state checkpoint, and EAGLE/MTP additionally drops the tail block from prefix-cache hits, so the checkpoints are largely unreachable and **the Mamba cache group never hits**. Upstream's production report was literally '0 hit tokens' ([#53504]); the fix moved an aggregate hit rate **0% -> 49.5%**."* … *"⭐ **#55760 ALONE IS NOT ENOUGH FOR OUR FAMILY** — it branches on `has_inner_state`, but Qwen3-Next declares `is_hybrid=True, has_inner_state=False`, so the branch is skipped."* — READ BODY.

**C10 — EAGLE-3 on-device (CoreML/ANE): benched, then acknowledged slower than baseline**
- **URL:** https://raw.githubusercontent.com/john-rocky/CoreML-LLM/refs/heads/main/docs/EAGLE3_INTEGRATION_STATE.md
- **VERBATIM:** *"Status: **Phase 2A + 2B done, Phase 3 benched, speculative currently slower than baseline.**"* … *"Phase 3 — iPhone 17 Pro bench | ⚠️ ran, **not faster than baseline 28.6 tok/s** (11–17 tok/s with fallback to T=1)"* … *"**Blocker 3 (11c): verify-vs-decode fp16 drift** — sets iPhone acceptance break-even at ~77%. Even a successful retrain landing at 50-60% would not produce a net speedup on device until 11c closes."* — READ BODY.

**C11 — Cross-model DFlash drafter on a fine-tuned target: fine-tune shift costs ~20 pp acceptance**
- **URL:** https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md
- **VERBATIM:** *"**DFlash alternatives ruled out same-day:** stock-vLLM DFlash hard-blocked on hybrid `layer_types` (vllm#40898, `NotImplementedError` at model build); cross-model Anbeeld qwen3.6 DFlash-Q4_K_M drafter on the llama.cpp lane attaches (dims match) but loses to Tess's own MTP GGUF (51.0/64.5 vs 63.2/66.9; accept 52–58% vs 72–85% — fine-tune shift costs ~20pp)."* — READ BODY. And: *"DFlash ruled out (only BASE-27B drafters exist → ~10% accept on the fine-tune; no Carnice-matched drafter)."* — READ BODY.

**C12 — MTP acceptance collapses at high n; the knee is real and reproducible**
- **URL:** https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md · https://github.com/noonghunna/club-3090/blob/master/docs/engines/VLLM.md
- **VERBATIM (vLLM engine doc):** *"n=3 is the empirical sweet spot. n=4 nominally hits higher TPS on code but 4th-position acceptance collapses to ~21%. Don't push higher."* — READ BODY.
- **VERBATIM (n-sweep, Tess W4A16):** *"n-sweep (single-variable boots, this ckpt): no-spec 70.1/70.0 → n=3 68.2/84.8 → **n=5 69.9/104.3 (knee: prose break-even, code +49%)** → n=6/8 REGRESS (83.8/84.5, tail accept 0.12–0.36)."* — READ BODY.
- **VERBATIM (cross-rig failure):** *"MTP accept only 49.1% on attended agent traffic (vs 80% reference warm-code) → n=5 past its knee → decode ~half reference"* — READ BODY.

**C13 — MTP refusal-sampling threshold violated by GGUF quantization → 0 % acceptance**
- **URL:** https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/discussions/32
- **Title VERBATIM:** *"UD-Q4_K_XL produces 0% MTP draft acceptance — blk.64.attn_*/ffn_* quantized below the threshold for MTP rejection sampling"* — TITLE ONLY (page fetched; discussion body did not render).

**C14 — vLLM silently skips a quantized MTP head → 0 % acceptance, no error**
- **URL:** https://raw.githubusercontent.com/devnen/qwen3.6-windows-server/v1.3.3/docs/MTP_HEAD.md
- **VERBATIM:** *"Multi-token prediction (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`) only works if the model weights ship an **MTP head in BF16**. The vLLM `Qwen3_5MTP` loader looks for tensors named `mtp.fc.*` and refuses (silently) to use them if they're quantised."* … *"The loader silently skips the quantised head, MTP runs, and you get **0 % draft acceptance**, no speedup, no error message."* … *"**If it's near 0.0, your quant's MTP head got silently skipped.**"* — READ BODY (via the docs sweep).

**C15 — PEFT-based block-diffusion drafting: published negative result**
- **URL:** https://arxiv.org/abs/2607.12422
- **VERBATIM:** *"Despite these advantages, PEFT-BD does not yield a practical speedup in our Qwen3-0.6B experiments."* … *"the drafter is parameter-efficient but not compute-efficient."* … *"the drafter must be substantially cheaper to execute than the verifier. Longer accepted prefixes alone cannot compensate when draft computation remains verifier-scale."* — READ BODY.

**C16 — KV-cache reuse to rescue long-range drafting: "the central negative result of this study"**
- **URL:** https://arxiv.org/abs/2604.26412
- **VERBATIM:** *"the hybrid drafter increases HF-measured MAT only from 5.01 to 5.04 (+0.6%). Our profiling further indicates that the additional cross-attention path introduces roughly 5–10% extra drafting latency. Taken together, these numbers do not support a meaningful end-to-end speedup claim in the current pipeline, and suggest that any wall-clock gain would be marginal at best."* — READ BODY.

**C17 — Meta's multi-token prediction: fine-tuning variant explicitly failed**
- **URL:** https://arxiv.org/abs/2404.19737
- **VERBATIM:** *"Finetuning LLama 2 with multi-token prediction does not significantly improve performance. We tried to finetune LLama 2 with 4-token prediction but this did not yield significant improvements compared to the baseline. We suppose that this new loss changes the initialization too brutally and never really recovers."* — READ BODY.

**C18 — Static draft-vocabulary pruning: no speedup, by its own accounting**
- **URL:** https://arxiv.org/abs/2605.27390
- **VERBATIM:** *"The static method FR-Spec (Static 32k) fails to achieve tangible speedups (1.00×) because the latency reduction in projection is strictly negated by the drop in acceptance rate."* — READ BODY.

**C19 — SGLang-JAX EAGLE-3: high acceptance, no throughput gain yet**
- **URL:** https://raw.githubusercontent.com/tails-mpt/SpecJAX/refs/heads/main/README.md
- **VERBATIM:** *"The sglang-jax EAGLE3 pipeline is functional (correct outputs, ~60–66% acceptance rates) but throughput gains are pending upstream optimization of the verify/tree-building path."* — READ BODY.

**C20 — AdaFlash: SGLang's GatedDeltaNet support blocks efficient variable-length verification**
- **URL:** https://arxiv.org/abs/2607.19223
- **VERBATIM:** *"We attribute this to an engineering limitation: SGLang's current support for the Gated DeltaNet architecture does not yet allow a fully efficient implementation of variable-length verification scheduling, so the adaptive length head incurs additional overhead under high-concurrency serving on this model."* — READ BODY.

**C21 — club-3090 ik_llama MTP slugs retired as non-functional**
- **URL:** https://github.com/noonghunna/club-3090/blob/master/docs/engines/IK_LLAMA.md
- **VERBATIM:** *"`ik-llama/iq4ks-mtp` — the engine's last functional slug — was retired 2026-08-12, following `iq4ks-mtp-vision` and `iq4ks-two-stage` earlier the same day, when the single-card qwen surface consolidated onto vLLM."* — READ BODY.

**C22 — vLLM DFlash: anchor sampling and CUDA-graph paths are explicitly not supported**
- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
- **VERBATIM:** *"sample_from_anchor=True is not supported for DFlash."* … *"%s draft attention (%s) does not support full CUDA graphs; running the draft eagerly."* … *"# PIECEWISE cudagraphs are not supported for dflash."* — READ BODY (via docs sweep).

**C23 — SpecForge: unsupported combinations are hard-rejected rather than degraded**
- **URL:** https://github.com/sgl-project/SpecForge
- **VERBATIM:** *"Unsupported combinations are rejected during config validation or run assembly instead of falling back to an older trainer."* — READ BODY.

**C24 — DFlash2 quantized-drafter calibration: the "theoretically tidier" variant measured worse**
- **URL:** https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16
- **VERBATIM:** *"A variant whose k/v Hessians also blended the context-KV precompute's input distribution — which is the theoretically tidier calibration — measured 7% *worse* greedy acceptance (3.12 vs 3.34 tokens per step, 118 vs 126 tok/s end to end) and is not what ships here."* — READ BODY (via docs sweep).

---

## D. HARDWARE-RULED-OUT for one RTX PRO 6000 Blackwell 96 GB (sm_120), single node

Anything requiring >1 GPU, >96 GB, or multi-node. Each row states the **verbatim** constraint.

| # | What is ruled out | Why (verbatim) | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|
| D1 | Production EAGLE-3 training for Llama-3.1-8B | *"\| Production \| Llama 3.1 8B Instruct \| **8x A100 80 GB** \| ~2 h (1 epoch, 200k samples) \|"* — 8 GPUs, 640 GB aggregate | https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/eagle.mdx | READ BODY |
| D2 | EAGLE canonical training claim | *"trainable (within 1-2 days) and testable on **8x RTX 3090 GPUs**. So even the GPU poor can afford it."* — 8 cards | https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B | READ BODY |
| D3 | Head-training GPU-hours on H200 | *"Training a single drafter requires **128 GPU-hours** for Llama-3.1-8B-Instruct, **288 GPU-hours** for GPT-OSS-20B, and **320 GPU-hours** for Qwen3-30B-A3B-Instruct. All computations are performed on **NVIDIA H200 GPUs**."* | https://arxiv.org/abs/2605.10453 | READ BODY |
| D4 | NeMo disaggregated DFlash training | *"torchrun --standalone --nproc_per_node=2 -m nemo_automodel.recipes.llm.train_dflash …"* — 2 processes minimum | https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/dflash.mdx | READ BODY |
| D5 | SpecForge checked-in 27B DFlash2 recipe | *"The checked-in Qwen3.6-27B recipe owns a **two-GPU local stack**: one configured GPU runs the target capture server and another runs the trainer."* | https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md | READ BODY |
| D6 | NeMo EAGLE recipe disk cost in online mode | *"GPU memory is therefore **dominated by the target model size**."* — the frozen verifier must fit alongside the drafter | https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/eagle.mdx | READ BODY |
| D7 | SpecForge offline EAGLE-3 (disk-heavy but GPU-light) | *"Offline \| Only used during data preparation \| **Huge (e.g. ultrachat+sharegpt will need 12TB storage)** \| **as low as 1 GPU**, as only need to accommodate the draft model"* — compute is fine, 12 TB of disk is the wall | https://raw.githubusercontent.com/sgl-project/SpecForge/9b8873dfb03ed3bc10b105249f8ec6092e01b574/docs/basic_usage/training.md | READ BODY |
| D8 | SpecForge v0.3.0 disaggregated training result | *"On our **8xH20 testbed**, a topology with **3 SGLang servers and 5 trainer workers** improves end-to-end training throughput by approximately **10%**"* | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | READ BODY |
| D9 | SpecForge reported draft-model serving evals | *"Qwen3.6-27B on **2 × A100** (TP2, BF16…)"; *"Qwen3.5-397B-A17B on **8 × B200** (TP8…)"; *"Kimi-K3 on **8 × B300**"* | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | READ BODY |
| D10 | Red Hat 70B EAGLE-3 speculator evaluation | *"hardware: **4xA100**"* | https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3 | READ BODY |
| D11 | TensorRT-LLM EAGLE-3 reference deployment | *"trtllm-serve <exported checkpoint> … --tp_size **8**"* — 8-way tensor parallel in the official example | https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/examples/post_training/modelopt/speculative.md | READ BODY |
| D12 | Speculators dataset regeneration | *"**Time required:** ~10 mins on **2x H100 GPUs** (for 1K samples)"* | https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/response_regeneration.html | READ BODY |
| D13 | Official Qwen3.6 serving recipes enable MTP at TP=8 | `vllm serve Qwen/Qwen3.6-27B --tensor-parallel-size 8 … --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` | https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/raw/main/README.md | READ BODY |
| D14 | DeepSeek-V3 MTP head in full precision | *"671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights"* — ~685 B params total, far beyond 96 GB at any usable precision | https://huggingface.co/deepseek-ai/DeepSeek-V3 | READ BODY |
| D15 | Meta's multi-token-prediction training campaign | *"In aggregate, training all models reported in the paper required around **500K GPU hours** of computation on hardware of type A100-80GB and H100."* | https://arxiv.org/abs/2404.19737 | READ BODY |
| D16 | club-3090's own EAGLE-3-on-Qwen3-Next path needed TP=2 | *"Dual 3090 (TP=2) + AutoRound INT4 + EAGLE-3 — boots + serves coherent output"*; and *"At 24 GB the target (~17 GB) + EAGLE-3 drafter (~3 GB) + Mamba state + KV cache leaves ~0-2 GB headroom."* | https://raw.githubusercontent.com/noonghunna/club-3090/refs/tags/v0.10.2/docs/engines/SGLANG.md | READ BODY |
| D17 | club-3090 DFlash2 TP=4 / NVLink row explicitly non-transferable | *"⚠️ **This is a 4-card NVLink rig** — the reference rig has 2 PCIe-only 3090s, so these do not transfer to the `dual` tiers."* | https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md | READ BODY |
| D18 | MTP finetune on Qwen3-Next-80B needed FSDP | 1 epoch, 7,473 GSM8K samples on **4 GPU FSDP** — i.e. >1 GPU for the 80B target | https://huggingface.co/inference-optimization/Qwen3-Next-80B-A3B-Instruct-GSM8K-MTP-finetuned | READ BODY |

**sm_120-specific notes that are NOT "ruled out" but constrain the same researcher** (from club-3090's hardware ledger, all READ BODY):
- NVFP4 **KV** crashes on stock vLLM on consumer Blackwell: *"`nvfp4` KV crashes on consumer Blackwell (sm_120/121) on stock vLLM — `--kv-cache-dtype nvfp4` routes to the trtllm-gen FP4 FMHA, which has no build there"* — https://github.com/noonghunna/club-3090/blob/master/docs/DTYPE_MATRIX.md. A community FA2+XQA route exists but *"it requires a locally patched image and a V-scale write fix"* and has a *"silent-corruption failure mode"*.
- FP8 KV is **storage-only** on sm_120: *"Neither FA3 nor the trtllm-gen FMHA builds for those arches."*
- W8A8 does not build for sm ≥ 10.0.
- The DFlash2 row on Ampere shows `bf16 KV + FLASH_ATTN is mandatory`: *"on Ampere sm_86 FLASH_ATTN takes ONLY bf16/fp16 KV … The same config on fp8 KV forces FlashInfer and measures 74 / 138 (~40% slower)."* — the same code path applies to sm_120, so a 96 GB card can host a much longer bf16-KV context, but the KV-dtype lever is constrained.
- club-3090 explicitly lists RTX PRO 6000 Blackwell as in-family with the 5090: *"The 5090 (sm_120), RTX PRO 6000 Blackwell (sm_120), and DGX Spark GB10 (sm_121) share this behavior"*.

---

## Search log

### `web_search` queries (verbatim, in order — engine: DeepSeek Harness `web_search`)

1. `EAGLE-3 speculative decoding limitations future work arXiv 2503.01840` / `MTP head quantization robustness acceptance rate collapse 2026` / `speculative decoding draft head negative result does not help reasoning` / `Qwen3-Next MTP vLLM SGLang speculative decoding support`
2. `DFlash2 drafter speculative decoding arXiv` / `MTP head count vs acceptance rate ablation multi-token prediction` / `EAGLE-3 head training cost GPU hours 8xH100` / `"speculative decoding" head fine-tuned target model acceptance drop`
3. `SpecForge EAGLE-3 training hardware requirements 8 GPU` / `MTP head training cost GPU hours DeepSeek V3 multi-token prediction` / `EAGLE-3 head does not generalize fine-tuned target model` / `"multi-token prediction" head ablation number of heads acceptance rate`
4. `EAGLE-3 limitations "future work" speculative decoding draft head training` / `SpecForge EAGLE-3 training GPU requirement` / `vLLM speculative decoding docs eagle3 mtp medusa supported methods` / `"MTP" head 0% acceptance quantized GGUF llama.cpp discussion`
5. `llama.cpp MTP multi-token prediction support PR rejected not planned` / `ik_llama.cpp MTP speculative decoding PR` / `"draft model" robustness fine-tuned target distribution shift speculative decoding negative` / `Medusa heads repo archived unmaintained 2026`
6. `speculative decoding acceptance rate collapse long context reasoning 2026 paper` / `"draft head" OR "MTP head" does not generalize quantized target model` / `EAGLE-3 Medusa head training single GPU feasibility` / `speculative decoding "not planned" OR "wontfix" eagle medusa vllm`
7. `HASS harmonized representations speculative sampling limitations` / `Medusa heads training cost 8x A100 hours` / `Qwen3-Next MTP head vLLM NEXTN SGLang support` / `layer skipping self-speculation LayerSkip limitations draft head`
8. `"JetSpec" parallel tree drafting speculative decoding arXiv 2026` / `"SpecBundle" speculative decoding` / `MTP head fine-tuning FastMTP arXiv 2509.18362` / `speculative decoding draft head fails reasoning benchmark negative result 2026`
9. `vLLM speculative decoding "Medusa" deprecated removed 2026` / `SGLang EAGLE3 Qwen3-Next patch upstream merge request` / `"JetSpec" arXiv 2606.18394 parallel tree drafting` / `speculative decoding head robustness distribution shift draft model base model mismatch`
10. `FasterDecoding Medusa repo issue not planned maintainer response` / `SafeAILab EAGLE issue closed wontfix training head` / `"SpecBundle" lmsys blog speculative decoding phase 2` / `vLLM speculators MTP finetuning Qwen3-Next single GPU`
11. `EAGLE-3 draft head training 8xH100 requirement reproduce single GPU` / `Qwen3-Next 80B MTP head fine-tuning multi-GPU FSDP requirement` / `speculative decoding Medusa head open question unsolved 2026` / `vLLM Speculators MTP finetune hardware requirement GPU`

**Delegated `web_search` queries (subagent sessions):** 39 further queries, verbatim, listed in `prior_art/REPORT.md` §"Search log" (covering Ouroboros, GLIDE, Hydra, HASS, DFlash, FastMTP, MTP-D, long-context acceptance decay, head training cost, draft-head robustness under quantization, and survey discovery).

### Direct retrieval (URL → result)

| URL | Result |
|---|---|
| `https://raw.githubusercontent.com/noonghunna/club-3090/main/docs/engines/SGLANG.md` | HTTP 200, **empty body** (branch is `master`, not `main`) |
| `https://raw.githubusercontent.com/noonghunna/club-3090/refs/tags/v0.10.2/docs/engines/SGLANG.md` | HTTP 200, **full body read via `web_fetch`** — the LEAD confirmed, text identical to `master` |
| `https://codeload.github.com/noonghunna/club-3090/tar.gz/refs/heads/master` | HTTP 200, 9.3 MB — full repo extracted to `/tmp/club/club-3090-master`, read locally |
| `https://arxiv.org/abs/{2509.18362, 2510.07535, 2603.23911, 2604.26412, 2605.10453, 2607.12422, 2607.26627, 2602.06036, 2605.27390, 2607.19223, 2602.01469, 2602.23881, 2605.29707, 2607.05147, 2402.05109, 2401.15077, 2406.16858, 2503.01840, 2408.15766, 2401.10774, 2404.16710, 2404.19737, 2412.19437}` | HTTP 200 each; `citation_title` verified for **all 23** — no fabricated IDs remain in this report |
| `https://arxiv.org/html/2605.10453v1` | HTTP 200, 812 KB — full body read (future-work + Limitations extracted verbatim) |
| `https://inco.ai/blog/dflash2/` | HTTP 200 via `web_fetch` — full body read |
| `https://raw.githubusercontent.com/z-lab/dflash/main/README.md` | HTTP 200 via `web_fetch` — DFlash paper = arXiv 2602.06036 |
| `https://raw.githubusercontent.com/sgl-project/SpecForge/9b8873df…/docs/basic_usage/training.md` | HTTP 200 via `web_fetch` (the `main` path 404s) |
| `https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/examples/post_training/modelopt/speculative.md` | HTTP 200 via `web_fetch` |
| `https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/{eagle,dflash}.mdx` | HTTP 200 via `web_fetch` |
| `https://raw.githubusercontent.com/devnen/qwen3.6-windows-server/v1.3.3/docs/MTP_HEAD.md` | HTTP 200 via `web_fetch` |
| `https://raw.githubusercontent.com/john-rocky/CoreML-LLM/refs/heads/main/docs/EAGLE3_INTEGRATION_STATE.md` | HTTP 200 via `web_fetch` |
| `https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/` | HTTP 200 — full body read |
| `https://github.com/noonghunna/club-3090/issues/1052` | HTTP 200 — title only: *"[bug] vllm/qwen38-27b-dual-fast - speculative decoding stops working after some time dragging TPS into the 20s"* |
| `https://github.com/vllm-project/vllm/issues/{52873, 47825, 26402, 40880}` | HTTP 200 — body + state extracted from embedded HTML |
| `https://github.com/vllm-project/vllm/pull/{40898, 34163, 52816, 48375, 50021, 40914, 47490, 52216, 55760, 55861, 45295}` | HTTP 200 — state/`mergedBy`/`mergedTime`/`closedTime`/`commitsCount`/`mergeCommitSha` extracted from embedded JSON |
| `https://github.com/sgl-project/sglang/pull/{10657, 20370}` | HTTP 200 — state + closing comment extracted |
| `https://github.com/sgl-project/sglang/issues/19406` | HTTP 200 — state CLOSED (title: Marlin repack alignment error) |
| `https://github.com/vllm-project/vllm/pull/10657.diff` | HTTP 200 — 4,787 bytes, 2 files |
| `https://api.github.com/repos/vllm-project/vllm/issues/{52873, 52873/comments}` | **HTTP 403 — API rate limit exceeded** (both direct and via `web_fetch`) |
| `https://huggingface.co/api/models/{unsloth/Qwen3.6-27B-MTP-GGUF, sakamakismile/Qwen3.6-27B-NVFP4}/discussions` | **HTTP 000** via `curl`; discussion pages render without the body via `web_fetch` → reported TITLE ONLY |
| `https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/raw/main/README.md` | HTTP 200 via `web_fetch` — full card read (contains the official Qwen3.6 MTP recipe at TP=8) |
| `https://export.arxiv.org/api/query?…` | **HTTP 429 / no entries** — arXiv API unusable this session |
| `https://arxiv.org/list/cs.CL/2609` | **HTTP 404** |

### Correction carried forward
The task brief listed "HASS (arXiv 2408.15766)" without a title. The verified title is **"Learning Harmonized Representations for Speculative Sampling"** (ICLR 2025) — not a hardware-aware method. Also flagged: the brief's implied paper set contains three IDs that `web_search` associated with the wrong papers — `2402.05099` is *Hydragen*, not *Hydra* (the real Hydra is `2402.05109`); `2403.15304` is a Knowledge Tracing paper, not *GLIDE*; `2406.04433` and `2402.13718` are not *Ouroboros*. **Ouroboros has no verified URL and is therefore excluded from all claims.** Separately, club-3090 cites `vllm#39931` as "DeltaNet rollback support" — that URL resolves to a **MERGED TurboQuant PR** ("[Feature] TurboQuant: support hybrid models and uniform quantization"), so the cited blocker issue could not be located at that number.

---

# ADDENDUM (added after the main report) — GitHub abandonment sweep, recovered and independently spot-checked

A fourth sweep thread (GitHub issues/PRs across vLLM, SGLang, llama.cpp, TensorRT-LLM, Medusa, EAGLE) was stopped and then recovered. Its highest-value rows are reproduced below. **I independently re-fetched and verified** the state badges and the load-bearing verbatim quotes for C25–C29 and O1–O3; rows marked *(reported, not re-verified)* come from that thread alone.

**Method correction (supersedes my earlier note):** GitHub issue/PR comment bodies *are* retrievable — they live as `"body":"…"` inside the `<script type="application/json" data-target="react-app.embeddedData">` blob. Critically, `curl`/`urllib` must **keep** the proxy for GitHub; unsetting it yields HTTP 000 on some paths. Exact state badge is `data-status="issueClosedNotPlanned"` / `"issueClosed"` / `"issueOpened"` / `"pullMerged"` / `"pullClosed"`.

## C-addendum (ATTEMPTED-AND-ABANDONED)

| # | Question | Who tried | What happened | URL | Verbatim | Evidence |
|---|---|---|---|---|---|---|
| C25 | Does MTP acceptance collapse on a *quantized* target, and is it a vLLM bug? | @noonghunna (filed 2026-08-19) | Filed as an engine bug, then **retracted by its own author** and `CLOSED` as `NOT PLANNED` after a controlled matrix | https://github.com/vllm-project/vllm/issues/52873 | **"Update after full investigation — this is checkpoint-specific, not a vLLM defect. Closing."** … *"the permanent draft-acceptance collapse reported here turns out to be **specific to the particular quantized checkpoint we were serving**, not a vLLM engine bug."* … *"So the trigger is that checkpoint's weights on the MTP-proposer path — not the engine, and not AutoRound-INT4 as a class. The ngram result localizes it to the MTP proposer's own persistent state (the shared verification / target-GDN path stays healthy)."* … *"Also correcting the title's premise: it is **not** tied to sequence position 32768 — it's a cumulative-generated-work threshold (~14.5k with async scheduling, ~28k without), independent of context position and surviving context resets."* … *"One honest note for maintainers: the collapse being *permanent, engine-global, and surviving a fresh conversation* is notable — if you consider hardening the MTP proposer against a marginal checkpoint worthwhile, I'm happy to share the repro; I'm just not filing that as a bug."* | READ BODY (re-verified) |
| C26 | Can EAGLE/NextN self-draft against a **quantized target embedding**? | @Thireus; closed by @Fridge003 | **CLOSED UNMERGED by a maintainer with no comment at all** (created 2026-09-07T18:56:37Z, closed 2026-09-07T20:33:17Z, 1 commit, `mergeCommitSha: null`) | https://github.com/sgl-project/sglang/pull/38366 | *"EAGLE / NextN self-draft speculative decoding cannot start on a ModelOpt MIXED_PRECISION checkpoint whose token embedding is NVFP4, although the same checkpoint serves correctly without speculation."* … *"`EagleDraftWorker.init_lm_head()` hands the draft a bare weight tensor from the target's `get_embed_and_head()`, which is the whole table only while it is dense; an NVFP4 table is weight plus `weight_scale`, `weight_scale_2` and an `e2m1_lut` buffer, and the dequantizing gather reads all four off the module, so the draft gathers packed bytes with its own `UnquantizedEmbeddingMethod`."* Errors: *"The size of tensor a (5120) must match the size of tensor b (2560) at non-singleton dimension 1"* / *"CHECK_EQ(input.size(1), weight.size(0)) failed. 2560 vs 5120"* | READ BODY (state re-verified) |
| C27 | Is DFlash compatible with KV-cache quantization? | @noonghunna; closed by @benchislett | **CLOSED** — declared unsupported; workaround is a different backend, not a fix | https://github.com/vllm-project/vllm/issues/41559 | (maintainer, 2026-08-11) **"Closing for now. We have support via FLASHINFER (CUTLASS) backend. If specific feature support is desired, please create fine-grained issues for each."** — issue title: *"[Bug] DFlash speculative decoding fundamentally incompatible with all KV cache quantization (fp8, turboquant) due to non…"* | READ BODY (state re-verified: `issueClosed`) |
| C28 | Does MTP acceptance collapse at KV-slot boundaries in llama.cpp? | @ronnieops; maintainer @am17an | **CLOSED `NOT_PLANNED`** — closed on a one-line template reply while the reporter says the root cause "remains open" | https://github.com/ggml-org/llama.cpp/issues/23636 | Maintainer: **"Follow the issue template"**. Reporter's final words: **"KV slot boundary root cause (collapse at specific ctx-size values) remains open. b9318 improves boundary positions but does not eliminate them entirely."** Data reported: a 256-token ctx delta flips MTP from 2.1× (AR 78 %) to 1.04× (AR 0.6 %), repeating at ~2048-aligned boundaries | READ BODY (state re-verified: `issueClosedNotPlanned`) |
| C29 | Can a model be quantized while keeping its EAGLE drafter? | TensorRT-LLM users | **Officially NOT SUPPORTED** | https://github.com/NVIDIA/TensorRT-LLM/issues/3207 | User: *"So to me it seems that there is currently no working solution for quantizing the model with an EAGLE drafter."* NVIDIA (@hchings, 2025-04-02): *"the [Support Matrix] of the Eagle example README suggests that FP8/INT8 are not supported. @laikhtewari or other colleagues please confirm."* | READ BODY (state re-verified: `issueClosed`) |
| C30 | Medusa with tensor parallelism / custom choice trees | @southfreebird, @saif-amdouni | **Two Medusa issues STALE-CLOSED as `NOT_PLANNED`** by github-actions, plus a third for MTP acceptance | https://github.com/vllm-project/vllm/issues/16477 · https://github.com/vllm-project/vllm/issues/20813 · https://github.com/vllm-project/vllm/issues/38339 | Boilerplate (all three): **"This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"** — #16477 *"Medusa hangs when tp>1"*; #20813 *"Specifying Medusa Choice Tree"*; #38339 Step-3.5-Flash MTP acceptance **"Current vLLM (main) \| 2.4%-4.6%"** vs *"v0.15.1 + Step-AI patch \| 97%-100%"* vs *"sglang \| ~50%"*, log line *"Avg Draft acceptance rate: 3.1%"* | READ BODY *(reported, not re-verified)* |
| C31 | MTP × TurboQuant × cudagraph in the Genesis patch cycle | @noonghunna; patcher @Sandermage | Gap explicitly left open by the third-party patcher | https://github.com/vllm-project/vllm/issues/40831 | **"we did not test MTP at all in the v7.13 cycle."** … *"MTP × TurboQuant × cudagraph is a **separate bug class** that v7.13 does not cover."* … *"the MTP code path goes through `EagleProposer` (or `MTPProposer` depending on family) which is a different machinery that we have not yet read carefully. Worth saying out loud: the depth of 'I don't know what I don't know' here is real."* | READ BODY *(reported, not re-verified)* |
| C32 | Was llama.cpp's first MTP attempt kept? | @ngxson et al. | PR #15225 (GLM-style MTP) **closed unmerged, superseded** | https://github.com/ggml-org/llama.cpp/pull/15225 | *"implemented via #22673"* | TITLE ONLY *(reported, not re-verified)* |

## B-addendum (OPEN)

| # | Question | Who is still asking | What's missing | URL | Verbatim | Evidence |
|---|---|---|---|---|---|---|
| O1 | Why does an NVFP4 drafter give 0 % acceptance under CUDA-graph capture? | @soyr-redhat (vLLM, label `speculative-decoding`) | Fix for the cudagraph interaction | https://github.com/vllm-project/vllm/issues/54997 | *"An MTP/eagle draft model with **compressed-tensors NVFP4 (w4a4)** MLP layers gives **0% spec-decode acceptance** with the default engine settings. Disabling CUDA-graph capture fixes it completely."* — reported numbers: 0 % default / 77.5 % cudagraph-off / 92.3 % eager. Title: *"compressed-tensors NVFP4 weights produce 0% acceptance in the eagle/MTP drafter under CUDA-graph capture (works with --e…"* | READ BODY (state re-verified: `issueClosed`) |
| O2 | Silent ~0 % acceptance from a **quantized DFlash2 draft**, no error, no warning | @noonghunna (filed **2026-09-11**, OPEN) | Any load-time guard; the failure is indistinguishable from a weak drafter | https://github.com/sgl-project/sglang/issues/39087 | *"A **quantized** DFlash2 draft checkpoint loads into `DFlash2DraftModel`, serves, and produces drafts that are rejected ~100% of the time. There is **no error and no warning** — acceptance simply collapses from ~3.7 to ~1.0 and decode drops below no-drafter speed."* … *"The same weights **unquantized** work correctly. This is the silent counterpart to #36599 (quantized NextN draft, `modelopt_fp4`), which fails loudly with a tensor-shape mismatch instead."* | READ BODY (re-verified) |
| O3 | Why do compressed-tensors quantized targets break the NEXTN/MTP head at construction? | @rnxrx (SGLang) | The MTP head must be added to the quantizer's `ignore` list (`"re:mtp\..*"`) | https://github.com/sgl-project/sglang/issues/38574 | *"The MTP head is **not part of the HF model graph** that the quantizer traces, so it is neither quantized nor listed in `ignore`. Its weights in the checkpoint are plain bf16 (typically re-attached from the base model after quantization)."* … *"SGLang's NEXTN path constructs the draft model's `Linear` layers through the same `CompressedTensorsConfig`… `_get_scheme_from_parts` raises because there is **no dense Linear scheme for int4-weight + fp8-activation** in SGLang"* | READ BODY (re-verified) |
| O4 | `deepseek_nextn` hardcodes `quant_config=None` for `modelopt_fp4` | SGLang #36599 | A one-line fix is known but unlanded | https://github.com/sgl-project/sglang/issues/36599 | *"`deepseek_nextn.py` hardcodes `quant_config = None` whenever the target model is `modelopt_fp4`"* … workaround: *"With that one-line change … the NEXTN draft loads and speculative decoding runs: single-stream 14.2 → 20.6 tok/s on our 2-Spark setup."* … *"this override silently discarded it, which makes the flag a no-op for this case."* | TITLE ONLY *(reported, not re-verified)* |
| O5 | EAGLE-3 trained with **Speculators** cannot be loaded by vLLM | @pavelgein | Config schema mismatch between trainer and engine | https://github.com/vllm-project/vllm/issues/54526 · https://github.com/vllm-project/speculators/issues/1065 | *"Cannot load an Eagle3 model, trained with Speculators"* / *"Config mismath between Speculators and vLLM"* | TITLE ONLY *(reported, not re-verified)* |
| O6 | DFlash2 draft 0 % acceptance at float16 on Intel XPU | @martyrant, @hlin99, @yma11 (Intel) | XPU backend support | https://github.com/vllm-project/vllm/issues/55250 | — | TITLE ONLY *(reported, not re-verified)* |
| O7 | NEXTN/MTP MoE weights fail under TP>1 (GLM-5-Next) | @Kokoro2336 | TP-aware MTP weight sharding | https://github.com/sgl-project/sglang/issues/36653 | — | TITLE ONLY *(reported, not re-verified)* |
| O8 | Does Medusa support non-Llama/Mistral architectures? | @Shubin-vadim (2025-03-04) | **Zero maintainer replies** | https://github.com/FasterDecoding/Medusa/issues/132 | *"Hey guys! Is there any plan to support other types of LLMs besides Llama and Mistral?"* | READ BODY (state re-verified: OPEN) |
| O9 | Multimodal MTP in the EAGLE reference repo | @1benwu1 (open since 2026-02-28) | No maintainer answer | https://github.com/SafeAILab/EAGLE/issues/331 | *"Thx for the great work. I wonder if the authors would consider updating for MM MTP"* | TITLE ONLY *(reported, not re-verified)* |
| O10 | EAGLE's own unchecked TODO | EAGLE maintainers | — | https://github.com/SafeAILab/EAGLE | unchecked: *"- [ ] Support official EAGLE-3 for Qwen-3. - [ ] EAGLE-4."* | TITLE ONLY *(reported, not re-verified)* |

**Repo-health datapoint (reported by the recovered thread, not re-verified):** EAGLE last commit 2026-02-19 (maintained); SpecForge last commit 2026-09-10 (maintained); **Medusa last commit 2024-04-18 (abandoned)**; `FasterDecoding/hydra` and `FasterDecoding/self-speculative` both return **HTTP 404 — they do not exist as repos**.

## Correction to the main report

**B4 is wrong as written.** vLLM issue **#43559 is CLOSED** (`data-status="issueClosed"`, verified). The fix PR **#48375 remains OPEN** (created 2026-07-12, 2 commits, `mergeCommitSha: null`, verified). So the accurate statement is: the issue was closed while its fix PR stayed unmerged — which is why club-3090 still vendors the patch and reports it as required on v0.29.0. My earlier row implied the issue itself was open.

## Cross-cutting pattern (new, from the recovered rows)

**"Quantized target breaks the draft head" is reported in BOTH engines, independently, and the dominant complaint is SILENCE.** vLLM: #54997, #55250, #51581, #26402, #52873. SGLang: #36599, #38574, #36653, #39087, #38366. TensorRT-LLM: #3207 (declared unsupported outright). The structural cause is stated most clearly in SGLang #38574: *"The MTP head is **not part of the HF model graph** that the quantizer traces, so it is neither quantized nor listed in `ignore`."* — i.e. quantizers skip the head, and engines then guess what precision it is in.
