# Speculative-Decoding Drafting Heads in Inference Engines — GitHub Prior-Art Sweep

Sweep date: 2026-09-13. Window prioritised: 2025-06 → 2026-09.
Repos swept: vLLM, SGLang, TensorRT-LLM, HF transformers, llama.cpp, Speculators, SpecForge, EAGLE, Medusa.
Every row below was actually fetched and read unless marked TITLE ONLY. No quote is reconstructed.

---

## 1. ATTEMPTED-AND-ABANDONED

### A1. MTP acceptance collapse at KV-slot boundaries — llama.cpp — closed **not_planned**
- **Question:** Does MTP draft acceptance collapse at specific `--ctx-size` values because of a KV-slot boundary?
- **Who tried:** `ronnieops` (reporter); `am17an` (maintainer) responded.
- **What happened:** Reporter demonstrated a 256-token context delta flipping MTP from 2.1x to 1.04x
  (AR% 78% → 0.6%), repeating at ~2048-aligned boundaries. Maintainer replied with a one-line
  template demand. Thread stalled; closed not_planned.
- **URL:** https://github.com/ggml-org/llama.cpp/issues/23636
- **VERBATIM (maintainer `am17an`, 2026-05-25):** "Follow the issue template"
- **VERBATIM (reporter `ronnieops`, final comment):** "KV slot boundary root cause (collapse at specific ctx-size values) remains open. b9318 improves boundary positions but does not eliminate them entirely."
- READ BODY

### A2. Qwen3-Next GDN + MTP permanent draft-acceptance collapse — vLLM — closed **not_planned**
- **Question:** Why does MTP draft acceptance die permanently, engine-wide (0% until restart), on Qwen3-Next GDN?
- **Who tried:** `noonghunna` (opened the issue; also wrote the closing retraction).
- **What happened:** Author isolated the failure to one third-party quantized checkpoint and withdrew
  the vLLM bug claim. **This is the single most on-point "MTP head fails on a quantized target"
  datapoint captured in this sweep.**
- **URL:** https://github.com/vllm-project/vllm/issues/52873
- **VERBATIM (closing comment):** "Update after full investigation — this is checkpoint-specific, not a vLLM defect. Closing."
- **VERBATIM:** "the permanent draft-acceptance collapse reported here turns out to be **specific to the particular quantized checkpoint we were serving**, not a vLLM engine bug."
- **VERBATIM (result matrix):** "`Avuja/Qwen3.8-27B-int4-AutoRound` (what we hit it on) + MTP | collapses 99%→0%, permanent, at ~14.5k cumulative generated tokens" … "Official **FP8** checkpoint + MTP | **no collapse**" … "Same Avuja checkpoint but **ngram** drafter instead of MTP | **no collapse**"
- **VERBATIM (the admitted open remainder):** "One honest note for maintainers: the collapse being *permanent, engine-global, and surviving a fresh conversation* is notable — if you consider hardening the MTP proposer against a marginal checkpoint worthwhile, I'm happy to share the repro; I'm just not filing that as a bug."
- READ BODY

### A3. EAGLE/NextN cannot share a quantized target embedding — SGLang — closed-unmerged by maintainer, **no comment**
- **Question:** Can EAGLE/NextN self-draft start when the target checkpoint's token embedding is quantized (NVFP4)?
- **Who tried:** `Thireus` (author); `Fridge003` (closer).
- **What happened:** Author supplied a complete fix (share the *module*, not the bare weight tensor,
  keyed on tensor identity), CPU tests, byte-identical target-forward verification, and +50–105%
  decode numbers. CI red. Closed by maintainer 2026-09-07 with **no comment**.
- **URL:** https://github.com/sgl-project/sglang/pull/38366 (companion: https://github.com/sgl-project/sglang/pull/38364)
- **VERBATIM (author's statement of the defect):** "EAGLE / NextN self-draft speculative decoding cannot start on a ModelOpt MIXED_PRECISION checkpoint whose token embedding is NVFP4, although the same checkpoint serves correctly without speculation."
- **VERBATIM (mechanism):** "`EagleDraftWorker.init_lm_head()` hands the draft a bare weight tensor from the target's `get_embed_and_head()`, which is the whole table only while it is dense; an NVFP4 table is weight plus `weight_scale`, `weight_scale_2` and an `e2m1_lut` buffer, and the dequantizing gather reads all four off the module, so the draft gathers packed bytes with its own `UnquantizedEmbeddingMethod`."
- **VERBATIM (observed errors):** "`The size of tensor a (5120) must match the size of tensor b (2560) at non-singleton dimension 1`" and "`CHECK_EQ(input.size(1), weight.size(0)) failed. 2560 vs 5120`"
- **Closure reason: SILENCE.** `Fridge003 closed this  Sep 7, 2026`. No rejection rationale was posted.
- READ BODY

### A4. MTP × TurboQuant × cudagraph — vLLM — closed by the third-party patcher, gap left documented-open
- **Question:** Does TurboQuant KV cache work with MTP speculative decoding?
- **Who tried:** `noonghunna` (reporter), `Sandermage` (third-party patcher, Genesis v7.13).
- **What happened:** Patcher closed the thread for the *ngram* path only and explicitly refused to
  open the MTP follow-up. Self-declared ignorance of the MTP path is the key verbatim.
- **URL:** https://github.com/vllm-project/vllm/issues/40831
- **VERBATIM (`Sandermage`):** "we did not test MTP at all in the v7.13 cycle."
- **VERBATIM:** "MTP × TurboQuant × cudagraph is a **separate bug class** that v7.13 does not cover."
- **VERBATIM:** "the MTP code path goes through `EagleProposer` (or `MTPProposer` depending on family) which is a different machinery that we have not yet read carefully. Worth saying out loud: the depth of 'I don't know what I don't know' here is real."
- **VERBATIM (status table):** "MTP | broken (per your Probe 9) | open — not covered by v7.13" ; "eagle / eagle3 | untested by us | unknown" ; "draft_model | untested by us | unknown"
- **VERBATIM (refusal to own the gap):** "One concrete thing — we won't open the new MTP issue … the primary report should come from where the failure is reproducible."
- READ BODY

### A5. DFlash non-causal attention vs KV-cache quantization — vLLM — closed with a backend workaround
- **Question:** Is DFlash spec decoding compatible with KV cache quantization (fp8, turboquant)?
- **Who closed it:** `benchislett`.
- **What happened:** Declared fundamentally incompatible (DFlash needs non-causal attention), then
  closed pointing at an alternative backend and asking for narrow follow-ups instead.
- **URL:** https://github.com/vllm-project/vllm/issues/41559
- **VERBATIM (closing comment, `benchislett`, 2026-08-11):** "Closing for now. We have support via FLASHINFER (CUTLASS) backend. If specific feature support is desired, please create fine-grained issues for each."
- **VERBATIM (title-level claim):** "DFlash speculative decoding fundamentally incompatible with all KV cache quantization (fp8, turboquant) due to non-causal attention requirement"
- READ BODY

### A6. Medusa speculation hangs when tp > 1 — vLLM — stale-closed **not_planned**, no technical resolution
- **Question:** What causes Medusa speculation to hang when tensor parallelism > 1?
- **Who tried:** `southfreebird`; closed by `github-actions`.
- **URL:** https://github.com/vllm-project/vllm/issues/16477
- **VERBATIM (`github-actions`, 2025-07-11):** "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!"
- **VERBATIM (`github-actions`, 2025-08-10):** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- READ BODY

### A7. Medusa choice-tree control — vLLM — stale-closed **not_planned**
- **Question:** Can a user specify the Medusa choice tree in vLLM?
- **Who tried:** `saif-amdouni`; closed by `github-actions`.
- **URL:** https://github.com/vllm-project/vllm/issues/20813
- **VERBATIM (`github-actions`, 2025-11-12):** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- READ BODY

### A8. Step-3.5-Flash MTP acceptance 2.4–4.6% vs sglang ~50% vs patched 97–100% — vLLM — stale-closed **not_planned**
- **Question:** Why is vLLM's Step-3.5-Flash MTP acceptance 2.4–4.6% when a patched v0.15.1 gives 97–100% and sglang gives ~50%?
- **Who tried:** `elinx`; closed by `github-actions`.
- **URL:** https://github.com/vllm-project/vllm/issues/38339
- **VERBATIM (reporter's measurement table):** "Current vLLM (main) | 2.4%-4.6% | N/A" vs "v0.15.1 + Step-AI patch | 97%-100% | 58%-83%" vs "sglang | N/A | ~50%"
- **VERBATIM (observed log line):** "Per-position acceptance rate: 0.031, Avg Draft acceptance rate: 3.1%"
- **VERBATIM (`github-actions`, 2026-07-27):** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- READ BODY

### A9. TensorRT-LLM: quantizing a model that has an EAGLE drafter — answered "not supported"
- **Question:** Is there any working way to quantize (FP8/INT8) a model served with an EAGLE drafter?
- **Who tried:** `geaned`; answered by `hchings` (NVIDIA).
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/issues/3207
- **VERBATIM (`geaned`, the ask):** "So to me it seems that there is currently no working solution for quantizing the model with an EAGLE drafter. If so, could you please give me an advice on which speculative decoding technique to try that works correctly at the moment and supports FP8 quantization?"
- **VERBATIM (`hchings`, NVIDIA, 2025-04-02):** "Hi @geaned, the [Support Matrix](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/eagle#support-matrix) of the Eagle example README suggests that FP8/INT8 are not supported. @laikhtewari or other colleagues please confirm."
- READ BODY

### A10. `deepseek_nextn` hardcodes `quant_config=None` for modelopt_fp4 — SGLang — reported, fix known, still OPEN
- **Question:** Why is a quantized NextN draft head built in BF16 while its checkpoint tensors are FP4-packed?
- **Who tried:** `chuck-ads`.
- **URL:** https://github.com/sgl-project/sglang/issues/36599
- **VERBATIM (the defect):** "`deepseek_nextn.py` hardcodes `quant_config = None` whenever the target model is `modelopt_fp4`"
- **VERBATIM:** "That's correct for DeepSeek V3/R1 NVFP4 checkpoints, whose MTP head ships in BF16 — but **GLM-5.3-Flash NVFP4 quantizes its NextN layer**"
- **VERBATIM (workaround that worked):** "Keep the quant config for the NextN draft … With that one-line change … the NEXTN draft loads and speculative decoding runs: single-stream 14.2 → 20.6 tok/s on our 2-Spark setup."
- **VERBATIM (why the official flag is a no-op):** "it took `--speculative-draft-model-quantization modelopt_fp4` just to get a quant config to the draft at all (the draft doesn't auto-detect from the checkpoint the way the target does) — and then this override silently discarded it, which makes the flag a no-op for this case."
- READ BODY

### A11. GLM-5.3-Flash NextN/MTP TP8 embedding-gather OOB — SGLang — closed-unmerged, author self-closed
- **Question:** Why does NextN/MTP crash on GLM-5.3-Flash at TP8 with an embedding gather out of bounds?
- **Who tried:** `Kokoro2336`.
- **URL:** https://github.com/sgl-project/sglang/pull/37791 (duplicates: https://github.com/sgl-project/sglang/pull/37783 , https://github.com/sgl-project/sglang/pull/37782 )
- **What happened:** Author closed his own PR one day after opening it. `Kokoro2336 closed this  Sep 4, 2026`. No maintainer rationale captured.
- TITLE ONLY for the closure rationale (page read, no rejection comment present).
- READ BODY (title + closure actor)

### A12. PD decode radix caching with EAGLE/EAGLE3 — SGLang — closed-unmerged, author self-closed
- **Question:** Can decode radix caching be enabled together with EAGLE/EAGLE3 under prefill/decode disaggregation?
- **Who tried:** `ByronHsu` (co-authored with Manikvsin).
- **URL:** https://github.com/sgl-project/sglang/pull/39150
- **What happened:** `ByronHsu closed this  Sep 12, 2026`, one day after opening. Commit title: "Allow decode radix caching with EAGLE and EAGLE3".
- READ BODY (title + closure actor)

---

## 2. CLOSED / SHIPPED

| # | Question it answers | Who closed it | Artifact | URL | Status | Evidence |
|---|---|---|---|---|---|---|
| C1 | Does llama.cpp support MTP heads? | am17an (merged, 28 commits) | PR #22673 "llama + spec: MTP Support" | https://github.com/ggml-org/llama.cpp/pull/22673 | **merged** | READ BODY |
| C2 | Does llama.cpp support EAGLE-3? | ggerganov (merged, 28 commits) | PR #18039 "[Speculative decoding] feat: add EAGLE3 speculative decoding support" | https://github.com/ggml-org/llama.cpp/pull/18039 | **merged** | READ BODY |
| C3 | How did llama.cpp finally land GLM-style MTP? | ngxson | PR #15225 closed; implemented via #22673 | https://github.com/ggml-org/llama.cpp/pull/15225 | **closed-unmerged, superseded** | READ BODY |
| C4 | Is DFlash2 merged in vLLM? | SubSir | PR #52816 "[Spec Decode] DFlash2: local convolution + candidate selector" | https://github.com/vllm-project/vllm/pull/52816 | **merged** | READ BODY |
| C5 | Does Speculators cover MTP heads? | Red Hat / vLLM | Speculators README, "MTP Finetuning Support" | https://github.com/vllm-project/speculators | **shipped** | READ BODY |
| C6 | Is the EAGLE reference repo maintained? | SafeAILab | EAGLE README + commits | https://github.com/SafeAILab/EAGLE | **maintained**, last commit 2026-02-19 | READ BODY |
| C7 | Is SpecForge maintained? | sgl-project | SpecForge repo + commits | https://github.com/sgl-project/SpecForge | **maintained**, last commit 2026-09-10 | READ BODY |
| C8 | Is Medusa maintained? | FasterDecoding | Medusa repo + commits | https://github.com/FasterDecoding/Medusa | **effectively abandoned**, last commit 2024-04-18 | READ BODY |
| C9 | Do Hydra / self-speculative exist as standalone repos? | — | both HTTP 404 | https://github.com/FasterDecoding/hydra ; https://github.com/FasterDecoding/self-speculative | **do not exist** | READ BODY |
| C10 | Is vLLM PR #39931 about DeltaNet rollback? | JartX | PR #39931 is "TurboQuant: support hybrid models and uniform quantization" | https://github.com/vllm-project/vllm/pull/39931 | **merged — lead was mis-numbered** | READ BODY |
| C11 | Is vLLM #52873 a live vLLM bug? | noonghunna (self) | Issue #52873 | https://github.com/vllm-project/vllm/issues/52873 | **closed not_planned** | READ BODY |
| C12 | Is vLLM #41559 a live vLLM bug? | benchislett | Issue #41559 | https://github.com/vllm-project/vllm/issues/41559 | **closed** | READ BODY |
| C13 | Status of the other named vLLM leads | — | #40914, #48375, #50021, #43559 | https://github.com/vllm-project/vllm/pull/40914 ; https://github.com/vllm-project/vllm/pull/48375 ; https://github.com/vllm-project/vllm/pull/50021 ; https://github.com/vllm-project/vllm/issues/43559 | **#40914 OPEN, #48375 OPEN, #50021 OPEN, #43559 closed (COMPLETED)** | READ BODY (state metadata only) |

---

## 3. OPEN

| # | Who is asking | What's missing | URL | VERBATIM |
|---|---|---|---|---|
| O1 | `soyr-redhat` | NVFP4 compressed-tensors drafter gives 0% acceptance under CUDA-graph capture | https://github.com/vllm-project/vllm/issues/54997 | "An MTP/eagle draft model with **compressed-tensors NVFP4 (w4a4)** MLP layers gives **0% spec-decode acceptance** with the default engine settings. Disabling CUDA-graph capture fixes it completely." Table: default 0% / cudagraph off 77.5% / `--enforce-eager` 92.3%. Label `speculative-decoding`. Page shows state **Closed**; only one comment; no maintainer explanation posted. |
| O2 | `martyrant`, `hlin99`, `yma11` | DFlash2 draft 0% acceptance at float16 on XPU (bf16 works) | https://github.com/vllm-project/vllm/issues/55250 | Title: "[Bug]: DFlash2 draft gets 0% acceptance with --dtype float16 on XPU (bf16 works) — Qwen3.8-27B + incoai/Qwen3.8-27B-DFlash2". Labels `quantization,intel-gpu`. OPEN, 10 comments. |
| O3 | `eisbaw`, `jschmied` | DFlash fused-KV projection silently corrupts any weight-quantized drafter | https://github.com/vllm-project/vllm/issues/51581 | Title: "DFlash fused-KV projection calls F.linear on a sliced qkv_proj weight — breaks (and can silently corrupt) any weight-quantized drafter". Label `quantization`. OPEN. |
| O4 | `pavelgein` (+ `he-yufeng`, `fynnsu`) | An EAGLE-3 model trained with Speculators cannot be loaded by vLLM | https://github.com/vllm-project/vllm/issues/54526 | Title: "[Bug]: Cannot load an Eagle3 model, trained with Speculators". OPEN. Companion: https://github.com/vllm-project/speculators/issues/1065 "[Bug]: Config mismath between Speculators and vLLM". |
| O5 | `eleqtrizit`, `josephkern`, `WindChimeRan`, `alkari` | MTP + NVFP4 weight shape mismatch, long-running and stale | https://github.com/vllm-project/vllm/issues/35031 | Title: "[Bug]: MTP Speculative Decoding with NVFP4: Weight Shape Mismatch". Labels `bug,stale`. 13 comments. |
| O6 | `noonghunna` | Quantized DFlash2 draft silently yields ~0% acceptance in SGLang, no error and no warning | https://github.com/sgl-project/sglang/issues/39087 | "A **quantized** DFlash2 draft checkpoint loads into `DFlash2DraftModel`, serves, and produces drafts that are rejected ~100% of the time. There is **no error and no warning** — acceptance simply collapses from ~3.7 to ~1.0 and decode drops below no-drafter speed." Table: compressed-tensors 1.03 accept-len / 0.004 rate vs bf16 3.71 / 0.61. Filed 2026-09-11. |
| O7 | `rnxrx` | compressed-tensors + NEXTN/MTP head fails at construction; error does not name the module | https://github.com/sgl-project/sglang/issues/38574 | "Serving a compressed-tensors checkpoint with `--speculative-algorithm NEXTN` crashes at model construction … The failing module is the **MTP head's** `mtp.layers.0.self_attn.qkv_proj` (found by instrumenting `prefix`; the error itself does not say)." Mechanism: "The MTP head is **not part of the HF model graph** that the quantizer traces, so it is neither quantized nor listed in `ignore`." Workaround: add `"re:mtp\\..*"` to `quantization_config.ignore`. |
| O8 | `Kokoro2336` | NEXTN/MTP MoE weights fail to load under TP>1 for Glm5NextForConditionalGeneration | https://github.com/sgl-project/sglang/issues/36653 | "NEXTN/MTP speculative decoding fails to load MTP MoE weights under TP>1 for Glm5NextForConditionalGeneration (GLM-5.3-Flash)". OPEN. |
| O9 | `doptime`, `blazingbhavneek`, `ormandj` | AWQ/INT8 quantized Qwen3.5-27B cannot start at all | https://github.com/sgl-project/sglang/issues/19406 | "Currently, loading the AWQ or INT8 quantized versions of `Qwen3.5-27B` fails entirely on the latest SGLang `main` branch." Label `inactive`; closed. The fix attempt https://github.com/sgl-project/sglang/pull/20370 was **closed-unmerged after the author self-closed it**. |
| O10 | `Shubin-vadim` (Medusa) | Medusa supports only Llama/Mistral; no maintainer answer in 18 months | https://github.com/FasterDecoding/Medusa/issues/132 | "Hey guys! Is there any plan to support other types of LLMs besides Llama and Mistral?" — asked 2025-03-04, still OPEN, zero maintainer replies. Repo's last commit is 2024-04-18. |
| O11 | `1benwu1` (EAGLE) | Multimodal MTP is not supported in the EAGLE reference repo | https://github.com/SafeAILab/EAGLE/issues/331 | "Thx for the great work. I wonder if the authors would consider updating for MM MTP" — OPEN since 2026-02-28. |
| O12 | EAGLE maintainers (their own TODO list) | Official EAGLE-3 for Qwen-3 is still unchecked, as is EAGLE-4 | https://github.com/SafeAILab/EAGLE | README "## Todo": "- [x] Training code of EAGLE-3. - [x] Support LLaMA-4. - [ ] Support official EAGLE-3 for Qwen-3. - [ ] EAGLE-4." Also: "We strongly recommend using [SpecForge](https://github.com/sgl-project/SpecForge) for out-of-the-box training of EAGLE-3 with SGLang." |
| O13 | Medusa maintainers (their own ROADMAP.md) | Batched inference, fine-grained KV cache management, and vLLM/llama.cpp/exllama integration all unchecked | https://github.com/FasterDecoding/Medusa/blob/main/ROADMAP.md | "## Functionality - [ ] Batched inference - [ ] Fine-grained KV cache management" … "### Serving - [ ] [vllm](https://github.com/vllm-project/vllm)" |
| O14 | `hernandez42` | Qwen3.x MTP is not supported in llama.cpp at all | https://github.com/ggml-org/llama.cpp/issues/28051 | Title: "Support Qwen3.x MTP (Multi-Token Prediction) for speculative decoding" — OPEN, filed 2026-08-30. Body is collapsed in the served HTML; title-level only. |
| O15 | `cplusplus2` | MTP retains inter-request state → non-determinism and degradation | https://github.com/ggml-org/llama.cpp/issues/26425 | Title: "Eval bug: MTP retains inter-request state causing non-deterministic output and model degradation (Qwen3.6-35B-A3B-MTP)" — OPEN. |
| O16 | `zsogitbe` | MTP/draft spec decode is not exposed in the public C API | https://github.com/ggml-org/llama.cpp/issues/27469 ; https://github.com/ggml-org/llama.cpp/pull/27788 | Title: "Feature Request: Expose Speculative Decoding (MTP / Draft) in public C API (llama.h) for downstream bindings" — OPEN. |
| O17 | `Mewo518`, `wenhaoli-xmu`, `wittycheng` | Medusa head training produces NaN losses / KV-cache questions, unanswered | https://github.com/FasterDecoding/Medusa/issues/139 ; https://github.com/FasterDecoding/Medusa/issues/134 ; https://github.com/FasterDecoding/Medusa/issues/137 | TITLE ONLY. Repo shows 51 open issues, 0 recent maintainer activity. |
| O18 | `h-guo18` | EAGLE-3 produces gibberish on Llama-3.1-8B in TRT-LLM 1.2.0rc8 | https://github.com/NVIDIA/TensorRT-LLM/issues/11125 | Title: "[Bug]: Gibberish outputs on Llama8B with Eagle3", label `bug`. OPEN. |
| O19 | `transcend-0` | transformers has no multi-candidate / tree speculative search | https://github.com/huggingface/transformers/issues/39684 | Title: "Add multi-candidate & tree search for assisted decoding (speculative decoding)" — OPEN. |
| O20 | `Kissmetothemoon` | transformers has no budget-bounded acceptance control for assisted decoding | https://github.com/huggingface/transformers/issues/48636 | Title: "[Feature Request] Approximate Speculative Decoding (ASD): training-free, budget-bounded acceptance for greedy assisted decoding" — OPEN. |
| O21 | `VaggelisGian` | transformers assisted decoding lacks batched logits processors | https://github.com/huggingface/transformers/issues/48390 | TITLE ONLY. |
| O22 | `pavelgein` (speculators) | Speculators' Eagle3 conversion silently writes random weights on checkpoint-key mismatch | https://github.com/vllm-project/speculators/issues/1114 | TITLE ONLY. |
| O23 | `doptime` / SGLang | Speculative decoding (DFLASH) crashes with `--kv-cache-dtype nvfp4` | https://github.com/sgl-project/sglang/issues/36001 | TITLE ONLY. Related: https://github.com/sgl-project/sglang/issues/36010 |
| O24 | SGLang users | MTP accept rate degrades to 0.19 on Qwen3.5 with GPQA at 32K+ context | https://github.com/sgl-project/sglang/issues/30763 | TITLE ONLY. |
| O25 | SGLang users | gpt-oss-120b with the nvidia EAGLE3-v3 head crashes | https://github.com/sgl-project/sglang/issues/32226 | TITLE ONLY. |
| O26 | SGLang users | GLM-5.2 EAGLE/MTP fails with DSA attention when attn_tp_size > 1 | https://github.com/sgl-project/sglang/issues/30296 | TITLE ONLY. |

---

## 4. Cross-cutting patterns (observed only — no recommendations)

1. **"Draft head fails on a quantized target" is reported in both major engines, in both directions** —
   quantized *target* breaking the *head* (vLLM #54997, #55250, #35031, #51581; SGLang #36599, #38574,
   #36653, #39087, #38366) and a quantized *drafter* breaking itself.
2. **Silent failure is the recurring complaint.** vLLM #54997 ("0% acceptance … output is incoherent"),
   SGLang #39087 ("no error, no warning"), vLLM #51581 ("can silently corrupt"), speculators #1114
   ("writes randomly initialized weights … instead of failing").
3. **MTP heads are structurally absent from the HF model graph**, so quantizers skip them (SGLang #38574)
   and engines then guess at their config (SGLang #36599) — the flag meant to fix it is a no-op.
4. **Long-context / boundary acceptance collapse recurs independently in three places:**
   llama.cpp #23636 (KV-slot boundary), vLLM #52873 (cumulative-work threshold ~14.5k tokens),
   llama.cpp #26425 (inter-request state).
5. **Reference-repo health has diverged sharply:** EAGLE alive (last commit 2026-02-19) and its EAGLE-3
   training handed off to SpecForge (last commit 2026-09-10); Medusa fossilised (last commit
   2024-04-18) with 51 open issues and no maintainer replies; Hydra and self-speculative have no repo.

---

## 5. Search log (verbatim)

### vLLM
```
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aunmerged+eagle
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+eagle&sort=updated-desc
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+medusa
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+mtp
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+wontfix+speculative&sort=updated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+eagle3&sort=updated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+mtp&sort=updated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+medusa&sort=updated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+%22acceptance+rate%22+speculative&sort=updated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+label%3A%22speculative-decoding%22&sort=updated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+delta
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+deltanet+rollback
https://github.com/vllm-project/vllm/issues/52873
https://github.com/vllm-project/vllm/issues/52873?_pjax=1
https://github.com/vllm-project/vllm/issues/52873?plain=1
https://github.com/vllm-project/vllm/issues/43559
https://github.com/vllm-project/vllm/pull/39931
https://github.com/vllm-project/vllm/pull/40914
https://github.com/vllm-project/vllm/pull/48375
https://github.com/vllm-project/vllm/pull/50021
https://github.com/vllm-project/vllm/pull/52816
https://github.com/vllm-project/vllm/issues/54997
https://github.com/vllm-project/vllm/issues/55250
https://github.com/vllm-project/vllm/issues/54526
https://github.com/vllm-project/vllm/issues/35031
https://github.com/vllm-project/vllm/issues/16477
https://github.com/vllm-project/vllm/issues/20813
https://github.com/vllm-project/vllm/issues/38339
https://github.com/vllm-project/vllm/issues/55605
https://github.com/vllm-project/vllm/issues/55357
https://github.com/vllm-project/vllm/issues/28312
https://github.com/vllm-project/vllm/issues/51581
https://github.com/vllm-project/vllm/issues/41559
https://github.com/vllm-project/vllm/issues/40831
```

### SGLang
```
https://github.com/sgl-project/sglang/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+eagle3
https://github.com/sgl-project/sglang/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+eagle
https://github.com/sgl-project/sglang/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+mtp
https://github.com/sgl-project/sglang/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+medusa
https://github.com/sgl-project/sglang/issues?q=is%3Aissue+is%3Aclosed+eagle3
https://github.com/sgl-project/sglang/issues?q=is%3Aissue+is%3Aclosed+mtp
https://github.com/sgl-project/sglang/issues?q=is%3Aissue+quantized+draft+speculative
https://github.com/sgl-project/sglang/issues?q=is%3Aissue+offloader+tied
https://github.com/sgl-project/sglang/issues?q=is%3Aissue+cute_dsl+cuda+graph+hang
https://github.com/sgl-project/sglang/issues/19406
https://github.com/sgl-project/sglang/pull/20370
https://github.com/sgl-project/sglang/pull/38366
https://github.com/sgl-project/sglang/pull/38364
https://github.com/sgl-project/sglang/pull/37791
https://github.com/sgl-project/sglang/pull/39150
https://github.com/sgl-project/sglang/pull/31214
https://github.com/sgl-project/sglang/issues/39087
https://github.com/sgl-project/sglang/issues/38574
https://github.com/sgl-project/sglang/issues/36599
https://github.com/sgl-project/sglang/issues/36653
https://github.com/sgl-project/sglang/issues/36001
https://github.com/sgl-project/sglang/issues/30763
https://github.com/sgl-project/sglang/issues/32226
```

### TensorRT-LLM
```
https://github.com/NVIDIA/TensorRT-LLM/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+medusa
https://github.com/NVIDIA/TensorRT-LLM/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+eagle
https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue+is%3Aclosed+medusa
https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue+is%3Aclosed+eagle
https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue+is%3Aclosed+mtp
https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue+medusa
https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue+eagle3
https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue+mtp
https://github.com/NVIDIA/TensorRT-LLM/issues/3207
https://github.com/NVIDIA/TensorRT-LLM/issues/11125
https://github.com/NVIDIA/TensorRT-LLM/issues/15022
https://github.com/NVIDIA/TensorRT-LLM/issues/15182
```

### llama.cpp
```
https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+eagle
https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+medusa
https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+mtp
https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+mtp
https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+eagle
https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+medusa&per_page=25&sort=created&order=desc
https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+%22multi-token+prediction%22&per_page=25&sort=created&order=desc
https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+%22won%27t+fix%22+speculative+OR+draft+OR+medusa+in:comments&per_page=25
https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+label:wontfix&per_page=25&sort=created&order=desc
https://github.com/ggml-org/llama.cpp/issues/23636
https://github.com/ggml-org/llama.cpp/issues/23658
https://github.com/ggml-org/llama.cpp/issues/28051
https://github.com/ggml-org/llama.cpp/issues/26425
https://github.com/ggml-org/llama.cpp/pull/15225
https://github.com/ggml-org/llama.cpp/pull/22673
https://github.com/ggml-org/llama.cpp/pull/23269
https://github.com/ggml-org/llama.cpp/pull/18039
https://github.com/ggml-org/llama.cpp/pull/27788
```

### transformers
```
https://github.com/huggingface/transformers/issues?q=is%3Aissue+eagle+speculative
https://github.com/huggingface/transformers/issues?q=is%3Aissue+medusa
https://github.com/huggingface/transformers/issues?q=is%3Aissue+mtp+speculative
https://github.com/huggingface/transformers/issues?q=is%3Aissue+is%3Aclosed+medusa
https://github.com/huggingface/transformers/issues?q=is%3Aissue+is%3Aclosed+eagle
https://api.github.com/search/issues?q=repo:huggingface/transformers+medusa+OR+eagle&per_page=30&sort=created&order=desc
https://api.github.com/search/issues?q=repo:huggingface/transformers+%22assisted+decoding%22+in:title&per_page=30&sort=created&order=desc
```

### Ecosystem
```
https://github.com/vllm-project/speculators
https://github.com/vllm-project/speculators/issues?q=is%3Aissue
https://github.com/vllm-project/speculators/commits/main
https://github.com/vllm-project/speculators/issues/1065
https://github.com/vllm-project/speculators/issues/1114
https://raw.githubusercontent.com/vllm-project/speculators/main/README.md
https://github.com/SafeAILab/EAGLE
https://github.com/SafeAILab/EAGLE/issues?q=is%3Aissue
https://github.com/SafeAILab/EAGLE/issues/331
https://github.com/SafeAILab/EAGLE/commits/main
https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md
https://github.com/FasterDecoding/Medusa
https://github.com/FasterDecoding/Medusa/issues?q=is%3Aissue
https://github.com/FasterDecoding/Medusa/issues/132
https://github.com/FasterDecoding/Medusa/commits/main
https://raw.githubusercontent.com/FasterDecoding/Medusa/main/README.md
https://raw.githubusercontent.com/FasterDecoding/Medusa/main/ROADMAP.md
https://github.com/sgl-project/SpecForge
https://github.com/sgl-project/SpecForge/commits/main
https://github.com/orgs/FasterDecoding/repositories
https://github.com/FasterDecoding/hydra             -> HTTP 404
https://github.com/FasterDecoding/self-speculative  -> HTTP 404
https://api.github.com/search/issues?q=repo:FasterDecoding/Medusa+is:issue+state:open&per_page=15&sort=created&order=desc
https://api.github.com/repos/{speculators,SafeAILab/EAGLE,FasterDecoding/Medusa,sgl-project/SpecForge} -> HTTP 403 (core rate limit exhausted)
```

---

## 6. Known gaps / corrections to the brief

- **vLLM PR #39931 is NOT a DeltaNet-rollback PR.** It is "[Feature] TurboQuant: support hybrid models
  and uniform quantization" by JartX, **MERGED**. No DeltaNet-rollback issue was found under that number,
  nor under `is:issue delta`.
- **vLLM #40914, #48375, #50021 are all still OPEN, not closed-unmerged.** #52816 is MERGED. #43559 is
  closed (state_reason COMPLETED). Their comment bodies were not transcribed before the time call.
- **Not located:** TensorRT-LLM Medusa deprecation/removal PR; SGLang offloader tied-weights bug;
  SGLang CUTE_DSL Ampere cuda-graph hang.
- **Not fetched:** HF discussions `unsloth/Qwen3.6-27B-MTP-GGUF#32` and
  `sakamakismile/Qwen3.6-27B-NVFP4#7`.
- **Method note for reproducibility:** GitHub issue-comment text is NOT in the served HTML for
  JS-rendered issue pages, but IS present in the `react-app.embeddedData` JSON blob as `"body":"..."`
  fields. PR pages under some repos (e.g. SGLang) still serve comment text in plain HTML.
  `raw.githubusercontent.com` is blocked for `curl` but works via the `web_fetch` tool;
  `api.github.com` core quota was exhausted (60/60) and search quota is 10/hr.
