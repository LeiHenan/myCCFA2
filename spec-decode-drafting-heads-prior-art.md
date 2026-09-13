# Speculative-Decoding Drafting Heads in Inference Engines — Prior Art / Gap Map
Window prioritised: 2025-06 → 2026-09. Sweep date: 2026-09-13.
Repos swept: vLLM, SGLang, TensorRT-LLM, HF transformers, llama.cpp, Speculators, SpecForge, EAGLE, Medusa.

> Note on this sweep's completeness: the parent called time before the TensorRT-LLM, transformers and
> ecosystem legs were fully mined. Sections marked **[PARTIAL]** are known-incomplete. Nothing below is
> guessed; every row carries a URL and a READ BODY / TITLE ONLY marker.

---

## CLOSED / SHIPPED

| # | Question it answers | Who closed it | Artifact | URL | Status | Evidence |
|---|---|---|---|---|---|---|
| 1 | Does llama.cpp support MTP heads? | am17an (merged) | PR #22673 "llama + spec: MTP Support" | https://github.com/ggml-org/llama.cpp/pull/22673 | **merged** | READ BODY |
| 2 | Does llama.cpp support EAGLE-3? | ggerganov (merged) | PR #18039 "[Speculative decoding] feat: add EAGLE3 speculative decoding support" | https://github.com/ggml-org/llama.cpp/pull/18039 | **merged** | READ BODY |
| 3 | How did llama.cpp finally land GLM-style MTP? | ngxson | PR #15225 (closed) → implemented via #22673 | https://github.com/ggml-org/llama.cpp/pull/15225 | **closed-unmerged, superseded** | READ BODY |
| 4 | Is DFlash2 in vLLM, and since when? | SubSir (merged) | PR #52816 "[Spec Decode] DFlash2: local convolution + candidate selector" | https://github.com/vllm-project/vllm/pull/52816 | **merged** | READ BODY |
| 5 | Does Speculators (vLLM) cover MTP heads? | Red Hat / vLLM | Speculators README — "MTP Finetuning Support" | https://github.com/vllm-project/speculators | **shipped** | READ BODY |
| 6 | Is the EAGLE reference repo still maintained? | SafeAILab | EAGLE README + commits | https://github.com/SafeAILab/EAGLE | **maintained** (README even today; last commit 2026-02-19) | READ BODY |
| 7 | Is SpecForge maintained? | sgl-project | SpecForge repo | https://github.com/sgl-project/SpecForge | **maintained** (last commit 2026-09-10) | READ BODY |
| 8 | Is Medusa maintained? | FasterDecoding | Medusa repo + commits | https://github.com/FasterDecoding/Medusa | **effectively abandoned** (last commit 2024-04-18) | READ BODY |
| 9 | Are Hydra / self-speculative standalone repos? | — | 404 on both | https://github.com/FasterDecoding/hydra , https://github.com/FasterDecoding/self-speculative | **do not exist** (HTTP 404) | READ BODY |
| 10 | Does vLLM support MTP + quantized targets? | — | issues #54997, #35031 | see OPEN | **not closed as fixed** | READ BODY |
| 11 | GLM-5.3-Flash NextN/MTP TP8 embedding OOB | Kokoro2336 (self-closed) | SGLang PR #37791 | https://github.com/sgl-project/sglang/pull/37791 | **closed-unmerged, self-closed** | READ BODY |
| 12 | TRT-LLM: is FP8/INT8 quantization of an EAGLE drafter supported? | hchings | Issue #3207 | https://github.com/NVIDIA/TensorRT-LLM/issues/3207 | **answered: not supported** (issue still open/triaged) | READ BODY |

---

## ATTEMPTED-AND-ABANDONED (highest value)

### A1. MTP acceptance collapse at KV-slot boundaries — llama.cpp, closed **not_planned**
- **Question:** Does MTP draft acceptance collapse at specific `--ctx-size` values due to a KV-slot boundary?
- **Who tried:** `ronnieops` (reporter); `am17an` (maintainer) responded.
- **What happened:** Reporter showed a 256-token context delta flipping MTP from 2.1x to 1.04x
  (AR% 78% → 0.6%). Maintainer replied with a one-line template demand; the thread ran out of steam
  and the issue was closed **not_planned**.
- **URL:** https://github.com/ggml-org/llama.cpp/issues/23636
- **VERBATIM (maintainer `am17an`, 2026-05-25):** "Follow the issue template"
- **VERBATIM (reporter `ronnieops`, final comment):** "KV slot boundary root cause (collapse at specific ctx-size values) remains open. b9318 improves boundary positions but does not eliminate them entirely."
- READ BODY

### A2. Qwen3-Next GDN + MTP permanent draft-acceptance collapse — vLLM, closed **not_planned**
- **Question:** Why does MTP acceptance die permanently engine-wide (0% until restart) on Qwen3-Next GDN?
- **Who tried:** `noonghunna` (author of the issue AND of the closing comment).
- **What happened:** Author retracted his own bug: he isolated it to one third-party quantized
  checkpoint, not vLLM. Closed as not-a-vLLM-defect. **This is the single most on-point
  "MTP head fails on a quantized target" datapoint in the sweep.**
- **URL:** https://github.com/vllm-project/vllm/issues/52873
- **VERBATIM (closing comment):** "Update after full investigation — this is checkpoint-specific, not a vLLM defect. Closing."
- **VERBATIM:** "the permanent draft-acceptance collapse reported here turns out to be **specific to the particular quantized checkpoint we were serving**, not a vLLM engine bug."
- **VERBATIM (result matrix):** "`Avuja/Qwen3.8-27B-int4-AutoRound` (what we hit it on) + MTP | collapses 99%→0%, permanent, at ~14.5k cumulative generated tokens" … "Official **FP8** checkpoint + MTP | **no collapse**" … "Same Avuja checkpoint but **ngram** drafter instead of MTP | **no collapse**"
- **VERBATIM (the open remainder):** "One honest note for maintainers: the collapse being *permanent, engine-global, and surviving a fresh conversation* is notable — if you consider hardening the MTP proposer against a marginal checkpoint worthwhile, I'm happy to share the repro; I'm just not filing that as a bug."
- READ BODY

### A3. EAGLE/NextN cannot share a quantized target embedding — SGLang, closed-unmerged by maintainer
- **Question:** Can EAGLE/NextN self-draft start when the target checkpoint's token embedding is quantized (NVFP4)?
- **Who tried:** `Thireus` (author), `Fridge003` (closer).
- **What happened:** Author supplied a full fix (share the *module*, not the bare weight tensor,
  keyed on tensor identity), tests, byte-identical target-forward verification, and speed numbers.
  Only CI-failing. Closed by maintainer 2026-09-07 with **no comment**.
- **URL:** https://github.com/sgl-project/sglang/pull/38366 (companion: https://github.com/sgl-project/sglang/pull/38364)
- **VERBATIM (author's statement of the defect):** "EAGLE / NextN self-draft speculative decoding cannot start on a ModelOpt MIXED_PRECISION checkpoint whose token embedding is NVFP4, although the same checkpoint serves correctly without speculation."
- **VERBATIM (mechanism):** "`EagleDraftWorker.init_lm_head()` hands the draft a bare weight tensor from the target's `get_embed_and_head()`, which is the whole table only while it is dense; an NVFP4 table is weight plus `weight_scale`, `weight_scale_2` and an `e2m1_lut` buffer, and the dequantizing gather reads all four off the module, so the draft gathers packed bytes with its own `UnquantizedEmbeddingMethod`."
- **VERBATIM (error):** "`The size of tensor a (5120) must match the size of tensor b (2560) at non-singleton dimension 1`" / "`CHECK_EQ(input.size(1), weight.size(0)) failed. 2560 vs 5120`"
- **Closure reason: NO COMMENT.** `Fridge003 closed this Sep 7, 2026`. Rejection reason is *silence* — flag this explicitly.
- READ BODY

### A4. MTP × TurboQuant × cudagraph — vLLM, closed by the third-party patcher with the gap documented as still open
- **Question:** Does TurboQuant KV work with MTP spec decode?
- **Who tried:** `noonghunna` (reporter), `Sandermage` (third-party patcher, Genesis v7.13).
- **What happened:** Patcher closed the thread for the *ngram* path only and explicitly refused to
  open the MTP follow-up. **Self-declared ignorance of the MTP path is the key verbatim.**
- **URL:** https://github.com/vllm-project/vllm/issues/40831
- **VERBATIM (`Sandermage`):** "we did not test MTP at all in the v7.13 cycle."
- **VERBATIM:** "MTP × TurboQuant × cudagraph is a **separate bug class** that v7.13 does not cover."
- **VERBATIM:** "the MTP code path goes through `EagleProposer` (or `MTPProposer` depending on family) which is a different machinery that we have not yet read carefully. Worth saying out loud: the depth of 'I don't know what I don't know' here is real."
- **VERBATIM (refusal to own the gap):** "One concrete thing — we won't open the new MTP issue … the primary report should come from where the failure is reproducible."
- **VERBATIM (status table):** "MTP | broken (per your Probe 9) | open — not covered by v7.13" / "eagle / eagle3 | untested by us | unknown"
- READ BODY

### A5. DFlash non-causal attention vs KV-cache quantization — vLLM, closed **by design / superseded**
- **Question:** Is DFlash spec decoding compatible with KV cache quantization (fp8, turboquant)?
- **Who closed it:** `benchislett`.
- **What happened:** Declared incompatible (DFlash needs non-causal attention), then closed with a
  "use a different backend" resolution and a request to file narrow follow-ups.
- **URL:** https://github.com/vllm-project/vllm/issues/41559
- **VERBATIM (closing comment, `benchislett`, 2026-08-11):** "Closing for now. We have support via FLASHINFER (CUTLASS) backend. If specific feature support is desired, please create fine-grained issues for each."
- **VERBATIM (title-level claim):** "DFlash speculative decoding fundamentally incompatible with all KV cache quantization (fp8, turboquant) due to non-causal attention requirement"
- READ BODY

### A6. Medusa on TP>1 — vLLM, stale-closed **not_planned** with no technical resolution
- **Question:** What causes Medusa speculation to hang when `tp > 1`?
- **Who tried:** `southfreebird`; closed by `github-actions`.
- **URL:** https://github.com/vllm-project/vllm/issues/16477
- **VERBATIM (`github-actions`, 2025-07-11):** "This issue has been automatically marked as stale because it has not had any activity within 90 days. It will be automatically closed if no further activity occurs within 30 days. Leave a comment if you feel this issue should remain open. Thank you!"
- **VERBATIM (`github-actions`, 2025-08-10):** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- READ BODY

### A7. Medusa choice-tree control — vLLM, stale-closed **not_planned**
- **Question:** Can a user specify the Medusa choice tree in vLLM?
- **Who tried:** `saif-amdouni`; closed by `github-actions`.
- **URL:** https://github.com/vllm-project/vllm/issues/20813
- **VERBATIM (`github-actions`, 2025-11-12):** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- READ BODY

### A8. Step-3.5-Flash MTP acceptance 2.4–4.6% vs sglang ~50% vs patched 97–100% — vLLM, stale-closed **not_planned**
- **Question:** Why is vLLM's Step-3.5-Flash MTP acceptance 2.4–4.6% when a v0.15.1+Step-AI patch gives 97–100% and sglang gives ~50%?
- **Who tried:** `elinx`; closed by `github-actions`.
- **URL:** https://github.com/vllm-project/vllm/issues/38339
- **VERBATIM (reporter's own measurement table):** "Current vLLM (main) | 2.4%-4.6% | N/A" vs "v0.15.1 + Step-AI patch | 97%-100% | 58%-83%" vs "sglang | N/A | ~50%"
- **VERBATIM (log):** "Per-position acceptance rate: 0.031, Avg Draft acceptance rate: 3.1%"
- **VERBATIM (`github-actions`, 2026-07-27):** "This issue has been automatically closed due to inactivity. Please feel free to reopen if you feel it is still relevant. Thank you!"
- READ BODY

### A9. TensorRT-LLM: quantizing a model with an EAGLE drafter — answered "not supported"
- **Question:** Is there a working way to quantize (FP8/INT8) a model that has an EAGLE drafter?
- **Who tried:** `geaned`; answered by `hchings`.
- **URL:** https://github.com/NVIDIA/TensorRT-LLM/issues/3207
- **VERBATIM (`geaned`, the ask):** "So to me it seems that there is currently no working solution for quantizing the model with an EAGLE drafter. If so, could you please give me an advice on which speculative decoding technique to try that works correctly at the moment and supports FP8 quantization?"
- **VERBATIM (`hchings`, NVIDIA, 2025-04-02):** "Hi @geaned, the [Support Matrix](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/eagle#support-matrix) of the Eagle example README suggests that FP8/INT8 are not supported. @laikhtewari or other colleagues please confirm."
- READ BODY

### A10. `deepseek_nextn` hardcodes `quant_config=None` for modelopt_fp4 — SGLang, OPEN, flagged but not fixed
- **Question:** Why does a quantized NextN draft head get built in BF16 while checkpoint tensors are FP4-packed?
- **Who tried:** `chuck-ads`.
- **URL:** https://github.com/sgl-project/sglang/issues/36599
- **VERBATIM (the defect):** "`deepseek_nextn.py` hardcodes `quant_config = None` whenever the target model is `modelopt_fp4`"
- **VERBATIM:** "That's correct for DeepSeek V3/R1 NVFP4 checkpoints, whose MTP head ships in BF16 — but **GLM-5.3-Flash NVFP4 quantizes its NextN layer**"
- **VERBATIM (the workaround that worked):** "Keep the quant config for the NextN draft … With that one-line change … the NEXTN draft loads and speculative decoding runs: single-stream 14.2 → 20.6 tok/s on our 2-Spark setup."
- **VERBATIM (why the flag is useless):** "it took `--speculative-draft-model-quantization modelopt_fp4` just to get a quant config to the draft at all … and then this override silently discarded it, which makes the flag a no-op for this case."
- READ BODY

---

## OPEN (explicitly unsolved)

| # | Who is asking | What's missing | URL | VERBATIM |
|---|---|---|---|---|
| O1 | `soyr-redhat` | NVFP4 compressed-tensors drafter gives 0% acceptance under CUDA-graph capture | https://github.com/vllm-project/vllm/issues/54997 | "An MTP/eagle draft model with **compressed-tensors NVFP4 (w4a4)** MLP layers gives **0% spec-decode acceptance** with the default engine settings. Disabling CUDA-graph capture fixes it completely." — table: default 0% / cudagraph off 77.5% / `--enforce-eager` 92.3%. Label: `speculative-decoding`. State: **Closed** (not not_planned); only one comment; no maintainer explanation. |
| O2 | `martyrant`, `hlin99`, `yma11` (Intel) | DFlash2 draft 0% acceptance at float16 on XPU | https://github.com/vllm-project/vllm/issues/55250 | Title: "[Bug]: DFlash2 draft gets 0% acceptance with --dtype float16 on XPU (bf16 works) — Qwen3.8-27B + incoai/Qwen3.8-27B-DFlash2". Labels: `quantization,intel-gpu`. OPEN, 10 comments. |
| O3 | `eisbaw`, `jschmied` | DFlash fused-KV projection silently corrupts any weight-quantized drafter | https://github.com/vllm-project/vllm/issues/51581 | Title: "DFlash fused-KV projection calls F.linear on a sliced qkv_proj weight — breaks (and can silently corrupt) any weight-quantized drafter". OPEN. |
| O4 | `pavelgein` (+ `he-yufeng`, `fynnsu`) | An EAGLE-3 model trained with Speculators cannot be loaded by vLLM | https://github.com/vllm-project/vllm/issues/54526 | Title: "[Bug]: Cannot load an Eagle3 model, trained with Speculators". OPEN. Companion: https://github.com/vllm-project/speculators/issues/1065 "[Bug]: Config mismath between Speculators and vLLM". |
| O5 | `eleqtrizit`, `josephkern`, `WindChimeRan`, `alkari` | MTP + NVFP4 weight shape mismatch (long-running, `stale`) | https://github.com/vllm-project/vllm/issues/35031 | Title: "[Bug]: MTP Speculative Decoding with NVFP4: Weight Shape Mismatch". Labels `bug,stale`. 13 comments. |
| O6 | `noonghunna` | Quantized DFlash2 draft silently yields ~0% acceptance in SGLang, no error/warning | https://github.com/sgl-project/sglang/issues/39087 | "A **quantized** DFlash2 draft checkpoint loads into `DFlash2DraftModel`, serves, and produces drafts that are rejected ~100% of the time. There is **no error and no warning** — acceptance simply collapses from ~3.7 to ~1.0". Table: compressed-tensors 1.03 accept-len / 0.004 rate vs bf16 3.71 / 0.61. Filed 2026-09-11 by the same reporter whose vLLM issue #52873 was closed as checkpoint-specific. |
| O7 | `rnxrx` | compressed-tensors + NEXTN/MTP head fails at construction; error does not name the module | https://github.com/sgl-project/sglang/issues/38574 | "Serving a compressed-tensors checkpoint with `--speculative-algorithm NEXTN` crashes at model construction … The failing module is the **MTP head's** `mtp.layers.0.self_attn.qkv_proj` (found by instrumenting `prefix`; the error itself does not say)." Workaround: add `"re:mtp\\..*"` to `quantization_config.ignore`. Mechanism: "The MTP head is **not part of the HF model graph** that the quantizer traces, so it is neither quantized nor listed in `ignore`." |
| O8 | `Kokoro2336` | NEXTN/MTP MoE weights fail to load under TP>1 for Glm5NextForConditionalGeneration | https://github.com/sgl-project/sglang/issues/36653 | "NEXTN/MTP speculative decoding fails to load MTP MoE weights under TP>1 for Glm5NextForConditionalGeneration (GLM-5.3-Flash)". OPEN. |
| O9 | `doptime`, `blazingbhavneek`, `ormandj` | AWQ/INT8 quantized Qwen3.5-27B cannot start; blocks quantized-target deployment generally | https://github.com/sgl-project/sglang/issues/19406 | "Currently, loading the AWQ or INT8 quantized versions of `Qwen3.5-27B` fails entirely on the latest SGLang `main` branch." Label `inactive`; closed. The fix PR https://github.com/sgl-project/sglang/pull/20370 was **closed-unmerged after the author self-closed it**. |
| O10 | `Shubin-vadim` (Medusa) | Medusa only supports Llama/Mistral; no maintainer answer in 18 months | https://github.com/FasterDecoding/Medusa/issues/132 | "Hey guys! Is there any plan to support other types of LLMs besides Llama and Mistral?" — asked 2025-03-04, still OPEN, zero maintainer replies. Last repo commit 2024-04-18. |
| O11 | `1benwu1` (EAGLE) | Multimodal MTP not supported in the EAGLE reference repo | https://github.com/SafeAILab/EAGLE/issues/331 | "Thx for the great work. I wonder if the authors would consider updating for MM MTP" — OPEN since 2026-02-28. |
| O12 | EAGLE maintainers (their own TODO) | **Official EAGLE-3 for Qwen-3 is still an unchecked TODO**, as is EAGLE-4 | https://github.com/SafeAILab/EAGLE | README "## Todo" list: "- [x] Training code of EAGLE-3. - [x] Support LLaMA-4. - [ ] Support official EAGLE-3 for Qwen-3. - [ ] EAGLE-4." Also: "We strongly recommend using [SpecForge] for out-of-the-box training of EAGLE-3 with SGLang." |
| O13 | Medusa maintainers (their own ROADMAP) | Batched inference, fine-grained KV cache mgmt, and vLLM/llama.cpp/exllama integration all unchecked | https://github.com/FasterDecoding/Medusa/blob/main/ROADMAP.md | "## Functionality - [ ] Batched inference - [ ] Fine-grained KV cache management" … "### Serving - [ ] [vllm](https://github.com/vllm-project/vllm)" |
| O14 | `hernandez42` | Qwen3.x MTP not supported in llama.cpp at all | https://github.com/ggml-org/llama.cpp/issues/28051 | Title: "Support Qwen3.x MTP (Multi-Token Prediction) for speculative decoding" — OPEN, filed 2026-08-30. |
| O15 | `cplusplus2` | MTP retains inter-request state → non-determinism + degradation | https://github.com/ggml-org/llama.cpp/issues/26425 | Title: "Eval bug: MTP retains inter-request state causing non-deterministic output and model degradation (Qwen3.6-35B-A3B-MTP)" — OPEN. Same failure family as vLLM #52873 and llama.cpp #23636. |
| O16 | `zsogitbe` | MTP/draft spec decode not exposed in the public C API | https://github.com/ggml-org/llama.cpp/issues/27469 and https://github.com/ggml-org/llama.cpp/pull/27788 | "Feature Request: Expose Speculative Decoding (MTP / Draft) in public C API (llama.h) for downstream bindings" — OPEN. |
| O17 | `h-guo18` | EAGLE-3 produces gibberish on Llama-3.1-8B | https://github.com/NVIDIA/TensorRT-LLM/issues/11125 | Title: "[Bug]: Gibberish outputs on Llama8B with Eagle3", label `bug`, TRT-LLM 1.2.0rc8. **[PARTIAL]** |
| O18 | `transcend-0` | transformers has no tree/multi-candidate speculative search | https://github.com/huggingface/transformers/issues/39684 | "Add multi-candidate & tree search for assisted decoding (speculative decoding)" — OPEN. **[PARTIAL]** |
| O19 | `Kissmetothemoon` | transformers has no budget-bounded acceptance control for assisted decoding | https://github.com/huggingface/transformers/issues/48636 | "[Feature Request] Approximate Speculative Decoding (ASD): training-free, budget-bounded acceptance for greedy assisted decoding" — OPEN. **[PARTIAL]** |
| O20 | `pavelgein` (speculators) | Speculators' Eagle3 conversion silently writes random weights on key mismatch | https://github.com/vllm-project/speculators/issues/1114 | Title: "[Bug]: Eagle3 conversion writes randomly initialized weights when a checkpoint key does not match, instead of failing" — OPEN. |

---

## Cross-cutting patterns (no advice, just the observed shape)

1. **The "quantized drafter" failure is reported in BOTH engines, in BOTH directions.**
   Quantized *target* breaking the *head* (vLLM #54997, #55250, #35031, #51581; SGLang #36599,
   #38574, #36653, #39087, #38366), and quantized *drafter* breaking itself.
2. **Silent failure is the recurring complaint.** vLLM #54997 ("0% acceptance… output is incoherent"),
   SGLang #39087 ("no error, no warning"), vLLM #51581 ("can silently corrupt"),
   speculators #1114 ("writes randomly initialized weights … instead of failing").
3. **MTP heads are structurally not in the HF model graph**, so quantizers skip them
   (SGLang #38574) and engines then guess (SGLang #36599).
4. **Long-context / boundary-condition acceptance collapse recurs independently in three repos:**
   llama.cpp #23636 (KV-slot boundary), vLLM #52873 (cumulative-work threshold), llama.cpp #26425
   (inter-request state).
5. **The reference repos have diverged in health:** EAGLE alive (2026-02-19) and EAGLE-3 training
   handed off to SpecForge (2026-09-10); Medusa fossilised (2024-04-18) with 51 open issues and no
   maintainer replies; Hydra and self-speculative have no repo at all.

---

## Search log (verbatim)

### vLLM
```
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aunmerged+eagle
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+eagle&sort%3Aupdated-desc
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+medusa
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+mtp
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+wontfix+speculative&sort%3Aupdated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+eagle3&sort%3Aupdated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+mtp&sort%3Aupdated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+medusa&sort%3Aupdated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+%22acceptance+rate%22+speculative&sort%3Aupdated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+label%3A%22speculative-decoding%22&sort%3Aupdated-desc
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+delta
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+deltanet+rollback
https://github.com/vllm-project/vllm/issues/52873
https://github.com/vllm-project/vllm/issues/43559
https://github.com/vllm-project/vllm/issues/52873?_pjax=1
https://github.com/vllm-project/vllm/issues/52873?plain=1
https://github.com/vllm-project/vllm/pull/52816
https://github.com/vllm-project/vllm/pull/40914
https://github.com/vllm-project/vllm/pull/48375
https://github.com/vllm-project/vllm/pull/50021
https://github.com/vllm-project/vllm/pull/39931
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
https://github.com/vllm-project/vllm/issues/39931
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
https://github.com/FasterDecoding/hydra            -> HTTP 404
https://github.com/FasterDecoding/self-speculative -> HTTP 404
https://api.github.com/search/issues?q=repo:FasterDecoding/Medusa+is:issue+state:open&per_page=15&sort=created&order=desc
https://api.github.com/repos/{vllm-project/speculators,SafeAILab/EAGLE,FasterDecoding/Medusa,sgl-project/SpecForge}  -> HTTP 403 (core rate limit exhausted)
```

### NOT completed before the time call (gaps)
- TensorRT-LLM Medusa deprecation/removal PR — not located.
- HF discussions `unsloth/Qwen3.6-27B-MTP-GGUF#32` and `sakamakismile/Qwen3.6-27B-NVFP4#7` — not fetched.
- vLLM PR #40914 / #48375 / #50021 thread text — pages fetched and state confirmed (OPEN / OPEN / OPEN; #52816 MERGED) but comment bodies not extract-then-quoted.
- vLLM issue #43559 and #55605/#55357 bodies read only in part.
- SGLang offloader tied-weights bug and CUTE_DSL Ampere cuda-graph hang — searched, not positively identified.
- vLLM #39931 is **NOT** a DeltaNet-rollback PR — it is "[Feature] TurboQuant: support hybrid models and uniform quantization" by JartX, **MERGED**. The reported "DeltaNet rollback support" issue was not found under that number or under `is:issue delta`.
