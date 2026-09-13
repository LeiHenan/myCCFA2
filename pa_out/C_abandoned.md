# C — ATTEMPTED-AND-ABANDONED: Speculative-Decoding DRAFT MODELS

**Scope:** standalone small draft models, distilled drafts, self-distillation, online/on-the-fly draft training, draft fine-tuning. Emphasis on things *tried and given up*, closed **wontfix / not planned / stale / duplicate / by design**, or reported as **negative results**.

**Method note on evidence quality:** GitHub issue-search HTML *and* the GitHub search API were used. The API returns `state_reason`, which is how "not planned" is confirmed below. **GitHub issue HTML frequently does not render comments**, so where a maintainer reply was not retrievable, the row quotes the *issue author's own words* and the closure state is reported separately. No maintainer quote has been invented.

**Confirmed limitation of this sweep:** `github.com` HTML and `api.github.com` were intermittently rate-limited/throttled during the sweep from this network. Rows marked `TITLE ONLY` are ones where only the search-result title and API-reported state were retrievable.

---

## A. vLLM — standalone draft model: removed, then re-introduced (the canonical removal)

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| A1 | Standalone ("classic") draft-model speculative decoding — the original v0 path | vLLM | **Removed / not supported in v0.10+ and absent from V1** (later re-added); documented as a permanent known incompatibility for that version range | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/README.md | READ BODY |
| A2 | Same removal, as experienced by a user on v0.13.0 | forever10086 / vLLM | **not planned** (+ `stale`); closed 2026 | https://github.com/vllm-project/vllm/issues/31883 | READ BODY |
| A3 | Draft-model spec decode **on CPU** | ganeshr10 / vLLM | **not planned** (+ `stale`) | https://github.com/vllm-project/vllm/issues/28384 | READ BODY |
| A4 | Question "why is this not supported anymore?" | luxisme, tomasruizt / vLLM Forums | **Removal confirmed by the PR author who re-added it** | https://discuss.vllm.ai/t/standalone-draft-model-spec-decode-support-in-v0-x-and-v1/2241 | READ BODY |
| A5 | Re-introduction of draft models to V1 | tomasruizt / vLLM | **Reversed the removal** (context for A1–A4) | https://github.com/vllm-project/vllm/pull/24322 | READ BODY |

**VERBATIM QUOTE (A1):** "Speculative decoding with draft models is not supported in `vllm<=0.10.0`" — vLLM official docs, "Known Feature Incompatibility", https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/README.md

**VERBATIM QUOTE (A2):** "NotImplementedError: Speculative decoding with draft model is not supported yet. Please consider using other speculative decoding methods such as ngram, medusa, eagle, or mtp." — runtime error text pasted by issue author forever10086, https://github.com/vllm-project/vllm/issues/31883

**VERBATIM QUOTE (A3):** "NotImplementedError: Draft model speculative decoding is not supported yet. Please consider using other speculative decoding methods such as ngram, medusa, eagle, or mtp." … "This branch is the PARD implementation of Speculative decoding for V0. However, this was done a few months ago and is unsupported with V1." — ganeshr10, https://github.com/vllm-project/vllm/issues/28384

**VERBATIM QUOTE (A4):** "Standalone draft model is removed in v0.x releases (x>10) … And v1 does not support this either. Can I know why this is not supported anymore?" — luxisme, https://discuss.vllm.ai/t/standalone-draft-model-spec-decode-support-in-v0-x-and-v1/2241

**VERBATIM QUOTE (A4b):** "@RunLLM is wrong. Support for draft model was reintroduced to V1 this week in this PR: https://github.com/vllm-project/vllm/pull/24322 (I'm the PR author)." — tomasruizt (the PR author), same URL

---

## B. vLLM — MLP speculator (draft-head family) left un-routed

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| B1 | `mlp_speculator` method — still exposed in config/docs/examples but never routed in V1 | fenghourun / vLLM | **not planned** (closed 2026-07-07) | https://github.com/vllm-project/vllm/issues/47825 | READ BODY |
| B2 | IBM `llama3-70b-accelerator` MLP speculator checkpoints | kylesayrs / vLLM | **not planned** + `stale` (closed 2026-06-09) | https://github.com/vllm-project/vllm/issues/34106 | READ BODY |

**VERBATIM QUOTE (B1):** "mlp_speculator is still exposed as a speculative method (in the SpeculativeMethod literal, auto-detected from model_type, with a model impl, config parser, docs page, and example) but nothing in vllm/v1/ routes it, so it crashes at engine init." … "grep -rn "mlp_speculator" vllm/v1/ returns nothing (neither the classic nor the V2 runner has a proposer for it)." — fenghourun, https://github.com/vllm-project/vllm/issues/47825

**VERBATIM QUOTE (B2):** "MLP speculation using the ibm-ai-platform/llama3-70b-accelerator models seems to be broken. Attempting to run the following code using method="mlp_speculator" or method="draft_model" results in the following traceback" … "AttributeError: 'MLPSpeculatorConfig' object has no attribute 'num_attention_heads'" — kylesayrs, https://github.com/vllm-project/vllm/issues/34106

---

## C. vLLM — draft-model / draft-vocabulary pruning (FR-Spec): built, benchmarked, rejected

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| C1 | FR-Spec draft-vocabulary pruning as a first-class vLLM feature | jmamou / vLLM | **not planned** (closed 2025) | https://github.com/vllm-project/vllm/issues/24506 | READ BODY |
| C2 | `fr-spec` implementation PR (105 commits) | eitanturok / vLLM | **closed-unmerged** ("Closing this PR in favor of #29334") | https://github.com/vllm-project/vllm/pull/24343 | READ BODY |
| C3 | fr-spec, second attempt | eitanturok / vLLM | **closed-unmerged** | https://github.com/vllm-project/vllm/pull/29334 | READ BODY |
| C4 | "Spec decode with probs" (V1) | vLLM | **closed-unmerged** | https://github.com/vllm-project/vllm/pull/20459 | TITLE ONLY |

**VERBATIM QUOTE (C1):** "I have implemented the FR-Spec approach at the logits processor level, using AllowedTokenIdsLogitsProcessor. This implementation does not prune the draft model itself but allows evaluating acceptance rates under different draft pruning ratios." — jmamou, https://github.com/vllm-project/vllm/issues/24506
*(Author-reported MT-Bench draft acceptance rate: vanilla 27.8%; pruning ratio 0.1→28.3%, 0.25→28.6%, 0.5→28.6%, 0.75→27.2%, 0.9→25.9%, 0.99→18.8% — i.e. no gain at moderate pruning, degradation at aggressive pruning.)*

**VERBATIM QUOTE (C2, reviewer):** "We definetly get a speedup over vanilla. In eagle num-spec-tokens=1, the drafter forward pass takes 4% of the time of the target forward pass. So the drafter forward pass is not nearly a huge bottleneck, so we don't expect fr-spec to speed things up that much." … "frspec seems to increase throughput by 1.52% ( =100*(121.99-120.16)/120.16 ) in the benchmark above." — keyboardAnt (reviewer), https://github.com/vllm-project/vllm/pull/24343

**VERBATIM QUOTE (C2b, author):** "Closing this PR in favor of #29334 ." — eitanturok, https://github.com/vllm-project/vllm/pull/24343

---

## D. Draft-model *training* — SpecForge (SGLang): removed methods, unplanned RFCs, non-reproducible results

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| D1 | **VP-Drafter** training mode for DFlash | Ulitochka / SpecForge | **Silently removed**; config left stale, falls back to plain DFlash | https://github.com/sgl-project/SpecForge/issues/679 | READ BODY |
| D2 | Backbone/TTT-internal position trimming for **online EAGLE3** training | julyanghar / SpecForge | **not planned** (closed 2026-08-27) | https://github.com/sgl-project/SpecForge/issues/706 | READ BODY |
| D3 | SpecForge's "TTT" (test-time-training) online loop as an implementation of EAGLE-3 | haiduo / SpecForge | **Disputed as not the paper's concept** | https://github.com/sgl-project/SpecForge/issues/227 | READ BODY |
| D4 | DFlash draft training on OCR (70k samples) | xinanjiao / SpecForge | **Negative result**: 98% train acceptance → 10% inference acceptance | https://github.com/sgl-project/SpecForge/issues/533 | READ BODY |
| D5 | Reproducing official DFlash training numbers | 5SSjw / SpecForge | **Negative result**: 4.72× official vs 2.86× best after 142k steps | https://github.com/sgl-project/SpecForge/issues/469 | READ BODY |
| D6 | **Online** (on-the-fly) DFlash training | UltramanKuz / SpecForge | **OOM; user requests the abandoned offline path instead** | https://github.com/sgl-project/SpecForge/issues/521 | READ BODY |
| D7 | Disaggregated online training topology | junzhang-zj / SpecForge | **Negative result**: ~3.2× slower than DP | https://github.com/sgl-project/SpecForge/issues/718 | READ BODY |
| D8 | Official Qwen3-8B EAGLE3 online training recipe (LR 1e-4) | heiheiha798 / SpecForge | **Negative result**: grad-norm outliers, unstable | https://github.com/sgl-project/SpecForge/issues/577 | READ BODY |
| D9 | PRISM — multi-layer/step-disaggregated draft model | Akemiiii / SpecForge | **Open RFC challenging EAGLE-3 scaling cost** | https://github.com/sgl-project/SpecForge/issues/444 | READ BODY |

**VERBATIM QUOTE (D1):** "the VP-Drafter-specific prefix sampling, masking, and loss logic has been removed." … "As a result, the current qwen3-8b-dta.json configuration seems to silently fall back to regular DFlash training instead of enabling vp_drafter." — Ulitochka, https://github.com/sgl-project/SpecForge/issues/679

**VERBATIM QUOTE (D2):** "Are maintainers open to backbone-internal position trimming in the online TTT loop (behind a default-off flag), or is loss-end trimming the preferred boundary?" — julyanghar; issue closed as **not planned**, https://github.com/sgl-project/SpecForge/issues/706

**VERBATIM QUOTE (D3):** "TTT implementation is not the original EAGLE-3 paper's concept, but merely data augmentation." — haiduo, https://github.com/sgl-project/SpecForge/issues/227

**VERBATIM QUOTE (D4):** "This training used 70,000 data points, achieving a 98% acceptance rate during training. I used a version of VLLM that supports Dflash for inference, but the average acceptance rate was only 10%." — xinanjiao, https://github.com/sgl-project/SpecForge/issues/533

**VERBATIM QUOTE (D5):** "official baseline: speedup: 4.72x, τ: 5.97" vs "Step 142,000: speedup: 2.86×, τ: 3.55" … "In my runs, improvements are clearly diminishing, but even after ~12 epochs the metrics are still moving upward." — 5SSjw, https://github.com/sgl-project/SpecForge/issues/469

**VERBATIM QUOTE (D6):** "I would like to know if there are any plans for developing offline training for Dflash? The online training has caused the OOM issue for me" — UltramanKuz, https://github.com/sgl-project/SpecForge/issues/521

**VERBATIM QUOTE (D7):** "I'm training a draft model for Qwen3.6-35B-A3B using SpecForge and observing a significant wall-clock time difference between the disaggregated topology and the original DP (data-parallel) approach." — junzhang-zj (title: "~3.2x slower than DP"), https://github.com/sgl-project/SpecForge/issues/718

**VERBATIM QUOTE (D8):** "the default --learning-rate 1e-4 may be too aggressive for the Qwen3-8B EAGLE3 setting, at least during early training." … "with the official 1e-4 LR, the run shows many very large gradient-norm outliers in the middle of training" — heiheiha798, https://github.com/sgl-project/SpecForge/issues/577

**VERBATIM QUOTE (D9):** "SGLang also implemented multi-layer eagle worker. However, these changes may increase draft model inference overhead and the training cost." — Akemiiii, https://github.com/sgl-project/SpecForge/issues/444

---

## E. `speculators` (vLLM-project draft-training library) — rejected scope and removed training modes

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| E1 | Support for draft architectures **other than EAGLE-3** | scimg / speculators | **not planned** + `stale` | https://github.com/vllm-project/speculators/issues/231 | READ BODY |
| E2 | Multi-node draft training | Liccol / speculators | **not planned** | https://github.com/vllm-project/speculators/issues/356 | READ BODY |
| E3 | Ulysses sequence parallelism for draft training | momo609 / speculators | **not planned** | https://github.com/vllm-project/speculators/issues/338 | READ BODY |
| E4 | DFLASH windowed-attention backward Triton operator | Leslie360 / speculators | **not planned** | https://github.com/vllm-project/speculators/issues/1105 | READ BODY |
| E5 | **Hybrid training mode** (`--on-generate cache`) | WindChimeRan / speculators | **Removed, breaking change** (PR open) | https://github.com/vllm-project/speculators/pull/1036 | READ BODY |
| E6 | Legacy training accuracy & acceptance reports | speculators | **Removal PR** | https://github.com/vllm-project/speculators/pull/1102 | TITLE ONLY |

**VERBATIM QUOTE (E1):** "Any plan to support more kinds of drafts other than Eagle-3?" — scimg; closed as **not planned**, labels `enhancement`, `stale`, https://github.com/vllm-project/speculators/issues/231

**VERBATIM QUOTE (E5):** "Removes hybrid training mode (breaking) --on-generate cache kept freshly generated hidden states so later epochs could reuse them. It was broken on the Mooncake backend: MooncakeTransfer never overrode cache(), so it inherited the no-op hook on HiddenStatesTransfer, and because the cache branch bypassed the delete path entirely, every generated sample leaked into the store while nothing was ever cached — MooncakeTransfer.get_cached() returns None unconditionally, so each epoch regenerated everything from scratch." — WindChimeRan, https://github.com/vllm-project/speculators/pull/1036

**VERBATIM QUOTE (E4):** "At production sizes the dense backward needs ~69GB (OOM on 8×A800) and ignores the window structure." — Leslie360, https://github.com/vllm-project/speculators/issues/1105

---

## F. llama.cpp — MTP/draft heads: abandoned PR and "speculation is slower" results

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| F1 | MTP (nextn) speculative head for Qwen3.8-Flash-Next | routhjim / llama.cpp | **closed-unmerged** (author closed after bot flagged AI-generated PR text) | https://github.com/ggml-org/llama.cpp/pull/27842 | READ BODY |
| F2 | SYCL MTP on Intel Arc | R-SITES / llama.cpp | **not planned**; negative: 21% slower at 100% draft acceptance | https://github.com/ggml-org/llama.cpp/issues/23533 | READ BODY |
| F3 | Gemma 4 MTP merge (`#23398`) | UncleMart / llama.cpp | **not planned**; negative: 40 tok/s → ~4 tok/s | https://github.com/ggml-org/llama.cpp/issues/24266 | READ BODY |
| F4 | Pipelining `draft-mtp` → `ngram-mod` | ElSnacko / llama.cpp | **not planned** + `stale`; "no speedup, only verification overhead" | https://github.com/ggml-org/llama.cpp/issues/23184 | READ BODY |
| F5 | External DFlash drafter invariant at temperature 0 | llama.cpp | **not planned** | https://github.com/ggml-org/llama.cpp/issues/27975 | TITLE ONLY |
| F6 | Draft-cache replay honouring `p_min` | llama.cpp | **not planned**; benchmark inflation ~10× | https://github.com/ggml-org/llama.cpp/issues/26100 | TITLE ONLY |
| F7 | Gemma E2B draft | llama.cpp | **not planned** | https://github.com/ggml-org/llama.cpp/issues/22337 | TITLE ONLY |
| F8 | Speculative context memory fitting | llama.cpp | **not planned** | https://github.com/ggml-org/llama.cpp/issues/25408 | TITLE ONLY |
| F9 | Gemma 4 MTP "heuristic" n-max | llama.cpp | **not planned** | https://github.com/ggml-org/llama.cpp/issues/24768 | TITLE ONLY |
| F10 | SYCL `draft-mtp` memory/slowdown | llama.cpp | **not planned** | https://github.com/ggml-org/llama.cpp/issues/23203 | TITLE ONLY |

**VERBATIM QUOTE (F1):** "AI-generated content : While code is allowed to be generated by AI, please write the PR description and commit messages on your own without the help of AI." — ggml-gh-bot (automated PR checker), https://github.com/ggml-org/llama.cpp/pull/27842

**VERBATIM QUOTE (F2):** "MTP is ~21% slower than generating without speculation, despite 100% draft accuracy. This is the opposite of the expected result (typically 1.5-2x speedup on CUDA)." — R-SITES, https://github.com/ggml-org/llama.cpp/issues/23533

**VERBATIM QUOTE (F3):** "#23398 When I use gemma 4 12B on this new merge my tokens/sec drop to ~4 tokens/sec, previously I was getting 40+ tokens/sec." — UncleMart, https://github.com/ggml-org/llama.cpp/issues/24266

**VERBATIM QUOTE (F4):** "Adding ngram-mod independently on top provides no speedup, only verification overhead." — ElSnacko, https://github.com/ggml-org/llama.cpp/issues/23184

---

## G. Ollama — draft/DFlash support

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| G1 | DFlash speculative decoding in Ollama (`muse-glimmer:30b-nvfp4-dflash`) | jakubtomas-cz / Ollama | **not planned**; user measured no effect | https://github.com/ollama/ollama/issues/17683 | READ BODY |
| G2 | DRAFT-layer memory accounting for Gemma 4 MTP | yarrrly / Ollama | **Duplicate** | https://github.com/ollama/ollama/issues/17951 | READ BODY |

**VERBATIM QUOTE (G1):** "I'm able to get 25 tps which is respectable and on par with Meta video on their release blog … without the speculative decoding. This indicates to me that the muse-glimmer:30b-nvfp4-dflash doesn't really use the DFlash speculative decoding." — jakubtomas-cz; closed as **not planned**, https://github.com/ollama/ollama/issues/17683

**VERBATIM QUOTE (G2):** "A model created with a DRAFT layer (Gemma 4 E2B plus the official MTP drafter) loads and runs speculative decoding correctly, but Ollama cannot measure the draft model's memory and then reports the whole model as 315 MB in /api/ps and ollama ps." — yarrrly, https://github.com/ollama/ollama/issues/17951

---

## H. Other engines — draft/assisted decoding declined or disproven

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| H1 | Add speculative decoding (two-model draft/verify) to TGI | OliverFM / HF TGI | **not planned** + `Stale` (2023, canonical) | https://github.com/huggingface/text-generation-inference/issues/729 | READ BODY |
| H2 | Assisted Generation support in TGI | sujithjoseph / HF TGI | **not planned** + `Stale` | https://github.com/huggingface/text-generation-inference/issues/314 | READ BODY |
| H3 | Medusa acceleration in TGI | eurus-ch / HF TGI | **not planned**; "acceleration in doubt" | https://github.com/huggingface/text-generation-inference/issues/1503 | READ BODY |
| H4 | EAGLE-3 acceptance length at larger batch size | Wokzy / TensorRT-LLM | **not planned**; negative scaling | https://github.com/NVIDIA/TensorRT-LLM/issues/9208 | READ BODY |
| H5 | EAGLE-3 one-model config with FP8 target | ValeGian / TensorRT-LLM | **not planned**; FP8/INT8 unsupported for draft | https://github.com/NVIDIA/TensorRT-LLM/issues/7842 | READ BODY |

**VERBATIM QUOTE (H1):** "Adding this feature would require making TGI more generic, so that one can run multiple models at once. We would need to make sure that this does not degrade performance or reliability for the single model use case." — OliverFM, https://github.com/huggingface/text-generation-inference/issues/729

**VERBATIM QUOTE (H3):** "Speculative(3) seems to accelerate significantly, but I didn't observe the same result with vLLM benchmark and in real use on the same A30*4. Their results look like Medusa doesn't have that much effect." — eurus-ch, https://github.com/huggingface/text-generation-inference/issues/1503

**VERBATIM QUOTE (H4):** "Eagle3 acc len decreases with bigger batch size" — Wokzy (issue title), closed as **not planned**, https://github.com/NVIDIA/TensorRT-LLM/issues/9208

**VERBATIM QUOTE (H5):** "The Support Matrix of the Eagle example README suggests that FP8/INT8 are not supported. Even if the Eagle3 model is in FP16, using it with an FP8 target model with the one-model configuration would result in the following error" — ValeGian, https://github.com/NVIDIA/TensorRT-LLM/issues/7842

---

## I. vLLM — negative results and declined draft-model features (systematic sweep, `reason:not_planned`)

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| I1 | **Distilled** small draft model (logits distillation, "very high" acceptance) | maiiabocharova / vLLM | **not planned**; ~30% *slowdown* | https://github.com/vllm-project/vllm/issues/15025 | READ BODY |
| I2 | Small Qwen3 model as draft for Qwen3-32B-FP8 | la1ty / vLLM | **not planned**; slower (13–14 → 10–11 tok/s) | https://github.com/vllm-project/vllm/issues/21278 | READ BODY |
| I3 | **Multimodal** draft models (external drafters) | benchislett / vLLM | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/33458 | READ BODY |
| I4 | Speculative Speculative Decoding (async draft/verify overlap) | celsowm / vLLM | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/36037 | READ BODY |
| I5 | Tree speculative decode | DingYiBin / vLLM | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/37396 | READ BODY |
| I6 | Dynamic pruning of EAGLE-3 draft trees | supertanziang / vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/41823 | READ BODY |
| I7 | Dynamic Speculation Length w/ confidence early-exit | jmamou / vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/36657 | READ BODY |
| I8 | Persistent in-place drafting attention metadata | chanh / vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/49488 | READ BODY |
| I9 | n-gram + suffix in `model_runner_v2` | lio1226 / vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/38069 | READ BODY |
| I10 | "Tracking Spec Decode Support" matrix | MatthewBonanni / vLLM | **not planned** (tracking abandoned) | https://github.com/vllm-project/vllm/issues/27691 | READ BODY |
| I11 | CI/CD regression testing for spec decode | rahul-tuli / vLLM | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/28135 | READ BODY |
| I12 | Draft model with shorter context than target | dsingal0 / vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/7859 | READ BODY |
| I13 | EAGLE3/DFlash TTFT at P99 | KlyzhenkoVadim / vLLM | **not planned**; TTFT regression | https://github.com/vllm-project/vllm/issues/39790 | READ BODY |
| I14 | Lookahead decoding port to vLLM core | SupreetSinghPalne / vLLM | **closed-unmerged**, branch deleted | https://github.com/vllm-project/vllm/pull/23388 | READ BODY |
| I15 | "Automate Speculative Decoding" RFC | vLLM | **not planned** (2024, canonical) | https://github.com/vllm-project/vllm/issues/4565 | TITLE ONLY |
| I16 | Speculative Streaming (draft-free) | vLLM | **not planned** (2024, canonical) | https://github.com/vllm-project/vllm/issues/2943 | TITLE ONLY |
| I17 | GPT-OSS 20B as drafter | vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/33133 | TITLE ONLY |
| I18 | EAGLE-3 draft model length > 2048 | vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/23072 | TITLE ONLY |
| I19 | EAGLE-3 second-token acceptance collapse | vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/33330 | TITLE ONLY |
| I20 | Non-parallel spec-decode draft/target hidden-size mismatch | vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/37966 | TITLE ONLY |
| I21 | Medusa choice-tree specification | vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/20813 | TITLE ONLY |
| I22 | Medusa with TP > 1 (hang) | vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/16477 | TITLE ONLY |
| I23 | Deploying the speculative model on a second device | vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/12200 | TITLE ONLY |
| I24 | Per-sequence speculative decoding | vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/17984 | TITLE ONLY |
| I25 | Medusa throughput at concurrency 2 | vLLM | **not planned**; slower than naive | https://github.com/vllm-project/vllm/issues/10031 | TITLE ONLY |
| I26 | Medusa vs baseline performance | vLLM | **not planned**; "poor performance than baseline" | https://github.com/vllm-project/vllm/issues/6777 | TITLE ONLY |
| I27 | Speculative performance generally | vLLM | **not planned**; "almost same or lower" | https://github.com/vllm-project/vllm/issues/5239 | TITLE ONLY |
| I28 | "why speculate decoding is slower than normal decoding?" | vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/8439 | TITLE ONLY |

**VERBATIM QUOTE (I1):** "I trained the small model using logits distillation of main model, so it has a good level of generation (acceptance rate is very high) Still I get consistent performance drop ~30% in terms of speed when using 5 speculative tokens, when I reduce number speculative tokens - speed increases, but the best speed in achieved when using main model only without speculative." — maiiabocharova, https://github.com/vllm-project/vllm/issues/15025

**VERBATIM QUOTE (I2):** "But it turns out that it leads to a slower inference speed (around 10-11 tokens/s). I don't know if it's normal." — la1ty, https://github.com/vllm-project/vllm/issues/21278

**VERBATIM QUOTE (I3):** "Currently, multimodal EAGLE drafters are supported, but only for EAGLE drafters and only when not using parallel drafting. This is due to some complexity in managing the MRoPe position embeddings and multimodal state that has not been explored for external drafters." — benchislett, https://github.com/vllm-project/vllm/issues/33458

**VERBATIM QUOTE (I5):** "I was testing tree speculative decoding and noticed that this feature is not yet fully implemented. It also doesn't appear to be on the current roadmap." — DingYiBin, https://github.com/vllm-project/vllm/issues/37396

**VERBATIM QUOTE (I8):** "Why the existing fast path doesn't apply … The fast path is not applicable to sequential MTP/EAGLE drafting." — chanh, https://github.com/vllm-project/vllm/issues/49488

**VERBATIM QUOTE (I12):** "Cannot handle cases where distributed draft workers generate no tokens" — error text pasted by dsingal0 (target 128K ctx, draft 32K ctx), https://github.com/vllm-project/vllm/issues/7859

**VERBATIM QUOTE (I13):** "While the expected improvement in Time-Per-Output-Token (TPOT) is confirmed, the TTFT degradation appears to be an inherent overhead of the speculative decoding process that might be more pronounced than previously understood." — KlyzhenkoVadim, https://github.com/vllm-project/vllm/issues/39790

**VERBATIM QUOTE (I14):** "Port the Lookahead decoding feature from PR #72 in vllm-project/vllm-gaudi into HabanaAI/vllm-fork. This enables lookahead decoding in the Gaudi fork." — SupreetSinghPalne; PR closed, branch deleted, never merged, https://github.com/vllm-project/vllm/pull/23388

---

## J. Papers — explicit negative results on draft models / draft training

| # | What was tried | Who / project | Outcome | URL | Evidence |
|---|---|---|---|---|---|
| J1 | PEFT-BD: LoRA-like adapter as a block-diffusion drafter on the *same* backbone | Abdurrahman Javat et al. | **Negative result** (self-described) | https://arxiv.org/abs/2607.12422 | READ BODY |
| J2 | Test-time training (TTT) as the fix for long-range draft decay | KVShot authors | **Negative result**: TTT does not fix decay; speedups marginal | https://arxiv.org/abs/2604.26412 | READ BODY |
| J3 | Diffusion/block-parallel drafters on multimodal targets | Multimodal SD survey | **Negative results at high resolution** (0.83×–0.90×) | https://arxiv.org/abs/2608.20743 | READ BODY |
| J4 | Online LoRA alignment of draft vocabulary + parameters | EvoSpec authors | **Negative framing**: full-parameter online updates too costly | https://arxiv.org/abs/2605.27390 | READ BODY |

**VERBATIM QUOTE (J1):** "Despite these advantages, PEFT-BD does not yield a practical speedup in our Qwen3-0.6B experiments." … "the drafter is parameter-efficient but not compute-efficient." … "Longer accepted prefixes alone cannot compensate when draft computation remains verifier-scale." — Javat et al., https://arxiv.org/abs/2607.12422

**VERBATIM QUOTE (J2):** "Existing work attributes this decay to train-inference mismatch and proposes test-time training (TTT) as a remedy, yet we observe that long-range decay persists even in TTT-trained drafters." … "Extensive evaluations on Qwen3-8B show that KV-Reuse improves long-range acceptance, although end-to-end speedups remain marginal under current training pipelines." — KVShot authors, https://arxiv.org/abs/2604.26412

**VERBATIM QUOTE (J3):** "At 8K, all settings fall to or below the autoregressive baseline, with speedups of 0.90× and 0.83×" … "on the Qwen3-VL family, EAGLE-3 fails to provide speedup on SGLang (0.71× for 4B and 0.88× for 8B)." … "Even when we explicitly train a DFlash-style block-parallel drafter for Qwen3-VL-8B with multimodal data using SpecForge, it reaches only 2.14× speedup on HF, below the 2.60×" — Multimodal SD survey authors, https://arxiv.org/abs/2608.20743

**VERBATIM QUOTE (J4):** "Online alignment improves draft quality, but full-parameter updates introduce substantial memory and latency overhead." — EvoSpec authors, https://arxiv.org/abs/2605.27390

---

## Cross-cutting patterns (observations only, no recommendations)

1. **"not planned" is the dominant closure mode**, not "wontfix" — the literal strings `wontfix` and the `not planned` *label* returned **zero** results on vLLM, SGLang, SpecForge and speculators issue searches. The `state_reason: not_planned` field (plus the `stale` label) is where the signal actually lives.
2. **Draft-model support in vLLM was removed and later restored** (A1–A5). Any claim that vLLM "doesn't support draft models" is version-dependent: not in `<=0.10.0`, present again after PR #24322.
3. **Draft *pruning* / cheaper-drafter work was rejected on measured grounds, not on principle** (C1–C3): reviewer measurement was ~1.5% throughput.
4. **The strongest repeated negative result is "the drafter isn't the bottleneck"** — vLLM #15025, #21278; llama.cpp #23533, #24266; TGI #1503; and paper J1 all report slowdown or negligible gain despite high/perfect acceptance.
5. **Online / on-the-fly draft training is the least reproducible area**: SpecForge reports training-vs-inference acceptance gaps (D4), non-reproducible baselines (D5), OOM (D6), slower disaggregated topologies (D7), unstable LR (D8), and a disputed TTT implementation (D3); J2 reports TTT does not fix long-range decay.
6. **Draft training libraries narrowed scope rather than expanded it**: speculators declined non-EAGLE-3 drafts (E1), multi-node (E2) and SP (E3), and removed a hybrid training mode (E5).

---

## Search log (every query verbatim + engine)

**web_search (DeepSeek web search):**
1. `speculative decoding draft model removed deprecated wontfix github issue`
2. `vLLM remove MLP speculator speculative decoding v1 draft model deprecated`
3. `speculators library huggingface draft model training abandoned issue`
4. `online speculative decoding draft model training negative result paper`
5. `llama.cpp speculative decoding draft model "not planned" issue closed`
6. `ollama speculative decoding draft model issue wontfix closed`
7. `EAGLE draft model training tooling SGLang removed deprecated`
8. `self-speculative decoding layer skipping abandoned negative result`
9. `vLLM V0 removal speculative decoding methods no longer supported draft_model`
10. `"self-speculative decoding" abandoned "we did not observe" draft`
11. `SGLang EAGLE draft model training tooling deprecated removed`
12. `Medusa vs EAGLE criticism draft model "not worth" overhead`
13. `"online speculative decoding" draft model training not worth it follow-up 2026`
14. `speculative decoding draft model "by design" wontfix maintainer rejected`
15. `arxiv 2026 speculative decoding draft model negative result no speedup`
16. `EAGLE draft head training abandoned replaced by MTP 2026`
17. `speculative decoding draft model paper 2026 "we find that" no improvement over EAGLE`
18. `"draft model" speculative decoding removed deprecated vLLM 0.10 release notes`
19. `SGLang speculative decoding draft model support removed issue`
20. `"not planned" speculative decoding draft github issue 2026`
21. `"DistillSpec" draft model distillation abandoned follow-up negative`
22. `self-speculative decoding layer skip abandoned "we found" overhead`
23. `DeepSeek MTP head separate draft model not needed argument`
24. `huggingface transformers assisted generation assistant model deprecated removed`
25. `arXiv 2026 speculative decoding draft model "does not help" limitation`
26. `"speculative decoding" survey 2026 draft model limitations open problems`
27. `online draft model training on-the-fly speculative decoding abandoned latency overhead`
28. `draft model distillation speculative decoding negative result acceptance rate low`
29. `"online speculative decoding" 2026 draft model retraining "not" practical`
30. `arXiv 2026 withdrawn retracted speculative decoding draft model`
31. `Medusa heads "not planned" OR deprecated OR removed speculative decoding 2026`
32. `self-speculative decoding early exit "not supported" OR abandoned inference engine 2026`
33. `"Training Domain Draft Models for Speculative Decoding" limitations negative findings ICLR 2025`
34. `huggingface transformers assisted generation assistant model slow overhead issue`
35. `vLLM speculative decoding draft model deprecation removal PR 2025 "removed"`
36. `SpecForge online training EAGLE3 vs offline "not worth" OR OOM OR slower`

**GitHub issue-search HTML URLs (fetch + strip):**
37. `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+speculative+draft+wontfix`
38. `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+label%3A%22not+planned%22+speculative`
39. `https://github.com/sgl-project/sglang/issues?q=is%3Aissue+is%3Aclosed+label%3A%22not+planned%22+draft`
40. `https://github.com/sgl-project/SpecForge/issues?q=is%3Aissue+draft`
41. `https://github.com/vllm-project/speculators/issues?q=is%3Aissue`
42. `https://github.com/ggml-org/llama.cpp/issues?q=speculative+draft+closed`
43. `https://github.com/ollama/ollama/issues?q=speculative+draft`
44. `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+medusa+speculator`
45. `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+MLP+speculator`
46. `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+%22draft+model%22+in%3Atitle`
47. `https://github.com/sgl-project/sglang/issues?q=is%3Aissue+%22draft+model%22+in%3Atitle`
48. `https://github.com/sgl-project/sglang/issues?q=is%3Aissue+speculative+deprecat`
49. `https://github.com/huggingface/transformers/issues?q=is%3Aissue+assisted+generation+draft`
50. `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+label%3Astale+speculative`
51. `https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+spec+decode`
52. `https://github.com/sgl-project/sglang/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+draft`
53. `https://github.com/vllm-project/speculators/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged`
54. `https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative`

**GitHub search API (api.github.com/search/issues, `reason:` qualifier):**
55. `repo:vllm-project/vllm is:issue reason:not_planned speculative`
56. `repo:vllm-project/vllm is:issue reason:not_planned "draft model"`
57. `repo:vllm-project/vllm is:issue reason:not_planned medusa`
58. `repo:sgl-project/SpecForge is:issue reason:not_planned`
59. `repo:vllm-project/speculators is:issue reason:not_planned`
60. `repo:sgl-project/sglang is:issue reason:not_planned draft`
61. `repo:sgl-project/sglang is:issue reason:not_planned speculative`
62. `repo:ggml-org/llama.cpp is:issue reason:not_planned draft`
63. `repo:ollama/ollama is:issue reason:not_planned speculative OR draft`
64. `repo:huggingface/transformers is:issue reason:not_planned assistant model speculative`
65. `repo:NVIDIA/TensorRT-LLM is:issue reason:not_planned draft speculative`
66. `repo:huggingface/text-generation-inference is:issue reason:not_planned speculative`
67. `repo:ml-explore/mlx-lm is:issue reason:not_planned draft speculative`  *(returned 0 results)*
68. `repo:vllm-project/vllm is:pr is:unmerged lookahead decoding`
69. `repo:vllm-project/vllm is:issue reason:not_planned lookahead`
70. `repo:vllm-project/vllm is:issue reason:not_planned "self-speculative"`  *(returned 0 results)*
71. `repo:huggingface/transformers is:issue reason:not_planned "assistant model"`  *(returned 0 results)*
72. `repo:vllm-project/vllm is:pr fr-spec`
73. `repo:vllm-project/vllm is:pr is:unmerged "draft model"`
74. `repo:sgl-project/SpecForge is:issue dta OR dflash OR drafter closed`
75. `repo:vllm-project/speculators is:issue closed draft model training`
76. `repo:huggingface/transformers is:issue reason:not_planned assisted OR speculative OR assistant`
77. `repo:vllm-project/vllm is:issue reason:not_planned medusa OR "mlp speculator" OR "draft model"`
78. `repo:vllm-project/speculators is:issue reason:not_planned OR label:stale`
79. `repo:sgl-project/sglang is:pr is:unmerged draft speculative`
80. `repo:vllm-project/vllm is:issue reason:not_planned "draft model" OR drafter OR eagle OR medusa`
81. `repo:sgl-project/sglang is:issue is:closed label:stale draft OR speculative`  *(returned 0 results)*
82. `repo:vllm-project/speculators is:pr is:unmerged training`

**Direct page fetches (issue/PR/paper URLs listed in the tables above; ~60 pages fetched and stripped to text).**

---

## Negative / empty results (worth recording)

- Literal `wontfix` search on vLLM issues: **no results**. The `not planned` *label* search on vLLM and SGLang: **no results**. Signal lives in `state_reason=not_planned` and the `stale` label.
- `repo:ml-explore/mlx-lm is:issue reason:not_planned draft speculative`: **0 results**.
- `repo:huggingface/transformers is:issue reason:not_planned "assistant model"`: **0 results** (transformers' assisted-generation issues are overwhelmingly closed *completed*, not declined).
- `repo:sgl-project/sglang is:issue is:closed label:stale draft OR speculative`: **0 results**.
- SGLang produced almost no "abandoned draft" signal via `not_planned` (5 hits, none about draft-model support being dropped) — its abandonment signal is concentrated in SpecForge instead.
