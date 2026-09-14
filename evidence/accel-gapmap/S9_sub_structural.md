# S9 — Structural model changes at inference time, no training
### (layer pruning/dropping · head & structured-width pruning · `num_hidden_layers` override · early exit / layer skipping)

Rig: 1× RTX PRO 6000 Blackwell 96 GB, sm120, single node, no NVLink/IB, vLLM 0.29.0 / SGLang 0.5.19, **Qwen3-4B dense full-attention**.
Every arXiv ID below was read back with `fetch.py arxiv <id>` and its returned TITLE is quoted in the row.
Honest scope note up front: **the engine-side surface is thin.** vLLM and SGLang ship *no* layer-pruning, head-pruning or
layer-skipping inference feature; the only shipped structural lever found is vLLM's generic `--hf-overrides` config override.
Most of the substantive evidence is therefore papers (A), unmet asks (B), and rejected/abandoned approaches (C).

---

### A9.1 | CLOSED
QUESTION: Does dropping whole transformer layers from an already-trained checkpoint work with NO retraining, and what does it cost?
WHO: Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, Daniel A. Roberts (Meta / MIT-CTP)
ARTIFACT: arXiv:2403.17887 — "The Unreasonable Ineffectiveness of the Deeper Layers" (title read back and confirmed)
URL: https://arxiv.org/abs/2403.17887 ; https://arxiv.org/html/2403.17887v2
STATUS: published (v1 26 Mar 2024, v2 3 Mar 2025)
EVIDENCE: READ BODY
QUOTE: "the simple heuristic performs quite poorly without healing the damage incurred by pruning: accuracy on the QA benchmarks decays rapidly to (near-) random with increased pruning fraction, and the loss begins to increase very rapidly even with small amounts of pruning."
QUOTE-2: "we explore reasoning related tasks (GSM8k and HellaSwag) and see that they are harmed by any amount of pruning"
NOTE (verbatim, what the headline no-retraining result actually covers): "Elaborating on the “optionality” of the final step, we find that the near-lack of performance degradation on question-answering benchmarks, cf. Figure 1 (d) and others in § 4.1, can be extended to greater pruning fractions with a small amount of finetuning."
NOTE: the paper's own accounting — QA-style multiple-choice survives; next-token loss and reasoning do not.

### A9.2 | CLOSED
QUESTION: Is there a genuinely training-free layer-removal method, and where does it break?
WHO: Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, Weipeng Chen (Baichuan Inc. / CAS)
ARTIFACT: arXiv:2403.03853 — "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect" (title read back and confirmed)
URL: https://arxiv.org/abs/2403.03853 ; https://arxiv.org/html/2403.03853v3
STATUS: published (v1 6 Mar 2024, v3 11 Oct 2024)
EVIDENCE: READ BODY
QUOTE: "When we remove 25% layers from Llama2-7B or Baichuan2-7B, the performance in generative tasks such as XSum and C3 deceases to nearly zero, although the performance decline was not as significant on the larger model of the 13B."
NOTE: the method is gradient-free (calibration set → Block Influence score → delete layers). Its own stated remedy is a retraining step that it says only "has the potential to mitigate this issue": "The post-training techniques discussed in Section 4.6 have the potential to mitigate this issue and warrant further exploration."

### A9.3 | CLOSED
QUESTION: Can early exit be added to a stock checkpoint without changing its weights?
WHO: Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A Aly, Beidi Chen, Carole-Jean Wu (Meta FAIR)
ARTIFACT: arXiv:2404.16710 — "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding" (title read back and confirmed)
URL: https://arxiv.org/abs/2404.16710 ; https://arxiv.org/html/2404.16710v4
STATUS: published (v1 25 Apr 2024, v4 18 Oct 2024)
EVIDENCE: READ BODY
QUOTE: "Our self-speculative decoding solution requires finetuning a model or pretraining it with our recipe, while the self-speculative decoding approach propoposed in Zhang et al. (2023) does not require changing a model’s weights."
QUOTE-2: "Moreover, LM heads in LLMs are trained to unembed embeddings from the last transformer layer. They were not trained to unembed from earlier layers."
NOTE (corroboration, different artifact — arXiv:2207.07061 "Confident Adaptive Language Modeling", title read back and confirmed, https://arxiv.org/html/2207.07061v2, READ BODY): "We train a dedicated linear classifier ... To avoid any impact on the core model’s performance, we train it as a second step where we freeze all parameters other than" — i.e. even the "lightweight" early-exit family trains an exit predictor.

### A9.4 | CLOSED
QUESTION: Is token-level early exit (skipping layers per token) achievable training-free in a decoder-only engine, and what is the blocker?
WHO: Yingtao Shen, An Zou (Peng Cheng Laboratory / Shanghai Jiao Tong University)
ARTIFACT: arXiv:2604.18396 — "River-LLM: Large Language Model Seamless Exit Based on KV Share" (title read back and confirmed)
URL: https://arxiv.org/abs/2604.18396 ; https://arxiv.org/html/2604.18396v3
STATUS: published (v1 20 Apr 2026, v3 25 May 2026)
EVIDENCE: READ BODY
QUOTE: "none of these strategies fully eliminates the penalty of KV Cache Absence. Neglecting KV integrity leads to severe performance degradation; recomputation significantly hampers the net acceleration; and imposing exit constraints severely limits the inherent potential of Token-level Exit."
QUOTE-2 (the paper's own limitations section): "As River-LLM is primarily optimized for token-level autoregressive decoding, its efficiency gains are most significant during the generation phase. Consequently, the speedup is less pronounced for prefill-dominant tasks, such as those within the MMLU benchmark, where a sequence-level exit strategy is currently applied."
QUOTE-3: "Our current evaluation focuses on representative models up to 8B parameters."
NOTE: KV Cache Absence is the concrete mechanism that breaks naive layer skipping in a decoder-only serving engine — a skipped layer leaves no K/V for later tokens. Hardware used: "All experiments are conducted on an NVIDIA A40 GPU." (single GPU, within rig class).

### A9.5 | CLOSED
QUESTION: Can structured WIDTH pruning (rows/columns, embedding dimension) be done post-training with no recovery fine-tuning?
WHO: Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman (Microsoft / ETH Zurich)
ARTIFACT: arXiv:2401.15024 — "SliceGPT: Compress Large Language Models by Deleting Rows and Columns" (title read back and confirmed)
URL: https://arxiv.org/abs/2401.15024 ; https://arxiv.org/html/2401.15024v2
STATUS: published (v1 23 Jan 2024)
EVIDENCE: READ BODY
QUOTE: "With SliceGPT we compress large models using a single GPU in just a few hours and maintain competitive performance on generation and downstream tasks even without RFT."
QUOTE-2 (abstract): "we show that SliceGPT can remove up to 25% of the model parameters (including embeddings) for LLAMA2-70B, OPT 66B and Phi-2 models while maintaining 99%, 99% and 90% zero-shot task performance of the dense model respectively."
NOTE: this is the one training-free *width* result found; it requires a calibration set and a per-model computational-invariance transform applied offline, and its 90% (Phi-2) figure is the weak end.

### A9.6 | CLOSED
QUESTION: Can inference bypass late/perturbed layers training-free, and does naive (static) early exit work?
WHO: Xuanming Zhang, Sining Zhoubian, Yuxuan Chen, Tianyi Tang, An Yang, Sean Du, Chujie Zheng, Fei Huang, Dayiheng Liu, Gao Huang, Jingren Zhou (Tsinghua / Alibaba Qwen)
ARTIFACT: arXiv:2606.21906 — "Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding" (title read back and confirmed)
URL: https://arxiv.org/abs/2606.21906 ; https://arxiv.org/html/2606.21906v1 ; https://github.com/vllm-project/vllm/issues/48080
STATUS: published (v1 2026)
EVIDENCE: READ BODY
QUOTE (static early exit, measured negative result): "As this execution probability p increases, the overall model accuracy undergoes a precipitous decline. This observation indicates that static truncation ignores the inherent variance in token complexity; a uniform exit policy prematurely interrupts the essential computation required for “hard” tokens, thereby destroying the model’s reasoning integrity."
QUOTE-2 (the paper's own limitations section): "While Confident Decoding offers a robust inference-time intervention, it is fundamentally constrained by the structural alignment of the unembedding matrix W_U with intermediate residual states. Although our theoretical framework bounds the projection noise, representations from shallow layers may still suffer from vocabulary mismatch. Furthermore, our approach mitigates the symptoms of the alignment tax during decoding rather than resolving its root cause during the training phase."
QUOTE-3 (a documented regression on a deeper model): "Qwen3.5-27B—a deeper model that gains strongly on reasoning benchmarks—shows a −3.0 drop on Air-Bench (67.0 → 64.0)."
NOTE: the method decodes from a near-final intermediate layer selected by entropy, needs the pretrained LM head applied to intermediate states, and reports "<2% latency increase" — i.e. it is an accuracy intervention, not a throughput win.

### A9.7 | CLOSED
QUESTION: Does vLLM or SGLang have a `--num-hidden-layers` / `num_hidden_layers` flag, and what is the supported way to override layer count?
WHO: vLLM maintainers (config surface); evidence from the rig's exact version
ARTIFACT: vLLM `ModelConfig.hf_overrides` / `--hf-overrides` (no dedicated `--num-hidden-layers` flag exists)
URL: https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/config/model.py ; https://github.com/vllm-project/vllm/issues/39428
STATUS: shipped (vLLM 0.29.0; field also present in the 0.28.0 checkout)
EVIDENCE: READ BODY
QUOTE: "hf_overrides: HfOverrides = field(default_factory=dict)  \"\"\"If a dictionary, contains arguments to be forwarded to the Hugging Face config. If a callable, it is called to update the HuggingFace config.\"\"\""
QUOTE-2 (proof the override is actually used to slice a model, from vLLM issue #39428): "run a 2-layer DeepSeek slice with --hf-overrides '{\"num_hidden_layers\": 2}', once with the fusion config under test and once with a plain baseline, feed the same prompt and compare logits."
NOTE: a dedicated `--num-hidden-layers` / `--num_hidden_layers` argument does **not** exist — grep of `/tmp/vllm-0.28.0` and of `/tmp/sglang-main/python/sglang/srt/server_args.py` returns no such CLI flag. The only path is the generic `--hf-overrides` dict.

### A9.8 | CLOSED
QUESTION: Has any layer-early-exit code actually been merged in the vllm-project org?
WHO: rootfs (contributor), merged by Xunzhuo (vLLM maintainer) in vllm-project/semantic-router
ARTIFACT: vllm-project/semantic-router PR #1210 — "feat: add 2D Matryoshka embedding with layer early exit"
URL: https://github.com/vllm-project/semantic-router/pull/1210
STATUS: merged
EVIDENCE: READ BODY
QUOTE: "This pull request adds support for the mmBERT-Embed-32K-2D-Matryoshka embedding model, a multilingual long-context embedding model with 2D Matryoshka capabilities (dimension reduction + layer early exit)."
NOTE: this is an *embedding* model whose checkpoint was trained for 2D Matryoshka; the merged code exposes the capability, it does not make an arbitrary pretrained checkpoint skippable. It is not the vLLM inference engine (semantic-router is a separate vllm-project repo). No equivalent merged layer-early-exit feature was found in vllm-project/vllm or sgl-project/sglang.

---

### B9.1 | OPEN
WHO IS STILL ASKING: xikronz (contributor) and ProExpertProg (vLLM maintainer) — vllm-project/vllm issue #39428, open, label `help wanted`
URL: https://github.com/vllm-project/vllm/issues/39428 ; https://github.com/vllm-project/vllm/pull/39480
ASKING QUOTE: "This would likely require some work to fix weight loading for models like DeepSeek when `--hf-overrides.num_hidden_layers` is overriden."
ASKING QUOTE 2 (the concrete breakage, comment by xikronz): "the current is_pp_missing_parameter only handles pipeline-parallel missing layers so without the guard those code paths hit a KeyError."
WHAT IS MISSING: `num_hidden_layers` can be overridden via `--hf-overrides`, but the weight loader is not guarded for layers that no longer exist, so a sliced checkpoint raises `KeyError` in the model's stacked-parameter paths. The DeepSeek fix (#39480) is still an unmerged proof of concept, and no supported/documented layer-slicing feature exists for a dense model like Qwen3-4B.
EVIDENCE: READ BODY
S3B: OCCUPIED (open PR #39480 — "[CI/Build][Model][DeepSeek] Enabling partial layer configs for deepseek models", state token `open`, verified by direct fetch)

### B9.2 | OPEN
WHO IS STILL ASKING: Xuanming Zhang / Sining Zhoubian et al. (Tsinghua + Alibaba Qwen) — vllm-project/vllm RFC issue #48080
URL: https://github.com/vllm-project/vllm/issues/48080
ASKING QUOTE: "1. Is `additional_config` the preferred configuration surface, or should this become a first-class `VllmConfig` field?"
ASKING QUOTE 2: "We propose adding **Confident Decoding** to vLLM: a training-free decoding strategy that dynamically selects logits from near-final intermediate layers instead of always using the final layer."
WHAT IS MISSING: no merged vLLM implementation. The RFC's own phase plan is unstarted ("| **PR 1** | `trough_utils.py`, v1 worker hooks, **Llama + Qwen3**, unit tests, `docs/features/confident_decoding.md` |"), the working prototype is described only as "Implementation prototype (vLLM 0.19.1 fork)", and the RFC states its own initial limitations — "Pipeline parallelism (`pp > 1`): disabled for correctness."
EVIDENCE: READ BODY
S3B: NO-ACTIVE-WORK-FOUND (queries: `grep -rni "entropy_selection|confident_decoding|trough" /tmp/vllm-0.28.0/vllm/` → no hits; `https://github.com/search?q=repo%3Avllm-project%2Fvllm+%22multi_layer_entropy_selection%22&type=issues` → did not complete, GitHub search throttled/timed out)

### B9.3 | OPEN
WHO IS STILL ASKING: smellslikeml — sgl-project/sglang issue #35987 "[Feature] Batch-wise Adaptive Pruning" (open, 22 days old at retrieval, no maintainer reply in the fetched body)
URL: https://github.com/sgl-project/sglang/issues/35987 ; https://arxiv.org/abs/2608.14003
ASKING QUOTE: "Gauging interest in a training-free, opt-in, default-off FFN-neuron-pruning knob in-tree."
ASKING QUOTE 2: "Questions: (1) lossy FFN-pruning in-tree or plugin? (2) integration shape acceptable (post_fill on replay stream + can_run_graph gating + prepare_bwap_batch hook)? (3) eligibility-broadening priorities?"
WHAT IS MISSING: a decision on whether training-free structured FFN-width pruning belongs in-tree, plus the CUDA-graph integration shape. The requester's own scope limits are stated: "Scope/limits: eligibility (unquantized float, bias-free, TP=1); ~2× pruned-FFN memory overhead (partly avoidable); AI-assisted."
EVIDENCE: READ BODY
S3B: NO-ACTIVE-WORK-FOUND (queries: `grep -rni "bwap|neuron_prun|teal|cats" /tmp/sglang-main/python/sglang/srt/` → no implementation (only unrelated substring matches); `grep -rn "pruning" /tmp/sglang-main/python/sglang/srt/server_args.py` → no hits; independent search `repo:sgl-project/sglang "neuron pruning"` type=issues → did not complete, GitHub search throttled)

---

### C9.1 | ABANDONED   (mid-network token pruning inside a running engine)
WHO: betacatZ (vLLM user), on Qwen2.5-7B
ARTIFACT: vllm-project/vllm issue #25756 — "[Usage]: how to do token prune"
URL: https://github.com/vllm-project/vllm/issues/25756
STATED REASON: "I am trying to perform token pruning on the Qwen2.5-7B model during the prefill stage in vLLM. Specifically, I prune 100 hidden_state tokens at layer 20. However, after pruning, the generated outputs are completely garbled, which should not happen."
STATED REASON 2 (the follow-on question that was never answered): "Do I need to manually modify the KV cache, slot_mapping, or other parameters to make token pruning work correctly during the prefill stage?"
EVIDENCE: READ BODY
S3B: NO-ACTIVE-WORK-FOUND (queries: `repo:vllm-project/vllm "layer pruning"` type=issues → 4 results, none on intra-layer token pruning; `repo:vllm-project/vllm "early exit" is:issue` → 34 results, none relevant; `grep -rni "layer_prun|layer drop|drop_layer|prune_layer" /tmp/vllm-0.28.0` → only an unrelated comment in `vllm/reasoning/cohere_command_reasoning_parser.py`)
CLOSURE CAVEAT (required disclosure): this issue carries `STATE_REASON: NOT_PLANNED` and was closed **only** by automation — "This issue has been automatically closed due to inactivity." There is **no human maintainer statement** and no human closure reason. The quote above is the reporter's own measured failure; it is not a maintainer decision, and this row must not be read as one.
REASON-MAY-HAVE-EXPIRED: the failure was measured against a source checkout of Sep 2025 (the reporter's environment clones vLLM and installs with `VLLM_USE_PRECOMPILED=1`), while the rig is vLLM 0.29.0; the coupling between hidden-state editing and KV/slot metadata may differ in 0.29.0.

### C9.2 | ABANDONED   (changing layer count / architecture without a process restart)
WHO: AlanFokCo (contributor) — sgl-project/sglang RFC issue #29363 (label `inactive`, closed by automation)
ARTIFACT: sglang issue #29363 — "[RFC] Cross-architecture model reload: `reload_model()` for architecture-changing model switches"
URL: https://github.com/sgl-project/sglang/issues/29363
STATED REASON (why the workaround was abandoned): "We have a working implementation of this as an external SGLang plugin (using the `HookRegistry` AROUND hook on `update_weights_from_disk`). However, it requires directly accessing ~20 private attributes across `ModelRunner`, `Scheduler`, and `TpWorker` (full list in the appendix). This is fragile — it broke once already when `init_memory_pool` moved into `ModelRunnerKVCacheMixin` — and not maintainable long-term."
STATED REASON 2 (the gap it was meant to close, including the layer-count case): "Currently, these scenarios require a full process restart (60-100s for a 32B model)." — preceded by: "several production scenarios require loading a model with a **different architecture** — different `num_hidden_layers`, `hidden_size`, attention type (GQA vs MLA), or model class (Dense vs MoE)"
EVIDENCE: READ BODY
S3B: NO-ACTIVE-WORK-FOUND (queries: `grep -rn "def reload_model\|reload_model" /tmp/sglang-main/python/sglang/srt/` → no definition, so the proposed API is not upstreamed; independent search `repo:sgl-project/sglang "reload_model"` type=issues → did not complete, GitHub search throttled)
CLOSURE CAVEAT (required disclosure): closed by automation only — "This issue has been automatically closed due to inactivity. Please feel free to reopen it if needed." No maintainer stated a reason; `STATE_REASON: COMPLETED` is a bot artifact, not a human judgement.
REASON-MAY-HAVE-EXPIRED: the stated fragility cites a specific internal refactor (`init_memory_pool` moving into `ModelRunnerKVCacheMixin`) in the SGLang revision the author used; in the local SGLang main checkout the memory-pool entry point is `Scheduler.init_memory_pools` and `ModelRunnerKVCacheMixin` survives only as a comment reference in an MLX stub, so the specific breakage cited may no longer apply at SGLang 0.5.19.

### C9.3 | ABANDONED   (threshold-based training-free adaptive FFN pruning under batching)
WHO: Yongmin Kim, Shota Takashiro, Yusuke Iwasawa, Takeshi Kojima, Yutaka Matsuo (University of Tokyo) — on the TEAL/CATS class of methods
ARTIFACT: arXiv:2608.14003 — "Batch-wise Adaptive Pruning: Periodic Neuron Activation-Aware Weight Pruning for Language Reasoning Model" (title read back and confirmed)
URL: https://arxiv.org/abs/2608.14003 ; https://github.com/sgl-project/sglang/issues/35987
STATED REASON: "the existing training-free adaptive pruning methods we evaluate severely degrade in this regime. Because a batch must share a single pruning mask, these methods aggregate activations across samples and then apply threshold-based selection; the threshold, calibrated offline on unaggregated activations, no longer matches the aggregated distribution, so the realized sparsity ratio drifts and accuracy on reasoning tasks collapses under batched inference."
STATED REASON 2 (in-engine measurement of the same failure, verbatim from the SGLang issue that cites this work): "Preliminary Results GSM8K n=50, ±6pp: 7B dense 92% → ρ=0.5 84% (−8pp) at 1.40× (ceiling); 1.5B dense 84% → ρ=0.25 78% (−6pp), ρ=0.5 catastrophic (34%). Accuracy = sparsity × model size."
EVIDENCE: READ BODY (abstract page for 2608.14003; the full text of that paper was not fetched — quote is from the retrieved arXiv abstract page)
S3B: NO-ACTIVE-WORK-FOUND (queries: `grep -rni "bwap|neuron_prun|teal|cats" /tmp/sglang-main/python/sglang/srt/` → no implementation; `grep -rn "pruning" /tmp/sglang-main/python/sglang/srt/server_args.py` → no hits)

### C9.4 | ABANDONED   (static / fixed-layer early exit)
WHO: Xuanming Zhang, Sining Zhoubian et al. (Tsinghua / Alibaba Qwen) — rejecting the naive baseline in their own study
ARTIFACT: arXiv:2606.21906 — "Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding" (§2.2 "Limitations of Static Early Exit")
URL: https://arxiv.org/abs/2606.21906 ; https://arxiv.org/html/2606.21906v1
STATED REASON: "As this execution probability p increases, the overall model accuracy undergoes a precipitous decline. This observation indicates that static truncation ignores the inherent variance in token complexity; a uniform exit policy prematurely interrupts the essential computation required for “hard” tokens, thereby destroying the model’s reasoning integrity."
EVIDENCE: READ BODY
S3B: NO-ACTIVE-WORK-FOUND (queries: `repo:vllm-project/vllm "early exit" is:issue` type=issues → 34 results, none proposing a shipped fixed-layer exit; `grep -rni "early.exit" /tmp/vllm-0.28.0` → only unrelated fast-path comments (KV offload, rejection sampler, spec decode); `grep -rni "early_exit" /tmp/sglang-main/python/sglang/srt/` → 2 hits, both unrelated scheduler/connector fast paths)

### C9.5 | ABANDONED   (recovery fine-tuning as the repair step for width pruning)
WHO: Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman (Microsoft / ETH Zurich)
ARTIFACT: arXiv:2401.15024 — "SliceGPT: Compress Large Language Models by Deleting Rows and Columns" (Figure 6 / recovery fine-tuning experiment)
URL: https://arxiv.org/abs/2401.15024 ; https://arxiv.org/html/2401.15024v2
STATED REASON: "Despite an extensive search, we were not able to find RFT parameters that enabled improved performance in the OPT models."
EVIDENCE: READ BODY
S3B: NO-ACTIVE-WORK-FOUND (queries: `repo:vllm-project/vllm "structured pruning"` type=issues → did not complete, GitHub search throttled; `grep -rni "layer_prun|prune_layer" /tmp/vllm-0.28.0` → no hits; `repo:vllm-project/vllm "layer pruning"` type=issues → 4 results, none on structured-width pruning)
REASON-MAY-HAVE-EXPIRED: the negative result is scoped to the OPT model family with WikiText-2 / Alpaca calibration; the rig's model is Qwen3-4B, so the OPT-specific finding does not transfer as a prediction for this rig.

---

### D9.1 | HARDWARE-RULED-OUT   (D.1 >1 GPU)
WHO: Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, Jingren Zhou (Alibaba)
ARTIFACT: arXiv:2312.04916 — "EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism" (title read back and confirmed)
URL: https://arxiv.org/abs/2312.04916 ; https://arxiv.org/html/2312.04916v2
HARDWARE QUOTE: "A server with 4 Nvidia A100-40GB GPUs is used for inference."
HARDWARE QUOTE 2 (the early-exit inference schedule the paper relies on): "Since the pipeline-based method uses 4 GPUs for a pipeline parallelism (PP) degree of 4, we allow KV recomputation to use a tensor parallelism (TP) degree of 1 or 4 for a fair comparison."
EVIDENCE: READ BODY
NOTE: EE-LLM's early-exit inference is built on Megatron-LM pipeline/tensor parallelism — its headline early-exit speedups are demonstrated on a 4-GPU server, which the single-GPU rig cannot reproduce.

### D9.2 | HARDWARE-RULED-OUT   (D.3 multi-node/NVLink/IB)
WHO: Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, Jingren Zhou (Alibaba)
ARTIFACT: arXiv:2312.04916 — "EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism"
URL: https://arxiv.org/abs/2312.04916 ; https://arxiv.org/html/2312.04916v2
HARDWARE QUOTE: "This scale is only limited by the hardware resources available to us, namely an 8-node cluster with 8 Nvidia A100-80GB GPUs in each node and hence 64 GPUs in total."
HARDWARE QUOTE 2 (the interconnect dependency the rig lacks): "for KV recomputation is only possible with high-end hardware like A100 GPUs connected by high-bandwidth communication."
EVIDENCE: READ BODY
NOTE: the second quote is the directly disqualifying one — the KV-recompute early-exit path's speedup depends on high-bandwidth GPU interconnect, and the rig is explicitly single-node with no NVLink/IB.

---

## SEARCH LOG (queries actually issued)
GitHub HTML search (returned results):
- https://github.com/search?q=repo%3Avllm-project%2Fvllm+%22layer+pruning%22&type=issues  (4 results)
- https://github.com/search?q=repo%3Avllm-project%2Fvllm+%22early+exit%22+is%3Aissue&type=issues  (34 results)
- https://github.com/search?q=repo%3Avllm-project%2Fvllm+%22num_hidden_layers%22&type=issues  (141 results)
- https://github.com/search?q=repo%3Asgl-project%2Fsglang+%22early+exit%22&type=issues  (18 results)
- https://github.com/search?q=repo%3Asgl-project%2Fsglang+%22layer+pruning%22&type=issues  (2 results)
- https://github.com/search?q=repo%3Avllm-project%2Fvllm+%22early+exit%22+is%3Apr&type=issues  (183 results; GitHub answered "The is:pr qualifier is not supported when searching issues")

GitHub HTML search attempted, did NOT complete (GitHub search hung / was throttled; no results obtained):
- repo:vllm-project/vllm "hf-overrides" "num_hidden_layers" (type=issues)
- repo:vllm-project/vllm "early exit" is:pr is:unmerged (type=issues)
- repo:vllm-project/vllm "skip layers" (type=issues)
- repo:vllm-project/vllm "head pruning" (type=issues)
- repo:vllm-project/vllm "structured pruning" (type=issues)
- repo:vllm-project/vllm "layer skip" (type=issues)
- repo:vllm-project/vllm "multi_layer_entropy_selection" (type=issues)
- repo:vllm-project/vllm "confident decoding" (type=issues)
- repo:vllm-project/vllm "early exit" is:unmerged (type=pullrequests)
- repo:vllm-project/vllm "num_hidden_layers" is:unmerged (type=pullrequests)
- repo:sgl-project/sglang "num_hidden_layers" (type=issues)
- repo:sgl-project/sglang "skip layers" (type=issues)
- repo:sgl-project/sglang "early exit" is:unmerged (type=pullrequests)
- repo:sgl-project/sglang "neuron pruning" (type=issues)
- repo:sgl-project/sglang "reload_model" (type=issues)
- repo:huggingface/transformers "num_hidden_layers" override (type=issues)
- https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+%22early+exit%22  (page returned "There was an error while loading. Please reload this page.")

Direct artifact fetches (`fetch.py gh` / `fetch.py get` / `fetch.py raw`):
- vllm-project/vllm issues: 39428, 25756, 48080 (bodies + comments read)
- vllm-project/vllm PR: 39480 (state token `open`), 19719 (state token `merged`, "[V1] Partial prefill skip for layers reusing shared KV cache" — not an early-exit feature)
- sgl-project/sglang issues: 29363, 35987
- vllm-project/semantic-router PR: 1210
- raw source: https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/config/model.py

arXiv read-backs (every ID confirmed by TITLE; `fetch.py arxiv <id>`):
- 2403.17887 "The Unreasonable Ineffectiveness of the Deeper Layers" — used (A9.1)
- 2403.03853 "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect" — used (A9.2)
- 2404.16710 "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding" — used (A9.3)
- 2207.07061 "Confident Adaptive Language Modeling" — used (note in A9.3)
- 2604.18396 "River-LLM: Large Language Model Seamless Exit Based on KV Share" — used (A9.4)
- 2401.15024 "SliceGPT: Compress Large Language Models by Deleting Rows and Columns" — used (A9.5, C9.5)
- 2606.21906 "Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding" — used (A9.6, C9.4)
- 2608.14003 "Batch-wise Adaptive Pruning: Periodic Neuron Activation-Aware Weight Pruning for Language Reasoning Model" — used (C9.3)
- 2312.04916 "EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism" — used (D9.1, D9.2)
- 2510.00546 "ThinkBrake: Efficient Reasoning via Log-Probability Margin Guided Decoding" — read back, then DROPPED (reasoning-stop, not layer exit; out of scope)
- 2211.15611 — read back returned "Special Cases of the Minimum Spanning Tree Problem under Explorable Edge and Vertex Uncertainty" (graph theory), NOT SkipDecode. DROPPED — no unverified ID is cited anywhere in this file.
- 2402.12345 — helper self-test only.

Local source greps:
- /tmp/vllm-0.28.0: `num_hidden_layers`, `early.exit`, `layer_prun|layer drop|drop_layer|prune_layer`, `hf_overrides`, `entropy_selection|confident_decoding|trough`
- /tmp/sglang-main: `num_hidden_layers` in `srt/server_args.py`, `early_exit|early exit`, `prune|skip_layer|layer_skip|drop_layer`, `bwap|neuron_prun|teal|cats`, `def reload_model|reload_model`, `ModelRunnerKVCacheMixin|def init_memory_pool`

## NEGATIVE FINDINGS
- **No `--num-hidden-layers` / `--num_hidden_layers` flag exists in vLLM or SGLang.** — queries tried: grep of `/tmp/vllm-0.28.0` (all `.py/.md`) for `num_hidden_layers` (only test/metrics uses, no CLI arg); grep of `/tmp/sglang-main/python/sglang/srt/server_args.py` for `num_hidden_layers` (zero hits); read of vLLM v0.29.0 `vllm/config/model.py` (only the generic `hf_overrides` dict).
- **No layer-pruning, head-pruning or layer-skipping inference feature in vLLM 0.29.0 / SGLang 0.5.19.** — queries tried: `repo:vllm-project/vllm "layer pruning"` (4 results, all unrelated: token prune, Linear.mask bug, Llama-Guard output pruning, KV-compression RFC); `repo:sgl-project/sglang "layer pruning"` (2 results: a batch-wise adaptive-pruning *feature request* and an architecture-reload RFC); grep of both checkouts for `layer_prun|prune_layer|drop_layer|skip_layer`.
- **Attention-head pruning without retraining: nothing found.** No merged or open engine work, and no citable paper row was established within budget. — queries tried: `repo:vllm-project/vllm "head pruning"` (type=issues) and `repo:vllm-project/vllm "structured pruning"` (type=issues) — both did NOT complete because GitHub search hung/was throttled; local grep of vLLM 0.28.0 for `prune_layer|layer_prun` returned nothing.
- **HF `transformers` `num_hidden_layers` override behaviour: not established, so no row is emitted.** — query tried: `repo:huggingface/transformers "num_hidden_layers" override` (type=issues) did not complete (throttled); web_search surfaced only unrelated config-error pages. Per the no-quote-no-row rule this is left as a gap rather than filled with a paraphrase.
- **No abandoned vLLM/SGLang PR specifically implementing layer dropping or early exit was confirmed.** — queries tried: `repo:vllm-project/vllm "early exit" is:pr is:unmerged` and the `type=pullrequests` variants for both repos, plus `https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+%22early+exit%22`; all failed to return results (throttle / "There was an error while loading"). The ABANDONED rows above are therefore issue-level and paper-level abandonments, not merged-PR post-mortems. **This is a real gap, not a claim that no such PR exists.**
- **No maintainer statement (vLLM or SGLang) closing layer/head/width pruning as out of scope** was found. The two closure-shaped artifacts found (#25756, #29363) were closed by automation with no human reason, and are disclosed as such.
