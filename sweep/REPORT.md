# Prior-art / gap-mapping sweep: OFFICIAL DOCS & MODEL CARDS for speculative-decoding drafting heads

Window prioritised: 2025-06 → 2026-09. Snapshot date 2026-09-13.
Repo snapshots read locally from `codeload.github.com` tarballs of `main` on 2026-09-13.

**Access notes (verified this session):**
- `curl` works: docs.vllm.ai, docs.sglang.io, github.com HTML, codeload.github.com tarballs, docs.nvidia.com.
- `curl` FAILS (HTTP 000): `huggingface.co` (all paths, incl. `/api/`), `raw.githubusercontent.com`.
- **`web_fetch` tool works** on huggingface.co, raw.githubusercontent.com, inco.ai.
- HF returns **401 "Invalid username or password"** for gated/nonexistent repos (e.g. `nvidia/Llama-3.1-70B-Instruct-Eagle3`, `lmsys/EAGLE3-LLaMA3.1-Instruct-8B`).

---

## CLOSED / SHIPPED

| # | Question it answers | Who closed it | Artifact (doc/flag/model card) | URL | Status | READ BODY or TITLE ONLY |
|---|---|---|---|---|---|---|
| 1 | What is vLLM's full speculative-method registry? | vLLM | `SpeculativeMethod` Literal: `ngram`, `medusa`, `mlp_speculator`, `draft_model`, `suffix`, `custom_class`, `eagle`, `eagle3`, `extract_hidden_states`, all `MTPModelTypes`, `dflash`, `dspark`, `ngram_gpu` | https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py | Shipped | READ BODY (local `vllm/config/speculative.py` L34–L79) |
| 2 | Is Medusa supported in vLLM? | vLLM | `"medusa"` in `SpeculativeMethod`; `MedusaModel: ("medusa","Medusa")` registry entry; `MedusaProposer`; auto-detect `hf_config.model_type == "medusa"` | https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py | Shipped, **no docs page** | READ BODY |
| 3 | vLLM `--speculative-config` JSON schema (all keys) | vLLM | `method`, `model`, `num_speculative_tokens`, `draft_tensor_parallel_size`, `max_model_len`, `parallel_drafting`, `rejection_sample_method`, `use_heterogeneous_vocab`, `prompt_lookup_min/max`, `suffix_decoding_*` | https://docs.vllm.ai/en/latest/features/speculative_decoding.html | Shipped | READ BODY |
| 4 | `method: "eagle"` exact config | vLLM | `{"model":"yuhuili/EAGLE-LLaMA3-Instruct-8B","draft_tensor_parallel_size":1,"num_speculative_tokens":2,"method":"eagle"}` | https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle.html | Shipped | READ BODY |
| 5 | `method: "eagle3"` exact config | vLLM | `{"model":"RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3","draft_tensor_parallel_size":2,"num_speculative_tokens":2,"method":"eagle3"}` | https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle.html | Shipped | READ BODY |
| 6 | `method: "mtp"` exact config | vLLM | `{"method":"mtp","num_speculative_tokens":1}`; Gemma 4: `{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}` | https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp.html | Shipped | READ BODY |
| 7 | `method: "draft_model"` exact config | vLLM | `{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5,"method":"draft_model"}` | https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model.html | Shipped | READ BODY |
| 8 | `method: "mlp_speculator"` exact config | vLLM | `{"model":"ibm-ai-platform/llama3-8b-accelerator","draft_tensor_parallel_size":1,"method":"mlp_speculator"}` | https://docs.vllm.ai/en/latest/features/speculative_decoding/mlp.html | Shipped | READ BODY |
| 9 | `method: "ngram"` exact config | vLLM | `{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}` | https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram.html | Shipped | READ BODY |
| 10 | `method: "suffix"` exact config | vLLM | `{"method":"suffix","num_speculative_tokens":8,"suffix_decoding_max_tree_depth":24,...}` | https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix.html | Shipped | READ BODY |
| 11 | `method: "dflash"` exact config | vLLM | `{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}` | https://huggingface.co/incoai/Qwen3.8-27B-DFlash2 | Shipped (vLLM main has `vllm/v1/worker/gpu/spec_decode/dflash2/`) | READ BODY |
| 12 | Parallel Draft Model (PARD) | vLLM | `parallel_drafting: true`; "Only compatible with EAGLE and draft-model methods." | https://docs.vllm.ai/en/latest/features/speculative_decoding/parallel_draft_model.html | Shipped | READ BODY |
| 13 | Custom proposer (`custom_class`) | vLLM | `speculative_config.method = "custom_class"`, `model = "your_module.YourCustomProposerClass"` | https://docs.vllm.ai/en/latest/features/speculative_decoding.html | Shipped, labelled **Experimental** | READ BODY |
| 14 | Dynamic speculative decoding | vLLM | doc page + limitation quote | https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding.html | Shipped | READ BODY |
| 15 | Adaptive verification | vLLM | "currently DSpark only" | https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification.html | Shipped | READ BODY |
| 16 | Hidden-state extraction for head training | vLLM | `method: "extract_hidden_states"` | https://docs.vllm.ai/en/latest/features/speculative_decoding/extract_hidden_states.html | Shipped | READ BODY |
| 17 | vLLM ↔ Speculators integration path | vLLM | speculators doc page | https://docs.vllm.ai/en/latest/features/speculative_decoding/speculators.html | Shipped | READ BODY |
| 18 | SGLang algorithm flag values | SGLang | `--speculative-algorithm`: `UNO`, `DFLASH`, `EAGLE`, `EAGLE3`, `STANDALONE`, `NGRAM`, `NEXTN` (alias of `EAGLE`) | https://docs.sglang.io/advanced_features/speculative_decoding.html | Shipped | READ BODY |
| 19 | SGLang EAGLE-3 exact flags | SGLang | `--speculative-algorithm EAGLE3` + `--speculative-draft-model-path ...` + `--speculative-num-steps/--speculative-eagle-topk/--speculative-num-draft-tokens` | https://docs.sglang.io/advanced_features/speculative_decoding.html | Shipped | READ BODY |
| 20 | SGLang MTP usage | SGLang | MTP runs through the EAGLE speculative workflow; example `--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2` | https://docs.sglang.io/advanced_features/speculative_decoding.html | Shipped | READ BODY |
| 21 | SGLang NEXTN flag | SGLang | `NEXTN` = alias of `EAGLE` | https://docs.sglang.io/advanced_features/speculative_decoding.html | Shipped | READ BODY |
| 22 | SGLang UNO | SGLang | `--speculative-algorithm UNO --uno-lora-path ...` | https://docs.sglang.io/advanced_features/speculative_decoding.html | Shipped | READ BODY |
| 23 | SGLang draft-model quantization flag | SGLang | `--speculative-draft-model-quantization`; `"unquant"` disables quant on draft | https://docs.sglang.io/advanced_features/speculative_decoding.html | Shipped | READ BODY |
| 24 | **Qwen3-Next MTP in vLLM** | Qwen + vLLM | `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct | Shipped | READ BODY |
| 25 | **Qwen3-Next MTP in SGLang** | Qwen + SGLang | `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct | Shipped | READ BODY |
| 26 | Qwen3-Next MTP source support | vLLM + SGLang | vLLM `Qwen3NextMTP` (`qwen3_next_mtp.py`, `model_type: qwen3_next_mtp`); SGLang `Qwen3NextForCausalLMMTP` | https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py | Shipped | READ BODY |
| 27 | vLLM DFlash/DFlash2 implementation | vLLM | `vllm/v1/worker/gpu/spec_decode/dflash/`, `dflash2/speculator.py`, `vllm/v1/spec_decode/dflash.py` | https://github.com/vllm-project/vllm/tree/main/vllm/v1/worker/gpu/spec_decode | Shipped on main | READ BODY |
| 28 | DFlash2 model card + engines | Inco AI / z-lab | SGLang `--speculative-algorithm DFLASH --speculative-num-draft-tokens 8`; vLLM `method:"dflash", num_speculative_tokens:7` | https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2 | Shipped | READ BODY |
| 29 | DFlash2 blog / design | Inco AI | "over 20% more output from every verification pass, for around 1% added cycle latency" | https://inco.ai/blog/dflash2/ | Shipped 2026-08-18 | READ BODY |
| 30 | EAGLE-3 canonical author model card | SafeAILab / Yuhui Li | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` | https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B | Shipped | READ BODY |
| 31 | Red Hat EAGLE-3 speculator (70B) | Red Hat AI | trained **with the `speculators` library**; vLLM eagle3 recipe; 4×A100 benchmark config | https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3 | Shipped 09/15/2025 | READ BODY |
| 32 | Medusa standalone inference scope | FasterDecoding | "single GPU and batch size 1 setting" | https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3 | Shipped (legacy) | READ BODY |
| 33 | DeepSeek-V3 MTP module size | DeepSeek-AI | "671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights" | https://huggingface.co/deepseek-ai/DeepSeek-V3 | Shipped | READ BODY |
| 34 | Medusa head-training weights | FasterDecoding | `medusa-vicuna-{7,13,33}b-v1.3` | https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3 | Shipped | READ BODY |
| 35 | SpecForge trainable head families | SGLang team | EAGLE3, P-EAGLE, EAGLE3.1, DFlash, **DFlash2**, Domino, DSpark | https://github.com/sgl-project/SpecForge | v0.3.0 (2026-08) | READ BODY |
| 36 | Speculators trainable head families | vLLM project | EAGLE-3, P-EAGLE, DFlash, DFlash2, DSpark, **MTP finetuning** (Qwen3-Next, Qwen3.5) | https://github.com/vllm-project/speculators | Shipped | READ BODY |
| 37 | Speculators MTP finetuning method | vLLM project / FastMTP | `MTPConverter` → finetune → stitch back; only MTP layers trainable | https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp.html | Shipped | READ BODY |
| 38 | NeMo AutoModel EAGLE recipe (EAGLE-1/2/3 + 3.1) | NVIDIA | `train_eagle1/2/3` recipes; hardware table | https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/eagle-speculative-decoding | Shipped | READ BODY (via https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/eagle.mdx) |
| 39 | NeMo AutoModel DFlash recipe (DFlash/DFlash2/Domino/JetSpec) | NVIDIA | `TrainDFlashRecipe`, `TrainDFlash2Recipe`, `TrainDominoRecipe`, `TrainJetSpecRecipe` | https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding | Shipped | READ BODY (via https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/dflash.mdx) |
| 40 | EAGLE-3 drafter for Qwen3.6-27B + engines | Ex0bit | SGLang EAGLE3 vs vLLM; full/compressed variants | https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3 | Shipped | READ BODY |
| 41 | MTP vs EAGLE-3 TPS for same base | Ex0bit | native MTP 121 tok/s (1.51×) vs EAGLE-3 chain 111 tok/s (1.39×) | https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ | Shipped | READ BODY |
| 42 | DFlash2 drafter → W4A16 quantization recipe | syvai | 1.92B / 3.85 GB bf16 → 1.19 GB W4A16 | https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16 | Shipped | READ BODY |
| 43 | SGLang SpecForge docs hub | SGLang team | — | https://docs.sglang.io/SpecForge/ | Shipped | TITLE ONLY (hub landed 200; content read from repo) |

---

## Documented limitations / unsupported claims (VERBATIM quotes)

### vLLM

1. **Pipeline parallelism × spec decode** — https://docs.vllm.ai/en/latest/features/speculative_decoding.html
   > "1. Pipeline parallelism is not composable with speculative decoding as of `vllm<=0.15.0`
   > 2. Speculative decoding with draft models is not supported in `vllm<=0.10.0`"
   (`docs/features/speculative_decoding/README.md`, "Known Feature Incompatibility")

2. **Heterogeneous vocab / probabilistic draft sampling** — same URL
   > "`use_heterogeneous_vocab` currently supports greedy draft sampling only. Probabilistic acceptance (temperature > 0 draft sampling) is not yet supported and will be added in a future release."

3. **`use_heterogeneous_vocab` key restriction** — same URL
   > "Probabilistic draft sampling (`draft_sample_method='probabilistic'`) is not yet supported when this option is enabled."

4. **`parallel_drafting` scope** — same URL
   > "Enable parallel draft token generation. Only compatible with EAGLE and draft-model methods."

5. **Adaptive verification scope** — https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification.html
   > "Adaptive verification needs per-position acceptance estimates, so today it is only supported for DSpark with a **confidence head**."

6. **Dynamic spec decode tested scope** — https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding.html
   > "Tested with Eagle, Eagle-3, and DFlash. Other SD methods may or may not work out of the box"

7. **MTP is family-gated** — https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp.html
   > "MTP only works for model families that support MTP in vLLM."
   > "If your model does not support MTP, use another method such as EAGLE or draft model speculation."

8. **MTP depth warning (maintainer code comment / runtime warning)** — https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py
   > "Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer ,which may result in lower acceptance rate"

9. **Gemma 4 assistant mis-dispatch** — https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp.html
   > "If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)` for a Gemma 4 assistant checkpoint, that release is treating the assistant as a generic draft model and may fail during initialization for multimodal Gemma 4 targets. Upgrade to a version with Gemma 4 MTP support instead."

10. **DeepSeek V4.1 has no classic MTP** — https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py
    > "DeepSeek V4.1 has no classic-MTP draft: its checkpoints ship DSpark stages under mtp.* (main_proj/markov_head/confidence_head) and carry no e_proj/h_proj/enorm/hnorm/hc_head weights. Use speculative method 'dspark' instead of 'mtp'."

11. **Unsupported method → hard error** — same file
    > `raise NotImplementedError(f"Unsupported speculative method: '{self.method}'")`

12. **MLP speculator known issue** — https://docs.vllm.ai/en/latest/features/speculative_decoding/mlp.html
    > "!!! warning "Known issue"
    >     `ibm-ai-platform/llama3-70b-accelerator` can fail with:
    >     `AttributeError: 'MLPSpeculatorConfig' object has no attribute 'num_attention_heads'`.
    >     Track status in [#34106] and [#34163]."

13. **Custom proposer is unstable API** — https://docs.vllm.ai/en/latest/features/speculative_decoding.html
    > "Using a custom class-based proposer backend. This is an experimental feature and the proposer interface is subject to breaking changes in future vLLM releases."

14. **DFlash: anchor sampling not supported** — https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
    > "sample_from_anchor=True is not supported for DFlash. DFlash uses a fixed 1+N query layout where the anchor is the bonus token."

15. **DFlash: CUDA-graph support** — same file
    > "%s draft attention (%s) does not support full CUDA graphs; running the draft eagerly."
    > "# PIECEWISE cudagraphs are not supported for dflash."

16. **Qwen3NextMTP prefix caching** — https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py
    > "Qwen3NextMTP currently does not support 'all' prefix caching, "

17. **Deprecated flag style** — https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model.html
    > "Note: Please use `--speculative-config` to set all configurations related to speculative decoding. The previous method of specifying the model through `--speculative-model` and adding related parameters such as `--num-speculative-tokens` separately has been deprecated."

18. **Old-format EAGLE checkpoint conversion** — https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle.html
    > "If you are using `vllm<0.7.0`, please use this script to convert the speculative model and specify `"model": "path/to/modified/eagle/model"` in `speculative_config`."

19. **Medusa gets no dedicated docs page.** `medusa` is a valid `method` value in `vllm/config/speculative.py` and has a model class (`vllm/model_executor/models/medusa.py`), but `docs/features/speculative_decoding/` contains no `medusa.md`. (Search of `docs/` for "medusa" returns zero hits.) URL: https://github.com/vllm-project/vllm/tree/main/docs/features/speculative_decoding — READ BODY.

### SGLang

20. **DFLASH constraints** — https://docs.sglang.io/advanced_features/speculative_decoding.html
    > "No `--enable-dp-attention`; `pp_size == 1`; disables overlap scheduler & mixed chunked prefill"

21. **NGRAM constraints** — same URL
    > "CUDA-only; no `--enable-dp-attention`; disables overlap scheduler & mixed chunked prefill"
    > "Ngram speculative decoding **only supports CUDA**."
    > "It currently **does not support** `--enable-dp-attention`."
    > "It disables the overlap scheduler and mixed chunked prefill."

22. **STANDALONE constraint** — same URL
    > "**Note:** Standalone speculative decoding currently **does not support** `--enable-dp-attention`."

23. **UNO constraints** — same URL
    > "UNO requires CUDA with FA3 for prefill and decode, tensor and pipeline parallel sizes of 1, and no DP attention or context parallelism."
    > "Grammar decoding, returned logprobs or hidden states, sampling penalties, `min_p`, logit bias, custom logit processors, strict thinking, and deterministic inference are not yet supported."
    > "Mixed chunked prefill is disabled for UNO."

24. **Overlap scheduler / V2 topk restriction** — same URL
    > "The overlap scheduler currently only supports `--speculative-eagle-topk 1`; **set `--speculative-eagle-topk 1` explicitly**."
    > "If you explicitly set `--speculative-eagle-topk > 1`, the server will error."
    > "If you omit `--speculative-eagle-topk`, auto-tuning may pick `topk > 1` for some models (e.g. Llama). This is incompatible with the overlap scheduler and may not always trigger an immediate config error, so set `--speculative-eagle-topk 1` explicitly."

25. **Draft model path is typically required** — same URL
    > "Draft model path/weights. **Typically required** for EAGLE/EAGLE3 and STANDALONE. For some MTP-enabled models, this can be omitted."

26. **NGRAM + page_size interaction** — same URL
    > "If `--speculative-ngram-max-bfs-breadth > 1` (thus `speculative_eagle_topk > 1`) and `page_size > 1`, use `--attention-backend flashinfer`; otherwise the server will error."

27. **UNO LoRA cannot combine with Multi-LoRA** — same URL
    > "`--uno-lora-path` loads UNO's fixed internal adapter and cannot be combined with request-selectable Multi-LoRA serving."

### Head × quantization interactions (the highest-value findings)

28. **vLLM silently ignores a quantized MTP head → 0% acceptance, no error** — https://raw.githubusercontent.com/devnen/qwen3.6-windows-server/v1.3.3/docs/MTP_HEAD.md
    > "Multi-token prediction (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`) only works if the model weights ship an **MTP head in BF16**. The vLLM `Qwen3_5MTP` loader looks for tensors named `mtp.fc.*` and refuses (silently) to use them if they're quantised."
    > "Other Qwen3.6-27B quants, `cyankiwi`, `groxaxo/Qwen3.6-GPTQ-Pro-4bit`, etc. , either OOM trying to allocate a fresh BF16 head on a 24 GB card, or quantise the head to INT4 along with the body. The loader silently skips the quantised head, MTP runs, and you get **0 % draft acceptance**, no speedup, no error message."
    > "**If it's near 0.0, your quant's MTP head got silently skipped.**"

29. **MTP + pipeline parallelism mutually exclusive on that build** — same URL
    > "`Qwen3_5MTP` worker init refuses pipeline parallelism on Qwen3-Next:
    > `NotImplementedError: Pipeline parallelism is not supported for this model`
    > This is a vLLM 0.19.0 limitation, not a hardware one. On that zip you pick MTP *or* PP (for big context), not both."

30. **BF16 drafter + quantized target needs explicit opt-out** — https://raw.githubusercontent.com/noonghunna/club-3090/refs/tags/v0.10.2/docs/engines/SGLANG.md
    > "Without this, the BF16 EAGLE-3 drafter silently inherits the target's `--quantization auto-round` flag and fails to load (it's a BF16 model, no AutoRound config). The fix is to explicitly opt the drafter out via `unquant`. Mandatory for any external-BF16-drafter + quantized-target combo."

31. **Quantized target `lm_head` breaks the DFlash2 selector upstream** — https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16
    > "The drafter shares the target's embeddings and lm_head (it ships neither); with a quantized target lm_head the repo's patch is needed (upstream refuses a non-bf16 lm_head for the candidate top-k)."

32. **DFlash2 drafter quantization is measurable but small** — same URL
    > "same acceptance as the bf16 drafter at greedy (3.34-3.65 vs 3.54 tokens per step), about 5% lower at the model's default sampling"
    > "A variant whose k/v Hessians also blended the context-KV precompute's input distribution — which is the theoretically tidier calibration — measured 7% *worse* greedy acceptance (3.12 vs 3.34 tokens per step, 118 vs 126 tok/s end to end) and is not what ships here."

33. **SGLang draft-quant default inherits target quant** — https://docs.sglang.io/advanced_features/speculative_decoding.html
    > "Quantization for the draft model. Use `"unquant"` to disable quantization on the draft even when the target is quantized." (default: "Same as target")

### Portability / engine-support gaps

34. **Qwen3-Next EAGLE-3 external drafter "PARKED"** — https://raw.githubusercontent.com/noonghunna/club-3090/refs/tags/v0.10.2/docs/engines/SGLANG.md
    > "**Status (2026-05-21): PARKED.** Not currently a shipped variant on this stack for Qwen3-Next."
    > "**EAGLE-3 is sub-MTP for Qwen3-Next, even on Blackwell where it works.** Ex0bit's own published numbers on the PRISM-PRO-DQ model card report native MTP = **121 TPS (1.51×)** vs EAGLE-3 chain = **111 TPS (1.39×)**. The model family has a strong built-in MTP head; routing through an external drafter is structurally slower."
    > "**cuda-graph capture on Ampere** | ❌ Hangs (CUTLASS CUTE Hopper-oriented) — must use `--disable-cuda-graph`"
    > "At 24 GB the target (~17 GB) + EAGLE-3 drafter (~3 GB) + Mamba state + KV cache leaves ~0-2 GB headroom."

35. **EAGLE-3 needs a target hook that SGLang does not ship for Qwen3.6** — https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3
    > "SGLang's dense Qwen3.6 model class (`Qwen3_5ForConditionalGeneration`) ships DFlash/MTP aux-hidden capture but not the EAGLE-3 target hook."
    > "Without it, the inherited Qwen3-VL hook sets a `layers_to_capture` list the Qwen3.6 decoder never reads — capture silently no-ops and the forward pass crashes."

36. **EAGLE-3 chain recommended over tree on hybrid GatedDeltaNet** — same URL
    > "Tree drafting (`--speculative-eagle-topk 4`) raises accept length to ~3.35 but is throughput-neutral on this hybrid GatedDeltaNet target (the tree-build + recurrent-verify cost cancels the acceptance gain) — **chain is recommended**."

37. **MTP not available in HF Transformers** — https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
    > "Multi-Token Prediction (MTP) is not generally available in Hugging Face Transformers."
    > "The efficiency or throughput improvement depends highly on the implementation. It is recommended to adopt a dedicated inference framework, e.g., SGLang and vLLM, for inference tasks."

38. **Medusa inference limited to single GPU / BS=1** — https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3
    > "We currently support inference in the single GPU and batch size 1 setting, which is the most common setup for local model hosting. We are actively working to extend Medusa's capabilities by integrating it into other inference frameworks, please don't hesitate to reach out if you are interested in contributing to this effort."

39. **DeepSeek MTP support is community-WIP** — https://huggingface.co/deepseek-ai/DeepSeek-V3
    > "Please note that MTP support is currently under active development within the community, and we welcome your contributions and feedback."

40. **EAGLE-3 on-device (CoreML/ANE): slower than baseline** — https://raw.githubusercontent.com/john-rocky/CoreML-LLM/refs/heads/main/docs/EAGLE3_INTEGRATION_STATE.md
    > "Status: **Phase 2A + 2B done, Phase 3 benched, speculative currently slower than baseline.**"
    > "Phase 3 — iPhone 17 Pro bench | ⚠️ ran, **not faster than baseline 28.6 tok/s** (11–17 tok/s with fallback to T=1)"
    > "**Blocker 3 (11c): verify-vs-decode fp16 drift** — sets iPhone acceptance break-even at ~77%. Even a successful retrain landing at 50-60% would not produce a net speedup on device until 11c closes."

41. **SpecForge: EAGLE-3/DFlash2 serving needs a matching SGLang version** — https://github.com/sgl-project/SpecForge
    > "Serving requires an SGLang version that includes DFlash2 support (SGLang PR #35371); the serving algorithm name remains `DFLASH`, and the exported `DFlash2DraftModel` config enables the new path automatically."

42. **SpecForge: unsupported combinations are hard-rejected** — https://github.com/sgl-project/SpecForge
    > "Unsupported combinations are rejected during config validation or run assembly instead of falling back to an older trainer."

43. **SpecForge: unvalidated acceptance on real server** — https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md
    > "loading the result in a real speculative-decoding server and measuring acceptance remains a GPU-serving validation step."

---

## Hardware and cost requirements for head TRAINING (verbatim quotes)

1. **NeMo AutoModel EAGLE recipe hardware table** — https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/eagle-speculative-decoding (source: https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/eagle.mdx)
   > "| Setup | Target Model | GPUs | Training Time |
   > |-------|-------------|------|---------------|
   > | MVP (quick test) | Llama 3.2 1B | 1x A100 80 GB | ~10 min (1 epoch, 1k samples) |
   > | Production | Llama 3.1 8B Instruct | 8x A100 80 GB | ~2 h (1 epoch, 200k samples) |"
   > "On CUDA, the target model is loaded in BF16 and frozen during training (the CPU fallback uses FP32). Only the small drafter is trained; standard EAGLE-3 uses one fused decoder layer plus an auxiliary projection. **GPU memory is therefore dominated by the target model size.**"
   > "`ttt_steps` | Required TTT unroll depth; maintained Llama examples use 4, and cost is linear per step"

2. **EAGLE (SafeAILab) canonical cost claim** — https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B (also https://github.com/SafeAILab/EAGLE)
   > "trainable (within 1-2 days) and testable on 8x RTX 3090 GPUs. So even the GPU poor can afford it."

3. **Speculators: MTP head training does NOT need the full verifier** — https://github.com/vllm-project/speculators
   > "**MTP Finetuning Support**: Added support for finetuning the native Multi-Token Prediction (MTP) heads of models like Qwen3-Next on domain-specific data, following the [FastMTP](https://arxiv.org/abs/2509.18362) approach. Because the MTP head is small (~100M–400M params), it can be trained on pre-extracted hidden states without loading the full verifier"

4. **Speculators: single-GPU training is a supported entry point** — https://docs.vllm.ai/projects/speculators/en/latest/cli/train.html
   > "Trains speculator models using either online or offline hidden states. Supports single-GPU and multi-GPU distributed training."
   > "**Single GPU:** drop the `torchrun` wrapper and call `speculators train` directly with the same arguments."

5. **Speculators: FSDP when the model does not fit** — https://docs.vllm.ai/projects/speculators/en/latest/cli/train.html
   > "**`--fsdp-shard`** (flag) Shard model parameters across GPUs with FSDP. By default, parameters are fully replicated (DDP-like). Enable this when the model does not fit in a single GPU's memory."

6. **Speculators: MTP finetune recipe detail** — https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp.html
   > "Only the MTP layers are trainable -- `embed_tokens` and `lm_head` are frozen and shared with the verifier."
   > "This approach is available for models that ship with native MTP support, such as Qwen3-Next and Qwen3.5."

7. **SpecForge: two-GPU local stack for a 27B DFlash2 head** — https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md
   > "The checked-in Qwen3.6-27B recipe owns a two-GPU local stack: one configured GPU runs the target capture server and another runs the trainer. Update its model/data paths and `cuda_visible_devices` lists for the local host."

8. **SpecForge: offline mode only needs the draft to fit** — same URL
   > "Online training captures target features while the run is active. It uses little disk space but keeps target inference available during training. Offline training reads feature checkpoints generated ahead of time, so only the draft model must fit on the training GPUs at the cost of substantially more storage."

9. **NeMo DFlash recipe launch (2 GPUs)** — https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/dflash.mdx
   > "torchrun --standalone --nproc_per_node=2 -m nemo_automodel.recipes.llm.train_dflash -c examples/speculative/dflash/qwen3_dflash.yaml"
   > "Multi-GPU wraps the draft in DDP and replicates the frozen target. Set `distributed.tp_size` to shard a large target across ranks (`qwen3_dflash_tp.yaml`); `distributed.cp_size` enables context parallelism, which cannot be combined with sequence packing or with `tp_size > 1`."

10. **Speculators: dataset regeneration cost** — https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/response_regeneration.html
    > "**Time required:** ~10 mins on 2x H100 GPUs (for 1K samples)"

11. **NeMo DFlash recipe: supported targets** — https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/dflash.mdx
    > "Supported targets: Qwen3 and Qwen3.5 (dense and MoE), including the multimodal `*ForConditionalGeneration` variants such as `Qwen/Qwen3.8-27B`. DFlash and JetSpec drafts serve on vLLM through its `dflash` method."

12. **NeMo EAGLE recipe: supported target architectures** — https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/eagle.mdx
    > "| Llama, Phi-3, Qwen3 dense, Qwen3-MoE | shared Llama-style dense draft |
    > | gpt-oss (`GptOssForCausalLM`) | dedicated draft reproducing gpt-oss YaRN RoPE |
    > | DeepSeek-V3 (`DeepseekV3ForCausalLM`) | dedicated MLA draft |
    > | Gemma4 (`Gemma4ForConditionalGeneration`) | thin dedicated draft (see below) |"

13. **Community EAGLE-3 retrain cost datapoint** — https://raw.githubusercontent.com/john-rocky/CoreML-LLM/refs/heads/main/docs/EAGLE3_INTEGRATION_STATE.md
    > "Revival cost: ~$0 Colab A100 ~25 min to run the already-written training script on `training_data_custom.pt`."
    > "Retrain for 2 epochs × ~30k samples. Target: acc[0] ≥ 0.5 against custom target"
    > "`eagle3_draft_best.pt` | 188 MB, 47.2 M params."
    > "Phase 1 — draft + fusion training | ✅ done on Colab. acc[0]=74.94%, acc[1]=40.6%, acc[2]=23.9%"

14. **Red Hat 70B EAGLE-3 speculator: training library + eval hardware** — https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3
    > "It was trained using the [speculators](https://github.com/vllm-project/speculators) library on a combination of the Aeala/ShareGPT_Vicuna_unfiltered and the `train_sft` split of HuggingFaceH4/ultrachat_200k datasets."
    > "Details **Configuration** - temperature: 0 - repetitions: 5 - time per experiment: 4min - hardware: 4xA100 - vLLM version: 0.11.0"

15. **Ex0bit EAGLE-3 drafter for Qwen3.6-27B: size + trainer** — https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3
    > "a small (~0.6 B trainable) draft head"
    > "| [`full/`](full) | 248 320 (full) | 3.1 GB | resume-training base; widest compatibility |
    > | [`compressed/`](compressed) | 32 000 (+ `d2t` map) | 1.1 GB | **recommended for serving — fastest** |"
    > "It was self-distilled on `Qwen3.6-27B-PRISM-PRO` completions (REAP + UltraChat + tulu-3 corpus). The current drafter is chain-trained (`parallel_draft_step: 1`); tree-aware retraining is the main lever for further gains."
    > "loadable by [SpecForge](https://github.com/sgl-project/SpecForge) or [NVIDIA TensorRT Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (which trained this drafter)"

16. **DFlash2 drafter size** — https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16
    > "the DFlash2 block drafter for Qwen3.8-27B (5 Qwen3-style layers, 1.92B parameters, 3.85 GB in bf16)"
    > "Hessians from the drafter's own inputs on 400 real prompts (~290k rows per layer)"

17. **EAGLE-3 fusion-layer selection is a training-side hyperparameter** — https://raw.githubusercontent.com/john-rocky/CoreML-LLM/refs/heads/main/docs/EAGLE3_INTEGRATION_STATE.md
    > "`eagle3_config.json` | `fusion_layers=[8,17,34]`, `hidden=1536`, `num_heads=8`, `num_kv_heads=1`, `head_dim=256`, `ffn=6144`, `embed_scale=39.1918...`, `ttt_k=3`, `model_id=google/gemma-4-E2B-it`"

---

## OPEN (explicitly still missing / asked for, with verbatim quote)

1. **No official docs page for Medusa anywhere in vLLM.** `medusa` is a first-class `method` value and has a model implementation, but `docs/features/speculative_decoding/` has no Medusa page, and SGLang's spec-decoding page never mentions Medusa. Gap: no official flag schema, no supported-model list, no limitation statement. Evidence: zero "medusa" hits under `vllm/docs/`. URL: https://docs.vllm.ai/en/latest/features/speculative_decoding.html

2. **`nvidia/Llama-3.1-70B-Instruct-Eagle3` does not exist** (HTTP 404 on the model page; HTTP 401 on `/raw/`). The URL requested for this sweep is dead. Working NVIDIA-adjacent EAGLE-3 cards found instead: https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3 and https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

3. **`lmsys/EAGLE3-*` is auth-walled** (HTTP 401 on `/raw/main/README.md`). Cannot confirm whether it is gated or renamed. URL: https://huggingface.co/lmsys/EAGLE3-LLaMA3.1-Instruct-8B

4. **No official EAGLE-3 × Qwen3-Next support in either engine; only third-party patched paths.** https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3
   > "SGLang's dense Qwen3.6 model class (`Qwen3_5ForConditionalGeneration`) ships DFlash/MTP aux-hidden capture but not the EAGLE-3 target hook."

5. **Upstream EAGLE-3 capture hook for `Qwen3_5ForConditionalGeneration` still unmerged** — https://raw.githubusercontent.com/noonghunna/club-3090/refs/tags/v0.10.2/docs/engines/SGLANG.md
   > "SGLang upstream merges the EAGLE-3 capture hook for `Qwen3_5ForConditionalGeneration` | We can drop the Ex0bit `patch_sglang_eagle3.py` vendor (or it ships baked into the drafter)."

6. **Single-3090 (24 GB class) EAGLE-3 blocked by an engine bug** — same URL
   > "Single 3090 + AutoRound INT4 + EAGLE-3 | ❌ Hits SGLang OffloaderV1 tied-weights bug on Qwen3-Next"
   > "`ValueError: functional_call got multiple values for keys ['linear_attn.attn.dt_bias', 'linear_attn.dt_bias']`"

7. **DFlash2 in vLLM required a PR at publication time** — https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
   > "pip install -U "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/52816/head""
   (Note: vLLM `main` as of 2026-09-13 does contain `vllm/v1/worker/gpu/spec_decode/dflash2/`, so it appears merged since.)

8. **DFlash2 for llama.cpp / Ollama still on unmerged PR branches** — https://inco.ai/blog/dflash2/
   > "git fetch origin pull/27342/head:pr-27342" (llama.cpp)
   > "git fetch origin pull/17865/head:dflash2" (Ollama)

9. **EAGLE-3 on llama.cpp is WIP and needs unmerged patches** — https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ
   > "EAGLE-3 chain (needs the WIP PR #18039 patches + the RS-rollback fix -- a one-shot llama.cpp patch script is documented alongside the drafter)"

10. **SpecForge Domino/DSpark cannot run without an explicit draft config** — https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md
    > "Domino and DSpark need their projector/head metadata, so they require an explicit draft config (or a pretrained warm-start source that contains `config.json`). The old Domino parser exposed an optional config flag, but its no-config branch immediately failed because those required projector fields had no defaults; the unified schema rejects that unusable combination early."

11. **Head-training cost for Medusa is not documented anywhere official.** The Medusa card gives inference scope only; no GPU-count or wall-clock training figure appears in the card or the FasterDecoding/Medusa README. URL: https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3

12. **No official engine documents head-training cost for DFlash2 or DSpark.** SpecForge lists them as trainable methods with example configs but states no GPU count or time; NeMo's DFlash guide gives only `--nproc_per_node=2`. URLs: https://github.com/sgl-project/SpecForge , https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/dflash.mdx

13. **Quantized-draft interactions are documented only in community sources, not in engine docs.** vLLM's spec-decoding docs say nothing about quantized MTP heads; the "silently skipped → 0% acceptance" behaviour is documented only at https://raw.githubusercontent.com/devnen/qwen3.6-windows-server/v1.3.3/docs/MTP_HEAD.md. SGLang documents only the `--speculative-draft-model-quantization unquant` escape hatch.

14. **Adaptive verification limited to DSpark** — https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification.html
    > "Adaptive verification needs per-position acceptance estimates, so today it is only supported for DSpark with a **confidence head**."

15. **SpecForge/NeMo surface methods with no published acceptance validation** — https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md
    > "loading the result in a real speculative-decoding server and measuring acceptance remains a GPU-serving validation step."

16. **Unverified in this sweep (flagged, not guessed):** model cards for `YuhuaiZhang/Medusa`, `meta-llama` MTP models, DeepSeek-R1 MTP section, `Qwen3` (non-Next) MTP, and SpecBundle phase-2 checkpoints. No URL was read for these, so no rows are claimed.

---

## Search log — every query and every URL fetched

### `web_search` queries (verbatim, in order)
1. `NeMo AutoModel "Train a DFlash Drafter" recipe GPU requirements` + `NeMo AutoModel EAGLE drafter training recipe hardware 8xH100`
2. `Qwen3-Next MTP speculative decoding vLLM SGLang support` + `Qwen3-Next-80B-A3B MTP multi-token prediction head model card`

### `web_fetch` URLs (verbatim, in order)
1. `https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-Eagle3/raw/main/README.md` — **HTTP 401**
2. `https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-Eagle3` — **HTTP 404** (404 page body)
3. `https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3` — HTTP 200
4. `https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ/raw/main/README.md` — HTTP 200
5. `https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3/raw/main/README.md` — HTTP 200
6. `https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding` — HTTP 200 (nav only, content truncated)
7. `https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/recipes-e2e-examples/dflash-speculative-decoding.md` — **HTTP 404**
8. `https://inco.ai/blog/dflash2/` — HTTP 200
9. `https://raw.githubusercontent.com/devnen/qwen3.6-windows-server/v1.3.3/docs/MTP_HEAD.md` — HTTP 200
10. `https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/raw/main/README.md` — HTTP 200
11. `https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16/raw/main/README.md` — HTTP 200
12. `https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/main/README.md` — HTTP 200
13. `https://raw.githubusercontent.com/john-rocky/CoreML-LLM/refs/heads/main/docs/EAGLE3_INTEGRATION_STATE.md` — HTTP 200
14. `https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/dflash.mdx` — HTTP 200
15. `https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/guides/speculative/eagle.mdx` — HTTP 200
16. `https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/refs/heads/main/phases/17-infrastructure-and-production/05-eagle3-speculative-decoding/docs/en.md` — HTTP 200
17. `https://raw.githubusercontent.com/noonghunna/club-3090/refs/tags/v0.10.2/docs/engines/SGLANG.md` — HTTP 200
18. `https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/README.md` — HTTP 200
19. `https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3/raw/main/README.md` — HTTP 200
20. `https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/README.md` — HTTP 200
21. `https://huggingface.co/lmsys/EAGLE3-LLaMA3.1-Instruct-8B/raw/main/README.md` — **HTTP 401**
22. `https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/raw/main/README.md` — HTTP 200

### `curl` URLs fetched (reachability probes + tarballs)
```
200  https://docs.vllm.ai/en/latest/features/spec_decode.html   (redirects to .../features/spec_decode/)
200  https://docs.sglang.io/advanced_features/speculative_decoding.html
000  https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-Eagle3
000  https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-Eagle3/raw/main/README.md
000  https://huggingface.co/
000  https://hf.co/
000  https://hf-mirror.com/nvidia/Llama-3.1-70B-Instruct-Eagle3/raw/main/README.md
000  https://huggingface.co/api/models/nvidia/Llama-3.1-70B-Instruct-Eagle3
401  https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-Eagle3/raw/main/README.md   (with proxy env)
401  https://hf-mirror.com/nvidia/Llama-3.1-70B-Instruct-Eagle3/raw/main/README.md  (with proxy env)
200  https://docs.vllm.ai/en/latest/features/speculative_decoding.html
404  https://docs.vllm.ai/en/latest/features/speculative_decoding/README.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/mlp.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/parallel_draft_model.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/speculators.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification.html
200  https://docs.vllm.ai/en/latest/features/speculative_decoding/extract_hidden_states.html
200  https://docs.sglang.io/SpecForge/
200  https://docs.vllm.ai/projects/speculators/en/latest/
200  https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/eagle-speculative-decoding
200  https://github.com/sgl-project/SpecForge
200  https://github.com/vllm-project/speculators
```

### Repos downloaded and read locally (via `codeload.github.com/.../tar.gz/refs/heads/main`)
```
https://codeload.github.com/vllm-project/vllm/tar.gz/refs/heads/main        -> vllm-main/      (124 MB extracted)
https://codeload.github.com/sgl-project/sglang/tar.gz/refs/heads/main       -> sglang-main/    (131 MB)
https://codeload.github.com/sgl-project/SpecForge/tar.gz/refs/heads/main    -> SpecForge-main/
https://codeload.github.com/vllm-project/speculators/tar.gz/refs/heads/main -> speculators-main/
https://codeload.github.com/SafeAILab/EAGLE/tar.gz/refs/heads/main          -> EAGLE-main/
https://codeload.github.com/FasterDecoding/Medusa/tar.gz/refs/heads/main    -> Medusa-main/
```

### Local files read (primary doc/code sources)
```
vllm-main/docs/features/speculative_decoding/README.md
vllm-main/docs/features/speculative_decoding/eagle.md
vllm-main/docs/features/speculative_decoding/mtp.md
vllm-main/docs/features/speculative_decoding/mlp.md
vllm-main/docs/features/speculative_decoding/draft_model.md
vllm-main/docs/features/speculative_decoding/speculators.md
vllm-main/vllm/config/speculative.py
vllm-main/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
vllm-main/vllm/model_executor/models/qwen3_next_mtp.py
vllm-main/vllm/model_executor/models/registry.py
sglang-main/docs/docs/advanced_features/speculative_decoding.mdx
sglang-main/python/sglang/srt/models/qwen3_next_mtp.py
SpecForge-main/README.md
SpecForge-main/docs/sections/basic_usage/training.md
speculators-main/README.md
speculators-main/docs/user_guide/algorithms/mtp.md
speculators-main/docs/user_guide/tutorials/train.md
EAGLE-main/README.md
```

### Blocked / failed access (report, do not retry blindly)
- `raw.githubusercontent.com` via `curl` → HTTP 000. Works via `web_fetch`.
- `huggingface.co` via `curl` (any path) → HTTP 000 / 401. Works via `web_fetch`.
- `hf-mirror.com` → 401 with proxy, 000 without.
- `docs.vllm.ai/en/latest/features/speculative_decoding/README.html` → 404 (index is `speculative_decoding.html`).
- `https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/recipes-e2e-examples/dflash-speculative-decoding.md` → 404 (correct path is `docs/guides/speculative/`).
