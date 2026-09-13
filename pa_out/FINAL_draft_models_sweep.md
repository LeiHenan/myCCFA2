# Prior-art / gap-map: Speculative decoding — DRAFT-MODEL DRAFTING and DRAFT TRAINING

Sweep date **2026-09-13**. Window prioritised: **2025-06 → 2026-09**; older items only where canonical.

**Label convention.** `READ BODY` = the page/PDF/abstract text was actually retrieved and read (by me or by a delegated sweep whose fetch is recorded). `TITLE ONLY` = only a search-result title, snippet, label line, or API state field was seen. No row is marked READ BODY on the strength of a guess.

**Coverage note.** Four parallel sub-sweeps were run (DFlash/architecture; quantisation/vocab; abandoned work; training data/cost). Their labels are carried through unchanged. Where I re-fetched and verified a claim myself it is marked `READ BODY (verified)`.

---

## A. CLOSED (someone shipped or published a working answer)

| # | Question it answers | Who closed it | Artifact (paper/PR/flag) | URL | Status | READ BODY or TITLE ONLY |
|---|---|---|---|---|---|---|
| A1 | Can a **standalone small draft model** (e.g. Qwen3-0.6B) draft for a larger target (e.g. Qwen3-4B) in vLLM v1? | tomasruizt (PR author) / vLLM, merged by benchislett | vLLM PR #24322 `feat: spec decode with draft models` — flag `--speculative-config '{"model":"Qwen/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":3,"disable_padded_drafter_batch":true}'` | https://github.com/vllm-project/vllm/pull/24322 | **merged 2026-01-19** | READ BODY (verified) |
| A2 | Same question, official docs form | vLLM project | `docs/features/speculative_decoding/parallel_draft_model.md` | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/parallel_draft_model.md | shipped (docs) | READ BODY (verified) |
| A3 | **Draft↔target vocabulary/tokenizer mismatch** — lossless speculative decoding across different tokenizer families | wan-danfeng / vLLM, merged by ywang96 | vLLM PR #38174 `Universal speculative decoding for heterogeneous vocabularies (TLI)` — flag `use_heterogeneous_vocab`; based on Timor et al., ICML 2025 oral, arXiv 2502.05202 | https://github.com/vllm-project/vllm/pull/38174 | **merged 2026-07-02** | READ BODY (verified) |
| A4 | Same, HuggingFace Transformers reference implementation | Timor et al. / HF | transformers PR #35029 `Universal Speculative Decoding CandidateGenerator` | https://github.com/huggingface/transformers/pull/35029 | merged | READ BODY (via A3's description) |
| A5 | Same, SGLang (cross-family STANDALONE drafts) | jmamou / SGLang | SGLang PR #22883 TLI (`--speculative-algorithm TLI`); vocab intersection built at startup; draft LM head pruned to `[intersection_size × hidden]` | https://github.com/sgl-project/sglang/pull/22883 | **Open** as of retrieval (33 commits) | READ BODY (verified) |
| A6 | SGLang STANDALONE silently corrupting output on vocab mismatch | jmamou / kpham-sgl / SGLang | SGLang PR #23838 `Validate vocabulary compatibility in STANDALONE mode` — `_validate_vocab_compatibility()`, 8 CPU unit tests | https://github.com/sgl-project/sglang/pull/23838 | **merged** (commit 5556631) after ~2 months stalled | READ BODY (verified) |
| A7 | **Lossless probabilistic rejection sampling for `draft_model`** (draft_probs were being dropped) | bedeks / vLLM, merged by mgoin | vLLM PR #40269 (closes #40149); wires real draft probabilities into `RejectionSampler` | https://github.com/vllm-project/vllm/pull/40269 | **merged 2026-05-14** | READ BODY (verified) |
| A8 | DFlash — block-diffusion drafter, single forward pass, target-conditioned | Jian Chen, Yesheng Liang, Zhijian Liu | arXiv 2602.06036 `DFlash: Block Diffusion for Flash Speculative Decoding` | https://arxiv.org/abs/2602.06036 | **published, ICML 2026 camera-ready** (v1 2026-02-05, v2 2026-05-28) | READ BODY (verified) |
| A9 | DFlash drafter **training** in the vLLM-native stack | Red Hat AI Model Optimization | Speculators v0.5.0 — `--speculator-type dflash`, `--block-size`, `--max-anchors`, `--draft-vocab-size`, `--num-layers`, `--target-layer-ids` | https://vllm.ai/blog/2026-05-28-speculators-v050 | shipped (v0.5.0) | READ BODY (verified) |
| A10 | DFlash **serving** in vLLM | ZhanqiuHu / vLLM | vLLM PR #38300 `Add DFlash speculators config parsing` | https://github.com/vllm-project/vllm/pull/38300 | **merged**, in `vllm>=0.20.0` | READ BODY (verified) |
| A11 | DFlash2 — DFlash + grouped dynamic depthwise conv + top-k candidate path selector | Inco AI (z-lab) | `DFlash 2: Keep Drafting Parallel`, 2026-08-18; code https://github.com/z-lab/dflash | https://inco.ai/blog/dflash2/ | shipped (blog + weights) | READ BODY (verified) |
| A12 | DFlash2 in vLLM as a **separate architecture** | SubSir / z-lab / vLLM (merged by maintainers) | vLLM PR #52816 — `qwen3_dflash2.py`, `v1/worker/gpu/spec_decode/dflash2/speculator.py` | https://github.com/vllm-project/vllm/pull/52816 | **merged 2026-08-21** | READ BODY (delegated) |
| A13 | DFlash2 in SGLang | SubSir / z-lab / SGLang | SGLang PR #35371 — conv+selector in `srt/models/dflash.py`, `dflash_worker_v2.py` | https://github.com/sgl-project/sglang/pull/35371 | **merged 2026-08-19** | READ BODY (delegated) |
| A14 | DFlash2 in llama.cpp | z-lab / SubSir / ggml-org | llama.cpp PR #27342 — `--spec-type draft-dflash` | https://github.com/ggml-org/llama.cpp/pull/27342 | merged (16 commits) | READ BODY (delegated) |
| A15 | Unified **draft-model training runtime** across 6+ drafting families | SpecForge team (LMSYS/SGLang) | SpecForge v0.3.0 — EAGLE3, EAGLE3.1, P-EAGLE, DFlash, DFlash2, Domino, DSpark; disaggregated online + offline; `specforge train --config run.yaml` | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | shipped (v0.3.0, 2026-08-04) | READ BODY (verified) |
| A16 | SpecForge training framework, measured | SpecForge team | arXiv 2603.18567 — "up to 9.9× faster EAGLE-3 training for Qwen3-235B-A22B" | https://arxiv.org/abs/2603.18567 | published | READ BODY (delegated) |
| A17 | **Target-agnostic** standalone parallel draft model (one draft serves a whole target family) | AMD (Zihao An et al.) | PARD, arXiv 2504.18583; weights `amd/PARD-Qwen3-0.6B`; vLLM flag `parallel_drafting: true` | https://arxiv.org/abs/2504.18583 ; https://huggingface.co/collections/amd/pard | published + shipped in vLLM | READ BODY (verified) |
| A18 | **Draft training-data curation**: regenerate responses with the target | SpecForge team | SpecForge v0.3 blog §"Feature source is not data policy"; SpecForge paper §7.1 (regeneration = +5.3% avg throughput) | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | shipped as documented practice + `regenerate_train_data.py` | READ BODY (verified) |
| A19 | **Data-efficient draft training** — which tokens are worth training on | Jiaming Fan et al. | SFDD / "Flatter Tokens are More Valuable for Speculative Draft Model Training", ICLR 2026, arXiv 2601.18902; code github.com/fjm9933/Flatness | https://arxiv.org/abs/2601.18902 | published (ICLR 2026) | READ BODY (delegated) |
| A20 | **SFT plateaus — fix it with on-policy distillation** | Haodi Lei et al. | Draft-OPD, arXiv 2605.29343 | https://arxiv.org/abs/2605.29343 | published (v2 2026-05-29) | READ BODY (verified) |
| A21 | **Draft depth/width**: how many layers should a drafter have? | DFlash authors | DFlash §5 + Table 6 — 5 layers (8 for Qwen3 Coder); speedup peaks at 5 even though τ keeps rising to 8 | https://arxiv.org/html/2602.06036v2 | published | READ BODY (delegated) |
| A22 | Same question for parallel EAGLE | Mude Hui et al. (Amazon) | P-EAGLE, arXiv 2602.01469 — 1→2 layers +33%, 4 layers +9.5% more; "Single-layer is thus optimal for AR EAGLE throughput" | https://arxiv.org/abs/2602.01469 | published | READ BODY (delegated) |
| A23 | Same, for semi-autoregressive drafting | Xin Cheng et al. (DeepSeek + PKU) | DSpark, arXiv 2607.05147 — "a 2-layer DSpark outperforms the 5-layer DFlash baseline across all domains" | https://arxiv.org/abs/2607.05147 | published | READ BODY (delegated) |
| A24 | **Draft vocabulary pruning** without the out-of-vocabulary penalty | Miles Williams et al. | SpecVocab, "Speculative Decoding with a Speculative Vocabulary", Findings of ACL 2026, arXiv 2602.13836 | https://arxiv.org/abs/2602.13836 | published (ACL 2026 Findings) | READ BODY (verified) |
| A25 | Canonical reduced-draft-vocab reference | Zhao, Pan, Han et al. | FR-Spec, ACL 2025, arXiv 2502.14856 | https://arxiv.org/abs/2502.14856 | published | TITLE ONLY |
| A26 | Reduced draft vocab is a **first-class training flag** | vLLM/Red Hat | Speculators `--draft-vocab-size`, `--token-freq-path`, `--t2d-path`, `--d2t-path` | https://raw.githubusercontent.com/vllm-project/speculators/main/docs/cli/train.md | shipped | READ BODY (verified) |
| A27 | EAGLE-3 acceptance drift in long context → fixed config-wise | EAGLE/vLLM/TorchSpec teams | EAGLE 3.1 — post-norm + FC-norm fix; "config-driven extension of the existing EAGLE 3 implementation" | https://raw.githubusercontent.com/vllm-project/vllm-project.github.io/refs/heads/main/_posts/2026-05-26-eagle-3-1.md | shipped | READ BODY (delegated) |
| A28 | Root-cause paper for that drift | Doğaç Eldenk et al. | "When Hidden States Drift", arXiv 2605.09992 | https://arxiv.org/abs/2605.09992 | published | READ BODY (delegated) |
| A29 | **Online / on-the-fly draft training**, disaggregated | SpecForge team | SpecForge v0.3 — patched `sglang==0.5.14` `--enable-spec-capture` + Mooncake; 3 capture servers + 5 trainers on 8×H20 ≈ +10% training throughput | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | shipped | READ BODY (verified) |
| A30 | Online + offline draft training unified behind vLLM native hidden-state extraction | Red Hat AI | Speculators v0.5.0 (vLLM ≥0.18.0 hidden-states extraction; vLLM no longer a Python dependency) | https://vllm.ai/blog/2026-05-28-speculators-v050 | shipped | READ BODY (verified) |
| A31 | Online speculator trained from live traffic in a serve→train flywheel | Together AI | Aurora, arXiv 2602.06932 + blog | https://arxiv.org/abs/2602.06932 | published + product | READ BODY (delegated) |
| A32 | Canonical online/on-the-fly draft updating | Xiaoxuan Liu et al. | Online Speculative Decoding, arXiv 2310.07177 (v4 2024-06-10) | https://arxiv.org/abs/2310.07177 | published (canonical) | READ BODY (delegated) |
| A33 | Canonical distillation-for-drafts reference | Yongchao Zhou et al. | DistillSpec, arXiv 2310.08461 | https://arxiv.org/abs/2310.08461 | published (canonical) | READ BODY (delegated) |
| A34 | MTP-head draft **fine-tuning** on domain data | vLLM/Red Hat Speculators (FastMTP) | Speculators MTP finetuning support; FastMTP arXiv 2509.18362 | https://github.com/vllm-project/speculators ; https://arxiv.org/abs/2509.18362 | shipped + published | READ BODY (delegated) |
| A35 | **Draft hidden size / architecture must match verifier today** — what is enforced | vLLM/Red Hat | Speculators `--draft-config` docs | https://raw.githubusercontent.com/vllm-project/speculators/main/docs/cli/train.md | shipped constraint | READ BODY (verified) |
| A36 | Draft **training works on a single GPU** (contra the assumption that it needs a cluster) | vLLM/Red Hat | Speculators `train` CLI: "Supports single-GPU and multi-GPU distributed training"; `--fsdp-shard`, `--gradient-checkpointing` | https://raw.githubusercontent.com/vllm-project/speculators/main/docs/cli/train.md | shipped | READ BODY (verified) |
| A37 | Community reference: **≈$3 / 2.5 h on one 24 GB GPU** to train a usable standalone 0.5B drafter | Vexp / Horizon | `vexp-ai/horizon-draft-0.5b` | https://huggingface.co/vexp-ai/horizon-draft-0.5b | shipped (weights) | READ BODY (delegated) |
| A38 | Reference point: training an EAGLE3 drafter for a **Qwen3-4B** target | huluhuluu (community) | `huluhuluu/Qwen3-4B-Instruct-2507-EAGLE3-ShareGPT-*` (1 layer, hidden 2560, draft vocab 32000 vs target 151936) | https://huggingface.co/huluhuluu/Qwen3-4B-Instruct-2507-EAGLE3-ShareGPT-full-context-epoch1-step35000 | shipped (weights) | READ BODY (delegated) |
| A39 | Draft model **works fine against a quantised target** | ravingamm (community, llama.cpp) | HF discussion on `z-lab/Qwen3.8-27B-DFlash2` #9 | https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/discussions/9 | reported working | READ BODY (delegated) |
| A40 | EAGLE3 drafter on a **W4A8-quantised** Kimi-K2.5 target, no accuracy regression | AMD ROCm / AITER / FlyDSL | ROCm blog: TPOT 42.73 ms → 27.41 ms (−35.9%) on 8× MI325X | https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-speculative/README.html | shipped (recipe + image) | READ BODY (verified) |
| A41 | Scale-up answer for speculative decoding in production engines | Liu, Yu, Park, Stoica, Cheung | "Speculative Decoding: Performance or Illusion?", arXiv 2601.11580, MLSys 2026 **oral** | https://arxiv.org/abs/2601.11580 ; https://mlsys.org/virtual/2026/oral/3782 | published (MLSys 2026) | READ BODY (verified) |
| A42 | Scaling laws for draft acceptance vs capacity/tokens/batch | Siyuan Yan, Mo Zhu, Guo-qing Jiang et al. | arXiv 2505.07858 `Scaling Laws for Speculative Decoding` (Scylla) | https://arxiv.org/abs/2505.07858 | published (v1 2025-05-08) | READ BODY (verified) |

---

## B. OPEN (explicitly unsolved, with evidence someone is still asking)

| # | The question | WHO is still asking | URL | What specifically is missing | VERBATIM quote of the asking | READ BODY or TITLE ONLY |
|---|---|---|---|---|---|---|
| B1 | Should draft **block size** be scheduled adaptively at inference time? | DFlash authors (Chen, Liang, Liu) | https://arxiv.org/html/2602.06036v2 | No adaptive block-size scheduler exists; only the observation that large blocks generalise down to small ones and that large blocks over-pay under compute-bound (large-batch) serving | "In practical serving scenarios, large blocks can increase verification cost under compute-bound settings (e.g., large batch sizes); reducing the block size in such cases can therefore yield better overall speedup. **We leave adaptive block-size scheduling to future work.**" | READ BODY (delegated) |
| B2 | Can low-acceptance requests be detected early enough to skip the draft block entirely? | DSpark authors (Xin Cheng et al., DeepSeek + PKU) | https://arxiv.org/html/2607.05147v1 | The parallel backbone always pays for a full γ-token block even when acceptance will be near zero; no difficulty-aware early exit exists | "**Although the prefix scheduler minimizes wasted target-model verification, DSpark still incurs a fixed draft-side cost to generate the initial γ-token block via the parallel backbone. For complex queries with inherently low acceptance rates, this upfront drafting compute is unrecoverable.** Future optimizations could introduce difficulty-aware early exiting within the draft model, enabling such requests to bypass full-block generation." | READ BODY (delegated) |
| B3 | Is there a lightweight predictor of per-position acceptance that actually works across workloads? | Xiaoxuan Liu, Jiaxiang Yu, Jongseok Park, Ion Stoica, Alvin Cheung (MLSys 2026 oral) | https://arxiv.org/html/2601.11580v2 | Their oracle upper bound shows up to 2.2× further headroom over the best fixed strategy; the predictor that would capture it does not exist | "Taken together, these results point to a promising direction for future work: **developing an accurate yet lightweight predictor capable of adapting to varying levels of acceptance behavior across workloads, requests and token positions.**" | READ BODY (verified) |
| B4 | How much of speculative decoding's remaining headroom is reachable at all? | Same MLSys 2026 paper | https://arxiv.org/html/2601.11580v2 | Measured-vs-oracle gap is unquantified for draft-model variants; they also defer context-overlap analysis | "**We leave an analysis that measures overlap with respect to this evolving context for future work.**" ; "no proposal mechanism can perfectly predict future target tokens. Hence, the oracle configuration serves purely as an upper bound" | READ BODY (verified) |
| B5 | Is there a better candidate **selector** for parallel drafters than pairwise top-k? | Inco AI (DFlash2) | https://inco.ai/blog/dflash2/ | Only a pairwise scoring selector and a grouped depthwise conv are implemented; the design space is explicitly declared open | "**Pairwise scoring is the simplest selector we could think of, and we believe there is plenty to explore.**" ; "Choosing is cheaper than predicting." | READ BODY (verified) |
| B6 | Can the **draft hidden size differ from the verifier**? | vLLM/Red Hat Speculators | https://raw.githubusercontent.com/vllm-project/speculators/main/docs/cli/train.md | Architecture-shaping is bounded by the verifier's hidden dim | "the draft `hidden_size` **must match the verifier (mismatch is not yet supported)**." | READ BODY (verified) |
| B7 | Can non-causal sliding-window draft attention be **deployed**? | vLLM/Red Hat Speculators | https://raw.githubusercontent.com/vllm-project/speculators/main/docs/cli/train.md | Training flag exists; no engine support | "Use non-causal (bidirectional) masking within draft blocks for sliding window attention layers. Full attention layers are always bidirectional. **Note: vLLM currently doesn't support these models.**" | READ BODY (verified) |
| B8 | Can **P-EAGLE** be trained offline, or beyond batch size 1? | SpecForge team | https://raw.githubusercontent.com/sgl-project/SpecForge/main/docs/sections/basic_usage/training.md | Support matrix lists P-EAGLE as online-disaggregated only, batch-size-1 only; offline columns are "No" | "P-EAGLE requires `training.batch_size=1` and reuses EAGLE3's server capture schema" ; support matrix row: `P-EAGLE | Yes, consumer DP, batch size 1 | No | No` | READ BODY (verified) |
| B9 | Can **VLM / multimodal** drafters be trained in the unified runtime? | SpecForge team | https://raw.githubusercontent.com/sgl-project/SpecForge/main/docs/sections/basic_usage/training.md | Text-only runtime; multimodal drafting remains unexplored at scale | "**VLM training, including Qwen2.5-VL, is not supported.** The unified runtime currently accepts text inputs only" ; "**online evaluation is not supported.** Evaluation requires precomputed offline features" | READ BODY (verified) |
| B10 | Is block-parallel/diffusion drafting ready for **multimodal** targets? | Yantao Li et al., arXiv 2608.20743 | https://arxiv.org/abs/2608.20743 | The paper's own framing: block-parallel generative drafting in multimodal models is an open question, and their empirical study finds it often **loses** to autoregressive | "While this transition is well studied in text-only LLMs, **its applicability to multimodal models remains an open question.** … however, block-parallel generative drafting remains largely unexplored." | READ BODY (delegated) |
| B11 | Does **FR-Spec-style draft-vocab trimming** apply to native MTP drafts in llama.cpp? | avifenesh / llama.cpp | https://github.com/ggml-org/llama.cpp/issues/25187 | Native MTP path has no `d2t` equivalent; no `convert_hf_to_gguf.py` producer; cross-architecture validation not done | "Before building a convert_hf_to_gguf.py producer path and turning this into a PR, **I'd like to check this is a direction the project wants**, and get input on a few open questions… **I have not verified this on real hardware** … I'm flagging this gap rather than papering over it." | READ BODY (verified) |
| B12 | Do **quantised MTP/NextN drafts** load correctly in SGLang? | chuck-ads / SGLang | https://github.com/sgl-project/sglang/issues/36599 | `deepseek_nextn.py` hardcodes `quant_config = None` for `modelopt_fp4`, making `--speculative-draft-model-quantization` a no-op for GLM-style checkpoints | "**Note it took `--speculative-draft-model-quantization modelopt_fp4` just to get a quant config to the draft at all** (the draft doesn't auto-detect from the checkpoint the way the target does) — and then this override silently discarded it, **which makes the flag a no-op for this case.**" | READ BODY (verified) |
| B13 | Why does EAGLE3 acceptance **collapse on ROCm**? | tanpinsiang / vLLM | https://github.com/vllm-project/vllm/pull/47854 | Fix PR is **open/unmerged**; root cause only partially isolated | "In the default ROCm backend route, the drafter can select `ROCM_ATTN`. **With that path, acceptance can collapse enough that speculative decoding stops helping.**" (2.83% acceptance / 1.08 accepted length vs 58.27% / 2.75 on TRITON_ATTN) | READ BODY (verified) |
| B14 | Long-range draft decay — does test-time training actually fix it? | KVShot authors, arXiv 2604.26412 | https://arxiv.org/abs/2604.26412 | TTT is claimed as the remedy by prior work; this paper finds it does not work, and its own fix still gives marginal end-to-end gains | "**Existing work attributes this decay to train-inference mismatch and proposes test-time training (TTT) as a remedy, yet we observe that long-range decay persists even in TTT-trained drafters.**" ; "end-to-end speedups remain marginal under current training pipelines" | READ BODY (delegated) |
| B15 | Official **Qwen3-8B EAGLE3 online recipe** appears unstable — is the LR wrong? | heiheiha798 / SpecForge | https://github.com/sgl-project/SpecForge/issues/577 | No resolution recorded; the shipped default remains 1e-4 | "the default `--learning-rate 1e-4` **may be too aggressive** for the Qwen3-8B EAGLE3 setting, at least during early training." ; "with the official 1e-4 LR, the run shows **many very large gradient-norm outliers** in the middle of training" | READ BODY (delegated) |
| B16 | Official DFlash numbers are **not reproducible** by users | 5SSjw / SpecForge | https://github.com/sgl-project/SpecForge/issues/469 | Official baseline 4.72×/τ 5.97 vs user's best 2.86×/τ 3.55 after 142k steps; no official reconciliation published | "official baseline: speedup: 4.72x, τ: 5.97" vs "Step 142,000: speedup: 2.86×, τ: 3.55" … "In my runs, improvements are clearly diminishing, but **even after ~12 epochs the metrics are still moving upward.**" | READ BODY (delegated) |
| B17 | Training acceptance 98% → inference acceptance 10%: why? | xinanjiao / SpecForge | https://github.com/sgl-project/SpecForge/issues/533 | No diagnosis recorded in-thread | "**This training used 70,000 data points, achieving a 98% acceptance rate during training.** I used a version of VLLM that supports Dflash for inference, but **the average acceptance rate was only 10%.**" | READ BODY (delegated) |
| B18 | Disaggregated online draft training is **~3.2× slower than plain DP** for one user | junzhang-zj / SpecForge | https://github.com/sgl-project/SpecForge/issues/718 | Contradicts the +10% headline; no published reconciliation | "I'm training a draft model for Qwen3.6-35B-A3B using SpecForge and observing a **significant wall-clock time difference between the disaggregated topology and the original DP (data-parallel) approach.**" (issue title: "~3.2x slower than DP") | READ BODY (delegated) |
| B19 | VP-Drafter was **silently removed** and the config silently falls back — is that intended? | Ulitochka / SpecForge | https://github.com/sgl-project/SpecForge/issues/679 | Config left stale in-tree; no deprecation notice | "the VP-Drafter-specific prefix sampling, masking, and loss logic **has been removed**." … "As a result, the current `qwen3-8b-dta.json` configuration seems to **silently fall back to regular DFlash training** instead of enabling `vp_drafter`." | READ BODY (delegated) |
| B20 | Is SpecForge's "TTT" the EAGLE-3 paper's concept at all? | haiduo / SpecForge | https://github.com/sgl-project/SpecForge/issues/227 | Dispute never formally resolved | "**TTT implementation is not the original EAGLE-3 paper's concept, but merely data augmentation.**" | READ BODY (delegated) |
| B21 | Online DFlash training OOMs; is the offline path coming? | UltramanKuz / SpecForge | https://github.com/sgl-project/SpecForge/issues/521 | At time of asking, no offline DFlash trainer; user blocked | "I would like to know if there are any plans for developing offline training for Dflash? **The online training has caused the OOM issue for me**" | READ BODY (delegated) |
| B22 | How big a drafter can be justified for multi-layer/step-disaggregated drafting, given training cost? | Akemiiii / SpecForge (open RFC) | https://github.com/sgl-project/SpecForge/issues/444 | Open RFC; no accepted design | "SGLang also implemented multi-layer eagle worker. **However, these changes may increase draft model inference overhead and the training cost.**" | READ BODY (delegated) |
| B23 | Is *training your own drafter* worth the money at all? | colizu2020 / NVIDIA DGX Spark forum | https://forums.developer.nvidia.com/t/training-a-personal-draft-model-for-qwen/378611 | No cost/benefit study published for the single-node case; answered only by anecdote | "**Is it worth the effort, or should I wait until the next Qwen release** which may or may not support the new speculative methods natively?" — reply: "**There's no free lunch here - training will cost you couple of hundreds with uncertain benefits**" (alexander.kachur) | READ BODY (verified) |
| B24 | Larger-scale draft training was not explored for lack of compute | DFlare authors, arXiv 2606.02091 | https://arxiv.org/abs/2606.02091 | Explicit resource limit; no scaling study | "the training cost of DFlare is high due to the large draft model and the scaled training corpus… **we were unable to explore this direction due to computational resource constraints. We leave the investigation of even larger-scale training to future work.**" | READ BODY (delegated) |
| B25 | Should TTT length be dynamic per sample? | SpecForge team, arXiv 2603.18567 | https://arxiv.org/abs/2603.18567 | Static TTT length wastes training cost across heterogeneous samples | "For cross-domain training, dynamically adjusting the TTT length based on the sample type could further reduce training cost, as not all samples require the same degree of training-time testing. **We leave the design and evaluation of such dynamic TTT strategies to future work.**" | READ BODY (delegated) |
| B26 | Can an **offline-trained** drafter be kept from going stale in production? | Together AI (Aurora) | https://www.together.ai/blog/aurora | Offline retraining cadence cannot track traffic drift | "**draft models go stale, acceptance rates drift, and offline retraining is too slow and too expensive to keep pace with live traffic.**" | READ BODY (delegated) |
| B27 | Is online adaptation even cost-effective at short generation lengths? | Weijie Shi et al. (SpecBlock), arXiv 2605.07243 | https://arxiv.org/abs/2605.07243 | They declined to evaluate adaptation where the backward pass cannot be amortised | "**HumanEval at 164 prompts and MT-Bench at 80 prompts are too short for the acceptance gain to amortize the backward cost of adaptation, so we do not evaluate adapt on them.**" | READ BODY (delegated) |

---

## C. ATTEMPTED-AND-ABANDONED

Every row below carries a **verbatim quote** and a URL. Method note carried from the sweep: GitHub's literal `wontfix` string and the `not planned` **label** return **zero** results on vLLM/SGLang/SpecForge/speculators; the signal lives in `state_reason = not_planned` plus the `stale` label. Where a maintainer comment thread was not retrievable, the row quotes the issue **author** and reports closure state separately.

### C-1. vLLM standalone draft models: shipped → removed → re-added

| # | What was tried | Who | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|
| C1 | Standalone draft-model speculative decoding (the original v0 path) | vLLM | **Removed**; documented incompatibility for that version range | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/README.md | READ BODY |
| C2 | Draft-model spec decode on v0.13.0 (user hit the removal) | forever10086 / vLLM | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/31883 | READ BODY (verified) |
| C3 | Draft-model spec decode **on CPU** | ganeshr10 / vLLM | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/28384 | READ BODY |
| C4 | "Why is this not supported anymore?" | luxisme / tomasruizt / vLLM Forums | Removal confirmed, then reversed | https://discuss.vllm.ai/t/standalone-draft-model-spec-decode-support-in-v0-x-and-v1/2241 | READ BODY (verified) |
| C5 | The formal removal decision itself | WoosukKwon / vLLM RFC #18571 | **Closed / `stale`**; draft-model decoding listed under "Features Temporarily Discontinued" | https://github.com/vllm-project/vllm/issues/18571 | READ BODY (verified) |

**Verbatim (C1):** "Speculative decoding with draft models is not supported in `vllm<=0.10.0`" — vLLM docs, "Known Feature Incompatibility".

**Verbatim (C2):** "NotImplementedError: Speculative decoding with draft model is not supported yet. Please consider using other speculative decoding methods such as ngram, medusa, eagle, or mtp." — runtime error pasted by forever10086. *(This is literally the Qwen3-0.6B → Qwen3-4B pairing the brief asks about, rejected as not-planned.)*

**Verbatim (C3):** "NotImplementedError: Draft model speculative decoding is not supported yet. Please consider using other speculative decoding methods such as ngram, medusa, eagle, or mtp." … "This branch is the PARD implementation of Speculative decoding for V0. However, this was done a few months ago and is unsupported with V1." — ganeshr10.

**Verbatim (C4):** "Standalone draft model is removed in v0.x releases (x>10) … And v1 does not support this either. Can I know why this is not supported anymore?" — luxisme. Reply: "@RunLLM is wrong. Support for draft model was reintroduced to V1 this week in this PR: https://github.com/vllm-project/vllm/pull/24322 (I'm the PR author)." — tomasruizt.

**Verbatim (C5):** "Features Temporarily Discontinued — Encoder-decoder models (e.g., Whisper) / **Draft model-based speculative decoding** / Neuron backend …" — WoosukKwon, RFC #18571. Labels on that RFC: `RFC`, `stale`.

### C-2. vLLM MLP speculator — left un-routed, closed not-planned

| # | What was tried | Who | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|
| C6 | `mlp_speculator` still exposed in config/docs but never routed in V1 | fenghourun / vLLM | **not planned** (closed 2026-07-07) | https://github.com/vllm-project/vllm/issues/47825 | READ BODY |
| C7 | IBM `llama3-70b-accelerator` MLP speculator checkpoints | kylesayrs / vLLM | **not planned** + `stale` (closed 2026-06-09) | https://github.com/vllm-project/vllm/issues/34106 | READ BODY |

**Verbatim (C6):** "mlp_speculator is still exposed as a speculative method (in the SpeculativeMethod literal, auto-detected from model_type, with a model impl, config parser, docs page, and example) but **nothing in vllm/v1/ routes it, so it crashes at engine init.**" … "grep -rn "mlp_speculator" vllm/v1/ returns nothing (neither the classic nor the V2 runner has a proposer for it)." — fenghourun.

**Verbatim (C7):** "MLP speculation using the ibm-ai-platform/llama3-70b-accelerator models seems to be broken. Attempting to run the following code using method="mlp_speculator" or method="draft_model" results in the following traceback" … "AttributeError: 'MLPSpeculatorConfig' object has no attribute 'num_attention_heads'" — kylesayrs.

### C-3. vLLM draft-**vocabulary pruning** (FR-Spec): built, benchmarked, rejected on measurement

| # | What was tried | Who | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|
| C8 | FR-Spec draft-vocab pruning as a first-class vLLM feature | jmamou / vLLM | **not planned** | https://github.com/vllm-project/vllm/issues/24506 | READ BODY |
| C9 | `fr-spec` implementation PR (105 commits) | eitanturok / vLLM | **closed-unmerged** | https://github.com/vllm-project/vllm/pull/24343 | READ BODY |
| C10 | fr-spec, second attempt | eitanturok / vLLM | **closed-unmerged** | https://github.com/vllm-project/vllm/pull/29334 | READ BODY |
| C11 | "Spec decode with probs" (V1) | andylolu2 / vLLM | **closed-unmerged** (blocker for #24322's non-greedy path) | https://github.com/vllm-project/vllm/pull/20459 | READ BODY (verified) |
| C12 | Near-identical target/draft vocab sizes (Qwen2.5 padding 128 vs 256) disabling speculation | benchislett / vLLM | **closed-unmerged** — "Closing as V0 is no longer a priority." | https://github.com/vllm-project/vllm/pull/13849 | READ BODY (verified) |

**Verbatim (C8):** "I have implemented the FR-Spec approach at the logits processor level, using AllowedTokenIdsLogitsProcessor. This implementation does not prune the draft model itself but allows evaluating acceptance rates under different draft pruning ratios." — jmamou. *(Author-reported MT-Bench acceptance: vanilla 27.8%; ratio 0.1→28.3%, 0.25→28.6%, 0.5→28.6%, 0.75→27.2%, 0.9→25.9%, 0.99→18.8%.)*

**Verbatim (C9, reviewer):** "We definetly get a speedup over vanilla. In eagle num-spec-tokens=1, **the drafter forward pass takes 4% of the time of the target forward pass. So the drafter forward pass is not nearly a huge bottleneck, so we don't expect fr-spec to speed things up that much.**" … "frspec seems to increase throughput by **1.52%** ( =100*(121.99-120.16)/120.16 ) in the benchmark above." — keyboardAnt.

**Verbatim (C11):** "**For small models, using draft probabilities is not worth the draft sampling/rejection sampling overhead.** I've added a "kill-switch" `enable_draft_probs` in SpeculativeConfig for those cases." — andylolu2 (PR closed unmerged; the functionality later landed via #40269).

**Verbatim (C12):** "Models such as Qwen 2.5 can have identical tokenizers but slightly different vocab sizes due to padding… resulting in a mismatch. **This disables speculative decoding**, which requires the tokens and respective probabilities match between both models." — benchislett; author's own closing words: "**Closing as V0 is no longer a priority.**"

### C-4. Stale / inactivity-bot closures (the fix existed; the issue died)

| # | What was tried | Who | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|
| C13 | SGLang STANDALONE silently corrupting output on vocab mismatch | jmamou / SGLang | Issue **auto-closed by the inactivity bot** (`inactive` label) while its fix PR was still unmerged | https://github.com/sgl-project/sglang/issues/24051 | READ BODY (verified) |
| C14 | The fix PR itself | jmamou / kpham-sgl / SGLang | Stalled ~2 months; author had to ask twice | https://github.com/sgl-project/sglang/pull/23838 | READ BODY (verified) |
| C15 | SGLang TLI (cross-family drafts) | jmamou / SGLang | **Still Open** (33 commits) while vLLM's equivalent merged | https://github.com/sgl-project/sglang/pull/22883 | READ BODY (verified) |

**Verbatim (C13):** "STANDALONE speculative decoding requires the draft model to share the same vocabulary as the target model. If the vocabularies differ, draft token IDs map to different strings in the target vocabulary — **the server starts without any error or warning and every accepted draft token is silently corrupted.**" … "The subtler case — **same vocab_size but different token-to-ID mappings** — should also be caught." — jmamou. Issue labels: `inactive`.

**Verbatim (C14):** "Rechecked this after **#24051 was auto-closed by the inactivity bot.** On current head ab75b613… the focused PR validation looks good" — ronhuafeng ; then "Thanks for the re-validation @ronhuafeng ! Good to know everything passes cleanly on current head. **Could a maintainer please merge this when convenient?**" — jmamou.

### C-5. Draft-**training** (SpecForge): removed modes, unplanned RFCs, non-reproducible results

| # | What was tried | Who | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|
| C16 | **VP-Drafter** training mode for DFlash | Ulitochka / SpecForge | **Silently removed**; stale config falls back to plain DFlash | https://github.com/sgl-project/SpecForge/issues/679 | READ BODY |
| C17 | Backbone/TTT-internal position trimming for online EAGLE3 | julyanghar / SpecForge | **not planned** (closed 2026-08-27) | https://github.com/sgl-project/SpecForge/issues/706 | READ BODY |
| C18 | SpecForge's "TTT" online loop as an EAGLE-3 implementation | haiduo / SpecForge | **Disputed** — not the paper's concept | https://github.com/sgl-project/SpecForge/issues/227 | READ BODY |
| C19 | DFlash draft training on OCR, 70k samples | xinanjiao / SpecForge | **Negative result** 98% train → 10% inference acceptance | https://github.com/sgl-project/SpecForge/issues/533 | READ BODY |
| C20 | Reproducing official DFlash training numbers | 5SSjw / SpecForge | **Negative result** 4.72× official vs 2.86× after 142k steps | https://github.com/sgl-project/SpecForge/issues/469 | READ BODY |
| C21 | **Online** (on-the-fly) DFlash training | UltramanKuz / SpecForge | **OOM**; user asks for the offline path | https://github.com/sgl-project/SpecForge/issues/521 | READ BODY |
| C22 | Disaggregated online training topology | junzhang-zj / SpecForge | **Negative result** ~3.2× slower than DP | https://github.com/sgl-project/SpecForge/issues/718 | READ BODY |
| C23 | Official Qwen3-8B EAGLE3 online recipe, LR 1e-4 | heiheiha798 / SpecForge | **Negative result** — grad-norm outliers, unstable | https://github.com/sgl-project/SpecForge/issues/577 | READ BODY |
| C24 | Multi-layer / step-disaggregated draft (PRISM) | Akemiiii / SpecForge | **Open RFC** challenging EAGLE-3 scaling cost | https://github.com/sgl-project/SpecForge/issues/444 | READ BODY |

**Verbatim (C16):** "the VP-Drafter-specific prefix sampling, masking, and loss logic **has been removed**." … "As a result, the current `qwen3-8b-dta.json` configuration seems to **silently fall back to regular DFlash training** instead of enabling `vp_drafter`." — Ulitochka.

**Verbatim (C17):** "Are maintainers open to backbone-internal position trimming in the online TTT loop (behind a default-off flag), or is loss-end trimming the preferred boundary?" — julyanghar. **Closed as not planned.**

**Verbatim (C18):** "**TTT implementation is not the original EAGLE-3 paper's concept, but merely data augmentation.**" — haiduo.

**Verbatim (C19):** "This training used 70,000 data points, achieving a **98% acceptance rate during training**. I used a version of VLLM that supports Dflash for inference, but the **average acceptance rate was only 10%**." — xinanjiao.

**Verbatim (C20):** "official baseline: speedup: 4.72x, τ: 5.97" vs "Step 142,000: speedup: 2.86×, τ: 3.55" … "In my runs, improvements are clearly diminishing, but even after ~12 epochs the metrics are still moving upward." — 5SSjw.

**Verbatim (C21):** "I would like to know if there are any plans for developing offline training for Dflash? **The online training has caused the OOM issue for me**" — UltramanKuz.

**Verbatim (C22):** "I'm training a draft model for Qwen3.6-35B-A3B using SpecForge and observing a **significant wall-clock time difference between the disaggregated topology and the original DP (data-parallel) approach.**" — junzhang-zj (issue title: "~3.2x slower than DP").

**Verbatim (C23):** "the default `--learning-rate 1e-4` **may be too aggressive** for the Qwen3-8B EAGLE3 setting, at least during early training." … "with the official 1e-4 LR, the run shows **many very large gradient-norm outliers** in the middle of training" — heiheiha798.

**Verbatim (C24):** "SGLang also implemented multi-layer eagle worker. However, **these changes may increase draft model inference overhead and the training cost.**" — Akemiiii.

### C-6. `speculators` library — scope narrowed, training mode removed

| # | What was tried | Who | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|
| C25 | Draft architectures other than EAGLE-3 | scimg / speculators | **not planned** + `stale` | https://github.com/vllm-project/speculators/issues/231 | READ BODY |
| C26 | Multi-node draft training | Liccol / speculators | **not planned** | https://github.com/vllm-project/speculators/issues/356 | READ BODY |
| C27 | Ulysses sequence parallelism for draft training | momo609 / speculators | **not planned** | https://github.com/vllm-project/speculators/issues/338 | READ BODY |
| C28 | DFLASH windowed-attention backward Triton operator | Leslie360 / speculators | **not planned** | https://github.com/vllm-project/speculators/issues/1105 | READ BODY |
| C29 | **Hybrid training mode** (`--on-generate cache`) | WindChimeRan / speculators | **Removed as a breaking change** | https://github.com/vllm-project/speculators/pull/1036 | READ BODY |
| C30 | Legacy training accuracy & acceptance reports | speculators | **Removal PR** | https://github.com/vllm-project/speculators/pull/1102 | TITLE ONLY |

**Verbatim (C25):** "Any plan to support more kinds of drafts other than Eagle-3?" — scimg; closed **not planned**, labels `enhancement`, `stale`.

**Verbatim (C28):** "At production sizes **the dense backward needs ~69GB (OOM on 8×A800)** and ignores the window structure." — Leslie360.

**Verbatim (C29):** "**Removes hybrid training mode (breaking)** --on-generate cache kept freshly generated hidden states so later epochs could reuse them. **It was broken on the Mooncake backend**: MooncakeTransfer never overrode cache(), so it inherited the no-op hook on HiddenStatesTransfer, and because the cache branch bypassed the delete path entirely, **every generated sample leaked into the store while nothing was ever cached** — MooncakeTransfer.get_cached() returns None unconditionally, so **each epoch regenerated everything from scratch**." — WindChimeRan.

### C-7. llama.cpp / Ollama / TGI / TensorRT-LLM — draft heads abandoned or proven slower

| # | What was tried | Who | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|
| C31 | MTP (nextn) speculative head for Qwen3.8-Flash-Next | routhjim / llama.cpp | **closed-unmerged** (bot flagged AI-written PR text) | https://github.com/ggml-org/llama.cpp/pull/27842 | READ BODY |
| C32 | SYCL MTP on Intel Arc | R-SITES / llama.cpp | **not planned**; **21% slower at 100% draft acceptance** | https://github.com/ggml-org/llama.cpp/issues/23533 | READ BODY |
| C33 | Gemma 4 MTP merge (#23398) | UncleMart / llama.cpp | **not planned**; 40 tok/s → ~4 tok/s | https://github.com/ggml-org/llama.cpp/issues/24266 | READ BODY |
| C34 | Pipelining `draft-mtp` → `ngram-mod` | ElSnacko / llama.cpp | **not planned** + `stale`; no speedup | https://github.com/ggml-org/llama.cpp/issues/23184 | READ BODY |
| C35 | DFlash in Ollama (`muse-glimmer:30b-nvfp4-dflash`) | jakubtomas-cz / Ollama | **not planned**; user measured no effect | https://github.com/ollama/ollama/issues/17683 | READ BODY |
| C36 | Speculative decoding (two-model draft/verify) in TGI | OliverFM / HF TGI | **not planned** + `Stale` (canonical, 2023) | https://github.com/huggingface/text-generation-inference/issues/729 | READ BODY |
| C37 | Assisted Generation in TGI | sujithjoseph / HF TGI | **not planned** + `Stale` | https://github.com/huggingface/text-generation-inference/issues/314 | READ BODY |
| C38 | Medusa acceleration in TGI | eurus-ch / HF TGI | **not planned**; "acceleration in doubt" | https://github.com/huggingface/text-generation-inference/issues/1503 | READ BODY |
| C39 | EAGLE-3 acceptance length at larger batch | Wokzy / TensorRT-LLM | **not planned**; negative scaling | https://github.com/NVIDIA/TensorRT-LLM/issues/9208 | READ BODY |
| C40 | EAGLE-3 one-model config with an FP8 target | ValeGian / TensorRT-LLM | **not planned**; FP8/INT8 unsupported for the draft | https://github.com/NVIDIA/TensorRT-LLM/issues/7842 | READ BODY |
| C41–C45 | Draft-cache replay `p_min`; external DFlash temp-0 invariant; Gemma E2B draft; spec-context memory fitting; SYCL `draft-mtp` memory | llama.cpp | all **not planned** | https://github.com/ggml-org/llama.cpp/issues/26100 , /27975 , /22337 , /25408 , /23203 | TITLE ONLY |

**Verbatim (C31):** "AI-generated content : While code is allowed to be generated by AI, please write the PR description and commit messages on your own without the help of AI." — ggml-gh-bot.

**Verbatim (C32):** "**MTP is ~21% slower than generating without speculation, despite 100% draft accuracy.** This is the opposite of the expected result (typically 1.5-2x speedup on CUDA)." — R-SITES.

**Verbatim (C33):** "#23398 When I use gemma 4 12B on this new merge my tokens/sec **drop to ~4 tokens/sec, previously I was getting 40+ tokens/sec.**" — UncleMart.

**Verbatim (C34):** "Adding ngram-mod independently on top provides **no speedup, only verification overhead.**" — ElSnacko.

**Verbatim (C35):** "I'm able to get 25 tps which is respectable and on par with Meta video on their release blog … without the speculative decoding. This indicates to me that the `muse-glimmer:30b-nvfp4-dflash` **doesn't really use the DFlash speculative decoding.**" — jakubtomas-cz.

**Verbatim (C36):** "Adding this feature would require making TGI more generic, so that one can run multiple models at once. We would need to make sure that this does not degrade performance or reliability for the single model use case." — OliverFM.

**Verbatim (C38):** "Speculative(3) seems to accelerate significantly, but I didn't observe the same result with vLLM benchmark and in real use on the same A30*4. **Their results look like Medusa doesn't have that much effect.**" — eurus-ch.

**Verbatim (C39):** "**Eagle3 acc len decreases with bigger batch size**" — Wozky (issue title), closed **not planned**.

**Verbatim (C40):** "The Support Matrix of the Eagle example README suggests that **FP8/INT8 are not supported.** Even if the Eagle3 model is in FP16, using it with an FP8 target model with the one-model configuration would result in the following error" — ValeGian.

### C-8. vLLM — the "negative result" tail (all `not planned`)

| # | What was tried | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|
| C46 | **Logits-distilled** small draft, "very high" acceptance | **not planned**; ~30% *slowdown* | https://github.com/vllm-project/vllm/issues/15025 | READ BODY (verified) |
| C47 | Small Qwen3 draft for Qwen3-32B-FP8 | **not planned**; 13–14 → 10–11 tok/s | https://github.com/vllm-project/vllm/issues/21278 | READ BODY |
| C48 | Multimodal external drafters | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/33458 | READ BODY |
| C49 | Speculative Speculative Decoding (async overlap) | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/36037 | READ BODY |
| C50 | Tree speculative decode | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/37396 | READ BODY |
| C51 | Dynamic pruning of EAGLE-3 draft trees | **not planned** | https://github.com/vllm-project/vllm/issues/41823 | READ BODY |
| C52 | Dynamic speculation length w/ confidence early-exit | **not planned** | https://github.com/vllm-project/vllm/issues/36657 | READ BODY |
| C53 | Persistent in-place drafting attention metadata | **not planned** | https://github.com/vllm-project/vllm/issues/49488 | READ BODY |
| C54 | n-gram + suffix in `model_runner_v2` | **not planned** | https://github.com/vllm-project/vllm/issues/38069 | READ BODY |
| C55 | "Tracking Spec Decode Support" matrix | **not planned** (tracking abandoned) | https://github.com/vllm-project/vllm/issues/27691 | READ BODY |
| C56 | CI/CD regression testing for spec decode | **not planned** + `stale` | https://github.com/vllm-project/vllm/issues/28135 | READ BODY |
| C57 | Draft model with shorter context than target | **not planned** | https://github.com/vllm-project/vllm/issues/7859 | READ BODY |
| C58 | EAGLE3/DFlash TTFT at P99 | **not planned**; TTFT regression | https://github.com/vllm-project/vllm/issues/39790 | READ BODY |
| C59 | Lookahead decoding port to vLLM core | **closed-unmerged**, branch deleted | https://github.com/vllm-project/vllm/pull/23388 | READ BODY |
| C60 | GPT-OSS 20B as drafter | **not planned** | https://github.com/vllm-project/vllm/issues/33133 | TITLE ONLY |
| C61 | EAGLE-3 draft model length > 2048 | **not planned** | https://github.com/vllm-project/vllm/issues/23072 | TITLE ONLY |
| C62 | EAGLE-3 second-token acceptance collapse | **not planned** | https://github.com/vllm-project/vllm/issues/33330 | TITLE ONLY |
| C63 | Non-parallel spec-decode draft/target hidden-size mismatch | **not planned** | https://github.com/vllm-project/vllm/issues/37966 | TITLE ONLY |
| C64 | Medusa choice-tree spec; Medusa TP>1 hang; Medusa concurrency 2; Medusa vs baseline; "Automate SD" RFC; Speculative Streaming; per-sequence spec decode; second-device draft; "why is spec decoding slower" | all **not planned** | /20813 , /16477 , /10031 , /6777 , /4565 , /2943 , /17984 , /12200 , /8439 | TITLE ONLY |

**Verbatim (C46):** "I trained the small model using **logits distillation** of main model, so it has a good level of generation (**acceptance rate is very high**) Still I get **consistent performance drop ~30%** in terms of speed when using 5 speculative tokens, when I reduce number speculative tokens - speed increases, but **the best speed in achieved when using main model only without speculative.**" — maiiabocharova.

**Verbatim (C47):** "But it turns out that it leads to a **slower inference speed** (around 10-11 tokens/s). I don't know if it's normal." — la1ty.

**Verbatim (C48):** "Currently, multimodal EAGLE drafters are supported, but only for EAGLE drafters and only when not using parallel drafting. This is due to some complexity in managing the MRoPe position embeddings and multimodal state that **has not been explored for external drafters.**" — benchislett.

**Verbatim (C50):** "I was testing tree speculative decoding and noticed that this feature is not yet fully implemented. **It also doesn't appear to be on the current roadmap.**" — DingYiBin.

**Verbatim (C53):** "Why the existing fast path doesn't apply … **The fast path is not applicable to sequential MTP/EAGLE drafting.**" — chanh.

**Verbatim (C57):** "Cannot handle cases where distributed draft workers generate no tokens" — error text pasted by dsingal0 (target 128K ctx, draft 32K ctx).

**Verbatim (C58):** "While the expected improvement in Time-Per-Output-Token (TPOT) is confirmed, **the TTFT degradation appears to be an inherent overhead** of the speculative decoding process that might be more pronounced than previously understood." — KlyzhenkoVadim.

**Verbatim (C59):** "Port the Lookahead decoding feature from PR #72 in vllm-project/vllm-gaudi into HabanaAI/vllm-fork. This enables lookahead decoding in the Gaudi fork." — SupreetSinghPalne; PR closed, branch deleted, never merged.

### C-9. Papers that tried it and reported it does not work

| # | What was tried | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|
| C65 | PEFT-BD: LoRA-style adapter as a block-diffusion drafter on the *same* backbone | **Self-described negative result** | https://arxiv.org/abs/2607.12422 | READ BODY (verified) |
| C66 | Test-time training as the fix for long-range draft decay | **Negative result** — decay persists | https://arxiv.org/abs/2604.26412 | READ BODY |
| C67 | Diffusion/block-parallel drafters on multimodal targets | **Negative results** at high resolution | https://arxiv.org/abs/2608.20743 | READ BODY |
| C68 | Online LoRA alignment of draft vocab + parameters | **Negative framing** of full-parameter online updates | https://arxiv.org/abs/2605.27390 | READ BODY |
| C69 | Serving-time drafter adaptation at short generation lengths | **Declined to evaluate** — backward pass cannot amortise | https://arxiv.org/abs/2605.07243 | READ BODY |

**Verbatim (C65):** "Despite these advantages, **PEFT-BD does not yield a practical speedup in our Qwen3-0.6B experiments.**" … "**the drafter is parameter-efficient but not compute-efficient.**" … "**Longer accepted prefixes alone cannot compensate when draft computation remains verifier-scale.**" — Javat, Kazakov et al.

**Verbatim (C66):** "Existing work attributes this decay to train-inference mismatch and proposes test-time training (TTT) as a remedy, yet **we observe that long-range decay persists even in TTT-trained drafters.**" … "**end-to-end speedups remain marginal under current training pipelines.**" — KVShot authors.

**Verbatim (C67):** "At 8K, all settings fall to or below the autoregressive baseline, with speedups of **0.90× and 0.83×**" … "on the Qwen3-VL family, **EAGLE-3 fails to provide speedup on SGLang (0.71× for 4B and 0.88× for 8B).**" … "Even when we explicitly train a DFlash-style block-parallel drafter for Qwen3-VL-8B with multimodal data using SpecForge, it reaches only 2.14× speedup on HF, below the 2.60×" — Multimodal SD survey authors.

**Verbatim (C68):** "Online alignment improves draft quality, but **full-parameter updates introduce substantial memory and latency overhead.**" — EvoSpec authors.

**Verbatim (C69):** "**HumanEval at 164 prompts and MT-Bench at 80 prompts are too short for the acceptance gain to amortize the backward cost of adaptation, so we do not evaluate adapt on them.**" — SpecBlock authors.

### C-10. Community negative results on shipped DFlash/DFlash2 drafters

| # | What was tried | Outcome | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|
| C70 | DFlash v1 / DFlash2 vs native MTP, two machines | **Slower than MTP** | https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/discussions/3 | READ BODY |
| C71 | DFlash2 long-context decay | v1 "worth nothing" by 32k | https://github.com/ggml-org/llama.cpp/pull/27342 | READ BODY |
| C72 | MTP drafter under rejection (GGUF/Metal) | **2.3× slower than no speculation** (issue still Open) | https://github.com/ollama/ollama/issues/17776 | READ BODY |

**Verbatim (C70):** "**Both DFlash (3.6) and DFlash 2 (3.8) are slower than MTP on the 2 machines that I've tested.** … **DFlash gains seem very not true.**" — Dnonmi ; "**In reality though performance uplift is very uneven.**" — zxbc2023 ; "**My only issue is that it uses more vram than MTP.**" — sleepyeldrazi.

**Verbatim (C71):** "DFlash v1 (n=5) **decays to 1.02x by 32k, i.e. by then it is worth nothing over plain decode.**" — dagnarf ; "it seems to be **not significantly better than MTP**" — treo (RTX 3090, MTP 57.79 t/s / accept 0.6317 vs DFlash2 62.21 t/s / 0.6153).

**Verbatim (C72):** "When drafts are rejected, mtp-q4_K_M falls to 5.14 tok/s, i.e. **2.3x slower than no speculation at all. The draft alone therefore costs slightly more than a full pass.**" … "This looks like an implementation cost in the GGUF speculative path rather than a model-format problem, but I have not read the code - I am only reporting measured effects." — otsoa-dubief.

---

## D. Hardware-ruled-out for a single-GPU researcher (1× RTX PRO 6000 Blackwell 96 GB, sm120, single node)

Criteria: requires **>1 GPU**, **>96 GB**, or **multi-node**. Rows are included only where the artifact's *reported number or shipped recipe* depends on that hardware.

| # | Artifact / result | Hardware it requires | Why it is out of scope | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|
| D1 | SpecForge v0.3 pure-draft-model serving table (Domino-B16 5.25× on GSM8K, etc.) | **2× A100 (TP2)** for the Qwen3.6-27B target | Target + draft exceed 1 GPU; result is a TP2 number | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | READ BODY (verified) |
| D2 | SpecForge v0.3 disaggregated online-training system result (+10%) | **8× H20**, 3 SGLang capture servers + 5 trainer workers | The measured "+10% over colocated" is defined by the 3+5 split | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | READ BODY (verified) |
| D3 | SpecForge training-throughput table (2.01×/1.39×/4.31×/9.99× vs EAGLE3) | **8× NVIDIA H200**, seq 4096 | "up to 9.9× faster EAGLE-3 training for Qwen3-235B-A22B" is an 8×H200 number; EAGLE ZeRO-2 OOMs at that scale | https://arxiv.org/abs/2603.18567 | READ BODY (delegated) |
| D4 | SpecBundle DFlash drafter for Qwen3.5-397B-A17B | **8× B200 (TP8)**, bf16 | Target alone is ~397B params; block-16 numbers (4.31× HumanEval) measured on 8×B200 | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | READ BODY (verified) |
| D5 | SpecBundle DSpark drafter for Kimi-K3 | **8× B300** | Kimi-K3 target far exceeds 96 GB; conc-1 3.14× measured on 8×B300 | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | READ BODY (verified) |
| D6 | SpecBundle EAGLE3 drafts (Step-3.5-Flash, Qwen3-32B, Kimi-K2.7-Code) | **4× H200** (conc 16) and **8× H200** (conc 8) | The published speedups are multi-GPU, multi-node-class results | https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/ | READ BODY (verified) |
| D7 | Ling-3.0-flash NEXTN + DSpark batch-1 campaign (TPOT 3.33 → 1.53 ms; DSpark 1120 tok/s) | **4× NVIDIA Blackwell**, TP4, ~63 GB weights **per rank** | Weights alone are ~252 GB across ranks; the whole post is a TP4 result | https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell | READ BODY (verified) |
| D8 | RepSpec (structural re-parameterised draft training) | **8× A100 (80 GB)** training, **2× A100** inference | Training rig is 8 GPUs; inference is 2 | https://openreview.net/forum?id=bqEi97qzzz | READ BODY (delegated) |
| D9 | PARD training recipe | **8× MI250X** | The published PARD adaptation runs (k=8, r=0.7, 4 epochs) are 8-GPU AMD | https://arxiv.org/abs/2504.18583 | READ BODY (delegated) |
| D10 | PRISM drafters | **8× A100-40G + 4× A100-80G**, "1 day to 2 weeks" | Datacenter-scale draft training; 12 GPUs total | https://arxiv.org/abs/2602.01762 | READ BODY (delegated) |
| D11 | "Speculative Decoding: Performance or Illusion?" measurements | **H100-80GB**; 1 GPU for 8B, **4 GPUs (TP4) for 70B/106B**; gift DGX server | The Llama3-70B draft-model result needs TP4; their own static-memory arithmetic puts Llama3-70B + Llama3.2-1B at **133.7 GiB** — over 96 GB | https://arxiv.org/html/2601.11580v2 | READ BODY (verified) |
| D12 | Llama3-70B draft-model pairing arithmetic | **133.7 GiB** static weights ((70.55 + 1.23)×10⁹ × 2 bytes); draft-side per-token KV 352 KiB vs 324 KiB for EAGLE | Explicitly exceeds 96 GB in bf16 | https://arxiv.org/html/2601.11580v2 | READ BODY (verified) |
| D13 | EDA parameter/data-efficient draft adaptation | **4× NVIDIA H200** (full retrain 5.1 h / 462 MB; EDA 2.0 h / 127 MB) | The 39.2%-of-time and 60.8%-of-cost claims are 4×H200 measurements | https://arxiv.org/abs/2603.09527 | READ BODY (delegated) |
| D14 | EAGLE3 on a W4A8 Kimi-K2.5 target (TPOT −35.9%) | **8× MI325X**, 497 GB W4A8 base | Target alone is ~497 GB even at INT4 | https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-speculative/README.html | READ BODY (verified) |
| D15 | SpecForge Qwen3.6-27B DFlash2 recipe | **2 GPUs** (one capture server, one trainer) | The checked-in recipe is explicitly two-GPU | https://raw.githubusercontent.com/sgl-project/SpecForge/main/docs/sections/basic_usage/training.md | READ BODY (verified) |
| D16 | Speculators DFlash / P-EAGLE quickstarts | **2 GPUs** (`--nproc_per_node 2`); P-EAGLE used 2 GPUs for extraction + 2 for training | Published commands are multi-GPU | https://vllm.ai/blog/2026-05-28-speculators-v050 | READ BODY (verified) |
| D17 | TokForge Qwen3-0.6B distiller | **2× RTX PRO 6000 Blackwell** (teacher GPU 0, student GPU 1) | Two-GPU teacher/student split | https://huggingface.co/darkmaniac7/TokForge-AccelerationPack-Draft | READ BODY (delegated) |
| D18 | EvoSpec online adaptation | **2× RTX 4090**, and separately a **5-GPU RTX PRO 6000 node** | Both configurations are multi-GPU | https://arxiv.org/abs/2605.27390 | READ BODY (delegated) |
| D19 | jukofyork DeepSeek-R1-DRAFT-0.6B-v3.0 | **6× RTX A6000 across 3 nodes**, ~2.3B tokens, 32k seq | Explicitly **multi-node** | https://huggingface.co/jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0 | READ BODY (delegated) |
| D20 | speculators DFLASH windowed-attention backward kernel | dense backward **~69 GB**, "OOM on 8×A800" | Per-GPU peak plus the 8-GPU reference rig | https://github.com/vllm-project/speculators/issues/1105 | READ BODY |
| D21 | Full-vocabulary draft training loss (pre-streaming-CE) | batch 16 × 32768 tokens × 128256 vocab × 4 B ≈ **250.5 GiB** of logits | Exceeds 96 GB by ~2.6×; the fix (Streaming Cross Entropy) is what makes it fit | https://nebius.com/blog/posts/training-speculative-decoders | READ BODY (delegated) |
| D22 | DFlash §5.3 serving results (Qwen3-4B/8B/Coder-30B, up to 5.1×, concurrency 1–32) | **single B200** (192 GB class, sm100) with FlashAttention-4 + Spec-v2 overlap | 1 GPU, but a different, larger-memory, different-arch part than sm120/96 GB; FA4 + Spec-v2 paths are not the sm120 path | https://arxiv.org/html/2602.06036v2 | READ BODY (delegated) |
| D23 | DFlash main experiments and ablations | **NVIDIA H200** (ablations: single H200; SGLang evals: single B200) | 1 GPU in each case, but H200/B200-class memory and kernels | https://arxiv.org/html/2602.06036v2 | READ BODY (delegated) |
| D24 | SFDD flatness-filtered draft training | **NVIDIA H800** GPUs | Reported training-speedup numbers are H800 | https://arxiv.org/abs/2601.18902 | READ BODY (delegated) |
| D25 | DeLS-Spec cost comparison (the *baseline* side) | Domino-FT **OOMs on a 48 GB L20** for Qwen3-8B | The baseline it beats is out of scope on 48 GB; the DeLS-Spec side (1.1 h / 9–10 GB) is **not** ruled out | https://arxiv.org/abs/2607.07409 | READ BODY (delegated) |
| D26 | Aurora continuous speculator learning | SGLang inference server **+** an asynchronous training server, hot-swapped | Two-server topology (paper + product blog) | https://arxiv.org/abs/2602.06932 | READ BODY (delegated) |

### D-note. Items explicitly NOT hardware-ruled-out on 1× RTX PRO 6000 96 GB / sm120

| # | Item | Evidence | URL | READ BODY / TITLE ONLY |
|---|---|---|---|---|
| N1 | vLLM standalone draft-model spec decode (Qwen3-32B + Qwen3-1.7B, `disable_padded_drafter_batch`) | The PR's own benchmark rig is **"an RTX PRO 6000 96GB"** — the same GPU class | https://github.com/vllm-project/vllm/pull/24322 | READ BODY (verified) |
| N2 | Speculators draft training | CLI: "Supports **single-GPU** and multi-GPU distributed training"; `--fsdp-shard` only if it does not fit | https://raw.githubusercontent.com/vllm-project/speculators/main/docs/cli/train.md | READ BODY (verified) |
| N3 | PARD-style parallel drafting via vLLM `draft_model` + `parallel_drafting` | Docs give a `-tp 1`, `--gpu-memory-utilization 0.8` single-GPU online command | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/parallel_draft_model.md | READ BODY (verified) |
| N4 | SGLang TLI cross-vocabulary drafts | Benchmarked on **single** H100 NVL 96 GB, A100 80 GB, and RTX A6000 48 GB | https://github.com/sgl-project/sglang/pull/22883 | READ BODY (verified) |
| N5 | llama.cpp FR-Spec-style MTP draft-vocab trimming | Measured on **a single RTX 5090 Laptop**, 128k ctx | https://github.com/ggml-org/llama.cpp/issues/25187 | READ BODY (verified) |
| N6 | Standalone 0.5B drafter training | "~2.5 h on a **single 24 GB GPU**, ~$3 of compute" on 30k R1-distilled traces | https://huggingface.co/vexp-ai/horizon-draft-0.5b | READ BODY (delegated) |
| N7 | Qwen3-4B EAGLE3 draft training (community) | 1 layer, hidden 2560, draft vocab 32000, bf16, SpecForge online run | https://huggingface.co/huluhuluu/Qwen3-4B-Instruct-2507-EAGLE3-ShareGPT-full-context-epoch1-step35000 | READ BODY (delegated) |
| N8 | DFlash2 GGUF draft for Qwen3.8-27B | **1.92 B params** bf16 (~3.8 GB) — draft side is small; the *target* is the memory question | https://huggingface.co/Akicou/Qwen3.8-27B-DFlash2-GGUF | READ BODY (delegated) |
| N9 | Q8 draft against a Q4 target | "Q8 DFlash2 works fine with a Q4 target — **the draft quant does not need to match the target**" | https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/discussions/9 | READ BODY (delegated) |

---

## Search log

Engines used: harness `web_search` (DeepSeek web search), arXiv abs/HTML pages, GitHub HTML pages and issue-search URLs, `api.github.com` (rate-limited), HuggingFace raw model cards / dataset cards / discussion JSON, engine docs sites, and direct `curl` (+`python3` HTML strip) or `web_fetch` for page bodies.

### A. `web_search` — queries run verbatim (my own pass)

| # | Query | Notes |
|---|---|---|
| 1 | `DFlash drafter speculative decoding` | found Inco/z-lab, SpecForge, AngelSpec |
| 2 | `DFlash2 speculative decoding draft model` | found HYHPING2023 checkpoint, Akicou GGUF |
| 3 | `speculative decoding draft model training 2026` | — |
| 4 | `SpecForge speculative decoding training library` | found sgl-project/SpecForge |
| 5 | `"Scaling Laws for Speculative Decoding" arXiv 2505.07858` | arXiv abs |
| 6 | `"Speculative Decoding: Performance or Illusion" MLSys 2026` | found MLSys slides PDF + HF papers |
| 7 | `DistillSpec knowledge distillation speculative decoding` | arXiv 2310.08461 |
| 8 | `online speculative decoding draft model training` | — |
| 9 | `draft model quantization speculative decoding` | found vLLM PR 13849, SGLang PR 23838, HF discussion |
| 10 | `speculative decoding draft model vocabulary mismatch` | found SGLang 23838 / 24051 |
| 11 | `EAGLE draft model training data curation` | — |
| 12 | `negative results draft model speculative decoding does not help` | found arXiv 2607.12422 |
| 13 | `"Speculative Decoding: Performance or Illusion" arxiv` | confirmed arXiv 2601.11580 |
| 14 | `SpecForge v0.3.0 draft models DFlash Domino DSpark` | blog mirrors |
| 15 | `speculators v0.5.0 vLLM blog draft model training` | blog.vllm.com.cn + .ai + raw md |
| 16 | `DFlash block diffusion speculative decoding paper arxiv` | Semantic Scholar record |
| 17 | `SGLang EAGLE draft model training documentation` | — |
| 18 | `vLLM speculative decoding draft model docs --speculative-config` | vLLM EAGLE / MLP docs |
| 19 | `"draft model" speculative decoding "future work" open problem 2026` | — |
| 20 | `"DFlash" block diffusion flash speculative decoding arxiv 2026` | arXiv 2602.06036 |
| 21 | `HuggingFace transformers PR 35029 speculative decoding vocabulary` | confirmed merged |
| 22 | `draft model quantization speculative decoding acceptance rate degradation` | found SGLang 36599, vLLM-Ascend 9834 |
| 23 | `speculative decoding draft model open problem "future work" 2026 arxiv` | found SpecBundle docs, DeLS-Spec, KVShot |
| 24 | `vLLM speculative decoding draft model issue "not planned"` | — |
| 25 | `SGLang SpecForge issue draft training problem` | — |
| 26 | `draft model training cost GPU hours speculative decoding 2026` | found PRISM, Horizon, jukofyork |
| 27 | `speculative decoding open questions 2026 draft model "remains an open"` | found multimodal SD survey, vLLM forum |
| 28 | `"Speculative Decoding: Performance or Illusion" future work research opportunities` | — |
| 29 | `vLLM forum standalone draft model spec decode support v1` | discuss.vllm.ai thread |
| 30 | `speculative decoding draft model training new research opportunities 2026` | — |
| 31 | `quantized EAGLE3 draft head W4A8 speculative decoding acceptance` | ROCm blog, FR-Spec, SpecQuant |
| 32 | `speculative decoding draft model quantization FP8 INT4 acceptance rate` | — |
| 33 | `"precision mismatch" speculative decoding quantized draft paper 2026` | HF dataset thaki-AI |
| 34 | `speculative decoding negative result draft model does not speed up` | vLLM #15025 |
| 35 | `speculative decoding draft model "we find that" does not improve 2026 negative` | Draft-OPD, MLA draft |
| 36 | `EAGLE draft model training data ShareGPT criticism limitation` | — |
| 37 | `draft model speculative decoding "stale" closed issue github 2026` | llama.cpp 25187, vLLM 45760 |
| 38 | `draft model quantization speculative decoding "does not" help acceptance` | SGLang 36599 |
| 39 | `speculative decoding draft model int4 quantization EAGLE draft quantized` | SentenceDiff, ginsongsong card |
| 40 | `quantized draft model speculative decoding vLLM issue accuracy loss` | vLLM #40149 |
| 41 | `self-distillation draft model speculative decoding 2026` | Draft-OPD, BudgetDraft, KnapSpec |
| 42 | `draft model fine-tuning cost GPU hours speculative decoding training budget` | NVIDIA DGX Spark forum |
| 43 | `"self-speculative" decoding draft model 2026 training` | — |
| 44 | `speculators library draft model training GPU requirements 8 GPU` | speculators docs/cli/train.md |
| 45 | `SpecForge draft model training hardware requirements GPU memory` | — |
| 46 | `"speculative decoding" draft model training "single GPU" 2026` | — |
| 47 | `speculative decoding draft model training "8xH100" OR "8xA100" cost 2026` | chutesai FP8 DFlash, dev.to DFlash drafter guide |
| 48 | `DFlash drafter training cost GPU hours z-lab` | — |
| 49 | `speculators library draft model training time H100 hours` | — |
| 50 | `speculative decoding draft model removed deprecated wontfix github issue` | (shared with sub-sweep C) |
| 51 | `vLLM remove standalone draft model speculative decoding v0.10 deprecated issue` | found RFC #18571 |
| 52 | `"speculative decoding" vllm "no longer supported" draft model removal discussion` | — |
| 53 | `Qwen3-0.6B draft Qwen3-4B vLLM speculative config benchmark` | found vLLM PR 24322 |
| 54 | `vLLM speculative decoding draft model "8xH100" cost` | — |
| 55 | `speculative decoding draft model open questions draft training 2026 arxiv` | — |

### B. `web_search` — queries run verbatim (delegated sub-sweeps; 4 sweeps, 36 + ~52 + 6 + 52 queries)

Recorded in full in the sub-sweep files: `/Users/leihenan/Desktop/myProject/pa_out/C_abandoned.md`, `DF_dflash_arch.md`, `E_data_cost.md`, `spec-decode-feature-interactions-sweep.md`. Representative examples: `speculative decoding draft model removed deprecated wontfix github issue`; `vLLM remove MLP speculator speculative decoding v1 draft model deprecated`; `llama.cpp speculative decoding draft model "not planned" issue closed`; `arXiv 2026 speculative decoding draft model "does not help" limitation`; `draft model distillation speculative decoding negative result acceptance rate low`; `DFlash speculative decoding paper`; `Domino speculative decoding draft model`; `DSpark semi-autoregressive speculative decoding DeepSeek arXiv`; `"how many layers" draft model speculative decoding optimal depth study`; `speculative decoding draft model training data curation 2026`; `speculators library vLLM draft model training cost GPU hours`.

### C. GitHub issue-search URLs used verbatim

```
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+speculative+draft+wontfix
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+label%3A%22not+planned%22+speculative
https://github.com/sgl-project/sglang/issues?q=is%3Aissue+is%3Aclosed+label%3A%22not+planned%22+draft
https://github.com/sgl-project/SpecForge/issues?q=is%3Aissue+draft
https://github.com/vllm-project/speculators/issues?q=is%3Aissue
https://github.com/ggml-org/llama.cpp/issues?q=speculative+draft+closed
https://github.com/ollama/ollama/issues?q=speculative+draft
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+medusa+speculator
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+MLP+speculator
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+%22draft+model%22+in%3Atitle
https://github.com/sgl-project/sglang/issues?q=is%3Aissue+%22draft+model%22+in%3Atitle
https://github.com/sgl-project/sglang/issues?q=is%3Aissue+speculative+deprecat
https://github.com/huggingface/transformers/issues?q=is%3Aissue+assisted+generation+draft
https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+label%3Astale+speculative
https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+spec+decode
https://github.com/sgl-project/sglang/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+draft
https://github.com/vllm-project/speculators/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged
https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged+speculative
```

API-form queries (`api.github.com/search/issues`, `reason:` qualifier) — verbatim:
`repo:vllm-project/vllm is:issue reason:not_planned speculative` · `... "draft model"` · `... medusa` · `... "mlp speculator"` · `... lookahead` · `repo:sgl-project/SpecForge is:issue reason:not_planned` · `repo:vllm-project/speculators is:issue reason:not_planned` · `repo:sgl-project/sglang is:issue reason:not_planned draft` · `repo:ggml-org/llama.cpp is:issue reason:not_planned draft` · `repo:ollama/ollama is:issue reason:not_planned speculative OR draft` · `repo:huggingface/transformers is:issue reason:not_planned assistant model speculative` · `repo:NVIDIA/TensorRT-LLM is:issue reason:not_planned draft speculative` · `repo:huggingface/text-generation-inference is:issue reason:not_planned speculative` · `repo:ml-explore/mlx-lm is:issue reason:not_planned draft speculative` (0 results) · `repo:vllm-project/vllm is:pr fr-spec` · `repo:vllm-project/vllm is:pr is:unmerged "draft model"`

### D. arXiv listings / direct arXiv pages consulted

`https://arxiv.org/abs/2505.07858` · `/abs/2601.11580` + `/html/2601.11580v2` · `/abs/2602.06036` + `/html/2602.06036v2` · `/abs/2602.01469` + `/html/2602.01469v1` · `/abs/2605.29707` + `/html/2605.29707v1` · `/abs/2607.05147` + `/html/2607.05147v1` · `/abs/2605.09992` · `/abs/2503.01840` · `/abs/2603.11053` · `/abs/2608.20743` · `/abs/2606.00144` · `/abs/2602.13836` · `/abs/2605.29343` · `/abs/2607.27269` · `/abs/2607.12422` · `/abs/2604.26412` · `/abs/2602.01762` · `/abs/2603.09527` · `/abs/2603.18567` · `/abs/2605.07243` · `/abs/2605.18810` · `/abs/2605.27390` · `/abs/2606.02091` · `/abs/2607.07409` · `/abs/2607.19223` · `/abs/2607.24434` · `/abs/2602.06932` · `/abs/2602.06019` · `/abs/2509.18362` · `/abs/2504.18583` · `/abs/2310.08461` · `/abs/2310.07177` · `/abs/2310.15141` · `/abs/2410.06916` · `/abs/2502.14856` · `/abs/2502.05202` · `/abs/2601.18902`

### E. Docs / blogs / model cards fetched and read

vLLM docs (`features/speculative_decoding/eagle`, `/mlp`, `/parallel_draft_model.md`, `/README.md`, `speculators` site) · SGLang / SpecForge docs (`docs/sections/basic_usage/training.md`, `docs/sections/concepts/DFlash2.md`, `docs/basic_usage/data_preparation.md`) · Speculators docs (`docs/cli/train.md`) · LMSYS blogs (SpecForge v0.3 2026-08-04; Ling-3.0-flash Blackwell 2026-08-21) · vLLM blog (speculators v0.5.0 2026-05-28; EAGLE 3.1 2026-05-26; speculators v0.3.0 2025-12-13) · Inco AI DFlash2 blog 2026-08-18 · ROCm blog (Kimi-K2.5 W4A8 + EAGLE3) · Nebius blog (training speculative decoders; SlimSpec) · Together AI Aurora blog · NVIDIA DGX Spark forum thread 378611 · HF model cards: `z-lab/Qwen3.8-27B-DFlash2` (+config.json, discussions 3 & 9), `HYHPING2023/checkpoint-draft-dflash2`, `Akicou/Qwen3.8-27B-DFlash2-GGUF`, `RadixArk/Qwen3.8-27B-DSpark`, `asherszhang/MiniMax-M2.7-EAGLE3-draft-vocab32k`, `vexp-ai/horizon-draft-0.5b`, `darkmaniac7/TokForge-AccelerationPack-Draft`, `alamios/Mistral-Small-3.1-DRAFT-0.5B`, `jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0(-GGUF)`, `huluhuluu/Qwen3-4B-Instruct-2507-EAGLE3-ShareGPT-*`, `amd/PARD-Qwen3-0.6B`, `RedHatAI/Qwen3-8B-speculator.peagle`, `RedHatAI/gpt-oss-120b-speculator.eagle3`, `AQ-MedAI/GLM-5.1-eagle3` (discussion 3), `deepseek-ai/DeepSeek-R1` (discussion 174), `weijiezz/SpecBlock-train-data-qwen`, `Doubleword/qwen3.6-specdec-calibration`, `thaki-AI/daily-paper-2026-07-25-spec-decode-quant-precision-mismatch`, `thaki-AI/daily-paper-2026-08-05-traffic-regime-serving-breakeven` · third-party deployment guides (`AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash/docs/dflash.md`, `JamePeng/llama-cpp-python` wiki) · Tencent AngelSpec docs (`concepts/dflash.md`, `dspark.md`, `dflare.md`, `dfly.md`, `mtp.md`, `eagle3.md`).

### F. Engines that returned nothing usable (recorded so they are not re-run blindly)

- `export.arxiv.org/api/query` — HTTP 429 "Too Many Requests" on **every** query attempted from this sandbox (5 distinct queries). No results obtained.
- `arxiv.org/search/` HTML scrape — returned **zero** parseable entries for 4 distinct queries.
- `api.github.com/repos/.../git/trees/main?recursive=1` for the paper-notes index — returned an empty tree (`total 0`).
- `github.com/<repo>/tree/...` directory-listing HTML — repeatedly timed out (60 s harness cap) and is unreliable; individual blob/raw URLs work.
- `raw.githubusercontent.com/.../refs/heads/main/_posts/2026-05-28-speculators-v050.md` — returns HTTP 000/empty; the working path is `.../main/_posts/...` (no `refs/heads/`).
- Direct `curl` to `huggingface.co` — HTTP 000 from bash for several cards; `web_fetch` succeeded on the same URLs.
- Search-result negatives: literal `wontfix` on vLLM — **no results**. The `not planned` *label* on vLLM and SGLang — **no results**. `repo:ml-explore/mlx-lm is:issue reason:not_planned draft speculative` — **0 results**. `repo:huggingface/transformers is:issue reason:not_planned "assistant model"` — **0 results**. `repo:sgl-project/sglang is:issue is:closed label:stale draft OR speculative` — **0 results**. "SpecTrain" — **no matching paper or artifact found by any query**; "SpecTr" resolves to arXiv 2310.15141 (optimal-transport draft selection, NeurIPS 2023, training-free), not a training method.

---

## Appendix — DRAFT-MODEL QUANTIZATION, REDUCED DRAFT VOCAB, AND PRECISION MISMATCH (consolidated; rows also folded into A/B/C above)

| # | Finding | Who | Artifact | URL | Status | READ BODY / TITLE ONLY |
|---|---|---|---|---|---|---|
| Q1 | **Quantizing the draft does not require matching the target's quant.** Q8 DFlash2 draft against a Q4_K target runs faster than both no-spec and MTP | ravingamm (community) | HF discussion `z-lab/Qwen3.8-27B-DFlash2` #9; RTX 3090, no-spec 33.16 → MTP2 48.32 → DFlash2 Q4 56.06 → DFlash2 Q8 **59.88** t/s | https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/discussions/9 | reported working | READ BODY (delegated) |
| Q2 | **A quantized target does not break an EAGLE3 draft.** Kimi-K2.5 W4A8 + EAGLE3: TPOT 42.73 → 27.41 ms (−35.9%), throughput 672 → 895 tok/s (+33.1%), **"with no measurable accuracy regression"** | AMD ROCm / AITER / FlyDSL | ROCm blog, 8× MI325X, concurrency 40 | https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-speculative/README.html | shipped (recipe + Docker image) | READ BODY (verified) |
| Q3 | DFlash + **FP8-quantized verifier** compounds | Red Hat AI | Speculators v0.5.0 blog: "Combining DFlash with an FP8 quantized verifier yields even greater gains" | https://vllm.ai/blog/2026-05-28-speculators-v050 | shipped | READ BODY (verified) |
| Q4 | **Quantized MTP/NextN drafts break in SGLang** when the target is `modelopt_fp4` — `quant_config` is hardcoded to `None`; `--speculative-draft-model-quantization` becomes a no-op | chuck-ads / SGLang | SGLang issue #36599; error `FusedMoE._load_w13: The size of tensor a (4096) must match the size of tensor b (2048)`; reporter's one-line fix gives 14.2 → 20.6 tok/s | https://github.com/sgl-project/sglang/issues/36599 | **Open** | READ BODY (verified) |
| Q5 | **FP8/INT8 were explicitly not supported for the EAGLE draft** in TensorRT-LLM one-model config (2026) | ValeGian / TensorRT-LLM | Issue #7842 | https://github.com/NVIDIA/TensorRT-LLM/issues/7842 | **not planned** | READ BODY |
| Q6 | **Reduced draft vocab (`draft_vocab_size`) + unconditional lm_head sharing = broken DFlash drafts** | Liuchenbing-2026 / vLLM-Ascend | vLLM-Ascend PR #9834: `_maybe_share_lm_head` "unconditionally replaced the draft model's lm_head with the target model's lm_head whenever method was in `("eagle", "dflash")`" — broke `RedHatAI/Qwen3-8B-speculator.dflash` with `draft_vocab_size=32000` | https://github.com/vllm-project/vllm-ascend/pull/9834 | fixed | READ BODY (delegated) |
| Q7 | **Reduced draft vocab is a supported training flag** with frequency-based token selection and explicit t2d/d2t maps | vLLM/Red Hat | Speculators: `--draft-vocab-size`, `--token-freq-path`, `--t2d-path`, `--d2t-path` | https://raw.githubusercontent.com/vllm-project/speculators/main/docs/cli/train.md | shipped | READ BODY (verified) |
| Q8 | **Vocab-pruned EAGLE3 draft in the wild**: lm_head pruned to top-32,000 tokens (~99.1% coverage), trimmed `[32000, 3072]` head is 6.25× smaller; carries `t2d`/`d2t` maps | asherszhang | HF card `MiniMax-M2.7-EAGLE3-draft-vocab32k` (target vocab 200064, draft vocab 32000); HumanEval mean accept length 2.59 | https://huggingface.co/asherszhang/MiniMax-M2.7-EAGLE3-draft-vocab32k | shipped (weights) | READ BODY (delegated) |
| Q9 | **Draft LM-head is the dominant draft-side cost at large vocab** — trimming to 32,768 hot tokens cut the draft LM-head kernel from 407.1 ms/655 calls to 61.6 ms/675 calls (**−84.9%**), 621 µs → 91 µs per call; end-to-end 83.9 → 85.1 tok/s (public map) / 86.5 (code-weighted) | avifenesh / llama.cpp | Issue #25187; single RTX 5090 Laptop, Qwen3.6-27B-NVFP4 trunk + Q6_K MTP draft | https://github.com/ggml-org/llama.cpp/issues/25187 | **Open research issue** | READ BODY (verified) |
| Q10 | **Draft-vocabulary pruning was rejected in vLLM on measured grounds** — reviewer measured 1.52% throughput gain and argued the drafter is only 4% of target forward time | keyboardAnt / eitanturok / vLLM | PR #24343 (105 commits) + PR #29334, both closed-unmerged; issue #24506 closed not-planned | https://github.com/vllm-project/vllm/pull/24343 | **closed-unmerged** | READ BODY |
| Q11 | **FR-Spec's acceptance-rate table** (author-run, logits-processor level): vanilla 27.8%; ratio 0.1→28.3%, 0.25→28.6%, 0.5→28.6%, 0.75→27.2%, 0.9→25.9%, 0.99→18.8% | jmamou / vLLM | Issue #24506 | https://github.com/vllm-project/vllm/issues/24506 | closed not-planned | READ BODY |
| Q12 | **Vocabulary speculation instead of vocabulary reduction** — SpecVocab picks a vocabulary subset per decoding step; "up to an 8.1% increase in average throughput over EAGLE-3" | Miles Williams et al. | arXiv 2602.13836, Findings of ACL 2026 | https://arxiv.org/abs/2602.13836 | published | READ BODY (verified) |
| Q13 | Canonical reduced-draft-vocab reference (SGLang ships it as `--speculative-token-map`) | Zhao, Pan, Han et al. | FR-Spec, ACL 2025, arXiv 2502.14856 | https://arxiv.org/abs/2502.14856 | published | TITLE ONLY |
| Q14 | **4-bit is claimed to be the sweet spot for drafts** — "experimentation has shown using more or less than 4-bits for speculative decoding is a waste of time anwyay"; also blocked by head count: "The 14 heads of `Qwen2.5-0.5B` doesn't allow for any of the other 4-bit quants to be made" | jukofyork | `jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0-GGUF` | https://huggingface.co/jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0-GGUF/blob/main/README.md | shipped (weights) | READ BODY (delegated) |
| Q15 | **Quantizing the draft is an accepted way to build a verification companion**: "Companion models can be obtained by: fine-tuning the draft model, **quantizing the draft model**, or directly selecting public models of similar scale" | Speculative Verification authors | arXiv 2509.24328, ACL 2026 Findings | https://arxiv.org/abs/2509.24328 | published | READ BODY (delegated) |
| Q16 | **Low-rank LM head instead of vocab truncation** — SlimSpec: "4-5× acceleration over the standard LM-head architecture while maintaining a competitive acceptance length", vs vocab truncation which "reduce[s] LM-head latency by only about 60%" and caps acceptance | Nebius | arXiv 2605.10453 | https://arxiv.org/abs/2605.10453 | published | READ BODY (delegated) |
| Q17 | **MLA conversion silently destroys draft acceptance** unless functionally reconstructed — "direct MHA/GQA-to-MLA conversion can sharply reduce this agreement… may be tolerable for standalone generation but substantially lower draft-token acceptance"; functional reconstruction improves acceptance in 37/64 matched cells | Weiye Shi, Fanxu Meng, Muhan Zhang | arXiv 2607.27269 | https://arxiv.org/abs/2607.27269 | published | READ BODY (verified) |
| Q18 | **Near-identical target/draft vocab sizes disable speculation entirely** (Qwen2.5 padding to multiples of 128 vs 256); fix PR closed-unmerged | benchislett / vLLM | PR #13849 | https://github.com/vllm-project/vllm/pull/13849 | **closed-unmerged** | READ BODY (verified) |

---

## Method / provenance note on the quantization rows

The dedicated quantization sub-sweep did not return within the session budget and was stopped. **Every row in the Appendix above (Q1–Q18) was retrieved and read directly in this session or by one of the three sub-sweeps that did report**, and each carries its own URL. No quantization claim in this document rests on a search-result snippet alone except where explicitly marked `TITLE ONLY` (Q13). Remaining un-retrieved leads on this sub-topic, listed for a follow-up pass and **not** used as evidence here: `SpecQuant – Speculative Decoding with Multi-Parent Quantization for Adaptive LLM Inference` (IEEE Xplore document 11646348) — TITLE ONLY; `thaki-AI/daily-paper-2026-07-25-spec-decode-quant-precision-mismatch` (main.tex fetch timed out) — TITLE ONLY; `ginsongsong/eagle3-kimik2.5-w4a8` HF card (curl failed, card not read) — TITLE ONLY.
