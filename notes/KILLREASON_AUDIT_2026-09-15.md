# KILL-REASON AUDIT — do the three maps close cells for reasons that are factually false **on this rig**?

**Date:** 2026-09-15 · **Auditor:** delegated subagent (parent `session-1efb6dac`) · **Cost:** 0 GPU·h
**Question asked:** *"the filter itself is broken"* — are hardware/architecture kill reasons factually wrong on the target machine?

**Rig under test (all facts below re-confirmed this session, `RUN`):**

```
$ ssh -p 26924 root@connect.weste.seetacloud.com 'nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv'
NVIDIA RTX 6000D, 85651 MiB, 12.0, 595.71.05
$ source /root/ccfa_env.sh && python -c "import vllm,torch;print(vllm.__version__, torch.__version__)"
0.29.0 2.13.0+cu130
```

Also confirmed by direct predicate call: `current_platform.get_device_capability() = DeviceCapability(major=12, minor=0)`,
`is_device_capability_family(120) = True`, `is_device_capability_family(100) = False`, **SGLang not installed** (`NEW_SERVER_ENV` §1,
not re-checked by me this session). Host path for every `file:line` below is
`<V> = /root/ccfa_venv/lib/python3.12/site-packages/vllm/` — the **installed** tree, which I read with `sed -n`/`grep -n` myself.

**Method.** Zero GPU, zero inference, zero benchmark. Only: source reads, `nm -D`, `--help`-style predicates, and Python
predicate calls that do not initialise CUDA. Every absence claim is phrased "I did not find it".

---

## §1 Scope — how many hardware/architecture kill reasons exist, and how many I could check

**How I enumerated them.** I grepped all three maps for hardware-shaped reason predicates
(`not supported on` / `does not support` / `only … SM100|SM90|Blackwell|Hopper` / `requires compute capability` /
`gated` / `gated out` / `falls? back to` / `hard-?gated|hard-?disabled`) and then read each hit in context.
The maps' own D-section counts are the authoritative denominator:

| Map | D-section / hardware verdict bucket | Items | In scope for this audit |
|---|---|---|---|
| `INFERENCE_ACCEL_GAP_MAP.md:13` | `130 HARDWARE-RULED-OUT` | 130 | D.1 >1 GPU (61), D.2 >96 GB (14), D.3 multi-node/NVLink (23), D.4 storage fabric (10) = **108 out of scope** (GPU-*count* / fabric kills, which the brief excludes). D.5 borderline (22) — I did not find a device-capability claim in the D.5 rows I read (`D109`–`D123`), so **0** in scope. |
| `KV_CACHE_GAP_MAP.md:10` | `94 HARDWARE-RULED-OUT` | 94 | D.1 >1 GPU (47), D.2 >96 GB (2), D.3 multi-node (26), D.4 storage (12) = **87 out of scope**. D.5 borderline (7) = **0** in scope. |
| `spec-decode-gap-map-2026-09-13.md` | `## D. What is ruled out BY HARDWARE` — D1 kernel/family gaps on sm120 (14), D2 parallelism (16), D3 training cost (16), D4 version traps (7), D5 single-GPU datapoint (3) | ~45 (not head-counted by that map) | **D1 = 14 in scope** (these are *device-capability* kills on sm120); D2/D3/D4/D5 = 31 out of scope. |

**In-scope population:**

> **≈ 28 hardware/architecture kill reasons are of the "sm120 lacks X / feature Y is gated / falls back to Z" form.**
> (180 device-*count*/VRAM/fabric/storage/training-scale kills were excluded as out-of-scope by the brief.)

**How many I could actually check first-hand on the rig: 11.** Those 11 are the ones whose deciding evidence is a
capability predicate, a shipped-tree source gate, or a binary symbol table — all readable with zero GPU. The other ~17 are
gated on artifacts **not present on this rig** (SGLang — row D1.9, D1.10, D1.11 all cite `sgl-project/sglang`; quantized /
MoE / MLA / hybrid model checkpoints; the optional `b12x` package; or a Blackwell datacentre card), i.e. they are
model-family or engine-absence kills rather than verifiable hardware claims for this box.

**Checked reasons, by verdict:**

| Verdict | Count | Cells |
|---|---|---|
| `HOLDS` | 6 | INF-A68 (D1a), INF-C137 (D1b), INF-C138 / INF-C139 / Spec-D1.13 (D1d), Spec-D2.6 (D1a) |
| `FALSE-ON-THIS-RIG` | **4** | Spec-D1.8 (D1a), Spec-D1.4 (D1b, **scope defect**, not a support defect), Spec-D1.5 (D1e), INF-E18/E19 (D1c) |
| `UNVERIFIABLE` | 1 (+1 mixture) | INF-C99 / C5 (D1c) — vendored split-KV |
| Total | **11 (one is a mixture: reason HOLDS, escalation defect)** | |

---

## §2 Table — every checked reason, verbatim, with the deciding evidence

Convention: **`<V>` = `/root/ccfa_venv/lib/python3.12/site-packages/vllm/`** on the rig.
"Map reason (verbatim)" is quoted from the map file at the cited line.

### D1a — device-capability predicates and decode-kernel reachability

| cell id | map's reason (verbatim) | verdict | deciding evidence | if FALSE: surviving content + §0.6 arithmetic |
|---|---|---|---|---|
| **Spec-D1.8** (`spec-decode-gap-map-2026-09-13.md:651`) | *"**trtllm-gen decode kernel gated behind `is_sm100_supported()`**"* — evidence *"the working trtllm-gen decode kernel is gated behind is_sm100_supported(), even though it imports and runs fine on sm_120"* (SGLang #36701) | **FALSE-ON-THIS-RIG** | (i) **SGLang is not installed here** — the quoted gate is SGLang's, and the brief's rig has no SGLang. (ii) In the engine that *is* installed, `is_sm100_supported` does not exist: `RUN` → `WARNING … Current platform cuda does not have 'is_sm100_supported' attribute.` (iii) The real gate explicitly **includes** sm12x: `<V>/utils/flashinfer.py:482-486` — `# SM90 and SM12x have XQA decode only.` / `if current_platform.is_device_capability(90) or current_platform.is_device_capability_family(120): return not is_prefill`. (iv) Predicate call `supports_trtllm_attention(is_prefill=False) = True`, `…(is_prefill=True) = False`. | Surviving content: the TRT-LLM/XQA **decode** path is reachable on sm120 in this engine; the map's own "corrected" half (prefill still gated) is the part that holds. Magnitude of the *prefill* half: TRT-LLM prefill would replace FA2 prefill. §0.6 upper bound — attention read at 4K is the only component a decode kernel can touch: `2 × 36 layers × 8 kv_heads × 4096 × 128 × 2 B = 3.02e8 B`; at the rig's assumed `1.52 TB/s` effective (`notes/POOL1_ENGINE_UNLOCKED_2026-09-15.md` §2.0) → `0.199 ms/step`; against a `7–12 ms` step → **1.7–2.8%** — sitting *on* the 2% line, and that is the *unreachable* upper bound (it assumes attention reads go to zero). The reachable gain is the delta FA2-paged-decode → XQA at bs=1, strictly smaller. **Does not clear the gate.** |
| **INF-A68** (`INFERENCE_ACCEL_GAP_MAP.md:125`) | *"Marlin built for 12.0; **Machete is Hopper-only by construction**"* | **HOLDS** | Not contradicted by anything I found: `MACHETE_ARCHS` is a build-time list, and I did not find an sm120 Machete entry. **I did not read `CMakeLists.txt` myself this session** — this row is accepted on the map's own `READ BODY` and on the fact that nothing in my probes depends on it. | — (bf16 dense Qwen3-4B never reaches Machete) |
| **INF-C137** (`INFERENCE_ACCEL_GAP_MAP.md:955`) | *"return (fa_version == 3 and current_platform.is_device_capability_family(90)) or ( fa_version == 4 and current_platform.is_device_capability_family(100) )" — … on sm120 both terms are False* | **HOLDS** | Read verbatim in the installed tree, `<V>/v1/attention/backends/fa_utils.py:306-308`: `return (fa_version == 3 and current_platform.is_device_capability_family(90)) or (` / `fa_version == 4 and current_platform.is_device_capability_family(100)` / `)`. And `RUN`: `get_flash_attn_version()` → `2`, `is_device_capability_family(90)` → `False`, `is_device_capability_family(100)` → `False`. | — |
| **INF-C138** (`INFERENCE_ACCEL_GAP_MAP.md:956`) | *"if not current_platform.is_device_capability_family(90): return False, \"FlashMLA Dense is only supported on Hopper devices.\""* / *"\"FlashMLA Sparse is only supported on Hopper and Blackwell DC devices.\""* | **HOLDS** | `<V>/v1/attention/ops/flashmla.py:57-58` (dense: `if not current_platform.is_device_capability_family(90):` → `return False, "FlashMLA Dense is only supported on Hopper devices."`) and `:76` (sparse: the string `"FlashMLA Sparse is only supported on Hopper and Blackwell DC devices."`). `RUN`: `is_device_capability_family(90)=False`, `is_device_capability_family(100)=False`, `is_device_capability_family(120)=True` ⇒ neither predicate can pass on 12.0. | — (Qwen3-4B is dense full attention, non-MLA) |
| **INF-C139** (`INFERENCE_ACCEL_GAP_MAP.md:957`) | *"\"indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs \""* / *"\"(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and \""* / *"\"earlier architectures are not supported.\""* | **HOLDS** | `<V>/v1/attention/backends/mla/indexer.py:57-61`: `if use_fp4 and not current_platform.is_device_capability_family(100):` then `raise ValueError("indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs " "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and " "earlier architectures are not supported.")`. All three strings verbatim in the installed tree. | — (MLA-only path) |
| **Spec-D1.13** (`spec-decode-gap-map-2026-09-13.md:656`) | *"NSA on SM120 declared an architectural dead end … \"This is not a bug — it is an architectural incompatibility.\""* | **HOLDS on this engine** | The quoted operator tested **SGLang**, which is not installed here, so I cannot test their mechanism. What I *can* test is the equivalent vLLM route: sparse non-MLA attention is not in the sm120 priority list (`notes/POOL1_ENGINE_UNLOCKED_2026-09-15.md` §1, C233, from `RUN`), and the MLA-sparse backends that do exist are MLA-gated. **I did not re-read `platforms/cuda.py` myself this session.** ⇒ I record this as **HOLDS by two independent routes, one of which (the vLLM backend table) I did not personally re-verify.** | — (Qwen3-4B has no sparse/NSA path) |

### D1b — "feature requires FA3/FA4" and the FlashAttention backend

| cell id | map's reason (verbatim) | verdict | deciding evidence | if FALSE: surviving content + §0.6 arithmetic |
|---|---|---|---|---|
| **Spec-D1.4** (`spec-decode-gap-map-2026-09-13.md:647`) | *"**FP8 KV as anything but storage**"* — *"\"Neither FA3 nor the trtllm-gen FMHA builds for those arches\""*; *"DFlash2 requires \"bf16 KV + FLASH_ATTN is mandatory\" (fp8 KV forces FlashInfer, ~40% slower)"* | **MIXTURE — quoted clause HOLDS; the heading over-claims (scope defect, not a support defect)** | **RUN, four independent lines:** (i) `FlashAttentionBackend.supports_kv_cache_dtype("fp8_e4m3") = False`, `("fp8") = False`, `("fp8_e5m2") = False` — the FA path really is closed, exactly as the clause says. (ii) `<V>/v1/attention/backends/flash_attn.py:228` returns the literal string `"FP8 KV cache requires FA3 on SM90 or FA4 on SM100"`, decided by `flash_attn_supports_kv_cache_dtype` (`<V>/v1/attention/backends/fa_utils.py:306-308`), which is `False` on 12.0. The engine then hard-raises: `<V>/v1/attention/backends/flash_attn.py:915-917` — `raise NotImplementedError(f"FlashAttention does not support {self.kv_cache_dtype}" " kv-cache on this device.")`. (iii) The **device** is *not* FP8-KV-incapable: `FlashInferBackend.supports_kv_cache_dtype("fp8_e4m3") = True` (`RUN`). (iv) The binary confirms the FA path has no fp8 KV route at all: `nm -D --defined-only <V>/vllm_flash_attn/_vllm_fa2_C.abi3.so \| grep -ci fp8` → **0**. | The heading "as anything but storage" is false as stated for this rig: **fp8 KV is usable, but only by leaving the FlashAttention backend** — and the backend you leave is the one this rig actually runs (see the log evidence below). Surviving content = the *forced-backend choice*, not an unavailable feature. §0.6: the choice is FlashAttention-vs-FlashInfer, and I **cannot bound the FlashInfer XQA penalty from source** — it is not a per-layer launch count I can multiply. **No magnitude claimed.** |
| **INF-A65 / INF-A46** (`INFERENCE_ACCEL_GAP_MAP.md:122`, `:103`) | A65: *"shipped (v0.29.0; SM120 FP8 CUTLASS path exists and dispatches first)"*; A46: *"opt-in; `VLLM_TRITON_USE_TD` unset auto-selects only on XPU"* | **HOLDS (signal in the right direction)** | The first-hand artifact check agrees with A65 rather than contradicting it: `<V>/_C_stable_libtorch.abi3.so` contains the architecture markers `sm_120` (8) and `sm_120f` (18) among its embedded cubin/PTX strings (`strings -a \| grep -Eo 'sm_[0-9]+[a-z]?'`), and the tree ships **60** files whose path contains `sm120` (`find <V> -name '*sm120*' \| wc -l`). **This is a lower bound on what is compiled, not proof that a specific kernel dispatches** — I did not run any GEMM. | — |

### D1c — kernels whose gate semantics differ between "family" and "≥ capability"

| cell id | map's reason (verbatim) | verdict | deciding evidence | if FALSE: surviving content + §0.6 arithmetic |
|---|---|---|---|---|
| **INF-E18 / INF-E19** (`INFERENCE_ACCEL_GAP_MAP.md:1562`, `:1573`) | E18: *"MXFP8 dense linear on SM120/SM121 through B12X silently drops to whatever"*; E19: *"The native B12X FP4 dense GEMMs (both MXFP4 and NVFP4) are gated behind a single"* | **FALSE-ON-THIS-RIG as a *hardware* claim** | `<V>/model_executor/kernels/linear/mxfp8/b12x.py:56` and `<V>/model_executor/kernels/linear/mxfp4/b12x.py:59` both read `if not current_platform.is_device_capability_family(120):` — i.e. the hardware gate **admits sm120 explicitly**. The actual blocker is a **software/optional-dependency** one: `RUN` → `import b12x` raises `ModuleNotFoundError: No module named 'b12x'`, and the kernel's own reason string is `(False, 'Install the B12X backend with \`pip install vllm[b12x]\`')` for **both** `B12xMxfp8LinearKernel.is_supported()` and `B12xMxFp4LinearKernel.is_supported()`. | Surviving content: on this rig these kernels are **one `pip install` away, not one hardware generation away** — a materially different "gap", because it is closable at zero hardware cost. §0.6: **magnitude not boundable and not needed** — the model is BF16 dense (no MXFP8/MXFP4 checkpoint on the rig), so the recoverable amount on *this* model is **0**. |
| **INF-A68-adjacent — "C99 / C5: sm120 lacks split-KV"** (`INFERENCE_ACCEL_GAP_MAP.md:64`; escalation at `notes/POOL1_ENGINE_UNLOCKED_2026-09-15.md` §2.2) | C99 verbatim: *"partly — harness drift is real; the `assert num_splits == 1` gate this fills is now also targeted by open PR #2758."* | **UNVERIFIABLE from zero-GPU evidence — my verdict differs from the escalation's** | What I confirmed first-hand: the vendored FA2 binary **does** carry the split path — `nm -D --defined-only <V>/vllm_flash_attn/_vllm_fa2_C.abi3.so \| grep -c run_flash_splitkv_fwd` → **512**, total `grep -ci split` → **561**, plus `flash::num_splits_heuristic`. **But symbol presence is not reachability on the executed path**, and the executed path is Python-gated: `<V>/vllm_flash_attn/flash_attn_interface.py:311-312` — `if num_splits > 1: raise NotImplementedError("FA2 does not support num_splits > 1")`. | **What this means for the escalation.** The claim "the split path is reachable on Qwen3-4B's exact 32:8 GQA decode shape" rests on the *upstream* `flash_api.cpp` GQA-swap branch. The **installed wrapper short-circuits before that C++ is reached** whenever Python passes `num_splits > 1`. Whether Python passes `>1` here depends on `FlashAttentionMetadataBuilder.build` (`<V>/v1/attention/backends/flash_attn.py:596-600`): `max_num_splits` is set only when `self.use_full_cuda_graph` is true, and is `1` when `VLLM_BATCH_INVARIANT`. Meanwhile the rig's own log **proves** FULL CUDA graphs run with FA2 on sm120: `/root/ccfa_results/2026-09-12/smoke/01-nospec.serve.log:46` — `Capturing CUDA graphs (FULL): 100%\|██████████\| 35/35`, and `:11` — `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>`, `'pass_config': {… 'enable_sp': False …}`, and `:16` — `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']`. That run did not raise, which is **inconsistent with the simplest reading** of `build()` and means I do not understand which of the two `num_splits` branches fires in the captured path. **I did not run the model, so I cannot tell which of these is true, and I do not claim either.** Honest statement: *the vendored FA2 **contains** split-KV (RUN); whether vLLM's FA2 path on sm120 **executes** it is undetermined by anything I read or ran.* |

### D1d — "restricted to Blackwell datacentre / FP4"

| cell id | map's reason (verbatim) | verdict | deciding evidence | if FALSE: surviving content |
|---|---|---|---|---|
| **INF-E14/E17/E19-adjacent family** — the maps' recurring `is_device_capability_family(100)` framing (e.g. `INFERENCE_ACCEL_GAP_MAP.md:125-127`, `:955`, `:958`) | Representative verbatim (`:957`): *"\"indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs \""*; and the C141 table quoted at `:961`: *"`current_platform.is_device_capability_family(100)` \| **rejected** — family is 120"* | **HOLDS as a *family* claim — but the maps' neighbouring `has_device_capability(100)` wording is a trap** | `RUN`: `is_device_capability_family(100) = False` on 12.0 (so the family-gated kernels really are rejected), **but `has_device_capability(100) = True` on the same device**. These two predicates give *opposite* answers on sm120. Any kill reason phrased as "requires compute capability 100 / `has_device_capability(100)`" is therefore **not** a valid sm120 exclusion — it passes on this device. | No surviving content claimed: I only use this to classify *wording*. I did **not** find a specific map cell whose kill rests on `has_device_capability(100)` alone, so I am **not** calling any additional cell FALSE. Recorded because it is the exact mechanism that has bitten this project three times. |

### D1e — build-level / kernel-family availability

| cell id | map's reason (verbatim) | verdict | deciding evidence | if FALSE: surviving content |
|---|---|---|---|---|
| **Spec-D1.5** (`spec-decode-gap-map-2026-09-13.md:648`) | *"**W8A8 kernels** — reported \"does not build for sm ≥ 10.0\""* | **FALSE-ON-THIS-RIG** | Three independent first-hand checks: (i) `<V>/_C_stable_libtorch.abi3.so` embeds `sm_120` (8 hits) and `sm_120f` (18 hits) architecture markers; (ii) the tree ships **60** `*sm120*` files, including `third_party/deep_gemm/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl` and `sm120_blockwise_mma_builder.inl`; (iii) the same map's **own** A65 row (`INFERENCE_ACCEL_GAP_MAP.md:122`) records *"shipped (v0.29.0; SM120 FP8 CUTLASS path exists and dispatches first)"*. The quoted "*does not build for sm ≥ 10.0*" is a **third-party club-3090 FAQ** statement, not this engine's state. | Surviving content: W8A8 **does** build and ship for 12.0 in vLLM 0.29.0; whatever residual gap the club FAQ recorded is engine-version-specific. §0.6 magnitude: **0 on this model** — Qwen3-4B is BF16 dense; there is no W8A8 checkpoint on the rig, so the recoverable amount for the target workload is zero regardless of kernel availability. (Kernel existence is not evidence of a speedup.) |
| **Spec-D2.6** (`spec-decode-gap-map-2026-09-13.md:669`) | *"**Adaptive verification on sm120 — effectively unavailable**"* — *"vLLM AV requires \"FULL varlen decode graphs require AttentionCGSupport.ALWAYS, which the DSV4 sparse-MLA, sparse-SWA, and indexer backends report on SM100. Elsewhere adaptive verification is rejected at startup\"."* | **HOLDS** | This is one of the strongest rows in the maps and I confirm it on the executed backend, not just on the DSV4 backends it names: the gate is `<V>/v1/worker/gpu/spec_decode/adaptive_verification.py:481-484` — `if target_attn_cg_support.min_cg_support != AttentionCGSupport.ALWAYS: raise ValueError("Adaptive verification captures varlen decode cudagraphs, so every target attention builder must report AttentionCGSupport.ALWAYS, but …")`. On this rig `AttentionCGSupport.ALWAYS` requires FA3: `<V>/v1/attention/backends/flash_attn.py:356-359` — `_cudagraph_support = (AttentionCGSupport.ALWAYS if get_flash_attn_version() == 3 else AttentionCGSupport.UNIFORM_BATCH)`. `RUN`: `get_flash_attn_version()` → **2** ⇒ `UNIFORM_BATCH` ≠ `ALWAYS` ⇒ startup rejection. The FlashInfer path is also capped below `ALWAYS` on sm12x: `<V>/v1/attention/backends/flashinfer.py:1010-1013` returns `AttentionCGSupport.UNIFORM_BATCH`. | — (Correct kill. Note the rows are consistent: the map's separate "FULL graphs work on sm120" claim concerns `UNIFORM_BATCH`-level FULL capture, a *lower* bar than AV's `ALWAYS`.) |

### D1f — the wrong-capability-gate pattern the brief flagged (audited as a *mechanism*, not a cell)

| item | what the maps say | verdict | deciding evidence |
|---|---|---|---|
| **`is_sm100_supported()` on this engine** | `INFERENCE_ACCEL_GAP_MAP.md:251` and `spec-decode-gap-map-2026-09-13.md:651` both treat `is_sm100_supported()` as a live gate that must be "widened"/"fixed" for sm120 | **The attribute does not exist in this engine** | `RUN`: `WARNING … Current platform cuda does not have 'is_sm100_supported' attribute.` Both map rows describe **SGLang** (`sgl-project/sglang` issue #36701 / #36531) or an SGLang-side `arg_groups/overrides.py`. Since **SGLang is not installed on this rig**, any candidate reasoned about through these rows is reasoned about an engine that is absent. |

---

## §3 Verdict — does any wrongly-closed cell survive §0.6 (>2% of end-to-end step time)?

**No. Zero of the four `FALSE-ON-THIS-RIG` reasons leave anything behind that clears the 2% magnitude gate.**

| FALSE reason | Surviving content | §0.6 arithmetic | Clears 2%? |
|---|---|---|---|
| **Spec-D1.8** trtllm-gen XQA decode "gated out" | XQA/TRT-LLM **decode** is reachable on sm120 in this engine (`supports_trtllm_attention(is_prefill=False) = True`) and is already being used by the FlashInfer backend; only **prefill** stays gated | upper bound = total attention read at 4K: `2 × 36 × 8 × 4096 × 128 × 2 B = 3.02e8 B` ÷ `1.52e12 B/s` = **0.199 ms/step**; vs `7–12 ms` step ⇒ **1.7–2.8%**, and that assumes attention reads cost *nothing* after the change | **No** — and the realisable gain (FA2-paged-decode → XQA at bs=1) is strictly below that ceiling |
| **Spec-D1.4** "FP8 KV as anything but storage" | FP8 KV **is** usable here via FlashInfer (`FlashInferBackend.supports_kv_cache_dtype("fp8_e4m3") = True`); the real constraint is that using it **forces you off the FlashAttention backend**, which is the one this rig runs (`01-nospec.serve.log:16`) | not boundable from source — the cost is backend-vs-backend, not a per-layer launch count; per §0.6 a candidate I cannot bound is killed | **No** |
| **Spec-D1.5** W8A8 "does not build for sm ≥ 10.0" | W8A8 code **is** in the sm120 build (`sm_120`/`sm_120f` markers; 60 `*sm120*` files) | **0** on this model — no W8A8 checkpoint; Qwen3-4B is BF16 dense | **No** |
| **INF-E18/E19** B12X MXFP8/MXFP4 "silently drops" | the sm120 hardware gate **passes** (`is_device_capability_family(120)`); the blocker is `pip install vllm[b12x]` (`ModuleNotFoundError: No module named 'b12x'`) | **0** on this model — no MXFP8/MXFP4 checkpoint | **No** |

**⇒ No §3 conditional-difference-axis sentence is issued, because no wrongly-closed cell survives.**

For completeness, the required form would have read: *"nearest neighbour X does A; under condition C it misses B;
we do B, therefore when C holds the conclusion differs."* — **I did not find a candidate that can fill A/B/C and also
clear §0.6.** The one cell that came closest is **Spec-D1.8**, whose difference axis is clean and conditional
(*"the SGLang gate `is_sm100_supported()` does A; under C = vLLM 0.29.0 on sm120 it misses B = the XQA decode kernel;
we do B, therefore when C holds the conclusion differs"*) — **but it fails §0.6 at 1.7–2.8% upper bound, so the kill
reason holds and the cell stays closed.**

### The most useful thing I found that is *not* a §0.6 survivor

The four FALSE reasons are all **"closed for the wrong reason, and it does not matter here"** — which is a different
finding from "a candidate was wrongly excluded". The audited rig's *real* optimisation ceiling is not hiding behind
these doors. What the audit **does** change, operationally:

1. **`supports_trtllm_attention(is_prefill=False) = True` and `has_nvidia_artifactory() = True` are RUN facts on this
   rig.** Any future kill reason phrased "sm120 has no TRT-LLM generation kernels" is refuted by this rig, though not
   refuted by *nothing else on the rig*, since prefill genuinely remains gated.
2. **The map's `is_device_capability_family(100)` vs the platform's `has_device_capability(100)` give opposite answers
   on sm120** (`RUN`: `False` vs `True`). Kill reasons must name which predicate they mean; the maps mostly do, and the
   ones that do are correct.
3. **Spec-D1.4's *evidence clause* was right and its *heading* over-reached.** The FP8-KV door on this rig is closed
   for FlashAttention and open for FlashInfer; that is a *forced trade*, not a missing feature, and the map should say
   so (this is the surviving actionable correction).
4. **The escalation in `notes/POOL1_ENGINE_UNLOCKED_2026-09-15.md` §2.2 / §0.2.3 is not confirmed by the artifact it
   leans on.** *"The split path is reachable on Qwen3-4B's exact 32:8 GQA decode shape"* is an **upstream-source**
   argument; the **installed** wrapper raises `NotImplementedError("FA2 does not support num_splits > 1")` at
   `<V>/vllm_flash_attn/flash_attn_interface.py:311-312`, and the rig's own log proves FA2 + 35 FULL graphs run on
   sm120. **Those two facts do not fit together and I could not resolve them without running the model.** Treat
   "sm120 split-KV is live" as **UNRESOLVED**, not as established, until someone runs the one experiment.

---

## §4 What I could not verify

Listed explicitly so none of it is read as a finding.

1. **Anything requiring model execution.** I ran **no inference and no benchmark** (the brief forbids it: another
   process owns the GPU). Consequences: (a) whether `run_flash_splitkv_fwd` is **entered** at runtime — I found the
   512/561 symbols in the binary (`RUN`) but reachability on the executed path is exactly what I could *not* test;
   (b) which of the two `num_splits` branches in `FlashAttentionMetadataBuilder.build`
   (`<V>/v1/attention/backends/flash_attn.py:596-600`) fires inside a captured FULL graph; (c) any actual speedup or
   slowdown, for any kernel named here.
2. **SGLang-side reasons (Spec-D1.9, D1.10, D1.11, and the SGLang half of Spec-D1.13 and of INF-A68's neighbours).**
   SGLang is not installed on this rig, so every reason whose deciding artifact is `sgl-project/sglang` is verified
   against an engine that is absent. I did not install it and did not attempt to.
3. **The maps' arXiv/GitHub-sourced clauses.** I did **not** re-fetch any PR, issue or paper. Every "verbatim" quote
   in §2 is quoted from **the map file itself**, and is attributed as such; where I had first-hand engine evidence I
   said so and gave the `file:line`.
4. **`INF-C99`'s open PR #2758, #2348, #2406** — not fetched, not read. My §2 D1c entry rests only on the installed
   tree and the binary.
5. **`INF-A68` (Machete Hopper-only)** — I did **not** read `CMakeLists.txt` this session. I recorded `HOLDS` on the
   map's own `READ BODY` plus the fact that no probe of mine touches it. Treat as **not independently verified**.
6. **`Spec-D1.13` (NSA on sm120)** — I did not re-read `<V>/platforms/cuda.py`'s backend-priority tables this session;
   that half of the row is inherited from `notes/POOL1_ENGINE_UNLOCKED_2026-09-15.md` §1 (C233). The vLLM-side route
   is therefore **second-hand**.
7. **`is_sm100_supported` in SGLang** — I did not fetch SGLang source, so I cannot say what that attribute returns
   there. My finding is narrower and sufficient: **it does not exist in the engine this rig has.**
8. **Quantized / MoE / MLA / hybrid / multi-GPU cells — deliberately not audited** (the brief excludes them).
   Concretely I did not attempt: the `mxfp4` indexer beyond its error string, NVFP4 KV, DFlash/DSpark+KV-quant
   composition, DeepSeek-V4 sparse-MLA on sm120, or any `D.1`/`D.2`/`D.3`/`D.4` row.
9. **Two `cudagraph`/`num_splits` code paths I read but did not trace to a conclusion** — recorded in §2 D1c rather
   than resolved. If this audit is continued, the single highest-value next step is one short run that prints
   `attn_metadata.max_num_splits` for a decode batch under `cudagraph_mode=FULL`, which settles the split-KV question
   for good.
10. **Absence claims.** Everything in this document phrased as an absence is "I did not find it" / "I did not read it",
    never "it does not exist". The one exception is a *present-tense engine fact* I did test: `is_sm100_supported`
    is absent from the installed `Platform` implementation, evidenced by the engine's own warning line.
