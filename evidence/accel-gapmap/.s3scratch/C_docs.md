# C — DOCUMENTATION gap map: weight+activation quantization for Qwen3-4B dense on RTX PRO 6000 Blackwell 96 GB, sm120

Mined from **documentation only** (no GitHub issues used). All items retrieved with
`python3 fetch.py get|raw <url>` from `/Users/leihenan/Desktop/myProject/evidence/accel-gapmap`.
Every QUOTE below is a verbatim continuous string from the page actually fetched.

**LABEL CONVENTION (read this before parsing).** The brief's `<A|B|C>` label encodes the
*state of the documentation*, not the state of a PR:
- **A = CLOSED** — the docs contain a definitive, unambiguous statement that answers the question
  (positive OR negative). The claim is settled *by the docs*.
- **B = OPEN** — the docs are SILENT, AMBIGUOUS, or SELF-CONTRADICTORY on the question. The
  QUOTE is the verbatim text that fails to settle it (a "not supported", a "TBD", a gate naming
  a different SM, or a contradiction with another doc).
- **C = ABANDONED** — a doc artifact that was retired / rotted (moved, redirect stub, dead link).

Everything in this file is **"the docs claim X"**. Nothing here is measured on our rig.
No item asserts what the RTX PRO 6000 actually does at runtime.

Scope note: KV-cache quantization and speculative decoding appear only where a doc couples them
to a weight+activation statement (flagged *ADJACENT*). Target model is Qwen3-4B dense (no MoE).

---

## Section A — LLM Compressor (vllm-project/llm-compressor + docs.vllm.ai)

### A.1 | CLOSED
WHO: vLLM project / LLM Compressor maintainers (Red Hat AI)
ARTIFACT: vLLM docs — "LLM Compressor / INT8 W8A8", the `!!! warning` block titled *Blackwell GPU Limitation*
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor/int8_w8a8.md ; https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w8a8.html
STATUS: published (live `latest` docs; warning present in both raw markdown and rendered HTML)
QUOTE: "!!! warning
    **Blackwell GPU Limitation**: INT8 is not supported on compute capability >= 10.0 (e.g., RTX 6000 Blackwell).
    Use [FP8 quantization](fp8.md) instead, or run on Hopper/Ada/Ampere architectures."
EVIDENCE: READ BODY
NOTE: This is the only statement found in ANY of the six requested doc sources that (a) names a
compute-capability threshold, (b) names "RTX 6000 Blackwell" by product, and (c) explicitly says
**not supported**. Rendered-page readback of the same warning (line 2960 of the fetched HTML text):
"Blackwell GPU Limitation : INT8 is not supported on compute capability >= 10.0 (e.g., RTX 6000 Blackwell). Use FP8 quantization instead, or run on Hopper/Ada/Ampere architectures."
sm120 reports capability (12,0) at runtime, so `>= 10.0` is satisfied by the letter of this gate.

### A.2 | CLOSED
WHO: vLLM project / LLM Compressor maintainers
ARTIFACT: vLLM docs — "LLM Compressor / FP8 W8A8" (intro paragraph + `!!! note`)
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor/fp8.md ; https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8.html
STATUS: published (live `latest` docs)
QUOTE: "Ada Lovelace, Hopper, and Blackwell GPUs are supported for W8A8."
EVIDENCE: READ BODY
SECOND QUOTE (same page): "FP8 computation is supported on NVIDIA GPUs with compute capability >= 8.9 (Ada Lovelace, Hopper, Blackwell).
    FP8 models will run on compute capability >= 7.5 (Turing) as weight-only W8A16, utilizing FP8 Marlin."
THIRD QUOTE (same page, kernel-selection note): "For block-quantized checkpoints on CUDA it tries, in order: a FlashInfer/DeepGEMM hybrid
    (Hopper only), DeepGEMM, CUTLASS, Marlin, Triton, Humming, then a PyTorch fallback. GPUs without native FP8 support (e.g. Turing/Ampere)
    land on weight-only (W8A16) Marlin."
NOTE: "Blackwell" is stated as a bare family name — the doc never disambiguates sm100 vs sm120.
The kernel chain names CUTLASS and Marlin as fallbacks but attaches no SM number to either.

### A.3 | CLOSED
WHO: vLLM project / LLM Compressor maintainers
ARTIFACT: vLLM docs — "LLM Compressor / INT4 W4A16", `!!! note` on compute capability
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor/int4.md ; https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int4.html
STATUS: published (live `latest` docs)
QUOTE: "INT4 computation is supported on NVIDIA GPUs with compute capability > 8.0 (Ampere, Ada Lovelace, Hopper, Blackwell)."
EVIDENCE: READ BODY
NOTE: No accuracy number is given on this page — it documents the recipe only.

### A.4 | CLOSED
WHO: vLLM project / LLM Compressor maintainers
ARTIFACT: LLM Compressor docs home — "Supported quantization schemes" table (has a dedicated **Compute Capability** column)
URL: https://docs.vllm.ai/projects/llm-compressor/en/latest/
STATUS: published (LLM Compressor v0.13.0 docs, `latest`)
QUOTE: "Supported quantization schemes

LLM Compressor supports applying multiple formats in a given model.

 Format 
 Targets 
 Compute Capability 
 Use Case 

 W4A16/W8A16 
 Weights 
 7.5 (Turing and up) 
 Optimize for latency on older hardware 

 W8A8-INT8 
 Weights and activations 
 7.5 (Turing and up) 
 Balanced performance and compatibility 

 W8A8-FP8 
 Weights and activations 
 8.9 (Ada Lovelace and up) 
 High throughput on modern GPUs 

 MXFP8 
 Weights and activations 
 10.0 (Blackwell) 
 Microscale FP8 

 NVFP4/MXFP4 
 Weights and activations 
 10.0 (Blackwell) 
 Maximum compression on latest hardware 

 NVFP4A16/MXFP4A16/MXFP8A16 
 Weights 
 7.5 (Turing and up) 
 Weight-only microscale compression 

 W4AFP8 
 Weights and activations 
 9.0 (Hopper and up) 
 Low-bit weights with dynamic FP8 activations 

 W4AINT8 
 Weights and activations 
 — (Arm CPU) 
 Low-bit weights with dynamic INT8 activations"
EVIDENCE: READ BODY
TABLE (rendered form of the same content):

| Format | Targets | Compute Capability | Use Case |
| --- | --- | --- | --- |
| W4A16/W8A16 | Weights | 7.5 (Turing and up) | Optimize for latency on older hardware |
| W8A8-INT8 | Weights and activations | 7.5 (Turing and up) | Balanced performance and compatibility |
| W8A8-FP8 | Weights and activations | 8.9 (Ada Lovelace and up) | High throughput on modern GPUs |
| MXFP8 | Weights and activations | 10.0 (Blackwell) | Microscale FP8 |
| NVFP4/MXFP4 | Weights and activations | 10.0 (Blackwell) | Maximum compression on latest hardware |
| NVFP4A16/MXFP4A16/MXFP8A16 | Weights | 7.5 (Turing and up) | Weight-only microscale compression |
| W4AFP8 | Weights and activations | 9.0 (Hopper and up) | Low-bit weights with dynamic FP8 activations |
| W4AINT8 | Weights and activations | — (Arm CPU) | Low-bit weights with dynamic INT8 activations |

NOTE: The `10.0 (Blackwell)` cell is the gate for **both** NVFP4 and MXFP4 weight+activation.
The table resolves "Blackwell" to the number **10.0** and stops there — it never writes 12.0 or SM120.
This table also directly contradicts A.1: here W8A8-INT8 is `7.5 (Turing and up)` with no
Blackwell exclusion. Both pages are served from `docs.vllm.ai`.

### A.5 | CLOSED
WHO: vLLM project / LLM Compressor maintainers
ARTIFACT: LLM Compressor docs — "Choosing the right compression scheme" (per-architecture section + "vLLM min. compute capability" table)
URL: https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/
STATUS: published (live `latest` docs)
QUOTE: " Scheme 
 Precision 
 Targets 
 GPU 
 vLLM min. compute capability 
 Use case 

 W4A16/W8A16 
 4 or 8 bit weights, 16-bit activations 
 Weights 
 Turing 
 7.5 
 Memory reduction on older hardware 

 W8A8-INT8 
 8-bit integer 
 Weights and activations 
 Turing 
 7.5 
 High throughput on older hardware 

 W8A8-FP8 
 8-bit floating point 
 Weights and activations 
 Lovelace 
 8.9 
 High throughput on modern GPUs 

 NVFP4 
 4-bit NVIDIA floating point 
 Weights and activations 
 Blackwell (SM100) 
 10.0 
 Maximum compression on latest hardware 

 MXFP4 
 4-bit MX floating point 
 Weights and activations 
 Blackwell (SM100) 
 10.0 
 Maximum compression; cross-platform compatible via OCP MX spec 

 MXFP8 
 8-bit MX floating point 
 Weights and activations 
 Blackwell (SM100) 
 10.0 
 High accuracy MX format; cross-platform compatible via OCP MX spec 

 W4AFP8 
 4-bit weights, FP8 activations 
 Weights and activations 
 Hopper 
 9.0 
 Low-bit weights with FP8 activations 

 W4AINT8 
 4-bit weights, INT8 activations 
 Weights and activations 
 Arm 
 - 
 Low-bit weights with INT8 activations"
EVIDENCE: READ BODY
SECOND QUOTE (same page, "Choosing the right compression scheme for your GPU hardware"): "NVIDIA Blackwell

 Minimum compute capability : 10.0

 Recommended : NVFP4 or MXFP4 for maximum compression

 Alternative : MXFP8 or FP8 for balanced compression and speed"
THIRD QUOTE (same page, FP4 subsection): "NVFP4 : NVIDIA's native 4-bit format with two-level micro-block scaling; requires calibration data for activation global scales

 MXFP4 : Microscaling FP4 format for cross-platform compatibility; per-group quantization (group_size=32) with E8M0 scales; no calibration data required if using RTN"
FOURTH QUOTE (same page, FP8 subsection): "MXFP8 : Microscaling FP8 format using per-group quantization (group_size=32) with E8M0 scales; fully dynamic activations with no calibration data required; supported on Blackwell (SM100) GPUs"
NOTE: This is the most explicit gate found: the docs spell the Blackwell row as **"Blackwell (SM100)"**,
not "10.0 and up". A reader on sm120 cannot tell from this page whether they are "Blackwell (SM100)".

### A.6 | CLOSED
WHO: vLLM project / LLM Compressor maintainers
ARTIFACT: LLM Compressor docs FAQ
URL: https://docs.vllm.ai/projects/llm-compressor/en/latest/faq/faq/
STATUS: published (live `latest` docs)
QUOTE: "There is minimal support for compressed-tensors models in SGLang, but it is not maintained nor tested by our team. Much of the integration relies on vLLM. For the most up-to-date and tested integration, vLLM is recommended."
EVIDENCE: READ BODY
SECOND QUOTE (same page, "Which model layers should be quantized?"): "Typically, all linear layers are quantized except the lm_head layer. This is because the lm_head layer is the last layer of the model and sensitive to quantization, which will impact the model's accuracy."
NOTE: The second quote is the only *accuracy-mechanism* guidance in the LMC FAQ; it is qualitative
and gives no delta.

### A.7 | CLOSED  (accuracy numbers — FP8_DYNAMIC, INT8 W8A8, W4A16 on Llama-3-8B; NO BF16 baseline)
WHO: vLLM project / LLM Compressor maintainers
ARTIFACT: LLM Compressor example READMEs — `quantization_w8a8_fp8`, `quantization_w8a8_int8`, `quantization_w4a16` (gsm8k, 250 samples, 5-shot)
URL: https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w8a8_fp8/README.md ; https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w8a8_int8/README.md ; https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a16/README.md
STATUS: published (repo `main`)
QUOTE (fp8 README): "We can see the resulting scores look good:

```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```"
QUOTE (int8 README): "```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.752|±  |0.0274|
|     |       |strict-match    |     5|exact_match|↑  |0.756|±  |0.0272|
```"
QUOTE (w4a16 README): "```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.728|±  |0.0282|
|     |       |strict-match    |     5|exact_match|↑  |0.720|±  |0.0285|
```"
EVIDENCE: READ BODY
TABLE (consolidated; the "Delta vs BF16" column is NOT in any doc — it is blank on purpose):

| Scheme | Algorithm | gsm8k flexible-extract | gsm8k strict-match | Stderr | BF16 baseline stated in doc? |
| --- | --- | --- | --- | --- | --- |
| FP8_DYNAMIC W8A8 | RTN / simple PTQ | 0.768 | 0.768 | ±0.0268 | **no** |
| W8A8 INT8 | SmoothQuant 0.8 + GPTQ | 0.752 | 0.756 | ±0.0274 / ±0.0272 | **no** |
| W4A16 INT4 | GPTQ (group 128) | 0.728 | 0.720 | ±0.0282 / ±0.0285 | **no** |

NOTE: All three numbers are on **meta-llama/Meta-Llama-3-8B-Instruct**, not Qwen3-4B. The vLLM-side
copy of the fp8 recipe (https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8.html)
repeats the identical 0.768 table. **None of the three pages states a BF16 score**, so no delta is
recoverable from LMC docs.

### B.1 | OPEN  (LMC repo and vLLM docs disagree on whether Blackwell is in the FP8 gate)
WHO IS STILL ASKING: the two doc surfaces of the same project (LLM Compressor repo vs vLLM docs)
ARTIFACT: vLLM docs — "LLM Compressor / FP8 W8A8" (Blackwell listed) vs LLM Compressor repo — `examples/quantization_w8a8_fp8/README.md` (Blackwell omitted), same recipe documented twice
URL: https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w8a8_fp8/README.md (vs A.2 above)
STATUS: both published, both live
QUOTE (LMC repo, no Blackwell): "> `fp8` compuation is supported on Nvidia GPUs with compute capability > 8.9 (Ada Lovelace, Hopper)."
EVIDENCE: READ BODY
WHAT IS MISSING: The vLLM docs copy says "compute capability >= 8.9 (Ada Lovelace, Hopper, Blackwell)";
the LLM Compressor repo copy omits Blackwell entirely and uses `>` rather than `>=`. Neither copy
names sm120, and the two disagree on both the operator and the GPU list.

### B.2 | OPEN  (LMC repo INT8 page carries no Blackwell limitation warning at all)
WHO IS STILL ASKING: LLM Compressor repo docs vs vLLM docs
ARTIFACT: LLM Compressor repo — `examples/quantization_w8a8_int8/README.md`, the INT8 W8A8 end-to-end recipe page
URL: https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w8a8_int8/README.md
STATUS: published (repo `main`)
QUOTE: "> `int8` compuation is supported on Nvidia GPUs with compute capability > 7.5 (Turing, Ampere, Ada Lovelace, Hopper)."
EVIDENCE: READ BODY
WHAT IS MISSING: This page is presented as the end-to-end INT8 W8A8 recipe and gives a gsm8k score
(0.752/0.756) with no caveat. It does **not** carry the "Blackwell GPU Limitation" warning that the
vLLM copy (A.1) carries. A reader who follows the LMC repo README to produce an INT8 W8A8 Qwen3-4B
checkpoint gets no warning that the vLLM docs say it will not run on cc >= 10.0.

### B.3 | OPEN  (W4A16 gate omits Blackwell; no sm120 disambiguation)
WHO IS STILL ASKING: LLM Compressor repo docs vs vLLM docs
ARTIFACT: LLM Compressor repo — `examples/quantization_w4a16/README.md`, the INT4 W4A16 end-to-end recipe page
URL: https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a16/README.md
STATUS: published (repo `main`)
QUOTE: "> `int4` mixed precision computation is supported on Nvidia GPUs with compute capability > 8.0 (Ampere, Ada Lovelace, Hopper)."
EVIDENCE: READ BODY
WHAT IS MISSING: vLLM's copy of the same recipe says "> 8.0 (Ampere, Ada Lovelace, Hopper, Blackwell)"
(A.3). The LMC repo copy stops at Hopper. No page states whether the W4A16 kernel path on sm120 is
the Marlin path, and none states a speed expectation.

### B.4 | OPEN  (the `< SM100` gate is written as a numeric comparison, not a family test)
WHO IS STILL ASKING: LLM Compressor maintainers (NVFP4 W4A4 example)
ARTIFACT: LLM Compressor repo — `examples/quantization_w4a4_fp4/README.md`, "Note:" line under the Quickstart section
URL: https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a4_fp4/README.md
STATUS: published (repo `main`)
QUOTE: "The resulting model `Meta-Llama-3-8B-Instruct-NVFP4` is ready to be loaded into vLLM.
Note: if running inference on a machine that is < SM100, vLLM will not run activation
quantization, only weight-only quantization."
EVIDENCE: READ BODY
WHAT IS MISSING: The gate is written `< SM100`. sm120 is *numerically* not `< 100`, but on the target
rig `torch.cuda.is_device_capability_family(100)` is FALSE — i.e. sm120 is not SM100-family. The doc
gives no way to decide which reading applies, and does not state what the fallback path is on sm120
(Marlin W4A16? something else?) or what accuracy that fallback yields.

### B.5 | OPEN  (NVFP4 calibration-data requirement stated, but no dense-model accuracy number)
WHO IS STILL ASKING: LLM Compressor maintainers (NVFP4 example)
ARTIFACT: LLM Compressor repo — `examples/quantization_w4a4_fp4/README.md`, step 2 "Prepare Calibration Data" and the end of the page (no step 4)
URL: https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a4_fp4/README.md
STATUS: published (repo `main`)
QUOTE: "Prepare the calibration data. `nvfp4` quantization generates per-tensor global scales and per-group (size 16) local quantization scales for the weights, as well as per-tensor global scales for the activations. Per-group local activation quantization scales are generated dynamically during inference time. We need some sample data to calibrate the global activation scales. Typically, a small number of samples is sufficient. In this example, we use a sample size of 20."
EVIDENCE: READ BODY
WHAT IS MISSING: This page ends at "We have successfully created an `nvfp4` model!" — it has **no
step 4 and no accuracy evaluation at all**, unlike the fp8/int8/w4a16 siblings. There is no
published LMC accuracy number for NVFP4 W4A4 on any dense model.

### C.1 | ABANDONED  (LMC example paths referenced by the main README no longer exist)
WHO: LLM Compressor maintainers
ARTIFACT: README links to `examples/quantization_w4a16_fp4`, `examples/quantization_w4a8_int8`, `examples/quantization_w8a8_mxfp8`, and `examples/quantization_w4a4_fp4/qwen3_example.py`
URL: https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a16_fp4/README.md ; https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a8_int8/README.md ; https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w8a8_mxfp8/README.md ; https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a4_fp4/qwen3_example.py
STATUS: HTTP 404 (each returned a 14-byte body)
STATED REASON: not stated anywhere; the top-level README still advertises the missing targets as live links.
QUOTE (verbatim from https://raw.githubusercontent.com/vllm-project/llm-compressor/main/README.md, lines 96-102): "### Weight and Activation Quantization
* [Activation quantization to `int8`](examples/quantization_w8a8_int8/README.md)
* [Activation quantization to `fp8`](examples/quantization_w8a8_fp8/README.md)
* [Activation quantization to MXFP8](examples/quantization_w8a8_mxfp8)
* [Activation quantization to `fp4` (NVFP4)](examples/quantization_w4a4_fp4)
* [Activation quantization to `fp4` (MXFP4)](examples/quantization_w4a4_mxfp4)
* [Activation quantization to `fp4` using AutoRound](examples/autoround/quantization_w4a4_fp4/README.md)"
EVIDENCE: READ BODY (of the 404 responses and of the README that points at them)
NOTE: Consequence — the two schemes that matter most for sm120 (MXFP8 W8A8, and any Qwen3-specific
NVFP4/W4A8 example) are the ones whose example docs are missing.

---

## Section B — TensorRT-LLM (NVIDIA)

### A.8 | CLOSED
WHO: NVIDIA / TensorRT-LLM maintainers
ARTIFACT: TensorRT-LLM docs — `docs/source/features/quantization.md`, section **Hardware Support Matrix**
URL: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/quantization.md
STATUS: published (repo `main`)
QUOTE: "## Hardware Support Matrix

| Model          |  NVFP4  | MXFP4  | FP8(per tensor)| FP8(block scaling) | FP8(rowwise) | FP8 KV Cache | NVFP4 KV Cache | W4A8 AWQ  | W4A16 AWQ | W4A8 GPTQ  | W4A16 GPTQ |
| :------------- | :---:   | :---:  | :---: | :---: | :---: | :---: | :---: | :-------: | :-------: | :--------: | :--------: |
| Blackwell(sm120)       |   Y     |   Y    |   Y   |   .   |   .   |   Y   |   .   |     .     |     .     |     .      |     .      |
| Blackwell(sm100/103)       |   Y     |   Y    |   Y   |   Y   |   .   |   Y   |   Y   |     Y     |     Y     |     Y      |     Y      |
| Hopper           |   .     |   .    |   Y   |   Y   |   Y   |   Y   |   .   |     Y     |     Y     |     Y      |     Y      |"
EVIDENCE: READ BODY
TABLE (the sm120 row alone, rendered):

| GPU | NVFP4 | MXFP4 | FP8 (per tensor) | FP8 (block scaling) | FP8 (rowwise) | FP8 KV Cache | NVFP4 KV Cache | W4A8 AWQ | W4A16 AWQ | W4A8 GPTQ | W4A16 GPTQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Blackwell(sm120) | Y | Y | Y | . | . | Y | . | . | . | . | . |
| Blackwell(sm100/103) | Y | Y | Y | Y | . | Y | Y | Y | Y | Y | Y |

NOTE: This is the **only** artifact in the entire six-source sweep that names **`sm120`** in a
weight+activation support matrix. It answers sm120 affirmatively for NVFP4, MXFP4 and per-tensor FP8,
and negatively for FP8 block scaling, FP8 rowwise, W4A8 AWQ, W4A16 AWQ, W4A8 GPTQ and W4A16 GPTQ.
(FP8 KV Cache / NVFP4 KV Cache columns are *ADJACENT* — out of scope.)
SECOND QUOTE (same page, prose above the matrix): "**The default PyTorch backend supports FP4 and FP8 quantization on the latest Blackwell and Hopper GPUs.**"
THIRD QUOTE (same page, note under the matrix): "```{note}
FP8 block wise scaling GEMM kernels for sm100/103 are using MXFP8 recipe (E4M3 act/weight and UE8M0 act/weight scale), which is slightly different from SM90 FP8 recipe (E4M3 act/weight and FP32 act/weight scale).
```"

### A.9 | CLOSED
WHO: NVIDIA / TensorRT-LLM maintainers
ARTIFACT: TensorRT-LLM docs — `docs/source/features/quantization.md`, section **Model Support Matrix** (Qwen rows)
URL: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/quantization.md
STATUS: published (repo `main`)
QUOTE: "| Qwen           |   .     |   .    |   .   |   .   |   .   |   Y   |  .  |     Y     |     Y     |     .      |     Y      |
| Qwen-2/2.5     |   Y     |   .    |   Y   |   .   |   .   |   Y   |  .  |     Y     |     Y     |     .      |     Y      |
| Qwen-3         |   Y     |   .    |   Y   |   .   |   .   |   Y   |  Y  |     .     |     Y     |     .      |     Y      |"
EVIDENCE: READ BODY
TABLE (Qwen rows, rendered; same 11 columns as A.8):

| Model | NVFP4 | MXFP4 | FP8 (per tensor) | FP8 (block scaling) | FP8 (rowwise) | FP8 KV Cache | NVFP4 KV Cache | W4A8 AWQ | W4A16 AWQ | W4A8 GPTQ | W4A16 GPTQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen | . | . | . | . | . | Y | . | Y | Y | . | Y |
| Qwen-2/2.5 | Y | . | Y | . | . | Y | . | Y | Y | . | Y |
| Qwen-3 | Y | . | Y | . | . | Y | Y | . | Y | . | Y |

NOTE: "Qwen-3" is the closest row to Qwen3-4B. Per this table Qwen-3 gets NVFP4 and per-tensor FP8
weight+activation, and **W4A16 AWQ / W4A16 GPTQ**, but **no MXFP4**. The matrix is model-family level
and does not enumerate parameter counts, so Qwen3-4B specifically is not named.
SECOND QUOTE (same page, multi-modal note): "The vision component of multi-modal models (BLIP2-OPT/BLIP2-T5/LLaVA/VILA/Nougat) uses FP16 by default.
The language component decides which quantization methods are supported by a given multi-modal model."

### B.6 | OPEN  (TRT-LLM's own overview prose contradicts its sm120 support matrix)
WHO IS STILL ASKING: TensorRT-LLM docs — `overview.md` vs `features/quantization.md`
ARTIFACT: TensorRT-LLM docs — `docs/source/overview.md`, sections "FP4 Support", "FP8 Support", and the "Advanced Quantization" bullets
URL: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/overview.md
STATUS: published (repo `main`)
QUOTE: "### FP4 Support
[NVIDIA B200 GPUs](https://www.nvidia.com/en-us/data-center/dgx-b200/), when used with TensorRT LLM, enable seamless loading of model weights in the new [FP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/#what_is_nvfp4), allowing you to automatically leverage optimized FP4 kernels for efficient and accurate low-precision inference."
SECOND QUOTE (same page): "### FP8 Support

On NVIDIA H100 and later GPUs, TensorRT LLM supports [FP8 quantization](./features/quantization.md), which can double performance and halve memory consumption compared to 16-bit floating point, with minimal impact on model accuracy."
THIRD QUOTE (same page, Advanced Quantization bullets): "- **FP4 Quantization**: Native support on NVIDIA B200 GPUs with optimized FP4 kernels
  - **FP8 Quantization**: Automatic conversion on NVIDIA H100 GPUs leveraging Hopper architecture"
EVIDENCE: READ BODY
WHAT IS MISSING: `overview.md` attributes FP4 to **B200** (sm100 datacenter) and FP8 to **H100**,
with no mention of Blackwell GeForce / RTX PRO / sm120. `features/quantization.md` (A.8) marks
NVFP4=Y and MXFP4=Y for `Blackwell(sm120)`. Both pages are live on `main`; neither is marked stale.

### B.7 | OPEN  (legacy `precision.html` matrix is stale and contradicts the current matrix)
WHO IS STILL ASKING: TensorRT-LLM docs — `reference/precision.html` vs `features/quantization.md`
ARTIFACT: TensorRT-LLM docs — `reference/precision.html` (rendered), sections "NVFP4 (Blackwell)" and "Support matrix"
URL: https://nvidia.github.io/TensorRT-LLM/reference/precision.html
STATUS: published (live docs site)
QUOTE: " NVFP4 (Blackwell) # 

 LLama and Mixtral can run in NVFP4 datatype. Those examples can be found in Llama examples."
SECOND QUOTE (same page, support matrix header): " Model
 
 FP32
 
 FP16
 
 BF16
 
 FP8
 
 NVFP4
 
 W8A8 SQ
 
 W8A16
 
 W4A16
 
 W4A16 AWQ
 
 W4A16 GPTQ"
THIRD QUOTE (same page, the Qwen row of that matrix; 10 cells = FP32, FP16, BF16, FP8, NVFP4, W8A8 SQ, W8A16, W4A16, W4A16 AWQ, W4A16 GPTQ): " Qwen
 
 Y
 
 Y
 
 Y
 
 .
 
 .
 
 Y
 
 Y
 
 Y
 
 Y
 
 Y"
EVIDENCE: READ BODY
WHAT IS MISSING: `precision.html`'s matrix has **no MXFP4 column, no sm120 row, no Qwen-3 row**, and
marks Qwen **FP8 = "." and NVFP4 = "."** — i.e. it says Qwen cannot do FP8 or NVFP4 at all, whereas
A.9's current matrix says Qwen-3 does both. The page carves NVFP4 out as "(Blackwell)" and gives the
only named models as Llama and Mixtral. No accuracy table appears anywhere on this page.

### A.10 | CLOSED  (*ADJACENT* — KV cache, recorded only because it constrains W/A choice)
WHO: NVIDIA / TensorRT-LLM maintainers
ARTIFACT: TensorRT-LLM docs — `docs/source/features/quantization.md`, NVFP4 KV Cache section
URL: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/quantization.md
STATUS: published (repo `main`)
QUOTE: "Note that currently TRT-LLM only supports FP8 weight/activation quantization when NVFP4 KV cache is enabled. Therefore, `--quant fp8` is required here."
EVIDENCE: READ BODY
NOTE: *ADJACENT* — KV-cache quantization is out of scope for this map. Included only because it is a
doc statement that couples a KV-cache mode to a **weight/activation** mode being forced to FP8.

---

## Section C — torchao (pytorch/ao)

### A.11 | CLOSED
WHO: Meta / PyTorch torchao maintainers
ARTIFACT: torchao docs — "Quantized Inference" workflow table (each row states its own arch requirement)
URL: https://pytorch.org/ao/main/workflows/inference.html (moved here from `torchao/quantization/README.md`)
STATUS: published (torchao `main` docs; page cites torch 2.12.0.dev and vllm 0.13.0 in its benchmark section)
QUOTE: "Float8DynamicActivationFloat8WeightConfig : Applies float8 dynamic symmetric quantization to both activations and weights. Requires CUDA ≥8.9, AMD MI350+, or Intel XPU. Supports PerTensor and PerRow granularity."
SECOND QUOTE (same table): "MXDynamicActivationMXWeightConfig (prototype): Applies mxfp8 or mxfp4 dynamic quantization to activations and weights. Requires NVIDIA SM100+ (Blackwell) or AMD MI350+."
THIRD QUOTE (same table): "NVFP4DynamicActivationNVFP4WeightConfig (prototype): Applies NVFP4 dynamic quantization to activations and weights with double quantization (per-tensor + per-block scales). Requires NVIDIA SM100+ (Blackwell)."
FOURTH QUOTE (same table, the mxfp4 row repeats): "MXDynamicActivationMXWeightConfig (prototype): Applies mxfp8 or mxfp4 dynamic quantization to activations and weights. Requires NVIDIA SM100+ (Blackwell) or AMD MI350+."
FIFTH QUOTE (same table, weight-only NVFP4 — for contrast, no arch gate): "NVFP4WeightOnlyConfig (prototype): Applies NVFP4 weight-only quantization."
EVIDENCE: READ BODY
NOTE: **Weight+activation** FP8 (`Float8DynamicActivationFloat8WeightConfig`) is gated only at
`CUDA ≥8.9` — no SM100 requirement, so sm120 clears it by the letter. Both **NVFP4 W4A4** and
**MXFP8/MXFP4 W8A8/W4A4** are gated at **`NVIDIA SM100+ (Blackwell)`**. Also note the asymmetry:
NVFP4 *weight-only* carries no arch gate while NVFP4 *weight+activation* requires SM100+.

### B.8 | OPEN  (torchao docs never mention sm120, RTX, or consumer Blackwell anywhere)
WHO IS STILL ASKING: torchao maintainers — the gate wording `SM100+` is never defined
ARTIFACT: torchao docs — "Quantized Inference" workflow table, prototype config rows (`NVFP4DynamicActivationNVFP4WeightConfig`, `MXDynamicActivationMXWeightConfig`)
URL: https://pytorch.org/ao/main/workflows/inference.html ; https://raw.githubusercontent.com/pytorch/ao/main/README.md
STATUS: published (live docs)
QUOTE: "NVFP4DynamicActivationNVFP4WeightConfig (prototype): Applies NVFP4 dynamic quantization to activations and weights with double quantization (per-tensor + per-block scales). Requires NVIDIA SM100+ (Blackwell)."
EVIDENCE: READ BODY
WHAT IS MISSING: A `grep -iE 'sm ?120|sm_120|12\.0|rtx|pro 6000|consumer|geforce|5090|sm121|sm_121'`
over the *entire* fetched torchao inference docs page and the torchao README returned **zero matches**
(the only `12.0` hits are the string `2.12.0.dev` inside a PyTorch version number — line 565/581/597
of the fetched text). "SM100+" is left undefined: it can be read numerically (sm120 qualifies) or
architecturally (sm120 does not). torchao is the one source in this sweep whose primary
weight+activation FP4 path is gated on a capability family rather than a numeric threshold.

### A.12 | CLOSED  (accuracy table WITH a BF16 baseline — the only one found in the whole sweep)
WHO: Meta / PyTorch torchao maintainers
ARTIFACT: torchao docs — "Accuracy benchmarks", meta-llama/Llama-3.1-8B via lm-eval
URL: https://pytorch.org/ao/main/workflows/inference.html
STATUS: published (live docs)
QUOTE: "All the following benchmarks are for meta-llama/Llama-3.1-8B using lm-eval ."
TABLE (verbatim cell contents from the fetched page, rendered):

| weight | activation | wikitext-perplexity | winogrande | checkpoint size (GB) |
| --- | --- | --- | --- | --- |
| bfloat16 | bfloat16 | 7.3315 | 0.7380 | 16.1 |
| float8_rowwise | float8_rowwise | 7.4197 | 0.7388 | 9.1 |
| int8_rowwise | bfloat16 | 7.3451 | 0.7340 | 9.1 |
| int8_rowwise | int8_rowwise | 7.4535 | 0.7285 | 9.1 |
| mxfp8 | mxfp8 | 7.6034 | 0.7316 | 9.32 |
| nvfp4 | nvfp4 | 8.4459 | 0.7135 | 6.05 |

QUOTE (reproduction commands, same page): "To reproduce, run the following command:

 // on an H100
 SKIP_VLLM = 1 ./benchmarks/quantization/measure_accuracy_and_performance.sh h100
// on a B200
 SKIP_VLLM = 1 ./benchmarks/quantization/measure_accuracy_and_performance.sh b200"
EVIDENCE: READ BODY
NOTE: This is the **only** artifact in the sweep that publishes a BF16 baseline alongside quantized
scores, so it is the only place a delta can be computed from docs. Deltas vs BF16 (computed here, not
quoted): float8_rowwise ppl **+0.0882 / winogrande +0.0008**; int8 W8A8 ppl **+0.1220 / winogrande
−0.0095**; mxfp8 ppl **+0.2719 / winogrande −0.0064**; nvfp4 ppl **+1.1144 / winogrande −0.0245**.
Model is Llama-3.1-8B, **not** Qwen3-4B, and the harness is run on H100/B200 — no sm120 run is published.

### A.13 | CLOSED
WHO: Meta / PyTorch torchao maintainers
ARTIFACT: torchao docs — "Performance benchmarks / e2e model level benchmarks" (Llama-3.1-8B, torch 2.9.0, vllm 0.13.0)
URL: https://pytorch.org/ao/main/workflows/inference.html
STATUS: published (live docs)
QUOTE: "All the following benchmarks are for meta-llama/Llama-3.1-8B using torch==2.9.0 and vllm==0.13.0 ."
TABLE (verbatim cell contents, rendered):

| GPU | weight | activation | prefill toks/s | decode toks/s | prefill_speedup | decode_speedup |
| --- | --- | --- | --- | --- | --- | --- |
| NVIDIA B200 | bfloat16 | bfloat16 | 59099.9 | 14380 | 1 | 1 |
| NVIDIA B200 | mxfp8 | mxfp8 | TODO(https://github.com/pytorch/ao/issues/3549) | - | - | - |
| NVIDIA B200 | nvfp4 | nvfp4 | 102786 | 15218.9 | 1.739 | 1.058 |
| NVIDIA B200 | float8_rowwise | float8_rowwise | 69313.7 | 15984 | 1.173 | 1.112 |
| NVIDIA H100 | bfloat16 | bfloat16 | 30946.5 | 6612 | 1 | 1 |
| NVIDIA H100 | float8_rowwise | float8_rowwise | 45312.5 | 8025.95 | 1.464 | 1.214 |
| NVIDIA H100 | int8_rowwwise | bfloat16 | 28231.9 | 4309.8 | 0.912 | 0.652 |
| NVIDIA H100 | int4 | float8_rowwise | TODO(https://github.com/pytorch/ao/issues/3550) | - | - | - |

QUOTE (reproduction, same page): "// on an h100
 SKIP_LM_EVAL = 1 ./benchmarks/quantization/measure_accuracy_and_performance.sh h100
// on a b200
 SKIP_LM_EVAL = 1 ./benchmarks/quantization/measure_accuracy_and_performance.sh b200"
EVIDENCE: READ BODY
NOTE: Both weight+activation FP4/MXFP8 speedups are published **only for B200 and H100** — no sm120,
no RTX PRO, no consumer row. The row name in the doc is misspelled `int8_rowwwise` (three w's);
quoted as-is. Note also that INT8 W8A8 on H100 is a **slowdown** (0.912 prefill / 0.652 decode) —
the only negative speedup published in the sweep.

### A.14 | CLOSED
WHO: Meta / PyTorch torchao maintainers
ARTIFACT: torchao docs — "Low-Precision FP8 Attention (Prototype)"
URL: https://pytorch.org/ao/main/workflows/inference.html
STATUS: published (live docs, marked Prototype)
QUOTE: "FP8 low-precision attention for inference, built on Flash Attention backends. Currently supports FA3 on Hopper (SM90) and FA4 on Blackwell (SM100)."
SECOND QUOTE (same section): "Requirements: PyTorch >= 2.11, Hopper or Blackwell GPU, Flash Attention 3 ( pip install flash-attn-3 --index-url=https://download.pytorch.org/whl/{cuda_version} )."
EVIDENCE: READ BODY
NOTE: *ADJACENT-ish* (attention, not weight/activation GEMM) but it is the sharpest example of
torchao naming **SM100** as "Blackwell" while the requirement line says only "Hopper or Blackwell GPU".
The two sentences on the same page disagree about how specific the gate is.

### A.15 | CLOSED
WHO: Meta / PyTorch torchao maintainers
ARTIFACT: torchao README — "Results" / highlight bullets for weight+activation FP8 and weight-only INT4
URL: https://raw.githubusercontent.com/pytorch/ao/main/README.md
STATUS: published (repo `main`)
QUOTE: "- **Float8 dynamic quantization**: [1.5-1.6x speedup on gemma-3-27b-it](https://huggingface.co/pytorch/gemma-3-27b-it-FP8/blob/main/README.md#results-h100-machine) and [1.54x and 1.27x speedup on Flux.1-Dev* and CogVideoX-5b respectively](https://github.com/sayakpaul/diffusers-torchao) on H100 with preserved quality"
SECOND QUOTE (same list): "- **Int4 weight-only**: [1.73x speedup with 65% less memory](https://huggingface.co/pytorch/gemma-3-12b-it-INT4) for Gemma3-12b-it on H100 with slight impact on accuracy"
EVIDENCE: READ BODY
NOTE: Every headline speedup is labeled **on H100**. The README does contain `torch.float8` /
`float8` / `nvfp4` kernel-install guidance ("# optional - install MSLK for float8 and nvfp4 inference
kernels") but attaches no SM number to it. No `torch.float4` dtype claim was found in the README.

### C.2 | ABANDONED  (torchao's own quantization README and mx_formats README are now redirect stubs)
WHO: PyTorch torchao maintainers
ARTIFACT: `torchao/quantization/README.md` (132 bytes) and `torchao/prototype/mx_formats/README.md` (345 bytes)
URL: https://raw.githubusercontent.com/pytorch/ao/main/torchao/quantization/README.md ; https://raw.githubusercontent.com/pytorch/ao/main/torchao/prototype/mx_formats/README.md
STATUS: replaced by redirects; content moved off-repo
STATED REASON (verbatim, entire file): "# Quantization

The quantization documentation has moved to the torchao docs:

https://pytorch.org/ao/main/workflows/inference.html"
SECOND (verbatim, entire mx_formats file): "# MX training and inference with native PyTorch

Training documentation has moved to the mxfp8 section of [Quantized Training](https://pytorch.org/ao/main/workflows/training.html#mxfp8).

Inference documentation has moved to the the mxfp8, nvfp4 and mxfp4 sections of [Quantized Inference](https://pytorch.org/ao/main/workflows/inference.html)."
EVIDENCE: READ BODY
QUOTE (consolidated verbatim, both stub files in order — each file is reproduced in full): "# Quantization

The quantization documentation has moved to the torchao docs:

https://pytorch.org/ao/main/workflows/inference.html

# MX training and inference with native PyTorch

Training documentation has moved to the mxfp8 section of [Quantized Training](https://pytorch.org/ao/main/workflows/training.html#mxfp8).

Inference documentation has moved to the the mxfp8, nvfp4 and mxfp4 sections of [Quantized Inference](https://pytorch.org/ao/main/workflows/inference.html)."
NOTE: Doc-rot consequence — vLLM's own TorchAO page still points readers at the dead anchor:
"Some benchmark numbers can be found [here](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks)."
(https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/torchao.md).
The `#benchmarks` anchor that vLLM links to exists only in the moved docs.

---

## Section D — SGLang

### A.16 | CLOSED
WHO: SGLang project (LMSYS) maintainers
ARTIFACT: SGLang docs — "Advanced Features / Quantization", table **Platform Compatibility**
URL: https://docs.sglang.io/advanced_features/quantization.html
STATUS: published (live docs site)
QUOTE: " Model NVIDIA GPUs AMD GPUs (MI300X/MI325X/MI350X) Ascend NPUs (A2/A3/A5) Notes 
 fp8 Yes Yes WIP Aiter or Triton backend on AMD 
 mxfp4 Yes Yes Yes (A5) On GPU: requires CDNA3/CDNA4 with MXFP support (uses Aiter). On Ascend NPU (A5): W4A4 MXFP4 for Qwen3 dense and MoE LLMs (MXFP4 weights + activations) — dense models support online dual-level MXFP4; offline W4A4_MXFP4 dense and MoE checkpoints (single-level) are auto-detected via modelslim . On Intel GPUs (XPU): W4A16 MoE experts on Xe2/BMG via sgl-kernel-xpu , enabled automatically with —device xpu (see Intel GPUs (XPU) ) 
 mxfp8 No No Yes (A5 for Diffusion, LLM Dense Linear and LLM MoE) Ascend NPU only; online + offline MXFP8 for Diffusion models (e.g., Wan2.2), LLM Dense Linear, and LLM MoE (FusedMoE, e.g. Qwen3-30B-A3B) on A5 series; uses CANN npu_dynamic_mx_quant / npu_quant_matmul (dense) and npu_grouped_matmul_swiglu_quant_v2 / npu_grouped_matmul (MoE) kernels 
 mxfp_w4a8 No No Yes (A5) Ascend NPU only; online W4A8 for Qwen3 dense LLM (MXFP4 weights + MXFP8 activations) on A5 series; offline W4A8_MXFP dense and MoE checkpoints are auto-detected via modelslim 
 blockwise_int8 Yes Yes No Triton-based, works on both platforms 
 w8a8_int8 Yes Yes No 
 w8a8_fp8 Yes Yes No Aiter or Triton FP8 on AMD 
 awq Yes Yes Yes Uses a JIT-compiled CUDA kernel on NVIDIA, Triton dequantize on AMD. Uses CANN kernels on Ascend 
 gptq No No Yes Removed on NVIDIA and AMD GPUs — use gptq_marlin instead. Uses CANN kernels on Ascend. Still supported on Intel CPUs with AMX 
 compressed-tensors Yes Yes Partial Aiter paths for FP8/MoE on AMD. Uses CANN kernels on Ascend, FP8 not supported yet"
EVIDENCE: READ BODY
NOTE: The table's NVIDIA column is a **single undifferentiated "Yes"** — there is no sm90/sm100/sm120
split anywhere in the method matrix. Two rows carry a parenthetical NVIDIA gate:
SECOND QUOTE (same table): "modelopt / modelopt_fp8 Yes (Hopper/SM90+) No No NVIDIA ModelOpt ; requires NVIDIA hardware"
THIRD QUOTE (same table): "modelopt_fp4 Yes (SM80-SM90 via Marlin; SM100+ native FP4) No No NVIDIA ModelOpt ; use Marlin W4A16 fallback on Ampere/Hopper and native FP4 backends on Blackwell; supports load-time BF16/FP16/FP8 MoE conversion with per-tensor FP32 activation scales"
FOURTH QUOTE (same table): "nvfp4_online Yes (Blackwell/SM100 or SM103) No No Online MoE-only NVFP4 weight quantization with per-token FP32 activation scales for BF16/FP16/FP8 checkpoints; use modelopt_fp4 for per-tensor FP32 activation scales; requires flashinfer_trtllm or flashinfer_trtllm_routed"
FIFTH QUOTE (same table): "gptq No No Yes Removed on NVIDIA and AMD GPUs — use gptq_marlin instead. Uses CANN kernels on Ascend. Still supported on Intel CPUs with AMX"
NOTE on the gate wording: `nvfp4_online` is gated at **"Blackwell/SM100 or SM103"** — sm120 is
conspicuously absent from that enumeration even though SGLang's GEMM-backend tables (A.17/A.18 below)
do list SM120 for FP4. Also note `gptq` is **removed on NVIDIA** entirely.

### A.17 | CLOSED
WHO: SGLang project (LMSYS) maintainers
ARTIFACT: SGLang docs — Quantization, `--fp8-gemm-backend` (Blockwise FP8 GEMM) table + auto-selection order
URL: https://docs.sglang.io/advanced_features/quantization.html
STATUS: published (live docs site)
QUOTE: " Backend Hardware Description 
 auto All Auto-selects based on hardware 
 deep_gemm SM90, SM100 JIT-compiled; enabled when DeepGEMM is installed 
 flashinfer_trtllm SM100 FlashInfer TensorRT-LLM backend; optimal for low-latency 
 flashinfer_cutlass SM100/120 FlashInfer CUTLASS groupwise FP8 GEMM 
 flashinfer_deepgemm SM90 Uses swapAB optimization for small M dimensions in decoding 
 cutlass SM120 sgl-kernel CUTLASS 
 triton All Fallback; widely compatible 
 aiter ROCm AMD AITER backend"
SECOND QUOTE (same section): "auto selection order: 1) DeepGEMM (SM90/SM100, installed); 2) FlashInfer TRTLLM (SM100, FlashInfer available); 3) CUTLASS (SM120); 4) AITER (AMD); 5) Triton (fallback)."
THIRD QUOTE (same section): "MXFP8 dense linear: auto uses flashinfer_cutlass on SM100 (else triton ). flashinfer_cutlass is fastest on most shapes; flashinfer_trtllm is faster only at small M."
EVIDENCE: READ BODY
NOTE: This is a **positive, explicit SM120 statement for weight+activation FP8 GEMM**: two backends
(`flashinfer_cutlass`, `cutlass`) list SM120, and SM120 is step 3 of the auto-selection order.
It is the only auto-selection-order text found that puts a numeric SM120 entry in a fallback chain.
Caveat: the doc does not state which of these serves **per-tensor** vs **blockwise** FP8 on sm120.

### A.18 | CLOSED
WHO: SGLang project (LMSYS) maintainers
ARTIFACT: SGLang docs — Quantization, `--fp4-gemm-backend` (NVFP4 GEMM) table
URL: https://docs.sglang.io/advanced_features/quantization.html
STATUS: published (live docs site)
QUOTE: " Backend Hardware Description 
 auto SM80+ Auto-selects: flashinfer_cutedsl on SM100; marlin on SM80-SM90; flashinfer_cutlass otherwise (including SM120) 
 flashinfer_cutlass SM100/120 FlashInfer CUTLASS backend 
 flashinfer_cudnn SM100/120 (CUDA 13+, cuDNN 9.15+) FlashInfer cuDNN backend 
 flashinfer_cutedsl SM100 FlashInfer CuTe DSL backend 
 flashinfer_trtllm SM100 FlashInfer TensorRT-LLM backend 
 marlin SM80-SM90 Weight-only W4A16 fallback for NVFP4 checkpoints"
SECOND QUOTE (same section): "On SM80-SM90, auto selects Marlin for NVFP4. NVFP4 GEMM requires FlashInfer to be installed."
EVIDENCE: READ BODY
NOTE: This is the **strongest positive sm120 statement found for NVFP4 weight+activation** in any
doc: the word `SM120` appears inside the auto-selection description ("flashinfer_cutlass otherwise
(including SM120)") and two backends are labeled `SM100/120`. It is also the only place in the whole
sweep that names a concrete **dependency** for FP4 on consumer Blackwell: **FlashInfer must be
installed**. This table sits in tension with A.16's `modelopt_fp4` row (`SM100+ native FP4`) and
`nvfp4_online` row (`Blackwell/SM100 or SM103`).

### B.9 | OPEN  (SGLang's own method matrix and backend matrix give different sm120 answers)
WHO IS STILL ASKING: SGLang docs, internally inconsistent between two tables on the same page
ARTIFACT: SGLang docs — "Advanced Features / Quantization", the `Platform Compatibility` table vs the `--fp4-gemm-backend` table (both on the same page)
URL: https://docs.sglang.io/advanced_features/quantization.html
STATUS: published (live docs site) — both tables on the SAME page
QUOTE (method matrix, gating native FP4 at SM100+): "modelopt_fp4 Yes (SM80-SM90 via Marlin; SM100+ native FP4) No No NVIDIA ModelOpt ; use Marlin W4A16 fallback on Ampere/Hopper and native FP4 backends on Blackwell; supports load-time BF16/FP16/FP8 MoE conversion with per-tensor FP32 activation scales"
QUOTE (backend table, same page, listing SM120): "flashinfer_cutlass SM100/120 FlashInfer CUTLASS backend"
EVIDENCE: READ BODY
WHAT IS MISSING: The method-level table tells an sm120 reader that native FP4 is SM100+ and that the
fallback on non-SM100 is **Marlin W4A16 (weight-only)** — i.e. activations would not be quantized.
The backend-level table tells the same reader that `flashinfer_cutlass` runs on SM120. The page never
reconciles the two, never states whether `modelopt_fp4` on sm120 lands on native FP4 or on the
W4A16 Marlin fallback, and never states an accuracy or speed consequence either way.

### B.10 | OPEN  (SGLang publishes no sm120/sm100 accuracy delta for any quant method)
WHO IS STILL ASKING: SGLang maintainers — the Quantization page has no accuracy section
ARTIFACT: SGLang docs — "Advanced Features / Quantization", intro paragraph and AutoRound section
URL: https://docs.sglang.io/advanced_features/quantization.html
STATUS: published (live docs site)
QUOTE: "Quantized models must be validated via benchmarks post-quantization to guard against abnormal quantization loss regressions."
EVIDENCE: READ BODY
WHAT IS MISSING: This is the page's entire stance on accuracy — it tells the reader to go measure.
A `grep -iE 'gsm8k|MMLU|accuracy|perplexity'` over the fetched page returns no accuracy table and no
numeric score for any scheme. The only adjacent numbers are an AutoRound failure report:
SECOND QUOTE (same page, AutoRound section): "Qwen2.5-VL-7B auto_round:auto_gptq format: Accuracy is close to zero. GPTQ format: Fails with: Output"
THIRD QUOTE (same page): "SGLang API Usage only supports auto-round-int8 quantization method now, more quantization methods are on the way."

---

## Section E — bitsandbytes / AutoAWQ / GPTQModel

### A.19 | CLOSED
WHO: bitsandbytes-foundation maintainers
ARTIFACT: bitsandbytes README — "System Requirements / Accelerator support" table
URL: https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/README.md
STATUS: published (repo `main`; the README itself notes it reflects the development branch)
QUOTE: "##### Legend:
🚧 = Planned |
〰️ = Partially Supported |
✅ = Supported |
❌ = Not Supported"
SECOND QUOTE (same table, the x86-64 Linux NVIDIA row — the row that applies to a discrete RTX PRO 6000; column order is Platform / Accelerator / Hardware Requirements / LLM.int8() / QLoRA 4-bit / 8-bit Optimizers): "    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM60+ minimum<br>SM75+ recommended</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>"
THIRD QUOTE (same table, the only `SM12x` token anywhere in the sweep. It appears twice with the same shape — once under the `arm64` platform heading, one row above the `🍎 macOS 14+` heading — i.e. it describes an aarch64 + NVIDIA host, NOT an x86-64 RTX PRO 6000): "    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM121</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>"
EVIDENCE: READ BODY
NOTE: For x86-64 + NVIDIA the requirement is **"SM60+ minimum / SM75+ recommended"** — a floor only,
which sm120 clears numerically. The library's own README carries a version caveat for this table:
"Note: this table reflects the status of the current development branch. For the latest stable release, see the
[document in the 0.50.0 tag](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.0/README.md#accelerator-support)."
No accuracy numbers and no weight+activation scheme (bnb's schemes here are LLM.int8() W8A16-ish and
QLoRA 4-bit weight-only) — see DEAD ENDS.

### A.20 | CLOSED
WHO: AutoAWQ maintainers (casper-hansen)
ARTIFACT: AutoAWQ README — "Install / Prerequisites / NVIDIA"
URL: https://raw.githubusercontent.com/casper-hansen/AutoAWQ/main/README.md
STATUS: published (repo `main`; AutoAWQ is archived upstream in the wider ecosystem, but the live README states no deprecation on the fetched page)
QUOTE: "- NVIDIA:
  - Your NVIDIA GPU(s) must be of Compute Capability 7.5. Turing and later architectures are supported.
  - Your CUDA version must be CUDA 11.8 or later."
SECOND QUOTE (same README, the compute-bound caveat — the closest thing to a limitation note in any of these three): "In the scenario of being compute-bound, which happens at higher batch sizes, you will not gain a speed-up using a W4A16 quantized model because the overhead of dequantization will slow down the overall generation. This happens because AWQ quantized models only store the weights in INT4 but perform FP16 operations during inference, so we are essentially converting INT4 -> FP16 during inference."
EVIDENCE: READ BODY
NOTE: "Compute Capability 7.5 ... Turing and later" is a floor with no upper bound and no sm120
mention — sm120 is not excluded by the letter of this text. The second quote is the clearest doc
statement found that **W4A16 is a memory play, not a compute play**, and can be a net loss at high
batch. **AutoAWQ is weight-only (W4A16)** — it is not a weight+activation scheme; see DEAD ENDS.

### B.11 | OPEN  (GPTQModel gates its Blackwell kernel at sm100 and never mentions sm120)
WHO IS STILL ASKING: GPTQModel maintainers (ModelCloud)
ARTIFACT: GPTQModel README — "Platform and HW Support" table, the note beneath it, and the Swordfish Kernel acknowledgement entry
URL: https://raw.githubusercontent.com/ModelCloud/GPTQModel/main/README.md
STATUS: published (repo `main`, release notes through 08/31/2026)
QUOTE (Platform and HW Support table): "| Platform | Device |  | Optimized Arch | Kernels |
|---|---|---|---|---|
| 🐧 Linux | NVIDIA GPU | ✅ | `Turing+` (`sm_75+`) | Machete, Marlin, Exllama V3 / EXL3, Exllama V2, AWQ GEMM/GEMV, ParoQuant CUDA/Triton, GGUF CUDA/Triton, QQQ, BitBLAS, Triton, BitsAndBytes, Torch |"
SECOND QUOTE (same README, immediately under that table): "`Marlin` and JIT CUDA kernels now support NVIDIA `Turing+` (`sm_75+`) GPUs."
THIRD QUOTE (same README, acknowledgements): "Swordfish Kernel: Blackwell (`>= sm100`) GPTQ/AWQ kernel from [AlpinDale](https://x.com/AlpinDale). [Paper](https://blog.alpindale.net/posts/swordfish/)"
EVIDENCE: READ BODY
WHAT IS MISSING: The supported-arch cell is the string "`Turing+` (`sm_75+`)" — an open-ended floor
with no upper bound, so sm120 is neither claimed nor excluded. The newest Blackwell-specific kernel
the project ships is gated as **`>= sm100`**, and `sm120` appears nowhere in the README. There is no
statement of which GPTQModel kernel is expected to serve a W4A16 or W4A8 checkpoint on sm120, and no
accuracy table. GPTQModel's schemes are **weight-only / W4A8-W4A16**; see DEAD ENDS.

---

## Section F — NVIDIA Model-Optimizer

### A.21 | CLOSED
WHO: NVIDIA / Model Optimizer maintainers
ARTIFACT: Model-Optimizer docs — `examples/hf_ptq/README.md`, "Support Matrix / Hugging Face Supported Models" + footnotes
URL: https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/examples/hf_ptq/README.md
STATUS: published (repo `main`; release notes through 2026)
QUOTE: "| Model | fp8 | int8_smoothquant | int4_awq | w4a8_awq_beta<sup>1</sup> | nvfp4<sup>5</sup> |
| :---: | :---: | :---: | :---: | :---: | :---: |"
SECOND QUOTE (same table, the Qwen3 row): "| QWen3, 3.5 MOE, Next <sup>6</sup> | ✅ | - | - | - | ✅ |"
THIRD QUOTE (same page, footnote 5): "*<sup>5.</sup>A selective set of the popular models are internally tested. The actual model support list may be longer. NVFP4 inference requires Blackwell GPUs and TensorRT-LLM v1.2 or later*"
FOURTH QUOTE (same page, footnote 1): "*<sup>1.</sup>The w4a8_awq_beta is an experimental quantization scheme that may result in a higher accuracy penalty.*"
FIFTH QUOTE (same page, the W4A8 note): "*<sup>3.</sup>W4A8_AWQ is only available on some models but not all*"
EVIDENCE: READ BODY
NOTE: "NVFP4 inference requires Blackwell GPUs" is the **loosest** NVFP4 hardware gate found anywhere
in the sweep — it says "Blackwell" with no SM number, which is the one wording that read literally
would cover sm120. No accuracy numbers appear anywhere in this file.

### B.12 | OPEN  (Model-Optimizer's FP8 footnote stops at Ada/Hopper and omits Blackwell)
WHO IS STILL ASKING: Model-Optimizer docs vs vLLM docs (A.2)
ARTIFACT: Model-Optimizer docs — `examples/hf_ptq/README.md`, "Technical Resources" footnotes, item 1 (FP8 format availability)
URL: https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/examples/hf_ptq/README.md
STATUS: published (repo `main`)
QUOTE: "1. The [FP8 format](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) is available on the Hopper and Ada GPUs with [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) greater than or equal to 8.9."
EVIDENCE: READ BODY
WHAT IS MISSING: This is the FP8 availability statement in NVIDIA's own PTQ documentation, and it
names **Hopper and Ada only**. It is `>= 8.9`, so sm120 clears it numerically, but no Blackwell
variant is named — while vLLM's FP8 page (A.2) explicitly adds Blackwell and Model-Optimizer's own
NVFP4 footnote (A.21) says "Blackwell". The same file's own NVFP4 entry therefore claims a *newer*
architecture than its FP8 entry.

### A.22 | CLOSED
WHO: NVIDIA / Model Optimizer maintainers
ARTIFACT: Model-Optimizer docs — `examples/hf_ptq/README.md`, NVFP4 technical notes (recovery guidance)
URL: https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/examples/hf_ptq/README.md
STATUS: published (repo `main`)
QUOTE: "1. The [NVFP4](https://blogs.nvidia.com/blog/generative-ai-studio-ces-geforce-rtx-50-series/) is one of the new FP4 formats supported by NVIDIA Blackwell GPU and demonstrates good accuracy compared with other 4-bit alternatives. NVFP4 can be applied to both model weights as well as activations, providing the potential for both a significant increase in math throughput and reductions in memory footprint and memory bandwidth usage compared to the FP8 data format on Blackwell. For higher accuracy with NVFP4 PTQ, we recommend `nvfp4_mlp_only`, `nvfp4_experts_only`, or `nvfp4_omlp_only`. `nvfp4_mlp_only` restricts NVFP4 quantization to MLP (and MoE) layers only, leaving attention layers in higher precision. `nvfp4_experts_only` quantizes only expert layers (`*mlp.experts*` and `*block_sparse_moe*`), ideal for MoE models. `nvfp4_omlp_only` extends MLP-only by also quantizing the `o_proj` layer, providing a middle ground between full NVFP4 and MLP-only quantization."
EVIDENCE: READ BODY
NOTE: This is the **strongest statement of an accuracy-recovery strategy for NVFP4 weight+activation**
in the sweep: rather than reporting a delta, NVIDIA documents partial-quantization recipes
(`nvfp4_mlp_only`, `nvfp4_omlp_only`) as the accuracy lever. The footnote's own link target is a
**GeForce RTX 50-series blog** (consumer Blackwell), while the requirement footnote (A.21) says only
"Blackwell GPUs" — the consumer/datacenter split is never stated in text.

### A.23 | CLOSED
WHO: NVIDIA / Model Optimizer maintainers
ARTIFACT: Model-Optimizer docs site — "Quantization Formats / Supported Model Formats" table (Linux section)
URL: https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html
STATUS: published (live docs site)
QUOTE: " FP4

 Per-Block FP4 Weight & Activations

 GPUs: Blackwell and Later

 PyTorch
 
 TensorRT, TensorRT-LLM"
SECOND QUOTE (same table, FP8 row): " FP8

 Per-Tensor FP8 Weight & Activations

 GPUs: Ada and Later

 PyTorch, ONNX*
 
 TensorRT*, TensorRT-LLM"
THIRD QUOTE (same table, W4A8 row): " W4A8 (INT4 Weights, FP8 Activations)

 Block-wise INT4 Weights, Per-Tensor FP8 Activations

 Uses AWQ Algorithm

 GPUs: Ada and Later"
FOURTH QUOTE (same table, W4A16 row): " W4A16 (INT4 Weights Only)

 Block-wise INT4 Weights, F16 Activations

 Uses AWQ Algorithm

 GPUs: Ampere and Later"
EVIDENCE: READ BODY
NOTE: **"GPUs: Blackwell and Later"** for per-block FP4 weight+activations is the single most
sm120-permissive hardware gate found in the sweep — "and Later" is the only wording that plainly
admits anything after Blackwell. Contrast with LLM Compressor's `10.0 (Blackwell)` (A.4) and
torchao's `SM100+ (Blackwell)` (A.11). W4A8 (INT4 weights + FP8 activations) is gated at
**"GPUs: Ada and Later"**, which sm120 also clears.

### A.24 | CLOSED
WHO: NVIDIA / Model Optimizer maintainers
ARTIFACT: Model-Optimizer docs — "PyTorch Quantization" guide, `auto_quantize` and QAT sections
URL: https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html
STATUS: published (live docs site)
QUOTE: "We recommend QAT for 10% of the original training epochs. For LLMs, we find that QAT fine-tuning for even
less than 1% of the original pre-training duration is often sufficient to recover the model quality."
SECOND QUOTE (same page, auto_quantize): "You may specify a effective_bits constraint such as 4.8 for mixed precision quantization using NVFP4_DEFAULT_CFG & FP8_DEFAULT_CFG .
AutoQuantize will automatically quantize highly sensitive layers in FP8_DEFAULT_CFG while keeping less sensitive layers in NVFP4_DEFAULT_CFG 
(and even skip quantization for any extremely sensitive layers) so that
the the final mixed precision quantized model has an effective quantized bits of 4.8.
This model would give a better accuracy than the model quantized with vanilla NVFP4_DEFAULT_CFG since
the more aggressive NVFP4_DEFAULT_CFG quantization was not applied for the highly sensitive layers."
THIRD QUOTE (same page, PTQ calibration): "PTQ can be achieved with simple calibration on a small set of training or evaluation data (typically 128-512 samples) after converting a regular PyTorch model to a quantized model."
EVIDENCE: READ BODY
NOTE: This is the strongest "accuracy recovery requires ..." text found: NVIDIA's stated recovery
levers are (a) mixed NVFP4+FP8 at `effective_bits ≈ 4.8`, and (b) QAT for <1% of pre-training.
The words "accuracy penalty" / "recover" appear — but **no numeric delta is given on this page.**

### A.25 | CLOSED
WHO: NVIDIA / Model Optimizer maintainers
ARTIFACT: Model-Optimizer docs — `examples/llm_eval/README.md` (evaluation harness entry points; MX format coverage note)
URL: https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/examples/llm_eval/README.md
STATUS: published (repo `main`)
QUOTE: "> **_NOTE:_** `MXFP8_DEFAULT_CFG` is one the [OCP Microscaling Formats (MX Formats)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) family which defines a set of block-wise dynamic quantization formats. The specifications can be found in the [official documentation](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). Currently we support all MX formats for simulated quantization, including `MXFP8 (E5M2, E4M3), MXFP6 (E3M2, E2M3), MXFP4, MXINT8`. However, only `MXFP8 (E4M3)` is in our example configurations, users can create their own configurations for other MX formats by simply modifying the `num_bits` field in the `MXFP8_DEFAULT_CFG`."
SECOND QUOTE (same page, the quant-cfg selector): "# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG|MXFP8_DEFAULT_CFG]"
THIRD QUOTE (same page, MMLU description): "[Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300). A score (0-1, higher is better) will be printed at the end of the benchmark."
EVIDENCE: READ BODY
NOTE: Decisive for the "does NVIDIA publish accuracy numbers" question: this is Model-Optimizer's
dedicated accuracy-evaluation README, it documents MMLU / gsm8k / simple-evals / lm-eval-harness
entry points **for exactly these quant configs**, and it contains **no accuracy table at all** — it
prints scores at run time and never tabulates them. It also confirms NVIDIA carries **no MXFP4 W4A4
example config** (only `MXFP8 (E4M3)`), and does not carry a plain NVFP4-vs-BF16 comparison.

---

## Section G — vLLM core (not one of the six requested sources; included because it is the documented deployment target for LLM Compressor output and it is the only place `sm120` and `SM121` appear together)

### A.26 | CLOSED
WHO: vLLM project maintainers
ARTIFACT: vLLM docs — "b12x Linear and MoE Backends" (feature page, `--linear-backend` / `--moe-backend` flags)
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/b12x.md
STATUS: published (repo `main`; the page documents the `vllm[b12x]` optional extra)
QUOTE: "[b12x](https://pypi.org/project/b12x/) provides optional CUDA kernels for
NVIDIA SM120 and SM121 GPUs. Install the dependency with:

```bash
uv pip install "vllm[b12x]"
```"
SECOND QUOTE (same page, Supported Configurations table): "| Backend | Supported configurations |
| ------- | ------------------------ |
| Linear | Per-tensor FP8, 128x128 block FP8, MXFP8, NVFP4, and MXFP4 |
| MoE | Tensor-parallel MXFP4 weights with BF16 or MXFP8 activations; NVFP4 weights with BF16, NVFP4, or MXFP8 activations |"
THIRD QUOTE (same page): "b12x uses MXFP8 activations by default for MXFP4 MoE and the checkpoint's
activation format for NVFP4 MoE. MXFP4 falls back to BF16 when its A8 path does
not support the model configuration. Set `VLLM_B12X_MOE_FP4_FORCE_A16=1` to
force BF16 activations for either FP4 weight format."
FOURTH QUOTE (same page): "Dense W4A16 layers are not handled by b12x and continue to use another
compatible backend such as Marlin. The b12x MoE backend does not support expert
parallelism, expert maps, EXL3, or NF3."
FIFTH QUOTE (same page): "b12x linear kernels participate in automatic selection after established
optimized backends and before emulation."
EVIDENCE: READ BODY
NOTE: This is the **only artifact in the sweep whose stated GPU scope is exactly `SM120 and SM121`**,
and it enumerates weight+activation support concretely: **per-tensor FP8, 128x128 block FP8, MXFP8,
NVFP4, MXFP4** for Linear. Three constraints matter for a Qwen3-4B dense target: (1) the MoE row is
irrelevant to a dense model; (2) "Dense W4A16 layers are not handled by b12x" means the W4A16 path
falls to Marlin; (3) MXFP4 has an **A8 path that may not support the model configuration** and then
falls back to BF16 activations. The doc never states accuracy for any of these.

### A.27 | CLOSED
WHO: vLLM project maintainers
ARTIFACT: vLLM docs — "NVIDIA Model Optimizer", note on NVFP4 GEMM kernel auto-selection and the documented no-native-FP4 fallback
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/modelopt.md
STATUS: published (repo `main`)
QUOTE: "!!! note
    For NVFP4 checkpoints, vLLM selects a GEMM kernel automatically at load
    time from the backends available on the current platform (CUTLASS,
    FlashInfer, Marlin, and others). On GPUs without a supported native FP4
    GEMM kernel, vLLM falls back to weight-only (W4A16) execution via Marlin
    and logs a warning; this may reduce throughput for compute-heavy
    workloads. Use `--linear-backend` to override the automatic selection
    (this replaces the deprecated `VLLM_NVFP4_GEMM_BACKEND` environment
    variable). Values relevant to NVFP4 include `cutlass`,
    `flashinfer_cutlass`, `flashinfer_cutedsl`, `flashinfer_trtllm`,
    `flashinfer_cudnn`, and `marlin`; see `KernelConfig` on the
    [Engine Arguments](../../configuration/engine_args.md) page and shown by
    `vllm serve --help=KernelConfig`. For `W4A16_NVFP4`, `auto` currently
    selects Marlin. Models with BF16 activations can explicitly select the
    FlashInfer CuTe-DSL backend with `--linear-backend flashinfer_cutedsl`."
SECOND QUOTE (same page, the immediately following note — the only place in the sweep that uses the term "SM100-family"): "!!! note
    For models quantized to MXFP8 with BF16 activations on SM100-family GPUs,
    use `--linear-backend flashinfer_trtllm` to select FlashInfer's TensorRT-LLM
    GEMM backend."
THIRD QUOTE (same page, supported ModelOpt checkpoint formats — note MXFP8 is listed but MXFP4 W4A4 is not): "vLLM detects ModelOpt checkpoints via `hf_quant_config.json` and supports the
following `quantization.quant_algo` values:

- `FP8`: per-tensor weight scale (+ optional static activation scale).
- `FP8_PER_CHANNEL_PER_TOKEN`: per-channel weight scale and dynamic per-token activation quantization.
- `FP8_PB_WO` (ModelOpt may emit `fp8_pb_wo`): block-scaled FP8 weight-only (typically 128×128 blocks).
- `NVFP4`: ModelOpt NVFP4 checkpoints (use `quantization="modelopt_fp4"`).
- `W4A16_NVFP4`: weight-only ModelOpt NVFP4 checkpoints with 16-bit
  activations (use `quantization="modelopt_fp4"`).
- `MXFP8`: ModelOpt MXFP8 checkpoints (use `quantization="modelopt_mxfp8"`)."
EVIDENCE: READ BODY
NOTE: This is the **only documentation in the sweep that describes the runtime behaviour when a GPU
lacks a native FP4 GEMM kernel**: a silent, warning-logged downgrade from NVFP4 W4A4 to **W4A16 via
Marlin** — i.e. activations stop being quantized. The doc does **not** enumerate which GPUs "without a
supported native FP4 GEMM kernel" are, does not name sm120 in that set, and states no accuracy
consequence for the fallback ("may reduce throughput" is a throughput statement only). It is also the
only place the term **"SM100-family"** appears, which is the family predicate the target rig fails
(`torch.cuda.is_device_capability_family(100)` is FALSE on sm120).

### B.13 | OPEN  (vLLM's core quant-hardware matrix has no Blackwell column at all)
WHO IS STILL ASKING: vLLM maintainers — the compatibility chart predates/omits Blackwell
ARTIFACT: vLLM docs — `docs/features/quantization/README.md`, section "Supported Hardware" (compatibility chart) and the note beneath it
URL: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/README.md
STATUS: published (repo `main`)
QUOTE: "## Supported Hardware

The table below shows the compatibility of various quantization implementations with different hardware platforms in vLLM:"
SECOND QUOTE (same page, the table header and the FP8/W8A8 + INT8 rows): "| Implementation            | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | Arm CPU |
| ------------------------- | ----- | ------ | ------ | --- | ------ | ------- | --------- | ------- | ------- |
| AWQ                       | ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| GPTQ                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| Marlin (GPTQ/AWQ/FP8/FP4) | ❌    | ✅︎*    | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| llm-compressor INT8 (W8A8)| ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ✅︎      | ✅︎      |
| llm-compressor INT8 (W4A8)| ❌    | ❌     | ❌     | ❌  | ❌     | ❌      | ❌        | ❌      | ✅︎      |
| llm-compressor FP8 (W8A8) | ❌    | ❌     | ❌     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |"
THIRD QUOTE (same page, the legend): "- Volta refers to SM 7.0, Turing to SM 7.5, Ampere to SM 8.0/8.6, Ada to SM 8.9, and Hopper to SM 9.0.
- ✅︎ indicates that the quantization method is supported on the specified hardware.
- ❌ indicates that the quantization method is not supported on the specified hardware."
FOURTH QUOTE (same page, the escape hatch): "!!! note
    This compatibility chart is subject to change as vLLM continues to evolve and expand its support for different hardware platforms and quantization methods.

    For the most up-to-date information on hardware support and quantization methods, please refer to [vllm/model_executor/layers/quantization](../../../vllm/model_executor/layers/quantization) or consult with the vLLM development team."
EVIDENCE: READ BODY
WHAT IS MISSING: The legend enumerates SM 7.0 / 7.5 / 8.0/8.6 / 8.9 / 9.0 and **stops at Hopper**.
There is no Blackwell column, no SM100/SM120/SM121 entry anywhere in the table, and the reader is
referred to source code. Note that `llm-compressor INT8 (W4A8)` is ❌ on every GPU column and ✅ only
on Arm CPU — consistent with A.4's `W4AINT8 — (Arm CPU)` row.

---

## ACCURACY TABLES FOUND (summary of what is quotable, and what is not)

| Source | Scheme(s) with a published number | Baseline published? | Model | Hardware named |
| --- | --- | --- | --- | --- |
| torchao docs (A.12) | fp8 rowwise W8A8, int8 W8A8, mxfp8 W8A8, nvfp4 W4A4 | **YES (bfloat16 row)** | Llama-3.1-8B | H100 / B200 (repro cmds only) |
| LLM Compressor repo (A.7) | FP8_DYNAMIC, INT8 W8A8, W4A16 GPTQ | **no** | Llama-3-8B-Instruct | none stated |
| vLLM LMC fp8 page (A.2) | FP8_DYNAMIC | **no** | Llama-3-8B-Instruct | none stated |
| TensorRT-LLM (A.8, A.9, A.10) | none — matrices are Y/. only | no | — | sm120 / sm100-103 / Hopper / Ada / Ampere |
| SGLang (A.16–A.18) | none | no | — | SM90/SM100/SM103/SM120/SM121 (backends only) |
| bitsandbytes (A.19) | none | no | — | SM60+/SM75+/SM121 |
| AutoAWQ (A.20) | none | no | — | CC 7.5+ |
| GPTQModel (B.11) | none | no | — | Turing+/sm_75+/≥sm100 |
| Model-Optimizer (A.21–A.25) | none | no | — | "Blackwell and Later", "Blackwell GPUs" |
| vLLM core (A.26, A.27) | none | no | — | SM120/SM121, "SM100-family" |

**Net position for the target rig.** Exactly one doc source publishes a BF16-referenced accuracy
delta for weight+activation quantization (torchao, on Llama-3.1-8B, on H100/B200). Zero sources
publish a Qwen3-4B number. Zero sources publish an sm120-measured number for any scheme. Only three
sources name `sm120`/`SM120` in a weight+activation context at all — TensorRT-LLM (A.8, a Y/`.` matrix
with no numbers), SGLang (A.17/A.18, GEMM-backend tables listing SM120) and vLLM core (A.26, b12x) —
and **none of the three attaches an accuracy or a speed figure to sm120**. Every numeric accuracy
delta and every numeric speedup in the entire sweep was measured on H100, B200, or an unnamed GPU.

---

## QUERIES

Every URL issued with `fetch.py` (mode in parentheses), in order. No `web_search` queries were used;
no `api.github.com` calls were made; `web_fetch` was not used.

### LLM Compressor
1. (get) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/README.md
2. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w8a8_fp8/README.md
3. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w8a8_int8/README.md
4. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a16/README.md
5. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a4_fp4/README.md
6. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w8a8_mxfp8/README.md — 404
7. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w8a8_fp8/qwen3_example.py — 404
8. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a4_fp4/qwen3_example.py — 404
9. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a16_fp4/README.md — 404
10. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a8_int8/README.md — 404
11. (raw) https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a16/qwen3_example.py — 404
12. (get) https://github.com/vllm-project/llm-compressor/tree/main/docs
13. (get) https://github.com/vllm-project/llm-compressor/tree/main/examples
14. (get) https://docs.vllm.ai/projects/llm-compressor/en/latest/
15. (get) https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/
16. (get) https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-model/
17. (get) https://docs.vllm.ai/projects/llm-compressor/en/latest/schemes/nvfp4/ — not a real path (nav-only response)
18. (get) https://docs.vllm.ai/projects/llm-compressor/en/latest/schemes/fp8/ — not a real path
19. (get) https://docs.vllm.ai/projects/llm-compressor/en/latest/schemes/w4a16/ — not a real path
20. (get) https://docs.vllm.ai/projects/llm-compressor/en/latest/faq/faq/

### TensorRT-LLM
21. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/quantization.md
22. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/reference/precision.md — 404
23. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/README.md
24. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/deployment-guide/deployment-guide-for-onnx.md — 404
25. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/installation/supported-hardware.md — 404
26. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/reference/supported-hardware.md — 404
27. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/overview.md
28. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/quantization-int4.md — 404
29. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/quantization/README.md
30. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog5_Disentangling_Reuse_KV_Cache.md — 404
31. (get) https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source
32. (get) https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/features
33. (get) https://nvidia.github.io/TensorRT-LLM/reference/precision.html

### torchao
34. (raw) https://raw.githubusercontent.com/pytorch/ao/main/README.md
35. (raw) https://raw.githubusercontent.com/pytorch/ao/main/torchao/quantization/README.md — redirect stub (C.2)
36. (raw) https://raw.githubusercontent.com/pytorch/ao/main/torchao/prototype/mx_formats/README.md — redirect stub (C.2)
37. (raw) https://raw.githubusercontent.com/pytorch/ao/main/torchao/prototype/README.md
38. (get) https://github.com/pytorch/ao/tree/main/torchao/quantization
39. (raw) https://raw.githubusercontent.com/pytorch/ao/main/docs/source/workflows/inference.rst — 404
40. (raw) https://raw.githubusercontent.com/pytorch/ao/main/docs/source/workflows/training.rst — 404
41. (raw) https://raw.githubusercontent.com/pytorch/ao/main/docs/source/index.rst
42. (get) https://pytorch.org/ao/main/workflows/inference.html

### SGLang
43. (raw) https://raw.githubusercontent.com/sgl-project/sglang/main/docs/advanced_features/quantization.md — 404
44. (raw) https://raw.githubusercontent.com/sgl-project/sglang/main/docs/advanced_features/quantize_and_deploy.md — empty
45. (raw) https://raw.githubusercontent.com/sgl-project/sglang/main/docs/platforms/amd_gpu.md
46. (raw) https://raw.githubusercontent.com/sgl-project/sglang/main/README.md
47. (get) https://github.com/sgl-project/sglang/tree/main/docs
48. (get) https://docs.sglang.io/advanced_features/quantization.html
49. (get) https://docs.sglang.io/advanced_features/quantize_and_deploy.html — empty
50. (get) https://docs.sglang.io/index.html
51. (get) https://docs.sglang.io/llms.txt

### bitsandbytes / AutoAWQ / GPTQModel
52. (raw) https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/README.md
53. (raw) https://raw.githubusercontent.com/casper-hansen/AutoAWQ/main/README.md
54. (raw) https://raw.githubusercontent.com/ModelCloud/GPTQModel/main/README.md

### NVIDIA Model-Optimizer
55. (raw) https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/README.md
56. (raw) https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/examples/llm_ptq/README.md — 404
57. (raw) https://raw.githubusercontent.com/NVIDIA/TensorRT-Model-Optimizer/main/README.md (same content as 55)
58. (raw) https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/examples/hf_ptq/README.md
59. (raw) https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/docs/source/guides/_support_matrix.rst — 404
60. (raw) https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/CHANGELOG.rst
61. (raw) https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/examples/llm_eval/README.md
62. (raw) https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/examples/llm_ptq/recipes/README.md — 404
63. (raw) https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/modelopt/torch/quantization/config.py
64. (get) https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html
65. (get) https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html
66. (get) https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html

### vLLM core
67. (get) https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html
68. (get) https://docs.vllm.ai/en/latest/features/quantization/b12x.html
69. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/README.md
70. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/b12x.md
71. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/supported_hardware.md — 404
72. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor.md — 404
73. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/torchao.md
74. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor/README.md
75. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor/fp8.md
76. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor/int4.md
77. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor/int8_w4a8.md
78. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor/int8_w8a8.md
79. (raw) https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/modelopt.md
80. (get) https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w8a8.html
81. (get) https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8.html
82. (get) https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int4.html

Local greps issued (targeted, non-network):
- `grep -n -iE 'sm ?120|sm_120|12\.0|rtx|pro 6000|consumer|geforce|5090|sm121|sm_121'` over `ao_inference_site.txt` and torchao `README.md` — **0 substantive matches** (only `2.12.0.dev` version strings). Basis for B.8.
- `grep -rn -iE 'is_device_capability_family|compute_capability|get_device_capability|sm_?120|12, 0|\(12, 0\)|major == 12'` over all fetched LMC / TensorRT-LLM / torchao docs — **the only hit in the entire doc set was the `Blackwell(sm120)` row of the TensorRT-LLM hardware matrix (A.8)**. No documentation anywhere in the sweep references `is_device_capability_family`.
- `grep -n -iE 'blackwell|not supported|limitation|compute capability'` over all fetched vLLM LLM-Compressor quant pages — basis for A.1 / A.2 / A.3 / Section G.
- `grep -n -iE 'sm1[0-9]{2}|sm_1[0-9]{2}|blackwell|geforce|rtx|pro 6000|consumer|datacenter|compute capability|requires'` over Model-Optimizer `hf_ptq/README.md` and the MO support-matrix page — basis for A.21 / A.23 / B.12.

---

## DEAD ENDS

Explicit "the docs do not say this" statements. Each one is a place where a claim **could not** be
sourced and must not be inferred from this file.

1. **No documentation in the sweep publishes an accuracy delta for any scheme on Qwen3-4B.**
   The only accuracy tables found anywhere are on **Llama-3.1-8B** (torchao, A.12/A.13) and
   **Llama-3-8B-Instruct** (LLM Compressor, A.7). Qwen3-4B accuracy numbers do not exist in the docs.

2. **No documentation in the sweep publishes a BF16 baseline next to an LLM Compressor number.**
   torchao's table (A.12) is the *only* one with a bfloat16 row. The three LMC gsm8k scores
   (0.768 / 0.752 / 0.728) are therefore **not deltas** — LMC docs state no reference score, so no
   "accuracy recovery" percentage is quotable from LMC docs for FP8, INT8, or INT4.

3. **LLM Compressor docs state NO accuracy number for NVFP4, MXFP4, MXFP8, or W4A8 on any model.**
   The NVFP4 example README (B.5) ends before its evaluation step; the MXFP8 and W4A8-INT8 example
   READMEs 404 (C.1). Only FP8_DYNAMIC, INT8 W8A8 and W4A16 GPTQ have published LMC scores, and all
   three are on Llama-3-8B-Instruct.

4. **TensorRT-LLM docs state NO accuracy numbers for any quantization mode.** `features/quantization.md`
   is a Y/`.` support matrix only (A.8/A.9). `reference/precision.html` (B.7) is a Y/`.` matrix only.
   There is **no NVFP4-vs-MXFP4-vs-FP8-vs-BF16 accuracy table anywhere in TensorRT-LLM docs** for
   dense models. The docs assert only that FP8 has "minimal impact on model accuracy" (overview.md)
   without a number.

5. **No TensorRT-LLM doc explains the `Blackwell(sm120)` vs `Blackwell(sm100/103)` kernel difference.**
   The matrix silently gives sm120 `MXFP4 = Y` and sm100/103 `MXFP4 = Y` but withholds FP8 block
   scaling, FP8 rowwise, and all four W4A8/W4A16 AWQ/GPTQ cells from sm120. The docs never say *why*,
   never state which kernel serves NVFP4 on sm120, and never state an expected throughput ratio.

6. **torchao docs never mention sm120, RTX, GeForce, consumer Blackwell, "5090", or "PRO 6000".**
   Verified by grep over the full inference docs page and README (see Local greps). The `SM100+`
   gate on NVFP4 W4A4 and MX W8A8/W4A4 (A.11) is **undefined as to whether sm120 counts**.

7. **torchao docs publish no speedup for any weight+activation scheme on sm120/consumer Blackwell.**
   Both e2e tables (A.13) are B200 and H100 only. The `mxfp8` B200 cell and the `int4+float8_rowwise`
   H100 cell are literally `TODO(<issue>)` — the docs are **tracked-open for those two cells**:
   "TODO(https://github.com/pytorch/ao/issues/3549)" and "TODO(https://github.com/pytorch/ao/issues/3550)".
   (Recorded as a doc quote; the linked issues were NOT fetched — issues are out of scope for this sweep.)

8. **bitsandbytes documents no weight+activation quantization scheme.**
   Its README covers "8-bit optimizers", "LLM.int8()", and "QLoRA ... 4-bit" — i.e. weight-only /
   weight+16-bit-activation. There is no W8A8 or W4A4 scheme and no accuracy table. Its NVIDIA
   requirement is a floor ("SM60+ minimum / SM75+ recommended") with no sm120 statement. The only
   `SM12x` token in the entire sweep is the **arm64** row's `SM121`, which does not describe an
   x86-64 RTX PRO 6000 host.

9. **AutoAWQ is weight-only (W4A16) and publishes no weight+activation scheme and no accuracy table.**
   Its only numeric claims are speed/memory ratios ("speeds up models by 3x and reduces memory
   requirements by 3x compared to FP16") with no hardware named and no accuracy delta. Its
   requirement line is a floor (CC 7.5) with no sm120 mention.

10. **GPTQModel publishes no accuracy table and never writes `sm120`.**
    Its kernels are weight-only/W4A8-W4A16-oriented; its newest Blackwell kernel (Swordfish) is
    documented as `>= sm100`. No doc states which kernel serves a Qwen3-4B W4A16 checkpoint on sm120.

11. **SGLang publishes no accuracy number for any quantization method.**
    Its only accuracy-adjacent text is a warning to "validate via benchmarks post-quantization" and
    an AutoRound failure report ("Qwen2.5-VL-7B auto_round:auto_gptq format: Accuracy is close to
    zero"). No gsm8k/MMLU/perplexity table exists on the Quantization page.

12. **SGLang's method matrix has no sm90/sm100/sm120 row granularity for the core methods.**
    `fp8`, `mxfp4`, `w8a8_int8`, `w8a8_fp8`, `awq`, `gptq_marlin`, `compressed-tensors` all carry a
    single undifferentiated "Yes" (or "No") under "NVIDIA GPUs" (A.16). The only NVIDIA sub-gates are
    on `modelopt/modelopt_fp8` (Hopper/SM90+), `modelopt_fp4` (SM80-SM90 via Marlin; SM100+ native FP4),
    and `nvfp4_online` (Blackwell/SM100 or SM103).

13. **No documentation states whether the INT8 W8A8 `>= 10.0` exclusion (A.1) also kills W4A8 INT8.**
    vLLM's INT8 W4A8 page (int8_w4a8.md) was fetched and contains **no** Blackwell limitation warning
    of any kind; the core matrix (B.13) marks `llm-compressor INT8 (W4A8)` ❌ on every GPU column and
    ✅ only on Arm CPU. The docs do not resolve whether sm120 can run W4A8.

14. **No documentation from any source states that Qwen3-4B specifically is validated for any scheme.**
    TensorRT-LLM's matrix row is the family-level "Qwen-3" (A.9); Model-Optimizer's row is
    "QWen3, 3.5 MOE, Next" (A.21). Neither enumerates parameter counts, and neither is a dense-only
    claim — the MO row explicitly bundles MoE variants.

15. **NVIDIA Model-Optimizer publishes no accuracy table.**
    Its dedicated accuracy-evaluation README (A.25) documents the *harnesses* (MMLU, gsm8k,
    simple-evals, lm-eval-harness) for exactly the quant configs of interest and then reports nothing.
    The strongest accuracy statement in all of Model-Optimizer docs is qualitative: "demonstrates good
    accuracy compared with other 4-bit alternatives" (A.22) — a comparison with no numbers and no
    named alternatives.

16. **No doc in the sweep states a calibration-data minimum for W8A8-FP8, and the W4A16/W8A8-INT8
    "512 samples" figure is framed as a starting heuristic, not a requirement.**
    Verbatim, the strongest such text is: "512 samples is a good place to start (increase if accuracy
    drops)" (LLM Compressor w4a16 and w8a8_int8 example READMEs) and "Start with 512 samples for
    calibration data, and increase if accuracy drops" (vLLM int4.md Best Practices). By contrast NVFP4
    *does* state a calibration requirement (B.5: "We need some sample data to calibrate the global
    activation scales ... we use a sample size of 20"), and MXFP4/MXFP8 are documented as needing none
    (A.5: "no calibration data required if using RTN").

17. **No documentation anywhere in the sweep contains the string `is_device_capability_family`, nor
    any runtime capability-family check for sm120.** The only sm120 token found in the entire fetched
    corpus was the TensorRT-LLM matrix row `| Blackwell(sm120) ...` (A.8). This is the single largest
    documentation gap relative to the target rig.

18. **`web_search` was not used and no GitHub issue/PR pages were fetched**, per the task's
    documentation-only constraint. Where a doc pointed at an issue (e.g. torchao's `TODO(.../issues/3549)`),
    the issue was deliberately **not** opened; those cells are recorded as documented-open, not as
    resolved or unresolved by maintainers.

19. **No documentation states the accuracy cost of the documented NVFP4 → W4A16 Marlin fallback.**
    vLLM's ModelOpt page (A.27) documents that on a GPU without a native FP4 GEMM kernel the runtime
    silently downgrades to weight-only W4A16 via Marlin and warns — but it attaches only a throughput
    caveat ("this may reduce throughput for compute-heavy workloads") and **no accuracy statement**.
    No doc states which GPUs are in that fallback set, so no doc states whether an sm120 host lands
    there. Combined with DEAD END 3 (no LMC NVFP4 accuracy number at all), there is **no documented
    accuracy figure for either branch** of the NVFP4 path on a non-SM100 GPU.

20. **No documentation gives a per-scheme expected speedup for weight+activation quantization on
    sm120.** The only sm120-specific performance text found is qualitative: b12x "participate[s] in
    automatic selection after established optimized backends and before emulation" (A.26) and the
    SGLang auto-selection ordering (A.17/A.18). Every numeric speedup in the sweep is H100 or B200
    (torchao A.13, torchao README A.15).

21. **No documentation was found that states FP8 block-scaling *can* run on sm120**, and no doc
    explains the asymmetry in the TensorRT-LLM matrix (A.8) between `Blackwell(sm120)` and
    `Blackwell(sm100/103)` for FP8 block scaling, FP8 rowwise and all four W4A8/W4A16 AWQ/GPTQ cells.
    The only counter-signal is SGLang's `--fp8-gemm-backend` table listing `cutlass SM120` (A.17),
    which does not say whether that backend serves per-tensor or blockwise FP8.
