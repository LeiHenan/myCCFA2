# The sm89 / 24 GB / PCIe-only hardware-difference axis — what is newly possible, newly broken, newly expensive

**Date:** 2026-09-15
**Scope:** the hardware-difference axis ONLY. KV-cache quantization/management and speculative-decoding
*algorithms* are covered elsewhere in this repo and are deliberately **not** re-mapped here
(see §0 skip-list).
**Machine under study:** `ssh schoolserver` = 8× RTX 4090 24 GB, **sm89 (Ada, CC 8.9)**, driver **550.67
(CUDA 12.4)**, PCIe-only, no NVLink, 2 NUMA nodes, 192 CPU cores, ~1007 GB RAM. Shared box.

## How to read the evidence labels (non-negotiable in this repo)

| Label | Meaning |
|---|---|
| **RUN** | I executed this on `schoolserver` this session and am reporting the raw output. |
| **FETCH** | I retrieved the artifact this session; the quote is verbatim from it. |
| **READ-SRC** | Retrieved from a source file (raw.githubusercontent / a repo page) — verbatim. |
| **DERIVED** | My arithmetic on labelled inputs. Not a measurement, not a quote. |
| **SUBAGENT** | A delegated agent reports it; **it is NOT verified by me**. Named as such. |
| **UNVERIFIED** | I could not retrieve or reproduce it. Listed in §5. |

**Rule inherited from `notes/FILTER_3AXIS.md` §7:** a subagent's retrieval is not a conclusion. Anything
load-bearing below is either **RUN** or **FETCH**. Where I rely on **SUBAGENT** material I say so in the row.

---

## §0 Scope boundary — what the other maps already own (grepped before proposing anything)

Grepped `KV_CACHE_GAP_MAP.md` (3540 lines), `spec-decode-gap-map-2026-09-13.md` (1191 lines),
`INFERENCE_ACCEL_GAP_MAP.md` (2159 lines) for `sm89|sm_89|ada lovelace|4090`, and read each hit.

**Finding: every one of those three maps is scoped to a DIFFERENT rig** — one RTX PRO 6000 Blackwell
**96 GB, sm120**, driver 580, 208 cores:

- `KV_CACHE_GAP_MAP.md:4` — *"Target setup: ONE GPU (RTX PRO 6000 Blackwell 96 GB, sm120, driver 580) … no NVLink/InfiniBand/cluster"*
- `INFERENCE_ACCEL_GAP_MAP.md:4` — *"ONE GPU — RTX PRO 6000 Blackwell, 96 GB, sm120 … vLLM 0.29.0, SGLang 0.5.19"*

So the maps are a **skip-list for algorithms**, not for this axis. Total sm89/Ada content across all three:
4 + 3 + 9 hits, and **all** of them are incidental mentions of "RTX 4090" as somebody else's testbed
(e.g. `INFERENCE_ACCEL_GAP_MAP.md` C250 QServe, C302 Quest; `spec-decode-gap-map` D5.3 batch-invariance
validated on 4090/4080/3060; `KV_CACHE_GAP_MAP.md:430` a docs PR adding *"the CUDA SM89+ requirement for
`fp8_per_token_head`"*). **None of them is a hardware-axis map.**

**Explicitly skipped as already-owned:** KV quant/offload/hierarchical tiering; spec-decode
algorithms/drafters/verification; sparse attention; scheduler/prefix-cache policy. If a candidate below
touches one of those, it is only because the *binding constraint is the hardware*, and I say so.

---

## §1 What is ruled OUT on sm89 — the tombstone half

Ordered by how load-bearing the gate is. "Hard gate" = an arch/ISA/driver boundary that software cannot
cross; "soft default" = a default that a flag or a different kernel can change.

### 1.1 Block-scaled FP8 (and FP4/FP6) — HARD GATE, and it is the sharpest one

This is the single most consequential tombstone and it has an exact numeric boundary.

**RUN** — I asked the compiled vLLM 0.26.0 extension directly on this machine:

```
$ python -c "...ops.cutlass_scaled_mm_supports_block_fp8(cc) for cc in [...]"
device capability: DeviceCapability(major=8, minor=9) int 89

cutlass_scaled_mm_supports_fp8:        80=False 86=False 89=True  90=True 100=True 120=True
cutlass_scaled_mm_supports_block_fp8:  80=False 86=False 89=False 90=True 100=True 120=True
cutlass_group_gemm_supported:          80=False 86=False 89=False 90=True 100=True 120=False
```

**RUN** — and what that does to kernel selection on sm89 (`init_fp8_linear_kernel`, real invocation):

| kernel in `_POSSIBLE_FP8_BLOCK_KERNELS[CUDA]` | can implement on **sm89**? | reason string returned by the engine |
|---|---|---|
| `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` | **False** | `base is not supported due to Flash…` |
| `DeepGemmFp8BlockScaledMMKernel` | **False** | `Currently, only Hopper and Blackwell GPUs are sup…` |
| `CutlassFp8BlockScaledMMKernel` | **False** | `The device compute capability of89 is not supporte…` |
| `MarlinFP8ScaledMMLinearKernel` | **False** | `To apply FP8 Marlin on high-capability GPUs, pleas…` |
| `TritonFp8BlockScaledMMKernel` | **True** | *(empty)* |

**READ-SRC (verbatim, the gate itself)** — `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:240-247`,
in the shipped tree at `/home/user/myCCFA/.venv/lib/python3.11/site-packages/vllm/`:

```python
    @classmethod
    def is_supported(cls, compute_capability=None):
        if not CUTLASS_BLOCK_FP8_SUPPORTED:
            return (
                False,
                "The device compute capability of"
                f"{compute_capability} is not supported.",
            )
        return True, None
```

**READ-SRC (verbatim)** — `.../scaled_mm/deep_gemm.py:47-52`:

```python
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "DeepGEMM is only supported on cuda platform"
        if not is_deep_gemm_supported():
            return False, "Currently, only Hopper and Blackwell GPUs are supported."
        return True, None
```

**SUBAGENT (not verified by me)** — the underlying ISA reason, from PTX ISA 9.4, `mma` → Target ISA Notes:
*"Support for `.kind`, `.block_scale`, `.scale_vec_size` qualifier requires `sm_120a`"*, and
`wgmma.mma_async` *"Requires `sm_90a`."* Reported from the PTX ISA doc fetched by the gating subagent;
full report at `research/sm89-gating-inventory.md`. If this is right, block scaling is not merely
"sm90-only in one library" but is architected away on Ada.

**The decisive consequence (this is the part that is new):** block-FP8 is not *blocked* on sm89 — it
silently **falls back to a Triton kernel**. That means a DeepSeek/Qwen3-block-FP8 checkpoint **loads and
runs** on this box, with no error, on the one kernel nobody chose for performance. That is the classic
"condition C changes the answer" shape: on sm90 the same checkpoint gets CUTLASS/DeepGEMM; here it gets
Triton, and **no artifact I found measures the gap.**

### 1.2 FlashAttention-3 — HARD GATE

**FETCH (verbatim)** — `https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md`:
> *"FlashAttention-3 is optimized for Hopper GPUs (e.g. H100)."*

and the install path is literally a `hopper/` subdirectory (`cd hopper; python setup.py install`).
The same README confirms FA2 **does** cover this box:
> *"FlashAttention-2 with CUDA currently supports: 1. Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100)."*

**SUBAGENT (not verified by me):** repo owner `tridao`, flash-attention issue #1121, answering exactly
"does FA3 support 4090": *"4090 is Ada, not Hopper"*; and vLLM `fa_utils.py` only selects FA3 when
`device_capability.major == 9`. **Consequence to carry:** FA3/FA4-gated features are unavailable, which
is why the subagent reports vLLM has no FP8-KV path on sm89 (`"FP8 KV cache requires FA3 on SM90 or
FA4 on SM100"`). Treat as SUBAGENT until re-fetched.

### 1.3 TMA / `cp.async.bulk` / thread-block clusters / DSMEM — HARD GATE

**SUBAGENT (not verified by me)**, CUDA C++ Programming Guide 12.4 §7.29 and PTX ISA 9.4 Target ISA Notes:
> *"To offload these computations, Compute Capability 9.0 introduces Tensor Memory Acces (TMA)."*
> `cp.async.bulk`: *"Requires sm_90 or higher."* — while `cp.async`: *"Requires sm_80 or higher."*

So Ada has the **Ampere-level** async-copy path and none of the Hopper machinery. This is the reason
CUTLASS 3.x collectives have no sm89 builder (SUBAGENT: no `sm89_*` in
`include/cutlass/gemm/collective/builders/`).

### 1.4 FP8 per-tensor W8A8 — **NOT** ruled out; this is a live path with a driver-shaped floor

Worth stating plainly because the received wisdom ("Ada has no usable FP8") is **false**:
- **RUN:** `cutlass_scaled_mm_supports_fp8(89) = True` — *and* `CutlassFP8ScaledMMLinearKernel` returns
  **True** for per-tensor on sm89 while `FlashInferFP8ScaledMMLinearKernel` returns **False**
  (`requires compute capability 100 and above.`).
- **SUBAGENT (not verified by me):** the gating condition is `CUDA_VERSION >= 12040` for sm89
  (`scaled_mm_entry.cu`), i.e. **CUDA 12.4 is the floor for FP8 on Ada.** Our driver is **550.67 = CUDA 12.4**
  — exactly on the boundary, verified by **RUN** (`nvidia-smi`: `Driver Version: 550.67  CUDA Version: 12.4`).

**A driver-shaped double bind (DERIVED from RUN facts, and this one is ours specifically):**
driver 550.67 is *simultaneously* (a) the **minimum** that enables CUTLASS per-tensor FP8 on sm89, and
(b) the **maximum** that pins us to vLLM ≤ 0.26.0. One version down loses FP8; one version up loses the
engine. That coincidence is a property of *this box*, not of Ada in general.

### 1.5 Tombstone summary table

| # | Artifact | URL / source | Verified state | Verbatim quote | Gate |
|---|---|---|---|---|---|
| 1 | Block-scaled FP8 / MXFP8 on sm89 | **RUN** `cutlass_scaled_mm_supports_block_fp8(89)` in vLLM 0.26.0 | False (89); True (90) | engine reason string: `"The device compute capability of89 is not supported."` | **HARD** |
| 2 | DeepGEMM on sm89 | **RUN** `DeepGemm...is_supported` | False | `"Currently, only Hopper and Blackwell GPUs are supported."` | **HARD** |
| 3 | CUTLASS FP8 MoE group GEMM on sm89 | **RUN** `cutlass_group_gemm_supported(89)` | False | *(function returned False; reason not quoted)* | **HARD** |
| 4 | FlashInfer FP8 linear on sm89 | **RUN** `FlashInferFP8...is_supported` | False | `"requires compute capability 100 and above."` | **HARD** |
| 5 | FlashAttention-3 on sm89 | **FETCH** FA README | Hopper-only | `"FlashAttention-3 is optimized for Hopper GPUs (e.g. H100)."` | **HARD** |
| 6 | TMA / `cp.async.bulk` / clusters | SUBAGENT: CUDA Guide 12.4 §7.29; PTX ISA 9.4 | min sm_90 | `"Compute Capability 9.0 introduces Tensor Memory Acces (TMA)."` | **HARD** |
| 7 | `wgmma` | SUBAGENT: PTX ISA 9.4 | sm_90a only | `"Requires sm_90a ."` | **HARD** |
| 8 | FP4/FP6 MMA | SUBAGENT: PTX ISA 9.4 | sm_120a only | `".e3m2 , .e2m3 and .e2m1 … requires sm_120a"` | **HARD** |
| 9 | Per-tensor FP8 W8A8 | **RUN** `cutlass_scaled_mm_supports_fp8(89)` | **True** (CUDA ≥ 12.4) | — | *available* |
| 10 | Block-FP8 *execution* on sm89 | **RUN** `TritonFp8BlockScaledMMKernel` can-implement | **True** | *(empty reason string)* | **SOFT** (silent fallback) |

---

## §2 What is NEWLY RELEVANT here (and a non-issue on H100/B200)

Six findings, all **RUN** unless marked. §2.2–§2.5 are the ones I did not expect and cannot find
discussed anywhere.

### 2.1 The standard topology tool sees nothing here — and my attempt to see more failed

**RUN** — `nvidia-smi topo -m` reports **`SYS` for all 28 GPU pairs**, and
`torch.cuda.can_device_access_peer` returns **False for every pair**. On that evidence the box looks
like a flat, uniformly-bad PCIe fabric.

**RUN** — my first attempt to see more produced a striking-looking table. **Read it as a measurement bug,
not a result** (explanation immediately below the table):

| pair group | one-way copy BW as *measured* (INVALID — see caveat) |
|---|---|
| GPU **0** → any other GPU | 21.6 – 21.8 GB/s |
| GPU **1** → 2 | 118.5 GB/s |
| GPU **1** → 3 / 4 / 6 | 135.0 / 168.2 / 168.3 GB/s |
| GPUs among **{2,3,4,5,6,7}** | 136.4 – 178.1 GB/s |
| 0 ↔ 1 | 21.7 GB/s |

Apparent spread **21.6 → 178.1 GB/s (8.2×)**, invisible to `nvidia-smi topo` and to
`can_device_access_peer`. **I am not claiming this: the fast half is physically impossible and is
withdrawn below.** The numbers are reported only so the failed probe is on the record.

**NUMA layout (RUN, from `/sys/bus/pci/devices/*/numa_node`):** GPUs **0,1,2,3 → node 0**;
GPUs **4,5,6,7 → node 1**; all `speed=16.0 GT/s PCIe width=16` (Gen4 ×16); NUMA distance
**10 (intra) vs 21 (inter)**. So the fast set **{2,3,4,5,6,7} straddles both NUMA nodes**, and the slow
GPUs 0 and 1 are exactly the two cards a co-tenant is using (`nvidia-smi` RUN: GPU0 5745 MiB / 100 % util;
GPU1 23017 MiB; GPU7 23133 MiB).

**Honest caveat — do not over-read this row, and I now believe the *absolute* numbers are an artifact.**
Consumer Ada is **PCIe Gen4 ×16 ⇒ ~32 GB/s per direction is the physical ceiling** (RUN: `nvidia-smi
--query-gpu=pcie.link.gen.max` = `4` for all eight GPUs; SUBAGENT: the Ada whitepaper's "PCI Express
Interface" row is `Gen3 Gen4 Gen4` for 2080 Ti / 3090 Ti / **4090**). **The 136–178 GB/s figures exceed that
ceiling by 4–5.6×, so they cannot be genuine inter-GPU PCIe traffic.** The likely methodological cause, which
I identified only after the environment broke and could not re-test: my probe called
`torch.cuda.synchronize()` (no argument), which synchronises the **current** device's stream, not both
devices' — so for a `b.copy_(a)` issued on device *dst*, part of the transfer may not be inside the timed
window, inflating the apparent rate. **The correct measurement is per-device event pairs on *both* devices.**
Consequences, stated plainly:

- The **slow mode (21.6–21.8 GB/s, GPU 0)** is in the right band for a Gen4 ×16 device↔device copy and is
  corroborated by an independent practitioner figure for granted-P2P on a 4×4090 box (**19–22 GB/s**,
  SUBAGENT) and by my own pinned-H2D measurement (25.56 GB/s, RUN). **Treat the slow mode as approximately
  real.**
- The **fast mode is not established at all.** Its *bimodality* (GPU 0 always slow, {2..7} always fast)
  reproduced across three probe runs with two different buffer sizes, so the **structure** is probably
  real; the **magnitude is an artifact of under-synchronisation** and must not be quoted.
- This is therefore a **measurement bug, not a finding**, and I am recording it as such rather than
  presenting 8.2× as a result. Any candidate built on it (C3) is correspondingly weakened.

### 2.2 NCCL all-reduce is ~14.5 GB/s and **does not care about the socket boundary**

**RUN**, `torch.distributed` + NCCL, 10 iters min, 1–128 MB fp32, busBW = 2(n−1)/n·S/t:

| GPUs | 1 MB | 16 MB | 64 MB | 128 MB |
|---|---|---|---|---|
| `[2,3]` (same NUMA node) | 9.04 GB/s | 14.07 | 14.33 | **14.50** |
| `[2,4]` (**cross** NUMA node) | 9.20 GB/s | 13.76 | 13.93 | **14.43** |
| `[2,3,4,5]` (TP=4, spans nodes) | 9.26 GB/s | 15.65 | 16.32 | **16.44** |
| `[0,2,3,4,5,6,7]` (TP=8) | 9.15 GB/s | 13.45 | **7.11** | **4.34** |

Two things fall out:

- **Same-socket ≈ cross-socket (14.50 vs 14.43 GB/s).** Whatever transport NCCL selects, it is
  **insensitive to the NUMA/socket boundary.** This is the *solid* half of this section: it was measured
  with `torch.distributed` barrier semantics (not the broken probe of §2.1) and it agrees with
  `nvidia-smi topo`'s flat `SYS` reading — both say NCCL is staging through the host regardless of which
  socket the peer sits on. So the earlier "8.2× asymmetry NCCL ignores" claim is **withdrawn**: there is no
  evidence NCCL is ignoring anything.
- **TP=8 *degrades* at large messages** — busBW falls 9.15 → 4.34 GB/s and per-call time rises
  0.183 ms → 49.4 ms as the message grows 128×. That is *much* worse than the 4× data growth; it is a
  large-message collapse specific to the 8-rank ring, and GPU 0 (the tenant-occupied card) is in that ring.
  **This one is unaffected by the §2.1 bug** — it comes from the NCCL run, not the copy probe.

### 2.3 There is a ~0.10 ms fixed cost per all-reduce, and it is the decode-relevant term

**RUN / DERIVED** — at 1 MB the per-call time is **0.106 ms at TP=2, 0.158 ms at TP=4, 0.183 ms at TP=8**,
and at TP=2/TP=4 it is nearly independent of message size at the small end (1 MB → 16 MB moves it only
0.106 → 1.135 ms, i.e. the fixed term dominates below ~1 MB). Since a decode step's all-reduce messages are
tiny (a few kB), the *bandwidth* term is irrelevant for decode, and the **fixed ~0.11 ms at TP=2** is what a
token pays, once per all-reduce, per layer. Note the direction: the per-call cost **rises** with rank count
(0.106 → 0.183 ms) rather than trading against it, so the total is hurt twice as TP grows — more calls *and*
a costlier call.

**DERIVED (arithmetic on the RUN latencies; inputs labelled):** for a 40-layer dense transformer with the
usual 2 all-reduces per layer (attention output + MLP down), at TP=2 that is 80 × 0.107 ms ≈ **8.6 ms/token
of pure collective latency**, i.e. a **~116 tok/s single-stream ceiling** imposed by the interconnect alone,
before any compute. This is the "more cards is not a free win" claim made quantitative — and note the
mechanism is *latency*, not bandwidth, which is the opposite of the usual PCIe story.

### 2.4 24 GB forces sharding *for the KV cache*, not for the weights

**DERIVED (weights: standard bf16 arithmetic; KV: Qwen3-14B-class config, 40 layers × 8 KV heads × 128 dim,
2 (K+V) × 2 bytes):**
- 14B bf16 weights ≈ **28 GB** → does not fit one 24 GB card (consistent with the stated framing).
- KV per token ≈ **164 kB/token** → a **128k-token** context needs ≈ **21 GB** of KV.
So at long context the KV cache **alone** approaches a full card, *on top of* the weights. The binding
constraint on context length here is therefore **aggregate VRAM ÷ KV-per-token**, and the way you buy it
is TP degree — which is precisely the knob §2.2/§2.3 show to be latency-taxed. The hardware axis couples
"how much context" to "how much interconnect", on a fabric that is the weakest part of the box.

### 2.5 Non-issues here that are the *whole problem* on H100/B200

For contrast, so the axis is stated in both directions (all **DERIVED** from the RUN facts above):
- **HBM bandwidth is not the constraint at decode.** 1008 GB/s/card × 8 = **8.06 TB/s** aggregate vs a
  single H100's 3.35 TB/s. This box has *more* aggregate HBM bandwidth than a DGX H100 node — the deficit
  is entirely in the fabric.
- **The fast path is not where the effort goes.** On Hopper/Blackwell the interesting work is FP8/FP4
  block-scaled kernels and TMA-fed pipelines (both HARD-gated out here, §1); on Ada the interesting
  question is what a *starved fabric* does to a scheme that assumes copious bandwidth.
- **Aggregate host RAM is a genuine asset** (1007 GB vs 24 GB/card, a 42:1 ratio), whereas on a 96 GB
  workstation card the ratio is ~10:1 — so host-tier techniques that are marginal there are structurally
  more attractive here. (I flag this as an *observation*, not a candidate: it is squarely inside the
  KV-offload map's territory, which §0 excludes.)

### 2.6 The engine already self-disables its fast collective path at TP>2 on PCIe-only

**SUBAGENT (not verified by me, and load-bearing enough to re-check first):** a practitioner log from a
4× RTX 4090 PCIe-only box quotes vLLM's own startup line —
> *"Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs"*

If that string is in the shipped tree, it is an **engine self-admission nobody has raised**, and it would
explain my §2.2 result without recourse to exotic transport behaviour: vLLM's custom (oneshot/twoshot)
all-reduce — the mechanism that is *supposed* to cut small-message latency — is switched off exactly in the
≥3-GPU PCIe-only regime this box lives in, so every decode step pays NCCL's generic path instead.
Grep target: `custom_all_reduce` / `"more than two PCIe-only GPUs"` in
`vllm/distributed/device_communicators/`. **This is the cheapest high-value re-verification in this
document and should be done before anything else.**

Corroborating practitioner measurement (SUBAGENT, 4×4090, same regime): per-step weights 12.94 GiB/card at
~939 GiB/s = 13.8 ms ⇒ a 72 tok/s compute ceiling, but **measured 31 tok/s (32 ms)** — i.e. ~18 ms/step of
allreduce + launch overhead that "does not shrink at all even with FP8", with the author's conclusion that
*"On a 4-card P2P-unsupported PCIe configuration this is structurally unrecoverable."* My §2.3 independent
RUN measurement (0.107 ms × 80 all-reduces ≈ 8.6 ms) is the same order but ~2× smaller, and mine is a bare
NCCL call rather than in-engine — the gap between the two is itself worth closing.

### 2.7 A tenancy artifact that is really a research condition

**RUN:** on this shared box, GPUs 0 and 1 (the two slow ones) are occupied by another user, and the
NCCL TP=8 ring must include them. §2.2 shows the 8-rank collective collapsing to 4.34 GB/s on large
messages. Whether that collapse is *caused* by the slow member is untested (I could not run P2P=0 and
could not free GPU 0) — see §3 C3 and §5. But the *condition* is real and is the normal state of a shared
multi-tenant box: **one tenant's placement decision sets another tenant's collective throughput.**

---

## §3 Candidate problems whose difference axis is the hardware

Form: *"nearest neighbour X does A on hardware H; under condition C it misses B; doing B would differ
because …"*. Magnitudes are labelled **(measured)** or **(estimated)**. Each has a zero-GPU or
minutes-long bounding experiment.

---

### C1 — The Ada block-FP8 fallback is unmetered (strongest)

**Statement.** vLLM's FP8 kernel selector (`init_fp8_linear_kernel`, `_POSSIBLE_FP8_BLOCK_KERNELS`) does A
— it dispatches block-scaled FP8 to `CutlassFp8BlockScaledMMKernel` / `DeepGemmFp8BlockScaledMMKernel` on
**sm90/sm100** — while on **C = sm89** every one of those returns `False` and it silently misses B: the
work lands on `TritonFp8BlockScaledMMKernel` **(measured: that is the only kernel that returns
can-implement=True on sm89)**. Doing B — i.e. measuring, and then recovering, the block-FP8 decode cost on
Ada — would differ because the fallback kernel was never performance-selected for this arch, the failure is
**silent** (the checkpoint loads), and the entire published block-FP8/DeepGEMM literature is sm90/sm100-only
(DeepGEMM README: *"NVIDIA SM90 or SM100 architecture GPU"*).

**Magnitude (estimated, and this is why it needs measuring):** block-FP8 W8A8 is usually a throughput win
over bf16 for memory-bound decode; the question is whether Triton-block-FP8 on Ada preserves that win or
inverts it. **I cannot give a defensible number** — that is the point of the candidate. The bounding
experiment is minutes long, below.

**Zero-GPU / minutes-long bounding experiment.** *Already done, and it is zero-GPU and took seconds:*
enumerate `_POSSIBLE_FP8_BLOCK_KERNELS[CUDA]` and call `is_supported_and_can_implement_kernel(k, cfg, 89)`
— that is the **RUN** table in §1.1 and it establishes the fallback without touching a model. The
follow-on (minutes, 1 GPU) is to time `TritonFp8BlockScaledMMKernel` vs `CutlassFP8ScaledMMLinearKernel`
(per-tensor) vs bf16 on a single GEMM shape (e.g. 4096×4096×4096, batch 1 and 128) using the engines'
own kernel classes — no server, no checkpoint.

**Why this is not already owned.** §0: the maps cover KV quant and spec-decode; neither is FP8 *weight/act*
kernel dispatch, and neither map contains an sm89 row.

**Strongest counter-evidence (§4).** See §4 C1 — this is where C1 could die.

---

### C2 — The interconnect sets the decode ceiling, and it does not fall as ranks decrease

**Statement (tightened with the sharpest gap statement found).** The canonical long-context cost model —
arXiv:2405.08944, *"Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance
Analysis"* — does A: it names PCIe as a first-class resource and then **explicitly assumes the term away**.
Both quotes **re-verified by me** in `arxiv.org/html/2405.08944v1`:

> *"(4) context switching is PCIE bound: offloading user 1's KV cache to the CPU DDR and loading user 2's KV
> cache to the HBM is bounded by the PCIE bandwidth."*

> *"Tensor Parallelism utilizes multiple devices for accelerating inference with negligible communication
> overhead. Linearly increasing the number of devices to 2, 4, and 8 introduces more HBM space, thus
> linearly increasing concurrency."*

with a running example of *"a 34B GPT-3.5 level model of 50K context on **A100 NVLink**"*. Under
**C = sm89 / 24 GB / PCIe-only / 2 NUMA** that "negligible communication overhead" assumption is false, and
I measured how false: a **~0.107 ms fixed cost per NCCL all-reduce that does not fall as ranks decrease**
(RUN: TP=2 0.106 ms, TP=4 0.158 ms, TP=8 0.183 ms at 1 MB — it *rises*) and that is **insensitive to the
socket boundary** (14.50 vs 14.43 GB/s, RUN). Doing B — costing serving against a *transport-set latency
floor* rather than a negligible-communication assumption — would differ because the deficit is not in the
bandwidth term at all: at TP=2 the floor is **already** 0.107 ms, so the interconnect cannot be optimised
away by choosing a smaller TP group. Only a different transport moves it.

**Magnitude (measured + derived):** ~0.107 ms/all-reduce at TP=2 (RUN). For 40 layers × 2 all-reduces,
≈ **8.6 ms/token ⇒ ≈116 tok/s ceiling** (DERIVED). The mechanism is *latency*, not bandwidth — the opposite
of the usual PCIe story — and it is the same *shape* as the generation-dependent effect already published in
arXiv:2605.30571 (*"launch-side overhead that becomes visible on fast GPUs but remains mostly hidden on
slower, bandwidth-bound GPUs"*, SUBAGENT quote, ID verified by me), which is why the question is already
legitimate and merely unasked on this topology.

**Zero-GPU / minutes-long experiment.** A pure **zero-GPU** re-analysis of the existing vLLM/SGLang
launch config: count `all_reduce` call sites per decode step for the target model from the model source
(no GPU), multiply by the measured 0.107 ms, and compare against the repo's existing decode-step timings.
That bounds the interconnect share of TPOT **without running anything**. A minutes-long confirmation is a
single-rank-sweep NCCL latency test (1 kB–1 MB, TP=2/4/8) which is what §2.3 already ran in ~2 minutes.

**Counter-evidence (§4 C2).** CUDA graphs may overlap or elide some collectives; and if the floor is an
NCCL software constant rather than a hardware property, a different transport could beat it (which is C3).

---

### C3 — **WEAKENED / magnitude withdrawn** — the fast islands may exist and no collective uses them

> **Status: the magnitude is a measurement artifact and is withdrawn (see §2.1 and §5.3 item 3).** What
> survives is only the *structural* observation, which is still worth recording: (a) `nvidia-smi topo -m`
> reports `SYS` for all 28 pairs and `can_device_access_peer` is `False` for all pairs (RUN), so the
> standard discovery surfaces expose **no** inter-GPU structure on this box; and (b) NCCL's all-reduce is
> **flat across the socket boundary** (14.50 vs 14.43 GB/s, RUN) despite the sockets being physically
> different (NUMA distance 10 vs 21, RUN). **Until the copy matrix is re-measured with correct
> per-device event synchronisation, there is no evidence of a large unexploited asymmetry.** Do not build
> on this candidate as written.

**Statement (as originally formed, magnitude now withdrawn).** NCCL does A — on H = NVLink/datacenter PCIe
it picks a transport from the topology NVML reports — while under **C = this box** it misses B: the standard
discovery surfaces are **blind** to any per-pair structure (`nvidia-smi topo -m` = `SYS` for all 28 pairs;
`can_device_access_peer` = False for all pairs — RUN), and the resulting collectives are **flat** across the
socket boundary (same-node 14.50 vs cross-node 14.43 GB/s, RUN). Doing B — a placement/collective scheme
that identifies any fast set and confines TP groups to it — would differ because the scheduler currently has
no way to see the difference.

**Magnitude:** ~~~8.2× on the raw copy path~~ **WITHDRAWN — artifact**; **0× realised** in NCCL today
(14.50 vs 14.43 GB/s, RUN — this half stands). The honest statement is: *the discovery surfaces are blind,
and the collectives are topology-insensitive; whether a profitable asymmetry exists is UNMEASURED.*

**Zero-GPU / minutes-long experiment.** *Zero-GPU:* parse `nvidia-smi topo -m` + `numa_node` and show that
no available API exposes per-pair bandwidth (already done — RUN), then grep the engines for any consumer of
per-pair bandwidth. *Minutes:* **re-run the §2.1 copy matrix with per-device CUDA event pairs on both
devices**, plus NVML PCIe throughput counters, to establish whether any fast path exists at all. This is the
first experiment to run once the environment is restored, because it either revives or finally kills C3.

**Counter-evidence (§4 C3).** This is the candidate most likely to die outright.

---

### C4 — **DEGRADED BY PRIOR ART** — the context/KV/TP trade on a 24 GB card

> **This candidate is substantially killed and is retained only as a tombstone.** While this document was
> being written, a delegated search found **arXiv:2608.23962, "More GPUs or a Smaller Cache? Tensor
> Parallelism versus KV Compression for Memory-Bound LLM Serving" (25 Aug 2026)** — **ID verified by me
> (FETCH: title read back from `arxiv.org/abs/2608.23962`)**. It is the C4 framing, published. **SUBAGENT
> verbatim:** *"When an LLM serving deployment runs out of KV cache room, there are two well-established
> ways out. Tensor parallelism shards the weights and the KV cache across two, four, or eight devices,
> buying memory headroom at the price of an all-reduce on every layer and a hardware bill that grows with
> the device count."* … *"almost nobody puts the two on the same cost axis."* … *"the boundary that decides
> between the strategies is model size relative to device memory, at roughly 36B parameters for an 80 GB
> card. Below that wall, compression dominates and extra GPUs are largely wasted spend; above it, tensor
> parallelism stops being a choice and becomes an entry ticket."*
>
> **The only surviving sliver** is that its simulator is calibrated on **A100 / A40 / H100 — never a 24 GB
> card**, and its cost axis for "more GPUs" is **dollars, not interconnect latency**. So the 24 GB /
> PCIe-only instantiation is unmeasured; the *tradeoff itself* is not new. Per `FILTER_3AXIS.md` §1a this
> is a "reason-does-not-expire" case, and per §0.4 the paper occupies the cell. **Downgraded accordingly;
> it should not be proposed as a fresh direction without reading 2608.23962 in full first.**

**Statement (as originally formed, for the record).** The KV maps do A — they evaluate KV compression at
fixed parallelism on 80–96 GB cards where the KV pool is large relative to the model — while under
**C = 24 GB × 8 over PCIe** it misses B: VRAM, not compute, forces the TP degree up (§2.4: 14B weights
≈28 GB; 128k-context KV ≈21 GB), and every step of TP degree is paid at the **fixed ~0.107 ms/all-reduce**
(RUN) plus the TP=8 large-message collapse (4.34 GB/s, RUN).

**Magnitude (DERIVED):** at TP=2 the interconnect ceiling is ≈116 tok/s (C2); at TP=8 the per-call latency
rises to 0.183 ms at small messages (RUN) — so **increasing TP to buy context costs decode rate twice**
(more calls × higher per-call latency). **DERIVED, not measured:** applying 2608.23962's own 80 GB
crossover of ~36B to a 24 GB card scales to roughly **~11B**, which would mean the "compression dominates"
regime largely collapses for this box. That is my arithmetic on their published boundary, **not their
claim**, and it is the one testable residue of C4.

**Zero-GPU experiment (still valid, and still cheap):** tabulate (context → KV bytes → minimum TP degree →
measured per-call latency × calls/step) from the RUN numbers in §2.2/§2.3 and compare against 2608.23962's
simulated boundary re-parameterised to 24 GB. Pure arithmetic; no GPU.

**Counter-evidence (§4 C4).** See the boxed note above — prior art occupies the framing.

---

## §4 The strongest counter-evidence for each candidate (what would kill it)

### C1 — the block-FP8 fallback

- **Kill condition 1 (most likely):** a **Triton block-FP8 kernel on Ada is already competitive**, so the
  fallback is not a gap. Nobody has published this, but I also did not measure it — and both engines ship
  Triton FP8 kernels that are actively maintained, which is weak evidence they are not pathological.
- **Kill condition 2: Per-tensor FP8 is the right answer on Ada anyway.** If Ada's FP8-with-FP32-accumulate
  runs at **half rate** (SUBAGENT: Ada whitepaper 330.3 vs 660.6 dense TFLOPS) and block scaling is
  architected away (PTX `sm_120a`), then "block-FP8 on Ada" may be a question with no useful answer, and
  the honest conclusion is "use per-tensor FP8 (RUN: supported) or bf16". This is a **serious** kill risk.
- **Kill condition 3: prior art on consumer-Ada FP8.** I did **not** find a paper or issue measuring
  block-FP8 on sm89 (§5). But QServe (arXiv:2405.04532) and APEX4 (arXiv:2606.08761) are Ampere/Ada
  *quantization* papers already in the inference map (C250, C253) — so the *neighbourhood* is occupied even
  though the specific gate is not. Re-check before committing.
- **Kill condition 4 (the filter doc's §0.6 trap):** this may be "unclaimed but small" — a silent fallback
  that costs a few percent. The minutes-long GEMM benchmark in C1 is designed to settle exactly this,
  and per `FILTER_3AXIS.md` §0.6 the rule is **measure the upper bound before spending GPU hours**.

### C2 — the transport-set latency floor

- **Kill condition 1:** the 0.107 ms is an **NCCL software constant** (e.g. a fixed kernel/launch/handshake
  cost), not a property of the fabric. If so it is a transport-tuning problem, not a hardware-axis result —
  and `NCCL_P2P_DISABLE`/`NCCL_SHM_DISABLE`/tree-vs-ring could move it. **I could not run the P2P=0
  control** (§5), so this is untested.
- **Kill condition 2:** CUDA graphs. vLLM captures decode in graphs; if collectives are captured and
  overlapped, the per-token exposure may be far below 80 × 0.107 ms. My 0.107 ms is a **synchronous,
  unoverlapped** NCCL call — an upper bound, not the in-engine cost.
- **Kill condition 3:** the model is not 40 layers, or uses fewer than 2 all-reduces per layer (fused
  QKV/MLP, or row-parallel-only where the second all-reduce is elided). The 116 tok/s figure is
  configuration-dependent arithmetic, not a measurement.

### C3 — the fast islands — **highest kill risk, magnitude already withdrawn**

- **Kill condition 0 (ALREADY FIRED): my own measurement was under-synchronised.** `torch.cuda.synchronize()`
  waits only on the current device's stream, so the cross-device copy timings are invalid, and the
  fast-path figures exceed the Gen4 ×16 physical ceiling by 4–5.6×. **The magnitude is withdrawn** (§2.1,
  §5.3 item 3). Only the slow mode (~21.7 GB/s) is approximately real.
- **Kill condition 1:** even after a correct re-measurement, the "fast" path may simply not exist; the
  bimodality could be a scheduling/occupancy artifact of GPU 0 being tenant-loaded (100 % util, RUN), not a
  topology property.
- **Kill condition 2: P2P is documented as disabled on all GeForce.** SUBAGENT: Puget Systems,
  *"NVIDIA driver 525.105.17 has P2P 'properly' disabled on GeForce."*; NCCL maintainer
  (kiskra-nvidia, NVIDIA/nccl #1637): *"As far as I know though, it's not supported on any recent GeForce
  hardware."* and *"On your hardware `NCCL_P2P_DISABLE=1` should have no effect as NCCL will auto-detect
  P2P connectivity and disable that transport layer on its own if it's not available."* If that is
  authoritative, C3 has no mechanism — and the 168 GB/s reading is an artifact.
- **Kill condition 3: prior art.** NUMA/root-complex-aware multi-GPU placement is **established**:
  Mobius (ASPLOS'23) — SUBAGENT verbatim — *"the naive sequential mapping scheme of existing pipeline
  systems is not PCIe topology-aware, thus can cause severe communication contention on the CPU root
  complexes of commodity GPU servers"* and *"Since commodity GPUs lack GPUDirect peer to peer
  (GPUDirect P2P) support, inter-GPU communication is first routed through CPU to DRAM and then
  transferred to the target GPU."* The inference-specific instantiation is a NeurIPS 2025 workshop paper
  whose abstract only *outlines* ablations. So the **idea** is not novel; only the **Ada-inference**
  measurement would be.

### C4 — the context/KV/TP trade — **KILLED**

- **Kill condition 1 (FIRED): arXiv:2608.23962** *"More GPUs or a Smaller Cache? Tensor Parallelism versus
  KV Compression for Memory-Bound LLM Serving"* — **ID verified by me (FETCH)** — publishes the exact
  tradeoff, with a device-memory-relative crossover (~36B on an 80 GB card) and the sentence *"almost
  nobody puts the two on the same cost axis."* C4's framing is therefore occupied. Its calibration is
  A100/A40/H100 and its GPU-cost axis is dollars, so the **24 GB / PCIe-only parameterisation** remains
  unmeasured — but that is a parameter sweep of a published model, not a new direction.
- **Kill condition 2:** reachable by existing flags ⇒ the filter doc's §4 auto-kill ("if the benefit lands
  in the interval the engine already covers, judge it a configuration exercise"). If
  `--tensor-parallel-size` + `--max-model-len` + an existing KV dtype already reach the joint optimum,
  there is nothing to build.
- **Kill condition 3:** the KV maps already contain the KV-pressure questions (B51 in
  `KV_CACHE_GAP_MAP.md` is literally *"The PCIe-bandwidth bottleneck model for KV offload ignores
  write-back cost, read-write contention, and MLA"*). C4 is only distinct if framed as *fabric latency*,
  not KV traffic.
- **Kill condition 4:** SUBAGENT: **Helix Parallelism, arXiv:2507.07120** (**ID verified by me, FETCH**)
  plus **vLLM RFC #34018** already own the KV-duplication-vs-TP-width coupling — *"When TP width exceeds
  the number of KV heads, it leads to inefficient KV duplication, limits parallelism, and constrains batch
  size."* For a 8-KV-head model, TP=8 is exactly at that ceiling, which is the mechanism C4 would have
  needed.
- **Kill condition 5:** prior art on the exact box. arXiv 2311.03687 (**verified by me**) benchmarks
  vLLM/LightLLM/TGI on **8× RTX 4090 24 GB PCIe4 ×16 dual-Xeon** — this box, running serving benchmarks,
  already published.

---

## §4b Published prior art that touches this exact box (the most dangerous rows)

These do **not** kill a hardware-axis claim but they sharply constrain what can be called new. Titles
verified by me (**FETCH**, `arxiv.org/abs/<id>` read back):

| Artifact | What it establishes | Does it kill the axis? |
|---|---|---|
| **arXiv:2311.03687**, *"Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models"* (**verified**) | Serving benchmarks (vLLM, LightLLM, TGI) on **8× RTX 4090 24 GB, PCIe4 ×16, 2× Xeon Gold 6230**. **Verbatim (FETCH):** *"the configuration NCCL_P2P_DISABLE=1 was applied. However, this configuration might impact the final performance, putting RTX4090 at a disadvantage against other platforms."* And later: *"This discrepancy might be due to the NCCL_P2P_DISABLE=1 setting."* | **NO — it is the strongest *motivation* for the axis.** The authors **state their own cross-hardware comparison is confounded** by the 4090 P2P workaround and cannot attribute it. The measurement is named as missing by the paper itself. |
| **arXiv:2604.27085**, *"Efficient Training on Multiple Consumer GPUs with RoundPipe"* (**verified**) | SUBAGENT quotes: *"4090 server: 8× NVIDIA RTX 4090 GPUs (24 GB VRAM each), Intel Xeon Gold 6330 CPU, 800 GB available DDR4 host memory, PCIe 4.0 (32 GB/s) interconnect"*; *"without utilizing any GPU peer-to-peer communication (NVLink), relying entirely on PCIe host-to-device transfers."* | **NO** — training/fine-tuning only; no inference serving, no NUMA placement. The hardware string is prior art. |
| **Mobius**, ASPLOS'23 | SUBAGENT quotes: root-complex-aware placement; *"commodity GPUs lack GPUDirect peer to peer (GPUDirect P2P) support, inter-GPU communication is first routed through CPU to DRAM"*. 3090-Ti, training. | **YES for C3's *idea*** — do not claim topology-aware placement as novel. Only an Ada-**inference** measurement would be new. |
| BNTU 2026 journal paper (SUBAGENT, DOI 10.21122/2309-4923-2026-1-54-59, **not verified by me**) | SUBAGENT verbatim: *"The results unequivocally indicate the unsuitability of Tensor Parallelism for systems without NVLink due to critical synchronization delays. It is proven that Pipeline Parallelism is the only viable strategy for PCIe clusters."* 2× RTX 3090, vLLM, 14B. | **PARTLY kills a naive TP-vs-PP proposal** — the qualitative conclusion is peer-reviewed. **Not** on 4090, **not** at 8 GPUs, **not** quantifying the latency term. |
| Quest (arXiv:2406.10774) — already row C302 in the inference map | RTX 4090 used for the FlashInfer implementation | **NO** — and per that map's own note, *"the RTX 4090 evaluation is the opposite of an A100-only constraint"*. |
| vLLM issue **#54059** (SUBAGENT, not verified) | 8×4090, no sparse-MLA attention path on Ada | **NO** — attention-backend availability, not the fabric. Worth re-fetching: it would corroborate §1.2. |
| **arXiv:2608.23962**, *"More GPUs or a Smaller Cache? Tensor Parallelism versus KV Compression for Memory-Bound LLM Serving"* (**verified by me**) | SUBAGENT quotes: *"Tensor parallelism shards the weights and the KV cache across two, four, or eight devices, buying memory headroom at the price of an all-reduce on every layer"*; *"almost nobody puts the two on the same cost axis"*; *"roughly 36B parameters for an 80 GB card"*. Calibrated on A100/A40/H100. | **YES — kills C4.** The TP-vs-KV-compression tradeoff is published. Residue: 24 GB is unmeasured and its GPU-cost axis is dollars, not fabric latency. |
| **arXiv:2507.07120**, *"Helix Parallelism…"* (**verified by me**) + **vLLM RFC #34018** | SUBAGENT: *"When TP width exceeds the number of KV heads, it leads to inefficient KV duplication, limits parallelism, and constrains batch size."* | **YES for the KV-duplication half of C4.** Owns the mechanism by which TP degree is capped by KV-head count. |
| **arXiv:2508.20274**, *"Predictable LLM Serving on GPU Clusters"* (SUBAGENT, not verified) | SUBAGENT: PCIe-aware placement **for LLM serving**, measured on A100: *"Latency-sensitive inference on shared A100 clusters often suffers noisy-neighbor interference on the PCIe fabric"*, TTFT p99 +10-15% at ≤5% cost. | **PARTLY kills C3's serving half** — PCIe-aware *serving* placement is published, on A100. Not consumer Ada, not 2-NUMA. |
| **local-inference-lab/rtx6kpro** community wiki (SUBAGENT, not verified by me) | SUBAGENT: 8-GPU all-reduce bus BW **22.2 GB/s default NCCL vs 37.6 GB/s tuned** on dual-socket PCIe-only; oneshot all-reduce **"+7% on 4 GPU (same NUMA) but does not help on 8 GPU cross-socket"**; crossover threshold **120 KB (4 GPU) → 48 KB (8 GPU, cross-NUMA)**; mechanism *"the system-scope barrier (`__threadfence_system`) must wait for visibility across the Infinity Fabric link"*. Blackwell, not Ada. | **NO — but it is the closest measured analogue to my C2/C3**, and my RUN numbers (14.4 GB/s, flat across the socket boundary) are in the same family. **Must be read in full before proposing anything about collectives on 2-NUMA.** |
| **nccl-tests #117** + **kentino.se 4×4090** + **club-3090** (SUBAGENT, not verified) | Host-staged 4090 fallback ≈**12 GB/s**; granted P2P on a 4×4090 box ≈**19–22 GB/s** (matching my measured slow mode 21.6–21.8 GB/s, RUN); 3090 P2P off→on 11.28→27.12 GB/s and 15.23→1.01 µs, with *"Reason about latency, not bandwidth."* | **NO**, and it **corroborates the slow mode** of §2.1 while leaving the fast mode unexplained. |
| **arXiv:2405.08944**, *"Challenges in Deploying Long-Context Transformers…"* (**ID verified + quotes re-verified by me**) | Names PCIe as a first-class resource, then explicitly assumes it away: *"Tensor Parallelism utilizes multiple devices for accelerating inference with negligible communication overhead."* Running example: **A100 NVLink**. | **NO — it is C2's gap statement.** The field's reference long-context model rests on an assumption my RUN data falsifies here. |
| **mtecnic/vllm-topology-bench** (SUBAGENT, README fetched by them, not me) — 4× RTX 3090 24 GB, no NVLink | 2×TP=2 = 1,437 tok/s vs 1×TP=4 ≈600 tok/s (+136 %); **explicitly denies the KV story**: *"the weights are the wall, not the KV"*; but flags *"Longer contexts would further favor whichever topology has more KV headroom."* | **PARTLY — it bounds §2.4.** Do not claim "KV forces TP"; claim "KV forces TP once context pushes KV past the weight footprint", and note **that crossover is unmeasured on 24 GB**. |

**Verdict — corrected on a second pass (I initially accepted a wrong negative and am recording the fix).**
An earlier draft of this document said *"no published controlled cross-generation comparison for LLM
inference was found."* **That is false and is retracted.** Controlled cross-generation LLM-inference
comparisons **are published**, and I verified three IDs myself (`arxiv.org/abs/<id>`, title read back):

| Artifact | What it establishes | Verified by me |
|---|---|---|
| **arXiv:2605.30571**, *"Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode"* | Batch-1 decode for 7–8B GQA transformers across **H100 / A100 / L40S / L4**, controlled bf16 SDPA. **The headline is a generation-dependent finding** — the achieved fraction of peak HBM bandwidth *falls as peak bandwidth rises* (L4 ≈81 % of its analytic memory floor, H100 only 27 %), and CUDA Graphs gives 1.259× on H100 but only 1.028× on L4. | ID **verified**; quotes **SUBAGENT** |
| **arXiv:2604.09048**, *"Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures"* | 50 LLMs across **10 NVIDIA GPUs including RTX 4090**, V100→H200. States generation-dependence directly (A100 10 % more energy-efficient than H100). | ID **verified**; quotes **SUBAGENT** |
| **arXiv:2405.08944**, *"Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis"* | **Quotes re-verified verbatim by me** in the v1 full text (below). | ID **verified**; **quotes FETCH-verified by me** |

**⇒ The defensible claim is NOT "nobody compared generations." It is narrower and better:** cross-generation
comparison is an **established, respected finding type** in this field (2605.30571, 2604.09048 and APEX4 all
publish *"the result does not transfer across generations"* as their contribution). What has **not** been
done is asking that already-legitimate question on the **multi-GPU serving-topology axis** — TP/PP degree,
PCIe-only vs NVLink, NUMA/socket placement — with generation as the controlled variable. The only artifact
combining hardware class with serving remains arXiv:2311.03687 (**verified by me**), and **its own authors
state the 4090 arm is confounded by `NCCL_P2P_DISABLE=1` and that they cannot attribute their cross-hardware
results.**

**Why 2605.30571 is load-bearing for C2, not just a correction.** Its mechanism is *"launch-side overhead
that becomes visible on fast GPUs but remains mostly hidden on slower, bandwidth-bound GPUs."* The 4090
(1008 GB/s) sits **between** the L4/L40S it was visible on and the H100 it dominated — so whether that
overhead transfers to Ada is genuinely open, and it is the *same shape* as C2's ranks-insensitive latency
floor. This is positive evidence that the C2 mechanism is a recognised phenomenon in this field, measured
on other hardware, and unmeasured here.

**The single sharpest gap statement I found (quotes re-verified by me in `arxiv.org/html/2405.08944v1`):**
the canonical long-context cost model names PCIe as a first-class resource —

> *"(4) context switching is PCIE bound: offloading user 1's KV cache to the CPU DDR and loading user 2's
> KV cache to the HBM is bounded by the PCIE bandwidth."*

— and then **explicitly assumes the term away** in the very next paragraph:

> *"Tensor Parallelism utilizes multiple devices for accelerating inference with negligible communication
> overhead. Linearly increasing the number of devices to 2, 4, and 8 introduces more HBM space, thus
> linearly increasing concurrency."*

with a running example of *"a 34B GPT-3.5 level model of 50K context on **A100 NVLink**"*. So the field's
reference long-context analysis is built on *"TP communication is negligible"* — an assumption that is
false on this box, where TP communication is the measured ceiling (C2). **That is a sharper and more
citable gap than "nobody discussed this."**

**Two further résumé-level facts from the delegated prior-art search (SUBAGENT, not verified by me), each
of which narrows what may be called new:**

1. **Two independent sources state the 4090 measurement cannot currently be made cleanly** — arXiv
   2311.03687 (**verified by me**) and a practitioner log — both citing the `NCCL_P2P_DISABLE=1`
   workaround. That is a citable *"the measurement is missing"* statement, which is the honest
   justification for the axis.
2. **The generation axis is used for kernel authoring; the interconnect axis for serving; the
   consumer-vs-datacenter-class axis only for training.** *Generation × serving-on-consumer-PCIe* is where
   the delegated search found nothing. That is this document's slot — and it is a statement about **who can
   measure**, not about who thought of it.
3. **A practitioner reports SGLang's official cookbook has no verified cell for Ada (SM89)** (verified
   entries: H200 / RTX PRO 6000 / RTX 5090 / DGX Spark / GB300), and that SGLang's multimem all-gather
   SIGFPEs rank 0 at TP>1 on a PCIe-only 4090 box (SUBAGENT, not verified). If true, these are the same
   "nobody has Ada CI" point from the engine side and belong in §1 — **re-verify before citing.**

---

## §5 Search log — exact queries, surfaces, and what I could NOT verify

### 5.1 Surfaces actually used
| Surface | Method | Result |
|---|---|---|
| `schoolserver` GPU/PCIe/NUMA state | `nvidia-smi`, `/sys/bus/pci/devices/*/numa_node`, `/sys/devices/system/node/*/distance` | **RUN** (all facts in §2.1) |
| vLLM 0.26.0 shipped tree at `/home/user/myCCFA/.venv/lib/python3.11/site-packages/vllm` | direct `grep -n`, `sed -n`, and **live Python calls** into the compiled extension | **RUN / READ-SRC** (§1.1, §1.4) |
| NCCL all-reduce | `torch.distributed` + `nccl`, rank sweep | **RUN** (§2.2, §2.3) |
| GitHub issue search (`github.com/vllm-project/vllm/issues?q=…`) | curl | **FAILED — see §5.3** |
| `raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md` | `curl -sL --retry 4` | **FETCH** |
| `arxiv.org/abs/2311.03687`, `arxiv.org/abs/2604.27085`, `arxiv.org/html/2311.03687v2` | `curl` + title read-back | **FETCH / verified** |
| Delegated agents | two `subagent` runs | **SUBAGENT** (labelled everywhere above) |

### 5.2 Queries issued (verbatim)
- `https://github.com/vllm-project/vllm/issues?q=topology+aware+placement&state=all`
- `https://github.com/vllm-project/vllm/issues?q=PCIe+bandwidth+heterogeneous&state=all`
- `https://github.com/vllm-project/vllm/issues?q=tensor+parallel+PCIe&state=all`
- `https://github.com/vllm-project/vllm/issues?q=numa+GPU+placement&state=all`
- `https://github.com/vllm-project/vllm/issues?q=P2P+disabled&state=all`
- Local greps across `KV_CACHE_GAP_MAP.md`, `spec-decode-gap-map-2026-09-13.md`, `INFERENCE_ACCEL_GAP_MAP.md`,
  `notes/DIRECTION_MAP_3AXIS.md`, `notes/FILTER_3AXIS.md`, `hw_ruled_out_kv_cache_offloading.md` for
  `sm89|sm_89|ada lovelace|RTX 4090|PCIe-only|no NVLink|NUMA`.
- Local greps of the vLLM tree for `sm89|sm_89|is_device_capability|get_device_capability|Ada`.
Subagent-issued queries are enumerated in `research/sm89-gating-inventory.md` and
`research/prior-art-consumer-ada-8x4090.md`.

### 5.3 What I could NOT verify — stated plainly, with the reason

1. **GitHub issue search is not usable from this box.** `github.com/<org>/<repo>/issues?q=<q>&state=all`
   returns a **React shell** whose embedded JSON contains only the pinned/roadmap issues; **all five
   queries above returned the identical four titles** (`[Roadmap] vLLM Roadmap Q3 2026`, `[Roadmap] Rust
   Frontend Feature Parity`, `[Model Support] Kimi K3 Tracking Issue`, `Issues · vllm-project/vllm`),
   proving the `q=` parameter is not applied server-side. `api.github.com` is rate-limited to zero.
   **⇒ Therefore: every "no upstream work found" claim for the non-hardware neighbours is
   `I did not find it`, NOT `it does not exist`.** The gating subagent reports the same failure and used
   `github.com/search?q=repo%3A…&type=issues` as a workaround.
2. **The `NCCL_P2P_DISABLE=1` control never completed.** Both the P2P=0 run and the sustained-clock
   transfer run were killed when the shared venv broke (§5.4). **So the question "does the field's
   standard 4090 workaround help or hurt on this box?" is UNANSWERED.** This is the single most valuable
   missing measurement and it is cheap (~2 min) once the environment is restored.
3. **§2.1's absolute fast-path bandwidth (136–178 GB/s) is an artifact of under-synchronisation and is
   WITHDRAWN.** `torch.cuda.synchronize()` with no argument waits on the *current* device's stream, not
   both devices', so cross-device copy timings are not trustworthy. The values also exceed the Gen4 ×16
   physical ceiling (~32 GB/s/direction) by 4–5.6×, which is the tell. **The slow mode (~21.7 GB/s) is
   approximately real; the fast mode's magnitude is not.** A correct re-measurement needs per-device CUDA
   event pairs on both devices. This is the single most important correction in this document.
4. **`nvidia-smi topo -m` reports no inter-GPU structure at all.** topo says `SYS` for all 28 pairs and
   `can_device_access_peer` says `False` everywhere. **Whether that is accurate (a genuinely flat,
   host-staged fabric) or merely uninformative is UNRESOLVED** — my copy probe that appeared to contradict
   it was itself broken (item 3), so at present the flat reading is the one with evidence behind it.
5. **No primary NVIDIA statement that P2P is disabled on RTX 4090 was located by me.** The subagent
   reports Puget Systems + an NVIDIA/nccl maintainer comment as the closest; the original NVIDIA
   developer-forum/press statement was not found. **Explicitly deferred to SUBAGENT.**
6. **Block-scaling ISA floor: `sm_120a` (not sm90) is SUBAGENT-reported only.** This materially changes the
   story (§1.1) and should be re-fetched from the PTX ISA doc before being treated as settled. Note the
   internal tension: `cutlass_scaled_mm_supports_block_fp8` returns **True for 90** (RUN), so "sm90 has
   block FP8 in CUTLASS" and "per-32 `ue8m0` block scaling needs sm_120a" are **different mechanisms** —
   the subagent itself flags this ("CUTLASS's `94_ada_fp8_blockwise` is *not* MXFP8… Don't conflate.").
7. **CORRECTED — my initial "no cross-generation comparison exists" claim was WRONG.** Controlled
   cross-generation LLM-inference comparisons **are** published: **arXiv:2605.30571** (H100/A100/L40S/L4,
   batch-1 decode) and **arXiv:2604.09048** (10 GPUs incl. RTX 4090) — both IDs **verified by me**. The
   corrected, narrower gap is the *multi-GPU serving-topology* axis (TP/PP degree, PCIe-only vs NVLink,
   NUMA placement) with generation controlled; see §4b. **What remains true: I did not find a TP-vs-PP
   scaling study on 8× 4090** (queries in §5.2 + subagent queries), and **no paper measuring block-FP8 on
   sm89.** `I did not find those.`
8. **MLPerf is UNSEARCHED, not negative.** No MLPerf surface was successfully fetched, so **do not claim
   MLPerf lacks a cross-generation analysis.**
9. **The cross-generation search was partially cancelled** — a 15-query batch was terminated mid-run, so
   that axis is **partially** searched, not exhaustive. Newly identified blocked surfaces: `arxiv.org/search`
   and `arxiv.org/list` → **HTTP 406** (no arXiv full-text search at all), `openreview.net` PDF/API → 403.
10. **Counter-evidence to §2.4's KV-forces-sharding chain (SUBAGENT, README fetched by them, not by me):**
    `mtecnic/vllm-topology-bench` on 4× RTX 3090 24 GB no-NVLink, Qwen3.6-35B-A3B-AWQ, reports 2×TP=2 =
    1,437 tok/s vs 1×TP=4 ≈ 600 tok/s (+136 %), and **explicitly denies KV is the driver**: *"Lowering
    max-model-len doesn't help — the weights are the wall, not the KV."* **Consequence: §2.4's claim must
    be read as "KV forces TP only once context pushes KV past the weight footprint" — and I did not find
    that crossover measured on 24 GB cards.** The weight term (≈28 GB for 14B bf16, DERIVED) dominates at
    short context; KV (≈21 GB at 128k, DERIVED) dominates at long context.
11. **The §2.6 vLLM "custom allreduce disabled" string is SUBAGENT-reported and NOT verified by me** — and
   it is the single highest-value cheap check remaining, because if it is real it explains §2.2 without any
   exotic transport hypothesis, and it is an engine self-admission nobody has raised. Grep
   `vllm/distributed/device_communicators/` for `"more than two PCIe-only GPUs"`. **Do this first.**
12. **§2.6's 18 ms/step practitioner figure and the rtx6kpro collective numbers are SUBAGENT-reported**,
    from community sources I did not fetch. They corroborate my RUN measurements in the same direction but
    must not be cited as established.
13. **`arXiv:2508.20274` ("Predictable LLM Serving on GPU Clusters") was not verified by me.** If it really
    does PCIe-aware placement *for LLM serving* on A100, it is the closest serving-side prior art to C3 and
    narrows that candidate further.

### 5.4 Operational note — the shared environment is now broken (not by this analysis)

At ~04:00 UTC during this session `/home/user/myCCFA/.venv` lost
`torch/lib/libtorch_global_deps.so` (the file is listed in `torch-2.11.0+cu128.dist-info/RECORD` but is
absent on disk, and `torch/lib/` now holds only 7 files). **`import torch` now fails on the whole venv**,
which blocks all further GPU work. `/` is at **92 % (141 GB free)**. I did not modify the venv. I cleaned
up my own artifacts: removed `/home/user/Qwen3.8-27B` (abandoned) and `/tmp/ssltmp`; my probe scripts and
raw logs remain under `/home/user/myCCFA/probes/hw/` (`nccl_ar.py`, `run_nccl.py`, `sustained.py`).
**Restoring the venv is a prerequisite for every experiment in §3.**

---

## §6 One-line summary of the surviving set

Ranked by (survives) × (magnitude) × (cost to bound). **Only two candidates survive; two are dead.**

- **C1 — block-FP8 silently falls back to Triton on sm89.** Strongest. The fallback is a **RUN** fact
  (zero-GPU, seconds to establish); the cost is unmeasured by anyone. Survives unless Triton block-FP8
  turns out competitive — which the minutes-long GEMM benchmark decides **before** any GPU hours are spent
  (the `FILTER_3AXIS.md` §0.6 rule).
- **C2 — a ~0.107 ms all-reduce latency floor that does not fall as ranks decrease.** Most quantitative;
  directly governs decode rate here. **Dies** if the floor is an NCCL software constant (e.g. because
  vLLM's custom all-reduce is disabled at TP>2 on PCIe-only, §2.6) rather than a fabric property — and
  that is exactly what the un-run `NCCL_P2P_DISABLE` control plus the §2.6 grep would settle.
- **C3 — topology-blindness observation (magnitude WITHDRAWN).** `nvidia-smi topo -m` reports `SYS` for all
  28 pairs and `can_device_access_peer` is `False` everywhere (RUN), while NCCL is flat across the socket
  boundary (14.50 vs 14.43 GB/s, RUN) — but **my raw-copy asymmetry was a measurement artifact and is
  withdrawn**, so there is currently **no evidence of a large unexploited margin**. Re-measure with correct
  per-device synchronisation before treating this as anything.
- **C4 — context/KV/TP joint optimum on 24 GB: KILLED** by arXiv:2608.23962 (**verified**), with
  arXiv:2507.07120 + vLLM RFC #34018 owning the KV-duplication mechanism. Retained in §3/§4 as a tombstone.

**Per `notes/FILTER_3AXIS.md` §0.6** ("unclaimed ∧ sufficiently large are near-mutually-exclusive in this
field"), the one thing this axis changes is *why* that trade-off is not forced: sm89 work is unclaimed not
because it is too small, but because **the upstream fleet is Hopper/Blackwell** — there is no sm89 CI on
which a maintainer could even measure it, and the one paper that ran serving on this exact box
(arXiv:2311.03687, verified) **openly states that its own 4090 arm is confounded by the P2P workaround and
that it cannot attribute its cross-hardware results.**

**The strongest honest form of the argument (corrected — see §4b).** Do **not** claim "nobody compared GPU
generations for LLM inference"; that is false (arXiv:2605.30571, arXiv:2604.09048). The argument that
survives is better than that one:

> Generation-dependence is already an **established, respected finding type** in this field — three verified
> artifacts (2605.30571, 2604.09048, APEX4) publish *"the result does not transfer across generations"* as
> their contribution, and the field's own reference long-context analysis (2405.08944) **explicitly assumes
> TP communication is negligible** on A100 NVLink. What has never been done is asking that already-legitimate
> question on the **multi-GPU serving-topology axis** — TP/PP degree, PCIe-only vs NVLink, NUMA placement —
> with generation as the controlled variable. The reason is not that nobody thought of it: it is that the
> **RTX 4090's P2P defect makes the Ada arm of that comparison unmeasurable by default**, a fact the only
> paper to attempt serving on this box states about its own results.

**So the difference axis is the hardware itself, and the honest claim is about who can measure it — not
about who thought of it.**

### The 1–2 strongest, in the required sentence form

> **C1.** The nearest neighbour — vLLM's FP8 kernel selector, and the whole block-FP8/DeepGEMM line of work
> (DeepGEMM README: *"NVIDIA SM90 or SM100 architecture GPU"*) — does A (dispatch block-scaled FP8 to
> CUTLASS/DeepGEMM) on hardware H = sm90/sm100; under condition C (= sm89 / 24 GB / PCIe-only) it misses B,
> because **every one of those kernels returns `can-implement = False` on this device (RUN, zero-GPU) and
> the work silently lands on `TritonFp8BlockScaledMMKernel`**. Doing B — measuring, then recovering, the
> Ada block-FP8 decode cost — would differ because the fallback was never performance-selected for this
> arch, the substitution is **silent** (the checkpoint loads without error), and the entire published
> block-FP8 literature is sm90/sm100-only. **Magnitude: unmeasured by anyone; I explicitly decline to
> estimate it — that is the experiment.** Bound: seconds (kernel-support enumeration, already done) then
> minutes on one GPU (time the three kernels on a 4096³ GEMM at batch 1 and 128). **Kill:** Triton
> block-FP8 is competitive, or per-tensor FP8 (RUN: supported on sm89, CUDA ≥ 12.4) is simply the right
> answer here.

> **C2.** The nearest neighbour — the spec-decode and KV maps, and the published TP-vs-KV-compression
> tradeoff (arXiv:2608.23962, verified) — does A (treat decode cost as a function of acceptance rate, KV
> size and batch composition) on hardware H = 80–96 GB sm90/sm120 with NVLink; under condition C (= sm89 /
> PCIe-only / 2 NUMA) it misses B, because there is a **measured ~0.107 ms fixed cost per NCCL all-reduce
> that does not fall as ranks decrease (RUN: TP=2 0.106 ms, TP=4 0.158 ms, TP=8 0.183 ms at 1 MB — it rises)
> and that does not respond to the socket boundary at all (14.50 vs 14.43 GB/s bus bandwidth, RUN)**. Doing
> B — budgeting decode against a *transport-set latency floor* instead of a bandwidth term — would differ
> because the standard intuition ("more TP ⇒ more comms ⇒ more cost, so shrink the group") has the wrong
> shape here: at TP=2 the floor is **already** 0.107 ms, so the interconnect is not something you can
> optimise away by choosing a smaller TP group — only a different transport moves it. **Magnitude:
> ~0.107 ms measured; ≈8.6 ms/token and a ≈116 tok/s single-stream ceiling DERIVED for 40 layers × 2
> all-reduces.** Bound: zero-GPU (count all-reduce call sites per decode step from model source × measured
> latency), then a ~2-minute rank sweep. **Kill:** it is an NCCL software constant, not the fabric — most
> likely if vLLM's custom all-reduce is disabled at TP>2 on PCIe-only (§2.6, SUBAGENT-reported, unverified)
> — or CUDA-graph capture overlaps it away.

> **C3 (do not propose as-is).** The nearest neighbour — Mobius (ASPLOS'23) and arXiv:2508.20274 — does A
> (topology-aware placement from the fabric the driver reports) on hardware H = 3090-Ti / A100 clusters;
> under condition C (= sm89, 8×24 GB, 2 NUMA, PCIe-only) the discovery surfaces report **nothing**
> (`nvidia-smi topo -m` = `SYS` for all 28 pairs, `can_device_access_peer` = `False` for all pairs, RUN) and
> NCCL is flat across the socket boundary (14.50 vs 14.43 GB/s, RUN), so it misses B (any per-pair structure
> at all). **But the one measurement that would have shown a profitable margin — my raw-copy asymmetry — was
> an under-synchronisation artifact and is WITHDRAWN, and P2P is documented as disabled on all GeForce.**
> So this candidate currently rests on a *blindness* observation with **no measured margin**, and must be
> re-measured (per-device CUDA events on both devices + NVML PCIe counters) before it is anything.
