# 池① 在新机（vLLM 0.29.0 / sm120 / Qwen3-4B）上的过筛 —— `REASON-MAY-HAVE-EXPIRED` 格

**日期**：2026-09-15 · **执行者**：子代理（会话 `session-…`），父代理 `session-1efb6dac`
**候选池**：`INFERENCE_ACCEL_GAP_MAP.md` 中 **263 个** `REASON-MAY-HAVE-EXPIRED` 格（这是本仓库**唯一**带该标签的地图；KV 地图与投机地图各 0 个）
**判据**：`notes/FILTER_3AXIS.md` 正文 + 附录 A（§1a / §1c / §0.4 / §0.5 / §0.6 / §3 / §4）
**成本**：**0 GPU·h**（全程遵守"零 GPU"约束：只做源码阅读与廉价命令；未跑任何模型推理或基准）
**机器**：`ssh -p 26924 root@connect.weste.seetacloud.com`（1× RTX 6000D 85,651 MiB，CC **12.0**，驱动 595.71.05，Python 3.12.3 / torch 2.13.0+cu130 / **vLLM 0.29.0**，flashinfer **0.6.18**）

**阅读约定**：文中所有 `file:line` 均指**已安装树**
`/root/ccfa_venv/lib/python3.12/site-packages/vllm/`（下文简写 `<V>/`），
是我用 `sed -n`/`grep -n` **亲自读到**的行号。凡未亲读者，一律写 **"未亲验"**。

---

## 0. 本轮最有价值的三条结论（比任何单格判决更重要）

### 0.1 池① 的系统性失败模式与前两轮**不同**

前两轮（KV / 投机地图）池① 的死因是"占位者仍在 OPEN"。
**263 格的死因分布完全不同**：

| 死因 | 数量级 | 机制 |
|---|---|---|
| **R1 约束未变**（理由仍成立） | ~90 格 | 架构门/厂商 roadmap/结构性缺失，在 0.29.0 源码里**逐字仍在** |
| **R2 落到了"本机没有的配置"**（量化 / MoE / MLA / 多卡 / ROCm / 第二引擎 SGLang / 训练） | ~120 格 | 模型是 **BF16 dense Qwen3-4B 单卡**，这些杠杆对本机**根本不参与** |
| **R3 量级 <2%** | 少数，但**本轮所有可算的存活者都死在这里** | 见 §2 算术 |
| **R4 上游在飞**（DRAFT/OPEN PR） | 至少 1 条新的、地图完全没记的 | 见 §0.3 |

⇒ **本轮没有出现"池① 够大但被占位"的重复**；出现的是 **"池① 的过筛筛子本身没适配新机"**（§9.1 的老问题以新形式复发）。

### 0.2 我在本机源码里读到的**真正改变了的事实**（可直接复用）

这些是**新的、地图没有的、本机亲验**的事实：

1. **`IS_QUANTIZED = False` 与 `IS_DENSE = False` 被硬编码，全模型一律**（`<V>/config/vllm.py:129-134`，含注释
   *"The optimizations that depend on these properties currently set to False in all cases."*，指向 issue **#25689**）。
   ⇒ 依赖它们的 `fuse_attn_quant` / `enable_sp` / `fuse_gemm_comms` 在 **O0–O3 全部关闭**（`<V>/config/vllm.py:245,268,291,314`）。
   **这一条一次性杀掉了池① 里一整簇"量化侧优化 + 融合"的格**（C197/C198/C259/C260 及 C247/C155 的触发条件）。
2. **sm120 上 `supports_trtllm_attention(prefill=False) == True`**（我**跑出来**的，非推断）：
   `has_nvidia_artifactory() = True`、`supports_trtllm_attention(is_prefill=True) = False`、
   `supports_trtllm_attention(is_prefill=False) = True`（`<V>/utils/flashinfer.py:467-489`）。
   ⇒ FlashInfer 在 sm120 上拿到 **XQA decode**，`AttentionCGSupport` 升到 `UNIFORM_BATCH`
   （`<V>/v1/attention/backends/flashinfer.py:974-1015`），因此 **FULL decode cudagraph 在 sm120 上真的成立**。
   **实证**：本机既有日志 `/root/ccfa_results/2026-09-12/smoke/01-nospec.serve.log:11` 里
   `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>`，`:46` 显示 **35 个 FULL 图被捕获**，
   `:16` `Using FLASH_ATTN attention backend`，`:28` `Using FlashInfer for top-p & top-k sampling`。
   ⇒ **这条"能力"已经发货，不构成缺口**（反向排除）。
3. **sm120 的 FA2 是完整的分片 KV（FlashDecoding）路径**，不是残缺的：
   - vLLM 内置 FA2 二进制 `_vllm_fa2_C.abi3.so` 里存在 **561 个** `split` 符号，含
     `flash::run_flash_splitkv_fwd<…>` 与 `flash::num_splits_heuristic`（我用 `nm -D` 亲验）。
   - 关键条件在 **FA2 的 GQA 交换分支**：`seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && …`
     （vllm-project/flash-attention `csrc/flash_attn/flash_api.cpp:646`，我经 HTTPS 读到），
     **只有该分支**调用 `set_params_splitkv`（同文件 `:756-761`）；
     非交换的 paged 分支是 `STD_TORCH_CHECK(num_splits <= 1, …)`（同文件 `:763`）。
   - Qwen3-4B 恰好满足：`num_attention_heads 32 > num_key_value_heads 8`，decode `seqlen_q == 1`。
   ⇒ 本节原结论写作 **C99 一族「sm120 缺 split-KV」在本机不成立**（详见 §2 的 C99 行）。

   > **⚠️ 2026-09-15 更正（`notes/KILLREASON_AUDIT_2026-09-15.md`，我已采纳并回读源码确认）**：
   > 上面这条**只有二进制那一半成立**。**符号存在 ≠ 可达** —— 已装封装在
   > `<V>/vllm_flash_attn/flash_attn_interface.py:311-312` 用 **Python** 把它闸掉了：
   > `if num_splits > 1: raise NotImplementedError(...)`。上游 `flash_api.cpp` 的 GQA-swap 分支
   > 在 Python 传入 `>1` 时**根本没机会执行**。
   > ⇒ **本节应记为 `UNRESOLVED`，不是 established。** 最便宜的收口：一次短运行打印 decode batch
   > 在 `cudagraph_mode=FULL` 下的 `attn_metadata.max_num_splits`。
   > **方法学教训**：本工作区已把「断言配置真的生效，而不是断言意图」写进纪律，
   > 却没有把同一条纪律用在**源码路径判定**上 —— `nm -D` 看到的符号，与 Python 层实际会不会走到，是两件事。
4. **`enable_qk_norm_rope_fusion` 在 CUDA 上被硬关**：`<V>/config/vllm.py:252,275,298,321` 在 **O0/O1/O2/O3 四档全为 `False`**；
   注册点 `<V>/compilation/passes/pass_manager.py:225`。融合核仅支持 head_dim ∈ (64,128,256)
   （`<V>/compilation/passes/fusion/qk_norm_rope_fusion.py:32`），Qwen3-4B 的 `head_dim=128` **在集合内**。

### 0.3 一条**地图完全没有记**的上游在飞占位者（§0.5 的又一次实证）

用独立检索（不走地图）查到 vLLM PR **#47979**
**[Perf] SM120 PCIe serving stack: SP/async-TP enablement, FlashInfer spec-decode FULL cudagraphs, and PCIe-safe multi-GPU comms**
—— 我读回的状态是 **`"state":"DRAFT"`**（`https://github.com/vllm-project/vllm/pull/47979`）。
同时该检索面还浮现 `#43232 Xqa decode kernels`、`#51696 [Kernel] Add native B12X linear, MoE, and causal attention backends`。
⇒ **"sm120 服务栈的 per-step 优化"这一整片区域已经有 DRAFT/在办工作**，地图一条都没记。
（**这是 DRAFT，我按 §1c 记为"在飞"，不写成"已发货"**。）

---

## 1. 短线表（Shortlist）

列义：**L?** = 该格"让位/受阻于的那个 X"是否真的落地（**只写我亲自读过的东西**）。

| 格 | 地图 | `REASON-MAY-HAVE-EXPIRED` 原文（截取，逐字） | **X 是否落地** | `file:line` 证据（我读的） |
|---|---|---|---|---|
| **C258** | INFERENCE_ACCEL | "likely yes — the stated regression is explicitly measured **\"on H100\"** (SM90, 132 SMs) … the rig is sm120 … **Nobody has published a sm120 measurement.**" | 理由字面**仍成立**：fusion 仍全档关闭 | `<V>/config/vllm.py:252,275,298,321`（O0–O3 皆 `"enable_qk_norm_rope_fusion": False`）；`<V>/compilation/passes/pass_manager.py:225`；`<V>/model_executor/models/qwen3.py:150-151`（Qwen3 确有 q_norm/k_norm） |
| **C260** | 同 | "yes — the reason is explicitly torch-version-dependent … and the v0.29.0 tag still ships **`IS_QUANTIZED = False`** rather than the torch-group-quant default the checklist wanted." | **未落地**（理由成立） | `<V>/config/vllm.py:129-134`（硬编码 `False` + 注释 + `#25689` 链接） |
| **C197** | 同 | "no — the stated comparison is Inductor-generated Triton vs vLLM's own CUDA kernel on NVIDIA, which is exactly the rig's configuration." | 地图自判"不适用"；本机再度确认：**BF16 + `custom_ops=['none']` ⇒ `enable_norm_fusion` 为 False** | `<V>/config/vllm.py:136-141`（`enable_norm_fusion` 定义）；本机日志 `01-nospec.serve.log:11`（`'fuse_norm_quant': False`, `'fuse_act_quant': False`, `'custom_ops': ['none']`） |
| **C198** | 同 | "partly — … `use_inductor_graph_partition` exists at v0.29.0 but is still off in the O2/O3 defaults" | **未落地** | `<V>/config/vllm.py:291,314`（O2/O3 `"use_inductor_graph_partition": False`） |
| **C247 / C155** | 同 | C155: "no — the forced-`FULL` fallback is present verbatim in `vllm/config/compilation.py` at v0.29.0." | **未落地**，且**触发条件在本机不可达** | `<V>/config/compilation.py:1257-1275`（`set_splitting_ops_for_attn_fusion` → `cudagraph_mode = CUDAGraphMode.FULL`）；`<V>/config/compilation.py:1150`（仅当 `fuse_attn_quant and not use_inductor_graph_partition`）；而 `fuse_attn_quant = IS_QUANTIZED = False` |
| **C168** | 同 | "YES — … vLLM 0.29.0 **ships the sampler enabled by default again**" | **已落地** ⇒ 无缺口 | `<V>/envs.py:856-860`（`else True`）；`<V>/envs.py:49`（`VLLM_USE_FLASHINFER_SAMPLER: bool = True`） |
| **C230** | 同 | "YES — … **the default was never rolled back.**" / S3b: "SOLVED (the default is True in vLLM 0.29.0)" | **已落地** ⇒ 无缺口 | 同上；本机日志 `:28` `Using FlashInfer for top-p & top-k sampling.` |
| **C137** | 同 | "**This is the single highest-value expiry to watch** … the stated reason expires if either (a) FlashAttention ships CuTe-DSL kernels for capability 12.x, or (b) vLLM lifts the `major == 10` branch. Checked at retrieval: neither has happened" | **未落地**（逐字复核） | `<V>/vllm_flash_attn/flash_attn_interface.py:72-86`（`_is_fa4_supported` 只放行 family 90/100/110，**拒绝 12.x**）；`<V>/v1/attention/backends/fa_utils.py:82-88`（默认 `major == 10 → 4`，**否则 FA2**）+ `:130-134`（`major >= 10 and fa_version == 3` 的降级注释） |
| **C138** | 同 | "The gate is a shipped capability-family check; it would expire if FlashMLA gained an sm120 build." | **未落地** | `<V>/v1/attention/ops/flashmla.py:53-56`（dense：`is_device_capability_family(90)` only）；`:60-72`（sparse：family 90 或 100，reason 串为 *"FlashMLA Sparse is only supported on Hopper and Blackwell DC devices."*）⇒ 12.0 两者皆不可达 |
| **C146** | 同 | "**Note on sm_120 specifically: `_resolve_gdn_prefill_backend` only enables the flashinfer/cutedsl GDN prefill paths for SM90 and cc-family-10.x**" | **未落地**（逐字成立） | `<V>/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:93-108`（docstring 明确 "Hopper (SM90)" 或 "Blackwell (SM10.x)"） |
| **C160** | 同 | "Yes, strongly — this is vLLM 0.9.2.dev311 on 2025-06-29, before sm120 had any dedicated FP8 CUTLASS sources. vLLM 0.29.0 now ships `scaled_mm_c3x_sm12…`" | **已落地**（sm120 CUTLASS 源确实在树内） | `<V>/third_party/deep_gemm/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl` 等**多个 `sm120_*` 文件**（`find` 亲验）；但**需要 FP8 模型**，BF16 路径不经过 |
| **C144** | 同 | "The three block-size sites branch on `is_device_capability_family(90)`/family(100), so they expire when they gain an explicit `120` branch." | **部分已落地**：FlashInfer 后端里 `120` 分支已存在 | `<V>/v1/attention/backends/flashinfer.py:526,844,950,987,1027,1049,1124,1155,2066` 共 9 处 `is_device_capability_family(120)` / `is_sm12x`。**未亲验** issue #56702 本身的具体三处 |
| **C147** | 同 | "Expires when the FlashInfer sm120 + NVFP4 + fp8-KV combination is fixed upstream" | **未亲验** | 我只确认了 `<V>/v1/attention/backends/flashinfer.py:970-975` 存在 *"Architectures with only fa2 (e.g. SM89, SM120) cannot consume FP8 queries"* 的注释；**没有**去核 NVFP4 的修复状态 |
| **C99 / C5** | 同 | C99: "partly — harness drift is real; the `assert num_splits == 1` gate this fills is now also targeted by open PR #2758." | **本机不成立**（FA2 的 split-KV 路径是活的） | 见 §0.2.3：`nm -D` 561 个 split 符号 + FA2 fork `csrc/flash_attn/flash_api.cpp:646,756-761,763` |
| **C104** | 同 | "yes — the 2026-05-29 \"no plans\" statement predates merged sm120 prefill via a different backend." | **能力层已落地、但与本机模型无关** | `<V>/platforms/cuda.py:158-163`（sm120 默认后端序 `FLASH_ATTN, FLASHINFER, TRITON_ATTN, …`）；我**跑出** `supports_trtllm_attention(is_prefill=True) = False`（TRT-LLM **prefill** 仍不在 sm120） |
| **C233** | 同 | "YES — the stated reason was the V0→V1 deprecation sweep, a constraint that no longer exists in vLLM 0.29.0" | 理由**失效** | 但本机**不可达**：`<V>/platforms/cuda.py` 的 `_get_backend_priorities` 非 MLA 分支里**没有** block-sparse 后端；sm120 non-MLA 优先级表只有 `FLASH_ATTN/FLASHINFER/TRITON_ATTN/FLEX_ATTENTION/TURBOQUANT` |
| **C169** | 同 | "YES — the measurement was taken against the Python V0/V1 frontend of early 2025 (**\"Current main API server is within 3% of the LLM performance already\"**)" | 理由**已过期**（前端重写） | 但**结构性下限**：detokenize 现在跑在**前端进程的后台协程**里 —— `<V>/v1/engine/async_llm.py:740-822`（`_run_output_handler` → `asyncio.create_task(output_handler())`）、`:651`（注释 *"A separate output_handler loop runs in a background AsyncIO task"*）⇒ 它**本就在关键路径之外** |
| **C172** | 同 | "YES — the crossover (~40 rows, vocab 160k) is a measured property of one Triton kernel version on GB300; a kernel change or a different vocab/batch mix moves it" | **上游在飞**（浮出 open PR #48913） | 我未亲读 #48913 正文（**未亲验**）；默认采样已走 FlashInfer（`01-nospec.serve.log:28`） |
| **C216 / C259 / C207 / C244 / C246 / C64 / C179** | 同 | （各自原文见附录 B） | **本机不可达 / 量级不足** | C216 是 ROCm/gfx950；C259 是 Helion 量化核；C207 属 SGLang（**本机未装**）；C244 的符号在 B=1 为**负**且属 SGLang；C246 是 FP8-KV 精度；C64 是 H100/1B 的文档测量；C179 地图自判 "no" 且已被 open PR 占位 |

**核验计数**：本轮**逐格过筛 263 个**（脚本抽取全部原文），其中**深读源码核验 20 个**（上表），
另**顺手核验 6 个**（C41/C51/C54/C55/C112/C156 —— 见附录 A），合计 **26 格的"X 是否落地"得到一手判定**。
其余 237 格：我读了它们的原文与地图 S3b 结论，但**没有亲自复核 X**，一律**不作为结论**。

---

## 2. 每个入选格的 §3 句子 · §0.6 量级算术 · 判定

### 2.0 量级算术的公共底座（全部来自源码/规格，不是猜）

**本机 decode 每步的时间下界（权重读取）**：
- Qwen3-4B `config.json`（我在机上 `cat` 到）：`hidden_size 2560`、`num_hidden_layers 36`、
  `vocab_size 151936`、`tie_word_embeddings true`、`torch_dtype bfloat16`、`num_attention_heads 32`、
  `num_key_value_heads 8`、`head_dim 128`。
- 权重流量 ≈ 4.0e9 参数 × 2 B = **8.0e9 B/token**。
- RTX 6000D 与 RTX PRO 6000 Blackwell 同档 GDDR7（**1.79 TB/s 规格**；见 [DigitalStorm 配置页](https://www.digitalstorm.com/configurator-info.asp?id=23513)、
  [RTX PRO 6000 Blackwell 96GB 产品页](https://www.overclockers.co.uk/nvidia-rtx-pro-6000-blackwell-workstation-edition-96gb-gddr7-graphics-card-gra-nvi-05589.html)）。
  取 85% 有效带宽 ⇒ **1.52 TB/s**。
- ⇒ **t_base(bs=1, 短上下文) ≈ 8.0e9 / 1.52e12 ≈ 5.3 ms**；含 attention/KV/激活开销取 **7–12 ms/步**。
- ⇒ **2% 门槛 = 0.14–0.24 ms/步**。**本节所有"每层节省"都要乘 36 层**。

**每核 launch 开销**：CUDA graph replay 下 Python 侧开销已归零，只剩**设备侧 launch latency ≈ 1.0–1.5 µs/核**
（先前轮次实测：n-gram 提议器 1 线程上限可回收 **0.103 ms/步 ≈ 0.9%**，见 `GOAL3_CLOSE` §1.15 —— 与本模型同量级带）。

---

### 2.1 C258 `enable_qk_norm_rope_fusion`（**唯一过 §0.6 的**）

**§3 条件性差异轴**
> 最近邻（vLLM 自己）做了 X = 把 QK-RMSNorm + RoPE 的三次逐层发射保持为未融合；
> 在**条件 C = 测量硬件是 sm90/H100（132 SM）**下它测到融合**更慢**，于是把该 pass 在 **O0/O1/O2/O3 全档设为 `False`**；
> 我们做 Y = 在 **sm120 / 14-SM 级 launch-bound 的小模型 decode** 上重新裁决；
> 因此**当 C 不成立（本机 = sm120、Qwen3-4B、batch 1、FULL cudagraph）时，结论可以不同**。

**§0.6 量级（算术写出）**
- 每层被融合的发射：`rms_norm(q_by_head)` + `rms_norm(k_by_head)` + `rotary_emb(q,k)`
  （`<V>/model_executor/models/qwen3.py:150-167`）⇒ **3 次发射 → 1 次**，净省 **2 次/层**。
- 沿用"每次发射 1.0–1.5 µs"（与决策 #106 的 0.103 ms/步同量级带）：
  `2 × 1.0–1.5 µs × 36 层 = 72–108 µs = 0.072–0.108 ms/步`。
- 相对 7–12 ms/步 ⇒ **0.6%–1.5%**。
- **乐观上界**（若把 RoPE 的 cache 索引与两次 RMSNorm 的 reduce 都算进去，取 3 µs/层）：
  `3 µs × 36 = 108 µs` ⇒ 仍 **≤1.5%**。
- **n\***：这是**每步稳态节省**，`C_setup = 0`（改一个 config 字段），故 `n* = 0`——**但稳态增益本身就是 0.6–1.5%**，门槛没过。

**判定：`KILLED (§0.6: 上界 0.6–1.5% < 2%)`**
**并且**：上游在 H100 上测得的是**负号**（issue #34391 正文，我读过：它只贴了 sweep 命令与
`enable_qknorm_rope_fusion ∈ {True, False}` 两个配置，**正文里没有任何数字**），
即融合核自身可能比三个小核更慢 ⇒ **最可能的真实符号是 0 或负**。

---

### 2.2 C99 / C5（SM120 split-KV / FlashDecoding）—— **本机不成立，无缺口**

**§3 条件性差异轴**
> 最近邻做了 X = 在 sm120 上支持 split-KV（FlashDecoding）；
> 在**条件 C = 上游 FlashAttention 主仓的 SM120 分支仍是 `page_table is None` 断言**下它覆盖不到 paged-KV decode；
> 我们做 Y = 检查 vLLM **自己 vendored 的 FA2 fork**：因此**当 C 不适用（vLLM 不走上游主仓、走 `_vllm_fa2_C`）时结论不同** —— 而结论是"**没有缺口**"。

**§0.6 量级**
- 若真有缺口，收益 = 把 decode 的 KV 读取并行到更多 split。
  attention 读量（4K ctx）= `2 × 36 × 8 × 4096 × 128 × 2 B ≈ 3.0e8 B` ⇒ `0.20 ms/步`，
  占 7–12 ms 的 **1.7–2.8%**（**本身就在门槛线上**）；要 32K ctx 才到 ~14 ms 的量级。
- 但源码层面**缺口不存在**（§0.2.3），故可回收量 = **0**。

**判定：`KILLED (无缺口：vendored FA2 的 split-KV 路径在 bs=1 GQA decode 上是活的)`**
**诚实边界**：我没有跑模型，"split 分支被走到"是**源码路径判定**（`seqlenq_ngroups_swapped` 三个条件
从 `config.json` 逐项对上都成立），**不是实测**。

---

### 2.3 C260 / C197 / C198 / C247 / C155 / C259 —— **量化侧一整簇**

**§3 条件性差异轴（以 C260 为例）**
> 最近邻做了 X = 让 Inductor 生成 group-quant 的 `QuantFP8.forward_native` 取代 CUDA 核；
> 在**条件 C = `IS_QUANTIZED` 为真（模型带量化）**下它才被启用，而该**反事实条件在 vLLM 0.29.0 里被硬编码为假**；
> 我们做 Y = **没有 Y** —— 因为 `<V>/config/vllm.py:129-134` 写着
> *"The optimizations that depend on these properties currently set to False in all cases."*
> ⇒ **当 C 在本机成立时（BF16 Qwen3-4B），这些 pass 根本不参与**。

**§0.6 量级**：可回收 = **0**（pass 未启用，且本机没有该 dtype 的模型）。
**n\***：不适用。

**判定：`KILLED (§0.6 = 0；pass 由硬编码 False 关闭，本机 BF16 路径不经过)`**

---

### 2.4 C168 / C230（FlashInfer 采样器默认）—— **已发货，无缺口**

**§3**
> 最近邻做了 X = sm120/sm121 上默认**关掉** FlashInfer sampler（PR #44405）；
> 在**条件 C = flashinfer 固定为 0.6.13（缺修复）**下这条禁用在当时是必要的；
> 我们做 Y = **不适用** —— `<V>/envs.py:856-860` 的默认已是 `else True`，`<V>/envs.py:49` 亦为 `True`，
> 且本机 flashinfer 是 **0.6.18**（`flashinfer.__version__` 亲验）⇒ **C 已消失，且上游已自己恢复默认**。

**§0.6 量级**：可回收 = 0（已启用）。本机日志 `:28` 亦证 `Using FlashInfer for top-p & top-k sampling.`
**判定：`KILLED (无缺口：默认已为 True，且本机确在用)`**

---

### 2.5 C169（多进程 detokenizer / 前端开销）

**§3**
> 最近邻做了 X = 保留 in-process detokenization；
> 在**条件 C = 前端是 2025 年初的 Python V0/V1 AsyncLLM**（当时测到"API server 已在 LLM 性能的 3% 以内"）下它付得起；
> 我们做 Y = 检查 2026 前端 —— **C 确实过期了**，但 `<V>/v1/engine/async_llm.py:740-822`
> 显示 detokenize 已在**前端进程的后台 asyncio 任务**（`output_handler`）里，`:651` 的注释直接写着
> *"A separate output_handler loop runs in a background AsyncIO task"*
> ⇒ **它本来就不在关键路径上，可回收量无法从"3% 口径"继承**。

**§0.6 量级**：需要一个我用源码**给不出**的数字（后台协程的隐藏程度取决于 GPU 步时长）。
即便照搬 3% 口径，也**只比 2% 门槛高 1 个百分点**，远不能支撑立项。
**n\***：`C_setup` 无法估算 ⇒ **不可算**（按 §4，不可算即不过）。
**判定：`KILLED (§0.6 源头不可从源码界定 + §4 n* 不可算)`**

---

### 2.6 C137（sm120 上 FA4 缺失）—— **理由未过期 + 量级不可界定**

**§3**
> 最近邻做了 X = sm120 默认走 FA2；
> 在**条件 C = FA4 的 CuTe-DSL 核只编译给 family 90/100/110**下它覆盖不到 sm120；
> 我们做 Y = 无（我不做核移植）；**C 在我读到的源码里逐字成立**
> （`<V>/vllm_flash_attn/flash_attn_interface.py:72-86` 的 family 白名单里**没有 120**）
> ⇒ **没有"我们做 Y"这一项，§3 句式不成立**。

**§0.6 量级**：**无法从源码界定** —— 没有任何本机/已发表的 sm120 FA2→FA4 对照；
FA4 的收益同时取决于 batch、head_dim、cudagraph 模式，"上界"我给不出数字。
按 §0.6 的字面要求（"**Do not propose anything whose magnitude you cannot bound from source**"）⇒ 直接出局。
**判定：`KILLED (§1c：X 未落地；§0.6：量级无法从源码界定)`**

---

### 2.7 其余入选格的判定（简写）

| 格 | §3 条件 C | §0.6 上界（算术） | 判定 |
|---|---|---|---|
| **C138** FlashMLA | C = 有 sm120 构建 | 0（`family(90)` / `family(90\|100)` 门在树内逐字仍在） | `KILLED (§1c 未落地)` |
| **C146** GDN prefill | C = cc 12.x 进白名单 | 0（docstring 只写 SM90/SM10.x）；且 Qwen3-4B 是**全注意力**，无 GDN 层 | `KILLED (§3 无差异轴)` |
| **C144** block-size 分支 | C = 加上显式 `120` | 0（`flashinfer.py` 已有 9 处 120 分支）；#56702 未亲验 | `KILLED (§0.6 = 0)` |
| **C147** NVFP4+fp8KV | C = 上游修好 | 0（本机 BF16、且默认后端是 FLASH_ATTN 不是 FLASHINFER） | `KILLED (§3 条件在本机不可构造)` |
| **C160** FP8 scaled_mm | C = 有 sm120 CUTLASS 源 | **已落地**（树内多个 `sm120_*` builder）⇒ 无缺口；且无 FP8 权重 | `KILLED (无缺口 + 本机不可达)` |
| **C104** trtllm-gen FMHA | C = sm120 有 trtllm-gen prefill | 我**跑出** `supports_trtllm_attention(prefill=True)=False` ⇒ prefill 仍缺；decode 已有 XQA | `KILLED (§1c 半落地且无差异轴)` |
| **C233** BlockSparse 重加 | C = V0 弃用约束消失 | 已消失，**但** sm120 non-MLA 后端表里没有 block-sparse 项；且从未有过性能主张 | `KILLED (§0.6 无量化收益主张)` |
| **C172** 小 batch top-p | C = 换 kernel / 换 vocab | 未亲验 #48913；本机 vocab 151,936 ≈ 地图所称 160k ⇒ C **未移动** | `KILLED (C 未移动)` |
| **C216** / **C259** / **C246** | C = ROCm / 量化 / FP8-KV | 0（本机 NVIDIA BF16） | `KILLED (平台/类型不可达)` |
| **C207** | C = SGLang | **本机未装 SGLang**（`ModuleNotFoundError`，见 `NEW_SERVER_ENV` §1） | `KILLED (引擎不在本机)` |
| **C244** | C = 高并发（B≥32）才转正 | 地图自述 **B=1 −5.6%**；且属 SGLang | `KILLED (符号反 + 引擎不在本机)` |
| **C64** | C = 非 H100/1B 堆栈 | 文档测量，无工程缺口可回收 | `KILLED (非优化类缺口)` |
| **C179** | — | 地图自判 "no"，且已被 open PR 占位 | `KILLED (§1c 理由未失效)` |

---

## 3. 单个最佳存活者

### **没有存活者。** 理由如下，且这是一个**有信息量的零**：

1. **§0.6 是决定性的一关**：263 格中我真正能**从源码写出量级算术**的只有 3 条
   （C258 的逐层发射节省、C99 的 attention 读量、C169 的前端开销）。
   - C99 的缺口**不存在**（§0.2.3，源码级否定）；
   - C169 的 `C_setup` 与隐藏程度**无法从源码界定**（§4 要求 n* 可算）；
   - **只有 C258 算出了完整上界：0.6%–1.5%，低于 2% 门槛**，且上游在 H100 上测到的是**负号**。
2. **其余 260 格连"可算"都做不到**，因为它们不是落在**本机不存在的配置**上
   （量化 / MoE / MLA / 多卡 / ROCm / SGLang / 训练 ⇒ ~120 格），
   就是**约束在 0.29.0 源码里逐字未变**（~90 格）。
3. **§0.4 的教训在本轮以新形式复发**：我找到的**唯一新增在飞占位者**（PR **#47979**，DRAFT）
   恰好覆盖了"sm120 服务栈 per-step 优化"这一整片 —— 而这正是若我强行从池① 挑一个大候选时会落进去的地方。
   ⇒ **池① 在本机时点上依然是「够大的都被 DRAFT/OPEN 占着，没被占的都不够大」。**

**若要重开池①，唯一有依据的方向（不在本报告结论内，仅供父代理判断）**：
把 §0.6 的"上界 <2% 即杀"**改成"先把 C258 类的符号测出来"** —— 因为 C258 是本轮**唯一**一条
"差异轴条件性成立 + 上界刚好够到门槛 + 无人占位"的格，而它的关键未知量是**符号**（H100 上为负），
而这**只能实测**（`-cc.pass_config.enable_qk_norm_rope_fusion=True` 与 `False` 的 A/B，n*=0，无需 setup 摊销）。
**成本量级**：1 次服务启动 × 2 臂 × 单个 4k prompt，远小于此前每一轮的 0.15–0.55 GPU·h。

---

## 4. 搜索 / 阅读日志

### 4.1 读过的文件（**主机侧**）
| 文件 | 用途 |
|---|---|
| `notes/FILTER_3AXIS.md`（165 行，全文） | 判据 + 附录 A（§1a/§1b/§1c/§0.4/§0.5/§0.6/§3/§4） |
| `notes/OCCUPANCY_LEDGER.md`（§一–§九，含 7.1/8/9.1–9.8） | 28+ 墓碑；已知死路；sm120 新机解锁范围 |
| `notes/GOAL3_CLOSE_2026-09-15.md`（§0/§1.18/§1.19/§1.21/§1.22/§2/§3/§4） | 池①/池② 的执行状态；C141/B112 先例；3% 量级带 |
| `notes/NEW_SERVER_ENV_2026-09-15.md`（全文） | 机器画像、正确性闸门 T1–T4 |
| `INFERENCE_ACCEL_GAP_MAP.md` | 263 个 `REASON-MAY-HAVE-EXPIRED` 格（**全部**用脚本抽取原文），及其 A/B/C/D/E 段结构 |

### 4.2 读过的文件（**新机侧**，`<V>` = `/root/ccfa_venv/lib/python3.12/site-packages/vllm/`）
| 路径（行号） | 读到的东西 |
|---|---|
| `<V>/config/vllm.py:129-134, 136-141, 225-335` | `IS_QUANTIZED/IS_DENSE` 硬编码 False；`enable_norm_fusion`；O0–O3 全表 |
| `<V>/config/compilation.py:30-64, 100-300, 1060-1290, 1375-1500` | `CUDAGraphMode` 枚举；`PassConfig` 默认；`set_splitting_ops_for_attn_fusion`；`resolve_cudagraph_mode_and_sizes` |
| `<V>/config/attention.py:41-52` | `flash_attn_max_num_splits_for_cuda_graph: int = 32` |
| `<V>/config/kernel.py:20-80` | `IrOpPriorityConfig`（`rms_norm` 默认列表来自平台） |
| `<V>/platforms/cuda.py:83-163, 375-402, 696-716` | sm120/MLA 后端优先级；`get_default_ir_op_priority`（Inductor ⇒ `["native"]`） |
| `<V>/platforms/interface.py:255-262, 1306-1313` | `import_ir_kernels`；`IrOpPriorityConfig.with_default(["native"])` |
| `<V>/vllm_flash_attn/flash_attn_interface.py:52-108, 130-215, 280-345` | `_is_fa4_supported` 白名单（无 120）；FA2 wrapper 的 `NotImplementedError("FA2 does not support num_splits > 1")` |
| `<V>/v1/attention/backends/flash_attn.py:145-200, 275-300, 375-545, 580-605, 1035-1200, 1755-1835` | `use_full_cuda_graph`；`max_num_splits` 决策；`vllm_flash_attn_version` 传递 |
| `<V>/v1/attention/backends/fa_utils.py:1-200` | `get_flash_attn_version`（`major==9→3`, `major==10→4`, **否则 2**） |
| `<V>/v1/attention/backends/flashinfer.py:512-545, 960-1035, 1040-1060, 2050-2085` | sm120 的 9 处 `family(120)` 分支；`get_cudagraph_support`；XQA decode 选择 |
| `<V>/v1/attention/ops/flashmla.py:1-80` | dense/sparse 能力族门（90 / 90\|100） |
| `<V>/v1/attention/backend.py:565-600` | `AttentionCGSupport` 四档语义 |
| `<V>/utils/flashinfer.py:414-515` | `has_flashinfer_cubin` / `has_nvidia_artifactory` / `supports_trtllm_attention` / `can_use_trtllm_attention` |
| `<V>/v1/worker/gpu_model_runner.py:7245-7300` | `_check_and_update_cudagraph_mode`（`min_cg_support` 构造） |
| `<V>/v1/worker/gpu_worker.py:101-105, 218, 428-457` | `use_v2_model_runner` 的使用 |
| `<V>/model_executor/models/qwen3.py:105-167, 276` | Qwen3 的 `q_norm`/`k_norm`（每头 RMSNorm）+ RoPE 顺序 |
| `<V>/model_executor/layers/layernorm.py:37-132` | `RMSNorm.forward_cuda → forward_native → ir.ops.rms_norm` |
| `<V>/ir/ops/layernorm.py:1-80` | `rms_norm` / `fused_add_rms_norm` 的 **torch-native** 实现 |
| `<V>/model_executor/layers/quantization/input_quant_fp8.py:27-120` | `QuantFP8` 有 `forward_cuda`（per-token-group FP8）与 `compile_native` |
| `<V>/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:93-118` | `_resolve_gdn_prefill_backend`（SM90 / SM10.x） |
| `<V>/compilation/passes/fusion/qk_norm_rope_fusion.py:1-60` | 融合模式定义 + `SUPPORTED_..._HEAD_DIMS = (64,128,256)` |
| `<V>/compilation/passes/pass_manager.py:37, 208, 225` | `enable_qk_norm_rope_fusion` 的注册点 |
| `<V>/v1/engine/detokenizer.py:24-70, 168-310` | `Fast/SlowIncrementalDetokenizer` |
| `<V>/v1/engine/output_processor.py:32, 132-450` | `OutputProcessor` / `RequestState.detokenizer` |
| `<V>/v1/engine/async_llm.py:180-184, 625-822` | `output_handler` 后台任务（detokenize 不在关键路径） |
| `<V>/envs.py:49, 130-260, 756-870, 1818-1819` | `VLLM_USE_FLASHINFER_SAMPLER=True` 默认；`VLLM_USE_BREAKABLE_CUDAGRAPH`；`VLLM_ENABLE_CUDAGRAPH_GC` |
| `<V>/third_party/deep_gemm/include/cutlass/…sm120_*` | 树内 sm120 CUTLASS builder（`find` 命中多个） |
| `/root/autodl-tmp/models/Qwen3-4B/config.json` | 36 层 / 32:8 GQA / head_dim 128 / tie_word_embeddings / bf16 |
| `/root/ccfa_results/2026-09-12/smoke/01-nospec.serve.log`（`:11,:16,:28,:37-46,:49`） | **既有运行日志**：`FULL_AND_PIECEWISE` + 35 张 FULL 图 + FLASH_ATTN 后端 + FlashInfer 采样器 |

### 4.3 跑过的命令（**零 GPU，全部亲执行**）
- `python -c "import vllm,torch; …"` → `0.29.0 /root/ccfa_venv/…/vllm/__init__.py`，`2.13.0+cu130`，`NVIDIA RTX 6000D (12, 0)`
- `python -c "import flashinfer; print(flashinfer.__version__)"` → **0.6.18**；`du -sh` flashinfer 111M / vllm 759M
- `python` 探针：`has_flashinfer_cubin()=False`、**`has_nvidia_artifactory()=True`**、
  `supports_trtllm_attention(prefill=True)=False`、**`…(prefill=False)=True`**、`has_flashinfer_moe()=False`、`has_flashinfer_b12x_moe()=False`
- 环境旋钮清点：`import vllm.envs` 枚举全部 `VLLM_*`，列出默认 `False/0/None` 的项
- `nm -D --defined-only _vllm_fa2_C.abi3.so | grep -ci split` → **561**；含 `flash::run_flash_splitkv_fwd<…>`、`flash::num_splits_heuristic`
- `grep -rl "cudagraph" /root/ccfa_results /root/myCCFA` → 命中既有 serve 日志，读到模式与捕获数
- HTTPS（`curl -sL --retry 4 -A "Mozilla/5.0 …"`）：
  - `github.com/vllm-project/vllm/issues/34391`（正文：**无数字**，只有 sweep 命令）
  - `…/issues/25094`（正文 + `EDIT : These numbers should be collected again with torch==2.10 and torch==2.11`；环境为 RTX 5090 / driver 580.82.07）
  - `…/issues/25689`（**全量优化默认清单**，含 `IS_QUANTIZED` 的来龙去脉）
  - `…/pull/47979`（**`"state":"DRAFT"`**）
  - GitHub 仓库内搜索 `issues?q=…`（4 条查询）→ 从内嵌 JSON 抽出 `titleHTML` + `pullRequestState`
  - `raw.githubusercontent.com/vllm-project/flash-attention/main/csrc/flash_attn/flash_api.cpp`（:291, :307-337, :457, :646, :756-763, :805-825）
  - `raw.githubusercontent.com/vllm-project/flash-attention/5824e6e/hopper/flash_api.cpp`（该 commit 的注释被 vLLM 源码引用，我取了该文件）
- `web_search` 两条（RTX 6000D / RTX PRO 6000 Blackwell 显存带宽规格）

### 4.4 **我未能核验的**（明确列出，不得当作缺失证据）
1. `vllm-project/flash-attention` 是**独立仓库**，不在安装树里。
   我读的是 **`main` 分支**的 `csrc/flash_attn/flash_api.cpp`，**不是** vLLM 0.29.0 构建时 pin 的那个 commit。
   ⇒ "vendored FA2 的 split-KV 分支是活的"这一判断，**源码路径**成立，但**未与该 pin 的源文件逐行对齐**。
2. 我**没有**运行任何模型推理，因此 **`num_splits_heuristic` 在 208-SM 上实际选出多少 split、以及它是否真的进入
   `run_flash_splitkv_fwd`，未实测**。
3. `C247` 声称的 "−38% on ROCm sparse MLA"、`C230` 的 "three CI failures"、`C172` 的 "~40 rows, vocab 160k"、
   `C258` 的具体回归数字 —— 这些**正文数字我都没能取到**（GitHub 评论体在无 JS 的 HTML 里未渲染；
   我只拿到 issue 描述体，其中 #34391 **没有任何数字**）。
4. `C147`（NVFP4 + fp8 KV 在 sm120 的修复状态）、`C144`（issue #56702 的三个具体站点）、
   `C172`（PR #48913 正文）、`C104`（issue #3263 的 in-thread 更正）、`C106`（issue #2555）、
   `C107`（FlashInfer #1147 / merged #4259/#4714/#4199）、`C114`（PR #42404）、
   `C139`/`C143`（MXFP4/MXFP8 核）、`C157`/`C183`/`C216`/`C240`/`C259` —— **未亲验**。
5. vLLM 的 `git log` / commit 历史**不可用**（site-packages 无 `.git`），因此"某 PID 何时改的"我一律不断言。
6. **SGLang 未安装**（`ModuleNotFoundError`），所以任何 SGLang 侧候选（C207/C244/C166/C161/C181 等）
   的"落地"我只能靠外部检索，本报告一律按"引擎不在本机"处理。

---

## 附录 A —— 顺手核验（地图内"YES"但另一侧的格）

| 格 | 我读到的 | 判定 |
|---|---|---|
| **C41**（RMSNorm+quant fusion 的 revert，原因限于 "CUDA < 12.4 on Ada / ROCm FP8"） | 本机 CUDA 13 + sm120 ⇒ 条件不适用；但该 pass 由 `enable_norm_fusion` 门控，**BF16 下为 False**（`<V>/config/vllm.py:136-141`） | 条件过期但**无杠杆** |
| **C51**（FlashInfer block_size16/head_size256 Blackwell workaround 已 revert） | 地图记 `merged #36987`；本机 flashinfer **0.6.18** ≥ v0.6 ⇒ 依赖侧满足 | 已落地 ⇒ **无缺口** |
| **C54**（chunked local attn + HMA 的回归限于 MI300X/Llama-4） | 条件为 AMD 专属；本机为 NVIDIA dense full-attention | 不可达 |
| **C55**（`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 的 disable 提案） | B200 + NVFP4 专属；本机 BF16 ⇒ 不参与 | 不可达 |
| **C112**（FA2 `.item()` sync 回归，RTX 6000 Pro 受害最重） | 地图记 `#47134 merged` 已修 + open `#48353` 第三处；**我未亲读任一 PR** | **未亲验** |
| **C156**（MLA Fully Support 的 revert，限于 absorbed-MLA + >96GB 多卡） | Qwen3-4B **非 MLA** ⇒ 不可达 | 不可达 |

## 附录 B —— §2.7 简写表引用的原文（逐字，供复核）

- **C216**："measured on ROCm 7.2.3 / gfx950 (MI355X) with GLM-5.3 MXFP4 and use_inductor_graph_partition."
- **C259**："yes, plausibly — the disabling PR calls itself \"temporarily\", the mechanism is a frozen-config bug rather than a hardware property"
- **C207**："**Yes, in the opposite direction from the usual.** `bench_serving` has since moved to `sglang.benchmark.serving` and been re-timed"
- **C244**："The fusion's sign flips with batch (−5.6% at B=1 → positive at B≥32), so it expires for a high-concurrency server; for a single-stream latency rig the fusion is a regression."
- **C246**："The culprit is a *kernel's* scale handling, not the dtype: in the same report `VLLM_USE_DEEP_GEMM=0` restores GSM8K 0.8302 from 0.7089"
- **C64**："measured on Llama-3.2-1B on H100 with the then-current Inductor/torch.compile stack."
- **C179**："no" / S3b: "OCCUPIED (open PR #38703 …; open PR #34876 …)"
- **C233**："YES — the stated reason was the V0→V1 deprecation sweep, a constraint that no longer exists in vLLM 0.29.0"
- **C234**（附）："The stated reason is a security/privacy default, not a performance or technical constraint"
