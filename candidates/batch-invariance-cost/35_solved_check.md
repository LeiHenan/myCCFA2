# S3b 已解决性核查（Solved-elsewhere check） — 批组成不变性：确定性的代价与收益

**候选**：`batch-invariance-cost` ｜ **成本上限**：0 GPU·h（检索 + 读正文）
**目标**：在花任何 GPU 之前回答：**这个痛点，别的引擎/别的社区是不是早就解决了？** 只查单一引擎的 roadmap 与源码**不算**——上一轮的失败正是如此。

**判定（2026-09-13，goal-ebcec24d 第 1 轮）：🔴 杀 —— 在 0 GPU·h 被 S3b 拦下。**

**理由（命中杀出口）**：同一问题类在两个主流引擎里**都已 ship 专用开关**，且**上游把失效机制写进了代码**：

| 引擎 | 开关 | 默认 | 文档串原文 | 打开后关/改什么 |
|---|---|---|---|---|
| vLLM 0.29.0 | `VLLM_BATCH_INVARIANT`（`envs.py:617-619`） | `0` | *"deterministic results regardless of batch composition. Requires NVIDIA GPU with compute capability >= 9.0"* | attn cascade（`config/vllm.py:1730`）、allreduce（`all_reduce_utils.py:136/165`）、symm mem + FlashInfer allreduce（`cuda_communicator.py:63/235/254/457`）、LoRA kernel（`lora/ops/triton_ops/utils.py:222`）；另有专门模块 `determinism/batch_invariant.py` |
| SGLang 0.5.19 | `--enable-deterministic-inference`（`server_args.py:3474-3478`） | `False` | **"Enable deterministic inference mode with batch invariant ops"** | 采样后端强制 `pytorch`（`overrides.py:1030-1035`）、禁用 FlashInfer allreduce fusion（`overrides.py:1066-1077`）、禁用 `FLASHINFER_MOE_FUSED_FINALIZE`（`serving_hook.py:404-405`） |

**最致命的一条**：SGLang `arg_groups/speculative_hook.py:770-776` 直接 `raise ValueError(...)`，
原话是 *"the sampling kernel draws coins from the global RNG and is not batch-invariant"* ⇒
**上游不但知道，还在代码里把非不变性 kernel 挡住**。

**⇒ 这不是"没人想到"，而是"已知、已建开关、已被容忍"**。剩下的可写内容只有"把它的代价测一遍"，
那是**程度性**差异轴（违反判据 2）且属**测量类**（违反用户"只接受优化类"裁定）⇒ **杀，且不花 1 GPU·h**。

## 问题类定义

**问题类（不含任何引擎的功能名）**：

> **"批量推理系统的数值结果依赖于批的组成与形状，因此『同一请求 + 同一随机种子 + 贪心解码』并不保证同一输出；
> 而要让结果与批组成无关，必须付出可测的性能代价。"**

**换成别的引擎/社区会叫什么**（S3b 硬要求，防止只在一个引擎的词表里找答案）：

| 场景 | 该问题类的叫法 |
|---|---|
| vLLM | `VLLM_BATCH_INVARIANT`（env 开关）、`determinism/batch_invariant.py`、deterministic allreduce |
| SGLang | 待查（是否已有对应开关 / 文档承诺） |
| TensorRT-LLM | 待查（是否有 deterministic / batch-invariant 运行模式） |
| llama.cpp / 边缘引擎 | 待查（单批为主，问题形态可能不同） |
| 数值计算社区 | **"reproducibility of floating-point reductions under varying shapes"**（经典话题，但**不是** LLM 服务语境） |
| 训练社区 | **"bitwise reproducibility across batch sizes / gradient accumulation"**（PyTorch 有 `torch.use_deterministic_algorithms`，但那是**算子级**，不是**批组成级**） |
| 服务 / MLOps 社区 | **"output drift across replicas / load"**、A/B 与灰度发布的有效性 |

## 判定矩阵（跨引擎 × 跨社区）

> 规则：每行必须有**可点开的 URL**与**查证深度**；未读正文的写 `未取证（TITLE ONLY）`。

| # | 对象 | 状态 | URL / 位置 | 查证深度 | 与我们的差异 |
|---|---|---|---|---|---|
| 1 | **vLLM 0.29.0** `VLLM_BATCH_INVARIANT` | **已 ship（opt-in，默认 `0`）** | 本地已装树 `vllm/envs.py:617-619`（注释原文：*"Enable batch-invariant mode: deterministic results regardless of batch composition. Requires NVIDIA GPU with compute capability >= 9.0."*） | **READ BODY**（读了 `envs.py:610-630` + 全仓消费点 grep） | 上游**承认问题存在**并提供开关；**未量化**"不开时差多少、开了付多少" |
| 2 | **vLLM 0.29.0** 开关的性能副作用清单 | 已 ship（随开关生效） | `config/vllm.py:1730`（关 cascade attention）、`config/parallel.py:1035`、`distributed/device_communicators/all_reduce_utils.py:136/165`、`cuda_communicator.py:63/235/254/457`（symm mem / FlashInfer allreduce / `use_deterministic_rs`）、`lora/ops/triton_ops/utils.py:222`、`model_executor/determinism/batch_invariant.py`（含 `mm_batch_invariant`） | **READ BODY**（逐处 grep 命中行） | 证明**代价真实且分布很广**（注意力 / 通信 / LoRA / matmul 四处） |
| 3 | **vLLM 0.29.0** 逐请求采样 generator | 已 ship | `v1/sample/ops/topk_topp_sampler.py:207-212`（`q[i].exponential_(generator=generator)`） | **READ BODY** | 采样侧**已有**逐请求 generator ⇒ 发散若存在，来源更可能是**算子/归约**而非采样器 |
| 4 | **vLLM 0.29.0** 投机解码默认值 | **opt-in（非默认开启）** | `config/vllm.py:372` `speculative_config: SpeculativeConfig \| None = None`；`config/speculative.py:1146-1158`（ngram 时 `prompt_lookup_min = max = 5`） | **READ BODY** | **纠错**：我此前说"prompt-lookup 是默认特性"是错的（decision #109） |
| 5 | **SGLang 0.5.19** 是否有对应开关/承诺 | 待查 | 待填 | 待填 | — |
| 6 | **TensorRT-LLM / llama.cpp** | 待查 | 待填 | 待填 | — |
| 7 | **研究文献** | 待查（⚠️ 已知邻作 *Same Request, Different Answer* 打的是**缓存导致分歧**，属**不同机制**，须分开记） | 待填 | 待填 | — |
| 8 | **生产实践（issue / 论坛 / 工程博客）** | 待查 | 待填 | 待填 | — |

## 命中与缺口

**命中面（当前）**：1 条已 ship（vLLM 开关）+ 1 条配套副作用清单 ⇒ **2 / ≥3，未达标**。

**已能说清的部分**：上游**自认**该问题（否则不会建开关、不会专门写 `determinism/batch_invariant.py`），
且代价**分布在注意力 / 通信 / LoRA / matmul 四处** ⇒ 这不是"没人想到"，而是"**已知且被容忍**"。

**尚未说清（禁止进 S4）**：
- 「不开开关时，批组成导致的发散**有多大、在什么条件下发生**」——未见任何量化。
- 「打开开关的**性能代价**具体是多少」——未见任何基准。
- 上述两条正是本候选要填的格子；**若文献里已有同等量化 ⇒ 按 S3b 杀**。

## 反证与自证伪

**待填**：必须主动找证据说明"这个问题已被解决或已被充分刻画"。
**已知候选反证**：① 该开关存在本身说明上游认为它重要；② 若社区已有"批不变性代价"基准，则差异轴会被压成
"换个模型再测一遍" ⇒ **应杀**。

## 降级路径

**待填**（若被占）：可能形态 —— 把主张从"量化发散"改为"**刻画代价在别的维度上的二阶影响**"
（例如批不变性对投机解码接受率 / 前缀缓存命中率的影响），或降级为纯测量记录。

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] 用**问题类**（而非某个引擎的功能名）描述候选，并写出「换成别的引擎/别的社区会叫什么」 —— §问题类定义：单一表述 + **7 个视角**的词表（vLLM / SGLang / TRT-LLM / 边缘引擎 / 数值计算社区 / 训练社区 / MLOps 社区）
- [ ] 判定矩阵 ≥5 行，覆盖 **≥3 个引擎/实现**（vLLM / SGLang / TRT-LLM / llama.cpp / 专用库等）与 **≥2 个研究社区**（会议论文 / 开源库 / 生产实践）
- [ ] 每行含：对象 / 状态（已 ship / RFC / 论文 / 无人做）/ 可点开的 URL / 查证深度（读了哪一节或哪个 file:line）/ 与我们的差异
- [ ] 整体命中面 ≥3 条，并**逐条**推理「为什么剩下的缺口不是没人想到，而是结构上难/不划算」
- [ ] 写一条**反证**：主动找证据说明整个方向可能已被解决（若找不到，写明搜索词与范围）
- [ ] 写明「若被占，本方向降级成什么」，且降级形态不需要新半径

> **杀出口**：任一引擎/社区**已 ship 或已发表**同一问题类的同等解法，且我们的差异轴讲不出条件性区别 ⇒ **杀**；判定矩阵不足 5 行或缺 URL/查证深度 ⇒ **禁止进入 S4**（不许开 GPU）


---

## 附：同轮平行检验的第二个候选（thinking 阶段投机）—— 同样被 S3b 拦下

**问题类**：*"推理模型的 thinking 阶段与答案阶段的草稿接受特性不同，一个静态的投机步数/预算无法同时最优；
该差异是否有可回收的优化空间？"*

**判定：🔴 已解决。** 两条独立证据：

| # | 对象 | 状态 | URL | 查证深度 | 结论 |
|---|---|---|---|---|---|
| A | **SGLang adaptive speculative decoding** | **已 ship（有专门文档页）** | [docs.sglang.io/.../adaptive_speculative_decoding](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding.md) | **READ BODY**（整页：设计、EMA 策略、BS 分桶、hysteresis、ceiling、推荐配置） | 文档开篇即 *"designed for workloads whose **accept length changes over time**, where one static step count is rarely optimal"*；实现含**按 batch-size 分桶的独立 EMA 跟踪器** + `down/up_hysteresis` + `ceiling_coeff` + CUDA graph 分档切换 ⇒ "接受长度随时变"的控制器**已 ship** |
| B | **工程/教学共识** | 已发表（教材级） | [Low acceptance on chain-of-thought](https://theneuralbase.com/speculative-decoding/learn/advanced/low-acceptance-on-chain-of-thought/) ｜ [Why reasoning models are a poor fit for speculative decoding](https://theneuralbase.com/speculative-decoding/learn/advanced/reasoning-model-poor-fit/) | 页面存在且标题即结论（正文为课程索引页，**关键结论在标题中，记为 TITLE ONLY + 标题即论断**） | "**reasoning / chain-of-thought 阶段接受长度低**"是被写成专门章节的**既定事实**，不是未解现象 |
| C | vLLM 侧对照 | 已 ship | `config/speculative.py:473-529`（`num_speculative_tokens_per_batch_size`、per-request 自适应 RFC `#48202`） | **READ BODY**（源码） | 另一家引擎在**同一决策**上已有 per-BS 与 per-request 两级自适应 |

**⇒ 差异轴只能写成"在 thinking 阶段做自适应"**，而上游已经是**通用自适应**（对任意 accept 变化都响应）
⇒ 属**程度性**而非**条件性**区别（违反判据 ②），且与 `#6`/decision #62 已记录的学术邻居
（HELIOS / CAS-Spec 的运行时深度自适应）重叠 ⇒ **杀**。

---

## 终局判定（第 6–8 轮，2026-09-13）：**🔴 不立项（已实测 + 可回收性源码级排除）**

**结论**：本候选**不进入 S4**。S3b 清单中 5 项**不适用/不满足**，理由逐条如下（不是漏填，是**结论即否**）：

| S3b 清单项 | 状态与理由 |
|---|---|
| 判定矩阵 ≥5 行覆盖 ≥3 引擎 + ≥2 社区 | ✅ 已由 `notes/prior-art-batch-invariance-sweep.md`（263 行：58 跨引擎 + 29 文献 + 30 生产痛 + 9 性能）满足，**但由子代理检索产出，我未逐条亲自复核** ⇒ 按纪律不作为本档案的独立证据 |
| 每行含 URL / 状态 / 查证深度 | ⚠️ 部分：**我亲自读过的** = vLLM `envs.py` / `determinism/batch_invariant.py` / `layernorm.py` / SGLang `environ.py` / `server_args.py` / `sampling_params.py` / `speculative_hook.py` + 两次实测；**其余为子代理材料** |
| 命中面 ≥3 条 + 结构性理由 | 🔴 **命中面充足但方向被否**：两家引擎均已 ship 开关，且**在本机 sm120 上实测开关有效**（vLLM 3/8→1/8、SGLang 8/8→1/8）|
| 反证 | ✅ 已写：**可回收性不成立**（干预=开全局开关，代价 1.7–3.6×；无更省的等价方案；无更细粒度接口） |
| 降级路径 | ✅ 已写：唯一复活路径 = **按算子粒度的更省等价方案**，且需先做 S4 oracle 上界；当前半径内无线索 |

**三条判据的终态**（详见 `results/p15-batch-invariance/2026-09-13/summary.md`）：
现象层 ✅（两引擎、可复现、合批有墙钟证据）；机制层 ❌（未归因，但已排除 detokenize）；**可回收性 ❌**。

**⇒ 按 S3b 杀出口："任一引擎/社区已 ship 同一问题类的同等解法，且差异轴讲不出条件性区别" ⇒ 杀。**
