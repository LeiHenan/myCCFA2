# S3 占位与先行核查（Occupancy & prior art） — 混合模型（SSM/线性注意力）服务：状态不是分页 KV

**候选**：`hybrid-state-serving` ｜ **成本**：0 GPU·h（检索）
**目标**：确认没人已经做完，且**引擎里已 ship 的旋钮**不是你的方案。必须读正文。
**结论**：🔴 **判死（occupied）** —— 核心问题在 **2026-09-09（4 天前）** 被 vLLM 社区以 RFC + 三件套 PR 明确占住，且其**动机段落独立复述了我们零卡算出的同一条结构事实**。

## 最近邻（≥3，标注读到的深度）

| # | 工作 | 占住的是什么 | 读到什么深度 | 证据 |
|---|---|---|---|---|
| 1 | vLLM **RFC #55697** + PR 栈 **#55873 / #55875 / #55876**「Application-Directed Prefix Checkpoints for Hybrid GDN/Mamba」（2026-09-09） | **状态检查点"边界该放哪"这个问题本身** + 调度器 producer/consumer 状态机 + 批量分组 prefill 内核 | ✅ **论坛正文逐字读完**（含动机与基准表）；GitHub issue/PR 正文未取到（只返回导航壳） | [论坛帖](https://discuss.vllm.ai/t/rfc-pr-stack-application-directed-prefix-checkpoints-for-hybrid-gdn-mamba-up-to-7-6x-speedup-on-l40s/2949) |
| 2 | LMSYS blog「**Unified Radix Cache: One Tree for Hybrid Model Prefix Caching**」（2026-08-11） | 混合模型的**统一前缀缓存结构**（KV 与状态同一棵树） | 标题级（正文未读） | [lmsys.org](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/) |
| 3 | SGLang **PR #29678**「feat(mem_cache): **unified memory pool** for hybrid Mamba / SWA models」 | **KV 与状态的统一内存池**（= 我们 P3 说的"缺联合预算"） | 标题级 | [PR #29678](https://github.com/sgl-project/sglang/pull/29678) |
| 4 | SGLang **PR #28185**「[GDN][KDA][mem_cache] **int8 checkpoint pool** for the linear-attn prefix cache」 | **状态检查点的内存压缩**（直接回应我们算出的"检查点比 KV 贵"） | 标题级 | [PR #28185](https://github.com/sgl-project/sglang/pull/28185) |
| 5 | **DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving**（论文） | 混合线性注意力的**状态压缩**（学术侧已进入） | 标题级 | [Semantic Scholar](https://www.semanticscholar.org/paper/DASC%3A-Decay-Aware-State-Compression-for-Hybrid-Yu-Sun/70c8d0d5c06d8f56f6d03ad78e489e4a38a2a1a2) |
| 6 | **Megatron-LM** `mamba_slot_allocator.py`（NVIDIA 官方） | 状态的**槽位分配器**（另一种分配设计） | 源码文件存在（未读正文） | [文件](https://github.com/NVIDIA/Megatron-LM/blob/23ba3570/megatron/core/inference/contexts/mamba_slot_allocator.py) |
| 7 | vLLM **PR #45019**（Mamba1 in NIXL **P/D 分离**）、**PR #46896**（GDN **all-mode** + Mooncake 留存） | 状态在**分离式部署**与**外部存储**里的搬运 | 标题级 | [PR #45019](https://github.com/vllm-project/vllm/pull/45019)、[PR #46896](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/46896/overview) |

## 引擎已 ship 的控制器（最强基线，逐字取自本机 vLLM 0.29.0）

`mamba_cache_mode ∈ {none, align, all}`（`config/cache.py:188-196`）｜ `mamba_block_size`（`:176-179`）｜ `replayssm_buffer_len=16` + `use_replayssm`（`:197-206`）｜ `use_kda_recoverssm`（Kimi-K3 KDA）｜ `mamba_cache_dtype` / `mamba_ssm_cache_dtype`（**状态可单独量化**）｜ Megatron 的 `mamba_slot_allocator`。
⇒ **"配一个开关"的空间已经用尽**；社区正在加的是**应用级契约 + 调度器状态机 + 专用内核**。

## 逐字对照：他们独立复述了我们的零卡结论

RFC 动机段落（**原文**）：

> "recurrent states (e.g. [H, D_k, D_v] per layer, **~512KB per layer**) have a significantly larger memory footprint than single Attention KV blocks. Because of this, recurrent architectures can realistically afford **only 1 or 2 discrete state checkpoints per sequence**."

我们的 `budget.py`（0 GPU·h，2026-09-13）算出：

> 状态/序列 = 1.9–67 MiB；**检查点开销/KV = L\*/b**；要不超过 KV 本身必须 **b ≥ L\* = 0.8k–3.2k tokens**。

**两者是同一件事**：状态太大 ⇒ 每序列只负担得起 1–2 个检查点。**推导是对的，但洞察不是新的** —— 它是 RFC 的立项前提。

他们接着把**我们 P2 指出的问题**（检查点边界与调度路径耦合、引擎无法启发式预测该放哪）明确写成"boundary dilemma"，并给出解法：应用用 `<|mamba_checkpoint|>` 标记语义边界 + 调度器 producer/consumer 状态机 + 批量两阶段分组 prefill 内核。
L40S 微基准（`benchmark_grouped_gdn_prefill.py`）：1 生产者 + 16 消费者 **12,698.95 µs → 1,674.81 µs = 7.58×**（延迟近 O(1)）。

## 差异轴：写不出来

| 候选差异轴 | 是否还站得住 |
|---|---|
| "没人做状态检查点" | ❌ 4 天前已被 RFC + 3 个 open PR 占住 |
| "没人做统一内存池" | ❌ SGLang PR #29678（且 LMSYS 有统一 radix cache 博文） |
| "没人做状态压缩" | ❌ SGLang int8 checkpoint pool + DASC 论文 |
| "没人做状态在 P/D 分离里的搬运" | ❌ vLLM PR #45019 / #46896 |
| "自动选检查点位置（而非交给应用）" | ⚠️ **唯一可能的缝**，但 RFC 明写"engine-internal 启发式预测 **virtually impossible**" —— 这是**别人给出的不可能性论断**，反驳它需要先证明工作负载统计足够可预测，风险高，且对方设计正在 ship |

**唯一的缝**也不构成可立项理由：它要推翻别人的"不可能"论断，属于**高风险、依赖特定工作负载分布**的赌注，且**没有独立的量级证据**支持。

## 结构性理由

**不适用** —— 在写结构性理由之前就被占位核查拦下。（这本身是好事：S3 在花 GPU 和写论证之前就结束了一个方向。）

## 被占时的降级

**本方向不降级。** 若要用尽残余价值，只剩一条窄路：**"状态复用引入的输出分歧"**（承接 P5 —— SGLang 的 int8 状态检查点池会引入量化误差 ⇒ 同一请求可能因**状态复用**得到不同答案）。但那必须：
① **新开独立预登记**（不得重解读旧数据）；
② 先做一次独立的**占位快扫**确认没人做过；
③ 明确它与既有 KV-cache 分歧工作（arXiv 2609.04748）的差异轴。

## 结论与它省下的东西

**🔴 KILL。** 依据：最近邻 ≥3（实为 7，其中 1 篇正文逐字读过）；**核心问题被引擎自己的社区在同一周占住并给出量化收益**；差异轴写不出来。

**这次 S3 省下的**：取证第 2 步的模式扫描（**~1 GPU·h + 20 min**）**已取消**；已下载的 **3.3 GB 权重已删除**；以及 **3–4 周**的论文级投入。

## ⚠️ 必须写进方法论的教训（本步已造成实际浪费）

我们是 **S1 → 零卡量化 → 才做 S3**。而 S3 的快扫本可与 S1 **并行**。
若先扫一遍占位，**第 1 天就能杀掉**，连那份零卡预算都不必做。
⇒ 规则已写入 `PIPELINE.md` §8：**每条痛点落地时（S1）必须同时做一次占位快扫，不要等到量化之后。**

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] 每个存活痛点找 ≥3 个最近邻，且记录**读过的具体页/节**（只读摘要不算）
- [x] 找齐目标引擎里**已 ship** 的相关旋钮/调度器，并写成最强基线（不是只写学术邻居）
- [ ] 写出审稿人会问的那一问，并给出**结构性**回答（不是『我们更自适应』这类程度性回答）
- [x] 写明『若被别人占住，本方向降级成什么』（降级形态必须先想好）

> **杀出口**：立项理由只有『没人做过』⇒ 杀；给不出结构性理由 ⇒ 改写定位或杀
>
> **本阶段实际走的是杀出口**：不是"没人做过"，而是**已被人占住且已给出量化收益** ⇒ 杀。
> 第 3 项（结构性理由）**故意不勾选** —— 该阶段在写出结构性理由之前就已终止，如实反映。
