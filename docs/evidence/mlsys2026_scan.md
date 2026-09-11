# MLSys 2026 论文集扫描（关闭 §11 第一条缺口）

**执行**：2026-09-11 ｜ **方式**：`mlsys.org/virtual/2026/papers.html?filter=titles`（JS 加载，需经社区全量清单）＋逐篇摘要核实
**规模**：录取 **135 / 504 = 26.8%**
**目的**：确认 MLSys '26 是否有论文占据 `#6`（drafter 网络算力分配）的格子；顺带复核 `#12`、`#1` 的拥挤度

> 注：`proceedings.mlsys.org/paper_files/paper/2026/hash/...` 形式**不存在（404）**；正确入口是 `mlsys.org/virtual/2026/`（oral/poster 编号页）。

---

## 一、Speculative Decoding 分区（4 篇，全部逐条核实）

| 论文 | 单位 | 做什么 | 是否占 `#6` 的轴 |
|---|---|---|---|
| **PRISM: Parametrically Refactoring Inference for Speculative Sampling Draft Models**（oral；arXiv `2602.01762`） | CUHK & ICT, CAS | 见 §二 | **不占机制格，但攻击动机** |
| **HELIOS: Adaptive Model And Early-Exit Selection for Efficient LLM Inference Serving**（oral；arXiv `2504.10724`） | UT Austin & NVIDIA | 见 §二 | **不占机制格，但挤压叙述** |
| Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding | CMU & MIT & UC Berkeley | 利用推理模型的激活稀疏做**自投机**，无需辅助 drafter | 否（自投机家族） |
| SpecDiff-2: Scaling Diffusion Drafter Alignment | UVA | 扩散 drafter 对齐，支持并行起草、降低拒绝 | 否（drafter 家族/训练侧） |
| Speculative Decoding: Performance or Illusion? | UC Berkeley | 生产级引擎（vLLM）上首次系统测量，揭示**真实 batch/负载下收益受限** | 否（测量论文；但它是"投机在高 batch 收益缩水"的权威佐证） |

## 二、两条最近命中的逐字证据

### PRISM —— 攻击 `#6` 的**动机**

> "the pursuit of better draft quality has driven a trend toward parametrically larger draft models, which inevitably introduces substantial computational overhead. While existing work attempts to **balance the trade-off between prediction accuracy and compute latency**, we address this fundamental dilemma through **architectural innovation**. We propose PRISM, which disaggregates the computation of each predictive step across different parameter sets, **refactoring the computational pathways of draft models to successfully decouple model capacity from inference cost**."

并宣称在已高度优化的引擎上把解码吞吐提升 **>2.6×**。

**为什么它不占格**：PRISM 是**训练期架构重构**（静态设计），不做运行时按 (bs, ctx) 分配。
**为什么它危险**：`#6` 的立论是"drafter 算力是要被分配的资源"；PRISM 主张"容量与推理成本可以**架构性解耦**"。**必须回答**：既然容量不必按成本付费，运行时分配还解决什么？——`#6` 的答案只能落在 PRISM 未覆盖的部分：**上下文相关**的成本（如 DFlash 在 185k 的**全上下文重扫**，`#54691` 实测 4.4× 损失），那不是参数规模问题，而是每步扫描范围问题。

### HELIOS —— 挤压 `#6` 的**叙述**

> "HELIOS employs **multiple models** and **dynamically switches between them** to collectively maximize the number of tokens that exit early… **only loads the weights of the most likely to be used layers**, yielding memory savings which is then re-purposed to increase batch sizes… employs **real-time profiling** … and **adaptively switches between models by tracking tokens in real-time**."

结果：**1.48× 吞吐、15.14× 更大 batch**。

**为什么它不占格**：它属于 **early-exit / EE-LLM 家族**（目标模型自身的层深 + 多模型切换），不是**独立 drafter 的网络容量**；且它切换的是**模型**，不是**同一个模型的大小**（与 MemSpec 的"选哪个 drafter"同类）。
**为什么它危险**：MLSys '26 上已经有一篇 oral 在卖"**运行时自适应深度 + profiler 驱动 + 省下的内存再分配给 batch**"。`#6` 若只写"我们按输入动态选深度"，会被直接对标 HELIOS。

## 三、对其它存活/归档项的影响

**`#12`（MoE EP 不均衡）——拥挤度进一步确认（MoE Inference 分区）**

| 论文 | 单位 | 做什么 |
|---|---|---|
| CRAFT: Fine-Grained Cost-Aware Expert **Replication** | UofT & AWS | 细粒度成本感知的专家副本策略，跨设备均衡负载（= placement/replication 半边） |
| From Tokens to Layers: Stall-Free Scheduling for MoE Serving with **Layered Prefill** | SNU | 沿**层维**切分以避免专家导致的负载不均衡（= 调度半边） |
| Demystifying the Mixture of Experts **Serving Tax** | UW | 系统测量：MoE 比等 FLOP 的稠密模型**差 2–3×**，并归类开销来源 |

⇒ 与已 ship 的 SGLang/NVIDIA **LPLB** 一起，构成 `#12`"机制已被占"的第三重证据。这**加强**（而非改变）此前结论：`#12` 仅作对冲，且必须先过 ≤3 天残余不均衡判定。

**归档项的相关命中（仅记录，不改变结论）**：KV Cache Management 分区有 Kitty（2-bit KV 量化）、OPKV（可召回稀疏）、SkipKV（跳过 thinking token 的 KV）、FlexiCache、Span Queries、MorphServe（运行时层交换 + KV 重设）；Sparsity 分区有 BLASST、Attribution-based Sparse Activation。⇒ 这些压在 `#16`/`#1` 的邻域，不占 `#1` 的"未提交 KV 分级"格。

## 四、结论

1. **`#6` 的机制格在 MLSys '26 仍未被占**（没有"运行时分配独立 drafter 网络容量"的论文）。
2. **但两侧各有一篇 oral**：PRISM 攻击动机（容量⟂成本）、HELIOS 挤压叙述（运行时自适应深度已有人卖）⇒ `#6` 的 pre-reg 新增**必答项 ④**：为什么在架构性解耦之后仍需要运行时分配（答案必须落在 PRISM 未覆盖的上下文相关成本上）。
3. **`#12` 的"已被占"结论在 MLSys '26 得到独立确认**。
4. §11 的 MLSys 条目可**关闭**；ASPLOS / ISCA / ATC / SOSP / SC '26 **仍未扫描**（本次尝试取社区清单失败），相关结论继续标"暂定"。
