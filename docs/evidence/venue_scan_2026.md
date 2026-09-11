# ASPLOS / SOSP 2026 扫描 + DA-MoE 正文精读

**执行**：2026-09-11（无卡）｜ **目的**：关闭 §11 的架构/系统会议缺口；精读 `#12` 最关键的在办工作

---

## 一、ASPLOS 2026（14.5% = 152/1048，Spring 9.6% + Summer 15.7%）

**Speculative Decoding 分区（2 篇）**

| 论文 | 单位 | 做什么 | 关联 |
|---|---|---|---|
| **DFVG**: Heterogeneous Architecture for Speculative Decoding with **Draft-on-FPGA and Verify-on-GPU** | SJTU 等 | 跨设备流水：FPGA 起草、GPU 验证 | `#17` 草稿放置（已归档）—— 强化"草稿放置/异构"已被占 |
| **SwiftSpec**: **Disaggregated** Speculative Decoding and Fused Kernels | ByteDance Seed & UChicago | 系统级分离 + 内核融合 | 本仓库已记录 `SwiftSpec 2506.11309` 占据"搬运草稿状态"（`#17` 归档依据） |

**`#6` 相关（RL 训练分区）**

| 论文 | 单位 | 做什么 | 判定 |
|---|---|---|---|
| **TLT: Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter**（arXiv `2511.16665`） | MIT HAN Lab & NVIDIA & ETH & IBM & UMass | "Train an **Adaptive Drafter** on idle GPUs during long-tail generation and use an **Adaptive Rollout Engine to select speculative-decoding strategies per input batch**" | **不占 `#6`**：它是**训练期**训练 drafter + 按 batch 选**策略**（长度/策略轴），不是运行时分配独立 drafter 的**网络容量**。**Tier B**（标题 + 会议列表一句话），正文未读 |

**其它相关**：MSCCL++（MSR，**面向推理的 GPU 通信抽象**）—— `#12` 通信侧的基础设施；MoE-APEX（自适应精度专家 offload）；Shift Parallelism（运行时切换并行策略）。

## 二、SOSP 2026（15.9% = 62/390）

| 论文 | 单位 | 为什么相关 | 判定 |
|---|---|---|---|
| **Democratizing MoE LLM Decoding via Barrier-Free Expert Parallelism** | USC & Google DeepMind & UT Austin | 与 `#12` **同赛道**（EP 解码的"无屏障"= 直接针对专家并行中的同步/不均衡） | ⚠️ **Tier B（仅标题级）**——**必须读正文**才能判是否与 `#12` 重叠，已列入 §11 缺口 |
| **MorphKernel: Taming Dynamism on GPUs（Cross-SM Kernel Fusion）** | SJTU IPADS | 明确针对"**dynamic and imbalanced GPU operators such as decoding attention and MoE expert activation**"，跨 SM 融合，平均 **1.3×** | 占 `#12` 的**dispatch/融合**半边（`#12` 剩通信侧） |
| Multi-LLM Serving at Production Scale（`Janus2026/Janus`） | PKU & JD 等 | 与 EP-LB 证据库 §0 的 Janus 一致（4 节点规模） | 佐证 `#12`/`#20` 拥挤 |

## 三、DA-MoE `2607.23099v2` 正文精读（Tier A，全文 HTML）

**身份**：NTU + **NVIDIA（Taipei）** + Harvard；`cs.AR`，2026-07-28；"conducted during an internship at NVIDIA"。

**它做什么（逐字）**

> "Existing serving systems typically select fused-MoE kernels using **static token-count buckets, ignoring the per-expert routing distribution that determines tile padding**, memory reuse, and kernel efficiency."

- 量化 **NVFP4 Tensor Core 上的 tile-padding 开销**（Blackwell 上 80 种 dense tile shape：大 tile 省复用但 padding 浪费，小 tile 免 padding 但重复载权重）。
- **DA-MoE**："a **GPU-resident kernel-dispatch runtime** … matches the live routing histogram to offline-tuned distributions and selects a near-optimal fused-MoE kernel **without CPU–GPU synchronization**"（用 **conditional CUDA Graphs** 做 GPU 侧分支）。
- 结果：fused-MoE 延迟几何平均 **1.16×（DeepSeek-V3）/ 1.29×（Kimi K2）**，峰值 1.40×/1.56×；端到端 MoE 层延迟 1.15×/1.26×（峰值 1.37×/1.51×）。

**对 `#12` 的影响**

1. **计算侧的"路由偏斜 + tile padding"响应已被占**，且带实测量级（1.16–1.29× geomean）。
2. ⇒ `#12` 若继续，"padding-aware" 必须严格限定在**通信侧**（all-to-all 的装箱/不均衡），否则就是 DA-MoE 的子集。
3. **纠正一条未核实的转述**：`docs/kimi_plan.md` 称 DA-MoE "**已量化 tile padding 开销且合入 FlashInfer**"——前半句成立；**后半句在正文（摘要/引言）中找不到依据**：它面向 **TRTLLM** 内核选择，未提及 FlashInfer。标为 **未核实**。

## 四、结论

- **`#6`**：ASPLOS/SOSP/MLSys '26 三处扫描后，**机制格仍未被占**；新增邻居 **TLT**（训练期 adaptive drafter + 按 batch 选策略），仍属**策略/长度轴**，不改主假设，但继续加厚"必须证明 depth ⟂ length"的必要性。
- **`#12`**：占位证据从"三重"升到**五重以上** —— LPLB（已 ship）+ MLSys '26 CRAFT / Layered Prefill / MoE Serving Tax + **SOSP '26 MorphKernel** + **DA-MoE 正文**（计算侧 padding 已占且带量级）+ **SOSP '26 Barrier-Free EP（Tier B，待读）**。⇒ 维持"先过 ≤3 天残余不均衡判定、否则归档"的门槛，且判定前**必须先读 Barrier-Free EP 正文**。
- 仍未扫描：**ATC / ISCA / SC '26**。
