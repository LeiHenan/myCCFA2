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

---

## 五、SOSP 2026 官方日程核实（2026-09-11 追加）

**社区清单的标题有误。** 官方 accepted/schedule 页给出真名：

> **StreamEP: Straggler-Tolerant MoE Decoding without Communication Barriers**
> Yizhuo Liang, Shaoyu Wang (USC), Jaeyong Song (SNU), Yanqi Zhou (Google DeepMind), Geon-Woo Kim (UT Austin), Guangrong He, Seo Jin Park (USC)
> 场次：Session 2B "AI Workflows and Performance Tuning"，10 月 1 日 10:40–12:00

**为什么这条最重要**：标题本身即 `#12` 的问题陈述 —— **straggler-tolerant（= per-rank 不均衡）+ without communication barriers（= 去掉 all-to-all 同步屏障）**，且作者含 Google DeepMind 与 USC（Seo Jin Park 组）。

**⚠️ 正文不可得**：SOSP 2026 会期为 **9 月 29 – 10 月 2 日**，论文未上 arXiv（多轮检索无果），ACM DL 未开放。
⇒ **状态：标题/作者/场次 = Tier A（官方程序页）；内容 = 不可得。**
⇒ **重检触发**：会后（约 10 月 2 日）或 ACM DL / arXiv 出现时，**这是判定 `#12` 之前第一优先要读的正文**。

**同会场另两条相关**：LLM-42《Enabling Determinism in LLM Inference with **Verified Speculation**》（MSR & UW，Session 3B）—— 即本仓库 RefShape 探针引用过的 LLM-42；MorphKernel（跨 SM 融合，Session 3D）。

## 六、ATC / ISCA / SC '26 与其它会议的入口状态

| 会议 | 入口状态 | 结论 |
|---|---|---|
| **USENIX ATC '26** | `usenix.org/conference/atc26/technical-sessions` **取不到**（fetch failed，与既有记录一致） | **入口受阻，非选择性跳过** |
| **ISCA '26** | 有索引页（Dagstuhl `einstein.dagstuhl.de/db/conf/isca/isca2026.html`），未展开 | 硬件方向，对本方案（`#12`/`#14`）优先级最低 |
| **SC '26** | 未找到程序/录用页（仅见零散新闻） | **入口受阻** |
| **NSDI '26**（此前**不在**本方案的会议清单里） | `usenix.org/conference/nsdi26/` 可访问 | ⚠️ **新发现**：**SwiftEP《Accelerating MoE Inference with Buffer Fusion and TMA Offloading》**（NSDI '26）—— 又一篇 MoE EP 工作，压在 `#12` 的通信侧 |

⇒ **方法论收获**：本方案原先的会议清单（MLSys/OSDI/SOSP/ASPLOS/ISCA/ATC/SC）**漏了 NSDI**；`#12` 相关的 MoE-EP 工作在 NSDI 上也有。
