# 空地测绘（Gap Survey）— 8 个子区域的「痛点 + 占位」双查

**日期**：2026-09-13 ｜ **方法**：PIPELINE.md §8 的新规则 —— **先查占位、再看痛点**（此前把顺序做反，白做了一份零卡预算，见 decision #73）
**成本**：0 GPU·h（检索）
**输出**：一张「已被占 / 仍空 / 待验证」的地图 + 一个选定方向

## 一、地图（8 个区域）

| # | 区域 | 最近的占位者（2026） | 判定 |
|---|---|---|---|
| 1 | **推理引擎内部机制**（KV/调度/状态/投机） | 混合状态：vLLM RFC #55697 + PR 栈（4 天前）｜KV 淘汰：红海｜自适应投机：PRISM / HELIOS（MLSys'26）｜chunked prefill 公平性：论文 | 🔴 **饱和**（本工作区已在此撞死一次） |
| 2 | **跨副本 / 集群 / 多租户** | Lyapunov-Guided KV Control（Euro-Par'26）｜SAGA: Workflow-Atomic Scheduling on GPU Clusters｜vLLM/SGLang 均已 ship cache-aware router | 🔴 饱和 |
| 3 | **训练–推理交界（RL rollout）** | TensorHub（弹性权重传输）｜RolloutPipe（分离式 RL 重叠）｜SparseRL-Sync（通信 ↓100×） | 🔴 **极饱和**（近 3 个月集中爆发） |
| 4 | **测量方法学 / benchmark 有效性** | *Systemic Measurement Bias in Production LLM Inference Benchmarks*（2605.24217，**打客户端排队/GIL**）｜*What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend*（2608.04714）｜*Speculative Decoding: Performance or Illusion?*（MLSys'26 **oral**）｜Pimp My LLM（EASE'26 配置调优） | 🟡 **活跃但未饱和，存在可细分空档** |
| 5 | **长上下文 / 记忆** | MAGMA（多图 agentic memory, ACL'26）｜长上下文 serving 论文密集（NLP 侧更挤） | 🔴 饱和 |
| 6 | **Agent 系统层** | *From LLM Inference to Agentic Workloads: Characterization…*（2608.15127）｜SpecBox（sandbox 调度）｜SAGA｜Continuum（KV TTL） | 🔴 **极饱和** |
| 7 | **可靠性 / 静默错误** | Ekka: Automated Diagnosis of Silent Errors in LLM Inference（**ICML'26**） | 🔴 被占 |
| 8 | **能效 / 成本** | DualScale（DVFS + PD 放置）｜*The Cost of Context*（输入 token 能耗）｜*Are LLMs Economically Viable…*（ACL'26 industry） | 🔴 被占 |

**测绘结论**：**"显而易见的那一层"已经被占满**——引擎内部机制、集群调度、RL 交界、Agent 系统四个区域全饱和，且占位者多在最近 3 个月内。
⇒ 在同抽象层继续找题目，撞车概率高（本工作区已实证一次）。

## 二、唯一仍可细分的区域：#4 测量方法学 —— 但要打对靶子

已有 4 个工作在这一层，**靶子各不相同**：

| 占位者 | 打的是什么靶 | **没有覆盖什么** |
|---|---|---|
| 2605.24217（2026-05） | **客户端排队偏差**：单进程 asyncio 客户端在 GIL 下把 TTFT/TPOT 抬高 ⇒ 提出多进程客户端 + NTPOT | 引擎/算法**自身控制回路**的度量（接受长度、投机步、暖机） |
| 2608.04714 | **后端副作用**：观察到的"模型行为"可能是推理后端造成的 | 性能度量的有效性 |
| *Performance or Illusion?*（MLSys'26 oral） | 投机解码**在高 batch 下收益缩水**这一宏观趋势 | 度量口径与工具链本身 |
| Pimp My LLM（EASE'26） | 超参**配置调优**（变异性建模） | 度量正确性 |

**⇒ 空档：投机解码（含 MTP / 多 token 预测）自身控制回路的度量方法学。** 靶子是「acceptance / 每步代价 / 暖机状态 / 分位数」这一组量，而不是客户端或宏观趋势。

## 三、这个空档有**独立的实践侧疼痛证据**（不是我推测的）

| 证据 | 说明了什么 |
|---|---|
| vLLM 论坛帖《vLLM 的 MTP 的标准测试方法是什么？》([2715](https://discuss.vllm.ai/t/vllm-mtp/2715)) | 从业者**在问"标准测法是什么"** ⇒ 没有公认方法 |
| vLLM 论坛帖《服务开启 MTP，如何评估指定维度的真实吞吐》([2658](https://discuss.vllm.ai/t/vllm-mtp/2658)) | 同上，且问的是**"真实吞吐"** ⇒ 他们怀疑现有数字不真实 |
| vLLM **PR #55283**「[Doc] **Clarify ITL vs TPOT** Prometheus metrics」 | 官方需要专门发 PR 澄清两个指标的差别 ⇒ **口径混淆是普遍现象** |
| vllm.cpp issue **#2770**「The OpenAI server **exports no speculative-decoding acceptance metric**, so acceptance is readable only from vllm-bench」 | 服务端**根本不导出接受率** ⇒ 关键量在可观测性上就是缺的 |

## 四、本工作区已有的四条**可复现**的度量伪影（自有证据，逐条可重跑）

| # | 伪影 | 本工作区实证 | 影响 |
|---|---|---|---|
| A1 | **`--dataset-name random` ⇒ 接受率恒为 1.00** | `#6` T0 期间的实测记录（decision #47–#50 段）；随机 token 必然被接受 ⇒ 投机解码"看起来免费" | 接受长度与吞吐被**根本性高估**；用它做的对比全部失真 |
| A2 | **暖机爬升随配置变化 ⇒ "N 次均值"系统性偏置赢家** | A1 网格 120 格逐次数据：bs=1 时 d5 爬 **11–13%**、d1/d3 只爬 6–7%；改稳态口径后差距从 25.6–32.6% 变 **26.3–31.8%**（decision #66） | 报告值被系统性压低，**且偏置方向与配置相关** ⇒ 排序会被扭曲 |
| A3 | **冷/热双峰 ⇒ 重复次数少时均值无意义** | A3 对照：同格三次重复 661/669/**1389**、750/757/**1941**（2–2.6×）（decision #65） | 3 reps 的结论不可靠 |
| A4 | **`mean_itl_ms` 是"每步"而非"每 token" ⇒ 每步代价被误算 accept_len 倍** | 18/18 格实测 `tpot ≈ itl/accept_len`；初版分析器就写错了这一步，**自测当场抓到**（decision #67） | "每步代价"是深度/长度权衡分析的核心量，误算会得出相反结论 |

**这四条的共同点**：它们都在**引擎给出的默认工具链口径内**，不需要特殊硬件，任何人按官方示例都会踩到；而且**每一条都能单独把结论带偏**。

## 五、选定方向

> **`specbench-methodology`** —— 投机解码/MTP 服务评测的方法学：把「接受长度 / 每步代价 / 暖机状态 / 服务端可观测性」这组量的度量口径系统化，量化它们的伪影，并给出可直接采用的正确口径与检测方法。

**为什么选它**：
1. 地图上**唯一未被同靶子占据**的区域（2605 打客户端、2608 打行为副作用、Illusion 打宏观趋势）；
2. 有**独立的实践侧疼痛证据**（论坛两帖 + 官方 doc PR + 服务端缺指标）；
3. 本工作区已有**四条可复现伪影**作为起点（不是从零猜痛点）；
4. **单卡可做**：不需要多机、不需要训练、不需要专有 trace，96 GB 卡足以；
5. **有 MLSys 先例**：*Performance or Illusion?* 是 MLSys'26 **oral** ⇒ 这个 venue 认这类工作。

**风险（写下来，供 S2/S4 打）**：
- ⚠️ 形态⑥「本质是个工程配置/文档问题」：若结论只是"用真实文本 + 多跑几次"，则退化为 best-practice 清单 ⇒ 必须给出**机制性解释**与**可迁移的检测方法**（例如"如何在不看源码的情况下判断一个报告的接受长度是否可信"）。
- ⚠️ 形态⑤「上界」：必须算出「伪影能把结论带偏多少」的**上界**（≥8% 才做）。
- ⚠️ 与 2605.24217 的重叠：必须在 S3 读其正文，逐条划清靶子边界。

## 六、下一步（都在本目标的轮次里）

1. 建 `candidates/specbench-methodology/`，写 S0–S1（痛点带量级与可复现命令）；
2. S2 伪问题筛查（重点打形态⑥）+ S3 占位核查（**读 2605.24217 正文**划边界）；
3. S4 上界：**零卡先算**"伪影幅度 / 真实效应"的比值（A1 的接受率伪影 ≈ 1.00 vs 真实 ~2.8 ⇒ 量级巨大；A2 ≈ 10–13%；A3 ≈ 2–2.6×）；
4. 服务器验证（单卡，tmux）：用 Qwen3-4B + dflash2 复现四条伪影，每条给出"错误口径 vs 正确口径"的对照数字；
5. S5 预登记 → S6 判定 → S7 归因；每步入库并推 GitHub。
