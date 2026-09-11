# GPT — 对 GPT 提案的审核与候选筛选

**日期**：2026-09-11
**审核对象**：一份 GPT 生成的选题提案（9 个方向 + 5 个推荐 + 1 个合并愿景「Speculation-Aware LLM Inference Runtime」）
**审核方法**：① 逐条核实其引用；② 与本轮已有击杀台账（47 条击杀理由，见 `old.md`）对齐；③ 只保留**真正未被覆盖的耦合**，其余标死。

> 配套文件：`V41.md`（本轮幸存候选）、`old.md`（引用纪律、判据框架、击杀台账）

---

## 0. 一句话结论

> **9 个方向里，7 个在本轮已被击杀或与其重复；引用基本真实但有两处关键定性错误；真正剩余的只有 2 个"耦合"，且它们是否成立取决于两个正在跑的针对性筛查。**

它自己列的"不太推荐"四项（新 attention 变体 / 新 eviction score / 更好的 draft model / 动态 K）——**我逐条同意**，而且每一条我都有实测的击杀依据（§3）。

---

## 1. 引用核查结果

**全部 6 个给出的 arXiv ID 都真实存在，标题一致——本轮无 phantom。** 这是这份文档比本轮早期若干材料强的地方。

| 引用 | 核查结果 |
|---|---|
| SpecDec++ `2405.19715` | ✅ 真实，**COLM 2025** |
| EAGLE `2401.15077` | ✅ 真实 |
| **RocketKV** `2502.14051`（它给的是 PMLR v267 behnam25a） | ✅ 真实，**ICML 2025**。"up to **3.7×** end-to-end speedup … on an NVIDIA A100" **逐字属实**——⚠️ 但原文限定 *"compared to the **full KV cache baseline**"*，**不是相对调好的稀疏基线** |
| SAGE-KV `2503.08879` | ✅ 真实《LLMs Know What to Drop》 |
| P/D-Serve `2408.08147` | ✅ 真实 |
| KV 综述 `2603.20397` | ✅ 真实（24 页 14 图，cs.LG） |
| **SparseSpec**（它未给 ID） | ✅ = `2512.01278`《Accelerating Large-Scale Reasoning Model Inference with **Sparse Self-Speculative Decoding**》——注意是**自投机**（同一模型兼任 draft/target），不是独立 drafter |
| **LServe** | ✅ 真实，MLSys 2025（`mlsys.org/media/mlsys-2025/Slides/3270.pdf`） |
| **Q-First**（它未给 ID） | ✅ = `2608.15473`《Most of Attention Needs Only the Query in **Disaggregated** LLM Decoding》——**是 PD 分离论文**，不是通用稀疏注意力工作 |

### ⚠️ 两处关键定性错误

**错误一：它把"把投机搬进 engine"当作方向 ① 的增量——但那已经有人做了。**
SpecTool `2512.15834` 摘要逐字：*"speculating tool calls and **forcing sequences to remain resident in the inference engine** to minimize overheads"*，并建议 *"a new **'tool cache' API endpoint**"*。
⇒ **engine 内驻留 + 工具调用投机已经是已发表工作**，不是缺口。（附带：它转述的 "6–21% agent end-to-end 节省" **不在该论文摘要里**；摘要是 *"throughput improvements of several hundred tokens per second"*。）

**错误二：它把 RocketKV 的 3.7× 当作稀疏注意力的当前水平——那是相对 full-KV 基线的数。**
这是本轮反复出现的同一个模式（见 `old.md` §2）：**一个相对弱基线的数字被当成整个空间的性质。**

---

## 2. 逐方向评估（与击杀台账对齐）

| # | 它的方向 | 判定 | 杀死它的具体依据 |
|---|---|---|---|
| **①** | Tool/Agent-aware Speculative Decoding | ⚠️ **部分重复** | step/action 级投机是最饱和的方向之一：最早 `2410.00079`（2024-09），Speculative Actions `2510.04371`（2025-10）已逐字陈述该范式，另有 ConfSpec、SpecGuard、AOSpec、Speculative Macro Commit（MLSP 2026）、SPORK、Skim、IdleSpec、SpecEyes 等 **20+ 系统**。**且 SpecTool 已把它做进 engine**。→ 唯一可能的剩余见 §4-A |
| **②** | Adaptive SpecDec / "K 改成计算预算" | ❌ **死** | 它自己承认"动态 K 不是新 idea"——而"改成预算"**也已发表**：LibraSpec `2608.08721` 逐字 *"reformulate dynamic speculative-length selection as expected-speedup optimization and derive a **marginal criterion**"*；SparseSpec-L `2607.27735` 逐字 *"extending the speculation horizon can reduce rather than improve speedup when the marginal acceptance probability falls below the relative drafting cost"*。另有 CAST / CaDDTree / GLANCE / AngelSpec。vLLM `#54749` 逐字：*"Six different signals are being proposed for this one decision right now. Batch size is what ships."* |
| **③** | Speculative KV Cache（P(commit) → 表示/驻留） | ⚠️ **一半死、一半待定** | **容量/预留那半死**：SpecMemo `2506.01986`、Nightjar（`10.1016/j.sysarc.2026.103889`）、MemSpec `2608.10362`、TransKV。→ **精度那半待定**，见 §4-B |
| **④** | KV Cache **Placement** 而非 Eviction（多 tier） | ❌ **死** | vLLM 已 ship `TieringOffloadingSpec` + out-of-tree `cache_policy_module_path`（*"no vLLM fork or patch required"*）+ fs/obj/p2p tiers；**Dynamo KVBM 已 ship G1 HBM/G2 DRAM/G3 NVMe/G4 S3 + TinyLFU/Count-Min-Sketch，且 `frequency≥2` 磁盘准入过滤器默认开启**；Mooncake / LMCache / SGLang HiCache（7 种驱逐策略 + 11 种存储后端）。**且 vLLM RFC `#54779` 内部已含一份完整的 cost-aware TinyLFU 准入设计，原型 `#54327` 开放中。** 它的"预测未来 reuse"是其中已被覆盖的部分 |
| **⑤** | Compute-before-Attention / 预测性 attention 预取 | ⚠️ **未决但天花板低** | 对应本轮的 C 级探针"prefetch/带宽仲裁"——**UNDECIDED**，缺的是"可归因于传输的 GPU 空闲"这一测量。tracker 拥挤：SGLang `#21846`（`PREFETCH`/`DEMOTE`/`PIN`）、vLLM `#42086`(CLOSED)/`#48445`/`#52113`/`#51428`、SGLang 已 ship `--hicache-storage-prefetch-policy`、AAAI 2026 已有异步 KV 预取。**算术上限 2–8%** |
| **⑥** | KV Prefetching | ⚠️ **同 ⑤** | 同一件事。它的论证"不牺牲准确率"是真实优点，但改变不了 **2–8% 的天花板 + 拥挤的 tracker** |
| **⑦** | Attention + 投机联合（confidence-aware attention budget） | ⚠️ **待定** | SparseSpec `2512.01278` 已经把**稀疏注意力 + 自投机**结合，但那不是"由接受概率门控的 attention 预算"。→ 见 §4-C |
| **⑧** | Speculation Scheduler（`Priority = P(commit)×SavedLatency − ComputeCost`） | ❌ **死** | **YieldSched 已被击杀**（"调度器以 token yield 为控制变量"——四条轴全部已实现：DeepSeek DSpark 已并入 vLLM、阿里 ECHO 已进 SGLang、ICLR 2026 一篇逐字预占）。而且这个公式**正是 vLLM `#54749` 作者造出来、并在唯一一次头对头测试中输给一张实测表的那个模型**：*"`K* = argmax_K E_accept(K) / (F(ctx) + K·M(bs))` … it selects K=0 at the high-batch tier at every context … would not have found the cell worth 29–36%"* |
| **⑨** | 合并愿景「Speculation-Aware LLM Inference Runtime」 | ❌ **不是研究问题** | 它是 ②③④⑤⑧ 的并集，而**每一块都已被占**。本轮的中心发现正是：**"把 N 个已被占的机制组合起来"是密度杀死你的地方**（K7 就是实例——两个已发表半边的组合，残余 1.1–1.3×）。它自己也承认需要"收敛成一个具体、可证伪的 research question" |

---

## 3. 它的"不太推荐"四项——我逐条同意，且都有实测依据

| 它的判断 | 我的依据 |
|---|---|
| ❌ 设计新 Attention | 稀疏注意力已被 LServe（MLSys'25）、RocketKV（ICML'25）、SAGE-KV、Q-First（`2608.15473`）等占实 |
| ❌ 新 KV eviction score | 同上 + `Random Attention` `2609.03430` 逐字：*"We show that the **selection signal contributes almost nothing**"*——它**动摇了整个驱逐打分层**（但那是负面结果，被"必须是加速论文"的约束排除） |
| ❌ 更好的 draft model | drafter 架构被 EAGLE-2/3、MTP、DFlash（ICML'26）、DFlare、JetSpec、Domino、P-EAGLE、DART、Spiffy 占满；Markov 替换被 xPress `2608.02438` / LiLiCorr `2608.20530` 占；树转换被 PCTree / DARTree 占 |
| ❌ 固定 K → 动态 K | 见 §2 ②，**已发表四次以上** |

---

## 4. 从中提取出的真正候选（只有两个半）

> 这三个是本轮台账里**没有被覆盖**的耦合。它们的存活性取决于两个正在跑的针对性筛查。

### A. Agent 级投机的 **commit/rollback 语义 + 概率化准入**（engine 内）

**增量**：不是"提前算工具调用"（SpecTool 已做），而是把**预测的工具结果**当作 engine 里的一等未提交状态：对应的 KV/attention 状态标记为 uncommitted，真结果到达时提交或回滚；并按 `P(commit | state, tool) × saved_latency > compute_cost + rollback_cost` 决定算多少分支。

**为何可能还没被覆盖**：20+ 个 step/action 级投机系统里，绝大多数活在 client/orchestrator 层；SpecTool 进了 engine 但做的是**驻留与吞吐**，不是**分支状态的提交语义**。

**风险（必须正视）**：
- 真正的问题不是"能不能投机"，而是 **agent 场景里工具结果通常改变后续**——所以 `P(commit)` 可能极低，投机窗口填不满。**这是量级问题，必须先测。**
- D3 探针已判定该方向"benchmarked baseline chain"完备，任何主张必须与 SpecTool 及 Speculative Macro Commit 正面比较。

**决定性实验**：在真实 agentic 轨迹上测 **工具延迟窗口内可被有效填充的比例**，以及给定 `P(commit)` 分布下的期望收益。
**判据**：若工具延迟窗口中可有效投机的比例 **<15%** ⇒ 杀。

---

### B. **投机不确定度 → KV 精度**（P(commit) 决定量化档位）

**增量**：不是"持有投机 KV 再提交/丢弃"（SpecMemo/Nightjar/MemSpec/TransKV 已占容量轴），而是 **P(commit) 决定 KV 的数值表示**：P=0.99 → FP16、P=0.8 → FP8、P=0.4 → INT4、P<0.2 → 不存。

**为何可能还没被覆盖**：现有混合精度 KV 工作按**层/通道/敏感度**分配精度，**不按"这个 token 会不会被接受"分配**。

**风险**：
- **量级**：投机 token 通常只占 KV 的一小部分，而且它们**很快就被提交或丢弃**——所以"给它们省精度"的空间可能很小。**必须先算这个比例。**
- **正确性**：低精度 KV 对**最终被提交**的 token 会造成精度损失，而投机解码的核心卖点是**无损**。这条要么被证明可忽略，要么整个方向崩掉。

**决定性实验**：测投机 token 占 KV 字节数的比例；再测按 P(commit) 分档量化后，**被提交 token 的输出分布偏移**。
**判据**：投机 KV 占总 KV **<10%** 或输出偏移不可忽略 ⇒ 杀。

---

### C. **Confidence-aware Attention Budget**（接受概率门控 attention 计算量）

**增量**：attention 的计算预算（full / sparse / 延后）是**该位置接受概率**的函数——而不是"哪些 KV 重要"。

**为何可能还没被覆盖**：SparseSpec `2512.01278` 结合的是**稀疏注意力 + 自投机**，其稀疏性来自注意力本身，不来自接受概率。

**风险**：与 ⑤⑥ 同族——**天花板可能只有个位数**，且必须证明**按置信度分配 attention 预算不损输出质量**（目前没有任何取到的正文给过这个测量）。

**决定性实验**：先测"接受概率"与"该位置需要多少 attention 精度"之间的相关性是否存在（无相关性 ⇒ 直接杀）。

---

## 5. 与其推荐相左的三点

1. **它最推荐的 ① 有一个未言明的致命风险**：agent 场景里 `P(commit)` 可能系统性偏低（工具结果改变后续），所以"投机窗口能否填满"是**先决问题**，不是可以放到后面的工程细节。
2. **它把 ②（动态 K / 计算预算）当作"很值得做"，但它自己在正文里已经承认"动态 K 不是新 idea"——而"改成预算"同样已发表**（LibraSpec 的 marginal criterion、SparseSpec-L 的 marginal acceptance）。这一条应当直接划掉。
3. **它把 ⑨ 当作"比 eviction score 高一个层级"，但层级高不等于可发表**。本轮 K7 就是"两个已发表半边的组合"，残余只有 1.1–1.3×。**组合愿景必须先收敛成一个可证伪的问题**——它自己也这么说。

---

## 6. 建议的下一步

1. **先做 §4-A 的量级测量**（工具延迟窗口中可有效投机的比例）——这是三个候选里唯一"先决问题清楚、且判据便宜"的一个。
2. **§4-B / §4-C 先做比例测量**（投机 KV 占比 / 置信度与 attention 精度的相关性），任一不成立即杀。
3. **不建议投入 ⑨ 这个合并愿景**，除非它能给出一个单一的可证伪命题——目前它是三个已占机制的并集。

**两个针对性筛查仍在跑**（`subfield_scan/r3_agent_spec.md`、`subfield_scan/r3_kv_precision_spec.md`），结果回来会更新本文件的 §4。
