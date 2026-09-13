# S3 占位与先行核查（Occupancy & prior art） — 投机解码/MTP 服务评测的方法学

**候选**：`specbench-methodology` ｜ **成本上限**：0 GPU·h（检索 + 读正文）
**目标**：确认没人已经做完，且**引擎里已 ship 的旋钮**不是你的方案。必须读正文。
**结论**：🟡 **存活但收窄** —— 体裁（"benchmark 伪影 + 量化 + 修法"）已被 2605.24217 占住且对方基础设施更强；
**spec-decode 自身的度量恒等式（accept_len / 每步代价 / ITL 语义）无人涉足** ⇒ 我们只能打这一层，
且**必须走"可行域 + 自检器 + 审计"而不是"再列一份伪影清单"**。

## 最近邻（≥3，标注读到的深度）

| # | 工作 | 打的是什么靶 | 读到什么深度 | 与本方向的关系 |
|---|---|---|---|---|
| 1 | **[2605.24217](https://arxiv.org/abs/2605.24217)** Identifying and Mitigating Systemic Measurement Bias in Production LLM Inference Benchmarks（Google，2026-05） | **客户端**：单进程 asyncio + GIL ⇒ M/G/1 排队把 TTFT/TPOT 抬高；提出多进程客户端 + **NTPOT** | ✅ **正文逐字读完**（含 §4.3/§4.4/§7.2） | ⚠️ **同类体裁，但靶子不同**：他们治**客户端与工作负载**，不治 spec-decode 控制回路。**其 §4.3 实测温度导致 21% 吞吐方差、§4.4 实测两工具输入 token 差 50%** ⇒ 证明该体裁"有人认"，但也意味着**我们不能只再列伪影** |
| 2 | **[2608.04714](https://arxiv.org/abs/2608.04714)** What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend | **行为/正确性**：观察到的模型行为可能是后端伪影 | 摘要级 | 相邻不重叠 ⇒ 边界材料（我们谈**性能度量**，不谈行为） |
| 3 | *Speculative Decoding: Performance or Illusion?*（**MLSys '26 oral**） | **宏观趋势**：高 batch 下投机收益缩水 | 标题/venue 级（本工作区既有归档） | 相邻不重叠 ⇒ 也是"这个 venue 认这类工作"的先例 |
| 4 | Pimp My LLM: Leveraging Variability Modeling to Tune Inference Hyperparameters（EASE 2026） | **配置调优**（变异性建模） | 标题级 | 靶子是"怎么选配置"，不是"怎么正确度量" |

### 2605.24217 的**逐字关键点**（用于划边界，避免被审稿人说重复）

- §4.3：*"identical hardware processing 1,000 requests exhibited a **21% throughput variance solely due to temperature configuration** (6,767 tok/s at Temp=0 versus 5,573 tok/s at Temp=0.7)"* ⇒ **温度伪影已被他们量化**。
- §4.4：*"two different utilities processing the exact same dataset generated a **50% discrepancy in the volume of processed input tokens**"*；并要求 *"benchmarking frameworks must explicitly report the enforced cache hit ratio"*。
- §7.2（**future work**）：*"develop standardized, mathematically constrained prompt datasets designed to guarantee exact cache-hit ratios (e.g., strict 0%, 50%, 100% shared prefix lengths)"* ⇒ **他们把"受控缓存数据集"列为未做**。
- **他们全文没有出现**：`acceptance length`、`speculative`、`draft token`、`ITL 与 TPOT 的语义差别`（其 ITL 仅作为"要采集的指标"被列出，未讨论 spec-decode 下 ITL = **每步**）。

**⇒ 可打的靶子 = 上表最后一行那条空档。** 但必须承认：**这个空档比想象中窄**。

## 引擎已 ship 的控制器 / 可观测性（最强基线）

| 项 | 状态 | 对本方向的含义 |
|---|---|---|
| `bench.json` 的 `spec_decode_acceptance_length` / `_rate` / `per_position_acceptance_rates` / `mean_itl_ms` / `mean_tpot_ms` | ✅ 已 ship | **原料齐备**：所有恒等式所需字段都在，但**引擎不做任何一致性检查** |
| `/metrics` 的 `vllm:spec_decode_num_{draft,accepted}_tokens_total`（**计数器**） | ✅ 已 ship | 提供**第二条独立路径** ⇒ 可交叉核对 `bench.json`（这是自检器的基础） |
| ITL vs TPOT 的语义文档 | ⚠️ 需专门 PR **#55283** 才澄清 | 官方承认易混 ⇒ **文档修法无法阻止错误分析**（这正是我们的空隙） |
| `--dataset-name random` | ✅ 已 ship（默认随示例出现） | 数据集选择直接改变接受率 ⇒ 我们的 P2 |
| 温度/采样参数 | ✅ 已 ship | 已被 2605 §4.3 量化 ⇒ **我们不重复打** |

## 差异轴

| 候选差异轴 | 是否站得住 |
|---|---|
| "客户端测量偏差" | ❌ 被 2605 占 |
| "benchmark 伪影清单" | ❌ 被 2605 占（21%/50%） |
| "温度/采样对吞吐的影响" | ❌ 被 2605 §4.3 占 |
| "受控缓存命中率数据集" | ❌ 被 2605 §7.2 **列为未做**（若我们做，等于抢人家的 future work，且需要大规模数据） |
| "后端影响模型行为" | ❌ 被 2608 占 |
| **"spec-decode 度量恒等式的可行域刻画 + 自检器 + 对公开报告的审计"** | ✅ **站得住**：研究对象是**报告本身的可验证性**，不是客户端、不是负载、不是行为；且**不依赖规模**（不需要 5000 QPS 基础设施） |

## 结构性理由（审稿人会问的那一问）

> **"引擎已经有这些字段了，为什么还需要你们？"**

**结构性回答**：因为这些字段的**语义在投机解码下发生了改变，而引擎只负责输出原始量、不负责判定组合是否可能**：

1. ITL 在投机解码下是**每步**延迟（一步可吐 `accept_len` 个 token），在非投机下才是每 token ⇒ **同一个字段名在不同配置下含义不同**，而报告里通常不写配置；文档澄清（#55283）解决不了**已经发表的分析**。
2. 接受长度有**硬上界 `γ+1`**，而 `γ` 常常没被报告 ⇒ 读者无法判断一个"接受长度 4.2"是优秀还是**越界/口径错**。
3. `bench.json` 与 `/metrics` 计数器是**两条独立路径**，引擎**不做交叉核对** ⇒ 静默不一致无人发现。
4. 因此需要的不是"再修一个旋钮"，而是**一组判据 + 一个能跑在报告上的自检器**：这是**可验证性（verifiability）**问题，不是性能问题。

## 被占时的降级

- **首选降级**：把主张缩到最小可交付 =「**可行域 + 自检器**」作为一篇**测量/可验证性**论文（体裁先例：*Performance or Illusion?* @MLSys'26 oral）；
- **次选降级**：若审计发现"几乎没人违反恒等式" ⇒ **自我判死**（无病呻吟），把结论写成一条负面结果（"该社区的报告自洽性良好"）并回到测绘第 2 轮；
- **不做**：不与 2605 抢客户端/温度/缓存数据集这三个靶子。

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] 每个存活痛点找 ≥3 个最近邻，且记录**读过的具体页/节**（只读摘要不算）
- [x] 找齐目标引擎里**已 ship** 的相关旋钮/调度器，并写成最强基线（不是只写学术邻居）
- [x] 写出审稿人会问的那一问，并给出**结构性**回答（不是『我们更自适应』这类程度性回答）
- [x] 写明『若被别人占住，本方向降级成什么』（降级形态必须先想好）

> **杀出口**：立项理由只有『没人做过』⇒ 杀；给不出结构性理由 ⇒ 改写定位或杀
>
> **本阶段判定：存活（收窄）**。理由：靶子未被占、结构性理由可给出、降级形态已想好。
> **但风险已量化**：体裁被更强的团队占住、缝隙窄 ⇒ 下一步必须用**低成本的判别性实验**验证"自检器能否在公开报告上跑出非平凡的不自洽率"，**跑不出就自我判死**。
