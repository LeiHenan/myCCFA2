# 案例研究 #3：用《从实践现场提炼研究课题》挖 myCCFA2 的实验现场

**日期**：2026-09-15 ｜ **现场**：`myCCFA2`（40 个实验目录 / 27 条决策记录 / 16 份入库 `summary.md` / 3 个候选池）
**方法论**：`docs/PRACTICE_TO_TOPIC_METHODOLOGY.md` **v2**
**结论**：**0 个新课题。** 但死因分布与前两轮不同——**三个候选全部死在"框架层面的占位"**，
而该仓库此前的判死主要是"在飞 PR 占位"与"效应量不足"。另查出**一处真实的入库缺口**。

---

## 0. 先承认边界：这个现场已经被挖过很多次

该仓库**自己已经给出结论**（`notes/COUPLING_MATRIX_2026-09-15.md:67-75`）：

> - 8 个给定方向：全占
> - 3 个候选池（①失败理由过期 ②从未讨论 ③因果裂缝）：全耗尽
> - 我自建机制：**14 条，0 幸存**（其中 5 条是我自己算错/测错后被自己否证）
> - 组件两两耦合矩阵：有意义格全占
> **⇒ "从第一性原理推导机制 → 查占位"这条路在当前时点的产出率 = 0/14。**
> 要改变产出率，必须换**分母**。

⇒ 本轮**不再走"推导机制"那条路**（它已被走到 0/14）。本轮只挖一个该仓库**系统性地没挖过的类别**：
**度量类原料**（§2 的第 3、4 类）。这与案例研究 #2 的结论一致——度量类是最容易被违反、最难自查的一类。

---

## 1. 记录（§2 五类，每条一句带数字）

### 1.1 失效

| # | 记录 | 出处 |
|---|---|---|
| F1 | `python -m vllm.benchmarks.serve` 在 0.29.0 **无 `__main__` 块** ⇒ import 后**静默退出、退出码 0** ⇒ 18 份 `.bench.log` **全 0 字节、0 份结果 JSON**，脚本里的 `\|\| echo "bench 失败"` **永不触发** | 决策 #123 |
| F2 | 网格 #2 结束后遗留**孤儿 `VLLM::EngineCore`（PPID=1、31.8 GB，来自另一个 venv 0.28.0）**，持有显存直到下一臂开始 | 决策 #123 |
| F3 | `--speculative-ngram-external-sam-budget` **默认 0** ⇒ 外置 SAM 路径被静默禁用；主实验 **B 臂干预实际未生效**，其 `recovery=0.0%` **不作否证证据** | `notes/prereg/sglang-draft-corpus.md:111` |
| F4 | `batch_invariant` + 前缀缓存：`config/cache.py` 里该名**零命中** ⇒ **没有任何地方拒绝或告警**，该组合在默认配置下被静默接受 | `notes/GOAL3_CLOSE_2026-09-15.md:111` |
| F5 | `speculative_config={"method":"dflash2"}` → pydantic `ValidationError`（合法值只有 `dflash`）⇒ v1 正式网格该臂**从未产生数据** | 本会话审计 |

### 1.2 反常

| # | 记录 | 出处 |
|---|---|---|
| A1 | `Initializing a V1 LLM engine` 是**硬编码字符串**（`v1/engine/core.py:124`）；**48 份** serve 日志**全部**含它，其中 **47 份同时含** `Using V2 Model Runner` ⇒ 全部历史实验跑的都是 V2，**从未发生降级** | 决策 #122 |
| A2 | `VLLM_USE_V2_MODEL_RUNNER` 的取值语义：**"未设"=auto**，但**空串** ⇒ `bool(int(""))` 抛 `ValueError` | 决策 #122 |
| A3 | `nm -D` 在 FA2 二进制里数出 **561 个** `split` 符号、含 `run_flash_splitkv_fwd`，但 Python 封装把它闸掉：`if num_splits > 1: raise NotImplementedError` ⇒ **符号存在 ≠ 可达** | `notes/POOL1_ENGINE_UNLOCKED:59-66` |
| A4 | **接受率对该类缺陷失明**：MTP 臂 PC off `0.8533` vs on `0.8512`（**−0.25%**），而**同臂贪心输出分歧 `0% → 75.9%`**（243/320，`p=9.0e-32`） | C1 仓 v9 |
| A5 | prefix-cache **块粒度随 drafter 变化**：`528 token`（nospec）→ `544 token`（MTP）⇒ 可缓存前缀集合本身是 drafter 的函数 | C1 仓 v9 |
| A6 | 贪心解码**跨引擎配置不可逐位复现**：nospec 对照臂自身分歧 `10.0%`（dense）/ `18.3%`（hybrid），onset 中位在文本 `0.62–0.95` 处 | 本会话 C1 重做 |

### 1.3 度量不可靠

| # | 记录 | 出处 |
|---|---|---|
| M1 | `sum(histogram)` 被当总步数 ⇒ **同一份数据**推出 **5.6×** 与 **17.1×** 两个"hybrid 边际成本"，最终判定 **~0** | 本会话审计 |
| M2 | **#54691 的四个 crux 格是"推断，不是测量"**；能约束可分性的 15 格网格**携带的是阈值而不是标签**，每个符号都是**未测标量 `a`** 的推论——`a` 在全部 15 格**都未测** | `notes/SB_SEPARABILITY_2026-09-15.md` |
| M3 | #54691 作者**自陈测量方法之误**：*"My measurement method is at fault. I get the cost by forcing all drafts to be rejected … If rejection does meaningful work, the numbers above are **worst-case rather than representative**."* | 同上 |
| M4 | 同一作者后来的固定 batch 诊断（S9）**未能复现** context 依赖：**95× context 范围内平坦到 −0.10 pp**，而 **batch 反而携带 +15.02 pp** | 同上 |
| M5 | v1 C-1 主判据**结构性不可辨识**：共同效应 **4.87–5.53×**，待测类别差异 **2%**（0.1810 vs 0.1847） | 本会话审计 |
| M6 | 我的 C1 重做：`run_id` **跨网格碰撞** ⇒ `analysis_core-dense.json` 的 ngram K=3 一行**无法从归档 raw 复算**（63.53 vs 62.64，1.4%） | `results/C1-redo/…/CORRECTION.md` |
| M7 | 62 个模型文件里 **3 个"尺寸全对、SHA256 不符"** ⇒ 模型对 "The capital of France is" 输出 `IIIIIIII`；**尺寸校验 100% 通过** | 本会话 C1 重做 |
| M8 | **同实例 nospec 基线是 `gen` 的函数**：`8.4%`（gen=64）→ `26.9%`（gen=256） | C1 仓 v9 |
| M9 | `num_draft_tokens` 的定义含 *"after subtracting drafts invalidated by structured-output constraints"* ⇒ **语法拒绝被静默折进分母** | `notes/DIRECTION_MAP_3AXIS.md:32` |
| M10 | 一条硬件引文被**静默编辑**过：CacheGen 的 *"an NVIDIA A40 GPU server **with four GPUs**"* 在引号内被删掉"四张 GPU" | `notes/DIRECTION_MAP_3AXIS.md:91` |

### 1.4 代理量失明（v2 新增类别）

| # | 记录 |
|---|---|
| P1 | **接受率**作为**正确性**的代理：一个让 76% prompt 输出分叉的缺陷只让它动 **−0.25%** ⇒ 二者**解耦**，不是口径问题 |
| P2 | **符号存在**作为**能力可达**的代理：561 个 split 符号 vs Python 层 `num_splits > 1` 直接 `NotImplementedError`（A3） |
| P3 | **退出码 0** 作为**"命令做了事"**的代理：`-m vllm.benchmarks.serve` 退出码 0 而 18 份产物全空（F1） |

### 1.5 不可测

| # | 记录 |
|---|---|
| U1 | v1 C-1 的 4 项必报量中，**TTFT p50** 与 **`prefix_cache_hits_total`** 从未取到（离线 API 口径下不可得） |
| U2 | #54691 的**标定对照**（`synthetic_acceptance_rates` 全 1.0）在最后一条评论里**仍在运行**，**从未报告结果** |

---

## 2. 过滤（§3 三道闸）

| 候选 | 闸1 改变谁的结论 | 闸2 变量在手 | 闸3 效应量够 | 判决 |
|---|---|---|---|---|
| **α 存在性证据 ≠ 可达性证据**：把"某能力是否可用"的判定从存在性证据（符号/日志串/退出码/字段/默认值）改为可达性证据（回读生效值），并给出 6 类证据的清单 | **改领域评价口径** | 是（8 个实例，均有 file:line 与复现） | 够：至少 **3 项独立调查的结论被它推翻或作废**；一次 **18 臂 × 2 轮网格 0 数据** | **KILL（框架被占 + 保质期短）** |
| **β pay/no-pay 曲面依赖未测标量**：已发表的投机"是否划算"边界，其 crux 格是推断而非测量 | 改同行默认假设 | 是（15 格符号矩阵 + 4 格成本分解已转录） | 够：**0.05 宽的窗口**（1.42 vs 1.47）撑起整个不可分性论证 | **降级**：批评体量，不是课题体量 |
| **γ 测量有效性的分类学**：把本仓库累积的失效收敛成一张可执行的审计清单 | 改领域评价口径 | 是（本文 §1 即原料） | 够 | **KILL（框架被占）** |

---

## 3. 占位核查（§5，限时）

### 3.1 α：**框架已被占，且修复正在上游发货**

- [vLLM #51401](https://github.com/vllm-project/vllm/issues/51401)：*"VLLM **silently ignores** falsey yaml configuration options"* —— 直接占住"静默忽略配置"这一格；
- [llama.cpp #24762](https://github.com/ggml-org/llama.cpp/issues/24762)：router **停止向子实例转发** `--parallel`/`--cache-type-*`/`--flash-attn`/`-ngl`（回归）—— 第二引擎的同类实例；
- [ConfLogger: Config Diagnosability Tool](https://www.emergentmind.com/topics/conflogger) —— 配置可诊断性已是工具方向；
- [FAILUREATLAS](https://export.arxiv.org/pdf/2607.17525)：*"A Taxonomy of Failure Modes in Multi-Provider LLM Serving Infrastructure"* —— **分类学这一格被占**；
- [vLLM PR #48030](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/48030/overview)：*"Log **fully resolved** pooling config at startup"* —— **"回读生效配置"这个修复正在上游发货**；
- 且该仓库**自己的规则**已写明：`缺`=引擎缺口，**保质期短，不立项**（`COUPLING_MATRIX:7`）。

⇒ **框架被占 + 修复在途 + 自身规则排除**，三条独立理由。**KILL。**

### 3.2 β：**框架未被直接占，但已被"自撤回实践"逼近**

- 未找到对 #54691/#54749 那个曲面的**再分析**；也未找到"crux 格是推断"这一具体主张的占位者；
- 但 [thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) 已把"**ERRATA.md 列出本研究自己撤回的主张** + 用 checker 从已提交数据重新推导、漂移即失败"做成了**仓库级实践**；
- ⇒ 按 §5.5，框架被占**不能**否掉具体主张；但该主张的体量是**一条批评**，不是一篇课题。

### 3.3 γ：**框架已被占**

- [Are We Learning Yet? A Meta-Review of Evaluation Failures Across Machine Learning](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/757b505cfd34c64c85ca5b5690ee5293-Paper-round2.pdf)（NeurIPS 2021）—— ML 评价失效的元评审；
- [verified-experiments](https://github.com/bhavsarp25/verified-experiments)、[llm-evalgate](https://github.com/LesterALeong/llm-evalgate) —— provenance / sanity / audit guards 与带误差棒的评测闸门。

⇒ **框架被占。**

---

## 4. 本轮真正的产出：不是课题，是三条结构判断 + 一个缺口

### 4.1 死因分布变了（这是与前两轮最不同的地方）

| 轮次 | 候选数 | 幸存 | 主要死因 |
|---|---|---|---|
| 案例研究 #1（RTX PRO 6000） | 4 | 0 | 2 被实测自证伪 / 2 被已发表工作占位 |
| 案例研究 #2（A800 / C1 重做） | 4 | **1** | 占位 1 / 效应量 1 / 自身缺陷 1 |
| **案例研究 #3（myCCFA2 全现场）** | 3 | **0** | **3 个全部死在框架层面的占位** |

⇒ **本轮的 0 不是"没挖到"，而是"挖到的每一条都已被更一般的工作覆盖"。**
这与该仓库的 0/14 结论**同向**，但给出了**不同的机制**：前两轮是"在飞 PR 抢先"，
本轮是"**元层面的框架已被 2021–2026 的综述/工具/上游 PR 拿走**"。

### 4.2 缺口：案例研究 #1 的头号资产**从未入库**

`docs/PRACTICE_TO_TOPIC_CASESTUDY_2026-09-15.md:107` 把
**`notes/BOTTLENECK_INVENTORY_2026-09-15.md`**（"13 格瓶颈清单"、"8 格已榨干、**4 格标未知**"）
列为"本案例留下的可复核资产"。核查结果：

```
git log --all --diff-filter=A -- '*BOTTLENECK*'   →  只命中 c2e2f268（即加入这条引用的那个提交）
grep -rn "BOTTLENECK_INVENTORY"  →  只命中案例研究自身那一行
```

⇒ **该文件从未存在于本仓库的任何一次提交中。** 而决策 #125 又把
"第三张图的 E 段（从未被讨论过）"确立为**可证伪的收口条件** —— 那个收口条件建立在一份不存在的清单上。

这是 §7.1（数字/资产溯源纪律）在**资产层面**的违规，与我在案例研究 #2 里犯的
"孤儿数字"（`0.8533/0.8512` 无源）**同型**，只是对象从"数字"换成了"文件"。

### 4.3 该现场的数据问题容量（§8）已到边界

- 要回答"α 是否成立"，需要的数据里**不存在的变量**是**上游是否已修**——本轮查证：**已在修**（PR #48030）⇒ §8.2 触发；
- 剩余分析只会是同一结论的更细版本 ⇒ §8.3 触发。

---

## 5. 对方法论的修订建议（第 4 条）

### 5.1 §8 停止规则应补一条**资产核查**（与输出绑定）

本轮的缺口（4.2）与案例研究 #2 的"孤儿数字"是同一类错误的两个变体：
**被归档的"资产"本身可能不存在。** 现有的 §7.1 只覆盖"数字"，§7.4 只覆盖"否定性结论"。

⇒ **加一条**：*封存时，凡在交付物里被**具名引用**的产物（文件、清单、表格、脚本），
必须附一条**可复现的存在性取证**（`git ls-files` / `ls` 的输出），
并对"仅被引用、未入库"的项**显式标注为缺失**。*

**理由**：交付物里的一句"本案例留下 X、Y、Z"会被人当作"X、Y、Z 可复核"。
只要没人去 `git ls-files`，缺失就可以无限期潜伏——这与 §7.4 的"否定性结论"同构：
**它不会被数据推翻，只会被取证推翻。**

### 5.2 §5 的"框架被占"需要与"保质期"联动

本轮 α 是**教科书式的框架被占**：上游 issue 已存在、修复 PR 已发货。
该仓库对这类情况已有规则（`缺`=保质期短，不立项），但**那条规则只在 `COUPLING_MATRIX` 里，没进方法论**。

⇒ **加一条**：*§5 的占位结论要再看一眼**上游是否正在修**。
"已被报为 bug"与"正在被修"是两种不同的占位，后者意味着**即使你现在做出来，发表时也已过期**。*

---

## 6. 一句话总结

> **挖不出新课题——但这次的"挖不出"比前两次更硬。**
> 三个候选（存在性≠可达性审计、pay/no-pay 曲面再分析、测量有效性分类学）
> **全部死在框架层面的占位**：上游 issue + 在途修复 PR + NeurIPS 2021 元评审 + 两个工具仓库。
> 与前两轮"在飞 PR 抢先""效应量不足"不同，本轮说明：
> **在这个现场，连"测量与评价"这一层也已经被占掉了。**
> 唯一确定的产出是一个**入库缺口**（头号资产从未入仓），以及**方法论的第 4 条修订**（封存前核验资产存在性）。
