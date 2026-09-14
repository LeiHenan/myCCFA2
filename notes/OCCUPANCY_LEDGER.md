# 已知被占图（Occupancy Ledger）—— 机器可读的负面结果库存

**日期**：2026-09-13 ｜ **来源**：`goal-ebcec24d` 第 5 轮 ｜ **成本**：0 GPU·h
**这份文件解决什么问题**：本工作区已经在 **18 个对象 + 8 条线索**上花了两轮测绘 + 四轮目标探测，
每一次的"为什么被否"都散落在 decision log 与各候选档案里。**结果是：我在第 2 轮重复发现了 4 天前已判死的深度轴。**
⇒ 把负面结果**集中、结构化、带证据**，让**任何未来的轮次在动手前先查这张表**。

**用法（硬性）**：
1. 新方向立项前，先在本表里 grep 关键词（问题类 / 引擎功能名 / 指标名）；
2. 若命中"🔴 已解决"，必须指出**差异轴是条件性的**（判据 ②）且**上游未覆盖该条件**，否则直接放弃；
3. 若命中"⚠ 条件性缺口"，**先算整机账**（判据 ③）再决定；
4. 新增条目一律带 **URL 或 file:line** 与**查证深度**（`READ BODY` / `TITLE ONLY`）。

**列约定**：状态 = 🔴 已解决（有人 ship 或已发表）／⚠ 条件性缺口／✅ 结构性缺口（本表当前 **0 条**）；
**墓碑** = 我们为它花过多少 GPU·h。

---

## 一、引擎内部机制层（全部 🔴）

| # | 问题类 | 占位者（带证据） | 查证深度 | 我们的墓碑 |
|---|---|---|---|---|
| 1 | 投机解码的草稿**长度/预算**自适应 | vLLM `num_speculative_tokens_per_batch_size`（`config/speculative.py:473-529`）+ RFC `#48202`（per-request effective proposal lengths）；SGLang `--speculative-adaptive-config`（[文档整页](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding.md)：per-BS EMA + hysteresis + ceiling） | READ BODY | 0 |
| 2 | 投机解码的**外置语料 / n-gram drafter** | vLLM prompt-lookup decoding（`config/speculative.py:1146-1158`，**opt-in**，`prompt_lookup_min=max=5`）；SGLang roadmap [#21052](https://github.com/sgl-project/sglang/issues/21052) 已勾项（外置语料 #21425、多 SAM 动态 API #22203、per-request trie #22737）+ 基准 #22569（*output-as-corpus proves SAM ≥2x*） | READ BODY | **≈0.90** |
| 3 | 语料的**跨进程/跨副本恢复** | 见 #2 的同一批占位者；我们实测"恢复比生成便宜 60×、逐位等价、跨进程可移植"（1.62% / 0.000%） | 自有实验 | （含在 #2） |
| 4 | **批组成不变性 / 确定性**（**第 6 轮定论：现象 ✅ 已实测（SGLang/sm120：批内 8 个相同请求→8 个不同 token 序列；开开关后回到 1 个）；机制 ❌ 未归因；可回收性 ❌ **第 7 轮源码级确定排除**（11/12 项改动属通信路径、TP=1 不执行；单卡成本在算子替换；且 `VLLM_BATCH_INVARIANT` 是全仓唯一开关、无更细粒度接口）⇒ **本线索关闭**；第 8 轮补：**vLLM 在 sm120 长序列上也实测证实**（默认 3/8 发散 → 开关 1/8，代价 1.7×），且**合批已有墙钟证据**（n=8 仅 n=1 的 2.0×））（⚠ 第 6 轮升级为条件性缺口：开关**存在但覆盖有洞**——vLLM #27433 仍 OPEN、多处 open issue 报告开了开关仍发散、SGLang 在 L40S 崩/chunked prefill 挂死、llama.cpp 无开关；**文献 A 类已 20+ 篇**） | vLLM `VLLM_BATCH_INVARIANT`（`envs.py:617-619`，默认 0，含 6 处副作用）；SGLang `--enable-deterministic-inference`（`server_args.py:3474-3478`，文档串即 "batch invariant ops"），且 `speculative_hook.py:770-776` **在代码里 `raise`** 挡住非不变 kernel | READ BODY | 0 |
| 5 | **前缀复用 / KV 生命周期**（跨请求、跨会话、跨副本） | agentic-kv-cache（68k 真实请求 / 393 会话）；Continuum（KV TTL）；vLLM RFC #55697；`--enable-session-radix-cache`（`server_args.py:1444`） | 部分 READ BODY | **≈0.6** |
| 6 | 混合状态（attention + Mamba/线性层）服务 | vLLM RFC #55697 + PR 栈（4 天前） | 检索 | 0 |
| 7 | 前缀复用与投机解码的**交互** | 我们自己的 p10 对照（投机关 onset=2 / 投机开 onset=3，2.09×/2.63×，8 条假设排除） | 自有实验 | （含在 ≈0.6） |
| 8 | 训练–推理交界（RL rollout） | TensorHub / RolloutPipe / SparseRL-Sync（2026 集中爆发） | 检索 | 0 |

## 二、测量方法学层（全部 🔴）

| # | 问题类 | 占位者 | 查证深度 | 墓碑 |
|---|---|---|---|---|
| 9 | 客户端伪影（排队 / GIL）抬高 TTFT/TPOT | *Systemic Measurement Bias in Production LLM Inference Benchmarks*（2605.24217） | 检索 | 0 |
| 10 | 后端副作用伪装成模型行为 | *What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend*（2608.04714） | 检索 | 0 |
| 11 | 投机解码在高 batch 下收益缩水（宏观趋势） | *Speculative Decoding: Performance or Illusion?*（MLSys'26 **oral**） | 检索 | 0 |
| 12 | 配置调优的变异性 | Pimp My LLM（EASE'26） | 检索 | 0 |
| 13 | **跨引擎指标语义一致性** | NVIDIA Dynamo `metrics-comparison.md`（PR #8055，36/48/14 指标映射）；AMD ROCm *Forced acceptance length*；InferenceBench §3.2（公式参照） | READ BODY（Dynamo） | **≈0.4**（S0–S8 全过） |
| 14 | **投机解码基准的数据集敏感性** | **[SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](https://huggingface.co/papers/2604.09557)**（专为投机解码做的数据集多样性基准）；[Runtime-Controlled Evaluation of Speculative Decoding Methods](https://www.mdpi.com/2079-9292/15/14/3179/review_report) | TITLE+摘要 / 评审页 READ BODY | 0 |
| 15 | 投机解码的**内部度量自洽性** | 我们实测 **0/24 违规**（引擎自洽）+ 官方 doc PR #55283 | 自有实验 | **≈0.3** |
| 16 | 暖机口径偏置 | 我们复算 **2.31 pp、符号不一致** ⇒ 不成立（自己否掉） | 自有实验 | （含在 ≈0.3） |

## 三、应用负载层（全部 🔴）

| # | 对象 | 占位者 | 墓碑 |
|---|---|---|---|
| 17 | 扩散 / 视频生成服务 | BlockServe / GenServe / Sol | 0 |
| 18 | 多阶段 RAG / 流水线运行时 | OpRAG / Scepsy / CrossPool / RAGPerf | 0 |
| 19 | 推荐系统推理 | Embedding Table Partitioning / ServerlessRec | 0 |
| 20 | 语音 / 流式推理 | VoxServe / PACE | 0 |
| 21 | 向量检索服务 | AdaptIndex / Puffin-Backed Vector Indexes | 0 |
| 22 | 多模态 / Omni 流水线 | LiveServe / FlashRT | 0 |
| 23 | 训练侧（非 RL） | InfiniPipe | 0 |
| 24 | 长上下文 / 记忆 | MAGMA + 密集长上下文服务论文 | 0 |
| 25 | Agent 系统层 | Continuum / Scepsy / SkillRouter / SpecNet-Agent | 0 |
| 26 | 可靠性 / 静默错误 | Ekka（ICML'26） | 0 |
| 27 | 能效 / 成本 | DualScale / The Cost of Context / Are LLMs Economically Viable | 0 |

## 四、⚠ 条件性缺口的唯一存量与其量化为负的结论

| # | 缺口 | 条件 C | 量化结果 | 判定 |
|---|---|---|---|---|
| 13′ | 跨引擎指标语义不一致 ⇒ 配置选错 ⇒ 真实损失 | 当同一份数据被两个引擎以不同语义度量时（实测 `accept_len` −32.6%、`itl/tpot` 1.07× vs 2.98×） | 用 A1 归档网格算"按引擎报告选最优 vs 按文档轴选"：**最大差 0.2%** | **<8% 门 ⇒ 杀** |

## 五、结论：**本表当前 ✅ 结构性缺口 = 0**

四条硬约束（**半径 ≤1 人周 / 单卡 / 无用户私有 trace / 只接受优化类**）的交集为空。
**取证口径**：18 对象 + 8 线索 + 四轮独立探测（批组成不变性、配置选错代价、thinking 阶段投机、数据集敏感性），
**全部命中"已 ship / 已发表 / 已被自己否证"**；累计 GPU 花费 ≈3.4 GPU·h（其中 `goal-ebcec24d` 为 **0**）。

**要重开本表，必须先放宽其中一条**（权限都在用户）：
① **半径**（引擎手术，3–4 周）→ 打开 #5/#6；② **多卡** → 打开训练侧/分布式；
③ **测量类**（已知被 #13/#14 占，需更窄切片）；④ **用户私有 trace** → 打开唯一被验证过能产出"文献里没有的数字"的输入源。

**维护规则**：

5. **🔴 的判定标准（v1.1.2 收紧）**：只有『已 ship **且** 在本代硬件/本负载上被验证有效 **且** 无未关闭的同类失败报告』才可标 🔴；若只是『有个开关』而有效性未验、或仍有 open 失败报告、或文档与实现不一致 ⇒ 记 **⚠**。本条来自 2026-09-13 的自我更正：第 3 轮据『两家都 ship 了开关』把 `batch-invariance-cost` 判死，而检索显示 vLLM #27433 仍 OPEN、多个 open issue 报告开了开关仍发散、修复 PR 仍在 merge。

**其余**：新增条目必须带 URL 或 `file:line` + 查证深度；**不得**把"我没搜到"写成"✅ 缺口"（本表不收录未取证的空白）。

---

## 六、2026-09-14 新增（`goal-a729588f` 三方向测绘 + p20/p21 实测后）

**为什么新增这一节**：两张方向地图（投机解码 1,191 行、KV cache 3,541 行）到位后，我在**同一天内**用**实测**验证了两条看起来最有希望的线索，
结果**两条都撞上 OPEN 的占位者**。这一节把两个新问题类登记进来，并**收窄第 4 行（批不变性）的适用前提**。

| # | 问题类 | 占位者（带证据） | 查证深度 | 我们的墓碑 |
|---|---|---|---|---|
| 9 | **动态投机调度（DSD）在 K=0 档的代价** | **PR [#53426](https://github.com/vllm-project/vllm/pull/53426)（OPEN）**标题即机制：*"Opt-in skip of the **K=0 draft sync forward** (MTP + DFlash, default off)"*；issue [#49548](https://github.com/vllm-project/vllm/issues/49548)（OPEN，含报告者自建仪器 `VLLM_DSD_K0_DIAG=1`）；issue [#48494](https://github.com/vllm-project/vllm/issues/48494)（OPEN，*"12–25% throughput penalty … even with an all-K=0 table"*）；PR [#47737](https://github.com/vllm-project/vllm/pull/47737)（OPEN，K=0 的 cudagraph 捕获 ZeroDivisionError） | READ BODY（4 条） | **≈0.55**（p20 三轮六臂） |
| 10 | **DFlash/DSpark × 前缀缓存的交互**（"命中仍要重算"） | issue [#47930](https://github.com/vllm-project/vllm/issues/47930)（**OPEN**，标题即我们的实测结论：*"DFlash/DSpark draft acceptance collapses with automatic prefix caching enabled"*）；PR [#47926](https://github.com/vllm-project/vllm/pull/47926)（OPEN/Draft，机制原文：dflash 需 target 辅助隐藏状态建 context KV，而**前缀缓存恢复的 token 从不过 target** ⇒ 读未初始化 KV；且明写 *"**MTP/EAGLE-style drafters … are unaffected**"*）；PR [#54163](https://github.com/vllm-project/vllm/pull/54163)（**OPEN**，*"the whole context was recomputed on every reply"*）；issue [#54094](https://github.com/vllm-project/vllm/issues/54094)（**OPEN**，环境栏正是 **RTX PRO 6000 Blackwell**） | READ BODY（4 条） | **≈0.15**（p21 四臂） |

**第 7 行（前缀复用 × 投机）的措辞须按实测收窄**：本表原先把它记成一类，**实测显示它是方法特异的** ——
在 vLLM 0.29 / Qwen3-4B / 4096-token prompt 上，**`dflash2` 第 2 遍前缀缓存命中 = 0（重算 100% prompt，256× 于不投机），
而 `eagle3`（2 blocks）与 `ngram`（1 block）与不投机同样正常**。⇒ 今后凡涉及此格，**必须按 drafter 类别分别陈述，不得写"EAGLE/MTP"**（上游 #47926 的机制说明与此一致）。

**第 4 行（批不变性）的适用前提被收窄（结论不变、理由改变）**：我此前关闭该线索的依据是"开关有效 ⇒ 干预点就是开关"，
但那批实验**全部在关闭前缀缓存下做**。p21 实测：**默认配置（前缀缓存开启）下 T=0 的确定性本身就不成立**
（同 prompt 冷 vs 热，文本在 `nospec` 7/8、`eagle3` 7/8、`ngram` **5/8** 相同；冷 vs 热的 token_logprobs 四臂 **8/8** 不同）。
⇒ **仍然不立项**，但理由从"已解决"改为"**未解决且已被他人占位**"（vLLM PR [#46592](https://github.com/vllm-project/vllm/pull/46592) 是他人 OPEN 的 canonical-chunking 实现，
跟踪 issue [#27433](https://github.com/vllm-project/vllm/issues/27433) 把 "Prefix caching support" 列为 help-wanted 未打勾）。

**本轮新增的正面数字（不立项，但已取证、可引用）**：投机在 **dense 4B / 4k prompt / 并发 8 / 前缀缓存开启** 下净赚 **+48.5%**（K=3）、**+11.2%**（K=1）；
**全 K=0 表 −31.7%**；**K=0 档的 ITL（17.4 ms）≈ 真投机档（18.0 ms）≫ 不投机（11.9 ms）**。
⇒ 与第 3 节"文献 A 类已 20+ 篇"的批不变性形成对照：**投机在本机小模型短上下文下是明确正收益**，地图 §1.9 那批负结果的条件（185k 上下文 / MoE / 长 CoT）与本机不同。

**✅ 结构性缺口计数不变：仍为 0。** 新增的两类都是 🔴（已被 OPEN PR/issue 占位）。

---

## 七、2026-09-14 续：**"OPEN" ≈ "正在施工"，不是"无人区"** —— 四簇独立线索、四次 S3b、全部被占位

**这一节是本轮最有价值的产出，比任何一个具体候选都重要。**

本节起因：我肉眼筛地图时漏掉了**主机内存 KV 卸载分层**这一簇（见 `notes/DIRECTION_MAP_3AXIS.md` §4.5），
用 `pipeline/tools/prescreen_map.py` 补扫捞回。补扫后又对它做了 S3b —— **同样被占位，而且占位者连地图自己都没记**：

| 格（地图标注 OPEN） | 地图原本引的证据 | **我 S3b 新查到的占位者** | 状态 |
|---|---|---|---|
| **B42** 卸载层之间没有背压检测 | #38470/#38448 等 | **vLLM PR [#50045](https://github.com/vllm-project/vllm/pull/50045) *"[KV Offloading] Back-pressure detection and remediation"*** | **OPEN**（标题与格子问题一一对应） |
| **B41** 分层卸载把所有等待请求都提升，冲刷主 DRAM 层 | #49902/#50014 | **vLLM PR [#49952](https://github.com/vllm-project/vllm/pull/49952) *"Reserve primary-tier headroom so speculative promotions don't starve running stores"*** | **CLOSED**（**merged 与否未取证**，不写成"已解决"） |
| **B44** host→device 回载坐在关键路径、未与 forward 重叠 | #38470/#38448 | **SGLang PR [#17843](https://github.com/sgl-project/sglang/pull/17843) *"[HiCache] Support direct io backend offload&load overlap"***（CLOSED）；**PR [#34519](https://github.com/sgl-project/sglang/pull/34519) *"fix(hicache): limit load-back pending to write-back"***（**MERGED**） | 部分已修 |

**四次独立 S3b 的汇总（本轮）**：

| # | 线索 | 我的实测投入 | S3b 判定 | 占位者 |
|---|---|---|---|---|
| 1 | DSD 的 K=0 档代价 | 0.55 GPU·h | 🔴 | PR #53426（OPEN，"K=0 draft sync forward"）、#49548、#48494、#47737 |
| 2 | DFlash/DSpark × 前缀缓存 | 0.15 GPU·h | 🔴 | #47930（OPEN）、#47926、#54163、#54094 |
| 3 | 前缀缓存 + batch-invariant | 0 GPU·h | 🔴 | PR #46592（OPEN），跟踪 issue #27433 列为 help-wanted |
| 4 | 主机内存 KV 卸载分层（B41/B42/B44） | 0 GPU·h（纯检索） | 🔴 | PR #50045（OPEN）、#49952、SGLang #34519（MERGED）/#17843 |

### 7.1 战略结论（必须记下来，否则下一轮还会犯）

**"地图上标 OPEN" 与 "无人占位" 是两件不同的事，而且在本领域它们高度负相关。**
原因是机制性的：一张三态地图的 **B 段是靠"有人还在问"的证据建起来的** ——
而**"有人还在问"恰恰是最容易长出 PR 的状态**。所以 B 段在很大程度上是**在飞工作的快照**，
不是**无人区的地图**。⇒ **B 段不该被当作候选池；它该被当作"施工路段警示牌"。**

**这解释了本轮的全部经历**：我按地图的"最开放"格去测，测一条撞一条，
四次全部命中 OPEN PR —— 不是运气差，是**筛选方向本身就偏了**：
我一直在"最热闹的地方"找"没人做过的事"。

**⇒ 对下一轮筛选的修订（写进 `notes/FILTER_3AXIS.md` 的待办）**：

1. **B 段（OPEN）应当降权**，而不是像现在这样被当作首要候选池。
2. **真正的候选池在两个地方**：
   - **C 段（试过被放弃）里"失败理由依赖某个已改变的约束"的那些**（FILTER §1a 原本就是这么设计的，
     但 §1 的"C 类优先"在本轮被 B 段的表观丰度盖过了 —— 这是我的执行偏差）；
   - **没有任何 issue/PR 讨论过的地方** —— 这最难找，但**唯一可能真正空闲**。
     可行的探法：从**已装引擎源码**里的 `TODO` / `not supported` / `NotImplementedError` 反查
     （`probes/p17-open-cells/scan_engine_todos.py` 已具备，274 条命中），
     再**逐条去 GitHub 搜是否已有人提** —— **搜不到才是信号**。
3. **S3b 必须前置到"花钱之前"，且证据要独立于地图**：本轮 #4 的占位者**地图自己都没记**
   （地图引的是另一批号）⇒ **不能拿地图当 S3b 的证据源**，地图只能当线索源。这条要写进 S3b 流程。

**✅ 结构性缺口计数仍为 0。**

---

## 八、2026-09-15 续：**B112（并发下前缀缓存命中率崩塌）由"关键词判死"升级为"机制不可达 + 匹配对照"**

**为什么单列**：这是三张地图里**唯一**一条我此前只能用关键词（`hybrid`）判"够不着"、却给不出机制的高价值格。
现在补上了机制**和**对照实验（`results/p22-dense-hit-collapse/2026-09-15/summary.md`，≈0.2 GPU·h）。

**上游被复核的因果断言**（vLLM C109 关闭理由，引 #42948 上 @stecasta 的插桩）：
*"the real bug is **131% pool overflow from the homogeneous `lcm=256` physical block layout vs DSv4-Flash's `{256, 64, 4, 8}`-block-size KV groups**"*。

**匹配对照（dense Qwen3-4B ⇒ 逐层 page size 均匀 ⇒ lcm = page size ⇒ 不溢出）2×2 结果 —— 四格逐位相同**：

| 配置 | KV 池 | 并发 | 冷命中率 | 热命中率 | 热态每请求重算 |
|---|---|---|---|---|---|
| loose_c1 | 319,328 | 1 | 0.0000 | **0.9981** | 8 |
| loose_c8 | 319,328 | **8** | 0.0000 | **0.9981** | 8 |
| tight_c1 | 146,448 | 1 | 0.0000 | **0.9981** | 8 |
| tight_c8 | 146,448 | **8**（占用 ~90%） | 0.0000 | **0.9981** | 8 |

⇒ **并发 8 + 池占用 ~90% 下崩塌不出现** ⇒ 该崩塌**需要异构 block size**；
报告者那条**一般性**机制假设（"引擎把自由块与已缓存条目混为一谈"）在本压力区间内**不成立**。
**⇒ B112 / C44 / C109 / C110 对本机：不可达（机制 + 对照），不只是关键词。**
**诚实边界**：未制造"请求大小混杂 / 抢占 / 极高 churn"（报告者的实际失败负载是 budget 受限的 agentic round-robin）；
且 4 格无重复（四格热命中率逐位相同是稳定性证据，但不是噪声估计）。

**同时固定一条仪器纪律（本目标第 5 处自我更正）**：**凡涉及前缀缓存的测量，必须先声明 drafter 类别** ——
p21 已证明 dflash 系把命中率压到 **0**（#47930），若沿用旧习惯带投机测命中率，本实验会得到**假阳性崩塌**。

---

## 九、2026-09-15 续：**筛子本身过时**（系统性偏差）＋ p30 结题 ＋ 三条 S3b 判决

### 9.1 系统性偏差：整池候选是在**一台已经不存在的机器**上过筛的

`pipeline/tools/prescreen_map.py` §0 硬编码硬件档为 **"单卡 RTX PRO 6000 96GB sm120"**，
这不是笔误 —— 它对应 2026-09-13 那台 AutoDL 机（`candidates/sglang-draft-corpus/00_intake.md:46`：
`RTX PRO 6000 Blackwell Server Edition（sm120）`, driver 580.82.09, CUDA 13, `memory.total 97887 MiB`），**该机已关机**。
现役是 `schoolserver`：**8× RTX 4090 24 GB, sm89, driver 550.67, CUDA 12.4**。

⇒ `HARDWARE_PAT` 在两个方向上同时错：
· 把一切 `multi-gpu|tensor-parallel|tp=[2-9]|nvlink|pcp|dcp|pipeline parallel` 判为"够不着"，
  而现机**有 8 张卡** ⇒ **该桶整桶属于 FILTER §1a**（失败理由依赖已改变的约束）；
· **不会**筛掉"需 96 GB 单卡"的条目，而那才是现在真正够不着的。
**修正动作**：凡"够不着"的结论在复述前必须重判；本文件以下条目一律以现机为准。

### 9.2 p30（思路 ⑥ Decode Dataflow 的**延迟侧**）：**杀掉**；第二次确认噪声地板规则

**上界**（§0.6 零卡）：p27 的 0.0548 ms/次 × 72 次/token = 3.95 ms/token，占地板 **34–59%** ⇒ 表面是 2% 杀线的 17–29 倍。
**实测**（真实依赖链 `x→gemm_i→all_reduce_i→gemm_{i+1}`）：`chain/no_comm` = **1.015**（p10 口径 1.045）
⇒ 暴露代价只有 **~1.5–4.5%**，比上界小一个数量级，**且低于同格噪声**（基线自身极差 **132.6%**）。
**根因**：p27 的 0.0548 ms 是"单发带同步"口径；背靠背 **0.0262–0.0270 ms**，链上暴露增量 **≈0.011 ms** ⇒ 上界高估 5–10×。
**这同时使 C2 的两种算法（0.107 ms 与 0.0548 ms）都不成立。**

**两条可用事实**：① decode 形状（5 KB）下本机集合通信有效带宽 **0.09 GB/s**，比同机 4 MB 时的 **12.4 GB/s** 低 **140×**；
② 因此**分块重叠在本机是反效果的**（把 1 个延迟事件变成 C 个），实测 `chunk8` **1.459** > `chain` 1.015。
**S3b 复核（`notes/S3B_PCIE_TP_OVERLAP_2026-09-15.md`）：`OCCUPIED`。** 决定性证据：
· **PRESERVE**（arXiv 2501.08192）已发表该机制本身（"prefetches … **during the communication operations**"）；
· **SiFAR**（2607.08973, MICRO 2026）做了匹配对照且**在小 batch 为负**："*there is not enough computation to overlap with communication*"；
· **FlashInfer PR #4393 已 MERGED**：PCIe 无 NVLink 定制 all-reduce **12 KiB 5.3 µs vs NCCL 205.1 µs（38.8×）**，
  且 **SGLang #34528（OPEN）** 在同一 fabric 上把 TPOT **21.14 → 13.62 ms**。
⇒ **这条轴在真实 fabric 上已发货的答案是"消除延迟"，不是"重叠"**，与本机实测（NCCL 26–55 µs）自洽。轴关闭。
**附带更正**：S3b 指出我的条件写反了 —— 税（2·L·t_ar）对模型规模**平**，而地板随模型规模**线性增长**，
故占比随模型**变大而缩小**；正确条件是"**足够大**"，不是"足够小"。

### 9.3 三条 S3b 判决（0 GPU·h，共 4 个子代理）

| 方向 | 判决 | 决定性反证 |
|---|---|---|
| ⑤ Reasoning-aware KV/Attention | **OCCUPIED** | **Random Attention**（2609.03430）跑了匹配对照并否定："*the selection signal contributes almost nothing*"；**RaaS**（ACL 2025 Findings, 2502.11147）已占 milestone-token 生命周期 |
| ⑥ Decode Dataflow | **PARTIALLY OCCUPIED** | **2605.30571**（"Memory-Bound but Not Bandwidth-Limited"）占 batch-1 decode 归因；**2512.01644** 占逐算子归因。存活切片仅剩"no-P2P host-staged 8×4090 双 NUMA"这一具体机型的字节-时间归因 |
| ⑧ Agentic LLM Runtime | **OCCUPIED** | **SAGA**（2605.00528）已把整个 workflow 当一等调度单位，并实测"*38% of execution time regenerating KV cache between agent steps*"；**ThunderAgent**（2602.13692）把 recompute 写进目标函数 |
| 数值漂移→决策翻转（p28 派生） | **PARTIALLY OCCUPIED** | "量级不预测翻转、margin+方向才预测"已被**两个互不相关的小组**发表（**MarginGate** 2605.30218；OpenReview QDOKyg7a5e, TMLR 在审） |

### 9.4 对**我自己** p28 头条数字的质疑（正在测，p31）

S3b 指出：p28 的"同 n 下 oracle vs random 差 3–31×"是**按 n 匹配而非按丢弃质量匹配**——
n=256 时 oracle 丢 0.0258、random 丢 0.4097，**质量差 16 倍**。
按质量横看 p28 自己的数据已**区间重叠**：oracle m=0.0880→TV 0.0880（比值 1.00）；
randn m=0.4097→TV 0.8060（1.97）；randn m=0.6204→TV 0.1926（0.31）。
⇒ **"440×/1323× 跨度"可能主要是质量效应而非结构效应。** `probes/p31-matched-mass/` 直接判它，
并同时记录已发表工作认定的真判据（top-1/top-2 logit margin 及其扰动 `d_gap`）。
**冻结判据**：匹配后 TV 比 <1.5× ⇒ 死；>3× 但 `d_gap` 分布重合（只有软指标差异）⇒ 也死。
