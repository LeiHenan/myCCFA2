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
