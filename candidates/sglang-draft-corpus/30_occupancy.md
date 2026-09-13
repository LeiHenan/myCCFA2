# S3 占位与先行核查（Occupancy & prior art） — 草案语料生命周期：跨会话/跨副本的冷启动罚金

**候选**：`sglang-draft-corpus` ｜ **成本上限**：0 GPU·h（检索 + 读正文）
**目标**：确认没人已经做完，且**引擎里已 ship 的旋钮**不是你的方案。必须读正文。

> **方法说明**：GitHub API 在本环境被限流（`API rate limit exceeded`），故占位核查走两条**可核**路径：
> ① **服务器上已装的 SGLang 0.5.19 源码树**（权威事实源，含 `file:line`）；
> ② 上游 issue/PR **正文**（经 jina reader 抓全文后逐项 grep，不是只看标题或摘要）。
> 所有引用都标了**读到的深度**；**未读正文的条目一律标注"未取证"**。

## 最近邻（≥3，标注读到的深度）

| # | 最近邻 | 读到的深度 | 它做了什么 | **它没做什么**（我们的差异轴） |
|---|---|---|---|---|
| **N1** | **[sgl-project/sglang issue #21052](https://github.com/sgl-project/sglang/issues/21052)**「[Roadmap] Further Ngram Speculative Decoding Support」（2026-03-20，label `roadmap` / `speculative-decoding`） | **正文全文**（抓取 627 行，逐项 grep） | 原文把 *"No external corpus support. The trie is only populated from the current decoding session's output tokens… there is currently no mechanism to load one."* 列为**第 1 号缺口**；已完成项含外置语料加载（#21425）、多 SAM 动态 HTTP API（#22203）、多 SAM 增强（#22294/#22471）、动态配额（#22538）、per-request Trie（#22737）、跨 scheduler 传输 | 未勾选项里与我们最近的是 `Transport SAM across scheduler (disagg, DP, etc)`；**"重启后恢复/持久化"在整份 roadmap 里没有任何条目**（逐项 grep 无 persist/restore/snapshot 相关词） |
| **N2** | **[PR #22538](https://github.com/sgl-project/sglang/pull/22538)**「[Spec][Ngram] 7/N: Dynamically select draft token counts from SAMs and Trie」（**Closed，未 merge**；2026-04-10 开，2026-08-17 关） | **正文 + 会话**（含作者自述与关闭事件） | 自述 *"This PR removes the old fixed trie/SAM draft-budget split… sources are merged in score order, capped only by `num_draft_tokens`"*，打分为 `score = source_prior * (w_specificity * specificity + w_confidence * confidence)`，并删掉 `external_sam_budget` 与 `min_trie_share` | **该 PR 关闭 ⇒ 0.5.19 仍是固定预算**（实测 `ngram.cpp:173`）；但**方向已被占** ⇒ 我们**不把"动态配额"当创新点**（仅作 S6 的一次免费观察，且只在固定预算留下可测缺口时才报） |
| **N3** | **[PR #22569](https://github.com/sgl-project/sglang/pull/22569)**「output-as-corpus and distractor corpora to benchmark dynamic spec tokens allocation」 | **仅标题 + roadmap 内转述**（*"Accept length benchmark: output-as-corpus proves SAM ≥2x accept length boost"*）—— **未读正文，百分比未取证** | 上游用 output-as-corpus 证明 SAM 带来 **≥2× 接受长度** | 它测**稳态收益**；我们测**冷窗口的恢复**（`recovery`），且上游没有"重启/新副本"这一维度 |
| **N4** | **HiCache**（`--enable-hierarchical-cache` / `--hicache-storage-backend {file,mooncake,hf3fs,nixl,aibrix}`） | **源码参数节**（`server_args.py:2722-2817`）+ 全仓 grep | 给 **KV cache** 做层级存储与持久化（host 内存 + storage backend + prefetch 策略） | **完全不覆盖草案语料**：`NgramCorpus` 是纯 CPU 的 SAM/trie 结构，**不在 KV 池里**；全仓 grep 未发现任何把 corpus/SAM 写进 storage backend 的路径 |
| **N5** | **agentic-kv-cache / Continuum / `--enable-session-radix-cache`** | **仅检索标题级 + `server_args.py:1444` 参数说明** ⇒ **正文未读，判定未取证** | 做 **KV** 的会话级生命周期 / TTL / 副本引用 | 对象不同：KV 有 per-token 状态、占显存、跨副本需传输；**草案语料是可纯拷贝的 CPU 结构** ⇒ 生命周期策略（选择/淘汰/共享）会得出**不同结论**。⚠️ **本条未读正文**，见 §未完成项 |

## 引擎已 ship 的控制器（写成最强基线）

| 已 ship 的机制 | 位置 | 在我们实验里如何被当作基线 |
|---|---|---|
| **会话内自动累积** | NGRAM trie 由解码过程 `insert` 填充（`ngram_worker.py` 插入路径） | **C 臂（饱和 7.78）** —— 引擎**不给任何外部帮助**时能到的高度 |
| **`POST /add_external_corpus`（多语料、运行中、异步线程）** | `http_server.py:998`；`tokenizer_control_mixin.py:193-266`；`external_corpus_manager.py:23-111` | **B 臂的干预手段**：**不自研接口**，只用引擎已 ship 的 API ⇒ 若它就能闭合冷窗口，创新点必须落在别处（否则判工程配置） |
| **`documents` 直喂文本（引擎自 tokenize）** | `tokenizer_control_mixin.py:224-243`（插 `SEPARATOR_TOKEN`；超 `max_tokens` 截断） | B 臂用 `documents` 而非 `file_path` ⇒ 绕开上轮 decision #95/#96 的 JSONL 格式约束 |
| **`--speculative-ngram-external-sam-budget`（固定配额）** | `server_args.py:2338`；`ngram.cpp:173` | **"引擎默认最优固定配置"**：用它的极值（≤7）与合理值做对照，判定固定配额是否留下可测缺口 |
| **`--speculative-ngram-max-trie-depth`（默认 18）** | `server_args.py:2346` | SAM 匹配长度被它掐住（roadmap 未勾选项之一），**本轮不动**，仅记为适用范围 |
| **前缀缓存 / radix cache** | `--disable-radix-cache` 可控 | 全网格**关闭**并在 C1 强制 `cached_tokens == 0` ⇒ 排除"更一般机制"解释 |

## 差异轴

逐条给出"最近邻做了 X，我们做 Y，因此在**什么条件下**结论不同"：

1. **对象轴**：最近邻（N3）优化 **SAM 的稳态收益**（output-as-corpus）；我们优化**语料状态在时间轴上的起点**（新进程/新副本的首通）。
   **条件不同处**：实例寿命 ≫ 冷窗口时两者结论一致（稳态都到 7.78）；当实例寿命 ~ 冷窗口量级（弹性扩缩、canary、serverless）时，**只有我们的量会变**。

2. **粒度轴**：最近邻（N2 与 N1 完成项）在**每步 draft 槽位分配**这一粒度决策（trie vs SAM 配额）；我们在**语料的存在性与恢复**这一粒度决策（进程/副本边界）。
   **条件不同处**：配额分配只在"trie 与 SAM 都有候选"时有意义；**首通时 trie 近乎空**，配额再优也无候选可给。

3. **预算轴**：最近邻把决策与 **`num_draft_tokens`** 耦合（≤7 个槽怎么分）；我们把决策与**语料装载成本 + 生命周期**耦合（装多少 token、何时装、重启后如何恢复）。
   **条件不同处**：`external_sam_budget` 的取值空间是 [0,7] 的整数（**离散且极小**），而语料装载的成本/收益**随语料规模连续变化** ⇒ 两者不是同一个优化问题。

## 结构性理由

**审稿人会问的那一问**：

> "**引擎已经 ship 了 `POST /add_external_corpus`，上游 roadmap 也把外置语料做完了（#21425/#22203/#22538），你凭什么说还有事可做？**"

**结构性回答（三条，逐条可核）**：

1. **"有 API" ≠ "有生命周期"**：`ExternalCorpusManager` 全文只有 `add` / `remove` / `list`（`external_corpus_manager.py:23-111`）；
   `NgramCorpus` 的 FFI 面只有 `insert` / `match_stateful` / `erase_states` / `load_external_corpus_named` /
   `remove_corpus` / `list_corpora`（`kernels/ops/speculative/ngram_corpus.py:45-135`）；C++ 侧 grep **无** dump/snapshot/serialize/save/export。
   ⇒ **结构上不存在**把语料状态带出进程的路径：新副本/重启只能**重新推送原始文本**（若原始文本在手）或**从零重学**。
   这是"缺接口"而非"没人想到"（TOPIC_METHODOLOGY §1.1 条件 3 的判据）。

2. **上游 roadmap 自己把这件事漏在外面**：整份 #21052 逐项 grep，**没有** persistence / restore / snapshot 条目；
   未勾选项里与我们最近的是 `Transport SAM across scheduler (disagg, DP, etc)` —— 那是**同一次运行内**的传输，
   **不是**跨重启的状态恢复。⇒ 我们打的不是"上游正在做的同一件事"。

3. **判据是数字，不是态度**：Go 条件为 `recovery ≥ 80%` 且效应 ≥3σ。
   若 `recovery` 高（≥80%）⇒ 说明"推送预置语料"这一现成原语**已足够**，创新必须落在**恢复成本/跨副本一致性**上（E5 条款，新开预登记）；
   若 `recovery` 低（≤0）⇒ 说明**现成原语不足**，缺口在配额/合并策略（与 N2 相关但 N2 已关闭）。
   **两种结果都有可写的结构性结论** ⇒ 不是"没人做过"式立项。

## 被占时的降级

| 若发生 | 降级成什么 |
|---|---|
| `recovery ≥ 80%`（预置语料完全够用） | 主张**收窄到恢复成本与跨副本一致性**：量化"重新推送语料"（装载耗时、CPU 内存、带宽）与"状态恢复"的差距；**新开独立预登记**（E5） |
| `recovery ≤ 0`（预置无效） | 主张**收窄到固定配额的缺口**：用 `external_sam_budget` 极值扫描量化固定配额限制了多少（与 N2 相关，须先读 #22538 完整 diff 划界）；缺口 <8% ⇒ **杀** |
| 上游在此期间 merge 了持久化 PR | 本线索**降级为现象记录**（冷窗口刻画 + 进程间噪声 + 可分辨性），按 §1B 形态结题 |

## 未完成项（诚实登记，不静默略过）

- **N5（agentic-kv-cache / Continuum）只读了标题级信息，正文未读** ⇒ 若本候选进入下一阶段，**必须**补读并逐条划边界；
  当前"对象不同（KV vs 语料）"的判定为 **L1，未取证**。
- **PR #22569 的百分比未取证**（仅从 roadmap 转述得到 "≥2×"）。
- **上游 #22203 / #21425 的完整 diff 未读**（读过 `server_args.py` 与 `http_server.py` 的**终态实现**，
  未读 PR 讨论中的设计取舍与权衡）。

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] 每个存活痛点找 ≥3 个最近邻，且记录**读过的具体页/节**（只读摘要不算） —— N1（issue #21052 **正文全文**，引用其第 229 行原文）、N2（PR #22538 **正文 + 会话**，引用作者自述与关闭事件）、N3（roadmap 内转述，**标注未读正文**）、N4（**源码参数节** `server_args.py:2722-2817` + 全仓 grep）、N5（**标注仅标题级、未取证**）
- [x] 找齐目标引擎里**已 ship** 的相关旋钮/调度器，并写成最强基线（不是只写学术邻居） —— §引擎已 ship 的控制器 6 项：会话内累积（C 臂）、`POST /add_external_corpus`（B 臂手段）、`documents` 直喂、`external_sam_budget` 固定配额、`max_trie_depth`、radix cache 开关
- [x] 写出审稿人会问的那一问，并给出**结构性**回答（不是『我们更自适应』这类程度性回答） —— 问句为"引擎已 ship API、上游 roadmap 已做完外置语料，凭什么还有事可做"；回答是三条结构性事实：① 管理器/FFI/C++ 三处**结构上无持久化路径**（给 `file:line`）；② 上游 roadmap **无该条目**（逐项 grep）；③ 判据是数字，高低两种结果都有可写结论
- [x] 写明『若被别人占住，本方向降级成什么』（降级形态必须先想好） —— §被占时的降级 3 行：`recovery≥80%` ⇒ 收窄到恢复成本/跨副本一致性（新预登记）；`recovery≤0` ⇒ 收窄到固定配额缺口（缺口 <8% 则杀）；上游 merge 持久化 PR ⇒ 降级为现象记录并按 §1B 结题

> **杀出口**：立项理由只有『没人做过』⇒ 杀；给不出结构性理由 ⇒ 改写定位或杀
> 本候选**未触发**：立项理由不是"没人做过"，而是"**结构性无持久化路径 + 上游 roadmap 未列该条目 + 判据是数字**"三条可核事实。
