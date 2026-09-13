# S1 痛点发现（Pain discovery） — 草案语料生命周期：跨会话/跨副本的冷启动罚金

**候选**：`sglang-draft-corpus` ｜ **成本上限**：0 GPU·h（可用 ≤1 GPU·h 做量级取证）
**目标**：只收集**真痛点**：有人真的疼、有量级、有可复现证据。此阶段禁止提方案。

> 本阶段按 PIPELINE §8 的新规则执行：**每条痛点落地时同时做一次占位快扫**（不等到量化之后再查）。
> 占位快扫的源码级结论见 `30_occupancy.md` 与 decision #99/#101。

## 候选痛点清单

### P1 全新实例的首通接受长度塌到 ~1.8/8（冷启动罚金）

- **现象**：SGLang 0.5.19 + NGRAM（γ=7）在**全新 serve 进程**上跑 32 条 ctx=4096 的真实 prompt，
  **第一遍全程**平均接受长度 **1.8365**（`spec_accept_rate` 0.1228）、每 128 token 需 **~70 次** verify step；
  同一进程第二遍即 **6.9744**（+280%），第四遍 **7.7827**（饱和，+323.8%）。
  32 请求总 wall **25.22 s → 8.67 s（2.91×）**。
- **谁在疼**：任何**短会话/低复用**的部署形态 —— serverless 冷启动、按需扩容的新副本、滚动重启后的实例、
  A/B 或 canary 新版本；这些形态下**第一段流量**全部落在冷窗口里。
- **量级**：**L2**（单点真机实测，160 格逐请求数据，三条采集路径）。
- **证据**：`/root/ccfa_results/2026-09-13/p12_smoke/smoke.jsonl`（160 行逐请求）+ 复现命令
  `bash probes/p12-draft-corpus/smoke.sh`（探针 `probes/p12-draft-corpus/probe.py --selftest` 可离线自测）；
  汇总见 `candidates/sglang-draft-corpus/40_measurability.md` §仪器冒烟。
- **反证**：① 稳态已达 7.78/8（97.7% 接受率）⇒ 这个 workload 上**接受长度没有更多空间**，
  可回收的只有"冷窗口那一段"；② 若生产流量是**长会话、prompt 高度复用**，冷窗口占总请求比例很低，
  罚金被摊薄 ⇒ 本痛点的严重程度**与部署形态强相关**（这一点在 S2/S4 里被显式承认为适用范围限制）。

### P2 引擎的语料状态**不持久**：进程重启即归零（上游自认的缺口）

- **现象**：`ExternalCorpusManager` 只提供 `add` / `remove` / `list` 三个操作，**没有 save/snapshot/restore**，
  也没有任何磁盘或跨进程路径 ⇒ 进程一重启，语料全丢，回到 P1 的 1.84。
  同一文件里还有上游自己留下的 FIXME（豁免：这是**被引用的上游源码原文**，是本痛点的证据内容，不是本仓库的待办）：`remove` 在加载进行中属于**未定义行为**（作者原话见证据）。
- **谁在疼**：需要**弹性扩缩容 / 滚动升级 / 多副本**的部署者 —— 每次新副本都要从零重新积累语料。
- **量级**：**L1→L2**（源码路径 + 本工作区实测的冷窗口数字 P1）。
- **证据**：`/root/autodl-tmp/venvs/sglang/lib/python3.12/site-packages/sglang/srt/speculative/external_corpus_manager.py:23-111`
  （全文已读）；第 86-87 行原文（豁免：引用上游源码，非本仓库待办）：`# FIXME(kpham-sgl): remove a corpus during a pending load is an undefined behaviour and should be explicitly prevented.`
- **反证**：上游可以把持久化当作"部署者的事"（语料是纯 CPU 结构，理论上可由外部重建）。
  若外部重建的成本与收益不成比例，这条痛点的"该由引擎解决"就站不住 —— S3 必须正面回答。

### P3 上游 roadmap 把"无外置语料"列为**首要缺口**，并给了 ≥2× 的实测收益

- **现象**：SGLang 官方 roadmap issue 原文第 229 行：
  *"**No external corpus support.** The trie is only populated from the current decoding session's output tokens.
  Ngram speculative decoding works best with a large reference corpus, but there is currently no mechanism to load one."*；
  该 roadmap 的已完成项里包含一条**基准结论**：*"Accept length benchmark: output-as-corpus proves SAM ≥2x accept length boost"*。
  另有未勾选项 *"SAM for user request's input"*、*"Expand SAM match length (currently still cap by `max_trie_depth`)"*、
  *"Transport SAM across scheduler in multi-scheduler system (disagg, DP, etc)"*。
- **谁在疼**：**上游维护者自己**（这是他们写的 roadmap，不是我的推测）。
- **量级**：**L2**（上游给出的 ≥2× 与我们的 4.24× 同量级、互相独立）。
- **证据**：[sgl-project/sglang issue #21052](https://github.com/sgl-project/sglang/issues/21052)（2026-03-20 建，label `roadmap`/`speculative-decoding`）。
- **反证**：这条痛点在**上游已被处置了大半**（外置语料、多 SAM 动态 API、动态分配 PR、跨 scheduler 传输都已有 PR）
  ⇒ 直接撞车风险高，**差异化只能在剩余未勾选项或"生命周期"维度**上找。

### P4 固定 trie/SAM 预算分割在"语料已预置"时会把 draft 槽位浪费掉

- **现象**：SGLang 0.5.19 的 C++ 侧仍是**固定预算**：
  `ngram.cpp:173` = `num_sams > 0 ? std::min(param_.external_sam_budget, total_draft_token_num) : 0`
  ⇒ 外置 SAM 每步最多只能占 `external_sam_budget` 个 draft 槽（且被 `arg_groups/speculative_hook.py:867-880` 限制为 `≤ num_draft_tokens−1 = 7`），
  剩下的槽留给**会话内 trie**。而首通时 trie 几乎是空的。
- **谁在疼**：预置了大语料的用户 —— 他们的 SAM 明明能给出高质量 draft，却受固定配额限制。
- **量级**：**L1**（源码路径 + 上游 PR #22538 自述 *"removes the old fixed trie/SAM draft-budget split"*，
  但**该 PR 状态是 Closed（未 merge）**，且其基准数字在 PR #22569 里 ⇒ **具体百分比未取证**）。
- **证据**：`/root/autodl-tmp/venvs/sglang/lib/python3.12/site-packages/sglang/kernels/jit/csrc/ngram_corpus/ngram.cpp:173`；
  上游 [PR #22538](https://github.com/sgl-project/sglang/pull/22538)（Closed 2026-08-17）与基准 [PR #22569](https://github.com/sgl-project/sglang/pull/22569)。
- **反证**：上游已用**另一条路径**（PR #22737 per-request Trie、#22538 动态分配、#22294/#22471 多 SAM 增强）处置中
  ⇒ 这条更可能变成"与上游赛跑"，**不宜作为主线**，只作为 S4/S6 的一条免费观察。

### P5 跨引擎/跨方法的接受长度**不可直接比**，使"语料有没有用"在公开数字里读不出来

- **现象**：同名指标语义随引擎与投机方法漂移：vLLM 0.29 与 SGLang 0.5.19 在**同模型、同负载、同客户端**下
  接受长度差 **−32.6%**（1.479 vs 2.194 @bs32）；`itl/tpot` 比值在同一客户端下因投机方法不同从 **1.07×** 变 **2.85×**
  （源码级解释：`mean_itl_ms` 记在**每个流式 chunk**，而 chunk 携带的 token 数随方法变化）。
- **谁在疼**：任何**依据公开数字选择投机/评测配置**的人 —— 他们无法判断"别人报的 2× 收益"是否可迁移。
- **量级**：**L3**（多点实测 + 与基线对照；本工作区 24 格 + 24 bench 两轮实测）。
- **证据**：`results/p09-engine-conformance/2026-09-13/conformance.md`、`results/p08-specbench/2026-09-13/consistency.md`、
  `notes/prereg/engine-metric-conformance.md`；复现：`probes/p09-engine-conformance/run_conformance.sh`。
- **反证**：计数类指标**完全一致（0.00%）**、吞吐口径仅差 **4%**（<8% 闸门）⇒ 不能说"跨引擎比较普遍失真"，
  只能说"**接受长度与延迟类**指标不可直接比"。这条痛点是**测量属性**，与 P1/P2 的**优化属性**不同类（见 S2 淘汰台账）。

## 逐条证据

| 痛点 | 证据形态 | 路径 / 链接 | 可否独立复现 |
|---|---|---|---|
| P1 | 逐请求原始数据（仓库外）+ 一行命令 | `/root/ccfa_results/2026-09-13/p12_smoke/smoke.jsonl`；`bash probes/p12-draft-corpus/smoke.sh` | ✅ 可（探针含 `--selftest`） |
| P2 | 源码全文（含作者 FIXME（豁免：上游原文引用）） | `…/sglang/srt/speculative/external_corpus_manager.py:23-111`（服务器已装树） | ✅ 可（`grep -n FIXME` 一行） |
| P3 | 上游 roadmap 正文 | [issue #21052](https://github.com/sgl-project/sglang/issues/21052) | ✅ 可（公开页，2026-09-13 取） |
| P4 | C++ 源码行 + 上游 PR 状态 | `…/kernels/jit/csrc/ngram_corpus/ngram.cpp:173`；[PR #22538](https://github.com/sgl-project/sglang/pull/22538) | ✅ 可（但百分比未取证 ⇒ L1） |
| P5 | 两轮实测派生表 + 预登记 | `results/p09-engine-conformance/2026-09-13/conformance.md`、`results/p08-specbench/2026-09-13/consistency.md` | ✅ 可（`run_conformance.sh`） |

## 不疼的反证

**主动找证据说明"这个痛点可能不存在或不重要"**（PIPELINE §4 S1 硬要求）：

| # | 反证 | 对哪个痛点构成威胁 | 现在的处置 |
|---|---|---|---|
| R1 | **稳态已饱和在 7.78/8**，说明该 workload 上语料几乎"一学就会"；若真实流量同样如此，冷窗口只占极小比例 | P1（罚金的**重要性**） | 承认；判据因此用**相对量** `recovery`（冷窗口闭合比例），并在适用范围里写明"与部署形态强相关" |
| R2 | 上游已有 PR 把这条线做完大半（外置语料 / 多 SAM / 动态分配 / 跨 scheduler 传输） | P2、P3（**未被治理**这一条） | 差异化只能落在**生命周期与恢复**（上游唯一没有 PR 的维度）+ **跨副本**；S3 必须给结构性理由 |
| R3 | 引擎**自带**会话内累积（第二遍就 6.97）⇒ "预置语料"本身是**现成原语**，不是创新 | 整条线索的**创新性** | 用户硬约束 #5 的直接适用：创新必须落在**引擎不覆盖的维度**（跨会话/跨重启/跨副本），否则判为工程配置 |
| R4 | 用**同一批 prompt** 既做语料又做测试，是"最好情形"，真实流量下语料靠跨请求重叠增长，收益会小很多 | P1 的**效应量可迁移性** | 承认；`E2` 条款已登记"外部分布语料"作为补测；结论必须写明这是**上界形态** |
| R5 | 冷窗口罚金在**并发>1** 时会因批内多样本而改变（NGRAM 命中率随 batch 组成变化） | P1 的**适用范围** | 本轮**只测串行**，并把并发明写为未测区间（预登记 §网格已声明） |

## 量级汇总

按量级排序（L1 以上进入下一阶段）：

| 序 | 痛点 | 量级 | 关键数字 | 来源类型 |
|---|---|---|---|---|
| 1 | **P1 冷启动罚金** | **L2** | 首通 **1.8365** vs 饱和 **7.7827**（**4.24×**）；wall **25.22 s → 8.67 s（2.91×）** | **自有实测**（160 格） |
| 2 | **P5 跨引擎接受长度不可比** | **L3** | **−32.6%**（1.479 vs 2.194）；`itl/tpot` 1.07× vs 2.85× | 自有实测（两轮 48 格/bench） |
| 3 | **P2 语料不持久** | L1→L2 | 无 save/restore 路径（源码）+ 作者 FIXME（豁免：上游原文引用）；重启即回到 1.84 | 源码 + P1 实测 |
| 4 | **P3 上游自认首要缺口** | L2 | 上游基准 *"SAM ≥2× accept length boost"* | **他人**（上游 roadmap） |
| 5 | **P4 固定预算分割** | L1 | `min(external_sam_budget, total_draft_token_num)`；上游 PR 自述要移除但 **Closed** | 源码 + 上游 PR |

**来自"别人"的痛点 ≥2 条**：P3（上游 roadmap 原文 + 上游基准）、P4（上游 PR #22538 自述 + 基准 PR #22569）、
以及 P5 里的 vLLM issue #19776 / doc PR #55283 与 InferenceBench §3.2（见 `candidates/engine-metric-conformance/10_pains.md`）。✅

**L0 条目：无**（本清单不含"只有说法没有数字"的条目）。

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] ≥5 条候选痛点，每条包含『现象 / 谁在疼 / 量级 L0–L3 / 证据 / 反证』五个字段 —— P1–P5 共 5 条，每条五字段齐全
- [x] 每条证据是**可复现命令**或**来源链接**；两者都没有的标 L0 并写『未取证』 —— P1 给 `bash probes/p12-draft-corpus/smoke.sh` 与仓库外原始数据路径；P2/P4 给 `file:line`；P3 给 issue 链接；P5 给派生表路径与 `run_conformance.sh`；P4 的百分比按实标 **L1 / 未取证**
- [x] 至少 2 条来自**别人**的痛点（生产事故报告、引擎 issue 里的实测数字、已发表测量论文的开放问题），不是自己推测的 —— **P3**（上游 roadmap issue #21052 原文第 229 行 + 上游基准 *"SAM ≥2× accept length boost"*）、**P4**（上游 PR #22538 自述与 #22569 基准）；旁证还有 vLLM issue #19776 与 doc PR #55283
- [x] 记录『不疼的反证』：主动找证据说明这个痛点可能不存在或不重要 —— §不疼的反证 R1–R5，其中 R3（引擎自带累积 ⇒ 预置语料不是创新）与 R4（同批 prompt 是最好情形）直接限制本线索的主张范围
- [x] 量级汇总表按 L1 以上条目排序，L0 条目不进入下一阶段 —— §量级汇总 5 条全部 ≥L1（最高 L3），**无 L0 条目**

> **杀出口**：全部条目 < L1（都没有数字）⇒ 回到搜证，不许进入 S2 谈方案
> 本候选**未触发**：最高量级 L3，且 P1/P5 为自有真机实测。
