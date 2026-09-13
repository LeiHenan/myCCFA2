# 三方向地图（推理加速 / 投机解码 / KV Cache）—— 汇总 + 我的亲验

**日期**：2026-09-13 起 ｜ **目标**：`goal-a729588f` ｜ **过筛标准**：`notes/FILTER_3AXIS.md`（**地图返回前冻结**）
**证据分层（重要）**：本文件区分 **① 我亲验**（读了源码行 / 跑了实验）与 **② 子代理材料**（不得直接当结论，见 FILTER §7）。

---

## 一、投机解码（子代理地图：276 行，374 个 provenance 标签，328 个不同 URL）

主交付：`spec-decode-gap-map-2026-09-13.md`（997 行）+ 5 份原始 sweep。
**子代理自报的方法学**：GitHub 评论体可经 `react-app.embeddedData` 取；状态以 `data-status` 徽章为准；`api.github.com` 几乎全程限流为 0；`export.arxiv.org/api` 全程 429/503，改用 `arxiv.org/search` HTML。**并自报 `web_search` 期间返回过伪造 arXiv ID**（2402.05099 实为 *Hydragen* 等）⇒ **地图中每个 arXiv ID 都经 `arxiv.org/abs/<id>` 读回标题与作者核验**（除标 TITLE ONLY 的 16 行）。

### 1.1 ⚠️ 被地图推翻的一条（**我接受更正**）—— 我早先的 observability 缺口已被 ship

| 项 | 内容 | 我的核实 |
|---|---|---|
| 命题 | vLLM **0.29.0 已 ship** 每请求投机指标：`--per-request-spec-decode-metrics {none,summary,detailed}`，返回 `mean_acceptance_length` / `draft_acceptance_rate` / `acceptance_histogram` / `num_spec_steps` / `num_accepted_draft_tokens` / `num_draft_tokens` / `num_spec_tokens`，并在 `detailed` 下给 `per_step_accepted`/`per_step_drafted` | ✅ **亲验**：`config/observability.py:48` 有该字段（默认 `"none"`），`config/vllm.py:1339-1343` 有校验 |
| 版本差证据 | 该 doc 文件 **v0.28.0 返回 404、v0.29.0 返回 200** ⇒ 新增于我们 pin 的版本（PR #48915，2026-09-09 发布说明） | 子代理材料（未亲自复核 404/200） |
| 含义 | 我此前引为痛点的 `vllm.cpp #2770`（"服务端不导出接受率"）**已被上游关闭** | 待核（TITLE 级） |
| 剩余空间（更窄） | schema 自述 *"experimental and its shape may change"*；`n>1` 时为 **`null`**；`num_draft_tokens` 定义含 *"after subtracting drafts invalidated by structured-output constraints"* ⇒ **语法拒绝被静默折进分母** | 子代理材料 |

### 1.2 ⚠️ 有人**正在**索取与我同题的观测能力（OPEN）

vLLM **PR #54748**（OPEN，needs-rebase）+ **RFC #54749**（OPEN，2026-09-01）。子代理给的引文几乎就是我的论点：
> *"Steps the schedule sent to K=0 increment no counter, so 'the schedule stopped speculating on purpose' and 'speculation is not running' look identical in metrics."*
> *"I have been measuring dynamic-SD schedules and trying to calibrate a cost model that picks K. The coefficients were guesses and the model picked the wrong tier; there is no counter that would have told me so from inside vLLM."*

**⇒ 判据含义**：这是**另一个团队正在做的同一件事** ⇒ 按 FILTER §1（C 类优先）与 §6（S3b 三问），这条**不能作为我的候选**（会被判"已有开放提案在办"）。

### 1.3 ✅ 与我**同一台硬件**的 issue，且状态是 **CLOSED NOT_PLANNED / 零评论**

SGLang **#36001**：提交于 **"GPU: RTX PRO 6000 Blackwell Workstation Edition, 1x, 96GB"**，配 **`--speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2`** 与 `--kv-cache-dtype nvfp4`。原话：
> *"any speculative-decoding draft-extend/verify step that ends up on the FlashInfer prefill/extend path with an NVFP4 KV cache will hit this deterministically."*
> *"there's currently no working `--speculative-draft-attention-backend` value for NVFP4 KV cache at all."*
> *"Dropping only `--kv-cache-dtype nvfp4` (everything else identical) boots and serves correctly."*

**⇒ 这是一条 C 类格子（试过、被关、零评论）**，且**硬件与我完全一致**。判据上很有价值；但按 FILTER §1a，需先问"失败理由是否依赖某个已改变的约束"——它依赖 **NVFP4 KV cache**，而**我们的实验全程用 bf16 KV**，所以我需要先解决"能否在本机加载 NVFP4 模型/是否值得引入该轴"。

### 1.4 🔴 一条与 vLLM 自家文档冲突的**已发表证明**

arXiv **2605.07698**（2026-05-08）原话：
> *"any speculative decoder with local mask access, Leviathan rejection, and rollback soundness samples from the locally projected distribution mu^proj rather than the grammar-conditional distribution mu^star... on Dyck grammars with Qwen3-8B, the total-variation gap can reach 0.996."*

vLLM 文档声称 *"algorithmically validated to be lossless"* —— **只在无约束采样下成立**；带语法掩码时该保证在"语法条件分布"意义下不成立。
（子代理材料，**我未读该文正文** ⇒ 按 FILTER §7 记 **未取证**。）

### 1.5 C 段要点（子代理材料）

- **两条维护者对"自适应投机长度"的明确拒绝**（vLLM 维护者 benchislett）：PR #44885 *"the surface area of this feature is too significant to justify the marginal and hardware-specific gains"*；PR #43522 *"I feel strongly that we must fundamentally avoid synchronization whenever possible."*
- **已 ship 的动态投机调度（DSD）有多条实测负结果**：vLLM #49986（**−31% @ctx=400**，K=3 静态 −21%）；#49548（`[[1,4,2],[5,512,0]]` 使 8 并发从 ~232 tok/s 崩到 24–157）；**#48494 最关键**：*"[DSD] incur[s] a 12–25% throughput penalty vs no-spec ... even with an all-K=0 table producing nearly zero draft tokens"* + *"The presence of the batch-size table is the trigger; K=0 tiers are not required."*
- SGLang 自家文档：*"If your workload is already stable and one static setting is well tuned, adaptive mode may not help much."*
- **诚实边界（子代理自报）**：**可访问记录中不存在**任何维护者用 "not planned"/"wontfix" 字样谈论投机组合的陈述；所有 `not planned` 关闭都是**零评论的状态变更**。它把这一点记为发现，而**没有编造引文**。

### 1.6 我**亲验**的补充（本轮）

- **`draft_model` 方法是合法的**（`config/speculative.py:75`）⇒ 地图所述"Qwen3-4B + 小 drafter 路径已 merge（PR #24322，基准机正是 RTX PRO 6000 96 GB）"**与安装版本一致**。**但本机只有 `Qwen3-4B` 与 `dflash2`，没有小 drafter** ⇒ 这是一条**本工作区从未触碰过的能力**（我们只做过 dflash/MTP 与 NGRAM）。
- **DSD 默认关闭**：`num_speculative_tokens_per_batch_size` 默认 **`None`**（`config/speculative.py:473`），开关是 `uses_dynamic_speculative_decoding()`（:1855-1856）⇒ §1.5 那批吞吐税**只影响显式配置该表的用户**。

---

## 二、KV Cache（子代理地图）

状态：**仍在运行**。到位后并入本文件。

---

## 三、待过筛的格子（按 FILTER §0–§7）

| 格 | 类型 | 一句话问题 | 我的初判 |
|---|---|---|---|
| **C-DSD** | 被放弃/负结果 | 已 ship 的动态投机调度在 K=0 表下仍付 12–25% 税（"表的**存在**即触发"） | **最接近可测**：默认关闭 ⇒ 可用我们已有仪器构造 K=0 表做**独立复现**；但需先答 FILTER §3（差异轴是否条件性）——若只是"vLLM 的 bug 且已有人报"，则价值有限 |
| **C-NVFP4-KV** | 被放弃（零评论） | 同硬件 issue：NVFP4 KV cache 下投机 draft-extend/verify 必崩 | 硬件一致，但**需引入 NVFP4 轴**（我们全程 bf16 KV）⇒ 半径与收益待估 |
| **B-observability** | 开放 | 引擎不导出"调度刻意 K=0"与"未在投机"的区分 | 🔴 **不可立项**：他人正在做（PR #54748 / RFC #54749） |
| **B-draft_model** | 开放（能力） | 我们**没有**小 drafter ⇒ 三种投机方法（dflash/NGRAM/draft_model）的同台对照从未在本机做过 | ⚠️ 属"能力补齐"而非"缺口"；需 FILTER §3 条件性理由 |
| **A-grammar-lossless** | 已发表证明 | 语法掩码下投机不保持"语法条件分布" | ⚠️ 纯理论/正确性，**不是优化类**；且已发表 |

**淘汰（本轮）**：B-observability（他人正在做）。
**待定**：C-DSD、C-NVFP4-KV、B-draft_model、A-grammar-lossless。
