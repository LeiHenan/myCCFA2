# 三方向地图（引擎执行代价 / 投机解码 / KV Cache 优化）—— 汇总 + 我的亲验

> **方向命名的修订（2026-09-14）**：第一个方向原叫"**推理加速**"，现改名为「**引擎执行代价**」。
> 依据是把吞吐写成 `吞吐 = 每步产出的 token 数 ÷ 每步执行代价` —— 三个方向恰好各占一项且互不重叠：
> ① **引擎执行代价** = **分母**（每步花多少）：算子与后端选择 / 编译与 CUDA graph / 权重与激活量化（非 KV）/
> 批调度与 host-bound 开销 / 采样与结构化输出开销 / tokenizer-detokenizer 开销 / 加速类测量方法学；
> ② **投机解码** = **分子**（每步产出多少 token）；③ **KV Cache 优化** = **约束与复用**（装得下多少、能复用多少，
> 它同时决定分子的上限与内存约束）。
> **改名的理由**：原词"推理加速"是**结果**而非**研究对象**（严格说三个方向都算推理加速），**无法据以筛选**；
> "引擎执行代价"是**可测的量**，且与本工作区既有仪器直接对齐（`spec_step_ms`、`ITL`、
> `Δln(tok/s) = Δln(接受) − Δln(每步代价)`，见 `results/p06-frontier/2026-09-13/A1_4k/mechanism.csv`）。
> ⚠️ 历史文档里出现的"推理加速"一词（含决策记录中对用户原话的引用）**一律保留原样**，不改写历史。
> 第三张地图的文件名沿用 `INFERENCE_ACCEL_GAP_MAP.md`（文件名只是文件名，方向名以本行为准）。

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

## 二、KV Cache（子代理地图：3,541 行，591 个不同 URL）

主交付：`KV_CACHE_GAP_MAP.md`（3,541 行 / 514 KB）。原始证据在 `evidence/kv-gapmap/`（S1–S14.md + S1–S14.verify.md +
`_index.jsonl` 697 个证据块 + `_in_A..D.tsv`）。
**证据分层**：以下**全部是子代理材料**（含一层对抗验证），**我尚未逐条亲验**。按 FILTER §7，进入 S1 的量级主张
**要么我亲自复现、要么标"未取证"**。

### 2.1 规模与可靠性（决定我该信多少）

| 段 | 条数 | 形态 |
|---|---|---|
| **A 已闭合** | 131 | 7 列表格（含 artifact URL 与 merged/shipped 状态） |
| **B 仍开放** | 116 | 每条带 WHO / URL / **提问者原话** / 缺什么 |
| **C 试过被放弃** | 185 | 六个子题（闭而未合 27 / wontfix 29 / 陈旧关闭 45 / 撤回 22 / **实测负结果 56** / 被自身局限推翻的论文 6） |
| **D 硬件排除** | 94 | D.1 >1 GPU (47) · D.2 >96 GB (2) · D.3 多节点/NVLink/IB (26) · D.4 存储网络 (12) · D.5 边界 (7) |

**这层对抗验证挣到了它的成本（我据此调整信任度）**：一条标 OPEN 的其实已被 merge 的 PR 修掉（vLLM #46971 ← #46972）；
一条标 ABANDONED 的产物其实仍开着（Dynamo #13794 ⇒ 移入 B95）；一条"3 个 PR 已 merge"覆盖了一个**未 merge** 的 PR（#54307）；
**一条硬件引文被静默编辑过** —— CacheGen 的 *"an NVIDIA A40 GPU server **with four GPUs**"* 在引号内被删掉"四张 GPU"，
从单卡翻成多卡（该论文因此移入 D 段）；**6 条 ABANDONED 因"只有陈旧机器人通知、无人话"而被整条丢弃**；
两个验证者还发现某段的中心方法学主张（"PR 评论体不渲染"）**是假的**，它此前掩盖了约 5 条真实的维护者关闭理由。

**引文保真度（子代理自审，未被掩饰）**：对 C 段 40 条做盲抽样回源机械比对 ⇒ **28/40 (70%) 在原 URL 逐字复现**，
6/40 是"真实引文嵌在检索到的框架文字里"，6/40 是转述/复合。因此 **185 条里有 74 条明确标注
`STATED REASON (composite: …)`**，其余 111 条才是单一连续引文。⇒ **任何 C 段的"理由"在用之前必须先看标签**。

### 2.2 我按 FILTER §0 的初筛（**只筛"本机可达"**）

先压掉三类（不逐条讨论）：

- **需要 hybrid Mamba/GDN 模型**（B1、B2、B4–B16、B19、B20、B80、B105、B112、B115）：我们的目标是 **Qwen3-4B（dense 全注意力）**，
  这类格子要引入一个全新的模型族才谈得上测。**不是"不可做"，是"不在现有半径内"** —— 若要打开，需单独立项并先算下载/适配成本。
- **需要多卡 / 多节点 / 存储网络**（B63–B68、B81–B87、B95、B96–B102）：本机 1 卡，直接排除（与 D 段 94 条一致）。
- **需要 MLA / DeepSeek 系模型**（B54–B57、B59、B69、B70）：Qwen3-4B 不是 MLA；且 B55/B56 已记录 MLA 在 **SM120 上本身就有问题**
  （*"Why can a NoPE MLA model not be served at all on SM120"*）⇒ 打开它等于先修硬件适配。

**剩下的、我这台机器够得着的格子，收敛到一个簇（这是本节最重要的结论）**：

| 格 | 一句话问题 | 与我已有资产的关系 |
|---|---|---|
| **B108** | 前缀缓存在 vLLM 的 **batch-invariant 模式**下**仍不支持**；跟踪 issue #27433 把"Prefix caching support"列为 **help-wanted 未打勾**，实现 PR **#46592 自 2026-06-24 起等待 code-owner 评审**；机制是 cache 命中长度不同 ⇒ prefill 被切成任意 chunk 边界（`VLLM_BATCH_INVARIANT_CANONICAL_PREFILL_CHUNK_BLOCKS`） | **我的 decision #116–#120 全部是在"前缀缓存关掉"下做的**（`--no-enable-prefix-caching` / `--disable-radix-cache`）⇒ **我在 p15/p16 里从未测过这个组合** |
| **B109** | vLLM V1 **基础调度器**的 KV block 生命周期 bug：`temperature=0` 同一 prompt **10/10 产生完全不同的输出**，且**在不开前缀缓存时也复现**（记者："points to a separate block lifecycle bug in the base scheduler's non-APC path"）；候选 TOCTOU 补丁 PR #37164 仍开着 | **与我 p15 测到的现象（BI=0 时 n=8 → unique 3/8）高度重合** —— 可能是同一现象的**上游正式拼写** |
| **B116** | **缓存复用本身改变确定性输出**：vLLM #54490/#54487（**2026-08-31 开，很新**）*"The minimal accepted prefix-cache configuration produces different text for two identical long-prompt requests, while the no-prefix baseline passes"* + *"The cache should preserve request semantics or reject the unsupported interaction"*；llama.cpp #28368 把同一问题延伸到 **logprob 位稳定性**（*"cache_prompt reuse changes computed logprobs on a plain (non-hybrid) transformer"*） | **我有现成仪器**（`probe_determinism.py` / `probe_repeat_once.py` / `compare_across.py`），且"同 prompt 两次、冷/热缓存对比"是**分钟级**实验 |
| **B107** | **EAGLE/MTP 的前缀缓存"末块丢弃"导致每次命中都要重算**（hybrid Qwen3.8 GDN 布局上 1,648 token/次）；姊妹 issue #51771 把"EAGLE/MTP block drop + prefix caching"记为 **untested** | **正是我三个方向的交集**（投机解码 × KV cache × 推理加速），且**我有 dflash2 与 eagle3** ⇒ 可在 dense Qwen3-4B 上直接测"开投机时一次缓存命中是否仍要重算" |
| **B114** | SGLang **radix cache + 确定性推理**路线图**两项未打勾**：*"FlashInfer Support"* 与 *"**Making Prefill with Radix Cache has the same output as Prefill without Radix cache**"*（issue 页面状态 COMPLETED 但复选框未勾 —— 记账不一致，能力问题仍在） | **我有 SGLang 0.5.19 + `--enable-deterministic-inference` 的现成 rig**（decision #117）；且我当时已撞到相邻限制：SGLang 的 `speculative_hook.py:770-776` 在 FlashInfer 下**直接抛错** |

**次一级（记下但排后）**：B21（空闲链表把命中块与未命中块交错 ⇒ 命中块先被逐出，PR #55998 只有单测、**无端到端命中率测量**）、
B22（cache-aware 准入排序，PR #54625；**前身机制的事后复盘只测到 +0.2%** ⇒ 门槛已知、偏低）、
B24（HBM `BlockPool` 无策略旋钮、请求优先级不影响 KV 保留；`gpu_eviction_policy` 不存在）、
B75（该抢占谁、选错要付多少）、B78（**KV 的 TTL/过期概念在两家引擎里都不存在**）、
B88/B103/B104（前缀缓存效果的**测量方法学**与 agent 负载下的可复现命中率）、B3（共享前缀短于 block 时命中恒为 0 —— 但属"找 bug"路线，见 decision #76 的告诫）。

### 2.3 合并后我**必须承认的一处自我矛盾**（比地图本身更重要）

我在 decision #117–#119 里把"批不变性"这条线索关掉，理由是：**开关有效 ⇒ 干预点就是开开关 ⇒ 不存在更省的等价方案**。
但这份 KV 地图显示：**那个"有效的开关"，在与"前缀缓存"组合时是"设计完成但未 ship"的**
（B108：跟踪 issue 把它列为 help-wanted 未打勾，实现 PR 等待评审逾两个月），
而**前缀缓存恰是 vLLM 0.29 的默认值**（我已亲验：`config/cache.py:138` `enable_prefix_caching: bool = True`，
且服务日志里有 `enable_prefix_caching=True`）。
⇒ **我此前"开关有效"的结论，是在"前缀缓存关闭"这个非默认条件下得到的**；在**默认配置**下它是否仍有效，
**上游自己把它标为未支持/未测**（B108），且**至少有三条独立的新报告说缓存复用会改变确定性输出**（B116，#54490/#54487，2026-08-31）。
**这不是"我已经做过 X"，而是"我做过 X 的那个条件不是默认条件"** —— 按 FILTER §3，这正是"差异轴是条件性"的形状。
**待我用零卡源码核 + 一次分钟级实验判定 B108 到底是"报错"还是"静默失效"**（见 §三 待筛表新增行）。

**段 D（硬件排除）与段 C（185 条被放弃）的价值**：它们不是候选，而是**墓碑**——避免我重复追已被硬件或已被他人否掉的路。
按 FILTER §0.2，我应把这两段的高频结论并进 `notes/OCCUPANCY_LEDGER.md`（待办）。

---

## 三、待过筛的格子（按 FILTER §0–§7）

| 格 | 类型 | 一句话问题 | 我的初判 |
|---|---|---|---|
| **C-DSD** | 被放弃/负结果 | 已 ship 的动态投机调度在 K=0 表下仍付 12–25% 税（"表的**存在**即触发） | **正在测**（p20，见 §四）：默认关闭 ⇒ 可用已有仪器构造 K=0 表做独立复现 |
| **B108** | 开放（help-wanted） | **默认的前缀缓存 + batch-invariant 模式**：跟踪 issue 把"Prefix caching support"列为未打勾，实现 PR 等待评审逾两月 | **新增头号待判**：我 p15/p16 全部在"前缀缓存关闭"下做的 ⇒ 这只差**一次分钟级实验 + 零卡源码核**就能判定是"报错"还是"静默失效" |
| **B116** | 开放（**2026-08-31 新开**） | **缓存复用本身改变确定性输出**（vLLM #54490/#54487；llama.cpp 侧连 logprob 都不稳） | **新增**：仪器现成（同 prompt 两次 / 冷热对比），分钟级可测 |
| **B109** | 开放（bug） | V1 **基础调度器** KV block 生命周期 bug：T=0 同 prompt **10/10 输出完全不同**，**不开前缀缓存也复现** | **新增**：与我 p15 的 `unique 3/8 @BI=0` 高度重合 ⇒ 可能是同一现象的上游正式拼写（需辨明是否同一根因） |
| **B107** | 开放（untested） | EAGLE/MTP 前缀缓存"末块丢弃"⇒ 每次命中仍要重算 1,648 token | **新增**：**三个方向的交集**，且我**已有 dflash2 + eagle3** ⇒ dense Qwen3-4B 上可直接测 |
| **B114** | 开放（路线图未勾） | SGLang radix cache + 确定性推理：*"Making Prefill with Radix Cache has the same output as Prefill without Radix cache"* **未打勾** | **新增**：SGLang 0.5.19 rig 现成；我此前已撞到相邻限制（FlashInfer 下 `speculative_hook.py:770-776` 直接抛错） |
| **C-NVFP4-KV** | 被放弃（零评论） | 同硬件 issue：NVFP4 KV cache 下投机 draft-extend/verify 必崩 | 硬件一致，但**需引入 NVFP4 轴**（我们全程 bf16 KV）⇒ 半径与收益待估 |
| **B-observability** | 开放 | 引擎不导出"调度刻意 K=0"与"未在投机"的区分 | 🔴 **不可立项**：他人正在做（PR #54748 / RFC #54749） |
| **B-draft_model** | 开放（能力） | 我们**没有**小 drafter ⇒ 三种投机方法同台对照从未在本机做过 | ⚠️ 属"能力补齐"而非"缺口"（EAGLE3 已下载 ⇒ 能力已补） |
| **A-grammar-lossless** | 已发表证明 | 语法掩码下投机不保持"语法条件分布" | ⚠️ 纯理论/正确性，**不是优化类**；且已发表 |

**淘汰（本轮）**：B-observability（他人正在做）；§2.2 压掉的三类（hybrid 模型族 / 多卡 / MLA）。
**待判（按"先便宜后贵"排序）**：**B108 → B116 → B114 → B109 → B107 → C-DSD → C-NVFP4-KV → B-draft_model**。
**排序理由**：B108/B116/B114 是**分钟级、零新能力、零新模型**就能判定的（仪器我全有），
而 C-DSD / C-NVFP4-KV 需要新配置或新硬件轴。**便宜的先杀掉** —— 这是 decision #118 教我的顺序（假设 → 廉价否证 → 不花 GPU）。

---

## 一·补：投机解码地图**补充材料**（394 行 / 430 个不同 URL）的要点与我核实的部分

### 1.7 一条直接约束本机的**硬事实**（OPEN）

**vLLM 无法组合 MTP + n-gram**（#46977，OPEN）：原话 *"Currently `--speculative-config` accepts a single method."* /
*"**ngram is effectively free (zero inference cost, no VRAM overhead) and excels precisely in those repetitive ranges,
but contributes nothing to base generation speed on its own.**"*
⇒ **在我们的 vLLM 0.29 上，dflash/MTP 与 NGRAM 每个引擎实例只能二选一**（本工作区历史实验也确实是分开做的）。
**未核实**：SGLang 侧是否同样互斥（待查）。

### 1.8 另一条已被明确拒绝的格子（跨引擎一致）

**vLLM 没有跨请求的全局 n-gram 提示查找缓存**（PR #44597，OPEN，有 merge 冲突）：*"The existing ngram proposer only scans
the current request context… **The default remains local.**"*；**llama.cpp 明确拒绝同一件事**（PR #26283：
*"We implement the per-request/prompt tree only."*）⇒ 跨引擎一致的"不做"。

### 1.9 🔴 **本文件至今最强的证据线：「投机解码常常在真实部署里输给不投机」**

**跨引擎、多来源、可引用的负结果（子代理材料，逐条有 URL）**：

| 来源 | 原话 / 数字 |
|---|---|
| llama.cpp PR #8648（**作者放弃自己的特性**） | *"it is **not worthwhile to invest more work into n-gram-based lookup decoding**"*；*"with Gemma 2 you actually get a **performance regression**"*；*"**~50 previous runs on an RTX 4090** to sufficiently populate the dynamic lookup cache in order to break even. After ~100 previous runs the speedup is ~10%."* |
| 某部署（2026-08-23，n-gram） | **113.00 → 96.42 tok/s**；*"Status: **CLOSED NEGATIVE** — keep the public recipe target-only"*；*"only 22 accepted tokens from 336 generated draft tokens … (6.55% reported acceptance)"* |
| llama.cpp #23533 | *"**MTP is ~21% slower than generating without speculation, despite 100% draft accuracy.** This is the opposite of the expected result."* |
| vLLM #15025（logits 蒸馏的 drafter） | *"acceptance rate is very high … Still I get **consistent performance drop ~30%** … **the best speed is achieved when using main model only without speculative.**"* |
| vLLM #16258 | *"regardless of the configuration … is **Pareto worse** than the inference without the n-gram model"* |

**⇒ 这批证据的策略含义（我的判断）**：投机解码的收益**不是接受率的函数**——存在"接受率 100% 却慢 21%"
（llama.cpp #23533）与"接受率很高却慢 30%"（vLLM #15025）的实测。⇒ 真问题不是"如何提高接受率"，
而是「**什么条件下这套机制整体不划算**」，而这正是我已有的仪器（**每步代价** `spec_step_ms`、
**逐位置接受率**、`Δln(tok/s) = Δln(接受) − Δln(每步代价)`，全部在 `results/p06-frontier/2026-09-13/A1_4k/mechanism.csv`）
**能直接量、而历史上一直被接受率掩盖的量**。

### 1.10 我**亲验**的新能力（本轮）

| 项 | 结果 |
|---|---|
| `eagle3` 是否为合法投机方法 | ✅ `config/speculative.py:69` 列出 `"eagle","eagle3","extract_hidden_states",MTP...,DFlash...` |
| 能否识别 EAGLE3 命名的 checkpoint | ✅ `config/speculative.py:1285` `elif "eagle3" in self.draft_model_config.model.lower()` |
| Qwen3-4B 的公开 EAGLE3 drafter 是否可取 | ✅ HF API 经镜像返回规范化 repo 名（302 → `...-full-context-epoch1-step30000`）⇒ repo 有效 |
| 我们的 `dflash2` 声明了什么 | `block_size: 8`，**无 `n_predict`**（与第 2 轮审计一致） |
| 磁盘 | 余 80 G ⇒ 足够 |

**⇒ 这是一条本工作区从未碰过的评测能力**：可把 **EAGLE3 小 drafter** 与现有 **dflash2（MTP 类）**、
**NGRAM** 做**同机同负载对照**（vLLM 0.29 官方文档的 `draft_model` 示例目标正是 Qwen3-4B）。

### 1.11 子代理对**自身 D 段（硬件排除）的更正**（值得记）

它加了 G6 "NOT hardware-ruled-out" 表，理由：**PR #24322 自己的基准机就是 "an RTX PRO 6000 96GB"**（与本机同类）；
Speculators 支持单卡训练；PARD 有 `-tp 1` 单卡命令；**Qwen3-4B 的 EAGLE3 drafter 已公开**；
一个可用的 0.5B drafter *"~2.5 h on a single 24 GB GPU, ~$3 of compute."*
⇒ D 段里的 128–320 H200-GPU·h 是**面向前沿模型**的数字，**对本机规模不适用**。

### 1.12 量化与投机的关系（被 G5 收窄）

**不是"量化必然破坏 drafter"**，而是**特定量化头/加载路径会静默失败**。反证（成功案例）：
*"Q8 DFlash2 works fine with a Q4 target — **the draft quant does not need to match the target**"*（3090 实测：
no-spec 33.16 → MTP2 48.32 → DFlash2 Q4 56.06 → **Q8 59.88** t/s）；Kimi-K2.5 W4A8 + EAGLE3：TPOT 42.73 → 27.41 ms（**−35.9%**）。
结构性成因（SGLang #38574）：*"The MTP head is not part of the HF model graph that the quantizer traces,
so it is neither quantized nor listed in `ignore`."*

---

## 一·补2：投机解码地图 **D 段（硬件排除，32 条）对我们这台机器的含义**

**子代理在 D 段开头就给了一条我该记住的前提**：*"**sm120 ≠ sm100**. `is_device_capability_family(100)` is FALSE on sm120
(120//10 = 12 ≠ 10). vLLM's `platforms/cuda.py` special-cases family(120) separately."*

### 1.13 单卡被硬件排除的（本机不可做，别再追）

| 类 | 条目 | 关键引文 |
|---|---|---|
| **量化 KV 路线** | NVFP4 KV cache（D1.3）、FP8 KV 作非存储用（D1.4）、NVFP4 CUTLASS for SM120（D1.6）、W8A8（D1.5） | *"nvfp4 KV crashes on consumer Blackwell (sm_120/121) on stock vLLM — `--kv-cache-dtype nvfp4` routes to the trtllm-gen FP4 FMHA, which has no build there"* |
| **量化 MoE 路线** | NVFP4 MoE（D1.1）、MXFP8 MoE 原生路径（D1.2）、MXFP8 grouped GEMM（D1.12） | *"The NVFP4 MoE backend selection code only checks for SM9.0 (Hopper) and SM10.x family … but not SM12.0"* |
| **DFlash/DSpark 的量化组合** | **D1.14（最硬）**：*"DFlash spec decode **cannot** compose with any of `fp8_e5m2`, `fp8_e4m3`, or `turboquant_4bit_nc` — it is locked to `bfloat16` KV cache."* / *"block-diffusion speculative decoding and 4-bit KV are currently mutually exclusive on exactly the memory-bound cards that want both."* | 对我们无实际损失：**我们全程 bf16 KV** |
| **自适应验证（Adaptive Verification）** | **D2.6**：sm120 上**启动即被拒**（需 `AttentionCGSupport.ALWAYS`，仅 SM100 后端报出） | 参考测量是 TP=8 × 8×B300 |
| **并行/多节点**（本机 1 卡，全部排除） | DSD + DP（D2.1）、MTP + PP>1（D2.2）、spec decode + dp-attention（D2.3）、draft-DP/TP-1 方案（D2.4，**均未 merge**）、spec decode + P/D 分离（D2.7）、SwiftSpec（D2.8，8×Hopper）、draft 协同训练（D2.9）、Cascade/EcoSpec（D2.10，4 卡 70B）等 | — |
| **drafter 训练** | D3：EAGLE-3 参考规模需 128–320 H200 GPU·h（D3.2）、生产配方 8×A100（D3.4）、SpecForge 双卡（D3.5） | ⚠️ 但 **D3.1**：*"trainable (within 1-2 days) and testable on **8x RTX 3090**"*；**G6 更正**说 0.5B drafter 约 2.5 h / 单张 24 G ⇒ **小规模 drafter 训练并非完全排除** |

### 1.14 单卡 **可做** 的正面证据（地图自己给的）

| # | 证据 |
|---|---|
| D5.2 | **MTP + FP8 KV + chunked prefill 的一个 bug 正是在 "1× RTX PRO 6000 Blackwell (sm_120)" 上被定位的** ⇒ "sm120 上做投机是**可能的**；被 gate 掉的是 FP4/FP8 **数据中心**内核" |
| D5.1 | 单卡 Blackwell 的 MTP n-sweep：*"MTP n=3 → 100 tok/s (27B) / 170 tok/s (35B MoE); MTP **n=5 → 125 tok/s, annotated '+4% over n=3, marginal'**"*，逐位置接受率 0.87/0.72/0.60（社区材料，非一手） |
| D5.3 | batch-invariance 验证已在 4090/4080/3060/4×A10G 上做过；PR #52522 的 E2E 测试写明 *"requires one CUDA GPU with at least 32 GB"* |
| G6 更正 | PR #24322 的基准机就是 **RTX PRO 6000 96GB**（与本机同类）；PARD 有 `-tp 1` 单卡命令；**Qwen3-4B 的 EAGLE3 drafter 已公开**（我们已下载并校验） |

### 1.15 一条与我第 10 轮亲验**直接吻合**的版本陷阱（D4 段）

> *"**MRV2 is default in 0.29.0**, but spec-decode features require it while other features force fallback to MRV1. … vLLM will still fall back to use MRV1 if any of these features are configured."*
> 以及 *"we are considering Model Runner V1 deprecated and are targeting v0.32 for its removal."*

**⇒ 这解释了我第 10 轮看到的全部现象**：0.29 里 V2 是默认；我也**亲验**过"配 dflash2 时 V1 会硬报错、V2 正常"（不是静默降级）。
**并且** 与我亲验的 `VLLM_USE_V2_MODEL_RUNNER`（默认 `None`=auto）一致。

### 1.16 一条与我们实验设计**直接相关**的空白（D4 末条）

> **DSD 自己的 merged benchmark 是 BS1-only 且关闭前缀缓存** —— PR #45953 的每条命令都带 `--no-enable-prefix-caching`，
> 且只报 base/EAGLE3 的 TPOT（`2.91 / 2.97 / 2.91` ms）。⇒ *"DSD at concurrency on this class is essentially uncharacterized."*

**⇒ 这是"仍开放"格子里**唯一**同时满足"我方有仪器、单卡可做、且**上游自己的基准没覆盖**"的** —— **我把它列为头号待筛格子**。

---

## 四、本轮**实测 + S3b**结果（2026-09-14）：两条最强化验线索**双双被占位**，但**拿到 4 个可引用数字**

**成本**：≈0.7 GPU·h（p20 三轮六臂网格 0.55 + p21 四臂探测 0.15；全部清理，末态显存 0 MiB）。
**派生物**：`results/p20-dsd/2026-09-14/summary.md`、`results/p21-prefix-hit/2026-09-14/summary.md`。
**raw**：`/root/ccfa_results/2026-09-14/{p20_dsd_v3,p21_hit}/`。

### 4.1 两条线索的终局

| 线索 | 我测到了什么 | S3b 判定 | 占位者（OPEN） |
|---|---|---|---|
| **C-DSD**（DSD 的 K=0 档代价） | 并发 8 + **前缀缓存开启**：K=3 **+48.5%**、K=1 **+11.2%**、**全 K=0 表 −31.7%**（相对不投机）；K=0 档的 ITL 17.4 ms ≈ 真投机 18.0 ms ≫ 不投机 11.9 ms | **🔴 不立项** | **PR #53426**（"Opt-in skip of the **K=0 draft sync forward**"，OPEN，标题即机制）、issue **#49548**（OPEN，含自建 `VLLM_DSD_K0_DIAG=1`）、issue **#48494**（OPEN）、PR **#47737**（OPEN） |
| **B107**（前缀缓存命中仍要重算） | **dflash2 前缀缓存完全不命中**：第 2 遍 `prompt_tokens_cached=0`、`prefix_cache_hits=0/32768`、每请求重算 **4096=100%**；对照 `nospec` 只重算 **16**（**256×**）；**eagle3 32 / ngram 16 正常** ⇒ **方法特异**；且**延迟一拍**（第 3 遍才命中 32,512/32,768） | **🔴 不立项** | issue **#47930**（OPEN，标题即我的结论）、PR **#47926**（OPEN/Draft，机制原文直指 dflash 需 target 隐藏状态建 context KV、*"MTP/EAGLE-style drafters … are unaffected"*）、PR **#54163**（OPEN，*"the whole context was recomputed on every reply"*）、issue **#54094**（OPEN，1.04M prompt zero reuse，环境栏正是 **RTX PRO 6000 Blackwell**） |

### 4.2 四个可直接引用的数字（无论是否立项，都已取证）

1. **投机在本条件（dense 4B / 4k prompt / 并发 8 / 前缀缓存开启）下净赚 +48.5%**（K=3）——与地图 §1.9 那批"投机常输给不投机"的报告**条件不同**（他们多为 185k 上下文 / MoE / 长 CoT）。
2. **K=0 档比"完全不投机"还慢 31.7%**，且 **ITL 与真投机档几乎相同（17.4 vs 18.0 ms）** ⇒ **K=0 没有回到普通解码路径**。
3. **`dflash2` 下前缀缓存命中率 = 0（第 2 遍），命中仍重算 100% 的 prompt**；同一 rig 上 `eagle3`/`ngram`/不投机**只重算 1–2 个 block**。
4. **缓存冷/热本身会改变 T=0 的输出**：同 prompt 冷 vs 热，文本在 `nospec` 7/8、`eagle3` 7/8、**`ngram` 5/8** 相同（即 1–3 个不同）；**冷 vs 热的 token_logprobs 四臂全部 8/8 不同**。⇒ 地图 B116 在本机**独立复现**。

### 4.3 一条对上游因果措辞的实测纠正（方法论产出）

issue #48494 写 *"**The presence of the batch-size table is the trigger**; K=0 tiers are not required."* —— 把因果归给**"表的存在"**。
但我的 **K 匹配对照**（A2 静态K=3 **无表** vs A6 表常量K=3，**K 完全相同、唯一变量是有没有表**）测得 **−0.51%（噪声 3.63%）** ⇒ **表的存在本身零代价**，真正花钱的是 **K=0 那一档**（−31.7%）。
⇒ 这正好是 `notes/FILTER_3AXIS.md` §3 要求的那种"**条件性差异轴**"意识：**"同一现象在哪个变量上"必须用匹配对照钉死，不能沿用上游的因果措辞。** 我据此仍**不立项**（现象与机制都已被 #53426 占位）。

### 4.4 仍未判的（按"先便宜后贵"）

`B108`（前缀缓存 + batch-invariant 模式未支持；实现 PR #46592 是他人 **OPEN** 提案 ⇒ 预计同样被占位）、
`B116`（缓存复用改变确定性输出 ⇒ **我已独立复现**，但属**正确性**而非优化类，且其修法与 #46592 的 canonical chunking 重叠）、
`B114`（SGLang radix cache + 确定性推理两项未打勾 ⇒ 亦属正确性）、
`B109`（基础调度器 block 生命周期 bug ⇒ 已有 OPEN issue 与候选补丁 PR #37164）。
**⇒ 当前判断：三方向里"本机可达"的开放格，几乎全部落在「已被 OPEN PR/issue 占位」或「正确性而非优化类」两类上。**

### 4.5 用分诊工具补做的一遍"防漏"扫描（`pipeline/tools/prescreen_map.py`）

**为什么要补**：§2.2 的初筛是**我肉眼**做的，而本工作区的历史教训恰恰是"漏掉一个已判死的格子又重做一遍"。
新工具把 FILTER §0 里可机械化的部分（**硬件可达性 / 模型族可达性 / 是否优化类 / 有无占位信号**）做成三段分桶。
对 KV 地图实跑：**301 个格子 → 153 个"可能候选" / 47 个"正确性类" / 101 个"够不着"**。
（工具自测当场抓到两个真 bug：段标题 `## B. OPEN` 被当成格子；`\bnon-determin\b` **永远匹配不到** "non-deterministic"。
这正是"仪器必须先自测"的又一例。）

**补扫的结果：我肉眼那一遍漏掉了一整簇** —— **主机/CPU 内存的 KV 卸载分层**。它们在**本机完全可达**
（208 核 + 充足 RAM/磁盘 + 单卡），而且**属性能优化类**：

| 格 | 一句话问题 | 为什么我先前漏了 |
|---|---|---|
| **B44** | HiCache 的 **host→device 回载坐在关键路径上**，没有与 forward 重叠 | 我把 "HiCache" 误分到"跨实例/多卡"里了 —— 其实 L2 host 层是**节点内**的 |
| **B41** | 分层卸载**把所有等待中的请求都提升**，反复冲刷主 DRAM 层 | 同上 |
| **B42** | 卸载层之间**没有背压检测** | 同上 |
| **B45** | HiCache 备份线程**遇到任何存储后端异常就死**，泄漏排队中的备份与宿主内存 | 被关键词 "storage" 误伤（其实是节点内磁盘） |
| **B48** | vLLM 卸载框架该统一到哪套**分层语义** | 同上 |
| **B52** | 全量 host 卸载**很快触顶**：瓶颈从稀疏度变成**传输量** | 同上 |
| **B53** | vLLM 文件系统二级层**没有可配置的磁盘预算**，容量语义未定 | 同上 |
| **B35** | vLLM 仍**没有**通用 INT8 KV 支持，KIVI 那类 INT4 也没 ship | 我把它和"FP8/NVFP4 KV 被硬件排除"**混为一谈**了；INT8 与 FP8 不是同一条路 |
| **B71** | **永不结束上下文的请求**（流式视频 / 1M token 会话）的 KV 内存归谁管 | 关键词里没有触发词 |
| **B106** | **没有人**按 agent 真实运行的时间尺度去测**反复压缩** | 被归进"测量类"就放下了 |

**⇒ 结论（对筛选清单的修订）**：`notes/FILTER_3AXIS.md` §0.2 的"已知死路清单"需要补一句 ——
**"被硬件排除的量化 KV（FP8/NVFP4）≠ 量化 KV 整体"**；且**"HiCache / 卸载"不能被当成多卡词**。
这两条都是我这轮肉眼筛掉、工具捞回来的。
**新的候选取向（在等第三张图期间并行推进）**：**主机内存 KV 卸载分层的性能** —— 它是三方向的交集
（KV cache 的存续策略 × 引擎执行代价里的传输/重叠 × 与投机解码的交互），**单卡可测**，且上游的
**merged** 修复（#46972，KV-offload × MTP/Eagle 的命中修复）说明这条路径**上游是认的**。
⚠️ **但已知一个必须先处理的混淆**：#47930 的 DFlash × 前缀缓存缺陷会让"带 dflash 的卸载命中率"读数失真
⇒ 该簇的**任何测量都必须先跑 `eagle3`/`nospec` 作对照**（这正是 p21 建立的方法）。

---

## 五、按 rev8 候选池 ①（C 段中"失败理由可能已过期"）对 **KV 地图 185 条被放弃格**的完整过筛（2026-09-14，0 GPU·h）

**为什么做这一步**：rev8 把候选池①（`notes/FILTER_3AXIS.md` §1a）定为**第一优先级**，而我此前一直没执行它
（附录 A.1 已承认这个偏差）。KV 地图的 C 段有 **185 条**带引文的放弃记录 —— 这是现成的最富矿，但**全文 0 个
`REASON-MAY-HAVE-EXPIRED` 标签**。本轮把它补上。

**方法（可复现）**：解析 185 条 → 机械旗标（`HW_OLD` 硬件代际 / `ENGINE_OLD` 引擎版本 / `SUPERSEDED` 被取代 /
`STALE` 陈旧关闭 / `NO_KERNEL` / `FUTURE` / `MEASURED_NEG`）→ 对命中的逐条**读原文理由**并施 §1a。
旗标计数：`STALE 67 / MEASURED_NEG 50 / FUTURE 37 / SUPERSEDED 22 / HW_OLD 5 / ENGINE_OLD 4`。

### 5.1 关键设计：**"closed in favour of X" 只有在 X 真的落地时才成立**

这是本轮最有价值的一条方法学。22 条放弃记录的理由是"另有 PR/设计在做它" —— **若那个 X 自己也没落地，放弃理由即失效、该方向重新变成无人认领。**
我据此**逐条查了 X 的真实状态**（HTML 直取，不经 API）：

| 被放弃的格子 | 放弃理由（"由 X 取代/重复"） | **X 的真实状态** | 结论 |
|---|---|---|---|
| C5 | closed in favor of **#40835** | **MERGED**（Triton INT4 per-token-head KV quant） | 理由成立 |
| C6 | "Please close this. See **#38479**" | **MERGED**（TurboQuant 2-bit KV） | 理由成立 |
| C12 | "closed this in **#50094**" | **MERGED**（CPUOffloadingSpec → SharedOffloadRegion） | 理由成立 |
| C75 | "duplicates **#41847**" | **MERGED**（HBA/HMA by default for connectors） | 理由成立 |
| **C44** | "Closing as a duplicate of **#41565**" | **#41565 仍 OPEN**（TurboQuant `_continuation_prefill` workspace 欠分配） | **理由失效** |
| **C109 / C110**（soft-pin / popular-insert 保护） | "Superseded by **#43302**" | **#43302 CLOSED 未 merge**（multi-storage replication） | **理由失效**：被放弃所"让位"的那个方案自己死了 |
| C16（39-commit Extensible KV cache 系列） | "Replaced by **#56492**" | **#56492 = Draft**（njhill） | 理由尚未兑现，但**在飞（有人在做）** |
| C26（SGLang Router 分层树） | "re-split into **#39108/#39109/#39110**" | 三者**均 OPEN** | 理由成立（工作只是搬了地方） |
| C19 | "duplicates **#55159**" | **#55159 OPEN** | 理由成立（占位者仍在办） |

### 5.2 但"理由失效"≠"可立项"：三条失效项**全部撞在同一个不可达的结构原因上**

C44/C109/C110 的失效看似是好消息，**但把 C109 的理由读全会发现真正的根因**（引文来自 #42948 线程上 @stecasta 的 block-pool 插桩）：

> *"the real bug is **131% pool overflow from the homogeneous `lcm=256` physical block layout vs DSv4-Flash's `{256, 64, 4, 8}`-block-size KV groups**.
> Under that overflow, some eviction every alloc cycle is mandatory"*

⇒ 那个"级联崩塌"（地图 **B112**：命中率 94.3% @bs1–2 → 1.0% @bs4 → 0.3% @bs8）**是异构 block size 的后果**：
逐层 page size 取最小公倍数 ⇒ 物理池溢出 131%。**我们目标是 dense Qwen3-4B，逐层 block size 相同 ⇒ lcm 就等于 block size ⇒ 不溢出 ⇒ 该崩塌在原理上不该出现。**
**这是 `prescreen_map.py` 的 `MODEL_PAT` 把 B112/C44/C109/C110 判为"够不着"的机制解释** —— 之前只是关键词，现在有因果。
而真正的结构性修法（heterogeneous per-page-size pool）正是 **B113 的 RFC #42082 + WIP #42374（93 commits，均 OPEN）** ⇒ 也已被人占位。

### 5.3 C141 被**一次零卡源码核对**否掉（最干净的一次否证）

C141 曾是本轮最强候选（"llm-d 的文件系统 KV 缓存因**最小 staging 缓冲大于实际所需**，4.4 GB 有用 KV 导致约 10 GB 被搬运 ⇒ ~2.3× 数据放大"），
且它的放弃理由（*"deprecated while our experiments were ongoing, in favour of the secondary filesystem disk tier introduced directly into vLLM"*）**确实已过期**：产物已上游进 vLLM。
**于是去读 vLLM 0.29 的实际实现**（`v1/kv_offload/tiering/fs/manager.py`）：块大小取自 `primary_kv_view.strides[0]`（:167），
偏移是 `int(bid) * self._block_size`（:226/:240），传输是**精确的** `view[offset : offset + block_size]`（`io.py:110/:146`）。
**⇒ 上游版本里没有"按最小值超量分配的 staging 缓冲"这个东西**；那个 2.3× 是 `llm-d==0.23`（final release）自身的实现细节，
**没有随代码一起上游**。⇒ **C141 死**（0 GPU·h）。

### 5.4 池① 的结论

| 类别 | 条数（本类中已核） | 处置 |
|---|---|---|
| 理由成立（后继 X 已 merge / 工作搬到仍 OPEN 的地方 / 实测负结果且边界对本机更紧） | C5, C6, C12, C75, C26, C19, C16, C125, C132, C137, C94, C118, C152, C163 | 丢弃（理由仍成立） |
| 作者自行撤回（无缺陷） | C103, C112 | 丢弃 |
| **理由失效但不可达**（异构 block size 的 lcm 溢出 ⇒ 需 hybrid 模型；结构性修法已被 RFC #42082/#42374 占位） | **C44, C109, C110** | 丢弃（换轴后仍不可达） |
| **理由失效且可达** | **C1**（"It is for **V0**… make a new PR for **V1** when the time comes!"） | ⚠️ 唯一残留：但其残余范围（**HBM block pool 的驱逐策略**）正是 **B24** —— `gpu_eviction_policy` 不存在、issue #40268 `REOPENED`、维护者 njhill 已回复指向 #40004 等 ⇒ **有活跃请求，占位中** |
| **理由失效但缺陷未随代码上游** | **C141** | **死**（§5.3 源码核对） |

**⇒ 池① 对 KV 地图：幸存者 0 条。** 这一方面是坏消息（少了一整条来源），
另一方面它**证明了 rev8 的优先级排序本身是对的**：我此前从 B 段挑的四条全部撞墙，而池①虽然也没出候选，
但**每一条都在 0 GPU·h 内被判死且给出了机制级理由**（而不是"测得不对"）。
**⚠️ 一处未取证**：C16 的后继 #56492 只能确认是 **Draft**（HTML 状态字段未渲染出 `state`，我从 `Status: Draft` 徽章判定）——
若它其实已 merge，则 C16 归入"理由成立"。这一条标**未取证**，不据以下结论。

---

## 六、按 rev8 候选池 ① 对 **投机解码地图 C/G 段**的过筛（2026-09-14/15，0 GPU·h）

**为什么做**：§五 把池① 施于 KV 地图（185 条 → 幸存 0）。投机地图是另一半 —— 它的 C 段（C1–C9）+ G 段（跨引擎放弃与负结果尾部）
是同一类资产，且**没有任何人在做它的 E 段**（池②）。本轮补池①。

**结果：池① 对投机地图同样幸存 0 条。** 两个最有希望的候选都被 S3b 在**零 GPU** 拦下：

### 6.1 候选一：**SGLang jump-forward 解码被移除、且从未给出失败理由**（池① 形状最"干净"的一个）

**为什么它看起来是好候选**（引文全部来自 C6.4）：
- 维护者 zhyncs 的原话是 *"Remove jump forward to **simplify the code maintenance**"*，**不是技术失败**；
- 而下游复核者 yhay81 明确写道：*"There is **no documented model-quality or regex-correctness failure** that caused the removal. … I could not find a stronger public claim such as 'it was slower' or 'it produced incorrect regex output.'"*
- 维护者当年承诺 *"Maybe Jump forward will be implemented using speculative decoding **later on** by @hnyls2002"* —— **从未发生**；
- **Discussion #32352（2026-07）仍在问为什么，无人回答**；
- 且 vLLM 与 SGLang 现在**都不支持** `compute_ff_tokens`。
⇒ 这符合 FILTER §1a（移除理由**不是**技术性的 ⇒ 无约束可"过期"，但也无理由阻止重做）+ 落在 rev8 明列的**"采样与结构化输出开销"**范围内。

**S3b 判定：🔴 已被占位（我此前没查到，地图也没记）**
- **vLLM PR [#47885](https://github.com/vllm-project/vllm/pull/47885)（OPEN）**：*"feat(spec_decode): **Grammar-aware draft token sampling for structured outputs**"*（jmamou）
  ⇒ 结构化输出 × 投机解码这条轴**正在被做**。
- vLLM PR [#15490](https://github.com/vllm-project/vllm/pull/15490) *"[V1][Experimental] Jump-forward decoding"*（aarnphm）—— 实验性提案，状态渲染为 Closed/Open 混合（**未取证**，不据以下结论）。
- 第三方 `mudler/vllm.cpp` 已把它作为 `VT_ENABLE_JUMP_FORWARD`（默认 off）以 *"SGLang parity SW3"* 实现 ⇒ **连第三方复刻都有了**。
⇒ **杀**。这条同时也是"**地图没有记的占位者**"的又一例（rev8 §0.5 的实证从 1 例变成 3 例：KV 的 #50045/#49952/#34519，加上这里的 #47885）。

### 6.2 候选二：**投机解码消耗的 KV 容量是否付得起自己**（"draft-KV 税"）

**为什么它看起来是好候选**：地图 C3.13（vLLM #41559，**CLOSED COMPLETED、零评论**）里上游给出的**量化理由**是
> *"The spec decode throughput gains from DFlash **do not justify halving the KV pool** for long-context workloads."*
—— 即 upstream 自己把"投机收益 vs KV 池损失"当成一个**权衡**在算。而**我手里刚有这条权衡的一个实测点**：
p20 实测同一 rig 上投机臂的 KV 容量是 **246,794 token**、不投机是 **319,328 token**（**−22.7%**，draft 模型占显存所致），
而暖态吞吐是 **+48.5%** ⇒ 在 24×4k prompt 这个负载上收益压过了容量损失。**问题是这个不等式在什么条件下翻转**，而且它是**三方向的交集**（引擎执行代价 × 投机解码 × KV Cache）。

**S3b 判定：🔴 已被占位**
- **论文 "Windowed-MTP: **Removing the Full-Context Draft-KV Tax** at Million-Token Context"**（HF papers `2607.21535`，
  **已从 HF 论文页读回标题**；摘要正文未取到 ⇒ 标**部分未取证**）⇒ 这条轴**已有专门论文**，且题目用的词就是 "Draft-KV Tax"。
- **vLLM issue [#54691](https://github.com/vllm-project/vllm/issues/54691)（OPEN）**：*"[Bug][Spec Decode]: DFlash is a **net loss at long context (~185k)** … drafter re-scans full accumulated KV every cycle; **no per-sequence-length disable hook**"*
- 同簇文献：LongSpec、Vegas（ICML 2026）、Dustin（ICML 2026）、Nightjar —— 长上下文投机解码是一个**活跃子领域**。
⇒ **杀**。

### 6.3 池① 的全景结论（两张地图都已跑完）

| 地图 | 池① 条数（带引文的放弃记录） | 幸存 | 最高价值的一次否证 |
|---|---|---|---|
| KV Cache | 185 | **0** | **C141**：放弃理由（llm-d 弃用）确已过期，但读 vLLM 0.29 的 `tiering/fs/manager.py` 发现**超量 staging 缓冲根本没随代码上游** ⇒ 0 GPU·h 判死 |
| 投机解码 | C 段 C1–C9 + G 段（约 60 条带引文/负结果） | **0** | **jump-forward**：移除理由非技术性、维护者承诺未兑现、下游提问无人答、**看起来完美**——但 #47885 OPEN 正在做同一件事 |

**⇒ 两张既有地图的池① 均已耗尽，幸存者 0。** 按 rev8，剩下的唯一来源是**池②（从未被讨论过）**：
KV 线（子代理 `22470aaa` 在跑）、引擎执行代价线（子代理 `bac75c1d` 的 E 段在跑）、
**投机线目前无人做** —— 下一步补上。
