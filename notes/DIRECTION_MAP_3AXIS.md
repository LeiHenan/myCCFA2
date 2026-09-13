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
