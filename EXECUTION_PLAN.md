# 执行方案 — MLSys 选题

**版本**：v1.0 ｜ **日期**：2026-09-11 ｜ **状态**：W1 执行中
**本文档是唯一权威执行依据。** 上游四份文件（`docs/plan.md`、`docs/kimi_plan.md`、`docs/V41.md`、`docs/GPT.md`）保留为论证与证据记录，其**排序结论不再作为执行依据**；纪律条款见 `docs/old.md`。

---

## 0. 一句话

**主线 1 条**：#6 投机算力分配（Speculative Compute Allocation）。
**备线 2 条**：`#12`（MoE EP 对冲；**先过 ≤3 天残余不均衡判定**，再过才做 2 周）、`#7`（per-adapter KV 配额，待离散度探针判定）。
**暂缓 1 项**：`#4` —— 复核门**不通过**（见 §6 与 `notes/gap_hybrid_state.md`），带重开触发条件。
**结构风险（09-11 记录）**：能作为"主线候选"的**只有 `#6` 一条**；若 T1 判定共线而死亡，portfolio 内**没有第二个有强先验的主线**（`#12` 已冻结待 StreamEP、`#7` 仅算术量级、`#1` 待 E1、`#3` 先验 ≤6%、`#4` 暂缓）⇒ 见 §12 待决策项 **D1/D3**。
**主线定位收紧**：`#6` 只能是 **drafter 网络自身的层数/宽度**弹性（draft 长度与 verify 预算的自适应已 ship，见 `notes/p06-family-codecheck.md`）。
**判定探针 2 个**：#1 E1（1 天）、#3 约束解码扫描（1 周）。
**其余 14 项归档**（§9），其中 #2 的归档理由不是"被占"而是**基线反证**（见 §9）。

**当前阶段**：⏸ **停工等卡**（见 [`README.md`](README.md) 顶部的**解冻顺序**：先 ② pin + 工具链核对 → 再 ① 租卡 → 再 ③ smoke test）。

**命名法**：`#N` = 候选方向；`probes/pNN-<slug>/` = 第 NN 号方向的实验；`probes/aux-cN-<slug>/` = 辅助探针（与候选无关）。完整词表见 [`docs/GLOSSARY.md`](docs/GLOSSARY.md)。

---

## 1. 入选门槛（三条，逐条可打勾）

| # | 门槛 | 判据来源 |
|---|---|---|
| **①** | **机制格未被占**：没有 MERGED PR，且同一决策上没有 ≥2 个并行开放提案 | plan.md §0.2「方向 B」 |
| **②** | **≤2 卡 / ≤2 周能拿到可信数字** | plan.md §5.1「先判定、后投入」 |
| **③** | **量级来自实测或引擎常量**，不是算术、不是拼接 | old.md「47 条击杀里 38 条越界」 |

**适用规则（修正版，必须按此执行）**

- ① ② **对所有条目必过**。**② 的卡数按"判定阶段"计**：主线与判定探针必须在 ≤2 卡内拿到可信数字；备线/对冲若**判定阶段**本身需要更多卡（如 `#12` 的残余不均衡判定与 4–8 卡 EP 实验），必须在 §7 之类位置写明**卡数、门槛与"先判定、后上卡"的次序**，否则不得进入清单。
- ③ **对主线与备线必过**；
- **探针可零量级入场**，但必须同时带：≤2 周换成实测的路径 + 预登记杀判据（`notes/prereg/`）；
- 任何"量级不足 / 量级足够"的判定**必须自测**；算术值与拼接值只能用于**设计实验**，不能用于**判定方向**。

> 此规则是为了避免上一轮的缺陷：`kimi_plan.md` 声明了排序键却生成不出表格。**规则必须能推出清单，否则改规则。**

---

## 2. 存活清单

| 定位 | 方案 | 机制格状态 | 首个可信数字 | 量级来源 | 预登记判据 |
|---|---|---|---|---|---|
| **主线** `p06` | **#6 投机算力分配** | 空（DSpark 只做离线静态层数；SGLang 已 ship 的是 *verify* 预算调度，不是 drafter 算力） | 1–2 卡 / 2 周 | **实测**（DFlash DT=4 @185k：16 vs 71 tok/s，`#54691`） | §5 |
| **暂缓** `p04` | **#4 收窄版** | **复核门不通过**：共存半场由 vLLM `#50172`（OPEN，交付物=正确性）在办；策略/预算半场已由 **ROCm ATOM 实装**（ladder + 需求驱动 rung + LRU + 成本记账 + 每请求 `1 + num_speculative_tokens` 回滚 slot） | — | — | §6（重开触发） |
| **备线 A** `p12` | **#12 MoE EP 不均衡 + dispatch** | **已被占**：dispatch 半边由 SGLang+NVIDIA **LPLB**（2026-06-26，2 节点）ship；padding 半边由 **DA-MoE** 等占 | 2 周（需多卡 EP；**先过 ≤3 天判定**） | **投影 / 下界 / 极端尾巴**（3.28× 系读图投影、19% 是均衡路由下的下界、80× 仅 Maverick top-1） | §7（先判定、后上卡） |
| **备线 B** `p07`（待判定） | **#7 per-adapter KV 配额/准入** | 空（两引擎均无 KV 配额/分区；`#2929` 的 21 项全是适配器权重/算子/API） | ≤1 周 / 1 卡 | 算术 1.2–2×（**③ 未过，待换实测**） | §8.4（离散度 <5pp ⇒ 归档） |
| 探针 `p01` | **#1 投机 KV 分级 state（E1）** | 空 | **1 天** | 无（E1 即补） | §8.1 |
| 探针 `p03` | **#3 约束解码 dynamic × complex × concurrent** | 三交集空，邻域快速填充 | 1 周 / 不改代码 | 实测（+5.3% → +802.9%），先验字段级 ≤6% | §8.2 |
| 填缝探针 `aux-c1/c2` | C1 prefill CUDA-graph padding ／ C2 FlashInfer radix 被静默关闭 | 空 | 各 1 周 / 1 卡 | 代码事实 | §8.3 |

---

## 3. 已关闭的证据缺口

### 3.1 `arXiv:2606.29223` — **不是 #1 的藏身处（已关闭，2026-09-11）**

- **真实身份**：**DEX — Depth Exploration for LLM Decoding**（Weisi Yang / Zipeng Sun / Stephen Xia；arXiv:2606.29223v1 [cs.LG]，2026-06-28）。全文已读。
- **它做什么**：在**目标模型层深轴**上，把"选单一 exit depth"替换为"**并行探索多个候选深度**"，用 final-depth 参考验证，commit 时 **collapse 探索格**只保留可复用分支；形式化 EAD（Earliest Available Depth）与 resolution Δ(𝒳)，给出 selection / exploration overhead 与深度侧加速的界；对 speculative 与 distributed decoding 基线报告竞争性吞吐。
- **对 #1 的结论**：**不占**。全文没有"commit 概率 → KV 内存层级"的映射；"commit position" 是**解码提交位置**，"reusable branch states" 是**深度探索器的分支状态**，不是 KV tier。
- **对 #6 的副作用**：它是 **#6 的邻近先例**（"把算力分配到不同深度"这一叙述已存在）⇒ 必须执行 §5.3 的 **DEX 替换测试**。
- 引用时注明：本结论基于 arXiv HTML 全文（v1）。

---

## 4. W1–W3 执行清单

| 时间 | 动作 | 预登记判据 | 产出 |
|---|---|---|---|
| **D1 上午** | ~~关闭 `2606.29223`~~ ✅ 已完成（§3.1） | — | 结论存档 |
| **D1** | **#1 E1**：instrument `vllm serve` + EAGLE-3，逐步统计未提交草稿 KV / 已提交 KV，扫 ctx × batch × γ{3,5,7} + 一个 draft-tree 配置 | 全部 HBM 可行点上 p95 比值 **<5% ⇒ 按实测定死** | 比值表 + 生死结论 |
| **W1** | **#3 扫描**（batch 1→256 × 4 语法类 × spec on/off，不改代码；语法类定义以 `notes/prereg/p03-grammar-scan.md` 为唯一来源） | class-4 @batch≥32 **>0.85×** **或**瓶颈非结构性 ⇒ 归档（Go 的补集） | 语法类成本曲线 |
| **W1** | **#6 可行性前置**：确认可跑的**多层并行 drafter**、per-step draft 成本仪器；写 DEX 替换测试 | 若只有 EAGLE-3 可用 ⇒ #6 降级（深度旋钮不存在，§5.1） | 家族确认 + 测试段落 |
| **W2** | **#6 frontier map**：`bs × ctx × depth × width`，对最佳固定配置 | 无内点最优 / 无 (bs,ctx) 配置反转 / <8% ⇒ 杀 | frontier 图 |
| ~~W2~~ **已完成（09-11）** | **#4 复核门** | 判据：gap 是否是已占证据的子集 ⇒ **是（不通过）** | `notes/gap_hybrid_state.md` |
| **W1**（提前完成） | **#6 代码层家族检查**（无卡） | 多层 drafter 是否存在且可截断 ⇒ **是**；定位收紧为"drafter 网络深度/宽度" | `notes/p06-family-codecheck.md` |
| **≤2 周** | **#12** 对冲：区分 DA-MoE，或推高"体积→时间"转化率 | 到期不达标 ⇒ 归档 | 差异化结论 |
| **W3** | 收敛：**1 主线 + 1 备线** | 量级 ≥8%（自测 vs 部署基线）且机制可归因 | 主线锁定 |

---

## 5. 主线 #6 — 投机算力分配

### 5.1 家族事实（先读，决定实验是否成立）

| drafter 家族 | 层数 | 深度旋钮是否存在 |
|---|---|---|
| **EAGLE-3** | **1**（DSpark 配置："we set 1 for Eagle3"） | **不存在** ⇒ 只能作基线；其树宽/树深旋钮已被 AdaEDL/SpecDec++/Pacer/SVIP/DDD/ECHO/Graft 占满 |
| **DFlash** | 5 | 存在 |
| **DSpark** | 生产骨干 **3** 个 MoE 层；"a **2-layer** DSpark outperforms the **5-layer** DFlash baseline" | 存在 |
| MTP head（Qwen3-Next 等） | 单头 | 存在但粒度粗 |

⇒ **frontier 必须跑在多层的并行 drafter 上（DFlash / DSpark 类）**；只有 EAGLE-3 可用时，#6 直接降级。

**② 工具链核对已完成（2026-09-12）**：pin = vLLM main `9a35c08`；注册表含 `DFlashDraftModel`(qwen3_dflash) / `DFlash2DraftModel`(qwen3_dflash2)，`qwen3_dspark.py` 存在；HF 可用权重含 **`Qwen3-4B-speculator.dflash2`**（⇒ 24 GB 可跑）；**深度由 config 的 `num_hidden_layers` 驱动**（training 侧 `--num-layers`）⇒ T0 截断是改配置而非改代码。详见 `notes/p06-toolchain-check.md`。

**多层 drafter 的取数路径（W1 先看这里）**：`vllm-project/speculators`（DSpark/DFlash 上游实现）+ HF 权重（如 `mgoin/GLM-5.2-speculator.dspark-block16`）。有它，W2 的租卡才有意义；没有它，`#6` 因"只有 EAGLE-3"直接降级。

### 5.2 实验设计

- **定位**：`Speculative Compute Allocation`（**内部叫法、非既有术语**，见 §12 D4）— 每步决定给这一轮投机投多少算力；`depth × width × draft length × verify budget` 是同一预算的不同形态。
- **网格**：`depth × width × **γ{1,3,5,7}** × batch{1,8,32} × ctx{4k,32k,128k,185k}`，spec on/off 对照。**γ 因子不可省**——正交性检验必须同扫 depth 与 length（至少在 `width=默认树` 子集上）。
- **指标**：tok/s、per-step draft 成本、平均接受长度、相对**最佳固定配置**的端到端比。
- **Go（主假设：drafter capacity 与 draft length 正交）**：存在 (bs,ctx) 区域使最优 (depth, γ) 发生**非共线反转** **且** 存在内点最优 **且** 相对最佳固定配置 ≥8% **且** **四个必答项**全部通过：① 与已 ship 控制器（`adaptive_spec_params`，调 `speculative_num_steps`）的差异；② **为什么不能把 vLLM 的 per-batch K 查表（`num_speculative_tokens_per_batch_size`）扩到深度轴** —— 本线跑在 vLLM，这才是审稿人会问的那一问；SGLang 拒绝多层 worker（`enable_multi_layer_eagle=True is not supported`）只是**旁证**；须给**结构性**理由，否则退化为 plumbing；③ 每个 (bs,ctx) 格做 **adaptive-steps-only vs adaptive-steps+depth** 增量对照；④ **动机抗辩**：为什么在 PRISM（容量⟂成本的架构解耦，MLSys '26 oral）之后仍需要运行时分配——落点必须是 PRISM 不处理的**上下文相关成本**（如 185k 全上下文重扫）。
- **No-Go**：收益只出现在 185k 角落；或与已 ship 的置信度/成本表调度器无法区分；或 **`optimal depth ≈ f(optimal γ)`（共线）⇒ 退化为"另一个 adaptive draft length 实现" ⇒ 杀或并入 `#7`/`#12`**。
- **三级加速判定（决策级 ≠ 论文级）**：T0 旋钮验证（半天）→ **T1 反转探针**（`depth{1,3,5} × γ{1,3,7} × ctx{4k,128k} × bs{1}` ≈18 格，半天–1 天）→ T2 增量对照（1–2 天）。**T1 判主假设存废、T2 出最终 go/no-go；通过后才需要 2 周全网格**。注意 T1 的**不对称性**（截断只给下界）。详见 `notes/prereg/p06-frontier.md`。
- **基线纪律（与 §12 D1 一致）**：T1/T2 **固定跑 vLLM**，基线 = ① spec off（AR）② **最佳固定配置** ③ DSpark 生产调度器（置信度头 + STS + 离线 SPS 成本表 + ragged-verify）④ **vLLM `num_speculative_tokens_per_batch_size`**（per-batch K 查表）。SGLang `adaptive_spec_params` **逐字拒绝多层 worker**，不能充当 T2 对照（仅可作静态基线）。
- **两个"8%"不是同一个比较**：§5 决策规则 2 的"<8% 即止"是相对**部署中实际在跑的配置**；本条 Go 判据的"≥8%"是相对**最佳固定配置**（更严）。两者需同时成立才 Go。

### 5.3 DEX 替换测试（书面交付物，W1 完成）

回答一个问题：**"把这个机制读成『DEX 换成一个独立 drafter』，它还成立吗？"**

- 若答案是"能"⇒ delta 只落在模型替换上 ⇒ **改措辞或改方向**。
- 必须写清的区分轴：① 独立 drafter 网络深度（非目标模型 early-exit / self-speculation）；② **调度器级、跨请求**的分配（非 per-commit-position）；③ 与验证预算耦合。

**差异轴与判死条件已预登记**：`notes/p06-dex-differentiation.md`（2026-09-11，**数据采集前**）——三条区分轴（对象=独立 drafter 网络深度 / 粒度=调度器级跨请求 / 预算=与 verify 耦合）+ 三条判死条件。

---

## 6. 暂缓 — #4 收窄版（复核门：**已执行，不通过**，2026-09-11）

**已被占的（不得重做）**

| 证据 | 状态 | 内容 |
|---|---|---|
| vLLM `#53614` | **MERGED** 2026-09-06 | Kimi-K3 checkpoint 扩展到 spec decoding / EAGLE block rewind + partial prefix caching；"re-keyed from its **provisional boundary**" |
| vLLM `#50172` | OPEN | GDN `mamba_cache_mode="all"` × MTP，"declares **functional correctness** as the deliverable" |
| vLLM `#55760` | MERGED 2026-09-08 | Mamba + EAGLE 的 `prefix_cache_retention_interval` 默认值修复 |
| vLLM `#56142` | OPEN 2026-09-09 | hybrid shared-prefix checkpoint 边界修复 |
| vLLM RFC `#55697` + `#55873`/`#55875`/`#55876` | OPEN 2026-09-09 | 应用声明式 checkpoint 边界 + 调度状态机 + 批量 GDN kernel（L40S 7.58×） |
| SGLang `#30393` | MERGED | draft-cache packed/sidecar HiCache 集成 |

**门（2 小时，W2 执行）**：gap 声明必须**不是**"checkpoint 边界 / 状态提交语义"（已 MERGED/OPEN），只能是：

> **树验证下多分支 SSM 快照的内存放大 + checkpoint 淘汰策略 + 跨请求/agent 复用。**

**判定（2026-09-11 已执行）**：**不通过** —— 主张的 gap 是上表的子集（"内存放大"由 ATOM 的 `1 + num_speculative_tokens` 回滚 slot 与投机深度的耦合覆盖；"淘汰策略"由 ATOM 的 ladder + LRU + `checkpoints_dropped` 覆盖），唯一剩余空白（投机激活下 forking 类的跨请求 checkpoint 复用）属引擎原生 PR 领域且由 `#50172` 在办。

**重开触发**：① `#50172`/`#54637` closed 未合并或 >4 周无活动；② 其落地后暴露结构性缺陷（非调参可解）。详见 `notes/gap_hybrid_state.md`。

---

## 7. 备线 A — #12（先判定、后上卡）

- 定位：**非投机**推理系统对冲（非算术解、问题被生产验证）。
- **量级证据更正（2026-09-11）**：引用链里的"最硬数字"实为三类不同强度的证据 —— ① `3.28×` 归一层延迟是**读图得到的投影**（`archive/subfield_scan/rea1/k3/FORENSIC_REPORT.md`：IF 轴是 swept/constructed 参数，原文未说明数据如何产生，"Treat as a PROJECTION, not a production measurement"）；② `19%` all-to-all 是硬件实测，但路由被构造成**完全均衡** ⇒ 只是**下界**；③ `80×` 是 **Llama-4-Maverick（128 专家 top-1）的极端尾巴**（DeepSeek-V3 10–20×、Qwen3 ≲10；"所有前沿 MoE 都严重偏斜"不成立）。
- **最直接的竞争者（SOSP '26，正文待发表）**：**StreamEP: Straggler-Tolerant MoE Decoding without Communication Barriers**（USC & SNU & Google DeepMind & UT Austin）—— 标题即本方向的处境，**判定前第一优先要读**。
- **机制已被占（五重以上）**：① **DA-MoE** `2607.23099` **正文已读**——计算侧的"路由偏斜 → tile padding → 内核选择"已占，带实测 1.16×/1.29× geomean（**注**：`docs/kimi_plan.md` 称其"合入 FlashInfer"在正文中无依据，未核实）；② **SOSP '26 MorphKernel**（跨 SM 融合，明确针对 MoE expert activation，1.3×）；③ MLSys '26 三篇（CRAFT / Layered Prefill / MoE Serving Tax）；④ **SOSP '26 Barrier-Free EP**（Tier B，待读）；⑤ dispatch 半边由 **SGLang + NVIDIA 的 LPLB**（[2026-06-26 博客](https://lmsys.org/blog/2026-06-26-waterfill-lplb/)，per-layer dispatch LP，**2 个 Hopper 节点**，+0.84%–7.34%）ship；padding 半边由 **DA-MoE** 等占。
- **门（≤3 天，不上 4–8 卡）**：① 若 SOSP '26 **StreamEP** 正文已公开（真名；社区清单的 "Barrier-Free Expert Parallelism" 为**误标**），先读它确认是否已覆盖你的半边；**若届时仍不可得（SOSP 会后才有）⇒ 直接执行第 ② 步并在结论中标注该不确定性**，不得因"读不到"而停摆；② 在你自己的 model + workload 上测 **EPLB/LPLB 之后残余的 per-rank 不均衡**：残余 <8% ⇒ **归档**；≥8% 且可归因 ⇒ 才进入 2 周实验。
- 必须做到其一：**与 DA-MoE 正面区分的机制**，或把"体积降 20% → 时间降 9%"的转化率推上去。
- 到期未达标 ⇒ **归档，不续期**（避免对冲仓变沉没成本）。
- 注意：需多卡 EP，成本高于表格标注。

---

## 8. 探针

### 8.1 #1 E1（1 天）

- **测什么**：逐步的 `未提交草稿 KV 字节 / 已提交 KV 字节`（p50 与 p95）。
- **网格**：ctx × batch × γ{3,5,7} + 一个 draft-tree 配置。**硬件分支（预登记）**：24 GB 卡 ⇒ ctx 改 `{4k,16k,32k}`（或换 ≤4B target / 2×24 GB TP=2）；80 GB ⇒ `{4k,32k,128k}`。
- **判据三档**：全部可行点 p95 **<5% ⇒ 定死归档**；某可行点 **≥10% ⇒ 进入机制设计**；**5% ≤ p95 < 10% ⇒ 灰区，不结论**（补测释放延迟与峰值占用后再判）。细则与 D3 支线见 `notes/prereg/p01-e1-uncommitted-kv.md`。
- **为什么值得做**：这是全组合里**唯一能产出有效 V2 型击杀**的探针——1 天成本买的不是方案，是一个可信的否定权。且该比值**全库无人测过**（`docs/evidence/r4_graded_kv_tiers.md` 第 9 条）。
- **已知风险**：SpecMemo 的高精度约束；相对已 ship 二值处理只宽一步。

### 8.2 #3 约束解码扫描（1 周，不改代码）

- **网格**：batch{1,8,32,128,256} × 4 语法类 × spec on/off。**语法类定义以 `notes/prereg/p03-grammar-scan.md` 为唯一来源**（class-1 JSON schema／class-2 嵌套 JSON+regex／class-3 C++ 子集／class-4 Python 变体-Bython 类），本文件不再另列以免漂移。
- **Go**：class-4 @batch≥32 **≤0.85×** 无约束 **且** 瓶颈是结构性的。
- **先验（必须写进结论）**：`docs/evidence/constrained_decoding.md` T5 的字段级端到端余量 **≤6%**，"entering now means arriving after the result"。
- **定位**：低概率彩票，**不是主线**。

### 8.3 填缝探针（各 1 周 / 1 卡）

- **C1** prefill CUDA-graph padding：记录每次 replay 的 `padded/real`（`_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2`）。
- **C2** FlashInfer radix：Blackwell 上 FlashInfer 是默认后端，而 `attention_hook.py:570` 对其**静默** `disable_radix_cache=True`。

---

### 8.4 #7 per-adapter 离散度探针（≤1 周）

- **为什么先跑**：机制格空，但量级是算术值（1.2–2× goodput）⇒ **门槛 ③ 未过**，先换成实测。
- **测什么**：N ≥ 8 个 adapter（冷热两端）并发，同一 prompt 池；记 per-adapter prefix 命中率、`p90 − p10` 离散度、KV 占用分布。
- **判据**：**离散度 < 5 个百分点 ⇒ 归档 #7**；≥ 5 且存在"高占用低命中"adapter ⇒ 升为备线 B。
- 详见 `probes/p07-adapter-dispersion/README.md` 与 `notes/prereg/p07-adapter-dispersion.md`。

---

## 9. 归档清单（14 项）与理由

> 20 个候选 = **14 归档 + 1 暂缓（`#4`）+ 5 存活**（`#1 #3 #6 #7 #12`）。

| 方案 | 归档理由 |
|---|---|
| **#2 多租户保留准入** | 占位（Continuum `2511.02230`、SAGA `2605.00528`）+ **真正的杀手是基线反证**：本地负面结果逐字 "TTL-300s produced **byte-identical** results to LRU-leaf in every single run"、"**LRU-leaf is a stronger baseline than the literature treats it as**"；`rea_4_legacy.md` 自给实际 band 10–30%（非 89%）。仅剩"计费/记账耦合"⇒ 经济学，非本会场。 |
| #5 跨并发分支准入 | 量级是 **O8 估计、零实测**（流传的"2–3%"是 SpecTool 工具投机的数字，不是本机制）；准入语义已被 vllm-omni COW/fork 占 |
| #8 长 CoT 检查点 | 机制偏薄（checkpoint 格式 + 恢复路径），需搭调度策略才够一篇 |
| #9 dLLM 去同步 | 薄机制 + 赛道升温最快，只剩跨请求去同步 |
| #10 recompute-vs-load | `PrefixPlace 2608.01655` 已把 "recomputation or replica fetches" 写进 placement 目标函数 ⇒ "recompute 无人显式调度"不成立 |
| #11 branch group 可撤销 | 依附"分支类 workload 是否真实"，需 trace 佐证 |
| #13 elastic 投机 KV 预留 | 需 8×H200 |
| #14 片上 KV 预取 | Levy `2603.13430` §5.3 已发表失败记录；权重流量 58% > KV 32.7% |
| #15 每位置 attention 预算 | 逐位置稀疏成本中性的理论证明压着 |
| #16 投机不确定度 → KV 精度 | 算术天花板 ~1.2%；被拒 token 的 KV 本就丢弃 |
| #17 草稿放置 | Saguaro/SSD + vLLM RFC `#42109` 已占核心格 |
| #18 ragged K | D-cut `2607.14647` 已做 **cross-request** 验证预算重分配（摘要逐字） |
| #19 token-yield × KV-retention | 量级单位数 % |
| #20 block-aware 多节点放置 | 基线（Epoch+EPLB）须自建；1.1–1.3× |

---

## 10. 引用纪律（违反即返工）

1. **作废 `kimi_plan.md` §1/§7 的三条"引用不实"裁定**——三条均为误判：
   - **SpecTool** = `2512.15834`（Nichols et al.；本地有全文与 §3.2 逐字引文）。`2411.13547` 是同名的另一篇 2024 benchmark，**不是**本文档所引工作。
   - **AngelSpec** = `2607.25852`（腾讯混元；本地 13:22 有正文全文）。
   - **LibraSpec** = `2608.08721`（本地 13:27 取证报告含 abs 页字节数 42,471 B / HTTP 200）。
2. **不得引用 "~20%"**（`#43559` 实测 −0.67% / −2% / −4.8%）。
3. **不得主张"没人搬草稿状态"**（SwiftSpec `2506.11309`、StarSD `2601.21622` 已占）。
4. **引用 issue/PR 必须带 repo 前缀**（vLLM `#38392` ≠ SGLang `#38392`）。
5. **判断占位必须读正文，不得用摘要**（`docs/GPT.md` §7 的自我纠错）。
6. **证据分级**：Tier A（正文）/ Tier B（仅元数据）必须标注；本方案 §3.1 与 §6 的条目均为 Tier A（正文或 PR 元数据）。

---

## 11. 仍未关闭的证据缺口

| 项 | 影响 | 处置 |
|---|---|---|
| ~~**MLSys '26**~~ ✅ **已扫描（09-11）** | #6 / #12 | **缺口关闭**。135/504 = 26.8%；Speculative Decoding 分区 4 篇**均不占 #6 的机制格**，但两篇 oral 构成挤压：**PRISM**（`2602.01762`，训练期架构重构，主张 "decouple model capacity from inference cost" ⇒ 攻击动机）、**HELIOS**（`2504.10724`，early-exit 多模型动态切换 + 实时 profiler ⇒ 挤压叙述）。`#12` 的"已被占"获第三重证据（CRAFT 专家副本 / Layered Prefill / MoE Serving Tax）。详见 `docs/evidence/mlsys2026_scan.md` |
| ~~**ASPLOS '26 / SOSP '26**~~ ✅ **已扫描（09-11）** | #6 / #12 / #17 | **缺口关闭**。ASPLOS（152/1048）：SD 分区 DFVG + SwiftSpec（→ `#17` 草稿放置/异构已被占）；**TLT** `2511.16665` 为训练期 adaptive drafter + 按 batch 选策略（策略轴，不占 `#6`）。SOSP（62/390）：**MorphKernel**（跨 SM 融合，明确针对 MoE expert activation 等不均衡算子，1.3×）、**Barrier-Free Expert Parallelism**（Tier B，待读）。详见 `docs/evidence/venue_scan_2026.md` |
| ⛔ **ATC / SC '26 入口受阻**；ISCA '26 仅索引 | #12 / #14 | 非选择性跳过：ATC 程序页取不到、SC 未找到程序页、ISCA 为硬件方向（优先级最低）。涉及该方向时不得宣称"空缺" |
| ⚠️ **NSDI '26（原清单遗漏）** | #12 | 该会议此前**不在**本方案清单内；已发现 **SwiftEP**（MoE 推理的 buffer 融合 + TMA offload） |
| ⏳ **StreamEP 正文**（SOSP '26） | #12 | **真名已从官方程序页核实**：**StreamEP: Straggler-Tolerant MoE Decoding without Communication Barriers**（USC & SNU & Google DeepMind & UT Austin，Session 2B）—— 社区清单标题有误。**正文不可得**（SOSP 会期 9/29–10/2，未上 arXiv）。**判定 `#12` 前第一优先要读**；重检触发：会后或 ACM DL/arXiv 出现 |
| **TLT `2511.16665` 正文** | #6 | Tier B；**非阻塞**（只影响引用准确度，不影响主假设；读它可确认"按 batch 选策略"是否已覆盖 depth 轴） |
| TransKV 正文（TechRxiv 403） | #1 | "二值"刻画仅据摘要；引用时标注 Tier B |
| PrefixShield / Continuum 正文精读 | #2（已归档） | **仅影响已归档项**，不阻塞 |
| ~~DA-MoE（`2607.23099`）正文~~ ✅ **已精读（09-11，Tier A）** | #12 | 计算侧「路由偏斜 → tile padding → 内核选择」**已被占**，实测 1.16×(DSV3)/1.29×(Kimi K2) geomean；kimi 称其「合入 FlashInfer」在正文中**无依据** |
| Nightjar DOI 不一致 | #13（已归档） | **仅影响已归档项**，不阻塞 |
| Libra（OpenReview `WhxNwgGkAS`）403 | #19（已归档） | **仅影响已归档项**，不阻塞 |

---

## 12. 决策记录（**已于 2026-09-11 拍板；开跑后不得再改**）

| # | 决策 | **决定** | 约束 / 理由 |
|---|---|---|---|
| **D1** | `#6` 的"测量论文"降级形态 | **(a) + 预登记的扩展条款** | 见下方 **D1 条款**（改动即事后调整） |
| **D2** | frontier 网格 | **(a)** T1 最小 18 格 + T2 反转点 | 与 D1 绑定；扩大只按 D1 条款自动触发 |
| **D3** | `#1` 的 `R_reserve` 分支 | **(b) 立项** | 若 `R_reserve / R_byte > 2` ⇒ 开"**消除保守预留**"支线；该比值 `analyze.py` 已直接输出，**边际成本≈0**，且是 portfolio 内唯一不依赖 `#6` 的线 |
| **D4** | `#6` 名称标签 | **标注为内部叫法** | 已在 §5.2 与 `notes/prereg/p06-frontier.md` 注明"非既有术语" |

### D1 条款（预登记；**改动即事后调整**）

1. **T1 = 1 个家族 × 1 个引擎，引擎固定 `vLLM`。**
   理由：只有 vLLM 同时提供"已 ship 的 K 自适应基线（`num_speculative_tokens_per_batch_size`）"与"多层 drafter"。SGLang 的 `adaptive_spec_params` **逐字拒绝多层 worker**（`enable_multi_layer_eagle=True is not supported`）⇒ 在它上面跑不出 T2 所需的对照，只能作静态基线。
2. **扩展条款（同一预登记下自动触发；优先级：家族 ≫ 引擎）**
   - 若 T1 出现**反转但幅度在噪声边缘**，或**共线且效应量 ≥ 8%** ⇒ 在**同一预登记**下**追加第二家族**（DFlash ↔ DSpark）；
   - **不自动追加第二引擎**。仅当 vLLM 的格已跑通**且** SGLang 侧已验证多层 drafter 可承载时，才追加 SGLang 作**静态基线**对照。
3. **降级形态的安全前提**：若机制主张死亡**且共线结论在两个家族上都成立** ⇒ 此时才启动**新的、独立预登记的测量论文实验**（新 prereg、新数据）。**不得把旧数据重新解读成测量论文。**

---

## 13. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-09-11 | 首版。合并 `docs/plan.md` 与 `docs/kimi_plan.md` 的排序冲突，采纳三条硬门槛；关闭 `2606.29223`；将 #2 归档理由由"被占"更正为"基线反证"；#6 主线化并加入 DEX 替换测试与 drafter 家族层数约束；#4 收窄并加复核门；#12 加 2 周时间盒。 |
| v1.1 | 2026-09-11 | 命名统一：实验目录改 `probes/pNN-*`、辅助探针 `probes/aux-cN-*`，新增 `docs/GLOSSARY.md`；补入漏列的 **#7**（备线 C + 离散度探针）；明确 20 = 14 归档 + 6 存活。 |
| v1.2 | 2026-09-11 | **无卡复核两项**：`#4` 复核门**不通过** → 降为「暂缓」（带重开触发）；`#6` 代码层家族检查**通过**但定位收紧为"drafter 网络深度/宽度"（`adaptive_spec_params` 已 ship 步数自适应）。备线重排：A=`#12`、B=`#7`；20 = 14 归档 + 1 暂缓 + 5 存活。 |
| v1.3 | 2026-09-11 | **数据采集前**收紧 #6：新增基线 ④（SGLang `adaptive_spec_params`，且它对 `enable_multi_layer_eagle` 明确不实现）、三个必答项、adaptive-steps-only 增量对照与两条杀判据；新增 `notes/p06-dex-differentiation.md`（与 DEX 的三轴区分 + 三条判死条件）。 |
| v1.4 | 2026-09-11 | 采纳外部文献调研的可核部分：**主假设升级为正交性**（drafter capacity ⟂ draft length）、网格加 `γ` 因子、No-Go 加"共线即杀"、基线补 vLLM per-batch K 查表；§5.1 加多层 drafter 取数路径（`vllm-project/speculators` + HF 权重）；§11 MLSys '26 由"未覆盖"改为"有入口、尚未系统扫描"（原 proceedings 链接 404）；新增 `docs/reviews/2026-09-11-verdict-02-*.md`。 |
| v1.5 | 2026-09-11 | **MLSys '26 全量扫描**（`docs/evidence/mlsys2026_scan.md`）：§11 该缺口关闭；`#6` 邻居清单加 PRISM/HELIOS/SpecDiff-2/"Performance or Illusion?"，并新增**必答项 ④ 动机抗辩**；`#12` 拥挤度获独立确认（CRAFT / Layered Prefill / MoE Serving Tax）。ASPLOS/ISCA/ATC/SOSP/SC '26 仍未扫。 |
| v1.6 | 2026-09-11 | **ASPLOS '26 + SOSP '26 扫描**（`docs/evidence/venue_scan_2026.md`）与 **DA-MoE 正文精读**：`#12` 占位证据升到五重以上（+ MorphKernel、Barrier-Free EP[Tier B]、DA-MoE 实测量级），门槛加"先读 Barrier-Free EP"；`#6` 邻居加 TLT（训练期 adaptive drafter，策略轴）。ATC/ISCA/SC '26 仍未扫；§11 已清理两处被取代/过期的行。 |
| v1.7 | 2026-09-11 | **无卡工作收尾**：SOSP 官方程序页核实 StreamEP 真名（社区清单标题有误）并转为 ⏳ 待发表；ATC/SC 入口受阻、ISCA 仅索引、**新发现 NSDI '26 漏在清单外（SwiftEP）**；§11 按「已关闭 / ⏳待发表 / ⛔入口受阻 / 📦仅影响已归档」四类收尾。 |
| v1.8 | 2026-09-11 | 新增 **§12 待决策项**（D1 测量论文降级形态 / D2 网格是否扩大 / D3 `R_reserve` 分支 / D4 名称标签，含默认建议与"降级须新预登记"的安全前提）；§0 补**结构风险**（主线候选仅 `#6` 一条）；`#6` 邻居表补 `2511.12031`（compute-vs-copy 配比，Tier B）。 |
| v1.9 | 2026-09-11 | **D1–D4 拍板**：D1 = (a) + 预登记的扩展条款（T1 固定 vLLM；扩展优先家族、不自动加引擎；降级须新预登记）；D2 = (a)；D3 = (b)；D4 = 标注内部叫法。§12 由"待决策"改为"决策记录" |
| v1.10 | 2026-09-11 | 审计修正：§5.2 基线纪律与 **D1 决定同步**（固定 vLLM，基线改为 vLLM per-batch K 查表）+ 澄清"两个 8% 是不同比较"；§0 加"当前阶段"；README 入口表补 `docs/EXPERIMENT_GUIDE.md` 与 `docs/reviews/`；`upstream/README` 登记锚点核对副本；清理 `results/_patchtest`（草稿） |
| v1.11 | 2026-09-11 | **独立核查后修正 7 类**：② 门槛的卡数适用范围；§4 #3 归档条件"且"→"或"；§5.2 必答项 ② 改为 **vLLM 口径**并把"三个"改为"四个"；§7 门改用真名 **StreamEP** 并加"正文不可得"的 fallback、删除未核实的"并入 FlashInfer"事实引用；§8.1 补 24 GB 分支与灰区判据；§8.2 语法类以 pre-reg 为唯一来源；`notes/decision_log.md` 重建三节结构 + 归档台账 + 日期锚点 |
| v1.12 | 2026-09-11 | 解冻条件由"三选一"改为**有依赖的三步**（② pin + 多层 drafter 核对 → ① 租卡 → ③ smoke test），并把 ② 扩展为含"该 pin 上是否有可跑多层 drafter"的核对（`#6` 家族前提），使卡型选择有依据 |
| v1.13 | 2026-09-12 | **解冻 ② 完成**（pin = vLLM main `9a35c08`）：E1 锚点全部命中（补丁无需改）、多层 drafter 可得（含 `Qwen3-4B-speculator.dflash2`）、深度为 config 旋钮；新增 `notes/p06-toolchain-check.md`（含显存算术与卡型建议：24 GB 起步，80 GB 仅当 8B/185k/bs>1） |
