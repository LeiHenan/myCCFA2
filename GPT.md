# GPT — 对 GPT 提案的审核与候选筛选

**日期**：2026-09-11
**审核对象**：一份 GPT 生成的选题提案（9 个方向 + 5 个推荐 + 1 个合并愿景「Speculation-Aware LLM Inference Runtime」）
**审核方法**：① 逐条核实引用；② 与本轮已有击杀台账（47 条击杀理由，见 `old.md`）对齐；③ 对从中提取的每一个耦合做针对性对抗性筛查（三轮，报告在 `subfield_scan/r3_*.md`）

> 配套文件：`V41.md`（本轮幸存候选）、`old.md`（引用纪律、判据框架、击杀台账）

---

## 0. 结论

> **9 个方向全部被占或与其重复；从中提取的 3 个"真正未被覆盖的耦合"经三轮针对性筛查后全部判死。这份提案没有剩下可做的候选。**

它的引用**基本真实**（9 条 arXiv/会议引用全部解析成功，无 phantom），问题不在引用，而在**它对现状的定性**——而三处关键定性错误恰好都指向它自己的立论基础。

---

## 1. 引用核查结果

**9 条引用全部真实**——这是这份文档比本轮早期若干材料强的地方。

| 引用 | 核查结果 |
|---|---|
| SpecDec++ `2405.19715` | ✅ 真实，**COLM 2025** |
| EAGLE `2401.15077` | ✅ 真实 |
| **RocketKV** `2502.14051`（它给的是 PMLR v267 behnam25a） | ✅ 真实，**ICML 2025**。"up to **3.7×** end-to-end speedup … on an NVIDIA A100" **逐字属实**——⚠️ 但原文限定 *"compared to the **full KV cache baseline**"*，**不是相对调好的稀疏基线** |
| SAGE-KV `2503.08879` | ✅ 真实，**但是纯 eviction**（《LLMs Know What to Drop》）——**不是**它说的"稀疏注意力执行" |
| P/D-Serve `2408.08147` | ✅ 真实 |
| KV 综述 `2603.20397` | ✅ 真实（24 页 14 图） |
| **SparseSpec**（未给 ID） | ✅ = `2512.01278`，是**自投机**（同一模型兼任 draft/target） |
| **LServe** | ✅ 真实，MLSys 2025 |
| **Q-First**（未给 ID） | ✅ = `2608.15473`，是**PD 分离**论文，不是通用稀疏注意力工作 |

### 三处关键定性错误（都指向它自己的立论）

**① 它把 SpecTool 归为"client-side tool speculation"，据以论证"应把思想深入 engine 内部"——但 SpecTool 的 §3.2 标题就是 "Engine-side Speculative Tool Calling"。**
逐字：*"On a cache hit, the tool result is appended and the **KV-cache updated** so decoding can continue without eviction. Otherwise, the engine **falls back to baseline** behavior."* 并在 vLLM 实测 *"up to 196 tokens/second improvement … 32 gpt-oss-120b agents."*
⇒ **方向 ① 的立论基础不成立。**

**② 它把 RocketKV 的 3.7× 当作稀疏注意力的当前水平——那是相对 full-KV 基线的数。**
这正是 `old.md` §2 的头号失效模式：**一个相对弱基线的数字被当成整个空间的性质。**

**③ 它说 SpecTool 报告的 6–21% agent 端到端节省——那个数字在论文里是 client-side 的**（Nichols Fig. 6，tool latency 0.5–3 s，峰值在 2.0–2.5 s）。**engine 侧的增量实测只有 "an additional 2-3% time saved."**

---

## 2. 逐方向判定（9/9 死）

| # | 它的方向 | 判定 | 杀死它的具体依据 |
|---|---|---|---|
| **①** | Tool/Agent-aware Speculative Decoding | ❌ **死** | **SpecTool `2512.15834` §3.2 已是 engine-side**；**PASTE `2603.18897` 已有 in-engine 调度钩子并逐字提出概率化准入**（*"admits speculative work when it is likely to hide exposed tool time"*，`EnginePressure = DecodeLoad + γ·KVLoad`）；**SPORK `2607.03333` 已 fork KV cache 并接入 vLLM proposer**；**Speculative Actions `2510.04371` 已有准入规则**（Thms 3–5）；`2606.07846` 已有 *"expected-value rule with a failure-weighted cost term … behind a commit barrier"*；**vllm-omni `#4907` OPEN + PR `#4909` DRAFT 已在做 COW KV fork + rollback** |
| **②** | Adaptive SpecDec / "K 改成计算预算" | ❌ **死** | 它自己承认"动态 K 不是新 idea"——而"改成预算"**也已发表**：LibraSpec `2608.08721` 的 *"marginal criterion"*；SparseSpec-L `2607.27735` 的 *"marginal acceptance probability falls below the relative drafting cost"*。另有 CAST / CaDDTree / GLANCE / AngelSpec。vLLM `#54749` 逐字：*"Six different signals are being proposed for this one decision right now. Batch size is what ships."* |
| **③** | Speculative KV Cache | ❌ **死** | 容量/预留轴：SpecMemo `2506.01986`、Nightjar（`10.1016/j.sysarc.2026.103889`）、MemSpec `2608.10362`、TransKV。**精度轴见 §4-B** |
| **④** | KV **Placement** 而非 Eviction | ❌ **死** | vLLM 已 ship `TieringOffloadingSpec` + 策略钩子；**Dynamo KVBM 已 ship G1–G4 + TinyLFU/CMS，`frequency≥2` 磁盘准入过滤器默认开启**；Mooncake / LMCache / SGLang HiCache（7 种驱逐策略 + 11 种存储后端）。**RFC `#54779` 内已含完整 cost-aware TinyLFU 准入设计，原型 `#54327` 开放中** |
| **⑤⑥** | 预测性 attention 预取 / KV prefetch | ❌ **死** | tracker 拥挤（SGLang `#21846` 的 `PREFETCH`/`DEMOTE`/`PIN`、vLLM `#48445`/`#52113`/`#51428`）、SGLang 已 ship `--hicache-storage-prefetch-policy`、AAAI 2026 已有异步 KV 预取。**算术上限 2–8%** |
| **⑦** | Attention + 投机联合 | ❌ **死** | **Vegas（ICML 2026）**；**BudgetDraft `2606.00144`**（标题字面 *"**Acceptance-Aware** … Sparse-KV Speculative Decoding"*）；**且已是 flag**（vLLM `enable_adaptive_verification`）。详见 §4-C |
| **⑧** | Speculation Scheduler | ❌ **死** | YieldSched 已被击杀（四条轴全部已实现）。而这个公式**正是 vLLM `#54749` 作者造出来、并在唯一一次头对头测试中输给一张实测表的那个模型**：*"…it selects K=0 at the high-batch tier at every context … would not have found the cell worth 29–36%"* |
| **⑨** | 合并愿景 | ❌ **不是研究问题** | 它是 ②③④⑤⑧ 的并集，**每一块都已被占**。本轮的中心发现正是"把 N 个已被占的机制组合起来"是密度杀死你的地方（K7 即实例，残余 1.1–1.3×） |

---

## 3. 它的"不太推荐"四项——我逐条同意

| 它的判断 | 我的依据 |
|---|---|
| ❌ 设计新 Attention | LServe（MLSys'25）、RocketKV（ICML'25）、SAGE-KV、Q-First `2608.15473` 已占实 |
| ❌ 新 KV eviction score | 同上 + `Random Attention` `2609.03430` 逐字：*"We show that the **selection signal contributes almost nothing**"*——整个驱逐打分层被动摇（但那是负面结果，被"必须是加速论文"约束排除） |
| ❌ 更好的 draft model | drafter 架构被 EAGLE-2/3、MTP、DFlash（ICML'26）、DFlare、JetSpec、Domino、P-EAGLE、DART、Spiffy 占满；Markov 替换被 xPress / LiLiCorr 占；树转换被 PCTree / DARTree 占 |
| ❌ 固定 K → 动态 K | 见 §2 ②，**已发表四次以上** |

---

## 4. 三个提取耦合的死亡记录

> 这三个是我从提案里挑出的、本轮台账**没覆盖**的耦合。全部做过针对性对抗性筛查（`subfield_scan/r3_agent_spec.md`、`r3_kv_precision_spec.md`）。

### A. Agent 级投机的 commit/rollback 语义 + 概率化准入 → **DEAD**

**机制已被发表，而且不止一次：**

| 工作 | 层 | 提交/回滚什么 | 碰 KV？ |
|---|---|---|---|
| **SpecTool** `2512.15834` | **ENGINE** | 命中则提交工具结果 + 参数草稿 token，未命中回落到 client 往返 | **是** |
| **SPORK** `2607.03333` | 控制器 **+ engine** | 匹配则提交，否则串行回落；**fork KV cache**、接入 vLLM proposer 路径 | **是** |
| **PASTE** `2603.18897` | 编排 **+ engine** | 匹配则复用/提升，否则丢弃；**"scheduler hooks around the LLM serving engine"**、in-engine load-shaping hook | **是** |
| **Speculative Actions** `2510.04371` | 环境/编排 | 匹配才提交 + 修复路径；**Thms 3–5 已含置信度感知的选择性分支启动** | 否 |
| `2606.07846` | — | *"expected-value rule with a failure-weighted cost term … behind a **commit barrier**"* | — |
| **vllm-omni** `#4907` OPEN + PR `#4909` **DRAFT** | engine | **COW KV fork for "speculative action branches… rollback"** | **是**（实测 **37 GiB/branch，H200 上约 2 个分支**） |

⇒ **"engine 内部提交语义"和"概率化准入"两半都已被占。** 而且它不是 flag——但它是**四个已发布系统的重参数化**。

**量级（这是独立的第二重杀死）**：
- 6–21% 是 **client-side** 的数（tool latency 0.5–3 s）；**engine 增量实测只有 "an additional 2-3% time saved."**
- 筛查给的界：`S = p·min(D,L)/(D+L)·(1−ε)`，最大 0.5p 在 `D=L` 处——与原文最优点在 `D≈L` 吻合（21% 反解出 `ε≈0.48 @ p≈0.8`）
- B200（TPOT 1 ms、500-token CoT、D=0.5 s）：工具 2.5 s → 可隐藏 16.7% → **~7%**；工具 10 s → **~2%**
- SPORK：被拒探针只挽回 *"a median of 18 verified tokens"*，而 CoT ~2,000 token ⇒ **<1% 有用**

### B. 投机不确定度 → KV 精度 → **DEAD**

**占位者**：
- **MiKV** `2402.18096` 逐字：*"the important KV pairs must be kept at a relatively higher precision… retaining the evicted KV pairs in low-precision"* —— **这就是本耦合的映射，只是把 `P(commit)` 换成了 "importance"**
- 投机专用形式：**QuantSpec** `2502.10424`、**QSpec** `2410.11305`
- 不确定度→分档阶梯：**"Don't Waste Bits!"** `2604.04722`（CVPR 2026）——*"entropy-based uncertainty… dynamically selects KV precision from {2-bit, 4-bit, 8-bit, FP16}"*

**两重独立杀死：**

**① 算术天花板 ≈1.2%**（Llama-3-8B，128 KiB/token FP16）

| 场景 | 未验证窗口占 KV |
|---|---|
| B=32, L=4096, k=5（160 token） | **0.122%** |
| 同上按 INT4（4×）算可省 | **0.091%** |
| B=8, L=128K | **0.0038%** |
| 最佳情形（64 节点 draft tree, B=32, L=4096） | 1.56% → 净 **1.17%** |

**② 结构论证（更根本）**：*"**rejected tokens are discarded**, so precision only matters for tokens **already committed**"* —— 已实现的 commit 结果就是 1，**精度阶梯的低档位只作用于根本不会持久化的条目**。

**③ 正确性不可逆**：INT8 = +0.34 PPL、INT4 在 INT8 之上再 +1.65 PPL、2-bit *"drastic performance decline"*——**永久损伤，无回滚**。而投机解码的核心卖点是**无损**。

### C. Confidence-aware Attention Budget → **DEAD**

- **Vegas（ICML 2026）** 逐字：*"Vegas identifies critical KV cache entries **as a byproduct of verification** and computes attention only over these entries when drafting subsequent tokens."* 实测 **1.25×–2.81×**，相对 SOTA 稀疏注意力自投机方法 **1.15×–1.29×**
- **BudgetDraft** `2606.00144` 标题字面：*"**Acceptance-Aware** Multi-View Training for Sparse-KV Speculative Decoding"*
- **已经是 flag**：vLLM `enable_adaptive_verification`（PR `#47808`）——*"a global top-B over survival scores"*
- **理论性杀死**：逐位置稀疏在成本上**中性**——SparseSpec 的模型给出平均 attention 输入 `(k·s+1)/(k+1)·M`（单一标量 s）；逐位置把 `k·s` 换成 `Σs_i ≥ k·s_min`，**所以均匀 `s=s_min` 已经是最优成本**。加上"延后计算"本身不自洽（draft token 必须被处理）

---

## 5. 与其推荐相左的三点

1. **它最推荐的 ① 有双重致命问题**：立论基础错（SpecTool 已是 engine-side），且**量级不足**（engine 增量 2–3%）。
2. **② 应直接划掉**——它自己承认动态 K 不新，而"改成预算"同样已发表。
3. **⑨ 层级高不等于可发表**。K7 就是"两个已发表半边的组合"，残余仅 1.1–1.3×。**组合愿景必须先收敛成一个可证伪的命题**——它自己也这么说。

---

## 6. 仅存的残余（不是候选）

筛查给出一个**唯一未被覆盖**的残余：

> **多租户下、跨*并发*投机分支的准入 + KV 容量调度。** 现有全部准入规则都是**逐会话（per-session）**的，而 vllm-omni 的实测（**37 GiB/branch，H200 上约 2 个分支**）说明**扇出会被 KV 直接杀死**。

**但它不是候选**：量级风险为**单位数百分比**，且需要先测"KV 是否是绑定约束"。筛查自己的建议是：**先测 KV bindingness**，再决定要不要做。

---

## 7. 我在这份审核里犯的错（留档）

**我只读了 SpecTool 的摘要就下判断。** 我在本文件初版 §4-A 里写的是"SpecTool 进了 engine 但做的是**驻留与吞吐**，不是**分支状态的提交语义**"——**这是错的**：它的 §3.2 标题就是 "Engine-side Speculative Tool Calling"，正文明确包含 KV 提交与回滚。

这与 `old.md` §2 列为**头号失效模式**的那件事是同一个：**把从一个窄窗口看到的东西当成整体性质**。区别只是这次的"窄窗口"是**摘要**而不是测量格。

⇒ **新增规则**：**判断一个方向的占位情况，必须读正文，不得用摘要。** 摘要系统性省略机制层细节——而"机制层细节是否已被占"恰恰是判定的全部内容。

---

## 8. 顺带纠正的引用属性

| 引用 | 真实情况 |
|---|---|
| SAGE-KV `2503.08879` | 纯 **eviction**，不是"稀疏注意力执行" |
| **SpecGuard** `2609.11799` | **是后门检测**——我递过去的 ID 映射到了一篇无关论文 |
| ConfSpec `2602.18447` | ACL 2026，但是**步骤级 CoT 推理**，不是工具调用 |
| SMC `2609.03236` | **MLSP2026 是 workshop**，不是 CCF-A |
| Q-First `2608.15473` | **PD 分离**论文 |
| SparseSpec `2512.01278` | **自投机**，不是独立 drafter |

**15 个摘要页在 raw HTML 上做过撤回复扫：干净。**
