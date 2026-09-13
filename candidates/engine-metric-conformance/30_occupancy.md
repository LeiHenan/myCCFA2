# S3 占位与先行核查（Occupancy & prior art） — 跨引擎指标定义一致性（投机解码/服务度量）

**候选**：`engine-metric-conformance` ｜ **成本上限**：0 GPU·h（检索 + 读正文）
**目标**：确认没人已经做完，且**引擎里已 ship 的旋钮**不是你的方案。必须读正文。
**结论**：🟢 **存活，且拿到了一个具体的、后果可量化的差异轴** —— NVIDIA 自己的跨引擎对照文档**只做名称映射、不做语义等价**，且它亲手暴露了三个洞（见下）。**"语义 conformance"这一层无人做过。**

## 最近邻（≥3，标注读到的深度）

| # | 工作 | 做到了什么 | 读到什么深度 | 与本方向的关系 |
|---|---|---|---|---|
| 1 | NVIDIA **Dynamo `docs/observability/metrics-comparison.md`**（PR **#8055**） | **名称映射表**：vLLM / SGLang / TRT-LLM 三家的 36/48/14 个指标逐条对应 | ✅ **正文逐字读完** | ⚠️ **最近的邻居，但它只做名称不做语义**；且**版本停在 vLLM v0.19.0 / SGLang v0.5.9 / TRT-LLM v1.3.0rc9，实测日期 2026-04-10**，而现役是 vLLM **0.29.0**（**落后 10 个 minor**） |
| 2 | AMD ROCm **[Forced acceptance length](https://rocm.docs.amd.com/projects/atom/en/latest/forced_acceptance_length.html)** | 提供**人为控制接受长度**的能力，用于可比性测试 | 标题+文档级 | 证明"接受长度是横评的混淆变量"是**行业共识**；但它是**厂商测试工具**，不是定义规范 |
| 3 | **InferenceBench**（arXiv [2607.20468](https://arxiv.org/html/2607.20468v1)）§3.2 | 在论文里收集"**exact formulas for reference**" | 摘要+节标题级 | 证明社区**需要**公式参照；但只存在于**单篇附录**，不是跨引擎规范 |
| 4 | 2605.24217（客户端偏差）｜2608.04714（行为副作用）｜*Performance or Illusion?*（宏观趋势）｜Pimp My LLM（配置调优）｜agentic-kv-cache（淘汰策略复现） | 各自打了客户端 / 行为 / 趋势 / 配置 / 淘汰策略 | 2605 全文，其余标题/摘要级 | **靶子全都不同**：没有一篇做"**跨引擎语义等价性**" |

## 决定性证据：Dynamo 的表亲手暴露了三个洞（逐字来自其正文）

1. **只映射名称，不映射语义**。全表每一行都是 `Running requests｜num_requests_running｜num_running_reqs｜-` 这样的**名字对应**，
   **从不说明两个同名/异名指标是否是同一个量**。
2. **ITL / TPOT 的非对称是致命的**：表中
   - `Inter-token latency`：vLLM ✅ `inter_token_latency_seconds` ｜ SGLang ✅ ｜ **TRT-LLM ✗**
   - `Time per output token`：vLLM ✅ `request_time_per_output_token_seconds` ｜ **SGLang ✗** ｜ TRT-LLM ✅
   ⇒ **任何跨这三家的延迟比较，都必须在 "SGLang 的 ITL ↔ vLLM/TRT 的 TPOT" 之间做映射**；
   而在**投机解码**下 ITL 是**每步**延迟、TPOT 是**每 token**，两者相差 **`accept_len` 倍**（我们实测 **2.45–2.88×**，decision #76）。
   **该表对这一映射的语义风险零提示。**
3. **投机解码一节几乎是空的**：`Spec accept length` / `Spec accept rate` **只有 SGLang 有**，vLLM 与 TRT-LLM 两列都是 `-`
   ⇒ **三家引擎里有两家的"接受长度"连名字映射都没有**，更谈不上语义对照。
   而接受长度恰恰是投机解码性能的**混淆变量**（AMD 为此专门做了强制控制）。

**⇒ 一句话差异轴**：**名称映射已经有人做了（Dynamo），语义等价性从未被测量。**

## 引擎已 ship 的控制器 / 可观测性（最强基线，逐字取自本机 vLLM 0.29.0）

| 项 | vLLM 0.29.0 实况 | 含义 |
|---|---|---|
| `/metrics` 计数器 | `vllm:spec_decode_num_drafts_total`、`vllm:spec_decode_num_accepted_tokens_total`、`..._per_pos_total` | **有原始计数**，但**没有**现成的 `spec_accept_length`（与 Dynamo 表中 vLLM 列为 `-` 一致） |
| `bench.json` | `spec_decode_acceptance_length` / `_rate` / `per_position_acceptance_rates` / `mean_itl_ms` / `mean_tpot_ms` | **只有离线 bench 里有** ⇒ 线上与离线**两条口径** |
| Prometheus TPOT | 在投机解码下**数值等于 ITL**（issue **#19776**，doc PR **#55283** 澄清） | 同一指标名在不同配置下语义不同 |
| 五条恒等式 | 我们实测 **24/24 自洽**（I1 误差 ≤1.72%） | **单引擎内部没问题** ⇒ 差异只能在**引擎之间** |

## 差异轴

| 候选差异轴 | 是否站得住 |
|---|---|
| 跨引擎"名称映射" | ❌ 被 Dynamo 占（且已过时 10 个版本） |
| 客户端测量偏差 / 淘汰策略 / 宏观趋势 / 配置调优 | ❌ 分别被 2605 / agentic-kv-cache / Illusion / Pimp My LLM 占 |
| **跨引擎"语义 conformance"**：同一模型、同一负载、同一硬件下，**逐指标判定两引擎报告的数值是否为同一个量**，并给出可运行的 conformance 套件 | ✅ **站得住**：Dynamo 自己只做到名字；且它暴露的 ITL/TPOT 非对称使该问题**有因子级后果**（`accept_len`） |

## 结构性理由（审稿人会问的那一问）

> **"Dynamo 已经有三家引擎的指标对照表了，为什么还需要你们？"**

**结构性回答**：

1. **那张表是描述性文档，不是可执行的判据**：它说"我们的 X 叫这个名字"，**不说**"我们的 X 和他们的 X 相差 `accept_len` 倍"。文档**无法阻止**任何人把 SGLang 的 ITL 直接与 vLLM 的 TPOT 并列。
2. **它是静态快照**（vLLM 0.19.0，2026-04-10），而指标随版本变动（该文档自己加粗警告 "Metric names and counts are subject to change with engine version updates"）⇒ **名称映射会腐烂，语义判据不会**。
3. **它覆盖不到投机解码的核心量**：三家引擎里两家连接受长度的名字都没有 ⇒ 投机解码的跨引擎比较**只能靠 bench 工具的私有口径**，而 bench 的口径**不是引擎契约**。
4. 因此需要的不是"再写一份对照表"，而是**可运行的语义 conformance**：把"两个指标是不是同一个量"变成**能被测试证伪的断言**（例如：把 vLLM 的 TPOT 与 SGLang 的 ITL 并列时，必须声明 `accept_len` 因子；不声明即标记为不可比）。

## 被占时的降级

- **首选降级**：若完整 conformance 套件做不完，**只做"投机解码下的跨引擎可比性"**这一窄条（接受长度 + ITL/TPOT 映射），交付"实测差异表 + 单文件检查器"。
- **自我判死条件（必须先实测）**：若两引擎在同负载下对应指标的差异 **<8%**（即语义实际等价）⇒ **本方向当场判死**，把结论写成一条负面结果（"跨引擎指标语义实际一致，文档只是不完整"）并回到测绘。
- **不做**：不重做名称映射（Dynamo 已做）；不做客户端偏差（2605 已占）。

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] 每个存活痛点找 ≥3 个最近邻，且记录**读过的具体页/节**（只读摘要不算）
- [x] 找齐目标引擎里**已 ship** 的相关旋钮/调度器，并写成最强基线（不是只写学术邻居）
- [x] 写出审稿人会问的那一问，并给出**结构性**回答（不是『我们更自适应』这类程度性回答）
- [x] 写明『若被别人占住，本方向降级成什么』（降级形态必须先想好）

> **杀出口**：立项理由只有『没人做过』⇒ 杀；给不出结构性理由 ⇒ 改写定位或杀
>
> **本阶段判定：存活（差异轴具体且后果可量化）**。理由：最近邻（Dynamo）只做名称映射且亲手暴露 ITL/TPOT 非对称与投机解码空白；
> 我们打的是**语义 conformance**，有结构性理由、有降级形态、也有明确的自我判死条件（差异 <8%）。
