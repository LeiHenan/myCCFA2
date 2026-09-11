# gap 声明 — hybrid 状态（#4 收窄版）· 复核门结论

**执行**：2026-09-11（无卡，约 2 小时）｜ **结论**：**不通过 → #4 降为「暂缓」**
**证据**：`archive/engine_prs.md`（2026-09-11 10:53）、`archive/probe_c/atom.md`（12:29）、`archive/probe_c/t2_joint_allocation.md`

## 一、上表已覆盖了什么

| 证据 | 状态 | 覆盖的内容 |
|---|---|---|
| vLLM `#53614` | **MERGED** 09-06 | Kimi-K3 checkpoint 扩展到 spec decoding / EAGLE block rewind + partial prefix caching + KV connectors；"re-keyed from its **provisional boundary**" |
| vLLM `#50172` / `#54637` | **OPEN** | GDN `mamba_cache_mode="all"` × **MTP**，"**declares functional correctness as the deliverable**" —— "让投机与 hybrid checkpoint 共存"这半场已在办 |
| vLLM `#55760`（MERGED 09-08）、`#56142`（OPEN 09-09） | — | 默认值（`prefix_cache_retention_interval`）与 shared-prefix checkpoint 上界修复 |
| vLLM RFC `#55697` + `#55873/5/6` | **OPEN** 09-09 | **应用声明式** checkpoint 边界 + 调度状态机 + 批量 GDN kernel（L40S 7.58×）；且"only one physical checkpoint block per sequence" |
| SGLang `#30393` | **MERGED** | draft-cache packed/sidecar HiCache 集成（真机制） |
| **ROCm ATOM（shipped 引擎）** | — | **完整的 state checkpoint 子系统**：ladder（`--state-checkpoint-interval-tokens` 默认 8192）+ **需求驱动的额外 rung**（`checkpoint_demand_pos`）+ **LRU 淘汰**（`_next_victim`）+ 容量记账（`checkpoints_dropped`、`Lost-to-checkpoint`）+ 三种 transfer（fork/copy/none）+ **每请求 `1 + num_speculative_tokens` 个回滚 slot，且 ring size 与投机深度显式耦合** |

## 二、我主张的 gap（一句话）

> 多分支/投机激活时，forking 状态类（GDN 等）的 state checkpoint 该如何跨请求复用，以及其内存/时间预算如何分配。

## 三、为什么它不是上表的子集

**它恰恰是上表的子集。** 逐条对照：

1. **"内存放大"不是空白**：ATOM 已把回滚 slot 数量与投机深度**显式耦合**（`1 + num_speculative_tokens`，"ring size is coupled to speculation depth"）。这是已实装设计，不是待解问题。
2. **"淘汰策略"不是空白**：ATOM 已有 ladder + 需求驱动 rung + LRU + 计入 `checkpoints_dropped` 的容量记账；换策略只是替换 `_next_victim` **一个方法**，且文中注明"a new policy can change which checkpoint is spent and never what the gate answers"。
3. **唯一剩余空白**：forking 状态类在**投机激活**下的跨请求 checkpoint 复用 —— ATOM 对 forking 类**显式排除** spec（两条独立理由：spec 路径的 state index tensor 没有读侧对应物；spec step 提交 `1 + accepted_drafts`，fork 的 successor 承诺无法兑现，状态会裂到两个 group、无单一 read index 可跨），vLLM `#50172` 正在以**正确性**为交付物尝试解除。
4. 这条剩余空白落在 `docs/old.md` 点名的**"引擎原生位置"**：那里的工作以 PR 形式落地，**低论文数是警告而非空缺**。

## 四、第一个可信数字是什么

若要强行做：spec 激活下 forking 类的状态副本开销（字节/请求）。但 `#50172` 一旦落地，这个数字很可能已被其实现覆盖 ⇒ **不构成独立的量级证据**。

## 五、结论

**不通过。#4 降为「暂缓」**，不再占用备线名额。

**重开触发（满足其一即重开）**：

1. `#50172` / `#54637` 被 **closed 未合并**，或连续 **>4 周无活动**；
2. 它落地后暴露出**结构性**缺陷（非调参可解）——例如 spec 下 fork 的状态分裂无法用任何单一 read index 表达，需要一个全新的状态提交协议。
