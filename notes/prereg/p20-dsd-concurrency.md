# 预登记 — DSD（动态投机调度）在**并发 + 前缀缓存开启**下的代价

**登记时点**：2026-09-14，**数据采集前** ｜ 目标 `goal-a729588f`（三方向）｜ 成本上限 ≤0.3 GPU·h

## 为什么是这一格（来自投机解码地图 D4/B1.6）

上游自己的 merged benchmark（PR #45953）**只测 BS1 且关闭前缀缓存**（命令里全是 `--no-enable-prefix-caching`，
只报 base/EAGLE3 的 TPOT `2.91/2.97/2.91` ms）⇒ 地图原话：*"DSD at concurrency on this class is essentially uncharacterized."*
同时有多条 **OPEN** 报告称：
- vLLM #48494：*"[DSD] incur[s] a **12–25% throughput penalty** vs no-spec … even with an all-K=0 table
  producing nearly zero draft tokens"* + *"**The presence of the batch-size table is the trigger**; K=0 tiers are not required."*
- vLLM #49548：*"catastrophic aggregate-throughput collapse under concurrency"*（8 并发 ~232 → 24–157 tok/s）

## 主假设（可证伪）

> **H1**：在**并发 8 + 前缀缓存开启**下，**"表的存在"本身**（而非"表里 K 的值"）会造成可测的吞吐下降。
> 即：`all-K=0 表` 与 `静态 K=0`（等价于不投机）在**相同请求集**下吞吐显著不同。

**若 H1 成立** ⇒ 这是一个**结构性**开销（调度器/图/元数据路径），不是调参问题，且**上游自己的基准没覆盖**。
**若 H1 不成立**（两者在噪声内相等）⇒ 我方负载上不存在该开销，**本条按否证记录**。

## 观测轴与判据（数字写死）

**固定**：vLLM 0.29.0 / Qwen3-4B / sm120 / `--max-model-len 8192` / `--gpu-memory-utilization 0.55` /
**并发 8**（客户端并发，`vllm bench serve` 同一 prompt 集与顺序）/ 输出长度固定（`ignore_eos`）/
**前缀缓存开启**（与上游基准相反，这是本格的条件 C）/ 每格独立 serve / 每格重复 **3** 次。

**六臂（并发全部 = 8；drafter 统一用 dflash2 保证跨臂唯一变量是"表"）**：

| 臂 | 投机配置 | 意图 |
|---|---|---|
| A1 | 无 | 绝对基线 |
| A2 | dflash2, `num_speculative_tokens=3`，**无表** | 静态基线 |
| A3 | dflash2, `=0`，**无表** | 静态 K=0 → **应与 A1 等价**（H1 的对照零点） |
| A4 | dflash2, `=0`，**表 `[[1,8192,0]]`** | **H1 的核心测试**：表存在但全 K=0 |
| A5 | dflash2, `=3`，**表 `[[1,3,3],[4,8192,0]]`** | 生产式跳变表 |
| A6 | dflash2, `=3`，**表 `[[1,8192,3]]`** | 表 = 常数 K |

**主判据 P1（H1）**：`throughput(A3) − throughput(A4)` 的**相对差** ≥ **8%** 且方向为 A4 更慢，
且超过噪声（见 §噪声）；符号相反或 <8% ⇒ **H1 否证**。
**副判据 P2**：`throughput(A2) / throughput(A1)`（投机净收益）；若 <1 则记录为"投不划算"的独立复现。
**副判据 P3**：A2 的 `accept_len`（由 `/metrics` 的 `vllm:spec_decode*_total` **做差**得到）应 ≈2–3；
A3/A4 的 `num_drafts` 增量应为 **0**（证明"K=0 表确实没出 draft"——否则 H1 的对照无效）。
**副判据 P4（口径）**：A1 与 A3 的吞吐应在噪声内相等（若无 ⇒ 说明 K=0 路径本身有额外开销，须单列）。

## 噪声与可分辨性

同格 3 次重复，报 p50 与极差；**k = 3**（效应 ≥3× 重复极差才称效应）。
若噪声 >8% ⇒ 提高重复到 5 次（扩展条款 E1），并如实标注。

## 杀判据

| 结果 | 处置 |
|---|---|
| **P3 失败**（A3/A4 的 `num_drafts` 增量 ≠ 0） | **本轮作废**（对照不成立：表未生效） |
| **P4 失败**且差 ≥8% | 记录为**另一条**结构性开销（K=0 路径），H1 改用 A3 作对照重判 |
| **P1 <8% 或符号相反** | **杀 H1**（记为否证），不追加实验 |
| 任一臂 serve 未就绪 | 该臂作废；关键臂（A3/A4）缺 ⇒ 整轮重跑 |

## 扩展条款

- **E1**：噪声 >8% ⇒ 每格重复提到 5 次。
- **E2**：若 P1 成立 ⇒ 追加**并发 {1, 32}** 两点，判断开销是否随并发变化（**先登记后触发**）。
- **E3（明写不做）**：不做多卡/DP（上游文档：DSD 与 DP 不兼容，会自动关闭表）；不做 Eagle/DFlash 以外的方法（文档：只测过 Eagle/Eagle-3/DFlash）。

## 环境前提

驱动 580 / sm120；`VLLM_USE_V2_MODEL_RUNNER` 不设（auto ⇒ 实测选 V2）；`--enforce-eager`（文档：Full Cudagraph 只在 V2 上工作，eager 下无图差异）；
前缀缓存**开启**（本格的条件 C）；prompt 集固定顺序；`temperature=0`。

## 修订记录

| 日期 | 变更 | 时点 |
|---|---|---|
| 2026-09-14 | 首次登记：H1、P1–P4、A1–A6 六臂、杀判据、E1–E3 | **数据采集前** |
