# 预登记 — Frontier（drafter 算力分配前沿，**主线决胜**）

**登记**：2026-09-11 ｜ **最后一次修订**：2026-09-11（**数据采集之前**，见 §修订记录）
**探针**：`probes/p06-frontier/` ｜ **对应方案**：**#6 主线**

| 项 | 内容 |
|---|---|
| **量** | 端到端 `tok/s`、per-step draft 成本 (ms)、平均接受长度、**相对最佳固定配置与相对"已 ship 自适应"的端到端比** |
| **区间** | depth {1..N} × width {窄链式, 默认树, 宽树} × batch {1, 8, 32} × ctx {4k, 32k, 128k, 185k}，每格 ≥3 次重复 |
| **基线** | ① spec off（AR）② **最佳固定配置** ③ **DSpark 生产调度器**（置信度头 + STS + 离线 SPS 成本表 + ragged-verify）④ **SGLang `adaptive_spec_params`（已 ship 的运行时自适应控制器）** |
| **家族前提** | 必须 ≥1 个**多层并行 drafter**（DFlash 5 层 / DSpark 3 层 MoE）。**EAGLE-3 只有 1 层 ⇒ 无深度旋钮，只能作基线** |
| **截止** | W2 结束（≤2 周） |
| **附带交付物** | `notes/p06-dex-differentiation.md` 的定稿（三轴对照 + 三条判死条件的实测结果） |

## Go 判据（**全部**满足才 Go）

1. 存在**内点最优**（不是"越深越好"或"越浅越好"）；
2. 在 (bs, ctx) 平面上出现**配置反转**；
3. 相对**最佳固定配置** ≥8%；
4. **三个必答项**全部通过（见下）。

## 必答项（任一不过即不 Go）

1. **与已 ship 控制器的差异**：`adaptive_spec_params` 调的是 `speculative_num_steps`（长度轴）与 `speculative_num_draft_tokens`；本主线主张的是 drafter **网络层数/宽度**（深度轴）。
2. **为什么不能直接复用已 ship 控制器**：该控制器**逐字拒绝多层 worker** —— `enable_multi_layer_eagle=True is not supported (MultiLayerEagleWorkerV2 does not implement adaptive)`。
   **必须给出结构性理由**（例如深度轴的"接受率—成本"关系与步数轴不同、最优深度随 (bs, ctx) 反转），否则本工作退化为 plumbing。
3. **增量对照**：每个 (bs, ctx) 格上必须做 **adaptive-steps-only vs adaptive-steps+depth** 的对照；只有后者胜出且 ≥8%，深度轴才算独立贡献。

## 杀判据

| 条件 | 动作 |
|---|---|
| 收益只在 185k 角落 | 杀 |
| 无内点最优 / 无配置反转 / <8% | 杀 |
| **深度自适应相对 `adaptive-steps-only` 无增量** | 杀或改向"步数-深度联合"的窄问题 |
| 必答项 2 给不出结构性理由 | 改写定位或杀 |

## 已知的反对证据（必须写入结论，不得省略）

- 已 ship 的控制器在 batch 1/8/32/64 上已把候选步数压到 `[1,3,5,7] / [0,1,3] / [0,1] / [0]` ⇒ **高 batch 下"少投机"已被自适应解决**；深度轴必须在这种基线上仍有增量。
- DSpark 的离线消融已证明深度轴"活"（2 层胜 5 层），但那是**静态架构选择**，不是运行时分配。
- 多条反面约束（该控制器同时不支持 topk≠1、DP attention、two-batch overlap、pdmux）说明"状态原子切换"的适用范围本身很窄 —— 这既是机会也是工程量警告。

## 修订记录

| 日期 | 变更 | 时点 |
|---|---|---|
| 2026-09-11 | 初版 | 数据采集前 |
| 2026-09-11 | 加入基线 ④（`adaptive_spec_params`）、三个必答项、增量对照与两条新增杀判据 | **数据采集前**（本工作区尚无任何 frontier 数据） |
