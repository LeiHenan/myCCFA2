# 预登记 — Frontier（drafter 算力分配前沿，**主线决胜**）

**登记**：2026-09-11 ｜ **最后一次修订**：2026-09-11（**数据采集之前**，见 §修订记录）
**探针**：`probes/p06-frontier/` ｜ **对应方案**：**#6 主线**

## 主假设（Headline claim）

> **drafter capacity（网络层数/宽度）是与 draft length 正交的独立控制维度。**

即：不存在一个能把两者互相替代的单一配置——在某些 (bs, ctx) 区域，最优组合必须**同时**调整 depth 与 γ，且方向相反。
若该主假设不成立（两者共线），本工作就退化为"另一个 adaptive draft length 实现" ⇒ 见 §杀判据。

| 项 | 内容 |
|---|---|
| **量** | 端到端 `tok/s`、per-step draft 成本 (ms)、平均接受长度、**相对最佳固定配置与相对"已 ship 自适应"的端到端比** |
| **区间** | depth {1..N} × width {窄链式, 默认树, 宽树} × **γ {1,3,5,7}** × batch {1,8,32} × ctx {4k,32k,128k,185k}，每格 ≥3 次重复。**γ 因子是正交性检验的必需项**：至少在 `width=默认树` 子集上与 depth 同扫 |
| **基线** | ① spec off（AR）② **最佳固定配置** ③ **DSpark 生产调度器**（置信度头 + STS + 离线 SPS 成本表 + ragged-verify）④ **SGLang `adaptive_spec_params`**（已 ship 的运行时自适应）⑤ **vLLM `num_speculative_tokens_per_batch_size`**（per-batch K 查表，`dynamic_sd_lookup`）—— ④⑤ 按实际使用的引擎择一，**不得省略** |
| **家族前提** | 必须 ≥1 个**多层并行 drafter**（DFlash 5 层 / DSpark 3 层 MoE）。**EAGLE-3 只有 1 层 ⇒ 无深度旋钮，只能作基线**。取数路径：`vllm-project/speculators`（DSpark/DFlash 上游实现）+ HF 权重（如 `mgoin/GLM-5.2-speculator.dspark-block16`） |
| **截止** | W2 结束（≤2 周） |
| **附带交付物** | `notes/p06-dex-differentiation.md` 的定稿（三轴对照 + 三条判死条件的实测结果） |

## Go 判据（**全部**满足才 Go）

1. **正交性（主假设）**：存在 (bs, ctx) 区域，最优 (depth, γ) 发生**非共线反转**；
2. 存在**内点最优**（不是"越深越好"或"越浅越好"）；
3. 相对**最佳固定配置** ≥8%；
4. **三个必答项**全部通过（见下）。

## 必答项（任一不过即不 Go）

1. **与已 ship 控制器的差异**：它们调的是 `speculative_num_steps` / `num_speculative_tokens_per_batch_size`（长度轴）；本主线主张的是 drafter **网络层数/宽度**（深度轴）。
2. **为什么不能直接复用已 ship 控制器**：SGLang 控制器**逐字拒绝多层 worker** —— `enable_multi_layer_eagle=True is not supported (MultiLayerEagleWorkerV2 does not implement adaptive)`。
   **必须给出结构性理由**（例如深度轴的"接受率—成本"关系与步数轴不同、最优深度随 (bs, ctx) 反转），否则本工作退化为 plumbing。
3. **增量对照**：每个 (bs, ctx) 格上必须做 **adaptive-steps-only vs adaptive-steps+depth** 的对照；只有后者胜出且 ≥8%，深度轴才算独立贡献。

## 杀判据

| 条件 | 动作 |
|---|---|
| **`optimal depth ≈ f(optimal γ)`（共线 / 可互相替代）** | **杀或并入 `#7`/`#12`** —— 主假设不成立 |
| 收益只在 185k 角落 | 杀 |
| 无内点最优 / 无配置反转 / <8% | 杀 |
| 深度自适应相对 `adaptive-steps-only` 无增量 | 杀或改向"步数-深度联合"的窄问题 |
| 必答项 2 给不出结构性理由 | 改写定位或杀 |

## 已知的反对证据 / 拥挤邻居（必须写入结论，不得省略）

**引擎侧（比论文更强，且是真正的基线）**

- SGLang `adaptive_spec_params` 已 ship 运行时自适应（batch 1/8/32/64 → 候选步数 `[1,3,5,7]/[0,1,3]/[0,1]/[0]` + 迟滞 + 多套 CUDA-graph 原子切换），但**明确不支持多层 worker**。
- vLLM 有 per-batch K 查表（`dynamic_sd_lookup` ← `num_speculative_tokens_per_batch_size`；RFC `#48627` 的形态为 `[[1,64,3],[65,128,1],[129,512,0]]`）。
- ⇒ 高 batch 下"少投机"**已被解决**；深度轴必须在这个基线上仍有增量。

**文献侧（长度/树/选择轴，均不占本轴但构成拥挤）**

| 工作 | 实际控制的轴 | 备注 |
|---|---|---|
| AdaEAGLE `2412.18910` | **Adaptive Draft Structures**（含 draft length 建模） | 不是 drafter 网络容量 |
| OPT-Tree（TACL） | draft **tree** 结构（depth / branching） | 树轴 |
| AdaSD `2512.11280` / AdaptiveSD `2607.03876` | generation length / 多策略编排（CPU 受限） | 长度/策略轴 |
| **MemSpec** `2608.10362`（LCTES 2026） | **选哪个 drafter**（residency，内存预算） | 与"当前 drafter 多大"不同；全文 `KV cache`=0 |
| DSpark `2607.05147` | 离线**静态** depth/block 选择（2 层胜 5 层） | 证明该轴"活"，但非运行时 |
| Graft `2605.20104` | draft **树**深度（§4.4 因 CUDA-graph 静态形状放弃动态深度） | 类比而非先例 |
| MLSys '26：ReSpec / Sparse Self-Speculative Decoding / Beat the long tail | RL 训练侧、self-speculation、分布感知 | 均不占本轴 |

## 修订记录

| 日期 | 变更 | 时点 |
|---|---|---|
| 2026-09-11 | 初版 | 数据采集前 |
| 2026-09-11 | 加基线 ④、三个必答项、增量对照、两条杀判据 | 数据采集前 |
| 2026-09-11 | **主假设升级为正交性**；网格加 **γ 因子**（正交性检验必需）；基线补 ⑤ vLLM per-batch K 查表；加"共线即杀"；补拥挤邻居清单（含 MLSys '26 三篇与 DSpark 上游可用性） | **数据采集前**（本工作区尚无任何 frontier 数据） |
