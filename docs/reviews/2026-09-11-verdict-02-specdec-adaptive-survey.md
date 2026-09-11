# verdict-02 — 对"投机解码自适应控制文献调研"报告的评审（#6 主线资格）

**日期**：2026-09-11 ｜ **对象**：一份外部调研报告（结论：#6 未被充分占据但周围拥挤；#6 由 ★★★★★ 降为 ★★★★☆）
**方法**：逐条核实其引用与刻画（本地语料 + 实时检索）

## 一、核实为真的部分

| 报告说法 | 核实结果 |
|---|---|
| 热点集中在 when / how many / which tree / which draft，而非 drafter 自身算力 | **成立**。长度轴已被引擎占：vLLM `num_speculative_tokens_per_batch_size`（`vllm/v1/core/sched/scheduler.py` 的 `dynamic_sd_lookup`；RFC `#48627` 形态 `[[1,64,3],[65,128,1],[129,512,0]]`）+ SGLang `adaptive_spec_params` |
| MemSpec 回答"选哪个 drafter"，不是"当前 drafter 多大" | **成立**。`2608.10362`（LCTES 2026）；本地语料逐字：预算是 **draft-model residency**、*"Because only K drafts can reside in fast memory, MemSpec explicitly manages a small working set"*；全文 `KV cache`=0 / `prefix`=0 |
| DSpark/DFlash 使 depth/width knob 天然存在 | **成立且更有用**：DSpark 有上游实现（`vllm-project/speculators`）与 HF 权重（`mgoin/GLM-5.2-speculator.dspark-block16`）⇒ 多层并行 drafter 在 vLLM 上可取，**"只有 EAGLE-3"的风险解除** |
| 正交性实验（depth × width × length 同扫，看非共线反转） | **成立，本轮最好的方法论增量** ⇒ 已并入 pre-reg（网格加 γ、正交性升为主假设） |

## 二、硬伤

1. **MLSys 2026 三条链接全部 404**（`proceedings.mlsys.org/paper_files/paper/2026/hash/...`）。**论文本身真实存在且确是 MLSys 2026**，正确入口为 `mlsys.org/virtual/2026/`：[ReSpec `poster/3613`](https://mlsys.org/virtual/2026/poster/3613)、[Sparse Self-Speculative Decoding `poster/3510`](https://mlsys.org/virtual/2026/poster/3510)、[Beat the long tail `oral/3766`](https://mlsys.org/virtual/2026/oral/3766)。⇒ **结论可信、引用形式错误**。
   **副作用（正）**：本仓库 §11 曾把"MLSys '26 论文集"列为**未覆盖缺口** ⇒ 现改为"**有入口、尚未系统扫描**"。
2. **漏掉引擎侧占位证据**（比论文更强、且是真正的基线）：SGLang `adaptive_spec_params` 已 ship 且**逐字拒绝多层 worker**；vLLM 已有 per-batch K 查表。
3. 小问题：AdaEAGLE 原题为 "Adaptive Draft **Structures**"（含 draft length 建模），不止 length；"文献撞车测试"是误名，应为**实证正交性测试**（需要卡）。

## 三、对 `#6` 主线资格的判定

**维持主线资格，且理由比上一轮更硬。** 三条门槛全过：① 机制格空（有代码级排除证据）；② ≤2 卡/2 周（DSpark 上游可用）；③ 量级实测（4.4× 格 + DSpark 离线消融）。

主假设改写为：**drafter capacity 是与 draft length 正交的独立控制维度**；判据变为三条并列 + 一条显式杀判据（**共线即杀**）。
报告把 #6 降为 ★★★★☆ 的理由（"若 optimal depth ≈ optimal length 则创新性下降"）**正是这条杀判据** ⇒ 属"风险显式化"，不构成降级。
