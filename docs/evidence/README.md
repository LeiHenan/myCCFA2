# docs/evidence/ — 承重证据扫描（副本）

三份扫描的**副本**，原件位于未入库的冻结目录（来源见每节）。凡涉及"某方向空缺 / 已死 / 拥挤"的判断，一律回到这三份文件核对，不凭记忆、不凭摘要。

---

## 1. `r4_graded_kv_tiers.md` — 分级 KV state 子领域扫描（#1 的判定依据）

**原件**：`subfield_scan/r4_graded_kv_tiers.md`（2026-09-11 14:59 冻结）

**承重结论**

- 机制**未被占**（因此不是 V1 击杀），但三处反向压力同时存在：① SpecMemo 要求投机行保持高数值精度 ⇒ 压缩档可能破坏无损性；② **未提交投机 KV 占比全库无人测量**（该扫描第 9 条逐字确认）；③ 相对已 ship 的二值处理，分级优势**只有一步宽**。
- 结论原文：**"Run E1 before spending any more screening effort"** —— 这正是本工作区把 #1 保留为探针的原因。

**已知缺口（引用时必须标注）**

- **ASPLOS / ISCA / ATC / SOSP / SC / MLSys '26 论文集未覆盖**（ACM DL / CSDL 被拦）⇒ 架构会议方向上的"空缺"结论只是暂定。
- TransKV 正文从未读到（TechRxiv 403），"二值"刻画**仅据摘要**。
- `arXiv 2606.29223` 当时未能取到全文 ⇒ **本工作区已于 2026-09-11 关闭该缺口**：它 = DEX《Depth Exploration for LLM Decoding》，不占 #1。详见 `EXECUTION_PLAN.md` §3.1。

---

## 2. `r4_kv_prefetch.md` — 计算侧 KV 预取扫描（#14 的判定依据）

**原件**：`subfield_scan/r4_kv_prefetch.md`（2026-09-11 14:51 冻结）

**承重结论**

- O7（机制混淆）已被证实：存储层 host/remote→HBM **有**预测，计算侧 HBM→**片上** **无**预测 —— 只有这个合取幸存。
- 该合取**已有一条已发表的失败记录**：Levy `2603.13430` §5.3 逐字 *"essentially failing at this approach"*。
- stall 分解显示**权重流量 58% > KV 32.7%** —— 上限被压低。

⇒ #14 归档（负结果风险全场最高）。

---

## 3. `constrained_decoding.md` — 约束解码子领域扫描（#3 的先验来源）

**原件**：`archive/subfield_scan/constrained_decoding.md`（2026-09-11 12:36 冻结）

**承重结论**

- 三个候选 gap **全部 KILLED**，上限分别为 **1–2% / 1–6% / ≤2.5%**。
- XGrammar-2 自报端到端：**"the gap between the result of XGrammar 2 and the result without constraints is no more than 6%"**。
- T5 原文：**"this is not a subfield where a gap is hiding. It is a subfield that has been converged on … Entering now means arriving after the result."**
- 唯一未解的是**复杂 CFG**（XGrammar2 上 C++ / Python 变体有 ~2–10× 慢），而 CFGzip 自报 7.5× 已在其上 —— 这正是 #3 的三交集的最后缝隙。

⇒ **#3 只能当低概率彩票**：任何 Go 结论都必须同时写明这条 ≤6% 的先验（`EXECUTION_PLAN.md` §8.2）。

---

## 4. `mlsys2026_scan.md` — MLSys 2026 论文集扫描（#6 / #12 / §11 缺口）

**执行**：2026-09-11（无卡）｜ **结论**：`#6` 的机制格**未被占**，但两篇 oral 从两侧挤压 —— **PRISM** `2602.01762`（训练期架构重构，主张容量与成本解耦 ⇒ 攻击动机；已转为 pre-reg 的**必答项 ④**）、**HELIOS** `2504.10724`（early-exit 多模型切换 + 实时 profiler ⇒ 挤压叙述）；`#12` 的"已被占"获得第三重证据（CRAFT / Layered Prefill / MoE Serving Tax）。

**入口更正**：`proceedings.mlsys.org/paper_files/paper/2026/hash/...` **404**；正确入口 `mlsys.org/virtual/2026/`（oral/poster 编号页）。规模 135/504 = 26.8%。

**仍未覆盖**：ASPLOS / ISCA / ATC / SOSP / SC '26。

---

## 使用纪律

1. 三份文件都是**副本**；修改原件请改源目录，改完重新同步副本，并在 `notes/decision_log.md` 记录。
2. 引用其中的数字时标注 **Tier A（正文）/ Tier B（仅元数据）**，与 `docs/GPT.md` §7 的规则一致。
3. 缺口栏（尤其是架构会议论文集）在相关方向被重新提起时**必须重新检查**，不得直接沿用"空缺"结论。
