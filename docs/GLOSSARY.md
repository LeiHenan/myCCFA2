# 命名法（词表）

本仓库有 6 套编号，来自不同上游文档。**文件系统只用 §0 的规则**，其余编号仅出现在正文引用中。

## 0. 一眼看懂（文件系统规则）

| 形式 | 含义 | 例 |
|---|---|---|
| `#NN` | **候选方向**编号（1–20），沿用 `docs/plan.md` §1 排序表 | `#6` = 投机算力分配（**主线**） |
| `probes/pNN-<slug>/` | **第 NN 号方向的实验**（`pNN` 读作"#NN"） | `probes/p06-frontier/` = #6 的实验 |
| `notes/prereg/pNN-<slug>.md` | 该实验的预登记判据（与实验目录**同名**） | `notes/prereg/p06-frontier.md` |
| `results/pNN-<slug>/<日期>/` | 该实验的产物 | `results/p01-e1-uncommitted-kv/2026-09-12/` |
| `probes/aux-cN-<slug>/` | **辅助探针** C1–C9（来自 `docs/plan.md` §2.6，**与候选方向无关**） | `probes/aux-c2-flashinfer-radix/` |

> **唯一容易搞混的点**：`p03-*` 是**候选 #3** 的实验；辅助探针表里的 **C3** 是"受迫区间测量 → jump-forward"，写作 `aux-c3-*`。两者不是一回事——这正是 2026-09-11 改名的原因。

## 1. 候选方向 `#1`–`#20`（20 个）

- **存活 5 个**：`#1`（E1 探针）｜`#3`（grammar scan 探针）｜`#6`（**主线**）｜`#7`（备线 B，待判定）｜`#12`（备线 A，对冲）。
- **暂缓 1 个**：`#4`（复核门不通过，2026-09-11；重开触发见 `notes/gap_hybrid_state.md`）。
- **归档 14 个**：`#2 #5 #8 #9 #10 #11 #13 #14 #15 #16 #17 #18 #19 #20`（理由见 `EXECUTION_PLAN.md` §9）。

## 2. 实验名

| 实验名 | 属于 | 是什么 | 时间盒 |
|---|---|---|---|
| **E1** | #1 | 未提交草稿 KV / 已提交 KV 的测量（名出 `docs/GPT.md` §6） | 1 天 |
| **grammar scan** | #3 | 约束解码 `batch × 语法复杂度 × 投机开关` 开销扫描 | 1 周 |
| **hybrid gate** | #4 | 2 小时文献/PR 复核门（**不是实验**） | 2 小时 |
| **frontier** | #6 | `depth × width × bs × ctx → tok/s` 前沿图（**主线决胜**） | 2 周 |
| **adapter dispersion** | #7 | per-adapter prefix 命中率离散度探针 | ≤1 周 |
| **EP imbalance** | #12 | MoE expert-parallel 不均衡 + dispatch 对冲实验 | 2 周时间盒 |

## 3. 辅助探针 `C1`–`C9`

出处：`docs/plan.md` §2.6（**表内顺序即编号**，原文未标 C 号）。

`C1` prefill CUDA-graph padding｜`C2` FlashInfer radix（被静默 `disable_radix_cache=True`）｜`C3` 受迫区间测量→jump-forward｜`C4` byte 记账的 KV 准入｜`C5` RefShape 重开｜`C6` dLLM radix refresh｜`C7` 异构 rank 统一分页+策略｜`C8` drafter 侧 LoRA｜`C9` thinking-cache 生命周期。

本工作区**只启用 C1 / C2**（填缝探针）。

## 4. 只出现在上游正文的旧编号（**不要再用于新文件**）

| 编号 | 出处 | 含义 | 对应关系（已核对 `docs/plan.md` 的"来源"列） |
|---|---|---|---|
| `A1`–`A4`、`B1`–`B10` | `docs/V41.md` §1 | V41 候选分级（A 强 / B 中） | A1→#3｜A2→#4｜A3→#12｜A4→#6｜B1→#18｜B2→#13｜B3→#8｜B4→#9｜B5→#7｜B6→#11｜B7→#10｜B8→#17｜B9→#20｜B10→#2 |
| `GPT-①`–`GPT-⑨` + `A/B/C` | `docs/GPT.md` §2、§4 | GPT 原始提案九方向 + 三个提取耦合 | ①→#5｜②→已杀｜③→#1｜④→已杀｜⑤⑥→#14｜⑦→#15｜⑧→#19｜⑨→非命题｜A→#5｜B→#16｜C→#15 |
| `r3_*` / `r4_*` | `archive/` | 对抗性筛查**轮次**（不是候选） | `r4_graded_kv_tiers.md` = #1 的子领域扫描；`r4_kv_prefetch.md` = #14 的 |

## 5. 改名映射（2026-09-11）

| 旧 | 新 |
|---|---|
| `probes/p01-e1-uncommitted-kv/` | `probes/p01-e1-uncommitted-kv/` |
| `probes/p03-grammar-scan/` | `probes/p03-grammar-scan/` |
| `probes/p04-hybrid-gate/` | `probes/p04-hybrid-gate/` |
| `probes/p06-frontier/` | `probes/p06-frontier/` |
| `notes/prereg/p01-e1-uncommitted-kv.md` | `notes/prereg/p01-e1-uncommitted-kv.md` |
| `notes/prereg/p03-grammar-scan.md` | `notes/prereg/p03-grammar-scan.md` |
| `notes/prereg/p06-frontier.md` | `notes/prereg/p06-frontier.md` |
