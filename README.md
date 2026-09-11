# myCCFA — MLSys 选题工作区

本工作区用于**选定并执行一条 MLSys 论文主线**。当前阶段：**⏸ 停工等卡** —— 无卡工作（文献/引擎/仓库审计）已收尾；下一步必须用 GPU。

**解冻条件（三选一即可开工）**：① 拿到 1 张 24 GB 卡 → 跑 `p01` 的 E1（1 天）；② 给出 vLLM 的 pin commit → 我按该版本核对仪器化锚点；③ 跑通三层 smoke test。

## 入口

| 想看什么 | 打开 |
|---|---|
| **现在到底做什么、判据是什么** | **[`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)** ← 唯一权威执行依据 |
| 候选是怎么被筛出来的（论证与证据） | [`docs/`](docs/README.md) |
| 每个实验怎么跑、何时杀 | [`probes/`](probes/README.md) 与 [`notes/prereg/`](notes/prereg/) |
| 决策历史 | [`notes/decision_log.md`](notes/decision_log.md) |

## 当前结论（摘要）

- **主线 1 条**：#6 投机算力分配（Speculative Compute Allocation）。
- **备线 2 条**：`#12`（MoE EP 对冲；先过 ≤3 天残余不均衡判定）、`#7`（per-adapter KV 配额，待探针判定）。
- **暂缓 1 项**：`#4` —— 复核门不通过（`notes/gap_hybrid_state.md`，带重开触发）。
- **主线定位收紧**：`#6` 只主张 **drafter 网络自身**的层数/宽度弹性（步数自适应已 ship，见 `notes/p06-family-codecheck.md`）。
- **判定探针 2 个**：#1 E1（1 天，唯一的有效否决权）、#3 约束解码扫描（1 周，低概率彩票）。
- **归档 14 项 + 暂缓 1 项**（20 = 14 归档 + 1 暂缓 + 5 存活），其中 #2 的归档理由是**基线反证**（`TTL-300s` 与 `LRU-leaf` 逐字节相同），不是"被谁占了"。

## 目录导览

```
EXECUTION_PLAN.md   唯一权威执行依据（门槛 / 清单 / W1-W3 / 判据 / 引用纪律）
README.md           本文件
docs/               方案与评估文档（5 份上游）+ evidence/ 证据扫描 + reviews/ 评审记录
probes/             实验 runbook：p01-e1 / p03-grammar-scan / p04-hybrid-gate / p06-frontier / p07-adapter-dispersion / aux-c1,c2（命名见 docs/GLOSSARY.md）
notes/              decision_log.md（决策历史）+ prereg/（预登记杀判据）
results/            实验产物（默认不入库，只保留 README 与 .gitkeep）
upstream/           引擎源码检出位置（不入库；快照见 archive/）
archive/            【冻结证据库，~2.2 GB，不入库】原始抓取、引擎快照、trace 数据库
subfield_scan/      【冻结，不入库】子领域扫描工作副本（结论副本已入 docs/evidence/）
prior_art_fetch/    【冻结，不入库】vLLM / SGLang 源码快照 tar
```

## 三条门槛（入选规则）

1. **机制格未被占**：没有 MERGED PR，且同一决策上没有 ≥2 个并行开放提案。
2. **≤2 卡 / ≤2 周能拿到可信数字**。
3. **量级来自实测或引擎常量**（① ② 对所有条目必过；③ 对主线与备线必过；探针可零量级入场，但必须带预登记判据与 ≤2 周换实测的路径）。

## 纪律（违反即返工）

- **不得引用 "~20%"**（`#43559` 实测 −0.67% / −2% / −4.8%）。
- **不得主张"没人搬草稿状态"**（SwiftSpec `2506.11309`、StarSD `2601.21622` 已占）。
- **引用 issue/PR 必须带 repo 前缀**；**判断占位必须读正文，不得用摘要**。
- **`docs/kimi_plan.md` 的三条"引用不实"裁定已作废**（SpecTool/AngelSpec/LibraSpec 三条均为误判，见 `EXECUTION_PLAN.md` §10）。

## 快速开始

```bash
# 1) 读执行方案
less EXECUTION_PLAN.md

# 2) 跑今天的探针（runbook）
less probes/p01-e1-uncommitted-kv/README.md

# 3) 产物落到 results/<probe>/<date>/，结论回写 notes/decision_log.md
mkdir -p results/p01-e1-uncommitted-kv/$(date +%F)
```

## 体积与版本控制

- 入库：`EXECUTION_PLAN.md`、`README.md`、`docs/`、`probes/`、`notes/`、`results/README.md`。
- 不入库：`archive/`（2.2 GB：源码快照、160 MB trace 数据库、DuckDB 二进制）、`subfield_scan/`、`prior_art_fetch/`、`upstream/`、`results/` 产物。规则见 [`.gitignore`](.gitignore)。
- 远端：`git@github.com:LeiHenan/myCCFA.git`（分支 `main`）。
