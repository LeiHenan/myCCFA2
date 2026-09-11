# myCCFA — MLSys 选题工作区

本工作区用于**选定并执行一条 MLSys 论文主线**。当前阶段：**W1（判定探针并行期）**。

## 入口

| 想看什么 | 打开 |
|---|---|
| **现在到底做什么、判据是什么** | **[`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)** ← 唯一权威执行依据 |
| 候选是怎么被筛出来的（论证与证据） | [`docs/`](docs/README.md) |
| 每个实验怎么跑、何时杀 | [`probes/`](probes/README.md) 与 [`notes/prereg/`](notes/prereg/) |
| 决策历史 | [`notes/decision_log.md`](notes/decision_log.md) |

## 当前结论（摘要）

- **主线 1 条**：#6 投机算力分配（Speculative Compute Allocation）。
- **备线 2 条**：#4 收窄版（多分支 SSM 快照内存放大 + 淘汰，需过复核门）、#12（MoE EP 不均衡，非投机对冲，2 周时间盒）。
- **判定探针 2 个**：#1 E1（1 天，唯一的有效否决权）、#3 约束解码扫描（1 周，低概率彩票）。
- **归档 14 项**，其中 #2 的归档理由是**基线反证**（`TTL-300s` 与 `LRU-leaf` 逐字节相同），不是"被谁占了"。

## 目录导览

```
EXECUTION_PLAN.md   唯一权威执行依据（门槛 / 清单 / W1-W3 / 判据 / 引用纪律）
README.md           本文件
docs/               方案与评估文档（5 份上游）+ evidence/ 证据扫描 + reviews/ 评审记录
probes/             实验 runbook：e1 / c3 / frontier_drafter / gate_hybrid_state
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
less probes/e1_uncommitted_kv/README.md

# 3) 产物落到 results/<probe>/<date>/，结论回写 notes/decision_log.md
mkdir -p results/e1_uncommitted_kv/$(date +%F)
```

## 体积与版本控制

- 入库：`EXECUTION_PLAN.md`、`README.md`、`docs/`、`probes/`、`notes/`、`results/README.md`。
- 不入库：`archive/`（2.2 GB：源码快照、160 MB trace 数据库、DuckDB 二进制）、`subfield_scan/`、`prior_art_fetch/`、`upstream/`、`results/` 产物。规则见 [`.gitignore`](.gitignore)。
- 远端：`git@github.com:LeiHenan/myCCFA.git`（分支 `main`）。
