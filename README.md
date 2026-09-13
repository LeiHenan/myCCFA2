# myCCFA — MLSys 选题工作区

本工作区用于**选定并执行一条 MLSys 论文主线**。当前阶段（2026-09-13）：**选题 pipeline 就绪；主线 `#6` 已按预登记判据结题**。

- **`#6`（投机算力分配）= 机制层被结构性否证**：`depth* = d5` 在全部 8 个 `(bs,γ)` 组合上一致（稳态口径差距 26.3–31.8%，Welch p<0.05）；机制原因是**逐位置接受率悬崖**（1 层 drafter 从第 2 个位置起接受率为 0）叠加**drafter 每步代价只占 +5.9%** ⇒ 深度轴上不存在"浅↔深"权衡曲线，最优恒在可行域上界 ⇒ 判据②（内点最优）**结构性不可能满足**。判据⑦ 干净通过（16/16 格 `preemption = 0`）。详见 [`notes/audit-p06-design-2026-09-13.md`](notes/audit-p06-design-2026-09-13.md) 与 `notes/decision_log.md` #64–#68。
- **因此下一步是重新选题**，用本工作区新固化的 pipeline 走一遍 ⇒ **[`PIPELINE.md`](PIPELINE.md)**。

## 入口

| 想看什么 | 打开 |
|---|---|
| **要选新课题 / 让 Agent 带着选**（9 阶段流程 + 判据 + 校验器 + 脚手架） | **[`PIPELINE.md`](PIPELINE.md)** ← **最上位入口** ｜ 判据表 `python pipeline/check.py --list` |
| **本工作区内容地图**（哪个目录在哪个阶段用）+ `#6` 回溯案例 | [`pipeline/CONTEXT_INDEX.md`](pipeline/CONTEXT_INDEX.md) |
| 选题的**原理与实证索引**（真痛点四条件 / 伪问题七形态 / 十问 / 判定纪律） | [`docs/TOPIC_METHODOLOGY.md`](docs/TOPIC_METHODOLOGY.md) |
| **现在到底做什么、判据是什么** | **[`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)** ← 唯一权威执行依据 |
| 候选是怎么被筛出来的（论证与证据） | [`docs/`](docs/README.md) |
| 每个实验怎么跑、何时杀 | [`probes/`](probes/README.md) 与 [`notes/prereg/`](notes/prereg/) |
| **一步步怎么操作**（命令/注入点/回退表） | [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md) |
| **换机器前**（配置规格 + 验收脚本 + 降级方案） | [`docs/MACHINE_REQUIREMENTS.md`](docs/MACHINE_REQUIREMENTS.md) ｜ `bash docs/acceptance_check.sh` |
| **上机第一步**（服务器验收 + 卡分配 + smoke test） | [`docs/SERVER_BOOTSTRAP.md`](docs/SERVER_BOOTSTRAP.md) |
| 决策历史与评审记录 | [`notes/decision_log.md`](notes/decision_log.md) ｜ [`docs/reviews/`](docs/reviews/README.md) |

## 当前结论（摘要）

- **主线 `#6`：结题（机制层否证，现象层覆盖缺口已补/在补）** —— 见上。
- **`#1` E1：判死**（`R_byte` p95 0.072% ≪ 5% 门槛，12 格两次运行逐位一致）；**D3 支线关闭**（正当口径 1.0053，无 headroom）。
- **备线**：`#12`（MoE EP 对冲；先过 ≤3 天残余不均衡判定）、`#7`（per-adapter KV 配额，待离散度探针判定）。
- **暂缓**：`#4` —— 复核门不通过（`notes/gap_hybrid_state.md`，带重开触发）。
- **归档 14 项**，其中 #2 的归档理由是**基线反证**（`TTL-300s` 与 `LRU-leaf` 逐字节相同），不是"被谁占了"。

## 目录导览

```
EXECUTION_PLAN.md   唯一权威执行依据（门槛 / 清单 / W1-W3 / 判据 / 引用纪律）
README.md           本文件
docs/               方案与评估文档（5 份上游）+ evidence/ 证据扫描 + reviews/ 评审记录 + 上机（MACHINE_REQUIREMENTS / SERVER_BOOTSTRAP / EXPERIMENT_GUIDE）
PIPELINE.md         选题 pipeline（Agent 操作手册：9 阶段 S0–S8 + 操作协议 + 交互契约）
pipeline/           判据与工具：gates.json（判据事实源）/ check.py（校验器）/ new_candidate.py（脚手架）/ CONTEXT_INDEX.md（内容地图 + #6 回溯案例）
candidates/         候选方向档案（每方向一目录，S0–S5 产物；骨架由脚手架生成）
EXECUTION_PLAN.md   唯一权威执行依据（门槛 / 清单 / W1-W3 / 判据 / 引用纪律）
README.md           本文件
docs/               方案与评估文档（5 份上游）+ evidence/ 证据扫描 + reviews/ 评审记录 + 上机（MACHINE_REQUIREMENTS / SERVER_BOOTSTRAP / EXPERIMENT_GUIDE）
probes/             实验 runbook：p01-e1 / p03-grammar-scan / p04-hybrid-gate / p06-frontier / p07-adapter-dispersion / aux-c1,c2（命名见 docs/GLOSSARY.md）
notes/              decision_log.md（决策历史）+ prereg/（预登记杀判据）+ audit-*（设计审计）
results/            实验产物（派生表与结论入库；原始数据在仓库外 /root/ccfa_results/）
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
# 0) 要选课题：先用 pipeline 建档案（0 GPU），不要直接开跑实验
python pipeline/new_candidate.py --slug <新方向短名> --title "<中文标题>"
python pipeline/check.py --list                       # 看判据表
python pipeline/check.py --dir candidates/<短名> --through S0

# 1) 读执行方案（已选定方向怎么跑）
less EXECUTION_PLAN.md

# 2) 跑探针（runbook）
less probes/p06-frontier/README.md

# 3) 产物落到 results/<probe>/<date>/（原始数据在仓库外），结论回写 notes/decision_log.md
```

## 体积与版本控制

- 入库：`PIPELINE.md`、`pipeline/`、`candidates/`、`EXECUTION_PLAN.md`、`README.md`、`docs/`、`probes/`、`notes/`、`results/README.md` 与 `results/**` 的派生表/小结（`*.md`/`*.csv`/`*.bench.json`）。
- 不入库：`archive/`（2.2 GB：源码快照、160 MB trace 数据库、DuckDB 二进制）、`subfield_scan/`、`prior_art_fetch/`、`upstream/`、`results/` 产物。规则见 [`.gitignore`](.gitignore)。
- 远端：`git@github.com:LeiHenan/myCCFA.git`（分支 `main`）。
