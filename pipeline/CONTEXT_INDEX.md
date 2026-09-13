# 本工作区内容地图（整理结果）—— 每个目录在选题 pipeline 里的位置

**日期**：2026-09-13 ｜ **配套**：[`PIPELINE.md`](../PIPELINE.md)（流程）、[`pipeline/gates.json`](gates.json)（判据）
**这份文件解决一个问题**：这个文件夹里堆了几十份文档、探针、结果、扫描报告，**哪些是"选题的输入"、哪些是"执行期的产物"、哪些只是历史**。查这张表，不要靠猜。

## 1. 目录 → 阶段映射

| 路径 | 是什么 | 在 pipeline 里的位置 | 状态 |
|---|---|---|---|
| `PIPELINE.md` | **选题流程控制器**（Agent 操作手册） | 入口 | 现行 |
| `pipeline/gates.json` | **机器可读判据**（小节/清单/杀出口） | 判据事实源 | 现行 |
| `pipeline/check.py` · `new_candidate.py` · `lib.py` | 校验器 / 脚手架 / 公共库 | 执行工具 | 现行 |
| `candidates/` | **新候选方向的档案目录**（每方向一个子目录） | S0–S5 产物 | 新建 |
| `docs/MACHINE_REQUIREMENTS.md` · `docs/acceptance_check.sh` | 选机规格与验收脚本（驱动/CUDA/显存/磁盘） | **S0 输入**（算力预算） | 现行 |
| `docs/TOPIC_METHODOLOGY.md` | 原理与实证索引（真痛点四条件 / 伪问题七形态 / 十问 / 纪律 / 模板） | **S1–S2 判据来源** | 现行 |
| `docs/GLOSSARY.md` | 术语表（含"内部标签 ≠ 术语"的纪律） | 全程（协议 #8） | 现行 |
| `docs/evidence/*.md`（mlsys2026_scan / venue_scan_2026 / constrained_decoding …） | 会议与子领域扫描、占位证据 | **S1 搜证来源 + S3 占位核查** | 现行 |
| `subfield_scan/*.md` | 子领域专题扫描（KV 分级/预取） | S1/S3 输入 | 现行 |
| `prior_art_fetch/`（`fetch.py` / `ghsearch.py` / 引擎源码快照） | 先行工作抓取工具与源码快照 | **S3 工具**（读正文用） | 现行 |
| `upstream/` | vLLM 等上游锚点核对副本 | S3 工具 | 现行 |
| `notes/decision_log.md` | **决策记忆**（#1–#69，每条含决定/依据/影响文件） | **全程**：每阶段必须落一行 | 现行 |
| `notes/prereg/*.md` | 冻结的预登记副本（p01/p03/p06/p07） | **S5 产物** | 现行 |
| `notes/audit-p06-design-2026-09-13.md` | 在跑方向审计的方法样本（模式 C 的来源） | S4/S5/S7 反向审计范例 | 现行 |
| `notes/p06-*.md` | 单方向的家族/工具链/差异化笔记 | S3 产物样式 | 归档中 |
| `probes/pNN-<slug>/` | 探针代码（脚本 + 分析器，多含 `--selftest`） | **S6/S7 产物**（可复用为模板） | 现行 |
| `results/<probe>/<date>/` | 派生表与结论（`summary.md` / `*.csv` / `bench.json`） | S6–S8 产物（**原始数据在仓库外**） | 现行 |
| `evidence/<date>/` | 关键原始日志片段（serve log 等） | S7 归因证据 | 现行 |
| `EXECUTION_PLAN.md` | 当前主线的执行方案（版本化变更记录） | 已选方向的执行 | 现行 |
| `docs/plan.md` · `kimi_plan.md` · `V41.md` · `GPT.md` | **四份冻结的上游论证文档** | **只作论证与证据记录，不提供执行指令** | 冻结 |
| `archive/` | 早期筛选的临时产物（a1_screen / 抓取原文 / 校验脚本） | 历史 | 归档 |
| `docs/reviews/` | 早期评审与排名复核 | 历史 | 归档 |
| `docs/old.md` · `docs/SERVER_BOOTSTRAP.md` · `docs/EXPERIMENT_GUIDE.md` | 纪律条款 / 上机手册 / 实验指南 | 执行期参考 | 现行 |

> **两条硬规矩**：① 四份冻结上游文档**不得**作为执行依据（见 `EXECUTION_PLAN.md` 抬头与 `docs/SERVER_BOOTSTRAP.md` 冻结横幅）；② 实验原始数据一律在仓库外（本工作区 `/root/ccfa_results/<date>/`），入库的只有派生表与结论。

## 2. `#6` 回溯案例：同一个方向按 pipeline 走一遍会怎样

这是本文件最有用的部分——**用已经跑完的 `#6` 当教材**，标出每个阶段当时做对/做错了什么。

| 阶段 | `#6` 当时实际做的 | 判定 | 出处 |
|---|---|---|---|
| **S0** | 约束量化过（选机按驱动 ≥580 / CUDA 13 / 24 GB 起步；后换 96 GB 卡） | ✅ 做对 | decision #39、`docs/MACHINE_REQUIREMENTS.md` |
| **S1** | 痛点来源：DFlash 全上下文重扫实测 16 vs 71 tok/s、引擎内深度轴空白 | ⚠️ 部分：**没有先量级取证"深度不匹配"到底有多疼** | decision #37 |
| **S2** | 未做正式的七形态 + 十问筛查（当时没有这套流程） | ❌ 没做 | `docs/TOPIC_METHODOLOGY.md` §1.2 是事后补的 |
| **S3** | 查到"引擎内 drafter 深度轴空白"就继续 | ❌ **没有预判 γ 轴已被 ship + RFC 推进对增量上限的含义**；深度轴还被 HELIOS 等 early-exit 挤压 | decision #62 |
| **S4** | **没做**。T2 的 oracle 上界事后算出来只有 **+0.1%**（且只在 `bs=1`） | ❌ **致命**：若 S4 做了，方向在选题阶段就被拦下 | decision #53 |
| **S5** | 预登记做得较全（判据/杀条件/扩展条款），但 **γ 网格 {1,3,5,7} 实际只跑 {3,7}** | ⚠️ 网格与实际执行未对账；缺的正是唯一可能反转的角落 | decision #68 |
| **S6** | T0 ✅ 结局 A；T1 斜 ridge 但幅度在噪声内；T2 +0.1% ≪ 8% | ⚠️ 结论下得比证据早（后被降级为"条件性阴性"） | decision #53/#55 |
| **S7** | **做得最好**：16/16 格 `preemption=0`；逐位置接受率显示"1 层 drafter 从第 2 位置起接受率为 0"⇒ **结构性否证**（接受率是悬崖不是权衡，最优恒在边界） | ✅ 教科书式的负面结论 | decision #67 |
| **S8** | 结题 12 项已走，产物入库，负面结论保留并标注适用范围 | ✅ | `results/p06-frontier/**`、decision #64–#68 |

**这次回溯的结论（也就是 pipeline v1.0 为什么长这样）**：
1. **S4 是省钱的闸门**：`#6` 真正的死因在 S4 就该被发现（oracle 上界 +0.1%），而不是花完 GPU 后才发现。
2. **S5 的"网格对账"要写成硬要求**：预登记与实际执行的差异会静默吃掉唯一的关键角落。
3. **S2 不能省**：`#6` 的立项理由里混着"没人做过"的成分，这正是七形态里第一条。
4. **S7 做得好能让负面结论有价值**：把一个"没测出反转"的失败，变成"接受率悬崖 ⇒ 该轴不存在权衡"的可迁移结论。

## 3. 下次选题时最短的操作路径

```bash
# 1) 建档案（0 GPU）
python pipeline/new_candidate.py --slug <新方向短名> --title "<中文标题>"
# 2) 先只填 S0 的 6 项约束，并核一遍
python pipeline/check.py --dir candidates/<新方向短名> --through S0
# 3) 复用本仓库已有的输入，不要从零开始：
#    - 会议/子领域扫描  → docs/evidence/、subfield_scan/
#    - 占位与读正文工具 → prior_art_fetch/、upstream/
#    - 算力与卡型约束   → docs/MACHINE_REQUIREMENTS.md
#    - 判据与伪问题清单 → docs/TOPIC_METHODOLOGY.md
#    - 决策记忆         → notes/decision_log.md（先搜一遍，别重复踩坑）
# 4) 每阶段结束：check.py --through S<n> + 在 decision_log 加一行 + 推 GitHub
```
