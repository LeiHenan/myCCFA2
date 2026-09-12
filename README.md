# myCCFA — MLSys 选题工作区

本工作区用于**选定并执行一条 MLSys 论文主线**。当前阶段：**⏸ 停工等新机器** —— 无卡工作（文献/引擎/仓库审计）已收尾，但原 8×4090 服务器**不可用于主线实验**：其驱动 **550.67（CUDA 12.4）** 无法运行 **CUDA 13 构建**的 vLLM ≥0.28，而主线 `#6` 所需的 `DFlash2DraftModel` 恰好只从 vLLM 0.28.0 起存在；该机无 root，无法升级驱动。**选机规格与验收脚本见 [`docs/MACHINE_REQUIREMENTS.md`](docs/MACHINE_REQUIREMENTS.md)。**

**解冻顺序（按实验设计的依赖；**不是"三选一"**）**：

1. ✅ **② 已完成（2026-09-12，pin = vLLM main `9a35c08`）** —— 结论：**通过**。(i) `e1_patch.py` 锚点在该 pin 上**全部命中，无需修改**；(ii) 多层 drafter **可得**（注册表含 `DFlashDraftModel`/`DFlash2DraftModel`，`qwen3_dspark.py` 存在；HF 有 **`Qwen3-4B-speculator.dflash2`** 等 6 个权重）；(iii) 深度由 config 的 `num_hidden_layers` 驱动（训练侧 `--num-layers`）。详见 [`notes/p06-toolchain-check.md`](notes/p06-toolchain-check.md)。
   **两种结局都是净收益**：有 ⇒ 卡型按显存需求定；**没有 ⇒ `#6` 降级**（深度旋钮不存在），卡型退到 24 GB 即可，主线改为 E1 + `#7`。
2. **① 换机器（现在可做）**：按 [`docs/MACHINE_REQUIREMENTS.md`](docs/MACHINE_REQUIREMENTS.md) 选机 —— **硬门槛：驱动 ≥580（CUDA 13）**；卡 **≥24 GB 且 Ada/Hopper（FP8 KV 需 sm_89+）**；数据盘可用 **≥200 GB**。拿到机器先跑 `bash docs/acceptance_check.sh`（两条硬门槛：驱动 ≥580、`DFlash2DraftModel` 被引擎注册）。配置 **1×24 GB 起步** —— 配 `Qwen3-4B` + `Qwen3-4B-speculator.dflash2`；128k 格需 `--kv-cache-dtype fp8`。**只有**需要 8B target 或 bs>1 时才要 48/80 GB。显存算术见 `notes/p06-toolchain-check.md` §5。
3. **③ smoke test**（需卡）：三层验证（无投机 → `method:"ngram"` → 真 drafter）。EAGLE-3 加载失败**不影响 E1 的结论有效性**，但须在 `summary.md` 标注 drafter 家族。

## 入口

| 想看什么 | 打开 |
|---|---|
| **现在到底做什么、判据是什么** | **[`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)** ← 唯一权威执行依据 ｜ 解冻顺序见本文 §"解冻顺序" |
| 候选是怎么被筛出来的（论证与证据） | [`docs/`](docs/README.md) |
| 每个实验怎么跑、何时杀 | [`probes/`](probes/README.md) 与 [`notes/prereg/`](notes/prereg/) |
| **一步步怎么操作**（命令/注入点/回退表） | [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md) |
| **换机器前**（配置规格 + 验收脚本 + 降级方案） | [`docs/MACHINE_REQUIREMENTS.md`](docs/MACHINE_REQUIREMENTS.md) ｜ `bash docs/acceptance_check.sh` |
| **上机第一步**（服务器验收 + 卡分配 + smoke test） | [`docs/SERVER_BOOTSTRAP.md`](docs/SERVER_BOOTSTRAP.md) |
| 决策历史与评审记录 | [`notes/decision_log.md`](notes/decision_log.md) ｜ [`docs/reviews/`](docs/reviews/README.md) |

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
docs/               方案与评估文档（5 份上游）+ evidence/ 证据扫描 + reviews/ 评审记录 + 上机（MACHINE_REQUIREMENTS / SERVER_BOOTSTRAP / EXPERIMENT_GUIDE）
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
