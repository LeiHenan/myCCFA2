# docs/ — 方案与评估文档

这 5 份文档是 **`EXECUTION_PLAN.md` 的上游论证**。它们的**排序结论已不再作为执行依据**（执行依据见根目录 `EXECUTION_PLAN.md`），但其中的证据、引用与逐项论证仍然有效，改动前请先读它们的"不可靠之处"一栏。

## 阅读顺序

| 顺序 | 文档 | 是什么 | 不可靠之处（已核实） |
|---|---|---|---|
| 1 | [`V41.md`](V41.md) | 本轮幸存候选（A/B/C 三级 23 项）与各自增量 | 量级多为算术值；B10 的"服务端机制未被占"已被推翻；**A3（=#12）的 `3.28×` 出自读图投影**（IF 为构造轴，见 `docs/evidence/` 之外的 `archive/subfield_scan/rea1/k3/FORENSIC_REPORT.md`） |
| 2 | [`GPT.md`](GPT.md) | 对 GPT 提案的审核（统一标准重判）+ 标准一致性自审 | 已自我纠错（§7 只读摘要下判断）；其"引擎增量 2–3%"仅适用于工具投机方向 |
| 3 | [`plan.md`](plan.md) | V41 × GPT 综合评估（创新性/可行性/文献综述三维 + 六周规划） | **#2 排在并列第 2 是错的**（Continuum/SAGA 已占；真正杀手是基线反证）；#6 的"从未调过深度"过强（DSpark 已扫层数）；**#12 的"证据最硬"不成立**（投影/下界/极端尾巴，且机制已被 SGLang LPLB 占） |
| 4 | [`kimi_plan.md`](kimi_plan.md) | 独立再校准（20 项重排 + MLSys 契合度 + 量级证据等级） | **§1/§7 的三条"引用不实"裁定全部作废**；排序键不可复现（主键是自由文本、次级键被违反）；宣称的"三个检索通道"无产物；其 #12 的 `3.28×`、`80×` 亦属未核实的转述 |
| 5 | [`old.md`](old.md) | 引用纪律、判据框架、击杀台账（47 条击杀 / 38 条越界） | 仍有效；是本工作区纪律条款的来源 |

## 上机与选机（**执行侧**，与上面的"论证侧"分开）

| 文档 | 用途 |
|---|---|
| [`MACHINE_REQUIREMENTS.md`](MACHINE_REQUIREMENTS.md) | **选机规格书**：驱动 ≥580 的事实链、显存/磁盘预算、框架与权重清单（Tier A/B）、选机对照表 |
| [`acceptance_check.sh`](acceptance_check.sh) | **机器验收脚本**：`bash docs/acceptance_check.sh`（两条硬门槛：驱动 ≥580、`DFlash2DraftModel` 被引擎注册） |
| [`SERVER_BOOTSTRAP.md`](SERVER_BOOTSTRAP.md) | **上机之后**：环境初始化 + 卡分配 + 三层 smoke test + 结果回写纪律 |
| [`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md) | 命令级操作细节（注入点 / 回退表） |

## 评审记录

- [`reviews/2026-09-11-verdict-01-ranking-review.md`](reviews/2026-09-11-verdict-01-ranking-review.md) — 对 `plan.md` 与 `kimi_plan.md` 的独立裁决（结论：plan.md 排序水平明显更高）。

后续评审按 `reviews/YYYY-MM-DD-verdict-NN-<slug>.md` 追加。

## 证据扫描（副本）

[`evidence/`](evidence/) 放的是本轮**承重结论**的三份扫描副本（原件在未入库的 `archive/` 与 `subfield_scan/`，2026-09-11 冻结）。任何"某方向空缺/已死"的判断都应回到这三份文件核对，而不是凭记忆。

## 路径映射（旧 → 新）

| 旧路径 | 新路径 |
|---|---|
| `plan.md` | `docs/plan.md` |
| `kimi_plan.md` | `docs/kimi_plan.md` |
| `V41.md` | `docs/V41.md` |
| `GPT.md` | `docs/GPT.md` |
| `old.md` | `docs/old.md` |
| `review_opinion.txt` | `docs/reviews/2026-09-11-verdict-01-ranking-review.md` |
| `subfield_scan/r4_graded_kv_tiers.md` | 副本 → `docs/evidence/r4_graded_kv_tiers.md`（原件仍在 `subfield_scan/`） |
| `subfield_scan/r4_kv_prefetch.md` | 副本 → `docs/evidence/r4_kv_prefetch.md`（原件仍在 `subfield_scan/`） |
| `archive/subfield_scan/constrained_decoding.md` | 副本 → `docs/evidence/constrained_decoding.md`（原件仍在 `archive/`） |

> ⚠️ `archive/` 内的历史文档大量以**旧相对路径**引用上述文件（例如 `V41.md`、`data/syfi_coding_trace.duckdb`）。`archive/` 是审计痕迹，**不追改**；需要引用时以本表换算。
