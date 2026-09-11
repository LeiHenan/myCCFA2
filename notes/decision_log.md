# 决策日志

**规则**：任何方向的**生**与**死**都要记一条，附依据与证据位置。半年前的自己会感谢现在的自己。

---

## 2026-09-11 — 选题流程（上游轮次，归档备查）

| # | 决策 | 依据 | 产物 |
|---|---|---|---|
| 1 | 一轮流程提名并击杀 22 个方向；对全部击杀理由做对抗性复审 | 47 条可分离击杀理由：**38 条越界 (81%)**、6 条有效 | `docs/old.md` |
| 2 | 幸存候选 23 项（A/B/C 三级） | 各自增量 + 便宜判定实验 | `docs/V41.md` |
| 3 | GPT 的 9 个提案：**2 条 VALID KILL**（② K 改计算预算、④ KV placement 多 tier，均为"方向 B"）、5 条 NARROWED、4 条 UNDECIDED、1 条非命题 | 统一标准重判 + 引擎 tracker 拥挤度 | `docs/GPT.md` |
| 4 | 两份清单合并排序（20 候选 + 9 探针） | 三维评估（创新性 / 可行性 / 文献综述） | `docs/plan.md` |
| 5 | 独立再校准（20 项重排，5 项评级变化） | 声称的"三个检索通道" | `docs/kimi_plan.md` |

## 2026-09-11 — 评审轮

| # | 决策 | 依据 | 产物 |
|---|---|---|---|
| 6 | **裁决：`plan.md` 的排序水平明显更高**（kimi_plan 排序键不可复现；其 3 条"引用不实"裁定全部作废） | 逐条复核承重声明（本地证据库 + 实时网络 + 文件时间戳） | `docs/reviews/2026-09-11-verdict-01-ranking-review.md` |
| 7 | 采纳 kimi 的两处真修正：#2 降档、#6 的 DSpark/Graft 边界；丢弃其三条引用裁定 | 同上 | 同上 |

## 2026-09-11 — 终局轮（本工作区建立）

| # | 决策 | 依据 | 影响 |
|---|---|---|---|
| 8 | 确立**三条硬门槛**（机制格未被占 / ≤2 卡≤2 周 / 量级来自实测或引擎常量），并明确 ③ 对主线备线必过、探针对探针可零量级入场 | 修正"规则生成不了表格"的缺陷 | `EXECUTION_PLAN.md` §1 |
| 9 | **关闭 `arXiv 2606.29223`**：= DEX《Depth Exploration for LLM Decoding》，**不占 #1**（目标模型层深轴；无 commit 概率 → KV 分层映射） | 读到 v1 全文 | #1 的 E1 可开跑；DEX 转为 #6 的邻近先例 |
| 10 | **#2 归档**，且理由由"被占"更正为**基线反证**（`TTL-300s` 与 `LRU-leaf` 逐字节相同；`LRU-leaf` 比文献以为的强） | 本地负面结果 + `rea_4_legacy.md` 自给 band 10–30% | 归档 14 项 |
| 11 | **#6 升为主线**，定位改为"投机算力分配"，并加 **DEX 替换测试** + **drafter 家族层数约束**（EAGLE-3 = 1 层 ⇒ 无深度旋钮） | DSpark/Graft/DEX 三处证据 | `probes/p06-frontier/` |
| 12 | **#4 收窄**为"多分支 SSM 快照内存放大 + 淘汰"，并加 2 小时复核门 | 引擎侧已被合并/在办（`#53614` 等） | `probes/p04-hybrid-gate/` |
| 13 | **#12 加 2 周时间盒**，到期不达标即归档 | 单点转化率、需多卡 EP | §7 |
| 14 | 工作区重构：文档归 `docs/`、探针归 `probes/`、预登记归 `notes/prereg/`、产物归 `results/` | 便于执行与复现 | 本仓库结构 |
| 15 | 新增 `docs/EXPERIMENT_GUIDE.md`（W1–W3 逐步操作手册） | 方案需要可执行的命令级指导 | `docs/EXPERIMENT_GUIDE.md` |
| 16 | **E1 判据新增硬件条件分支**：24 GB 卡（4090/5090）路径下 ctx 网格 {4k,32k,128k} → **{4k,16k,32k}**（或换 ≤4B target / 2×24 GB TP=2） | 显存算术：8B 级 ≈128 KiB/token ⇒ 单请求 128k ctx ≈16 GiB，24 GB 卡不可行；且比值 `≈γ/ctx` 在短上下文信噪比更高 | 本分支在**任何数据采集之前**登记，不构成事后放宽 |
| 17 | **统一命名**：实验目录 `probes/pNN-*`（NN = 候选 #NN）、辅助探针 `probes/aux-cN-*`；新增 `docs/GLOSSARY.md` | 消除 `c3_constrained_scan` 与辅助表 C3 的语义冲突 | 全仓库路径已同步 |
| 18 | **补入 #7**（per-adapter KV 配额）为**备线 C（待判定）**，配 ≤1 周离散度探针 | 上一版 §2 存活清单漏列 #7，导致 14+5=19 账不平 | 20 = 14 归档 + 6 存活 |
| 19 | **#4 复核门不通过 → 降为「暂缓」**（无卡，2 小时） | 主张的 gap 是已占证据的子集：ATOM（shipped）已实装 ladder + 需求驱动 rung + LRU 淘汰 + 成本记账 + 每请求 `1+num_speculative_tokens` 回滚 slot；共存半场由 vLLM `#50172`（OPEN，交付物=正确性）在办 | `notes/gap_hybrid_state.md`；重开触发①`#50172` 关闭或停滞 ②其落地后暴露结构性缺陷 |
| 20 | **#6 代码层家族检查通过，但定位收紧**（无卡） | 多层 drafter 存在且层数可配置（`multi_layer_eagle_*`、`dflash.py` 的 `draft_config.num_hidden_layers`）；但 `adaptive_spec_params.py` 已 ship "runtime 调整 `speculative_num_steps`" + 多套 CUDA-graph 原子切换 ⇒ 不再以"算力分配"泛称，只主张 **drafter 网络深度/宽度** | `notes/p06-family-codecheck.md`；备线重排 A=`#12`、B=`#7`；20 = 14 归档 + 1 暂缓 + 5 存活 |
| 21 | **#6 预登记在数据采集前收紧**（无卡）：新增基线 ④ `adaptive_spec_params`、三个必答项、`adaptive-steps-only` 增量对照、两条杀判据；新增与 DEX 的三轴区分声明 | 新证据：已 ship 的自适应控制器**逐字拒绝多层 worker**（`MultiLayerEagleWorkerV2 does not implement adaptive`）⇒ 深度轴的空白更明确，但也要求结构性理由而非 plumbing | `notes/prereg/p06-frontier.md`（修订记录已标注"数据采集前"）、`notes/p06-dex-differentiation.md` |
| 22 | **采纳外部调研的可核部分，主假设升级为正交性**：`#6` 的头条主张改为「drafter capacity ⟂ draft length」；网格加 γ 因子；No-Go 加"共线即杀"；基线补 vLLM per-batch K 查表 | 该报告的区分经核实为真（MemSpec=draft residency；DSpark 有上游实现与 HF 权重 ⇒ 多层 drafter 可取）；其 MLSys '26 链接 404 但论文真实 ⇒ §11 缺口由「未覆盖」改为「有入口」 | `notes/prereg/p06-frontier.md`、`EXECUTION_PLAN.md` v1.4、`docs/reviews/2026-09-11-verdict-02-*.md`；**#6 维持主线** |

## 待决策（按到期顺序）

| 截止 | 决策 | 触发条件 |
|---|---|---|
| D1 结束 | #1 生或死 | E1 判据（`probes/p01-e1-uncommitted-kv/README.md` §5） |
| W1 结束 | #6 是否具备实验条件 | 是否存在可跑的多层并行 drafter |
| W1 结束 | #3 归档或继续 | class-4 @batch≥32 是否 ≤0.85× 且结构性 |
| W2 结束 | #4 归档或升为备线 A | 复核门 gap 声明 |
| ≤2 周 | #12 归档或保留 | 是否区分 DA-MoE / 转化率是否上升 |
| W3 结束 | **锁定 1 主线 + 1 备线** | 量级 ≥8%（自测 vs 部署基线）且机制可归因 |
