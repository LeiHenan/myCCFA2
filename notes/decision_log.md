# 决策日志

**规则**：任何方向的**生**与**死**都要记一条，附依据与证据位置。半年前的自己会感谢现在的自己。
**时段**：第 1–5 条 = 上游选题流程；6–7 = 评审轮；8 起 = 本工作区建立后。

---

## 一、选题流程（上游轮次，归档备查）

| # | 决策 | 依据 | 产物 |
|---|---|---|---|
| 1 | 一轮流程提名并击杀 22 个方向；对全部击杀理由做对抗性复审 | 47 条可分离击杀理由：**38 条越界 (81%)**、6 条有效 | `docs/old.md` |
| 2 | 幸存候选 23 项（A/B/C 三级） | 各自增量 + 便宜判定实验 | `docs/V41.md` |
| 3 | GPT 的 9 个提案：**2 条 VALID KILL**（② K 改计算预算、④ KV placement 多 tier，均为"方向 B"）、5 条 NARROWED、4 条 UNDECIDED、1 条非命题 | 统一标准重判 + 引擎 tracker 拥挤度 | `docs/GPT.md` |
| 4 | 两份清单合并排序（20 候选 + 9 探针） | 三维评估（创新性 / 可行性 / 文献综述） | `docs/plan.md` |
| 5 | 独立再校准（20 项重排，5 项评级变化） | 声称的"三个检索通道" | `docs/kimi_plan.md` |

## 二、评审轮

| # | 决策 | 依据 | 产物 |
|---|---|---|---|
| 6 | **裁决：`plan.md` 的排序水平明显更高**（kimi_plan 排序键不可复现；其 3 条"引用不实"裁定全部作废） | 逐条复核承重声明（本地证据库 + 实时网络 + 文件时间戳） | `docs/reviews/2026-09-11-verdict-01-ranking-review.md` |
| 7 | 采纳 kimi 的两处真修正：#2 降档、#6 的 DSpark/Graft 边界；丢弃其三条引用裁定 | 同上 | 同上 |

## 三、终局轮（本工作区建立后）

| # | 决策 | 依据 | 产物 |
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
| 18 | **补入 #7**（per-adapter KV 配额）为**备线 C（待判定）**，配 ≤1 周离散度探针 | 上一版 §2 存活清单漏列 #7，导致 14+5=19 账不平 | 20 = 14 归档 + 1 暂缓 + 5 存活（v1.2 起 #7 改称「备线 B」） |
| 19 | **#4 复核门不通过 → 降为「暂缓」**（无卡，2 小时） | 主张的 gap 是已占证据的子集：ATOM（shipped）已实装 ladder + 需求驱动 rung + LRU 淘汰 + 成本记账 + 每请求 `1+num_speculative_tokens` 回滚 slot；共存半场由 vLLM `#50172`（OPEN，交付物=正确性）在办 | `notes/gap_hybrid_state.md`；**重开触发**：① `#50172`/`#54637` **closed 未合并** 或连续 **>4 周无活动**；② 其落地后暴露**结构性**缺陷（非调参可解）|
| 20 | **#6 代码层家族检查通过，但定位收紧**（无卡） | 多层 drafter 存在且层数可配置（`multi_layer_eagle_*`、`dflash.py` 的 `draft_config.num_hidden_layers`）；但 `adaptive_spec_params.py` 已 ship "runtime 调整 `speculative_num_steps`" + 多套 CUDA-graph 原子切换 ⇒ 不再以"算力分配"泛称，只主张 **drafter 网络深度/宽度** | `notes/p06-family-codecheck.md`；备线重排 A=`#12`、B=`#7`；20 = 14 归档 + 1 暂缓 + 5 存活 |
| 21 | **#6 预登记在数据采集前收紧**（无卡）：新增基线 ④ `adaptive_spec_params`、三个必答项、`adaptive-steps-only` 增量对照、两条杀判据；新增与 DEX 的三轴区分声明 | 新证据：已 ship 的自适应控制器**逐字拒绝多层 worker**（`MultiLayerEagleWorkerV2 does not implement adaptive`）⇒ 深度轴的空白更明确，但也要求结构性理由而非 plumbing | `notes/prereg/p06-frontier.md`（修订记录已标注"数据采集前"）、`notes/p06-dex-differentiation.md` |
| 22 | **采纳外部调研的可核部分，主假设升级为正交性**：`#6` 的头条主张改为「drafter capacity ⟂ draft length」；网格加 γ 因子；No-Go 加"共线即杀"；基线补 vLLM per-batch K 查表 | 该报告的区分经核实为真（MemSpec=draft residency；DSpark 有上游实现与 HF 权重 ⇒ 多层 drafter 可取）；其 MLSys '26 链接 404 但论文真实 ⇒ §11 缺口由「未覆盖」改为「有入口」 | `notes/prereg/p06-frontier.md`、`EXECUTION_PLAN.md` v1.4、`docs/reviews/2026-09-11-verdict-02-*.md`；**#6 维持主线** |
| 23 | **全仓审计 + 三项修正**：① #6 写入**三级判定**（T0/T1/T2，决策级 3–4 天 ≠ 论文级 2 周）并标注**截断下界的不对称性**；② #12 写入**≤3 天残余不均衡判定门**，并更正其"证据最硬"标签（3.28× 投影 / 19% 下界 / 80× 极端尾巴；mechanism 已被 LPLB 占）；③ `__pycache__` 出库并加 ignore | 审计发现：1 条断链（`EXECUTION_PLAN → GLOSSARY.md`）、3 处陈旧表述（"证据最硬"×1、"14 归档 + 6 存活"×1、过时的待决策行）、pyc 误入库 | 见本轮 commit；`EXECUTION_PLAN.md` §2/§5/§7、`probes/p06-frontier/README.md`、`docs/README.md` |
| 24 | **MLSys '26 全量扫描完成，§11 该缺口关闭**（无卡）：`#6` 机制格未被占，但新增 **必答项 ④ 动机抗辩**（PRISM）与 HELIOS 的叙述挤压；`#12` 拥挤度获第三重证据 | 扫描 135/504；Speculative Decoding 分区 4 篇逐条核实；两条 oral 逐字取证 | `docs/evidence/mlsys2026_scan.md`；`notes/prereg/p06-frontier.md`（必答项 ④）；`EXECUTION_PLAN.md` v1.5 |
| 25 | **ASPLOS '26 + SOSP '26 扫描完成 + DA-MoE 正文精读**（无卡）：`#12` 占位证据升至五重以上（+ MorphKernel 1.3×、Barrier-Free EP[Tier B]、DA-MoE 实测 1.16×/1.29×），门槛加"先读 Barrier-Free EP"；`#6` 邻居加 TLT（ASPLOS '26，训练期 adaptive drafter + 按 batch 选策略，策略轴）；纠正 kimi 关于 DA-MoE"合入 FlashInfer"的未核实转述 | 逐篇核实 ASPLOS 152/1048 与 SOSP 62/390 的清单；DA-MoE 读 arXiv HTML 全文 | `docs/evidence/venue_scan_2026.md`；`EXECUTION_PLAN.md` §7/§11 与 v1.6；ATC/ISCA/SC '26 仍未扫 |
| 26 | **写入三条待决策项**（D1 测量论文降级形态 / D2 网格是否扩大 / D3 `R_reserve` 分支）+ D4 名称标签；§0 补"主线候选仅 `#6` 一条"的结构风险 | 用户决策所需；且 portfolio 空洞是当前最大结构性风险 | `EXECUTION_PLAN.md` §0/§12 与 v1.8 |
| 27 | **D1–D4 拍板**：D1 = (a) + 预登记的扩展条款（T1 固定 vLLM；扩展优先家族、不自动加引擎；降级须新预登记）；D2 = (a) 最小 18 格；D3 = (b) `R_reserve` 支线立项；D4 = 名称标为内部叫法 | D1 的具体化依据：SGLang 的 `adaptive_spec_params` 逐字拒绝多层 worker ⇒ 只有 vLLM 能同时提供"已 ship 的 K 自适应基线"与"多层 drafter"；家族扩展（DFlash↔DSpark 均已发布）远比引擎扩展便宜，而审稿人最可能问的是跨家族 | `EXECUTION_PLAN.md` §12 改为决策记录、v1.9；`notes/prereg/p06-frontier.md`（引擎固定 + 扩展条款）；`notes/prereg/p01-e1-uncommitted-kv.md`（D3 次要终点）|
| 28 | **审计修正 5 处**：§5.2 基线纪律与 D1 未同步（仍写 SGLang 基线）→ 改为 vLLM per-batch K 查表；澄清"两个 8%"（部署基线 vs 最佳固定配置）；README 入口表补 `docs/EXPERIMENT_GUIDE.md` 与 `docs/reviews/`；`upstream/README` 登记 `_snapshot` 锚点核对副本；清理 `results/_patchtest` | 新审计类别（仓库体积/孤儿文档/arXiv ID 格式/README 索引/草稿归属）中发现 1 处真不一致 + 2 处可发现性问题 | `EXECUTION_PLAN.md` v1.10、`README.md`、`upstream/README.md`、`notes/decision_log.md` |

| 29 | **独立核查外部审计报告 → 修正 7 类**：① `decision_log` 三节结构被上一轮"整块排序"破坏（标题被排到块尾、两节只剩表头）→ 重建；② §1 门槛② 明确"按判定阶段计"的卡数适用范围；③ §4 #3 归档条件"且"→"或"（原为 Go 判据补集的错写）；④ §5.2 必答项 ② 改为 vLLM 口径（SGLang 仅旁证）并把"三个"改为"四个"；⑤ §7 门改用真名 StreamEP + 加"正文不可得"fallback、删除未核实的"并入 FlashInfer"事实引用；⑥ §8.1 补 24 GB 分支与灰区判据；§8.2 语法类以 pre-reg 为唯一来源；⑦ 归档台账与日期锚点补入本日志 | 外部审计 21 条中经原文核对**确认 12 条为真**（含 2 条由我上一轮引入），其余为措辞/归属问题或已由既有批注覆盖 | `EXECUTION_PLAN.md` v1.11、`notes/decision_log.md`、`notes/prereg/p06-frontier.md`、`notes/prereg/p03-grammar-scan.md`、`notes/p06-family-codecheck.md` |

| 30 | **解冻条件改为有序三步**：② pin + 多层 drafter 核对 → ① 租卡 → ③ smoke test；② 扩展为含"该 vLLM pin 上是否存在可跑的多层 drafter"（否则卡型与主线都会选错） | 原"三选一"写法掩盖了依赖：卡型取决于多层 drafter 是否可得（`#6` 家族前提） | `README.md`、`EXECUTION_PLAN.md` v1.12 |

| 31 | **解冻步骤 ② 完成**（pin = vLLM main `9a35c08`）：E1 锚点全部命中 ⇒ 补丁无需修改；多层 drafter **可得**（`DFlashDraftModel`/`DFlash2DraftModel` 已注册、`qwen3_dspark.py` 存在；HF 有 `Qwen3-4B-speculator.dflash2` 等 6 个权重）；深度为 **config 旋钮**（`num_hidden_layers` / 训练侧 `--num-layers`）⇒ T0 便宜 | "用最新 main"后按设计执行 ②：核对锚点 + 家族可得性 + 显存算术，结论是**无需为 #6 租 80 GB**（24 GB 起步，FP8 KV 覆盖 128k） | `notes/p06-toolchain-check.md`；`README.md` 解冻顺序；`EXECUTION_PLAN.md` §5.1 与 v1.13 |

| 32 | **更正 pre-reg 的显存估算**：128k 的 `KV ≈7 GB` → **18.0 GiB（FP16）/ 9.0 GiB（FP8）**，并补 T0 ≥16 GB / T1 ≥24 GB / T0-T1 不需多卡 / 同 KV dtype 四条约束 | 该数字写在拿到 Qwen3-4B 真实配置之前，照它执行会误以为"24 GB 卡跑 128k 无需 FP8"，直接影响租卡决策 | `notes/prereg/p06-frontier.md`、`EXECUTION_PLAN.md` v1.14 |

| 33 | **T0/T1 工具链交付 + 两条 pre-reg 增补**（无卡）：`T0-runbook.md`、`make_depth_variants.py`、`run_t1.sh`、`analyze_t1.py`；pre-reg 增补 **ridge 斜率主读数**（`γ*(depth)` 与 `spread`）与 **H1b 分支**（斜 ridge 转"预测式 horizon"，但**必须打赢已 ship 适配器**） | 上一轮外部评估提出"斜 ridge 可能比正交更有价值"；采纳其可核部分，同时补上纪律：H1b 不得退化为"什么结果都能成文" | `probes/p06-frontier/*`、`notes/prereg/p06-frontier.md`、`EXECUTION_PLAN.md` v1.15 |
| 34 | **修掉一个脚本真 bug**：`run_t1.sh` 中 `$d` 紧跟全角冒号 ⇒ bash 把多字节首字节并入变量名，`set -u` 报 `d?: unbound variable`；并让 `DRY=1` 走独立分支（原实现会等 300 秒健康检查） | 离线自测发现：dry-run 超时 → 定位到变量名解析与等待逻辑 | `probes/p06-frontier/run_t1.sh`（已复测：语法 ✔、dry-run 瞬时输出 117 行命令） |

| 35 | **服务器工作流就绪**：新增 `docs/SERVER_BOOTSTRAP.md`；**修 `.gitignore`** —— `results/**` 原规则把 `summary.md` 也忽略了，会让实验结论随机器丢失 ⇒ 改为"结论入库、原始数据不入库"；卡分配定为 GPU0=E1、GPU1=T0/T1、其余留空 | 用户已有 8×4090 服务器（PCIe、无 NVLink） | `docs/SERVER_BOOTSTRAP.md`、`.gitignore`、`results/README.md`、`EXECUTION_PLAN.md` v1.16 |

| 36 | **环境方案定案**：venv 优先、**不必 conda**；一个 env 只用一种包管理器；装 vLLM **先试 wheel + `ModelRegistry` 自检 DFlash/DSpark 注册**，缺了才编译 pinned commit | vLLM wheel 自带 CUDA 运行时，与 conda 的 cudatoolkit/pytorch-cuda 混用易出 undefined symbol；git 安装会源码编译 20–60 分钟 | `docs/SERVER_BOOTSTRAP.md` §2/§2b、`EXECUTION_PLAN.md` v1.17 |

| 37 | **上机入口唯一化**：`docs/SERVER_BOOTSTRAP.md` 顶部加「阅读顺序」，从该文件按序指向 T0 → T1 → E1 → 判据 → 全局背景；并显式禁止从四份冻结上游文档取执行指令 | 避免上机时在 20 个文件间找入口，或误读带勘误横幅的历史文档 | `docs/SERVER_BOOTSTRAP.md`、`EXECUTION_PLAN.md` v1.18 |

| 38 | **服务器实地勘探结论 + 修 `.gitignore`**：GPU2 空闲（0% 利用率、仅 754 MiB 占用）→ 定为主用卡；**HuggingFace 直连不可达**（000）但 `hf-mirror.com` 通（200）⇒ 用 `HF_ENDPOINT=https://hf-mirror.com`；anaconda 的 **torch 2.9.0+cu128 实测 `cuda.is_available()=True`** ⇒ 驱动 550.67（CUDA 12.4）不构成阻断；docker 不可用、无 sudo/tmux，但 `/opt/anaconda3/bin/conda` 可用（用 `-p ~/envs/...` 前缀装 tmux）；仓库根已有 `~/myCCFA/.venv`（Python 3.11.7 + pip）但**原 `.gitignore` 未忽略 `.venv/`** ⇒ 已补 `.venv/ venv/ env/` | 实地探测（三轮只读命令）；若在服务器上误提交 venv 会污染仓库。**⚠️ 本条"驱动 550.67 不构成阻断"的结论已被 #39 推翻**：torch 2.9+cu128 能算只说明"计算可用"，而 vLLM ≥0.28 是 CUDA 13 构建，550.67 无法加载 | `.gitignore`、`EXECUTION_PLAN.md` v1.19 |

| 39 | **换机器（= `EXECUTION_PLAN.md` §12 D5）：原 8×4090 判定不可用** —— 事实链：(i) `qwen3_dflash2` 模块**只从 vLLM 0.28.0（2026-08-26）起存在**，0.26/0.27 连文件都没有（404）⇒ 补注册表无效；(ii) vLLM 0.27.1/0.28.0/0.29.0 全部钉 `torch==2.13.0`，而 2.13.0 **只有 cu129/cu130**（cu128 索引最高 2.11.0）⇒ 必须 CUDA 13；(iii) NVIDIA 官方表：CUDA 13.x 要求驱动 **≥580**，本机 **550.67** 落在 12.x 区间（实测报错 `libcudart.so.13` 缺失 / `driver too old (found version 12040)`）；(iv) SGLang 0.5.19 同样钉 torch 2.13 + `flashinfer_python[cu13]` ⇒ 换引擎也救不了；(v) 无 root，不能原地升驱动。**产出**：`docs/MACHINE_REQUIREMENTS.md`（选机规格 + 显存/磁盘预算 + Tier A/B 清单）与 `docs/acceptance_check.sh`（验收脚本，两条硬门槛）；`docs/SERVER_BOOTSTRAP.md` 加冻结横幅与阅读顺序第 0 步。**prereg 一字不改**（引擎仍为 vLLM，只是换机器）；若最终只能用 CUDA 12.x 驱动则走 Tier B（引擎降级 ⇒ 需新增 prereg） | 逐 tag 核对模块存在性、PyPI 与 PyTorch 索引实测、NVIDIA 官方驱动表、原机实测报错 | `docs/MACHINE_REQUIREMENTS.md`、`docs/acceptance_check.sh`、`README.md`、`docs/SERVER_BOOTSTRAP.md`、`EXECUTION_PLAN.md` v1.20 |

| 40 | **数据采集前修掉一个会导致 T0 假阴性的坑**：`mgoin/Qwen3-4B-speculator.dflash2` 的 `config.json` **顶层没有 `num_hidden_layers`** —— 层数在 **`transformer_layer_config.num_hidden_layers`（嵌套 dict = Qwen3Config 形态）**，且同层 `layer_types`（逐层 SWA/Full 标记）长度必须与层数一致。原 `make_depth_variants.py` 只认顶层键与**列表型** `transformer_layer_config` ⇒ 会产出与源**逐字节相同**的"变体"，T0 将据此得出"所有深度行为一致"→ 错杀 `#6`。**修复**：支持嵌套 dict、同步截短 `layer_types`、selftest 改用真实形态，并加**护栏**（一处未改即报错退出）。依据：vLLM v0.29.0 `qwen3_dflash.py` 的 `range(self.config.num_hidden_layers)` 与 `_dflash_layer_causal()` 的 `layer_types[layer_idx]`，其中 `self.config` 即 `transformer_layer_config`；权重 `config.py` 继承 `speculators` 的 `DFlashSpeculatorConfig`（`transformer_layer_config: PretrainedConfig`） | 逐文件核对真实 config 键 + vLLM 源码；离线 selftest 复现真实形态 | `probes/p06-frontier/make_depth_variants.py`、`probes/p06-frontier/T0-runbook.md` |

## 四、归档台账（14 项）

> 满足"任何方向的生与死都要记一条"的规则；逐项理由见 `EXECUTION_PLAN.md` §9，此处只留一行索引。

| 方案 | 归档理由（一行） | 关键依据 |
|---|---|---|
| `#2` 多租户保留准入 | **基线反证**：`TTL-300s` 与 `LRU-leaf` 逐字节相同 | decision #10；`archive/` 负面结果 |
| `#5` 跨并发分支准入 | 量级是 **O8 估计、零实测**；准入语义已被 vllm-omni COW/fork 占 | §9 |
| `#8` 长 CoT 检查点 | 机制偏薄，需搭调度策略才够一篇 | §9 |
| `#9` dLLM 去同步 | 薄机制 + 赛道升温最快 | §9 |
| `#10` recompute-vs-load | `PrefixPlace 2608.01655` 已把 recompute/replica-fetch 写进目标函数 | §9 |
| `#11` branch group 可撤销 | 依附"分支类 workload 是否真实" | §9 |
| `#13` elastic 投机 KV 预留 | 需 8×H200 | §9 |
| `#14` 片上 KV 预取 | Levy §5.3 已发表失败记录；权重流量 58% > KV 32.7% | §9 |
| `#15` 每位置 attention 预算 | 逐位置稀疏成本中性的理论证明压着 | §9 |
| `#16` 投机不确定度→KV 精度 | 算术天花板 ~1.2%；被拒 token 的 KV 本就丢弃 | §9 |
| `#17` 草稿放置 | Saguaro/SSD + vLLM RFC `#42109` 已占核心格 | §9 |
| `#18` ragged K | D-cut `2607.14647` 已做 **cross-request** 预算重分配 | §9 |
| `#19` token-yield × KV-retention | 量级单位数 % | §9 |
| `#20` block-aware 多节点放置 | 基线（Epoch+EPLB）须自建；1.1–1.3× | §9 |

## 五、待决策（按到期顺序；**锚点：D0 = 新机器通过验收之日**，见 `docs/MACHINE_REQUIREMENTS.md` §4）

| 截止 | 决策 | 触发条件 |
|---|---|---|
| **机器到位时** | 该机器是否合格（不合格就不开跑） | `bash docs/acceptance_check.sh`：驱动 **≥580** ＋ `DFlash2DraftModel` 被引擎注册；缺一 ⇒ 只能走 Tier B（引擎降级，**需新 prereg**） |
| D0+1 结束 | #1 生或死 | E1 判据（`probes/p01-e1-uncommitted-kv/README.md` §5）＋ D3 支线（`R_reserve/R_byte > 2`） |
| D0+3–4 | #6 主假设存废（T1 → T2） | 见 `notes/prereg/p06-frontier.md` 三级判定 |
| D0+1 周 | #3 归档或继续 | class-4 @batch≥32 是否 ≤0.85× **或**非结构性 ⇒ 归档 |
| StreamEP 正文公开后 | #12 判定门是否可执行 | 见 §7 与 §11（正文不可得时的 fallback 已写明） |
| D0+2 周 | #12 归档或保留 | 是否区分 DA-MoE / 转化率是否上升 |
| D0+3 周 | **锁定 1 主线 + 1 备线** | 量级 ≥8%（自测 vs 部署基线）且机制可归因 |

> `#4` 已暂缓（复核门不通过，decision #19），不再占用截止项。

| 41 | **E1 插桩在 vLLM 0.29.0 上锚点全中（无需改补丁）** —— ① Hook A 锚点在 `vllm/v1/core/sched/scheduler.py` **恰好 1 次**（1916–1917 行），缩进逐字节一致；② `KVCacheManager.allocate_slots(self, request, num_new_tokens, ...)` 与包装器位置参数签名一致（343 行起）；③ 返回的 `KVCacheBlocks \| None` 具备 `get_block_ids()`。⇒ D1「引擎固定 vLLM」在新机器的 0.29.0 上成立，E1 可直接 `--apply` | 对 v0.29.0 源码逐项核对（此前只对 main `9a35c08` 验过） | `probes/p01-e1-uncommitted-kv/instrument/README.md` |

| 42 | **新机器环境三个坑的实测定位与修复（smoke 三层全绿之前必须先过）** —— ① **locale**：容器把 `LC_ALL=en_US.UTF-8` 写进环境但该 locale 未生成（`locale -a` 只有 C / C.utf8）⇒ `setlocale()` 失败 ⇒ python `import readline` 段错误（栈：`rl_initialize → _rl_init_locale → PyInit_readline`）⇒ **EngineCore 静默死亡**，只报 `Engine core initialization failed ... Failed core proc(s): {}` 且**无 Python traceback**；修复 = 任何 python 进程前 `export LC_ALL=C.UTF-8`。② **KV dtype**：模型 bf16 时传 `--kv-cache-dtype float16` ⇒ FlashAttention 报 `query and key must have the same dtype`；合法值只有 auto/float16/bfloat16/fp8*（无 `fp16`）。③ **96 GB 卡** 用 `--gpu-memory-utilization 0.5`（原 0.85 会预留 ~82 GB 且启动更慢），起服务 ~65 s 达 health=200 | 三次真实启动失败逐层定位（错误日志 + 段错误栈 + locale 复现实验） | `docs/MACHINE_REQUIREMENTS.md` §5、`docs/server_setup.sh`、`probes/p06-frontier/smoke_test.sh` |

| 43 | **三层 smoke test 在真机全绿 ⇒ `#6` 家族前提成立** —— ① 无投机：出文本；② ngram：服务正常；③ **真 DFlash2 drafter：`vllm:spec_decode_num_accepted_tokens = 23`**（不仅"服务起得来"，而是**真的参与了投机**）。测试配置：`Qwen3-4B` + `/root/autodl-tmp/models/dflash2`，`γ=7`，`ctx=32768`，`KV_DTYPE=bfloat16`，`GPU_UTIL=0.5`；vLLM 0.29.0 / torch 2.13.0+cu130 / RTX PRO 6000 Blackwell 96 GB / 驱动 580.82.09 | `probes/p06-frontier/smoke_test.sh` 实测 | `results/p06-frontier/2026-09-12/smoke/` |

| 44 | **T1 网格更正：长上下文格 128k → 32k**（**数据采集前**）—— 实测 `Qwen3-4B` 的 `max_position_embeddings = 40960`，vLLM 直接拒绝 `--max-model-len 140000`（`User-specified max_model_len is greater than the derived max_model_len`）。原 128k 出自 toolchain check 的假设、未核该字段。要跑 128k 必须加 rope scaling（YaRN），那会**改变被测量的模型**，超出本探针范围 ⇒ T1 定为 `ctx{4096,32768}` + `--max-model-len 40960`（仍保留 8× 上下文对比） | T1 首次启动即报错（数据采集前）；T1 此前无任何数据 | `notes/prereg/p06-frontier.md`、`EXECUTION_PLAN.md` §5、`probes/p06-frontier/run_t1.sh` |

| 45 | **T0 判定 = 结局 A（深度旋钮可操作）** —— `ctx=4096`、`γ=7`、真实文本（WikiText）、`KV=bfloat16`、`GPU_UTIL=0.5`：平均接受长度 **1.303→2.829**、tok/s **101.5→180.3**，随 depth{1..5} **单调增长**，lossless 检查通过 ⇒ 进 T1。为得到这个结论修掉三个会给出**错误答案**的陷阱：① 变体路径必须含 `dflash`（否则 vLLM 当成通用 draft_model，接受率恒 0）；② 数据集必须真实文本（random 数据集接受率恒 0）；③ 变体必须 `--prune-weights`（vLLM 严格加载）+ 变体与源权重逐位相同（已核 83 张量） | 真机实测（3 次失败定位 + A/B/C 对照） | `results/p06-frontier/2026-09-12/T0/summary.md`、`probes/p06-frontier/T0-runbook.md` §7 |

| 46 | **T1 完成（54/54 格）：脊线是斜的 ⇒ 严格正交性被否证，转 H1b 分支** —— `ctx=4096`：γ*(depth) = {1:3, 3:7, 5:7}，spread=4；`ctx=32768`：{1:3, 3:3, 5:7}，spread=4。tok/s（4k）：d1 84.0/91.1/86.7（γ=1/3/7）→ d3 91.4/108.6/111.6 → d5 98.5/136.9/157.4 ⇒ **最优草稿长度随 drafter 深度系统性漂移**，且深度越大越偏好长 γ。接受长度 0.26→1.85。⇒ 按预登记：斜 ridge **不自动判死**，但必须走 **H1b（预测式 horizon）**，且该策略**必须打赢 vLLM 已 ship 的适配器**（per-batch K 查表）才算 Go | T1 全网格实测（vLLM 0.29.0 + Qwen3-4B + DFlash2 截断 1/3/5 层；真实文本 WikiText；ctx{4096,32768}；γ{1,3,7}；bs=1；3 reps；KV=bfloat16；GPU_UTIL=0.5；EAGER=1 全网格统一） | `results/p06-frontier/2026-09-12/{summary.md,t1.csv,ridge.csv}`、`notes/prereg/p06-frontier.md` |

| 47 | **`#6` 判死（T2 增量不达标）** —— T1 实测表上做策略对照：臂1 `adaptive-steps-only`（固定最深 depth=5 + 按 ctx 选最优 γ）= 157.4（4k）/ 33.2（32k）；臂2 `adaptive-steps+depth` = 157.4 / 33.3 ⇒ **增量 +0.0% / +0.3%（均值 +0.1%）**，远低于预登记 **8%** 门槛。机制解释：**深度在每个 γ、每个 ctx 上单调有利** ⇒ 不存在"长度 ↔ 深度"的非退化权衡，能加深就加深。H1b 要求的"打赢已 ship 适配器"也做不到（per-batch K 查表本身就是选 γ） | T1 全网格实测（54 格）；判定依据 `results/p06-frontier/2026-09-12/T2-verdict.md` | `results/p06-frontier/2026-09-12/T2-verdict.md`、`notes/prereg/p06-frontier.md` |

| 48 | **γ 可行上限 = 7（边界疑点解除）** —— T1 中 depth≥3 的 γ* 落在网格上边界 γ=7，须排除"γ>7 更优"。实测 γ=8 与 γ=15 **均在引擎初始化崩**（`CUDA error: CUBLAS_STATUS_INTERNAL_ERROR`，证据 `evidence/2026-09-12/d1_g{8,15}.serve.log`）；权重 config 的 `block_size=8` + `speculative_tokens=7` 表明 γ=7 是**训练操作点也是可行上限** ⇒ 网格 γ∈{1,3,7} 已覆盖整个可行域 ⇒ #47 的 No-Go **不是网格太窄造成的假阴性** | 扩展探针实测 + 权重 config 核对 | `results/p06-frontier/2026-09-12/T2-verdict.md`、`evidence/2026-09-12/` |

| 49 | **E1 判定：`#1` 原形态判死 + D3「消除保守预留」支线触发** —— ① 主判据：`R_byte` p95 = **0.0716%–0.0723%**（γ=3 × ctx{4096,32768} × bs{1,8} × 3 reps，共 12 格，高度一致）≪ 5% ⇒ **按预登记杀**；且实测与解析 `γ/ctx = 3/4112 = 0.0727%` 吻合，而整网格内 `γ/ctx ≤ 7/4096 = 0.17%` ⇒ 全部可行点均在门槛下（70× 余量）。② 次要终点 D3：`R_reserve/R_byte` p95 = **5.39 > 2** ⇒ **触发立项**；但解析上 `R_reserve/R_byte ≈ block_size/γ = 16/3 = 5.33` 与实测吻合 ⇒ 该信号本质是**最后一个 block 的取整浪费**（绝对值 ~0.39% @4k，且随 ctx 增大而变小），**不是**系统性保守预留 ⇒ 立项时须带此量级警示 | 真机实测（插桩 Hook A/B，锚点在 vLLM 0.29.0 已核对）+ 解析对照 | `notes/prereg/p01-e1-uncommitted-kv.md`、`probes/p01-e1-uncommitted-kv/` |

| 50 | **流程教训：结果目录必须在仓库之外** —— 实例上每次同步仓库都用 `rm -rf /root/myCCFA && tar -x`，而结果原本写在仓库内 `results/` ⇒ **E1 首轮的 12 格原始产物被同步覆盖删除**（T0/T1 已拉回本机未受影响）。修复：`OUT_BASE` 改到 `/root/ccfa_results/<date>`（仓库外），并保留仓库只读/可覆盖的语义 | 自己造成的数据损失；E1 需重跑（~35 min） | 实例 `/root/ccfa_env.sh`、本 decision log |

| 51 | **更正 #49 的 D3 结论：原"触发"是指标定义假信号 ⇒ D3 关闭** —— 源码 + 原始记录双重确认 `KVCacheManager.allocate_slots` 返回的是 **"A list of new allocated blocks"（新增块）**，故旧 `R_reserve = 新块×block_size/ctx ≈ block_size/ctx`，与 `R_byte ≈ γ/ctx` 之比**恒为 `block_size/γ`**（γ<8 必然 >2）⇒ 预登记触发条件被结构性假信号点亮。**更正后口径**（`total_blocks = get_block_ids(req)` = 请求持有总块，含 lookahead 预留）实测：`R_reserve_total/committed` = **1.00533 / 1.00534 / 1.00529 / 1.00526**（ctx{4096,32768} × 并发{1,8}）⇒ 只多占 **0.53%**（最后一个 block 的取整）⇒ **无保守预留可消除，D3 支线关闭**（不立项）。**原判据 / 新判据 / 理由**：原判据 `R_reserve/R_byte > 2`；新判据为 `R_reserve_total/committed` 显著 > 1；理由是原指标量的不是"预留未用"而是"新增容量占比" | 读 vLLM 源码 + 原始 alloc 记录 + 更正后重测 4 格 | `results/p01-e1-uncommitted-kv/2026-09-12/VERDICT.md`、`probes/p01-e1-uncommitted-kv/instrument/{e1_patch.py,analyze.py}` |

| 52 | **阶段总结：两条线都由实测关闭** —— ① `#6` 投机算力分配：T0 结局 A、T1 斜 ridge、**T2 增量 +0.1% ≪ 8% ⇒ No-Go**（γ 上限实测 = 7，网格已覆盖可行域）；② `#1` 未提交投机 KV：**R_byte 0.072% ≪ 5% ⇒ 判死**，其 D3 变体也因 `R_reserve_total/committed = 1.0053` 而无 headroom（#51）。**portfolio 现状**：`#12` 冻结待 StreamEP 正文、`#7` 仅算术量级、`#3` 先验 ≤6%、`#4` 暂缓 ⇒ **主线空缺**，需按 §12 D1 条款第 3 条（新预先登记）或重排 portfolio。**本轮净收益**：一天 GPU 内用预登记判据关闭了两条无 headroom 的线，并留下一套可复用工具链（深度变体 + 真实文本 prompt + T0/T1 执行器 + ridge 判定 + E1 插桩）与一份真机踩坑清单 | T0/T1/T2 + E1 全部实测 | `results/p06-frontier/2026-09-12/T2-verdict.md`、`results/p01-e1-uncommitted-kv/2026-09-12/VERDICT.md` |

| 53 | **复核后降级：`#6` 不是"已成立的 No-Go"，而是「条件性阴性（coverage-limited）」** —— 三处覆盖缺口：① **只测 `bs=1`**，而 prereg 明文规定"(bs,ctx) 区域反转"在不确定时**必须扩到 `bs{1,8,32}` 再判、不得据 T1 阴性直接杀** ⇒ 这一步我**违反了自己的预登记**；② **长上下文只到 32k**（Qwen3-4B 上限 40960），而该方向的原始动机（DFlash 185k 全上下文重扫、16 vs 71 tok/s）**恰落在未测区间**，且该区间是最可能把权衡推向浅层的地方 ⇒ 动机所在处没测；③ **γ 只到 7**（对 DFlash2 已是架构上限——γ=8/15 实测崩、`block_size=8`——但**换 block 更大的族**（如 DSpark block=16）可测 γ∈{7,11,15}）；④ 仅有**截断**变体，缺 prereg 要求的"已发布浅层 checkpoint"对照（截断只给下界）。**已确证的部分**（在 bs=1、ctx∈{4k,32k}、DFlash2 截断、γ≤7 子空间内）：深度在每个 γ 上单调有利、最优配置为角点 (5,7)、深度感知的 oracle 上界增益 +0.00%（4k）/ +0.30%（32k）且在噪声内。**补测方案**：扩 bs（强制）→ 换 target 补 128k → 换族补长 γ → 补浅层 checkpoint 对照 | 用户复核质疑（5 条）；逐条对照 prereg 原文与手上数据 | `EXECUTION_PLAN.md` §0/§5、`results/p06-frontier/2026-09-12/T2-verdict.md` |

| 54 | **两处会毁掉可复现性的流程失误（均已修，但损失不可逆）** —— ① **逐次原始数据被删**：T1 每格跑了 3 次（54 个 `bench.json`），但写在我自己的仓库目录里，被"同步仓库 `rm -rf /root/myCCFA`"清掉 ⇒ **只剩 18 个格均值，p50/p95 再也算不出**（这是我此前把结果放仓库内的直接后果，与 #50 同源）；② **`.gitignore` 规则过窄导致核心产物静默不入库**：`results/**` 只放行 `summary.md` ⇒ `t1.csv`、`ridge.csv`、`T2-verdict.md`、E1 的 `VERDICT.md` 全部未跟踪，而 `git add -A` **不报错**、`git status` **看不出** ⇒ 必须用 `git ls-files` 读回（本次已放开 `results/**/*.{md,csv,bench.json}` 并补入，见 commit `5ee6075`）。**规则**：结果落仓库外 + 入库后必须 `git ls-files` 读回 + 逐次重复数据必须留档 | 用户质疑 P5（"没有看到方差和原始逐格数据"）逐条核对时发现 | `.gitignore`、`notes/decision_log.md` |

| 55 | **审计发现：重复之间的噪声约 ±10%，而 T1 的部分 ridge 判定落在噪声内** —— 唯一幸存的逐次数据（T1ext `d1/γ7` 同配置连跑 3 次）= **87.81 / 89.74 / 97.56 tok/s，极差 10.6%**；跨会话同配置为 86.7。据此：depth=5 的 γ\*=7（+15%）**站得住**；depth=1 的 γ\*=3（+5.1%）与 depth=3/4k 的 γ\*=7（+2.8%）**落在噪声内 ⇒「斜 ridge ⇒ 转 H1b」证据不足**。**站得住的是角点占优**（γ7 下 depth1→5：86.7→157.4，**+81%**，远超噪声）——注意该结论**不依赖** ridge 斜率。混淆成因：`--disable-shuffle` 未设 + vLLM 默认开启 prefix caching ⇒ 每次 rep 的 prompt 顺序与缓存状态不同。**修正**：`run_t1.sh` 加 `--disable-shuffle`、`REQS` 8→32、`REPS`≥5，并**逐次原始 JSON 入库** | 逐条复核用户 5 条质疑时发现（问题 1） | `probes/p06-frontier/run_t1.sh`、本 decision log |

| 56 | **换机器：新实例是 RTX 5090 / 32 GB（非原 96 GB 机器）** —— 用户提供 `connect.westb.seetacloud.com:24486`；容器 `autodl-container-85654ab736-7b31381a`、驱动 **580.76.05**（≥580 ✔，sm_120 ✔）、系统盘 30 GB + 数据盘 **250 GB**。**原 96 GB 实例已被释放**，其上未拉回的原始数据（E1 的逐事件 JSONL、E1v2 原始、模型/venv）随之丢失；已入库的结论与产物安全。**32 GB 的网格约束**（KV=144 KiB/token，权重+drafter+激活 ≈ 11 GiB）：`ctx=4k → bs≤32`、`ctx=32k → bs≤4`、`128k/bs=1` 已接近上限（18 GiB KV）⇒ **S2（≥128k 上下文）在这张卡上基本不可行**，需要更大显存（≥48 GB 更稳）。环境重建中（tmux/venv+vllm/模型/变体/prompt） | 连接新机器后实测 | 实例 `/root/ccfa_env.sh`、本 decision log |

| 57 | **S1 预登记（bs 扩展，数据采集前登记）** —— 按 prereg 的 bs 扩展条款，在同一预登记下补测 bs 维度：**4k 臂** `bs{1,8,32} × depth{1,3,5} × γ{3,7} × REPS=5`（90 格）；**32k 臂** `bs{1,2,4} × depth{1,3,5} × γ{3,7} × REPS=3`（54 格）。统一点：真实文本 prompt（32 条/ctx）、`--disable-shuffle`、`--enforce-eager`、`KV=bfloat16`、`--gpu-memory-utilization 0.9`、**逐次 bench.json 全入库**。**判据（写在此处，先于数据）**：若深 drafter 在**所有 bs 与所有 γ** 上都不劣 ⇒ 该子空间内 `#6` No-Go 成立（并须报分散度）；若任一浅层配置反超且超出噪声 ⇒ 重做策略级 T2 | 用户复核质疑（覆盖不足）；prereg 明文授权同一预登记下的 bs 扩展 | `probes/p06-frontier/run_t1.sh`、`notes/prereg/p06-frontier.md` |

| 58 | **第二台机器（RTX 5090 32 GB）环境重建完成，并得到一条重要的装包经验** —— 30 分钟内重建：tmux → venv（`uv` + **TUNA PyPI ≈51 MB/s**，60 秒下 3 GB；同一台机器上阿里云 PyPI 只有 ~113 KB/s，差 **450×**）→ 权重（ModelScope 7–9 MB/s；drafter 用 `aria2c -x16` 从 hf-mirror 秒级完成，**`hf download` 会卡死**）→ 深度变体（--prune-weights，d1–d5）→ prompt（32 条/ctx，从 5.4 亿字符语料取前 30 MB 切片）。**硬门槛复验**：torch 2.13.0+cu130 / CUDA 13.0 / RTX 5090 31.4 GiB / **`DFlash2DraftModel` 已注册** / 真算子通过。**smoke 三层全绿，第三层接受 23 tokens —— 与 96 GB 机器逐位相同**（跨机一致性）。**32 GB 的可行网格实测**：4k 下 KV=153,167 tokens ⇒ **bs=32 放得下**（需 131,072） | 换机后从零重建（用户提供新 SSH） | `docs/MACHINE_REQUIREMENTS.md` §5.1、实例 `/root/ccfa_env.sh` |

| 59 | **S1 开跑（bs 扩展，判据已先行登记于 #57）** —— 第一臂 `ctx=4096 × bs{1,8,32} × depth{1,3,5} × γ{3,7} × REPS=5`（90 格）在 RTX 5090 上运行；`--disable-shuffle` + 32 条 prompt/格 + `--save-result`，逐次原始 JSON 全部留存入库。第二臂 `ctx=32768 × bs{1,2,4} × …` 随后（32k 单请求 KV 4.5 GiB ⇒ bs≤4） | prereg bs 扩展条款 + decision #57 | `results/…/S1_4k/`、`results/…/S1_32k/` |

| 60 | **新增 `docs/TOPIC_METHODOLOGY.md`（可复用的选题方法论 v1.0）** —— 按用户要求把"重新选课题"的方法固化下来，**第一原则是"先证明疼，再证明没人治，最后才谈怎么治"**，明确"某方法没人用过"不是立项理由。内容：真痛点四必要条件 ｜ **伪问题七形态（每种都挂本工作区实证）** ｜ 自检十问 ｜ 量级等级 L0–L3 ｜ **审核流程图（mermaid，草案，待合并用户版本）** ｜ 三级判定 T0/T1/T2 ｜ **判定纪律七条**（噪声/覆盖审计/混淆控制/上界范围/仪器口径/环境前提/产物治理）｜ 五套可复制模板 + 结题 12 项审计清单 ｜ 实证索引（规则→decision 编号）。已在 `README.md` 与 `docs/README.md` 登记为**最上位入口** | 用户明确要求（3 条约束：强调真痛点/非伪问题、审核参考流程图、必须可复用） | `docs/TOPIC_METHODOLOGY.md`、`README.md`、`docs/README.md` |

| 61 | **S1 部分数据推翻了 #6 的"No-Go"：深度最优值随 batch 翻转（远超噪声）** —— 4k/γ=3/5 reps 实测 tok/s：**bs=1** d1 74.4 → d3 90.1 → **d5 107.2**；**bs=8** d3 **656** > d1 496 > d5 407；**bs=32** d3 **1081** > d1 862 > d5 **472**。三项排除假象：① 重复极差仅 1.5–3.1%（效应 38–56%）；② bs=32 三档 `completed=32/32`（无排队截断）；③ KV 容量三档均 ≥131k tokens（130,685/148,959/153,167），且 d5 的 `mean_ttft`=3705 ms vs d3 510 ms ⇒ 是**计算代价**而非显存。**同时更正两处错误论断**：原写"深度在每个 γ、每个 ctx 上单调有利"**是错的** —— T1 在 `ctx=32768, γ∈{1,3}` 上 d3>d5（33.3 vs 31.4 / 29.4 vs 27.0）。⇒ **`#6` 从"条件性阴性"转为「有实质证据支持」**：存在非退化的、运行点相关的深度选择问题；下一步 = 用统一参数（GPU_UTIL 0.90 + MAXSEQS 32）重跑完整 4k 网格（90 格）并补 32k 臂，然后按 (bs,ctx) 逐点做"最佳固定深度 vs 深度自适应"的 T2 | S1 前 45 格实测 + 三项假象排除 | `results/p06-frontier/2026-09-12/T2-verdict.md`、`EXECUTION_PLAN.md` §5 |

| 62 | **占位核查（应"是否会和 vLLM 撞"）：γ 轴已被 vLLM 占，drafter 深度轴在引擎内仍空但被学术界挤压** —— ① **γ 轴（草案长度）已 ship 且在继续做**：`num_speculative_tokens_per_batch_size`（per-batch K 查表，已在基线 ④）+ **开放 RFC `vllm-project/vllm#48202`「Per-request effective proposal lengths for adaptive speculative decoding」**（把 γ 自适应推进到**逐请求**粒度）+ **DSpark confidence-scheduled verification**（vLLM 官方博客 2026-08-14，即基线 ③）。⇒ "只调 γ"的基线会**越来越强**。② **drafter 深度轴**：本轮检索**未发现** vLLM 有 depth-adaptive drafter 的 PR/RFC ⇒ 机制格在**引擎内仍空**。③ **学术挤压**：early-exit / 级联家族已在做"运行时深度自适应"（HELIOS, MLSys'26 oral，早退 + 只加载可能用到的层 + 实时 profiler；CAS-Spec 级联自投机）⇒ 必须在**对象轴**上写清"改的是 drafter 而非 target 的早退"。**对判据的含义**：这**提高**了机制层（增量 ≥8% vs 已 ship 控制器）的门槛，但**不影响**现象层（最优深度随运行点变化）的真实性 | 检索 vLLM issue/博客/文档 + 学术邻居；RFC 正文抓取受限（GitHub 返回导航壳、docs 429），故仅按标题与来源定性，**待补正文核对** | `notes/prereg/p06-frontier.md`（待补邻居表）、本 decision log |

| 63 | **接手继续：96 GB 原机恢复 + 预登记 A1/A3/A4** —— `connect.westd.seetacloud.com:11640` 恢复为**原实例**（容器 `autodl-container-1f2343b3ae-527…` 同名、驱动 580.82.09、97,887 MiB），资产**全部保留**（venv 8.2 G / 模型 11 G / 深度变体 12 G / prompt / 结果 91 M），省去重建。硬门槛复验通过（torch 2.13.0+cu130 / `DFlash2DraftModel` 注册 / 真算子 / 95.0 GiB）。**下一步按用户给定的两层判据推进**：A1（4k 全网格含 γ=7，判据③ 深度与长度是否耦合）、A3（bs=16 的 KV 中立控制，判别计算 vs 显存）、A4（shipped-adapter 基线）。判据已在 prereg 登记（数据采集前） | 用户提供 SSH；实例身份与资产逐一核对 | `notes/prereg/p06-frontier.md`、`results/p06-frontier/2026-09-13/` |

| 64 | **A1（96 GB 卡，120 格）判定：判据①②③ 全部不成立 —— `depth* = d5` 在所有 8 个 (bs,γ) 组合上一致，且统计可分辨** —— 4k 全网格（`bs{1,8,16,32} × depth{1,3,5} × γ{3,7}` × 32 prompts × 5 reps，`GPU_UTIL=0.45` + `MAXSEQS=32` + eager，KV=bfloat16，零 serve 失败）：`depth*` 全为 **d5**，与次优差距 **25.6–32.6%**，而重复噪声仅 **3.0–9.5%** ⇒ 全部"可分辨"；**无交叉反转、无内点最优、最优深度不随 γ 移动**。⇒ 按用户两层框架：**机制层无证据支持**（"运行时在深度与长度之间分配算力"不成立）。**同时更正 decision #61**：S1（5090）看到的 `d3 > d5` **在 96 GB 卡上三种显存配置下均无法复现**（0.45 充裕 / 0.30 故意饿死 KV / 0.95 几乎不留空闲），且 `d5@bs32` 在同配置下 5090 为 472 tok/s、96 GB 卡为 1941 tok/s（**4×**）⇒ 该反转判为**机器/运行期假象**（最可能是共享宿主的 CPU 争用或未受控的启动状态），**不是运行点相关的机制** | A1 全网格实测 + A3 两轮显存对照 | `results/p06-frontier/2026-09-13/A1_4k/{bs_grid.md,bs_grid.csv}`、本 decision log |

| 65 | **测量层面的新发现：同一格的三次重复可呈"冷/热"双峰** —— 在 `GPU_UTIL=0.95 + MAXSEQS=256` 下，bs=32/γ=3 的同格三次重复：d3 = 661 / 669 / **1389**，d5 = 750 / 757 / **1941**（第三次约为前两次的 2–2.6×）。⇒ **重复次数少（3–5）时，"前几次偏冷"会系统性地把均值拉低**；本工作区此前多处 3 reps 的结论（含 T1 的 ridge 判定）都受此影响。**规则**：报数前先看**逐次序列**（是否仍在爬升），必要时应弃掉前 N 次或显式区分冷/热态；`--disable-shuffle` 只解决顺序问题，**不解决这一步** | A3 两轮对照的逐次数据 | 本 decision log、`docs/TOPIC_METHODOLOGY.md` §4.1（待补） |

| 66 | **审计①（读数口径）：A1 的判定用错了统计口径，且偏差方向对 `#6` 有利 —— 已用 v2 重算，结论反而更强** —— 逐次重读 120 个格的 `bench.json` 发现：同一格的前几次重复有**冷启动爬升，且爬升幅度随深度不同**（bs=1：d5 爬 11–13%，d1/d3 只爬 6–7%；bs≥8 <4%）。`analyze_bs_grid.py` v1 用**全部重复的均值** ⇒ **系统性低估深 drafter**，即偏向主假设。**修正**：v2 同时报 `mean` 与 `steady`（后一半重复），**以 steady 判 argmax**，并输出 `ramp_pct` 与 **Welch t 检验 p 值**（替换"差距 > 2×极差"的启发式，已加 `--selftest`）。**复核**：8/8 个 `(bs,γ)` 仍是 `depth*=d5`，差距 25.6–32.6% → **26.3–31.8%**（扩大），p<0.05 全显著 ⇒ **判据①②③ 在 4k 全部失败，且比 v1 口径更强**。附带发现：`.metrics` 是**跑完之后**抓的 ⇒ `kv_cache_usage_perc` 等 **gauge 一律归零、不可用**，只有 `*_total` 计数器有意义 | 用户要求"审计实验计划是否有问题"；逐字段重读原始产物 | `probes/p06-frontier/analyze_bs_grid.py`、`results/…/A1_4k/bs_grid.{md,csv}` |

| 67 | **审计②（机制层）：A 系列全程只读了 `output_throughput` 一个字段，而机制数据当时就已采到 —— 补齐后得到"接受率悬崖"，判据② 结构性不可能成立** —— 重读 `bench.json` 发现 `spec_decode_acceptance_length` / **`spec_decode_per_position_acceptance_rates`** / `mean_itl_ms` 全部已在库中但从没被分析 ⇒ 此前只能答"谁快"，答不了用户判据⑦ 要的"为什么快"。**先核对恒等式再用**（18 格实测）：投机解码下 `mean_itl_ms` 是**每步**延迟，`tpot ≈ itl/accept_len`（18/18 吻合 <1%）⇒ `step_ms = itl`，`Δln(tok/s) = Δln(接受长度) − Δln(每步代价)`（初版误写 `itl×accept`，自测当场抓到）。**结果**：逐位置接受率 **d1 = `0.28 0.06 0.00 0.00 0.00 0.00 0.00`**（γ=7）—— **1 层 drafter 从第 2 个位置起接受率为 0**，不是"更便宜的同一种东西"而是丧失多 token 能力的坏预测器；d3 = `0.49 0.19 0.07 0.02 0.01 0 0`、d5 = `0.71 0.45 0.29 0.17 0.11 0.06 0.04`。代价侧 d1→d5 每步仅 **+5.9%**（bs=32/γ=7：33.0→34.9 ms），接受长度 **+111%** ⇒ 接受长度随深度**超线性**、代价近线性 ⇒ **深度轴上不存在浅↔深权衡曲线，只有悬崖；最优恒在可行域上界 d5**。⇒ 判据②（内点最优）在本家族上**结构性不可能满足**，主假设（深度与长度可互相替代）被这一对事实**否证**。归因检验：**16/16 格 `preemption = 0`** 且 KV 容量（203,980）≥ 需求（131,072）⇒ **优势确实来自计算侧，不是显存/排队假象（判据⑦ 干净通过）** | 用户要求审计计划正确性；`bench.json` 逐字段重读 | `probes/p06-frontier/analyze_mechanism.py`、`results/…/A1_4k/mechanism.{md,csv}` |

| 68 | **审计③（覆盖缺口）：预登记 γ 网格是 `{1,3,5,7}`，实际只跑了 `{3,7}` —— 缺的正好是唯一可能反转的角落 ⇒ 预登记 A5 补格 + A1-32k 收窄** —— ① **覆盖缺口**：prereg §量 登记 γ 因子为 `{1,3,5,7}`，A1/S1 只跑 `{3,7}`（γ=8/15 崩溃后收窄未回补）。由 #67 的代价结构（drafter 代价占比 ∝ 1/γ）⇒ **反转只可能出现在 γ→1 角落**，而该角落**一次都没测**。**A5 预登记**（数据采集**前**）：4k `bs{1,8,32} × depth{1,3,5} × γ{1,2,5} × 5 reps`（135 格），参数与 A1 逐字相同以便合并；**证伪性预测 P1** d5 优势随 γ 单调收缩且 γ=1 处 ≤ +10%、**P2** 悬崖仍在、**P3** γ=1 处 d5 每步代价相对增幅大于 γ=7。**P1 成立 ⇒ 反转只存在于"已 ship 控制器本就把 γ 压到 0–1"的角落 ⇒ 判据⑥ 增量上限 = 0 ⇒ 按预登记判死归档**；P1 被反驳 ⇒ 本审计的结构论证有误，重开 32k 全网格。② **32k 臂结构性冲突**：32k 单请求 KV = 4.5 GiB ⇒ `bs=32` 需 144 GB > 96 GB **物理不可行**；而中高并发下 d5 比 d1 少 16% KV（242,403 vs 203,980 tokens）**必然触发驱逐**，按**判据⑦ 该类反超不可采纳**。⇒ 收窄为 **bs=8 单点**（`GPU_UTIL≈0.72 / MAXSEQS=8 / REQS=8`），并**新增可采纳前置条件：`num_preemptions_total == 0` 且 `GPU KV cache size ≥ 8×32k`**；不满足者只作现象记录、不计入机制判定。**本轮起不再为 `#6` 增加任何未预登记的网格** | 用户要求"确认计划正确性再继续实验"；prereg 逐条对账实际执行 | `notes/prereg/p06-frontier.md`、`notes/audit-p06-design-2026-09-13.md`、`EXECUTION_PLAN.md` v1.23 |
