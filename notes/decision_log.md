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
