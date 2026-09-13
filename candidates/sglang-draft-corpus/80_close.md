# S8 结题与归档（Close） — 草案语料生命周期：跨会话/跨副本的冷启动罚金

**候选**：`sglang-draft-corpus` ｜ **成本上限**：≤0.5 天
**目标**：把结论、适用范围、反证、产物一次写清；负面结论同等入库。

## 结论

**一句话**：SGLang NGRAM 服务在**全新实例**上的首通接受长度只有 **1.8360 / 8**（稳态 7.7696），
这是一个 **76.4%** 的冷启动罚金；用**引擎已 ship 的 `POST /add_external_corpus`** 在首个请求前装入
**模型自身续写**的 4127-token 语料（并显式打开 `--speculative-ngram-external-sam-budget 7`），
可把首通接受长度恢复到 **6.6787**，即 **`recovery = 86.2%`**、冷窗口 wall **2.45×**、
全链路（含生成语料那一遍）**4.05×** ⇒ **决策级 Go（条件性）**。

**判定逐条对照预登记**（`50_prereg.md`，冻结于 commit `207de66`，**采数前**）：

| 判据 | 阈值 | 实测 | 结果 |
|---|---|---|---|
| **C0** 现象可复现 | `control ≤ 0.5 × steady` 且对照同向 | 1.8360 ≤ 3.8848；**4 个独立进程全部 1.8360** | ✅ |
| **C1** 口径自洽 | 违例 0、`cached_tokens` 全 0 | 704 行：违例 **0**、`cached_tokens` **全 0** | ✅ |
| **C2** 可回收性 | `recovery ≥ 80%` | **86.2%** | ✅ |
| **C3** 可分辨性 | 效应 ≥ 3σ（k=3） | 效应 **+4.8427 token**；σ = 0.0000（两臂） | ✅（σ=0 的来源已在 S6 说明，非"效应无穷大"） |
| C4 wall 方向（辅助） | 方向与 C2 一致 | 24.98 s → 10.22 s | ✅ |

**三条必须与结论同时报的条件**：
① **语料语义是一等自变量**（prompt 原文 **−16.5%** vs 模型续写 **+263.8%**）；
② **配额必须显式打开**（默认 0 ⇒ 一切不发生，且 HTTP 装载路径不报错）；
③ **"恢复"在本轮仍是"在同一台机器上重新生成语料"**，不是从持久化状态恢复。

## 适用范围与反证

**适用范围（完整表见 `70_mechanism.md` §适用范围）**：
仅覆盖 **Qwen3-4B + NGRAM（γ=7）+ ctx 4096 + 串行 + radix cache 关闭 + 同模型续写语料**；
**未测且不得外推**：并发 > 1、ctx ≠ 4096、其他模型/家族、带 drafter 的投机方法（EAGLE/MTP/dflash）、
外部分布与跨模型语料、语料规模中间点、配额 1/2/4/5/6、**真正的跨进程/跨副本状态恢复**、多卡/跨机。

**已知反证（不许只写有利方向）**：

| # | 反证 | 影响 |
|---|---|---|
| R1 | **稳态已饱和在 7.7696 / 8（97.1%）** ⇒ 该 workload 上接受长度**没有更多空间**；可赚的只有冷窗口那一段 | 限制了主张的天花板；上界复核显示干预已达上界的 85.8% |
| R2 | **语料是"同模型对同批 prompt 的续写"，是最好情形**（模型"见过答案"） | 真实流量下语料靠跨请求重叠增长，收益会小 —— **本条未测** |
| R3 | **上游已把这条路做掉大半**（外置语料 #21425、多 SAM HTTP API #22203、动态配额 #22538、跨 scheduler 传输；基准 *"SAM ≥2×"*） | 新颖性风险；本候选的差异化只能落在**生命周期与恢复**，而那一维**本轮只做到"重新生成"**，未做到"状态恢复" |
| R4 | **T1-b 的负结果**：prompt 原文语料（131103 token，31.8× 于 output 语料）**反而有害** | 说明外置语料**不是免费杠杆**；用错语料会亏 |
| R5 | **σ = 0.0000 是确定性设置的性质**，不是"效应无限大" | C3 的强度被削弱为"远大于可分辨下限"；换 temperature > 0 必须重测 σ |
| R6 | **`/get_internal_state` 在 0.5.19 上 404** ⇒ 原计划的第二条独立采集路径（累计计数器做差）**不可用**；本判定只依赖响应体 + 客户端 wall（`/metrics` 的 batch 级 gauge 未纳入判定） | 交叉验证只有一条主路径 + 一条客户端路径 ⇒ 已在 S4 登记 |

## 产物清单

**派生表与结论（入库）**

| 产物 | 路径 |
|---|---|
| 本候选完整档案（S0–S8 + 冻结预登记） | `candidates/sglang-draft-corpus/00_intake.md … 80_close.md` |
| 冻结预登记（仓库级） | `notes/prereg/sglang-draft-corpus.md` |
| 派生结果（逐臂表格 + 判据对照） | `results/p12-draft-corpus/2026-09-13/summary.md` |
| 探针（**全部含 `--selftest`**） | `probes/p12-draft-corpus/{probe.py,analyze.py,verdict.py,corpus_client.py,build_corpus.py}` |
| 复现脚本 | `probes/p12-draft-corpus/{smoke.sh,run_corpus_experiment.sh,run_budget_contrast.sh,run_output_corpus.sh,run_validate.sh}` |
| 工具 | `probes/p12-draft-corpus/tools/{list_args.py,list_serve_args.py}` |

**原始数据（仓库外，不随仓库同步删除）**

```
/root/ccfa_results/2026-09-13/p12_smoke/        冒烟 160 格（三条采集路径验证）
/root/ccfa_results/2026-09-13/p12_corpus/       主实验 A_control×3 / B_treat×3 / C_steady(4 reps)
/root/ccfa_results/2026-09-13/p12_budget/       budget 极值对照 B0 / B7（同装 131103-token 语料）
/root/ccfa_results/2026-09-13/p12_outcorpus/    output-as-corpus：D_gen / D_treat / D_ctrl + corpus_output.jsonl
/root/ccfa_results/2026-09-13/p12_validate/     V_treat_r2/r3（重复）+ V_budget3（配额中间点）
/root/ccfa_results/p12_*_console.log            tmux 控制台全文（含每臂就绪时间与 PID）
```
每个目录含逐请求 `*.jsonl`、`*.serve.log`（**含 `server_args` 全量转储**，可核对每个参数）、`*.probe.log`、`*.corpus.log`。

**复现命令（自下而上，全部在 tmux 内）**

```bash
export LC_ALL=C.UTF-8; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
git archive --format=tar HEAD | ssh -p 11640 root@connect.westd.seetacloud.com \
  'rm -rf /root/myCCFA && mkdir -p /root/myCCFA && tar -x -C /root/myCCFA'
tmux new-session -d -s p12 -c /root/myCCFA "bash /root/myCCFA/probes/p12-draft-corpus/run_output_corpus.sh 2>&1 | tee /root/ccfa_results/repro.log"
# 判据统计（离线，可对任意一臂的 jsonl 重算）
/root/autodl-tmp/venvs/sglang/bin/python probes/p12-draft-corpus/verdict.py --dir <数据目录>
# 离线自测（零 GPU）
python3 probes/p12-draft-corpus/{probe,analyze,verdict,corpus_client,build_corpus}.py --selftest
```

**图**：本轮**未产图**（4 张表已足够表达：冷启动落差、语料语义反转、配额剂量-反应、上界复核）；如需图，数据在 `p12_validate/` 与 `p12_outcorpus/` 的 jsonl 里可直接重绘。

## 决策日志行

`notes/decision_log.md` 的 **#98–#104**（本目标）：
#98 S0 立项（用户三项授权）｜#99 占位快扫（已 ship 动态语料 API）｜#100 冒烟（冷启动罚金 4.24×，两处仪器缺陷）｜
#101 源码级占位与 API 缺口｜#102 主实验（干预被静默禁用，`recovery = 0%`，**不计入否证**）｜
#103 budget 极值对照（prompt 语料为负收益）+ **output-as-corpus Go**｜#104 验证轮（σ=0、剂量-反应）+ 结题。

## 后续动作

| 优先级 | 动作 | 理由（数字） | 成本估计 |
|---|---|---|---|
| **P0（下一阶段，需新开独立预登记）** | **真正的语料状态恢复**：把语料（token 块 JSONL，4127 token）**持久化并跨进程/跨副本恢复**，消灭"重新生成语料"那一遍 | **生成语料那一遍占全链路 25.04 s / 33.89 s = 73.9%**，是当前最大单项 | 应用层为主（语料是纯文本，可落盘/对象存储）；若需引擎内序列化则是新半径评估 |
| P1 | 补 **σ** 与 **并发 > 1** 两格 | 当前 σ 来自确定性设置；并发是真实服务形态 | ≤0.3 GPU·h |
| P1 | 补 **ctx 8k/32k** 与 **语料规模中间点** | 覆盖审计里明写为未测 | ≤0.4 GPU·h |
| P2 | 测**跨模型续写语料**（验证"语义决定符号"的机制推论） | S7 的可检验推论，未测 | ≤0.2 GPU·h |
| P2（对上游） | 把两条结构性发现回报上游：① `sam_budget=0` 时 HTTP 装载**静默无效**；② prompt 原文语料在 budget>0 下**降低**接受长度 | 两者都可复现、都有源码行 | 0 GPU·h |

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] 结题 12 项审计清单全部勾选（见 docs/TOPIC_METHODOLOGY.md §5.5） —— §12 项审计清单 逐项见下（①–⑫ 全部 `[x]`）
- [x] 结论写明**适用范围**与**已知反证**，不许只写有利方向 —— §结论（含三条必须同报的条件）＋§适用范围与反证（R1–R6，含"稳态已饱和""语料是最好情形""上游已做掉大半""prompt 语料有害""σ=0 的性质""缺第二条独立路径"六条反证）
- [x] 产物清单含：派生表、原始数据路径（仓库外）、复现命令、图 —— §产物清单含入库派生表、5 个仓库外原始数据目录、4 条复现命令（含离线 `--selftest`）；**图**一栏明写"本轮未产图及原因 + 重绘数据位置"
- [x] `notes/decision_log.md` 有对应行（决定 / 依据 / 影响文件） —— §决策日志行 列出 #98–#104，且校验器会在 `--repo` 模式下核对 `sglang-draft-corpus` 出现在日志中
- [x] 失败结论也入库（负面结果不得静默丢弃）；被取代的旧结论就地标注 —— 负面结果已入库：**主实验 `recovery = 0%`**（干预被静默禁用）、**prompt 语料 −16.5%**（反转）、**`/get_internal_state` 404**、**`ninja`/PATH 缺陷**；**旧结论就地标注**：decision #94 的"预置语料看不出效果"已在 `70_mechanism.md` §机制-1 与 decision #103 里改写为"当时未打开 `sam_budget`"
- [x] 若机制死亡但测量有价值 ⇒ **新开独立预登记**，不得把旧数据重新解读成测量论文 —— 本候选机制**未死亡**（`recovery = 86.2%`）；但**下一阶段（持久化恢复）已按 E5 条款要求新开独立预登记**，本轮数据只用于本轮判据，未重新解读

## 12 项审计清单（`docs/TOPIC_METHODOLOGY.md` §5.5）

- [x] **1. 判据与阈值是采数据前写下的吗？** —— 是。`50_prereg.md` + `notes/prereg/sglang-draft-corpus.md` 冻结于 commit `207de66`，**早于**任何正式实验（首个正式实验的 commit 为 `f4f8d70` 之后的补测；主实验脚本 `run_corpus_experiment.sh` 在该 commit 之后才首次运行）。审计中发现的**一处设计缺口**（漏写 `sam_budget`）已**如实登记为"数据采集后"修订**（`f4f8d70`），未静默改写。
- [x] **2. 报了吗 p50/p95 与重复次数？噪声 vs 效应量之比是多少？** —— 报的是**逐请求全量分布**（min/max/mean）与**独立进程重复数**（对照 4、处理 3、budget3 1、稳态 4 reps）；噪声：同进程 ≤2.2%（S4 实测）、进程间 σ = 0.0000；**效应 +4.8427 token ≈ 360× 同进程噪声**。
- [x] **3. 覆盖审计表填了吗？动机所在的区间测到了吗？** —— 填了（预登记 §网格 逐轴论证 7 轴）。**动机所在区间** = "全新实例的首通"，A/B/D_ctrl 三组共 5 个独立进程**全部命中该区间**；**未覆盖**（已明写）：并发 >1、ctx ≠ 4096、真实跨副本恢复。
- [x] **4. 基线里有已 ship 的适配器吗？** —— 有。基线①= 引擎默认（全新实例、空 trie）；基线②= 引擎自带**会话内累积**的饱和值；干预手段本身也是引擎已 ship 的 `POST /add_external_corpus`（非自研接口）。
- [x] **5. 混淆项（shuffle/温度/dtype/编译模式/引擎版本）全网格一致吗？** —— 一致：同 prompt 集同序、`temperature=0`、bf16、SGLang 0.5.19 固定、`--disable-radix-cache`、`--max-running-requests 32`、串行、`ignore_eos=true`、`max_new_tokens=128`；每臂独立进程，`server_args` 全量转储在每个 `*.serve.log` 里可核。
- [x] **6. 指标的口径对着源码核过吗？做过已知答案对照吗？** —— 核过：`spec_accept_length = completion_tokens / spec_verify_ct`（`tokenizer_manager.py:2811-2814`），上界 = `speculative_num_draft_tokens = 8`；已知答案对照 = C1 判据在 704 行上查 `accept ≤ 8` 与 `rate ∈ [0,1]`，**违例 0**；另核出 `/get_internal_state` **404**（原第二条路径作废）。
- [x] **7. 逐次原始数据入库了吗？`git ls-files` 读回验证过吗？** —— 逐次原始数据**按纪律留在仓库外**（`/root/ccfa_results/2026-09-13/p12_*`）；入库的是派生表 `results/p12-draft-corpus/2026-09-13/summary.md`，并用 `git ls-files` 读回验证（见本轮收尾提交）。
- [x] **8. 结果目录在仓库之外吗？** —— 是（`/root/ccfa_results/...`），仓库同步用 `rm -rf /root/myCCFA` 不会碰到它。
- [x] **9. 失败与异常（OOM、不可行域、崩溃）都记了吗？** —— 记了：`ninja` PATH 缺失（引擎启动失败）、`/get_internal_state` 404、`sam_budget=0` 使干预静默失效、prompt 语料负收益、预登记漏写固定量；**无 OOM**（每臂结束显存回 0 MiB）。
- [x] **10. 结论的适用范围写清了吗？** —— 写了（10 行表，逐轴"覆盖 / 未测"）。
- [x] **11. 被推翻的旧结论就地标注了吗？** —— 是：decision #94 的观测在 `70_mechanism.md` §机制-1 与 decision #103 里就地改写（"不是预置无效，而是当时未打开配额"）；`probes/p06-frontier/run_t1.sh` 的旧假 claim 早前（decision #76）已更正，本轮未再触碰。
- [x] **12. 归档/结题的理由可核吗（指向证据文件）？** —— 可核：每条判据都指向仓库外原始目录 + `verdict.py --dir` 可离线重算；机制的两条结构性发现都给了 `file:line`（`ngram.cpp:173`、`speculative_hook.py:867-880`、`ngram_worker.py:257-299`）。

> **杀出口**：无（但缺任何一项视为未结题）

---

## 结案增补（2026-09-13，R 阶段）：语料恢复已验证，并明确划出"可共享 / 不可共享"的边界

**新增已实测结论**（预登记 `notes/prereg/sglang-draft-corpus-r.md`，采数前冻结）：

1. **恢复比重新生成便宜 60×**：`cost_restore = 0.0539–0.4067 s` vs `cost_gen = 25.058 s` ⇒ **1.62%**。
2. **恢复效果逐位等价**：`R_regen` = `R_restore_docs` = `R_restore_xproc` = **6.6787431929115995**（相对差 **0.000%**）。
3. **跨进程可移植**：同一落盘文件在**另一个全新 serve 进程**上恢复，结果与同进程完全一致。
4. **合并语料更好**：`prompt 原文 + 续写`（135231 tok，装载 0.4067 s）⇒ accept **6.9467（+4.0%）**，
   达到稳态的 **93.0%**（`recovery` 由 86.2% 提升到 93.0%）。
5. **边界（源码级）**：`NgramCorpus` **14 个方法中无任何导出/序列化接口** ⇒
   **会话内学到的 trie 不可导出**；跨副本可共享的**只有外部语料文本这一半**。

**对"可写成什么"的影响（诚实评估，供后续决策）**：
- ✅ 可写成：**"NGRAM 草稿语料的跨副本恢复"** —— 有成本（1.62%）、有效果（等价 + 93.0% 恢复）、有源码级边界（trie 不可导出）。
- ⚠️ **新颖性风险仍在**：语料是**客户端持有的文本**，"把它发送到新副本"在工程上接近平凡；
  且母阶段已确认上游 #22569 覆盖了 *output-as-corpus* 的稳态收益。要立住，主张必须精确到
  **"新副本该从 0 学还是从 warm 副本继承"** 这一策略问题，并给出**整机收益**（本轮未测）。
- ⚠️ **真正的不可共享部分（trie）没有 API 可导出** ⇒ 若把它作为核心贡献，需要引擎内改动（超出中半径）。

---

## 终局增补（2026-09-13，P 阶段）：本候选**不作为论文立项**，按 §1B 形态收口

**决定性依据**（预登记 `notes/prereg/sglang-draft-corpus-p.md`，commit `9668fdf`，采数前冻结）：

1. **整机账抹平热路径收益**：预热盈亏平衡 **k ≈ 1.87 遍**，而该区间由**引擎自带会话内累积**覆盖
   ⇒ 外部预热的净收益 ≈ 0（P4 ❌，预登记明写"杀策略方向"）。
2. **跨租户迁移结构性失效**：`S_peer = S_bare = 3.064`（逐位）⇒ 长共享前缀被 `max_trie_depth=18` 挡在窗口外。
3. **"发送文本"有害**：`S_text` 比从 0 学**慢 17.4%**（accept 1.579 vs 1.906）。
4. **上游占位**：外置语料/多 SAM/动态配额/per-request trie 均已 ship 或有 PR；稳态收益已由上游基准
   （*output-as-corpus proves SAM ≥2x*）覆盖。
5. **可共享的只有一半**：`NgramCorpus` 14 个方法**无导出接口**，trie 不可跨副本继承。

**⇒ 结论**：本候选在**"优化类 + 可回收性"**判据下**已被自身数据杀死**，不作为论文立项。
**保留价值**：三阶段实测（母 86.2% / R 1.62% 成本 + 93.0% / P 整机账）+ 四条可迁移结论
（`sam_budget` 静默失效、语料语义决定符号、18-token 窗口限制跨租户迁移、预热盈亏平衡闭式条件）。
