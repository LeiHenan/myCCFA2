# S8 结题与归档（Close） — 跨引擎指标定义一致性

**候选**：`engine-metric-conformance` ｜ **成本**：累计 **≈0.5 GPU·h**（外加零卡源码/文档分析）

## 结论

**🟢 Go（决策级）—— 交付一个通过 S0–S5 全部闸门、且在服务器上实测验证过的候选。**

一句话：**跨引擎"对应指标"在语义上不等价，不等价幅度足以让比较失效（实测 32.6%），
而现有对照材料只做到"名称映射"。**

| 判据（采数前登记） | 实测 | 判定 |
|---|---|---|
| **C1** 计数类一致（<1%） | in/out/完成数 @bs32/off 两家**完全相同（0.00%）** | ✅ 通过 |
| **C2** 延迟单位（每步 vs 每 token） | vLLM/ngram **1.07×** vs SGLang/ngram **1.98×**（同方法同负载） | ✅ **跨引擎语义漂移** |
| **C3** 接受长度定义（≥8% ⇒ 不同） | @bs=32：vLLM **1.479** vs SGLang **2.194** ⇒ **−32.6%** | ✅ **成立** |
| **C4** 吞吐口径（≥8%） | 1341.0 vs 1392.2 tok/s ⇒ **4%** | ❌ 不成立 |

**S7 机制归因**：`mean_itl_ms` 的源码定义是**每个流式 chunk** 的间隔
（`vllm/benchmarks/lib/endpoint_request_func.py:241`），`mean_tpot_ms` 是**每 token**
（`benchmarks/serve.py:616`）⇒ **`itl/tpot ≈ 每个 chunk 携带的 token 数`**，而该数取决于**引擎的流式策略**
（dflash2 下 = accept_len ⇒ 2.85；NGRAM 下 vLLM ≈1、SGLang ≈accept_len）。

## 适用范围与反证

**成立范围**：vLLM 0.29.0 + SGLang 0.5.19 ｜ Qwen3-4B ｜ NGRAM（γ=7）｜ `ctx=4096`、`bs{1,32}`、`OUTLEN=128` ｜
单卡 RTX PRO 6000 Blackwell（sm120）｜ 同一客户端 `vllm bench serve`。

**已登记的反证 / 反例（不得省略）**：

1. **C1 完全一致（0.00%）** ⇒ **并非所有指标都不可比**：计数类（token/请求数）语义无歧义、可跨引擎直接用；
   不可比的是**延迟与接受长度这类"依赖内部策略"的量**。
2. **C4 仅 4%** ⇒ 端到端吞吐在同一客户端下差异不大（<8% 闸门）⇒ **不能说"跨引擎比较普遍失真"**。
3. **NGRAM 的有状态性**：SGLang 同会话内 `spec_accept_length` = 1.97/1.97/**2.65**，吞吐 444/1634/1607（第 1 rep 暖机）
   ⇒ 比接受长度**还必须控制缓存/语料状态**。最保守的逐 rep 读法仍支持 C3（区间不重叠）。
4. **未测**：EAGLE-3 / MTP / 带 drafter 的方法在两家引擎上的表现；其他 ctx 与并发；第三个引擎（TRT-LLM 因体积/许可排除）；
   SGLang 侧 `itl` 的**源码**定义（本轮只从数值推断其 chunk 行为）。
5. **环境陷阱已记录**：conda 的 `libstdc++` 只有 `GLIBCXX_3.4.29`，而 SGLang 的 JIT 内核要 `3.4.30`
   ⇒ 必须 `LD_PRELOAD` 系统的；否则报"加载 .so 失败"，**极易被误判为 GPU 不支持**（我们上一轮就这么误判过，已更正）。

## 产物清单

| 类别 | 路径 |
|---|---|
| 探针代码 | `probes/p09-engine-conformance/{install_sglang.sh,run_conformance.sh,analyze_conformance.py}`（均含用法与坑注释，分析器含 `--selftest`） |
| 派生结果（入库） | `results/p09-engine-conformance/2026-09-13/conformance.md` |
| 原始数据（**仓库外**） | 实例 `/root/ccfa_results/2026-09-13/p09_conformance/`（24 个 `bench.json` + `.metrics` + `.scrapes` + `.windows` + 4 个 `serve.log`） |
| 预登记 | `notes/prereg/engine-metric-conformance.md`（**采数前**冻结） |
| 候选档案 | `candidates/engine-metric-conformance/{00_intake,10_pains,20_screen,30_occupancy,40_measurability,50_prereg,60_decision,70_mechanism}.md` |
| 就地更正 | `probes/p06-frontier/analyze_mechanism.py`（"每步"→"每个流式 chunk"，结论数字不变但表述加限定） |

## 决策日志行

decision **#78–#83**（选轴 → S2/S3 → S4/S5 → S6 部分判定 → S7 归因 → 补测结案）。
**外加一条跨条目的更正**：#67/#76 里"ITL 是每步"的表述已在 #82 就地限定。

## 后续动作（若走论文级，必须先做这三件）

1. **控制 NGRAM 状态**（每 rep 重置 serve 会话，或显式声明语料状态）—— 否则 C3 的幅度会被状态漂移污染；
2. **扩到 ≥3 引擎 × ≥2 投机方法**（EAGLE-3 / MTP），把"语义漂移"从两个点变成一张表；
3. **量化对既有文献的影响**：在受控负载下复算若干公开的跨引擎比较，报告其误差量级
   —— 这一步决定它是"工具论文"还是"测量论文"。

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] 结题 12 项审计清单全部勾选（见 docs/TOPIC_METHODOLOGY.md §5.5）
- [x] 结论写明**适用范围**与**已知反证**，不许只写有利方向
- [x] 产物清单含：派生表、原始数据路径（仓库外）、复现命令、图
- [x] `notes/decision_log.md` 有对应行（决定 / 依据 / 影响文件）
- [x] 失败结论也入库（负面结果不得静默丢弃）；被取代的旧结论就地标注
- [x] 若机制死亡但测量有价值 ⇒ **新开独立预登记**，不得把旧数据重新解读成测量论文
