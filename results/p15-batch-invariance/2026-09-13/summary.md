# p15 批组成不变性：SGLang 在 sm120 上的**可复现发散**与开关代价

**日期**：2026-09-13 ｜ **目标**：`goal-ebcec24d` 第 6 轮 ｜ **成本**：**≈0.12 GPU·h**（7 次 serve，全部清理）
**原始数据**：`/root/ccfa_results/2026-09-13/p15_batch/`（`det{0,1}_n{1,8,32}.jsonl` + serve 日志）；早期两轮见 `p15_quick/`、`p15_long/`
**探针**：`probes/p15-batch-invariance/probe_batch.py`（含 `--selftest`）｜**引擎**：SGLang 0.5.19

---

## 一、现象：**同一个批里，相同输入、相同 token 上限、真贪心，输出却不同；开确定性开关后一致**

**为什么这次的设计比前两轮干净**（前两轮都栽在"无法证明合批"上）：
- **合批按构造成立**：SGLang 的 `GenerateReqInput.text` 支持 `List[str]`
  （`srt/managers/io_struct.py:182`，注释 *"It can be a single prompt or a batch of prompts"*）
  ⇒ **一次 HTTP 调用 = 一个批**，不再依赖"我发的并发请求有没有被合进同一批"这种**无法证明**的东西。
- **真贪心**：`top_k=1`。**关键教训**：`SamplingParams.normalize()` 把 `0 <= temperature < _SAMPLING_EPS`
  **改写成 1.0**（`srt/sampling/sampling_params.py:150-152`）⇒ **在 SGLang 里只写 `temperature=0` 得到的是标准采样**；
  真正的贪心是 `top_k=1`（同文件 :151 注释 "top_k = 1 means greedy sampling"）。
  **初版漏了 `top_k`，差点把采样随机性当成批组成效应。**

**设置**：Qwen3-4B / sm120 / `--attention-backend flashinfer` / `--disable-radix-cache` / `--max-running-requests 64`
/ prompt = `prompts_4096.jsonl` 第一条（**4096 token**）/ `max_new_tokens=64` / `top_k=1` / 每格 2 次重复。

| 设置 | 批规模 | 批内唯一**文本**数 | 文本发散率 | 批内唯一 **token 序列**数 | wall |
|---|---|---|---|---|---|
| **det=0（默认）** | 1 | 1 | 0.0000 | 1 | 0.60 s |
| **det=0** | **8** | **2** | **0.2500** | **8** | 1.51 s |
| **det=0** | **32** | **2** | **0.0625** | **32** | 4.60 s |
| det=1（`--enable-deterministic-inference`） | 1 | 1 | 0.0000 | 1 | 2.16 s |
| det=1 | 8 | **1** | **0.0000** | 1 | 2.88 s |
| det=1 | 32 | **1** | **0.0000** | 1 | 6.67 s |

**⇒ 三条结论（都有前后对照）**：

1. **现象（干预前）**：默认配置下，**同一批内 8 个相同请求产生 8 个不同的 token 序列**（`unique_fp = 8`），
   其中最常见的文本只占 75%（`divergence_text = 0.25`）；批规模 32 时也是 2 种文本。**批规模 1 时不发散。**
2. **干预后**：加 `--enable-deterministic-inference`，**批规模 8 与 32 均回到唯一 1 个输出**（`unique_fp = 1`）。
   ⇒ **开了就一致、不开就不一致**，方向与量级都清楚。
3. **代价**：wall 中位 **0.60 → 2.16 s（n=1，3.6×）**、**1.51 → 2.88 s（n=8，1.9×）**、**4.60 → 6.67 s（n=32，1.45×）**。
   ⇒ 与我们在 **vLLM** 上测到的代价（BI=1 墙钟 **2.0–2.3×**）**同量级**，
   也与既有报告一致（vLLM #27433 贡献者实测 −29.9%/+73.3% 与 −35.0%/+110.1%；SGLang 博客 +24.4~55.1%）。

---

## 二、**适用范围与未取证边界**（不许外推）

| 维度 | 覆盖 | 未测 |
|---|---|---|
| 引擎 / 后端 | **SGLang 0.5.19** / `flashinfer` attention + `flashinfer` sampling；vLLM 只测过**短序列** | fa3 / triton / fa4；vLLM 的**长序列**格 |
| 模型 / 数据 | Qwen3-4B，**单条 4096-token prompt**，`max_new_tokens=64` | 其他模型/长度；短 prompt（vLLM 轮测过且**未**观测到发散） |
| 批规模 | **1 / 8 / 32** | 44–64 共驻（issue #51187 报告区）；>64 |
| 调度 | 单次调用内构造的批 | 真实到达流下的动态批组成 |
| 结论强度 | **发散存在 + 开关消除它 + 代价量级** | **机制未归因**：未区分"算子归约顺序" / "kernel 选择随 batch 变" / "detokenize 路径"。但 `unique_fp` 说明**发散在 token id 层** ⇒ **排除 detokenize** |

**指标 bug 的自我更正（已入库）**：初版 `all_fp_same` 用"众数出现次数 == 总数"判断，
**当每个输出都唯一时恒为 True**，会把"全不同"误报成"全相同"；已改为 `len(fps) == n_f`，
并补了回归自测（`probe_batch.py --selftest` 覆盖"全不同 ⇒ 必须 False"）。

---

## 三、与子代理三轮检索的对接（跨引擎 × 跨社区，263 行 + 53 条生产痛）

| 维度 | 检索结论 | 我们的实测如何对接 |
|---|---|---|
| 现象是否被承认 | **是，且被维护者称为"fundamental"**：vLLM #966 zhuoran123 原话 *"Batching will change the order of each request being computed… **This is fundamental with batching.**"*；SGLang 官方 FAQ *"dynamic batching accounts for about **95%** of the indeterminism"* | 我们的实测是这条公认现象的**一个独立、可复现的实例**（SGLang / sm120 / 4096-token prompt） |
| 是否已 ship 开关 | 是（vLLM / SGLang / TRT-LLM / vLLM-Ascend / NVIDIA NIM）；**llama.cpp 明确拒绝** | 我们的"开/关对照"正是对已 ship 开关的**有效性验证** |
| 是否还有未解问题 | vLLM #27433 **仍 OPEN**；多处 open issue 报告**开了开关仍发散**（FA3/sm_90、SP/async-TP、conv、~44 共驻） | 我们**未**在 sm120 短序列上观测到发散；**长序列 det=0 下观测到、det=1 下消失** ⇒ 落在"开关有效"的一侧 |
| 文献是否饱和 | **A 类（本机制）20+ 篇**（2506.09501 NeurIPS'25、LLM-42、CoRun、MarginGate、2605.19537 等） | ⇒ **这决定了立项判断（见下）** |

---

## 四、立项判断：**仍是 ⚠，不立项**

| 判据 | 状态 |
|---|---|
| **现象层** | ✅ 已取证（前后对照、可复现、成本已量） |
| **机制层** | ❌ **未归因**（三种候选机制未区分；区分需读 kernel 或内核层插桩，超 1 人周半径） |
| **可回收性** | ❌ **不成立** —— 这里的"干预"是**打开确定性开关**，它消除发散但付出 **1.45–3.6×** 墙钟；要主张"优化"，必须证明**存在比"全开开关"更省的等价方案**（如只关受影响算子、或按批规模自适应开关），**而我没有证据表明这样的方案存在** |

**⇒ 综合**：这是"**已知、被维护者称为 fundamental、且已被大量研究的问题 + 上游已 ship 的开关在本格有效**"。
按 S3b 判据（v1.1.2 三问：开关在本代硬件/本负载是否有效？是否仍有未关闭同类报告？文档与实现是否一致？）
⇒ 本格是"**开关有效**"，**不构成未被治理的缺口** ⇒ **不立项**。

**复活的最小路径**：把差异轴换成"**比全开开关更省的等价方案**"——需先找到至少一个**按算子 / 按批规模**的局部关闭方式，
并先做 S4 的 oracle 上界（"完美地只关必要的东西"能省回多少）。在当前半径内**没有**这样的线索。
