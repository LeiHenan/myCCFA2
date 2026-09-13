# p15 批组成不变性：sm120 上的实测与**未取证边界**

**日期**：2026-09-13 ｜ **目标**：`goal-ebcec24d` 第 6 轮 ｜ **成本**：**≈0.05 GPU·h**（3 次 serve，均已清理）
**原始数据**：`/root/ccfa_results/2026-09-13/p15_{quick,long}/`｜**探针**：`probes/p15-batch-invariance/`

---

## 一、**已取证**的结论（可直接引用）

**设置**：vLLM 0.29.0 / Qwen3-4B / sm120 / `FLASH_ATTN` 后端 / `--enforce-eager --no-enable-prefix-caching`
/ 短 prompt（≈13 token）/ `max_tokens=48` / `temperature=0` / `seed=0` / 并发度 n∈{1,2,4,8,16} / 每格 3 次重复
⇒ **共 2 个 serve × 5 个并发度 × 3 重复**，逐格 8–16 个并发请求。

| 量 | BI=0（默认） | BI=1 | 判定 |
|---|---|---|---|
| **同批内一致性**（n 份相同请求的输出是否逐字相同） | 5/5 格 **unique=1，divergence=0.0000** | 5/5 格 **unique=1，divergence=0.0000** | 两臂都无批内发散 |
| **wall（n=1）** | 0.559 s | 1.265 s | **2.26×** |
| **wall（n=16）** | 0.621 s | 1.234 s | **1.99×** |
| **BI=0 vs BI=1 输出** | \multicolumn{2}{c}{**相同**} | 开关**未改变**结果 |

**⇒ 可引用的结论**：
> **在 sm120 + Qwen3-4B + 短请求（13→48 token）区间，`VLLM_BATCH_INVARIANT=1` 相对默认配置带来
> ≈2.0–2.3× 的墙钟代价（n=16 时 0.62 s → 1.23 s），而在该区间内未观测到任何批组成导致的输出差异
> —— 即"付出了不变性的代价，却没换到不变性的收益"。**

**代价数字与既有报告的对照**（子代理检索）：vLLM #27433 贡献者实测 **−29.9% 吞吐 / +73.3% 延迟（4090D）**、
**−35.0% 吞吐 / +110.1% 延迟（H20）**；我们实测的 **2.0–2.3× 墙钟** 与之**同量级且更保守**（我们用的是短序列、
`--enforce-eager`）。⇒ **代价侧结论可交叉印证。**

---

## 二、**未取证**的部分（不许当作结论）

| # | 我原本想说的 | 为什么不能说 |
|---|---|---|
| **U1** | "跨**并发度**输出一致 ⇒ 批组成不影响输出" | **没有合批证据**。我的探针用 `ThreadPoolExecutor` 发并发 HTTP 请求，**无法证明服务端把它们放进了同一批**。墙钟数字本身也不自洽：n=1 → 0.559 s、n=16 → 0.621 s（只涨 11%）——若 16 个请求真同批并行，应接近 n=1；若串行，应涨约 16×。**两个解释都不成立 ⇒ 我对自己测的是什么并不清楚。** 该结论**降级为未取证**。 |
| **U2** | "长序列 / 高并发（issue 报告的发散区间）也不发散" | **完全没测到**。SGLang 的 `input_ids` 路径因 `TypeError: Unexpected keyword argument 'seed'`（**SGLang 的 `sampling_params` 不接受 `seed`**）返回 500，n=1 与 n=32 两格全废。 |
| **U3** | "开关在 sm120 上有效" | 只能说"**在我测的短请求区间未观测到需要它修的东西**"，不能说它有效或无效。 |

**⇒ 因此本候选的 S3b 状态是 ⚠（未取证），不是 ✅，也不是 🔴。**

---

## 三、本轮踩到的四个仪器/API 缺陷（全部已记录，便于复用）

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| 1 | vLLM 引擎起不来 | **flashinfer 的 sampling JIT**（`flashinfer/sampling.py:68` → `run_ninja`）按**名字**找 `ninja`，而它在 `$VENV/bin/` | `export PATH="$VENV/bin:$PATH"`（**同类修复第 2 次**：p12 的 SGLang 侧已修过一次） |
| 2 | 60 次请求全 **404** | 探针写死了 SGLang 的 `/generate`，**vLLM 的 OpenAI server 没有该路由** | 探针改为 `--protocol {openai,generate}` |
| 3 | **`n must be 1 when using greedy sampling`** | vLLM 在贪心采样下禁用 `n>1` | 改为**并发提交 N 个相同请求**（`--mode concurrent`），更贴近"批组成"这一自变量 |
| 4 | SGLang 500 | **`sampling_params` 不接受 `seed`** | 未修（本轮就此停下）；修法明确：去掉 `seed` 或换协议 |

**可迁移教训**：**换引擎必须换协议**；**探针不该写死单一路由**；**"我发的并发请求" ≠ "服务端的同一批"**——后者需要独立的合批证据（vLLM 无迭代计数器，只有 token 级指标，故需另找口径）。

---

## 四、与既有证据的关系（子代理三轮检索，263 行 + 53 条生产痛证据）

| 方向 | 检索结论 | 对本候选的含义 |
|---|---|---|
| **问题是否被承认** | **是，且是"结构性"级**：vLLM #966 维护者 zhuohan123 原话 *"Batching will change the order of each request being computed… **This is fundamental with batching.**"*；SGLang 官方 FAQ 量化 *"dynamic batching accounts for about **95%** of the indeterminism"*；transformers #23017 维护者 *"**nothing we can do… other than increasing the precision**"* | **我第 3 轮的杀判据是错的**（"开关存在 ⇒ 已解决"）——已把该教训写进 S3b 判据（v1.1.2：已 ship 的开关必须过"三问"） |
| **方案是否已 ship** | 是：vLLM / SGLang / TRT-LLM / vLLM-Ascend / NVIDIA NIM 都有；**llama.cpp 明确拒绝**（维护者原话 *"any previously observed determinism was accidental"*） | 开关存在，但**每一家都有已记录的覆盖漏洞** |
| **是否还有未解问题** | 有：vLLM #27433 **仍 OPEN**（93 评论）、多个 open issue 报告**开了开关仍发散**、#42259（合作者署名的 RFC）仍在目录化新漏洞 | ⚠ 有缺口，但下面这条把它压住 |
| **文献是否已饱和** | **A 类（本机制）论文 20+ 篇**：2506.09501（NeurIPS'25，batch/GPU 数 → 最多 9% 准确率、9000-token 长度差）、LLM-42、CoRun、MarginGate（0.3–1.3% 解码步翻转）、2605.19537（五引擎 16.6pp）、2605.27763（22/55 安全翻转 → 0/55）、以及 2601.07239 的反方立场 | **⇒ 这是"已解决得很差 + 已被大量研究"，不是"未被治理的真痛点"** |

**⇒ 综合判定：本候选记 ⚠（条件性缺口，且我未能把条件坐实）。**
若要成为候选，差异轴必须写成**条件性**的（"上游开关在**条件 C** 下无效，而 C 在我的机器上可复现"），
而**本轮恰恰没有把 C 坐实**（U1/U2）。**⇒ 不立项，按未取证归档。**

---

## 五、下一步（若要继续，明确的最小动作）

1. **先解决"合批证据"**（否则一切批组成结论都不可信）：候选口径 —— 用 `/metrics` 的 `vllm:iteration_tokens_total`
   与 `vllm:generation_tokens` 做差推每步 token 数（合批时每步 token 数 ≈ 并发数），或改用**多进程**客户端
   绕过单进程 asyncio/GIL 的排队问题。**这是下一轮的第一件事，且必须在开新实验前完成。**
2. 修掉 SGLang 的 `seed` 问题，把**长序列 + 高并发**（4096 prompt / n=32）那一格补上——issue 报告的发散正在那一区间。
3. 只有当 1+2 都给出"**BI=0 发散 / BI=1 收敛**"的对照时，才谈得上 S4（oracle 上界）。
