# 独立审核报告：《LLM Serving「超越 KV cache 的计算复用」竞争格局调研》

**审核人**：本工作区（`Kimi选课题`）｜ **日期**：2026-09-14
**被审对象**：`llm-serving-computation-reuse-landscape.md`（100 行，来源未署名，按外部材料处理）
**审核方式**：① 逐条事实核查（含抽检其引用）；② 用本工作区**已实测**的数据对照其未取证主张；③ 独立判断其结论与建议

---

## 0. 审核结论（三句话）

1. **该文档的自我否定是正确且诚实的**——它自己推翻了原始 gap 表述，这一判断我**独立复核后同意**。
2. **但它有两处事实性遗漏，且遗漏的正是它声称"没人量化"的那部分**——本工作区已有实测数字可以直接填上。
3. **它给出的三个差异化建议中，建议 1 被我抽检的原文支持，是三者中最强的**；建议 2 的立论基础（HCache 的账）被它自己误读了一次。

**总判定：作为"选题地图"可用且质量较高；作为"gap 陈述"不能直接引用，需按 §4 修订后才能立项。**

---

## 1. 事实核查：抽检其引用（我实际读到的）

| 文档主张 | 我的核查 | 判定 |
|---|---|---|
| `2601.08343`：KV 复用对 judge 破坏决策不变性，best-of-N+verifier 的首个负面证据 | ✅ **已读 arXiv 摘要原文**。标题《When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges》。原文确认：*"end-task accuracy may appear stable, yet the judge's selection becomes highly inconsistent with dense prefill"*；提出 **Judge Consistency Rate (JCR)**；机制是 *"reuse systematically weakens cross-candidate attention, especially for later candidate blocks"*；结论 *"identify a previously overlooked failure mode"* | ✅ **准确**，且文档的转述没有夸大 |
| Tier 2「PIC 是 2024–2026 最拥挤的赛道」 | ✅ **与本工作区独立侦察一致**：我自己那份 KV 侦察里，位置无关复用列出 7 篇 2026 年论文（Irminsul / MiniPIC / COMB / LinearKV / Kamera / SemPIC / Leyline）+ SGLang RFC #30928，且 MiniPIC 的工程 PR 因精度 bug 被放弃 | ✅ **准确** |
| Tier 3「hidden state 粒度复用刚开头、无通用 runtime 对象」 | ✅ **与我 2026-09-14 的独立侦察一致**：我的三份侦察里没有出现"把 hidden state 做成通用 DAG 分支 runtime 对象"的工作 | ✅ **准确** |
| §二 判断 1：命中即跳过全层 forward，"KV reuse ≠ computation reuse 在精确前缀场景不成立" | ✅ **机制上正确**：有了全部层的 K/V，该前缀的后续计算确实不再需要 | ✅ **正确**，且是全文最有价值的一条纠正 |
| Tier 1.5 的 kernel 文献（Hydragen 32×、ChunkAttention、RelayAttention、Cascade Inference） | ⚠️ **我未逐篇核对**。本轮只抽检了 `2601.08343`。按 UNCERTAINTY 纪律记 **未取证** | ⚠️ 未核 |

**文档自己声明**："个别细节未逐篇精读原文核实"——**这个免责声明是诚实的**，且与我抽检的结果一致（抽到的那一条是准的）。

---

## 2. 我独立发现的**重要遗漏**（这是本次审核的主要增量）

### 遗漏 A：**"1→N 分支里 prefix caching 省了多少"——本工作区已经量过了**

文档 §三 建议 3 主张：

> 「**诚实的 workload 量化先行**：一篇扎实的『1→N 分支 workload 里 prefix caching 到底省了多少、剩多少在哪里』的 measurement 本身就是贡献（ETS 只测了树搜索 KV 共享度，**没人系统量化 hidden-state 级冗余**）」

**但本工作区在 2026-09-14 已经实测过一个直接相关的量**（`probes/e3_parallel_sampling.py`，
记录在 `E3判定_推理期并行已治理_2026-09-14.md`）：

```
model=Qwen3-4B  prompt=1024 tok  n=8  gen=32 tok  reps=3
A: N 个独立请求 (n=1)   1.0074 s      seqs=8  tokens_out=256
B: 单请求 n=N           0.9374 s      seqs=8  tokens_out=256
→ n=N vs N 个独立请求 = 1.075×，输出量相同
```

**含义**：vLLM 的 `n` **已经共享了 prefill**（若真算 8 次，差距会远大于 7.5%）。
⇒ 文档列为"没人做"的那类量化，**至少在一个形态上已被本工作区做过，且答案是"引擎已经做对了"**。

**但这条不能完全替代文档的建议**：我的测量是**逐字节相同的 prompt**（pure parallel sampling），
而文档关心的是 **TTS 分支**（分叉后各走各的）。两者是不同的 workload 形态。
**准确的说法是**：文档的建议 3 需要**收窄表述**为"TTS 形状（D≫P、分叉后后缀不同）下的量化"，
因为**同 prompt 形态已有人量过，且结论是已治理**。

### 遗漏 B：文档的复用量分解**缺了 block 粒度这一层**，而它有实测数字且直接决定分解结论

文档 §二 第 1 点写「前缀 prefill 全免」，第 3 点写「不复用的部分 = 分叉点之后的全部计算」。
**但"前缀"能命中多少，取决于 block 粒度的对齐**，本工作区有两个实测事实：

1. **hybrid 模型的最小可缓存前缀远大于 16**：vLLM 0.28 启动日志实测
   **Qwen3.5-4B = 528 token，27B = 784 token**（由 Mamba state page size 反推的 block size 决定）。
   ⇒ 共享前缀**短于 528 token 时命中恒为 0**；这些 workload 上"前缀全免"根本不会发生。
2. **spec-decode 与 prefix cache 的交互是 method-specific 的**：同一批实验中，
   `dflash2` 第 2 遍前缀缓存**命中为 0、重算 100% prompt（相对不投机是 256×）**，
   而 `eagle3`（2 blocks）与 `ngram`（1 block）**与不投机同样正常**。
   ⇒ 文档把"KV 复用"当成一个同质能力，**在投机解码开启时会按 drafter 类别分化**。

**为什么这条重要**：文档 §二 的分解是**算术式**的（P + D·N 的账），
但它假设"前缀一定命中"。**在 hybrid 与 spec-decode 场景下这个假设会失效**，
于是"剩余多少算力可省"的答案会显著不同。**这是文档结论的一个真实边界条件。**

### 遗漏 C：**judge 不变性这条线，本工作区有一条更便宜、已复现的证据**

文档建议 1 的核心卖点是「**质量/决策不变性保证**」，并引 `2601.08343` 作为"有反例衬底"。
**本工作区有两条同类的实测证据，且不需要 multi-agent 框架就能复现**：

| 证据 | 数字 | 出处 |
|---|---|---|
| 前缀缓存**冷/热状态**不同即改变输出（T=0） | `nospec` 7/8、`eagle3` 7/8、`ngram` **5/8** 的 prompt 输出不同；冷/热的 token_logprobs 四臂 **8/8** 不同 | 本工作区 p21 |
| 开 MTP 后同一请求输出变化 | 该工作区另一会话实测 PPL 3.34→15.83（Δ≈10 nats） | `spec-6000` 会话 |

**含义**：文档把"决策不变性"框在 **judge/verifier（multi-agent）** 语境里，
但**引擎默认配置（前缀缓存开启）下，temperature=0 的确定性本身就不成立**。
⇒ 这是一个**更广、更便宜、更容易被审稿人接受的入口**，
而文档完全没有提。**建议 1 若要做，应把入口从"judge"扩到"缓存状态即改变决策"。**

---

## 3. 对文档三个建议的独立判断

| 建议 | 我的判定 | 理由 |
|---|---|---|
| **1. TTS 跨阶段（generation→verification）复用 + 决策不变性保证** | 🟢 **三者中最强** | 抽检的 `2601.08343` 原文支持"这是新发现的失效模式"；本工作区又有两条可复现的同类证据（§2 遗漏 C）；且"跨阶段"确实没人做 |
| **2. 分支感知的 checkpoint/rollback（hidden-state 粒度）** | 🟡 **可用但立论需修正** | 文档说"HCache 只给了恢复场景的账"——**准确**；但它同时把 HCache 归为"证明 hidden state 是比 KV 更省的缓存对象"，而 HCache 的原话是"恢复成本降 6×、TTFT 快 1.93×"，**是特定恢复路径的收益，不是普适的"更省"**。立项前必须自己复算这笔账（本工作区 `hybrid_block.py` 类探针可直接算） |
| **3. 诚实的 workload 量化先行** | 🟡 **需要收窄** | 见 §2 遗漏 A：同 prompt 形态**已被本工作区量过且结论是"已治理"**；必须收窄到 TTS 形状（D≫P、分叉后后缀不同）才有增量 |

---

## 4. 修订后的 gap 陈述（可直接用于立项）

> **精确前缀复用已被基础设施化**（APC/RadixAttention），命中即跳过该前缀的全层 forward；
> **非前缀 PIC 已是红海**（CacheBlend/EPIC + 2026 年 7 篇以上）。
> **仍然未解的是：在推理时扩展（TTS）的"生成→验证/评判"跨阶段复用中，
> 复用的正确性没有契约**——已有一个反例（`2601.08343`：judge 选择与 dense prefill 不一致），
> 而引擎层面更基础的事实是**缓存状态本身就会改变输出**（本工作区实测 T=0 下冷/热 5/8–7/8 不同）。
> 因此本课题的问题是：**跨阶段复用需要什么样的不变量契约，才能同时拿到加速与决策等价？**

**必须写进 setup 假设的两条边界**（来自 §2 遗漏 B）：
1. hybrid 模型的最小可缓存前缀是 **528 / 784 token 量级**，短前缀命中为 0；
2. 前缀缓存与投机解码的交互**按 drafter 类别分化**（dflash 系命中归零，eagle/ngram 正常）。

---

## 5. 诚实边界

- 我**只抽检了一条引用**（`2601.08343`），其余条目的 venue/年份**未逐条核实**；
  文档自己的免责声明（"个别细节未逐篇精读"）与我的抽检结果一致。
- 我对建议 2、3 的保留**基于本工作区已实测的数据**，不是文献检索结论。
- 本报告**不构成**对该文档来源的可信度背书；它未署名，按外部材料处理。
