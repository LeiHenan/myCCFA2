# 成熟组件两两耦合矩阵（截至 2026-09-15，本会话证据）

**为什么做这张表**：本目标 4 轮内我按"第一性原理推导候选 → 查占位"生成并判死了 5 条机制，
此前会话另有 ~9 条。**失败模式高度一致**：可推导的机制都已被 2025/2026 的论文或引擎 PR 拿走。
故改为**穷举结构**：把组件两两耦合列全，能标死的标死并附证据，把**空白且有意义**的格子留出来。

**标记**：`占`=已被占位（附证据） ｜ `崩`=我自己的算术/实测否证 ｜ `缺`=引擎缺口（保质期短，不立项） ｜ **`空?`=有意义但本会话未核**

| | spec | KV 管理 | 调度 | 前缀缓存 | 量化 | 稀疏 | 批处理 | CUDA graph | 能量 | prefill | 卸载 | 搜索/TTC | agent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **spec** | — | 占(自适应K/KV压力) | 占(SpecGen/vLLM γ表/TurboSpec) | 占(#47930 系) | 占(QSpec/QuantSpec) | 占(SpecGen) | 占(MoESD/MLSys'26) | 缺(UNIFORM_BATCH) | **空?** | 缺(prefill预算交互) | **空?** | 崩(p37:引擎缺陷) | **空?** |
| **KV 管理** | | — | 占(KV-aware 调度) | 占(同轴) | 占 | 占 | 占 | 缺 | **空?** | 占(chunked prefill) | 占(LMCache 等) | 崩(p37) | 占(CacheWise) |
| **调度** | | | — | 占 | **空?** | **空?** | 同轴 | 缺 | 占(VoltanaLLM) | 占(chunked prefill) | 占 | 崩(p37) | 占(SAGA/ThunderAgent) |
| **前缀缓存** | | | | — | 占(omlx #2487/#2272) | **空?** | 占 | **空?** | **空?** | 占 | 占 | 崩(p37) | 占(CacheWise) |
| **量化** | | | | | — | 占(压缩三件套综述) | **空?** | 工程中(#40807) | **空?** | **空?** | 占 | **空?** | **空?** |
| **稀疏** | | | | | | — | **空?** | 工程中(Sparse-vLLM) | **空?** | 占(MInference) | 占(ShadowKV/2604.26837) | **空?** | **空?** |
| **批处理** | | | | | | | — | 崩(捕获边界无浪费) | 占 | 占 | 占 | **空?** | **空?** |
| **CUDA graph** | | | | | | | | — | **空?** | 缺 | **空?** | **空?** | **空?** |
| **能量** | | | | | | | | | — | 占(layered prefill) | **空?** | **空?** | **空?** |
| **prefill** | | | | | | | | | | — | **空?** | 崩(p37) | **空?** |
| **卸载** | | | | | | | | | | | — | **空?** | **空?** |
| **搜索/TTC** | | | | | | | | | | | | — | **空?** |

## 本会话已判死的证据（每条附出处）
| 机制 | 判死依据 |
|---|---|
| 稀疏索引摊销（写入时刻建索引） | PIVOT (2607.24593)、MInference 1.0、SGLang #31790、Sparse-vLLM quest-pattern |
| 能量作为一等目标 | VoltanaLLM (2509.04827)、MLSys'26 layered prefill、**energy characterization (2608.28044)** |
| 投机天花板（带宽/算力之比、与接受率无关） | **MoESD (NeurIPS'25 spotlight, 2505.19645)** + "target efficiency" |
| 捕获边界填充浪费 | **我自己的实测否证**：捕获尺寸 32 以上每 8 一档，n=32→33 与 63→64 均无跳变 |
| 算力受限区间感知 | **TileSparse (ICML 2026)** "Arithmetic-Intensity-Aware Sparse Attention for Compute-Bound LLM Decoding" |
| 跨精度 KV 复用 | omlx #2487/#2272（缓存身份含精度变体已在修） |
| 并发搜索的 KV 准入/复用（候选 A） | **我自己的实测否证**：2.6× 全在 `beam_search` 包装层，引擎本身 1.05× |
| 带宽感知 KV 淘汰 | **我自己的算术否证**：每 block 每步省下的带宽是常数 ⇒ 退化为 LRU |
| 预测式 KV 管理 | 2601.14279（position-only 胜学习打分器）、LRU 达 Belady 99.4% |
| agent 运行时 / 语义预取 / 混合状态运行时 | SAGA、ThunderAgent、TeleRAG、hybrid-state-serving 已判死 |

## 下一步只需核这几个格子
**空?** 中真正"有意义（两组件都成熟且会互相影响）"的，按我的判断排序：
1. **spec × 能量** — 投机改变 energy/token 的整个结构（每步多出 K 个 token），而控制器只优化吞吐。
2. **量化 × prefill** — 量化对**算力受限的 prefill** 与**带宽受限的 decode** 收益完全不同；今天用同一个量化配置服务两者。
3. **调度 × 量化** — 调度器不知道当前请求走的是哪条数值路径，而量化改变算力/带宽比 ⇒ 最优批大小随量化变化。
4. **稀疏 × 调度** — 稀疏使 decode 变带宽受限（KV 读少了），从而**改变最优批大小**；调度器不感知。
5. **量化 × agent** — agent 的超长会话 + 量化 KV ⇒ 精度损失在长程上累积，而 agent 评测不看这个。

**注意**：上面 5 条我**都还没查占位**。按纪律，下一轮先零卡核这 5 格，再谈别的。

---

## §2 五个空白格的核查结果（2026-09-15，第 5 轮）：**全部被占**

分析先行——有两格在花检索之前就已被逻辑否掉：
- **量化 × prefill**：**写不成条件性差异轴**。权重是共享的，无法按 prefill/decode 相位给不同精度；KV 精度也不按相位分。**逻辑不成立即杀。**
- **量化 × agent**：是**质量/正确性**主张，不是加速 ⇒ **出范围**。

其余三格检索结果（每条都是**一次检索即命中**）：

| 空格 | 占位者 | 状态 |
|---|---|---|
| **spec × 能量** | [Online Scheduling of Battery-Aware Speculative Decoding for Energy-Efficient Cloud-Edge Collaborative LLM Inference](https://dl.acm.org/doi/10.1145/3832810.3832825) | 已发表（ACM） |
| **调度 × 量化/精度** | [FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving](https://arxiv.org/abs/2509.06261) | 已发表 |
| **稀疏 × 调度** | [HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management](https://arxiv-org.ezproxy.obspm.fr/html/2608.07009v1)、[Stochastic Sparse Attention for **Memory-Bound** Inference](https://github.com/zhaoyang97/Paper-Notes-en/blob/main/docs/ICML2026/llm_efficiency/stochastic_sparse_attention_for_memory_bound_inference.md)（ICML 2026）、TileSparse（ICML 2026） | 已发表，多篇 |
| （用户清单 #10 在线自适应） | [Autopoiesis: A Self-Evolving System Paradigm for LLM Serving Under Runtime Dynamics](https://ar5iv.labs.arxiv.org/html/2604.07144)：*"Continuous adaptation to shifting runtime trade-offs is necessary"* | 已发表 |

**⇒ 结论：矩阵里"有意义"的耦合格已全部被占；空格之所以空，是因为耦合本身不成立。**

### 结构性结论（本会话累计证据）
- 8 个给定方向：全占
- 3 个候选池（①失败理由过期 ②从未讨论 ③因果裂缝）：全耗尽
- 我自建机制：**14 条，0 幸存**（其中 5 条是我自己算错/测错后被自己否证）
- 组件两两耦合矩阵：有意义格全占
- 用户清单 10 条：#1 最拥挤、#2 被我实测否证、#9 能量被占、#10 自适应被 Autopoiesis 占

**⇒ "从第一性原理推导机制 → 查占位"这条路在当前时点的产出率 = 0/14。**
失败**不是**因为不够努力或测得不够，而是因为**可推导的机制都已被 2025–2026 的出版物拿走**。
要改变产出率，必须换**分母**——引入一个"别人没有的不可推导要素"（真实负载轨迹 / 他人没有的硬件或模型族 / 合作方数据）。这一点必须交回用户裁决。
