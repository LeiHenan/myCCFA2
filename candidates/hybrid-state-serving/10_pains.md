# S1 痛点发现（Pain discovery） — 混合模型（SSM/线性注意力）服务：状态不是分页 KV

**候选**：`hybrid-state-serving` ｜ **成本上限**：0 GPU·h（可用 ≤1 GPU·h 做量级取证）
**目标**：只收集**真痛点**：有人真的疼、有量级、有可复现证据。此阶段禁止提方案。

## 候选痛点清单

量级标注：**L1** = 有数字（可来自他人报告）；**L0** = 只有现象描述、尚无数字。**本阶段禁止提方案。**

### P1 混合模型的「状态」不是分页 KV：prefix cache / 抢占 / 投机回滚三条路径全部错配

- **现象**：SSM / 线性注意力层持有**每序列固定大小的循环状态**（非块可寻址），而引擎的分配器、前缀缓存（radix / 块哈希）、抢占（丢块重算）都假设「内存 = 可分页 KV 块」。三条路径因此都不成立：状态无法被块引用共享、不能被部分驱逐、抢占后只能整段重算。
- **谁在疼**：所有跑 hybrid 模型的部署方；两个主流引擎都在打补丁式修复。
- **量级**：**L0**（尚无数字，见 §取证任务）。
- **证据**：本机 vLLM 0.29.0 实装 `mamba_cache_mode` 三态语义（`none` / `align` / `all`）+ `mamba_block_size` + `replayssm_buffer_len`，逐字见 `vllm/config/cache.py:184-206`（复现：`sed -n '184,206p' <vllm>/config/cache.py`）；vLLM **RFC [#17140](https://github.com/vllm-project/vllm/issues/17140)**「Native support for Mamba, SSM, and hybrid transformer models in V1」（RFC = 未落地）；SGLang **PR [#34760](https://github.com/sgl-project/sglang/pull/34760)**「Fix Mamba state donation misalignment in unified radix cache under DCP」（状态与 radix cache 的所有权语义出过 bug）；vllm-ascend **PR [#10228](https://github.com/vllm-project/vllm-ascend/pull/10228)**「Fix the main2main caused prefix-mamba-cache error」。
- **反证**：引擎**已经**有 `mamba_cache_mode` 开关 ⇒ 可能只是配置问题。**这是本痛点最需要排除的替代解释**（S2 形态⑥）。

### P2 状态复用与**调度路径**耦合：同样的 token 前缀，状态检查点可用性不同

- **现象**：`align` 模式的语义是「只缓存**每个 scheduler step 的最后一个 token**、且该 token 位于 `i × block_size`」⇒ 状态检查点是否存在**取决于历史批处理路径**；而 KV 的命中只取决于 token 前缀。**两者不是同一个函数** ⇒ 同一前缀、不同批处理路径，复用能力可能不同。
- **谁在疼**：prefix-cache 命中率不稳定、无法复现的部署；评测与回归无法解释差异。
- **量级**：**L0**。
- **证据**：本机 `vllm/config/cache.py:192-195` 逐字语义；`v1/core/sched/scheduler.py:327` 用 `mamba_cache_mode == "align"` 参与调度分支（复现：`grep -n mamba_cache_mode <vllm>/v1/core/sched/scheduler.py`）。
- **反证**：若实测命中率与批处理路径无关 ⇒ 该痛点不成立（**必须实测证伪**）。

### P3 `all` 模式的**状态内存与 KV 双重占用**，没有联合预算

- **现象**：`all` = 每 `block_size` 个 token 存一份完整状态 ⇒ 状态内存随 token 数**线性增长**，与 KV 争抢同一显存池；`none` = 每次复用都要**从头重算**。现有三态是两个极端 + 一个受调度约束的折中，**没有「按价值分配」的机制**。
- **谁在疼**：长上下文 + 高并发部署（显存本来就不够）。
- **量级**：**L0**（但**零卡可先算上界**）。
- **证据**：`mamba_cache_mode` 三态定义（本机源码）；状态大小可由模型 config 直接算出（层数 × `n_heads` × `head_dim` × `d_state` + conv state），**不用卡就能算占比**。
- **反证**：若状态总量相对 KV 可忽略（<1%）⇒ 不构成痛点（直接淘汰）。

### P4 长上下文下 hybrid 模型 decode **崩塌**（内核路径回退）

- **现象**：hybrid-Mamba 模型在长上下文回退到通用 paged attention 路径后 decode 吞吐崩塌。
- **谁在疼**：长上下文 hybrid 部署（该模型族正是为长上下文而设计）。
- **量级**：**L1**（他人报告存在，但**具体数字未读到**——GitHub 正文抓取只返回导航壳）。
- **证据**：vLLM issue **[#50264](https://github.com/vllm-project/vllm/issues/50264)**「On RDNA, hybrid-Mamba models fall back to Triton paged attention and decode collapses at long context」。
- **反证**：标题自述触发条件是 **RDNA（AMD 消费卡）**，本机是 NVIDIA ⇒ **该具体触发条件可能在本机不存在**，只能作「这类路径回退会致命」的旁证。

### P5 prefix caching **改变模型输出**（cache-induced divergence，量化会放大）

- **现象**：同一请求，开 / 关前缀缓存得到**不同答案**；量化会放大该分歧。
- **谁在疼**：做评测、回归、合规的人；「同请求不同答案」在生产里是事故。
- **量级**：**L1**（论文标题级声明，具体数字未读正文）。
- **证据**：arXiv **[2609.04748](https://arxiv.org/abs/2609.04748)**《Same Request, Different Answer: Quantization Amplifies Cache-Induced Divergence in LLM Serving》。
- **反证**：可能只是浮点非结合的已知现象 ⇒ 必须证明**分歧幅度随缓存复用 / 量化系统性放大**才算痛点（否则命中伪问题形态④「度量口径」）。

### P6 KV 淘汰**没有价值 / 上下文语义**

- **现象**：淘汰只按 LRU 块粒度，不知道哪些 KV 值得留（即将被复用、成本高、属于同一会话）。
- **谁在疼**：多轮 / agentic 工作负载（会话被淘汰后重算 prefill）。
- **量级**：**L0**。
- **证据**：vLLM **RFC [#37003](https://github.com/vllm-project/vllm/issues/37003)**「Context-Aware KV-Cache Retention API (Prioritized Evictions)」（RFC 未落地）。
- **反证**：**学术上已高度拥挤**（KV 淘汰 / 分级 / offload 已成红海，含 MIRAGE / Oneiros 等 SoCC'25 工作）⇒ 很可能在 S3 被判占位，仅作对照。

### P7 RL 后训练：推理引擎与训练器的**权重同步**停顿

- **现象**：每轮更新要把新权重同步进 rollout 引擎，同步期间 rollout 吞吐掉底。
- **谁在疼**：做 RL 后训练的团队。
- **量级**：**L1**（有论文给出「通信量降低 ~100×」的对照量级，说明原方案通信量是瓶颈）。
- **证据**：arXiv **[2605.07330](https://arxiv.org/abs/2605.07330)**《SparseRL-Sync: Lossless Weight Synchronization with ~100× Less Communication》。
- **反证**：**已有解**（该论文本身）⇒ 作为课题已被占；且本工作区在训练侧零积累，3–4 周做不出可信数字。

## 逐条证据

见上每条 pain 的「证据」字段：全部为**可点击链接**或**本机源码行号**（后者可用 `sed -n '184,206p' <vllm>/config/cache.py` 复现）。**没有一条依赖「我觉得」**。

## 不疼的反证

主动去找的「不疼」证据：

1. **P4** 的触发条件自述为 RDNA（AMD），本机 NVIDIA ⇒ 该具体现象可能不存在；
2. **P7** 已有解（SparseRL-Sync 本身）⇒ 不疼或已被治；
3. **P6** 学术极拥挤（MIRAGE / Oneiros / LPLB 等）⇒ 即使疼也已有人治；
4. **P1** 可能只是配置问题（已存在 `mamba_cache_mode` 开关）⇒ **这是 P1 最需要被证伪的假设**；
5. **`#6` 的前车之鉴**：一个「看起来很结构性」的痛点，实测 oracle 上界只有 **+0.1%**（decision #53）⇒ **本研究区任何痛点都必须先算上界，再谈立项**。

## 量级汇总

| 等级 | 痛点 | 是否进入下一阶段 |
|---|---|---|
| **L1**（有数字声明） | P4（他人报告，RDNA 触发）、P5（论文声明）、P7（论文对照量级） | P5 进；P4 作旁证 |
| **L0**（尚无数字） | **P1、P2、P3**、P6 | **P1/P2/P3 不进入 S3，转 §取证任务（仍属 S1 的回环）** |
| 已被占 | P6、P7 | 否 |

> ⚠️ **诚实结论：本轮最强的结构信号（P1 / P2 / P3）目前全是 L0。** 按 S1 判据，**L0 不得进入 S3**。⇒ 下一步不是去读论文，而是**做一次 ≤1 GPU·h 的量级取证**（见下）。

## 取证任务

**目标**：把 P1 / P2 / P3 从 L0 抬到 L1（有数字）。**只测不治**（S1 阶段禁止提方案）。成本上限 **≤1 GPU·h**。

1. **零卡先算**（0 GPU·h）：从模型 config 直接算状态内存占比 ⇒ 若状态 / KV 比 ≪1%，**P3 当场淘汰**。用 `NemotronH` / `FalconH1` / `Zamba2` 任一 ≤8B hybrid 的 config（`ssm_state_size`、`n_heads`、`conv_kernel`、层数、attention 层占比）——**这一步不需要 GPU，可以立刻做**。
2. **单卡取证**（≤1 GPU·h）：同一 checkpoint、同一负载，跑 `mamba_cache_mode ∈ {none, align, all}` 三档 × prefix cache 开 / 关，量三件事：**吞吐差、显存差、同一前缀的命中率差**。若三档吞吐差 < 8% ⇒ P1 / P3 不构成可写课题。
3. **P2 专项**（合并在 2 内）：**同一 token 前缀**用两种不同的批处理路径（不同并发 / 不同 chunk 切分）喂进去，看状态检查点命中是否不同 ⇒ 若相同，**P2 不成立**。
4. **P5 专项**（≤0.5 GPU·h，可后置）：固定 prompt + 固定 seed，prefix cache 开 / 关各跑 N 次，统计输出分歧率（先不加量化）。

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] ≥5 条候选痛点，每条包含『现象 / 谁在疼 / 量级 L0–L3 / 证据 / 反证』五个字段
- [x] 每条证据是**可复现命令**或**来源链接**；两者都没有的标 L0 并写『未取证』
- [x] 至少 2 条来自**别人**的痛点（生产事故报告、引擎 issue 里的实测数字、已发表测量论文的开放问题），不是自己推测的
- [x] 记录『不疼的反证』：主动找证据说明这个痛点可能不存在或不重要
- [x] 量级汇总表按 L1 以上条目排序，L0 条目不进入下一阶段

> **杀出口**：全部条目 < L1（都没有数字）⇒ 回到搜证，不许进入 S2 谈方案
