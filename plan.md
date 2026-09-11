# plan — 候选方案综合评估

**日期**：2026-09-11
**输入**：`V41.md`（本轮幸存候选 23 项） + `GPT.md`（GPT 提案审核后的 8 个不同对象）
**方法**：每个方案从**创新性**、**可行性**、**文献综述**三个维度独立评估，再给综合建议。

---

## 0. 评估口径（先读这一节）

### 0.1 立场：**不因为"有人做过类似工作"就否定**

本轮最大的方法论教训是：在先工作通常**只是定义了基线**，而不是关掉了路。因此本文件的评估**不把"已占"当作判据**，而是拆成三问：

| 维度 | 具体问什么 |
|---|---|
| **创新性** | ① 这个**确切机制**是否已被发表/已 ship？② 若已占，**剩余增量**是什么、有多大？③ 增量是**科学问题**还是**工程配置**？ |
| **可行性** | ① 硬件门槛（1 卡 / 单节点 / 多节点）？② 时间到第一个可信数字？③ 是 flag / bugfix / 插件 / 真机制？④ 基线要不要自建？ |
| **文献综述** | ① 最近的占位者覆盖了**哪个区间**、**没覆盖**哪个区间？② 该决策在**引擎 tracker** 里是否已有 ≥2 个并行提案（这**是**有效判据）？③ 关键事实的证据等级（Tier A 取到正文 / Tier B 仅元数据）？ |

**评级刻度**：创新性 ★1–5 ｜ 可行性 ★1–5 ｜ 文献密度 低/中/高（**低 = 机会，高 = 需找差异**）

### 0.2 只有两件事会真正否掉一个方案

1. **V1**：确切机制已 ship，**且剩余增量是一个 flag 或一个 bug fix**；
2. **V2**：**自测**相对**部署基线** <5%；
3. 以及**方向 B**：同一决策上 ≥2 个**并行**开放提案（这是引擎侧的拥挤，有效）。

**其余一切"有人做过"都只降低创新性评分，不构成否决。**

---

## 1. 候选总表（按综合建议排序）

| # | 方案 | 来源 | 创新性 | 可行性 | 文献密度 | 综合建议 |
|---|---|---|---|---|---|---|
| 1 | **投机 KV 分级 state**（位置 commit 剖面驱动） | GPT-③ | ★★★ | ★★★★★ | 中 | **立即判定（1 天 E1）** |
| 2 | **约束解码 dynamic × complex × concurrent** | V41-A1 | ★★★★ | ★★★★★ | 中高 | **优先** |
| 3 | **hybrid Mamba/GDN 前缀保留 × 投机** | V41-A2 | ★★★★ | ★★★★ | **低**（零论文） | **优先** |
| 4 | **server 端多租户保留准入** | V41-B10 | ★★★★ | ★★★★★ | 中 | **优先**（有 665K 轨迹） |
| 5 | **drafter 内部深度/宽度自适应** | V41-A4 | ★★★★ | ★★★ | 低 | **优先** |
| 6 | **跨并发投机分支的准入 + KV 容量调度** | GPT-① | ★★★★ | ★★★★ | 中高 | **优先** |
| 7 | **per-adapter KV 准入 + 驱逐策略** | V41-B5 | ★★★★ | ★★★ | 中 | 优先（次级） |
| 8 | **branch group 可撤销 / KV 记账** | V41-B6 | ★★★★ | ★★★ | 低 | 可做（1.5–3×） |
| 9 | **recompute-vs-load 显式调度** | V41-B7 | ★★★ | ★★★★ | 中 | 可做（5–20% TTFT） |
| 10 | **长 CoT 检查点/恢复** | V41-B3 | ★★★ | ★★★★ | 中 | 可做（1.3–2×） |
| 11 | **注意力感知预测 → 片上 KV 预取** | GPT-⑤⑥ | ★★★★ | ★★★ | 中高 | 需先判定 |
| 12 | **接受概率 → 每位置 attention 预算** | GPT-⑦ | ★★★ | ★★★ | 中高 | 需先判定 |
| 13 | **投机不确定度 → KV 精度** | GPT-B | ★★ | ★★★ | 高 | 需先判定（与 #1 并测） |
| 14 | **dLLM 去同步 denoise 循环** | V41-B4 | ★★★ | ★★★★ | 中 | 可做（1.12–1.4×） |
| 15 | **rank-imbalance dispatch + padding-aware all-to-all** | V41-A3 | ★★★ | ★★★ | 中 | 可做（证据最硬） |
| 16 | **elastic prefix-aware 投机 KV 预留** | V41-B2 | ★★★ | ★★ | 中 | 可做（需 8×H200） |
| 17 | **草稿放置（prefill/decode/拆分）** | V41-B8 | ★★★ | ★★ | 中高 | 需先判定 |
| 18 | **逐请求生成前准入 + ragged K** | V41-B1 | ★★ | ★★★ | **高** | 暂缓（须赢 2-D 表） |
| 19 | **token-yield × KV-retention 联合分配** | GPT-⑧ | ★★ | ★★★ | 高 | 暂缓（单位数 %） |
| 20 | **block-aware 多节点放置** | V41-B9 | ★★ | ★★ | 高 | 暂缓（1.1–1.3×，基线须自建） |
| — | K 改成计算预算 | GPT-② | ★ | — | 高 | ❌ **不做**（方向 B 有效） |
| — | KV placement 多 tier | GPT-④ | ★★ | — | 高 | ❌ **不做**（方向 B 有效） |
| — | 合并愿景 | GPT-⑨ | — | — | — | ❌ 非可证伪命题 |
| C1–C9 | 九个辅助探针 | V41-C | 各异 | ★★★★★ | 各异 | 见 §2.6（合计约 1 周） |

---

## 2. 逐项详评

### 2.1 第一梯队：一周内可判定，且增量明确

#### #1 投机 KV 分级 state（GPT-③）

- **创新性 ★★★**：**没有任何工作让 commit 概率去选内存层级**。最近的三个各持一块——**TransKV** 是**二值**（*"rejected KV is discarded without rollback"*，信号 = 已实现的 accept 事件、只在 commit 时一次）；**OasisKV** `2608.08097`（Microsoft）用 **draft token 的注意力质量**（relevance，非 P(commit)）做跨层预取；**VIA-SD** `2606.12243`（ICML 2026）的三态机器分档的是**验证算力、不是 KV 字节**。**⇒ 增量真实存在，但被三处证据压窄**：SpecMemo 要求投机行*"must retain high numerical precision to pass cumulative verification"*（压缩档破坏无损性）；未提交 KV 占比**从无测量**；分级相对"先原样留一步、再交给已 ship 的 tiering"的优势**只有一步宽**。
- **可行性 ★★★★★**：**整套里最便宜**。E1 只需 instrument 现成 `vllm serve` + EAGLE-3，逐步统计未提交草稿 KV 字节 / 已提交 KV 字节，扫 context × batch × γ{3,5,7}。**~1 天，不用建任何东西**，且能产出**有效的 V2 型证据**（本轮唯一一个）。
- **文献综述**：密度**中**。占位者：TransKV（Tier B，正文从未读到）、OasisKV、VIA-SD、CONF-KV `2605.24786`。引擎侧：`TieringOffloadingSpec`（N tier + 命名晋升与级联）、`cache_policy_module_path`、`kv_cache_dtype_skip_layers`、TRT-LLM `TokenRangeRetentionConfig` —— **但 tiering 路径里没有任何东西能看到 accept 信号**（≈80% 可用配置表达，**0% 可用概率表达**）。⚠️ **`2606.29223`《Depth Exploration for LLM Decoding》可能是藏身处**（摘要提及 "commit position" 与收缩 "exploration lattice"），PDF 取不到，**必须关闭**。
- **判定**：**先做 E1**。falsifier：p95 比值在全部 HBM 可行点上 <5% ⇒ 按实测定死。

#### #2 约束解码 dynamic × complex × concurrent（V41-A1）

- **创新性 ★★★★**：**该交集是空的**。XGrammar 2 是唯一动态引擎但**从未离开简单 JSON**；PSC 解决复杂语法但价格是 *"half to one minute per schema"* + 每 schema **3–6 GiB**（对逐请求动态语法不可支付）；CFGzip 只支持离线（自述 *"negates the advantages in dynamic contexts"*）；Gram2Token 只支持确定性语法。**同引擎实测差距**：JSON **+5.3%** → C++ **+50.7%** → Bython **+802.9%**。
- **可行性 ★★★★★**：**1 周，不改代码**，单卡。batch 1/8/32/128/256 × 4 个语法类 × spec on/off。
- **文献综述**：密度**中高**但方向不同——8+ 篇 2026 工作集中在**简单或离线**语法。**场地契合**：**XGrammar 是 MLSys 2025**（`proceedings.mlsys.org`），"目标会议 0 篇"已被证为**假阴性**。
- **判定**：**Go 判据**：class-4 在 batch ≥32 时 ≤0.85× 无约束。

#### #3 hybrid Mamba/GDN 前缀保留 × 投机（V41-A2）

- **创新性 ★★★★**：**全库零论文**——11 次 arXiv API 查询中**五次精确短语在全库返回字面 0**（`all:"EAGLE" AND all:"prefix cache"` → 0；`all:"radix attention" AND all:"speculative"` → 0）。而机制在**最高代价的模型类上被显式禁用**：`unified_tree_core.py:402` `self.is_eagle = params.is_eagle and ComponentType.MAMBA not in components`。
- **可行性 ★★★★**：1–2 卡，2 周。**`#53670` 的三臂消融就跑在 dual RTX 5090 上**——硬件门槛很低。
- **文献综述**：**密度低，但工程痕迹密集**。SGLang 的 bigram radix key **已 ship**（`radix_cache.py:59-60`）；vLLM 侧 RFC `#50438` **被维护者接管**（原作者公开抗议协调）、PR `#50897` **needs-rebase**、11 位 code owner 待审、另有第二个 RFC `#52817`。⚠️ **不得引用 "~20%"**（`#43559` 的 68 条时间线显示实测为 −0.67%/−2%/−4.8%）。
- **判定**：**优先**。幸存增量收窄为三条：(i) hybrid 类；(ii) 复用-vs-重算保留策略；(iii) MLA。

#### #4 server 端多租户保留准入（V41-B10）

- **创新性 ★★★★**：Khailo 全文 `GPU: 0`、`cluster: 0`——它是**客户端计时器**，没有引擎/调度器/租户；TraceLab 只做观测。**"外部性被命名 ≠ 机制被建出来"**。
- **可行性 ★★★★★**：**1 节点，≤2 周**，且**已有资产**：`data/syfi_coding_trace.duckdb`（665,453 轮）。
- **文献综述**：密度**中**。已被占的是"空闲预测 + 保留定价"（Khailo `2607.19214`、`2608.00101`、TraceLab `2606.30560`）；**未被占的是服务端机制本身**。轨迹数字：**28.4% 的会话 → 98.65% 的可避免成本**；断崖在 **5 分钟**（Anthropic TTL 边界）。
- **判定**：**优先**。对手：LRU / TTL / gap-aware 三臂对比。

#### #5 drafter 内部深度/宽度自适应（V41-A4）

- **创新性 ★★★★**：**经核验的空缺**——AdaEDL、SpecDec++、Pacer、SVIP **全部只调长度/停止，从未调过 drafter 的深度**。且**与 self-speculation 有结构性区别**：DEL/SpecBound/DSSD/LayerSkip 退的是**目标模型自己的浅层**，其机制（复用 verifier 的计算前缀）**无法迁移**到独立 drafter。
- **可行性 ★★★**：1–2 卡，2 周。
- **文献综述**：密度**低**（该具体机制）。对应格子有实测：**DFlash DT=4 在 ~185k 上下文下 16.0 tok/s vs 关掉投机 71 tok/s（4.4× 损失）**（`#54691`），其中 drafter 全上下文重扫占 **~182 ms 的 FA**。
- **判定**：**优先**。

#### #6 跨并发投机分支的准入 + KV 容量调度（GPT-①）

- **创新性 ★★★★**：**现有全部准入规则都是逐会话（per-session）的**——包括 PASTE 的 `EnginePressure = DecodeLoad + γ·KVLoad`；而并发分支之间存在真实资源耦合（vllm-omni 实测 **37 GiB/branch，H200 上约 2 个分支**）。**"会话独立"这个隐含假设是缺口。**
- **可行性 ★★★★**：先测 KV 是否绑定约束（vllm-omni 的 37 GiB/branch 是现成仪器），1–2 周。
- **文献综述**：密度**中高**——engine 侧已占：SpecTool §3.2（engine 内 KV 提交/回滚，vLLM 实测 +196 tok/s）、PASTE `2603.18897`（in-engine 调度钩子 + 概率化准入）、SPORK `2607.03333`（fork KV + vLLM proposer）、Speculative Actions `2510.04371`（Thms 3–5 准入规则）。**但它们都是单会话视角。**
- **判定**：先测 KV bindingness；非绑定 ⇒ 杀。

### 2.2 第二梯队：可做，但需明确差异

| # | 方案 | 创新性 | 可行性 | 文献综述要点 |
|---|---|---|---|---|
| **#7** | **per-adapter KV 准入 + 驱逐** | ★★★★ | ★★★ | SGLang `#2929` 的 **21 个打勾项全是适配器权重/算子/API，无一项关于 KV 池**；per-adapter prefix keying 只覆盖 **KEY**；**两引擎都无配额/分区/准入策略**。⚠️ 四篇"看似占位"的（ForkKV/aLoRA/ICaRus/LRAgent）**全是"共享"机制、不是配额策略**。判据：per-adapter 命中率离散度 <5% ⇒ 杀。1.2–2× goodput |
| **#8** | **branch group 可撤销 / KV 记账** | ★★★★ | ★★★ | TAPER 管的是**准入**、不是 fork 出去那部分 KV 的**记账**；SGLang 至今**直接 abort**——源码逐字 *"Beam groups cannot be retracted, so they are aborted instead of being requeued."* **1.5–3×** |
| **#9** | **recompute-vs-load 显式调度** | ★★★ | ★★★★ | vLLM `#53485` 的实际提案是 load-vs-**store** 的延迟 EMA 背压——**"recompute" 在它里面根本没出现**。**符号会随投机解码翻转**。判据：>5% 请求落在已 ship 默认的错误一侧。5–20% TTFT |
| **#10** | **长 CoT 检查点/恢复** | ★★★ | ★★★★ | **SGLang 并未默认 ship 部分抢占**——`schedule_batch.py::release_req` 把 host-KV 备份门控在 `disaggregation_mode == "decode"`，默认 `"null"` ⇒ **完整重算，与 vLLM 相同**。60k token 重算 ~20s vs 240MB PCIe 往返 ~10ms ⇒ **1.3–2×** |
| **#14** | **dLLM 去同步 denoise 循环** | ★★★ | ★★★★ | 关键背景：dLLM 的 **7 个已合并机制互不组合**（`dllm_hook.py` 强制关闭 overlap scheduler/HiCache/LMCache/FlexKV/流水并行/LoRA/分离式）。增量：`done.tolist()` 每步同步 + Python 循环。判据：f = h/(t_fwd+h)；<5% 杀、**≥10% 可发表**。1.12–1.4× |
| **#15** | **rank-imbalance dispatch + padding-aware all-to-all** | ★★★ | ★★★ | **证据最硬的一项**（真实 trace，非算术）：per-rank skew **5–25×**（Llama-4-Maverick >80×）→ **3.28× 归一化解码层延迟**；EPLB **两引擎都默认关闭**；现成 kernel **padding 到最大 rank**；已测转化率 **消息体积降 20% → 时间只降 9%**。**非投机论文** |
| **#16** | **elastic prefix-aware 投机 KV 预留** | ★★★ | ★★ | SGLang 每请求每步预留 **8 个 slot 而实际只需 1**。需 8×H200。判据：命中率与 goodput 变动 <5% ⇒ 杀 |

### 2.3 第三梯队：需先判定（增量真实但证据不足）

| # | 方案 | 创新性 | 为何只到"需先判定" |
|---|---|---|---|
| **#11** | **注意力感知预测 → 片上 KV 预取**（GPT-⑤⑥） | ★★★★ | **空单元格已精确定义**（两个决策分开：存储层**有**预测 / 计算侧**无**预测；空的是"预测 → 片上"）。**但反向证据很重**：Levy `2603.13430` §5.3 已发表失败记录（*"essentially failing at this approach"*）；AAAI-26 的**非预测版实测**是 **+15%/+7%/−2~−5%**（7:1 GQA 那档为负）；而 stall 分解显示**权重流量 58% > KV 32.7%**。**先测"KV stall 占长上下文 decode 墙钟的比例"——无人测过** |
| **#12** | **接受概率 → 每位置 attention 预算**（GPT-⑦/C） | ★★★ | Vegas（ICML 2026）*"identifies critical KV cache entries as a byproduct of verification"* 很近，但它选的是**哪些 KV 条目**、不是**每位置多少预算**。**逐位置稀疏在成本上可证明中性**（Σsᵢ ≥ k·s_min）⇒ 只能靠"同算力下提质"。**先测相关性是否存在** |
| **#13** | **投机不确定度 → KV 精度**（GPT-B） | ★★ | MiKV/QuantSpec/QSpec/"Don't Waste Bits!" 已占**按重要度分档**；候选的信号是 `P(commit)`（前瞻）vs 重要度（回溯）——**信号不同，但收益被"被拒 token 的 KV 本就丢弃"这条结构论证压住**。与 #1 并测即可 |
| **#17** | **草稿放置（prefill/decode/拆分）** | ★★★ | 靶心是 **vLLM `#42109`（OPEN，未合并）**《Disaggregated SD with Standalone Parallel Draft Model》，POC **1.4× TPOT**。**转移成本算术被更正了 30×**：草稿 KV = **112 KiB/token** vs 目标 128–320 ⇒ 真实比值 ~1.14:1。⚠️ **不得主张"没人搬草稿状态"**——SwiftSpec `2506.11309`、StarSD `2601.21622` 已占 |

### 2.4 暂缓（增量存在但量级或成本不划算）

| # | 方案 | 原因 |
|---|---|---|
| **#18** | 逐请求生成前准入 + ragged K | 文献密度**高**（CAST/CaDDTree/GLANCE/AngelSpec/LibraSpec/BanditSpec + `#54749` 的**六个并行信号**）。**量级未知**（既无"3–16%"支撑——那是我自己的拼接——也无 <5% 反证）。**必须打赢 2-D (bs,ctx) 表**（`#48944` 实测值 **1.29–1.36×**） |
| **#19** | token-yield × KV-retention 联合分配 | REA-4 给出的幸存增量，量级**单位数 %** |
| **#20** | block-aware 多节点放置 | **两个已发表半边的组合**（Epoch 构建 block plan 但明确拒绝喂给 placement；CAP 做了消费它的放置）。1.1–1.3×，且**基线 Epoch+EPLB 需自建** |

### 2.5 不做（仅有两条是有效判定）

| 方案 | 依据 |
|---|---|
| **K 改成计算预算**（GPT-②） | **方向 B**：`#54749`/`#54801`/`#47111` 三个**并行**开放提案；`#54749` 逐字 *"Six different signals are being proposed for this one decision right now. Batch size is what ships."* |
| **KV placement 多 tier**（GPT-④） | **方向 B**：RFC `#54779` + 原型 `#54327`、SGLang `#21846`、vLLM `#48445`/`#52113`/`#51428` —— **同一决策**上的多个并行提案；且 Dynamo KVBM 已 ship G1–G4 + TinyLFU/CMS |

> **注意**：这两条**不是**因为"有人做过"——而是因为**引擎侧已有多个并行的、相互竞争的提案在推进同一决策**。

### 2.6 辅助探针 C1–C9（合计约 1 周）

| 探针 | 成本 | 判据 |
|---|---|---|
| prefill CUDA-graph padding | **<1 周，1 卡** | 记录每次 replay 的 `padded/real`；`_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2` |
| FlashInfer radix 探针 | **1 周，1 卡** | Blackwell 上 FlashInfer 是默认后端，而 `attention_hook.py:570` 对其**静默** `disable_radix_cache=True` |
| 受迫区间测量 → jump-forward | 1 周 | 5.8µs 是 **mask 成本**而非跳步收益；**f·N ≥ 4–6 步/区间 ⇒ 1.3–1.4×** |
| byte 记账的 KV 准入 | ≤2 周 | 每步 bytes-read vs blocks-held；**≥10% 步决策分歧 且 e2e ≥30%** 才采纳 |
| RefShape 重开 | 待定 | 税 **+34.35%**；门控 p<14.6% 时 <5%。TML 的 `speculative:` 命中为 **0** |
| dLLM radix refresh | 2 周 | 1.4×；`#18724` 是初始版、block 粒度，且禁用 HiCache/LMCache/FlexKV |
| 异构 rank 统一分页 + 策略 | 2 周 | ~87% 适配器内存；LoRAServe 做的是 placement 不是 paging |
| drafter 侧 LoRA | 未测量 | `#52038` 只覆盖 DFlash 一个家族，吞吐基准标注 **PLANNED** |
| thinking-cache 生命周期 | 1 周 | 默认 **False**（opt-in），语义是"只保留 prompt 前缀"，只识别**一个**前导 span |

---

## 3. 六周执行规划（按"先判定、后投入"排）

| 周 | 动作 | 产出 |
|---|---|---|
| **W1** | 并发跑三个**判定性**探针：**#1 E1**（1 天）+ **#2 约束解码扫描**（1 周，不改代码）+ **#4 轨迹回放**（1 节点） | 三个可信数字 + 至少一次**有效的 V2 判定** |
| **W2** | **#3 hybrid**（1–2 卡）与 **#5 drafter 深度**（1–2 卡）的受控扫描；同时关闭 §4 的证据缺口 | 两个方向的实测 headroom |
| **W3** | 按 W1–W2 结果**收敛到 1–2 个**方向；启动 **#6/#7/#8** 中数据最好的一项 | 选定主线 |
| **W4–W6** | 主线做机制 + 基线；备线保留 | 第一个可投稿的结果 |

**资源**：W1–W2 全部可在 **1–2 卡**上完成；W3 起按需租（`#16` 需 8×H200，`#17` 需 2 节点，`#20` 需 4 节点）。

---

## 4. 必须在投入前关闭的证据缺口

| 项 | 影响 | 为什么重要 |
|---|---|---|
| **arXiv `2606.29223`《Depth Exploration for LLM Decoding》** | #1 / #13 | 摘要提到 *"commit position"* 与收缩 *"exploration lattice"* 到 *"retain only reusable branch states"*——**分级 KV 状态最可能的藏身处**；PDF 取不到 |
| **ASPLOS / ISCA / ATC / SOSP / SC / MLSys '26 论文集** | #11 / #15 | **未覆盖**（ACM DL、CSDL 被拦）⇒ **#11 的"空单元格"结论在架构会议方向只是暂定** |
| **TransKV 正文** | #1 | 从未读到（techrxiv 403），其"二值"刻画**仅据摘要** |
| **Libra**（ICLR 2026，OpenReview `WhxNwgGkAS`） | #18 / #19 | **UNRESOLVED**（403）；二手来源称其做"下一层专家激活的投机预测 + 热专家复制" |
| Nightjar DOI 不一致 | #16 | 实际返回 `S1383762126002079`，与记录的 `10.1016/j.sysarc.2026.103889` 不符 |
| OSDI '26 / ACL 2026 | — | **已全量覆盖，无占位者** |

---

## 5. 决策规则

1. **先判定，后投入**：任何方案在投入机制开发前，必须先有一个 **≤2 周、能给出可信数字**的探针。
2. **<8% 即止**：相对**部署中实际在跑的配置**（不是理想化"调好的"基线）的 headroom <8% ⇒ 终止。
3. **一次有效的 V2 胜过十次论证**：本轮 47 条击杀里 **38 条是越界**，根本原因是**没有一条建立在自己的测量上**。凡判"量级不足"，必须自测。
4. **"有人做过"只降创新性评分，不构成否决**——除非是 V1（确切机制已 ship 且剩余增量是 flag/bugfix）或方向 B（≥2 个并行开放提案）。
5. **机制混淆是双向错误**：既不要把 self-speculation 当独立 drafter（#5），也不要把存储层预取当计算侧预取（#11）。
6. **每个方向开工前先写清"它测的是哪个量、在哪个区间、对哪个基线"**——这一条能避免本轮全部六次"量搞错了"。

---

**配套文件**：`V41.md`（幸存候选细节）｜`GPT.md`（GPT 提案审核 + 标准一致性自审）｜`old.md`（引用纪律、判据框架、击杀台账与自身错误留档）
