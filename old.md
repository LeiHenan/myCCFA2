# old — 选题流程的教训与台账

**日期**：2026-09-11
**定位**：这是 `V41.md`（候选课题合集）的配套文件。**V41.md 只放候选方案本身与本末；本文件放流程性内容**——复审台账、失效模式、取证管道缺陷、引用纪律、判据框架、决策规则，以及我自己被撤回的错误。

---

## 1. 复审总账

一轮完整选题流程提名并击杀了 22 个方向，随后对**全部击杀理由**做对抗性复审（4 个复审簇、47 条可分离的击杀理由）：

| 复审簇 | 击杀理由数 | 越界 | **有效** | 其他 |
|---|---|---|---|---|
| REA-1 投机解码 | 8 | **8** | **0** | 0 |
| REA-2 约束解码 + KV 内存层级 | 13 | **10** | **3** | 1 |
| REA-3 dLLM + 多 LoRA + 长 CoT 服务 | 16 | **14** | **1** | 1 |
| REA-4 早期十项 | 10 | **6** | **2** | 2 |
| **合计** | **47** | **38 (81%)** | **6 (13%)** | **4** |

**"不够新"导致的击杀：0 条。**

四类结构性死因分布：① 引擎或竞品在定节奏（12）· ③ 量级天花板低于门槛（6）· ④ 场地轴不对（4）· ② 缝在扩展点或已指派的 checkbox 上（2）· 我自己的错误（2，分类有重叠）。

**最要害的一条元结论**（REA-2 逐字）：
> **"两簇里没有任何一条击杀成立于 V2——没有一条被筛选者自己对部署基线的测量所支持。每一个天花板都是论文自报、厂商博客，或建立在这些之上的算术。"**

⇒ **凡要判"量级不足"，必须自测。**

---

## 2. 核心失效模式

> **把"一个测量格"提升为"整个空间的性质"。**

| 我用错的量 | 它实际是什么 |
|---|---|
| "约束解码只剩 ≤6% 差距" | XGrammar 2 在 **BFCL-v3 tool-call JSON、RTX 5090、0.6–8B、batch 1** 上的单格；论文**根本没有 batch 维度**。同引擎的 CFGzip Table 3 实测 C++ **+50.7%**、Bython **+802.9%** |
| "KV layout 天花板 ≤1.23×" | 那是 **vAttention 的容量/碎片化收益**（连续虚拟内存，2024），不是带宽测量 |
| "jump-forward 只有 0.04%" | 那算的是 **mask 成本**，不是**跳过的 decode 步数**——范畴错误，不是界 |
| "流式服务相对调好的静态 γ 只有 1.20×" | StreamServe Table 8 是**消融表**，无 γ 列；"1.20" 在 HTML 里**只是 LaTeX 行 id** |
| "草稿 KV 只有 4 KiB/token" | 真实值 **112 KiB/token**（Qwen3-0.6B），错了约 **30×**；真实比值 ~1.14:1 而非 80:1 |
| khailo 证明引擎层保留机制 | Khailo 全文 `GPU: 0`、`cluster: 0`——它是**客户端计时器**，无引擎/调度器/租户 |

**衍生模式：把两个占位者混为一谈。** 实例：把 EPOCH 的术语 "block plan" 安到 CAP 头上（**CAP 全文 "block" 只出现一次，在参考文献 "MegaBlocks" 里**，且 CAP 完全没有 dLLM 内容）；把 SGLang 的 **LoRA 适配器**驱逐 ABC 当成 KV 驱逐策略；把 vLLM `#53485` 的 load-vs-**store** 当成 load-vs-**recompute**。

**衍生模式：把"扩展点存在"当成"问题已解决"。** vLLM 的 out-of-tree `cache_policy_module_path` 住在 `vllm/v1/kv_offload/cpu/policies/`——**它是 CPU offload tier 的钩子，不是 GPU 前缀缓存的**（`grep -i cache_policy|eviction_policy vllm/v1/core/` **零命中**，GPU 侧驱逐是硬编码的）。而且钩子暴露的是**接口不是答案**——vLLM 用 LRU/ARC、Dynamo 用 TinyLFU+CMS、Mooncake 用 CMS，**三个不同答案共存 = 问题还活着**。

**衍生模式：引擎原生位置上，低论文数是警告而非空缺。** 那里的工作以 PR 形式落地。同理，**一个 open PR 证明的是"引擎项目已经在跑"，不是"这个领域还空着"**（`#42109`、`#50438` 都是实例）。

---

## 3. 取证管道缺陷（会系统性坑后续工作）

| 缺陷 | 后果 | 规则 |
|---|---|---|
| **剥离 HTML 标签会静默删除 payload 里的评论** | "我重读了线程、没有撤回"这类结论**全部不可靠**。实测：`vllm_49548_issues.raw` 含作者撤回，剥离后的 `.txt` 里**没有** | 撤回检查必须对 **`.raw`** 做 `grep -a` |
| **`state="DRAFT"` 不等于已关闭** | 只识别 OPEN/MERGED/CLOSED 的枚举器会**低估 tracker 拥挤度**（方向 B 会把"其实很拥挤"判成"清空"） | 状态枚举必须含 DRAFT |
| **phantom ID 能通过"返回 200"的检验** | `sgl-project/sglang#52477`/`#55931` **不存在**；它们真实的身份是 **vLLM** 编号 | **必须测试你正在断言的那个 repo 限定形式** |
| **ar5iv 会软回退到摘要页** | 返回与 arXiv 摘要页 **MD5 完全相同**的 43,040 B——一个穿着 HTML URL 的摘要 | 必须比对字节数，或直接取 PDF |

**状态码/URL 骗人已有 5 种形态**：TechRxiv 403、arXiv 软 404、index-page 软 404、ar5iv 软回退、payload-only 评论。

**另有两条状态误读**：
- `stateReason: "COMPLETED"` 可能是 **stale bot** 关闭造成的（SGLang `#30263`：`ClosedEvent` 的 actor 是 `github-actions[bot]`，`closer: null`）——必须再看**是谁关的**。
- **GitHub roadmap 的纯文本导出会剥掉复选框**——SGLang `#14199` 状态 `CLOSED/COMPLETED` 但 **41 项里有 23 项未勾**（含 "Support overlap scheduling"、"Requests early exit"）。必须从原始 HTML 解析 `<li class="task-list-item">`。多 LoRA 的 `#2929` 踩的是同一个坑：**21 个打勾项全部是适配器权重，没有一项关于 KV 池**。

**第六种**：`## Correction:` / `I am retracting…` 这类**作者自我撤回**可能只在 payload 里——见第一条规则。本轮实测抓到 4 处真实更正/撤回（`#49548`、`#48627`、`#55265`、`vllm50505`）。

---

## 4. 引用纪律

> **来源只有在"取到正文、看到标题、并看到被引用的那句话"时才算数。**
> **状态码不构成证据。** 404 不证明不存在（版本化 DOI / 重命名 / 私有）；200 不证明存在（软 404 / 索引页 / ar5iv 回退）。

1. **判定"不存在"前必须跑 Crossref 标题查询**：`api.crossref.org/works?query.bibliographic=<title>&rows=3`。
   （前缀存活对照**探测不到**前缀内部被截断的 ID——TechRxiv 用版本化 DOI，`.../v1` 才是注册标识符。）
2. **撤回检查必须对 `.raw` 做 `grep -a`**，不得在被剥离的文本上做。
3. **引用 issue 必须带 repo 前缀**（vLLM `#38392` ≠ SGLang `#38392`）。
4. **必须测试你正在断言的那个 repo 限定形式。**
5. **状态枚举必须含 `DRAFT`**；`COMPLETED` 还要看**是谁关的**。
6. **roadmap 复选框必须从原始 HTML 解析** `<li class="task-list-item">`。
7. **引用任何数字前先写清：它测的是哪个量、在哪个区间、对哪个基线。**

**另两个陷阱类别**：
- **"问题陈述" vs "方案陈述"**：RFC `#52038` 那句 *"each domain needs its own drafter (e.g. 0.8B) held in memory"* 逐字属实，但它描述的是**该 RFC 要修掉的现状**，且修复已被原型验证。同一句话在两种读法下结论相反。
- **伪造的配置项名**：TensorRT-LLM 里 `kv_cache_host_memory_bytes` **不存在**（源码/文档/10 个 release 引用 0 命中）；真旋钮是 `kv_cache_config.host_cache_size`，默认 0。**凡引用旋钮名，必须在源码里 grep 到它。**

---

## 5. 判据框架（三方向）与两条反转

任何候选缝必须**三个方向都清**：

> **A（文献 → 引擎）**：文献前沿相对已 ship 的引擎是超前、持平、还是落后？
> 只有"文献确实沉默"才算。**引擎落后于文献 = 产品缺口，不是研究缺口。**
>
> **B（引擎 tracker → 文献）**：同一决策上是否已有 ≥2 个并行开放提案？
> **文献沉默 ≠ 无人竞争**（vLLM RFC `#54749` 逐字：*"Six different signals are being proposed for this one decision right now. Batch size is what ships."*）。
> ⚠️ 已知弱点：GitHub issue 搜索页因 GraphQL 返回 `"preloaded_records":{}` 而**不可完整枚举**，所以方向 B 的**否定**结论只是下界。
>
> **C（扩展点）**：该决策是否已是公开 plugin/policy hook？若是，在此处创新等于写插件。

**反转一**：在**引擎原生位置**，**低论文数是警告信号，不是空缺**——那里的工作以 PR 形式落地。
**反转二**：**一个 open PR 证明的是"引擎项目已经在跑"，不是"这个领域还空着"。**

---

## 6. 预注册决策规则

1. E0 **必须先测量级**（而非先做机制）；**相对"部署中实际在跑的配置" <8% ⇒ 终止**。
2. 任一候选只靠"我没搜到"支撑 ⇒ 直接判死。
3. 出现 ≥2 个通过 E0 的候选 ⇒ 取量级最高者深入。
4. 全部未通过 ⇒ **停止提名机制，改为要求放宽约束**（硬件 / 时间 / 领域 / 贡献类型）。

---

## 7. 引擎发货默认值的取证结论

用于回答"什么是部署基线"——**结论是：多数引擎不发货默认值，而是要求使用者自己挑。**

| 引擎 | 真实情况 |
|---|---|
| **vLLM** | `num_speculative_tokens: Field(default=None)`——**无数字默认值**（从草稿 checkpoint 解析，否则抛 ValueError）；`num_speculative_tokens_per_batch_size = None`——**出厂无表**（常被引用的 `[[1,4,2],[5,512,0]]` 是 **issue 报告者自己的配置**）；**无自动调参** |
| **SGLang** | 三个参数都是 `Optional[int] = None`；真实默认由 **`_auto_choose_speculative_params` 按模型架构**决定（Llama/Grok1 → (5,4,8)；DeepSeek/MoE/GLM/MiMo/其余 → (3,1,4)），**从不看 batch**；`speculative_adaptive = False`。⚠️ 其文档的"built-in default"块与代码里的 `DEFAULT_ADAPTIVE_CONFIG` **不一致**（漏了 BS64 槽、隐藏了 K=0 层） |
| **TensorRT-LLM** | `max_draft_len` **无默认值且 EAGLE 下必填**；`num_nextn_predict_layers` 是**已废弃别名**；文档用的是各算法**示例**，集合 **{3, 4, 6}**——**任何取到的正文里都不存在 `5`**。另有**单向永久关闭闩锁**（`acceptance_rate_window_size/threshold`） |
| **LMDeploy** | 静态 `num_speculative_tokens = 1` |
| **llama.cpp** | 当前默认 **3**；`16→3` 是真的（commit `d14ce3d`，PR `#23269`，5.33×），**但该 PR 正文无任何 benchmark**，且 `16` 本身是三天前由 PR `#22673` 引入的 ⇒ **是 churn，不是实测修正** |
| **Dynamo** | 未验证 |

---

## 8. 我自己被撤回的错误（留档）

| 我说过 | 事实 |
|---|---|
| `#49548` 证明"已 ship 的默认值是坏的" | **参与者 `Suppressor72` 撤回了因果论断**（`#51466` 以不支持关闭；实测中位差 −0.9%~−1.7%）；**且 vLLM 根本没有表**。撤回者是参与者，**不是** issue 作者 |
| "TensorRT-LLM 为同一旋钮发了三个冲突默认值（3/4/5-and-3）" | **按原文是错的**——无默认值、别名、示例集合 {3,4,6}，**没有 5** |
| "llama.cpp 把默认从 16 砍到 3" 当作实测修正 | 是**未做 benchmark 的 churn** |
| "张量布局上限 ≤1.23×" | vAttention 的**容量**收益，不是带宽 |
| SpecBudget 的 headroom 是 "3–16%" | **我自己的复合值**（把三篇不相关论文拼在一起），**没有任何来源包含它** |
| "SGLang `AdaptiveStepSlot` 已 ship" | **默认关闭**（`speculative_adaptive=False`） |
| SVIP 属 self-speculation | **SVIP 是专用 drafter 方法** |
| "jump-forward 只有 0.04%" | 算的是 **mask 成本**，不是跳步收益 |
| TechRxiv《Transactional KV Caching…》是 phantom | **它真实存在**（`10.36227/techrxiv.177101038.80960856/v1`；Crossref 200 / OpenAlex W7128772398）。我查的是**截断 ID**；`prefixes/10.36227/works` 对照（30,954 条）**探测不到前缀内截断** |
| `TODO(woosuk) … "arbitrarily chosen"` 指 γ | 那说的是 **ngram 窗口大小** |

**方法论自省**：我写下了"必须检查同线程撤回"这条规则，**却在框架文档里自己违反了它**——因为我的取证脚本用 tag-stripping，而撤回文本只在 payload 里。

---

## 9. 未决与未验证（不得当作结论）

| 项 | 状态 |
|---|---|
| **Libra**（ICLR 2026，OpenReview `WhxNwgGkAS`） | **UNRESOLVED**（OpenReview 403 拦截）。二手来源称其做"下一层专家激活的投机预测 + 热专家复制"，8×H200 |
| **TransKV**（= 上述 TechRxiv 论文）正文 | **不可读**（Cloudflare 403 × 8 种方法，Wayback 0 快照）。**仅元数据级验证** |
| **Loquetier** `2511.00101` 正文 | ar5iv fatal error + SSL EOF，仅摘要与 NeurIPS'25 场地 |
| **CoLoRA** | 产物是 **18 页幻灯片（4,401 字符）**，不是论文——数字不可审计 |
| SIGCOMM 2026《Balancing and Beyond…》`10.1145/3789240.3829201` | 付费墙，未取全文 |
| Dynamo 的 router 源码 | 仅厂商文档，非自测 |
| Mooncake `admission_threshold` | 仅文档 |
| llm-d GDS rollout versioning；KVBM `frequency>=2` 是否在 `main` | 未验证 |
