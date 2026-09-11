两个后台核查代理已停止（其剩余事项对结论无影响，决定性声明我已全部自己核实）。审核完成，未修改任何文件。

---

# 裁决：**plan.md 的排序水平明显更高；评估水平上 plan.md 也更严谨，但 kimi_plan.md 抓到两个真问题**

一句话：**排序上 plan.md 是"可复现的规则"，kimi 是"不可复现的判断"；评估上 kimi 有 2–3 处真实修正，但同时制造了 3 处硬错误——净额仍低于 plan.md。**

## 一、审核方法（不是文本对读）

我把两份文件里**决定排序的承重声明**逐条拉回本地证据库（2.2 GB `archive/` + 引擎源码）与实时网络复核，并用文件时间戳判定"谁当时拿得到什么"。关键时间线：

| 证据 | 落地时间 | 早于 plan.md(15:18)? |
|---|---|---|
| `archive/prefixshield.txt`（PrefixShield **全文**） | 09:20 | ✅ |
| `archive/subfield_scan/constrained_decoding.md`（T5 结论） | 12:36 | ✅ |
| `archive/dspark_deep_read.md`（DSpark 深度扫描） | 10:31 | ✅ |
| `archive/subfield_scan/rea1/k4/AngelSpec-2607.25852_body.txt`（**正文**） | 13:22 | ✅ |
| `.../REPORT_specbudget_citation_forensics.md`（LibraSpec 取证） | 13:27 | ✅ |
| Continuum `2511.02230` / SAGA `2605.00528` 摘要 | 已在 adjacent db.json | ✅ |

⇒ 两份文件面对的是**同一批证据**，差异不在信息可得性，而在使用纪律。

## 二、排序水平：plan.md 胜，且差距主要在这里

**plan.md**：排序键 4 级、等级刻度有序（立即判定→优先→可做→需先判定→暂缓→不做）、档次边界显式。我逐行验证：优先档、可做档、需先判定档的「可行性降序 → 密度升序 → 创新降序」**全部成立**，仅 2 处完全并列（#8–#10、#18–#19）属键未定义的无害 tie。

**kimi_plan.md**：排序键的第一级「综合建议」是自由文本标签（主线候选A / 快速判定 / E1 后定生死 / 备线 / 先测… / 可做… / 接近否决 / 暂缓偏不做），**没有定义等级顺序**，所以次级键在多数块内根本不生效；一旦按其**自己的**次级键检查即出现明确违反：

- 排名 15 的 #17（可行性 ★★）排在排名 16 的 #16（可行性 ★★★）之前；
- 排名 17 的 #18（★★★，拥挤高）排在排名 18 的 #13（★★，拥挤中）之前；
- "可做"块（排名 8–12 = #11/#9/#12/#8/#10）可行性序列 **3,4,3,4,4**、拥挤度序列 **低,中高,高,中,中** —— 两个次级键都不单调。

⇒ kimi 的表格实际顺序**无法由它声明的规则生成**，排序退化为不可审计的个人判断。这是两份文件在"排序"这一项上最本质的差别。

## 三、评估水平：plan.md 更严谨，但 kimi 有真实战果

### kimi 赢的（我独立复核为真）
1. **#2 降档正确，是本次最有价值的一条修正。** plan.md 写"未被占的是服务端机制本身"——**错**：Continuum（`2511.02230`，Hanchen Li/Alvin Cheung/Gonzalez/Ion Stoica，2026-09-08 更新）就是引擎侧 KV TTL 保留（工具暂停期间留 KV，并同时计重算/重载成本与排队延迟），SAGA（`2605.00528`）是 workflow-atomic 调度 + 跨工具边界 KV 复用预测；两篇都在本地库里，且 `prior-art-llm-inference-mechanisms.md` 的逐字引文里**直接点了 Continuum**。plan.md 却把它排第 2 并放进 W1。
2. **#4 措辞收窄正确**：DASC `2608.30386`（混合线性注意力服务的衰减感知状态压缩）、HYPIC `2607.01299`（混合注意力服务的位置无关缓存）真实贴边，"全库零论文"（靠 5 条精确短语查询）应改为"交互问题零论文"。
3. **#6 的新颖性边界更准**：DSpark `2607.05147` 明确扫过 drafter **层数**（"drafter depth (number of transformer layers)"，2 层胜 5 层），只是离线静态；Graft §4.4 逐字 "batched serving favors fixed CUDA graphs and batch-aligned verification shapes… instead of dynamically changing tree depth"。两份材料本地都有，**plan.md 的"从未调过 drafter 的深度"过强且未引用**。
4. 新增「MLSys 契合度 / 量级证据等级（实测·算术·拼接）」两个维度确实有用；vLLM `#53912` 我核实存在且标题即 "[Bug]: prefix caching + MTP still corrupts output on hybrid Mamba/GDN models"（[issue #53912](https://github.com/vllm-project/vllm/issues/53912)）。

### kimi 输的（我独立复核为假/误）
1. **三条"引用不实"裁定全是错的**，而它列出的输入里就包含 `archive/`：
   - **SpecTool** 引的是 Nichols et al. `2512.15834`（本地全文 460,562 B + §3.2 逐字引文；[Semantic Scholar 确认](https://www.semanticscholar.org/paper/Optimizing-Agentic-Language-Model-Inference-via-Nichols-Singhania/9b260665a76e31a07bcb99234fa21dadbca8144f)）。它说的 `2411.13547` 是同名的另一篇 —— 名字撞车，不是引用错。
   - **AngelSpec** = `2607.25852`：本地 13:22 就有**正文全文**，外部亦有[腾讯开源报道](https://www.aibase.com/news/30001)。
   - **LibraSpec** = `2608.08721`：本地 13:27 取证报告连 abs 页字节数（42,471 B / HTTP 200）都记录了，[arXiv 页](https://arxiv-org.ezproxy.obspm.fr/html/2608.08721v1)亦在。
   - §7 那条"找不到出处就删除"若照做，会删掉两条**正确且承重**的引用（LibraSpec 是 #18 判定的支柱，AngelSpec/DFly 恰是"D-cut 那件事"的更早证据）——这是**破坏**项目在 `GPT.md` §7 / `old.md` §2 里花了很大代价换来的更正。
2. **量级误植**：#5 的"engine 侧实测仅 2–3%"不是 #5 机制的测量，而是 SpecTool 在 agent 工具投机上的 engine 增量；`GPT.md` 自己把 #5 幸存增量标为 **O8 估计（单位数 %，无任何实测）**。kimi 把"无人测量"写成了"实测"。
3. **"本轮新增三个独立检索通道、全部经 arXiv 原页核实"没有留下任何产物**：全树在 15:00–17:00 只有 `GPT.md`/`plan.md`/`kimi_plan.md` 被写，最新证据产物停在 14:28；其"新发现"的 Sparse Prefix Caching/DASC/HYPIC/VeriCache 早在本地 `arxiv_sweep*.json`、`adjacent/db.json` 内，Graft/DSpark/AngelSpec 亦然。**真正本地不存在的只有两个 ID**：D-cut `2607.14647`、DA-MoE `2607.23099`（后者确认真实存在，[arXiv 2607.23099](https://arxiv-org.ezproxy.obspm.fr/html/2607.23099v2)，但"已合入 FlashInfer"我未能独立确认）。
4. **覆盖回退**：plan.md 的 **C1–C9 九个探针（合计约 1 周、含零成本判定）在 kimi 里完全不出现**；资源/硬件表消失；20 项中 16 项被压成一句话。
5. **机制类比滑移**：Graft 的 "dynamic depth" 是 draft **树**深度/验证预算，不是 drafter **网络**深度——把它当作"网络深度自适应被放弃"的直接先例，正是两份文件都同意的第 5 条规则（机制混淆）要防的错。

### 两份都漏了的（我的独立发现，对最终排序有实质影响）
- **#3 被两份文件同时高估**：`constrained_decoding.md`（12:36，早于两份文件）的 T5 结论是"**这不是一个藏着缺口的子领域…现在进入等于在结果之后到达**"，字段级端到端余量 **≤6%**，三个 gap 全 KILLED（上限 1–2% / 1–6% / ≤2.5%）。该结论在 `plan.md`/`V41.md`/`GPT.md`/`kimi_plan.md` 里**零引用**，而两份都把 #3 排第 3 并标"优先/快速判定"——plan.md 的 Go 判据还要求 ≥15% 余量（≤0.85×）。plan.md 用 KV 子领域的 "DEAD" 结论杀掉了 GPT-④，却没把约束解码子领域的结论带进排序层：**同一份语料、两套标准**。
- **#2 的真正杀手不是"被占"而是基线**：本地负面结果（68,266 请求 / 393 会话回放）逐字 "TTL-300s produced **byte-identical** results to LRU-leaf in every single run"、"**LRU-leaf is a stronger baseline than the literature treats it as**"，以及 `rea_4_legacy.md` 自给的 "realistic band: **10–30%** of re-prefill cost, not 89%"。plan.md 只引 28.4%→98.65%，kimi 只换了个降档理由——**两边都没把这条最强的反向证据摆上桌**。

## 四、量化评分（1–5）

| 维度 | plan.md | kimi_plan.md |
|---|---|---|
| 排序规则设计 | **5** | 2.5 |
| 排序可复现性（自洽） | **5** | 2 |
| 逐项证据分级与判据 | **5** | 3 |
| 引用核查准确率 | **4.5** | 2 |
| 覆盖完整性（对同一语料） | **4.5** | 2.5 |
| 新增决策信息 | 3 | **4** |
| 可执行性（探针/资源/门槛） | **5** | 3.5 |
| 校准诚实度 | **4** | 2.5 |
| **均值** | **≈4.5** | **≈2.8** |

## 五、建议：以 plan.md 为骨架 + 三个补丁，并明确丢弃三条错误裁定

1. **#2 降档**并按 Continuum/SAGA + "TTL≡LRU / 10–30% 实际band" 重新定位（kimi 方向对，理由要换）。
2. **#3 把 `constrained_decoding.md` 的 T5（≤6%、已收敛）写进排序理由**，否则"优先"不成立；plan.md 的 Go 判据应从"≤0.85×"改为"先量 base rate 再决定是否值得测"。
3. **#4/#6 采用 kimi 的收窄措辞**（DSpark 离线深度扫描、Graft 的 CUDA-graph 约束、DASC/HYPIC 邻域），并把 `#53912` 当**动机**而非**缺口**——同类修复 PR（`#47861`、`#45477`）已在飞，引擎 bug 会过期。
4. **明确作废** kimi 的 SpecTool/AngelSpec/LibraSpec 三条"引用不实"裁定——它们是错的，照做会污染已经校正的证据链。

**结论：`plan.md` 的排序与评估水平更高。** `kimi_plan.md` 的价值不在它自述的"三个检索通道"，而在它把**旧证据重新整合**时抓到的那两处边界（#2 占位、#6 的 DSpark/Graft 边界）——这部分应当吸收，其余（排序框架、引用裁定、新增证据声明）不可采信。