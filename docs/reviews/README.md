# docs/reviews/ — 评审记录

对方案文档的独立评审，按时间追加。新文件命名：`YYYY-MM-DD-verdict-NN-<slug>.md`。

| 文件 | 评审对象 | 结论 |
|---|---|---|
| [`2026-09-11-verdict-01-ranking-review.md`](2026-09-11-verdict-01-ranking-review.md) | `docs/plan.md` vs `docs/kimi_plan.md` | plan.md 的排序水平明显更高；kimi_plan 抓到 2 个真问题（#2 占位、#6 的 DSpark/Graft 边界），但制造 3 处硬错误（SpecTool / AngelSpec / LibraSpec 的"引用不实"裁定全部作废） |
| [`2026-09-11-verdict-02-specdec-adaptive-survey.md`](2026-09-11-verdict-02-specdec-adaptive-survey.md) | 外部「投机解码自适应控制」调研报告 | 核心判断成立（#6 周围拥挤但本轴未占）；MLSys '26 三条链接 404 但论文真实（改记 `mlsys.org/virtual/2026`）；漏引擎侧证据；**#6 维持主线**，主假设升级为正交性 |

## 评审已确认的事实（供后来者免于重复核查）

**已核实为真**

- Continuum `2511.02230`（引擎侧 KV TTL，工具暂停期间保留）、SAGA `2605.00528`（workflow-atomic + 跨工具边界 KV 复用预测）、PrefixShield `2608.01657`（多租户 prefix 准入责任记账）—— 三者真实，且**在 plan.md 成文前就已在本地库中**。
- DSpark `2607.05147` 确实扫过 drafter **层数**（"a 2-layer DSpark outperforms the 5-layer DFlash baseline"），但只做**离线静态**选择。
- Graft `2605.20104` §4.4 逐字："batched serving favors fixed CUDA graphs and batch-aligned verification shapes… instead of dynamically changing tree depth"（注意：其 "dynamic depth" 是 draft **树**深度，不是 drafter **网络**深度）。
- vLLM `#53912` 标题逐字："[Bug]: prefix caching + MTP still corrupts output on hybrid Mamba/GDN models in v0.28.0 (#43559 closed but unfixed)"；`#55766` 存在（hybrid GDN prefix-cache 命中后 NaN logits）。

**已核实为假（不得再引用）**

- `kimi_plan.md` §1/§7 的三条引用裁定：SpecTool = `2512.15834`（有本地全文与 §3.2 逐字引文）、AngelSpec = `2607.25852`（本地有正文全文）、LibraSpec = `2608.08721`（本地取证报告含 abs 页字节数）。
- "#5 engine 侧实测仅 2–3%"：这是 SpecTool **工具投机**方向的数字，不是 #5 机制的测量；`docs/GPT.md` 把 #5 幸存增量标为 **O8 估计（无实测）**。
- `kimi_plan.md` 宣称的"本轮新增：三个独立检索通道"：全树在 15:00–17:00 只有 3 个文件被写，最新证据产物停在 14:28；其"新发现"多数已在 `archive/` 的 `arxiv_sweep*.json`、`adjacent/db.json` 中（含 D-cut `2607.14647` 的**全文**）。

**本轮关闭**

- `arXiv 2606.29223` = DEX《Depth Exploration for LLM Decoding》⇒ **不占 #1**；作为 #6 的邻近先例，触发 DEX 替换测试（`EXECUTION_PLAN.md` §5.3）。
