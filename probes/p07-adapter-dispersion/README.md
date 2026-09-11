# #7 — per-adapter KV 配额/准入：离散度探针

**时间盒**：≤1 周 ｜ 1 卡 ｜ **预登记**：`notes/prereg/p07-adapter-dispersion.md`

## 为什么先跑它

机制格是空的（SGLang `#2929` 的 21 个打勾项全是适配器权重/算子/API；两引擎都无 KV 配额/分区/准入；四篇"看似占位"的 ForkKV/aLoRA/ICaRus/LRAgent 都是**共享**机制，不是**配额策略**），
但量级只有算术估计（1.2–2× goodput）⇒ 按三条硬门槛，**③ 未过**。所以先用一个便宜探针把它变成实测值。

## 测什么

同一进程并发 N 个 LoRA adapter（N ≥ 8，冷热两端都要有），负载用同一 prompt 池：

- `hit_i` = 第 i 个 adapter 的 prefix 命中率
- **离散度** = `p90(hit) − p10(hit)`（百分点），并报 `max/min`
- 附：各 adapter 的 KV 占用分布、请求到达间隔分布

## 判据

| 结果 | 动作 |
|---|---|
| **离散度 < 5 个百分点** | **归档 #7**（准入没有抓手） |
| 离散度 ≥ 5 且存在"高占用低命中"的 adapter | 升为**备线 B**，进入配额策略设计 |

## 产物

`results/p07-adapter-dispersion/<日期>/{summary.md, hit_rate.csv, run.log}`
