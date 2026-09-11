# C3 — 约束解码 dynamic × complex × concurrent 扫描（#3 判定探针）

**时间盒**：1 周 ｜ **硬件**：单卡 ｜ **不改引擎代码** ｜ **预登记**：`notes/prereg/p03-grammar-scan.md`

> **定位：低概率彩票，不是主线。** 任何结论都必须同时写出下面的先验。

## 1. 先验（必须写进 summary 第一段）

来自 `docs/evidence/constrained_decoding.md`（12:36 冻结）：

- XGrammar-2 自报端到端 **"the gap between the result of XGrammar 2 and the result without constraints is no more than 6%"**。
- 该子领域三个候选 gap **全部 KILLED**，上限 **1–2% / 1–6% / ≤2.5%**。
- T5 原文：**"this is not a subfield where a gap is hiding … Entering now means arriving after the result."**
- 唯一未解的是**复杂 CFG**（C++ / Python 变体在 XGrammar2 上 ~2–10× 慢），而 CFGzip 自报 7.5× 已在其上。

⇒ 本探针要回答的是：**"dynamic（逐请求）× complex（C++/Python 级）× concurrent（batch ≥32）"这个三交集是否还有结构性空间**，而不是"约束解码还有没有问题"。

## 2. 网格

| 维度 | 取值 |
|---|---|
| batch | 1 / 8 / 32 / 128 / 256 |
| 语法类 | class-1 JSON schema（简单）／class-2 中等（嵌套 JSON + regex）／class-3 C++ 子集／**class-4 Python 变体或 Bython 类**（复杂、逐请求生成） |
| 投机解码 | on / off（同一引擎） |

- 每格记录：相对**无约束**吞吐比、mask 计算耗时、CPU 占用、端到端 P50/P95 TTFT 与 TPOT。
- 至少两个引擎/后端交叉验证（例如 XGrammar-2 与一个已有后端），避免单后端特例。

## 3. 判据

| 结果 | 动作 |
|---|---|
| **class-4 @batch≥32 ≤0.85× 无约束**，且瓶颈**结构性**（无法用缓存/预编译消除） | Go：值得进入机制设计 |
| class-4 @batch≥32 >0.85× | **归档**（余量不足以支撑一篇论文） |
| 差异存在但可被工程手段抹平（预编译、mask 缓存、schema 复用） | 退化为 measurement paper ⇒ **归档** |

**"结构性"的判定标准**（必须逐条回答）：瓶颈是否随 schema 复杂度超线性增长？是否与 batch 耦合？是否无法在请求到来前的任意时刻摊销（即逐请求动态语法的价格是否真的不可支付）？

## 4. 产物

```
results/p03-grammar-scan/<YYYY-MM-DD>/
├── summary.md     # 先验 + 数字表 + "结构性"三条回答 + 结论
├── grid.csv
├── grammars/      # 4 类语法定义（可复现）
└── run.log
```
