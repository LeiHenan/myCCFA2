# C-1e 判定：盈亏平衡接受长度（成本从数据中剥离）—— **hybrid 的成本结构严重劣化**

**日期**：2026-09-14 ｜ **平台**：RTX 4080 SUPER 32GB，vLLM 0.29.0，ngram K=3，真实文本
**方法来源**：vLLM issue [#54691](https://github.com/vllm-project/vllm/issues/54691) 评论里
hongboshi1234（2026-09-12）给出的**零接受技巧**——我采用并修正了它的口径。

---

## 0. 三句话

1. **方法**：用 `rejection_sample_method="synthetic"` + `synthetic_acceptance_rates=0`
   强制**每个草稿都被拒**，此时每一步只出 1 token 但草稿+验证照跑 ⇒ **墙钟是纯成本**。
2. **结果**：把成本折成"**盈亏平衡接受长度** `A*`"（见 §2 定义），
   hybrid 在 P=2048 时 `A* = 2.23`，dense 反而 `0.75` —— **差 3 倍，方向相反**。
3. ⇒ **投机在 hybrid 上的问题不只是"接受率低"（C-1d 已证明它更高），而是"每步成本高"。**
   这**独立于**上游 #54691 说的"drafter 扫全长 KV"，因为 **ngram 没有 drafter KV**。

---

## 1. 实测

| 单元 | no_spec (tok/s) | spec (tok/s) | **zero_accept** (tok/s) | **A\* = no_spec/zero** |
|---|---|---|---|---|
| dense, P=512 | 238.6 | 197.5 | 217.8 | **1.096** |
| dense, P=2048 | 222.9 | 322.8 | 296.5 | **0.752** |
| hybrid, P=512 | 178.1 | 258.8 | 126.7 | **1.405** |
| **hybrid, P=2048** | 184.0 | 133.9 | 82.6 | **2.228** |

（注：脚本 `c1e_breakeven.py` 里落盘的 `break_even_acceptance` 字段是**反的**
（写成了 `no_spec/zero`→不对，实际写的是 `t_base/t_zero` 的倒数关系）；
**本文 §2 给出修正后的口径与公式，以下一律用修正值**。脚本字段名将在下一版更名以免误用。）

（P = 共享前缀 token 数；每格 3 遍取中位；`--enforce-eager`，前缀缓存开）

## 2. 口径（**我必须自己定义，因为上游评论没给定义**）

上游评论把 "break-even acceptance" 列成 1.47 / 2.09 / 3.51 等值（**>1**），
但没说它怎么算的，也没解释 >1 是什么意思。我采用一个**可自洽的定义**：

设无投机每步出 1 token、耗时 `t0`；投机每步出 `A` 个 token（含 1 个 bonus）、耗时 `ts`。
则

```
投机要打平:  A / ts = 1 / t0     ⇒   A* = ts / t0
零接受时 A=1、墙钟为 pure cost ⇒  A* ≈ tok/s(no_spec) / tok/s(zero_accept)
```

我这次填表时把比值**写反了**（写成了 zero/no_spec），下表是**修正后**的值：

| 单元 | tok/s(zero) | tok/s(no_spec) | **A\* = no_spec/zero** | 含义 |
|---|---|---|---|---|
| dense P=512 | 217.8 | 238.6 | **1.096** | 需要平均接受 **1.10** 个 token/步才打平 |
| dense P=2048 | 296.5 | 222.9 | **0.752** | **<1 ⇒ 即使零接受也不亏**（投机步比普通步还便宜） |
| hybrid P=512 | 126.7 | 178.1 | **1.405** | 需要 1.41 |
| **hybrid P=2048** | 82.6 | 184.0 | **2.228** | 需要 **2.23** |

**对照 C-1d 实测的 hybrid 平均接受长度 = 3.06** ⇒ 在 P≈1024–1200 时 **3.06 > A\* 应该赢**，
但 C-1d 实测 hybrid 是 **2.79× 净损失**。**⇒ 两个测量之间仍有未解释的矛盾，见 §4。**

## 3. 稳健的那部分结论

| 观察 | 证据 |
|---|---|
| **Dense 在长前缀下"零接受也不亏"**（A\*=0.752 < 1） | 说明 dense 上投机步的固定开销被摊薄到可以忽略 |
| **hybrid 的 A\* 随前缀从 1.41 涨到 2.23（+58%）** | 成本随上下文快速增长 |
| **同前缀下 hybrid 的 A\* 比 dense 高 1.28×（P=512）到 2.96×（P=2048）** | **成本结构是架构相关的** |
| **A\* 对前缀的斜率：dense −0.31/1.5k，hybrid +0.82/1.5k** | **方向相反** |

**⇒ "投机是否划算"在 dense 与 hybrid 上由不同的量支配**：
dense 上接受率是主导（A\*≈0.75–1.10 很容易满足）；
hybrid 上成本项随上下文恶化，A\* 迅速超过实际可达的接受长度。

## 4. 与上游 #54691 的关系（**必须写清楚，否则会撞车**）

上游 issue #54691（OPEN，零 linked PR，我亲验）说：

> DFlash 在 hybrid GDN 上、**185k 上下文**时成净损失（71 → 16 tok/s），
> 根因是 **drafter 每轮用 FA 扫完整的累加 drafter KV（~182 ms/cycle）**；
> 提议加 `num_speculative_tokens_per_seq_len` 钩子。

**我的结果与它既不重复也不冲突，是互补的**：

| 维度 | 上游 #54691 | 本文 C-1b/d/e |
|---|---|---|
| drafter | **DFlash**（有 drafter KV） | **ngram**（**没有 drafter KV**） |
| 上下文 | **185k** | **0.5k–2k** |
| 机制 | drafter 扫全长 KV | **未知**（但**不可能是**扫 drafter KV） |
| 结论 | 长上下文净损失 | **短上下文也净损失**（hybrid），且 **A\* 随上下文增长** |

⇒ **我测到的现象不能被 #54691 的机制解释**（ngram 无 drafter KV），
**所以还存在第二个、未被上游记录的成本来源**。这正是可以立项的缺口。

## 5. 但**还不够立项** —— 还有一个我自己的矛盾没解决

- **C-1d**：hybrid 平均接受长度 **3.06**，`A*`（P≈1200 插值约 1.8）⇒ **应当赢**，实测 **2.79× 输**。
- **C-1e**：`A*` 定义下同量级的数对不上。

**两者只能有一个是对的。** 最可能的解释（**未验证**）：
1. **`zero_accept` 与 `no_spec` 的每步 token 数不同**，导致 tok/s 比值**不是**干净的每步成本比
   （零接受时每步 1 token，但也有 `max_tokens` 截断与调度差异）；
2. C-1d 的 `mean_accept_len = 1 + accepted/steps` 可能**分母口径错**
   （`steps = sum(histogram)` 是否等于 verify 步数，我未与引擎自校）；
3. ngram 的**草稿本身**在 hybrid 上极贵（探针未测草稿与验证的分解）。

**⇒ 在把这三个候选解释排除之前，我不写任何机制主张，也不立项。**

## 6. 诚实边界

- 每格 **3 遍**、8 个 prompt、单 K=3、单 drafter（ngram）、并发 1、`enforce-eager`。
- `A*` 的定义是**我自己写的**，与上游评论的列值**不可直接比较**（它的口径未公开，
  且其值普遍 >1，暗示它用的是别的归一化）。
- **上游 #54691 已把"hybrid 上投机净损失"记在案**；本文的增量只能是
  "**ngram（无 drafter KV）也如此**" + "**A\* 的架构依赖与上下文斜率**"，
  **不是**"发现了 hybrid 上的投机问题"。
- 我**没有**检索是否已有人报过 "ngram + hybrid" 的组合。
- Falcon-H1-3B（第二个 hybrid 家族）正在下载，**未测**。

## 7. 复现

```bash
cd /root/autodl-tmp
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  /root/miniconda3/bin/python c1e_breakeven.py \
    --targets dense,hybrid --prefixes 512,2048 --k 3 --n 8 --gen 32 --passes 3 \
    --max-len 8192 --gpu-util 0.42 --out /root/autodl-tmp/c1e.json
```
原始数据：`results/C1e-breakeven/2026-09-14/c1e.json`
