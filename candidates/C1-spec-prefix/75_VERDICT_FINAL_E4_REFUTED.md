# C-1 最终判定（第三次修订，2026-09-15）：E4 主现象**被推翻**；留下一个真实的度量学结果

- 硬件：`ssh -p 38300`（RTX 4080 SUPER 32GB，128 核），vLLM 0.29.0，torch 2.12.1+cu130
- 模型：dense `Qwen3-4B`；hybrid `Qwen3.5-4B`（GDN 线性注意力 + 全注意力）；hybrid `Falcon-H1-3B`
- 原始数据：`results/C1m-step-cost/2026-09-15/`（`c1m_step_cost.json`、`c1m_falcon_patched.json`、`c3b.json`）
- 探针：`probes/c1l_ground_truth_steps.py`、`c1m_step_cost.py`、`c1n_coverage.py`、`c3b_invariant_detector.py`

> 本文**推翻** E4 主现象（"hybrid 上投机解码净亏、接受率反而更高"），并**取代** C-1j 及其全部修订。
> 保留全过程是为了让"错在哪一步"可审计——本课题线的错误链条本身是比结论更有价值的产出。

---

## 0. 结论先行

1. **E4 主现象不成立。** 在干净口径下（同一 prompt、`n=1`、丢首轮、真实步数）投机解码在
   **三个家族上全部为正收益**：dense 1.165×、hybrid(Qwen3.5) 1.155×、hybrid(Falcon-H1) 1.070×。
   "hybrid 上净亏"是 C-1e 归一化口径错误的产物（见 §3）。
2. **不是机制发现。** hybrid 与 dense 的**每步耗时几乎不随草稿长度 K 变化**
   （dense 23.5→24.7 ms，Qwen3.5 34.3→35.9 ms，Falcon 42.7→43.8 ms），
   即"验证更多草稿 token 几乎免费"。这一点**已公开**：
   [LMSYS/SGLang 2026-08-21 博客](https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell)
   在 Ling-3.0-flash（hybrid MoE，4×Blackwell）上明确写出
   *"Weight-bandwidth dominance has a counter-intuitive corollary: verifying more tokens is nearly free."*
   ⇒ 我这边的"机制定位"**没有新颖性**。
3. **留下一个真实的、可复现的度量学结果**（§4）：vLLM 0.29 的
   `step` 口径极易被误用，而**正确口径需要引擎内部计数器**；
   用错误口径会得到方向完全相反的结论（我本人连续错了三次）。

---

## 1. 错误链条（完整留档）

| # | 我曾经的结论 | 依据 | 为什么错 |
|---|---|---|---|
| 1 | C-1j：**slope 5.6×**，hybrid 每草稿 token 边际成本高 | `t_step = wall / sum(histogram)` | 把"起草步数"当成了"总步数" |
| 2 | C-1j §7：改为 **17.1×**，方向不变量级更强 | 用恒等式 `steps = gen/A` 反推 | 仍是同一口径错误的第二次表现 |
| 3 | C-1l/C-1m 初读：histogram **低估 4–9×** | 打桩得 88–99 步 vs histogram 6–21 | **这个"低估"判断本身也是错的**（见 §2） |
| 4 | C-1e：`A* = tok/s(无投机)/tok/s(零接受)` | 归一化 | 零接受臂走回滚/状态恢复，步数膨胀，基线不可用 |

## 2. 正确口径（这一节是唯一需要记住的）

给引擎循环打桩，并证明该计数与厂商自己的计数器一致：

```python
# 在 import 期打桩；spawn 出的 EngineCore 子进程同样生效
EngineCore.step / EngineCore.step_with_batch_queue  ->  向文件追加一个 tick
```

**恒等式（全部 6 个 cell 精确成立）**：

```
vllm:spec_decode_num_draft_tokens / K  ==  sum(histogram)      <- 精确相等
```

| cell | draft_tokens | ÷K | sum(histogram) |
|---|---|---|---|
| dense \| K=2 | 22 | 11.0 | 11 |
| dense \| K=3 | 33 | 11.0 | 11 |
| dense \| K=4 | 44 | 11.0 | 11 |
| hybrid \| K=2 | 14 | 7.0 | 7 |
| hybrid \| K=3 | 18 | 6.0 | 6 |
| hybrid \| K=4 | 24 | 6.0 | 6 |

⇒ **`sum(histogram)` 不是"总步数"，也不是"被低估"：它精确等于"真正发起了起草的步数"。**
一次解码步有两种：

- **起草步**（ngram 找到匹配）：提议 K 个 token，验证后接受 `A_cond` 个 → 计入 histogram
- **普通步**（ngram 没找到匹配）：只出 1 个 token → **不计入 histogram**

所以 `A × sum(histogram) = 总 token 数` **本来就不该成立**，我拿它当"恒等式检验"是错的。
`draft_acceptance_rate` 的分母（draft_tokens）是对的 ⇒ **接受率指标没有被破坏**。
被破坏的只有"拿 `sum(histogram)` 当总步数"这一种用法。

**这直接解释了为什么投机收益只有 1.15×**：

```
speedup = (1 - M) · 1  +  M · A_cond        M = 起草步占比
```

ngram 只在一小部分步上找到匹配（本工作负载 M≈0.13–0.20），
所以即使 `A_cond` 高达 1.8–2.5，端到端也只有 ~1.15×。

## 3. 干净测量（C-1m：同 prompt / 同 prefix / n=1 / 丢首轮 / 真实步数）

| cell | tok/s | 步数(打桩) | ms/步 | tok/步 | A_cond | accept |
|---|---|---|---|---|---|---|
| dense \| K=0 | 41.3 | 99 | 23.49 | 0.970 | — | — |
| dense \| K=2 | 47.4 | 88 | 23.00 | 1.091 | 1.82 | 0.41 |
| dense \| K=3 | 44.2 | 88 | 24.71 | 1.091 | 1.82 | 0.27 |
| dense \| K=4 | 48.1 | 88 | 22.68 | 1.091 | 1.82 | 0.20 |
| hybrid \| K=0 | 28.3 | 99 | 34.25 | 0.970 | — | — |
| hybrid \| K=2 | 30.1 | 89 | 35.86 | 1.079 | 2.14 | 0.57 |
| hybrid \| K=3 | 31.6 | 88 | 34.53 | 1.091 | 2.50 | 0.50 |
| hybrid \| K=4 | 32.7 | 88 | 33.35 | 1.091 | 2.50 | 0.38 |
| falcon \| K=0 | 22.7 | 99 | 42.74 | 0.970 | — | — |
| falcon \| K=3 | 23.6 | 93 | 43.76 | 1.032 | 1.44 | 0.15 |
| falcon \| K=4 | 24.3 | 93 | 42.44 | 1.032 | 1.44 | 0.11 |

**相对于各自的无投机基线：dense 1.165× / hybrid 1.155× / falcon 1.070× —— 全部为正。**
**`ms/步` 在 K=2..4 上基本平坦（斜率分别为 −0.16 / −1.25 / −0.42 ms/draft，量级都在噪声内）。**

> **边界**：K=0 的每步耗时（23.5 / 34.3 / 42.7 ms）里混有**模型本体**差异
> （3B vs 4B、是否含视觉塔）。因此"hybrid 每步贵 1.47×"**不能**归因于混合架构本身，
> 只能读到"每步耗时随 K 平坦"这个**族内**结论。

## 4. 留下来的真结果：一个度量学陷阱

vLLM 0.29 里同时存在三个都叫"步"的量，且只有一个是总步数：

| 量 | 含义 | 来源 |
|---|---|---|
| `sum(histogram)` / `num_spec_steps` | **起草步**数（非总步数） | 按请求累加器，仅 `scheduled_spec_token_ids and generated_token_ids` 时触发 |
| `vllm:spec_decode_num_drafts` | 总迭代数 | 引擎 Prometheus 计数器 |
| `vllm:iteration_tokens_total` count | 总迭代数 | Histogram("tokens per engine_step") |

**为什么这个陷阱值得写**：用 `sum(histogram)` 当分母会**翻转结论方向**——
我据此连续得到 5.6× 和 17.1× 的"hybrid 边际成本劣势"，而真实值是 ~0。
`disable_log_stats` 在 `vllm/entrypoints/llm.py:228-229` 被**默认置 True**，
使 Prometheus 计数器读数为 0，从而把使用者推向那个错误的分母。

**当前状态**：这是**候选**结果，尚未立题。要成为课题还需：
(a) 向上游确认这是否为已知/有意设计（`observe()` 的触发条件是否有意只统计起草步）；
(b) 在所有 spec 方法（eagle3/mtp/dflash）上复核，而不是只有 ngram。

## 5. 顺带确认的一个上游缺陷（C-3b 检测器的预测被实证命中）

Falcon-H1-3B + 投机解码在 vLLM 0.29 上**必然**崩溃：

```
vllm/v1/kv_cache_interface.py:879:  assert self.page_size_padded >= page_size
```

**且与 prefix caching 无关**（`pc=False` 与 `pc=True` 都崩）——
这修正了此前"需要 prefix caching 才触发"的判断。

- 根因（与 C-3b 静态检测器**事前预测**一致）：`falcon_h1.py:540` 的
  `get_mamba_state_shape_from_config` 调用 `mamba2_state_shape(...)` 时
  **不传 `num_spec`**，而消费端 `MambaSpec.page_size_bytes` 会乘 `(1 + num_speculative_blocks)`。
- 对照：`qwen3_5.py:405-417` 读了 `speculative_config.num_speculative_tokens` 并转发 ⇒ 不崩。
- **一行修复已验证有效**：把 `num_spec` 转发进 `mamba2_state_shape` 后
  Falcon-H1 + K=3 成功构建（`falcon_patch`，本次测量即用它取得 Falcon 数据列）。
- C-3b 检测器在 0.29.0 上报告 **13 个 SPEC_UNAWARE 生产方 / 8 个 SPEC_AWARE**，
  消费端确实计 spec blocks（`kv_cache_interface.py:857,887,891,894`）。

## 6. 复现

```bash
# ssh -p 38300 root@connect.westc.seetacloud.com
# 上传 probes/{c1l_ground_truth_steps,c1m_step_cost,c1n_coverage,c3b_invariant_detector}.py 到 /root/autodl-tmp/probes/
cd /root/autodl-tmp

# 1) 真实步数 + 每步成本（打桩必须开）
C1M_PATCH=1 C1M_STEP_FILE=/tmp/c1m_steps.txt \
  /root/miniconda3/bin/python probes/c1m_step_cost.py \
  --out /root/autodl-tmp/c1m_step_cost.json        # --ks 0,2,3,4 --passes 3

# 2) 第二家族（Falcon-H1 需先应用 num_spec 修复，否则必然崩溃）
C1M_PATCH=1 C1M_FALCON_PATCH=1 \
  /root/miniconda3/bin/python probes/c1m_step_cost.py --targets falcon \
  --models "falcon=/root/autodl-tmp/models/Falcon-H1-3B" --prefix-caching 0 \
  --out /root/autodl-tmp/c1m_falcon_patched.json

# 3) 静态不变量审计（零 GPU）
/root/miniconda3/bin/python probes/c3b_invariant_detector.py \
  /root/miniconda3/lib/python3.12/site-packages/vllm --json /root/autodl-tmp/c3b.json
```

## 7. 对课题线的后果

| 层 | 状态 |
|---|---|
| E4 主现象（hybrid 上投机净亏+接受率更高） | **推翻**（三家族全部正收益） |
| "hybrid 每草稿边际成本高" | **推翻**（~0；且该机制已被 LMSYS 公开） |
| "每步耗时随 K 平坦" | 成立，但**非新颖** |
| hybrid 每步贵 1.47× | **不可归因**（混有模型规模差异），降级为观察 |
| C-1j / C-1e / C-1i 的机制结论 | 全部作废（同一口径错误） |
| vLLM `step` 口径陷阱 | **候选**，需按 §4(a)(b) 补完 |
| Falcon-H1 + spec 必然崩溃 + 一行修复 | 已实证，属上游 #47635 范畴，非本课题新发现 |
| C-3b 静态检测器（13 命中，1 个已实证命中） | 检测器有效，但需要更多实证命中才能立题 |

**当前没有任何一个候选通过全部闸门。** 诚实结论：C-1 线上我追的"机制"是度量伪影，
E4 主现象不成立；本课题线目前**没有**可立项的课题，除非 §4 的度量学结果能补完。
