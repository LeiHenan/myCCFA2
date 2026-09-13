# p22 —— 对上游「命中率崩塌 = 异构 block 的 lcm 溢出」这一**因果断言**的**匹配对照**

**日期**：2026-09-15 ｜ **目标**：`goal-a729588f`（引擎执行代价 / 投机解码 / KV Cache 优化）
**预登记**：无独立预登记文件 —— 本条是 **FILTER 附录 A.6 那个"新机会类型"的首次实际执行**（见 §6 纪律说明）
**成本**：**≈0.2 GPU·h**（1 次 smoke + 4 次 serve，全部清理，末显存 0 MiB、无残留进程）
**仪器**：`probes/p22-dense-hit-collapse/{hit_metrics.py,run_hit_concurrency.sh}`（`hit_metrics.py --selftest` 通过）
**raw**：`/root/ccfa_results/2026-09-14/p22_hit_conc/`

---

## 1. 被检验的断言（上游原文）

vLLM 地图 **C109** 的关闭理由，引 #42948 线程上 @stecasta 的 block-pool 插桩：

> *"@stecasta's block-pool instrumentation on the #42948 thread shows **the real bug is 131% pool overflow from the homogeneous `lcm=256`
> physical block layout vs DSv4-Flash's `{256, 64, 4, 8}`-block-size KV groups**. Under that overflow,
> some eviction every alloc cycle is mandatory; the recent-hit set in this PR can only delay the eviction of blocks
> that have already served a lookup hit, which by itself doesn't help the agentic-rotation case."*

配套现象（地图 **B112** / vLLM #42948）：前缀缓存命中率 **94.3% @bs1–2 → 1.0% @bs4 → 0.3% @bs8**，
且报告者给的**一般性**机制假设是 *"vLLM conflates 'free physical block' with 'cached prefix entry'"*。

**为什么这条值得复核**：它点名了一个**可单独拿掉的变量**（逐层 block size 是否异构），
而**引文里没有任何只变那个变量的对照** —— 所有既有崩塌报告（#42948、#48435、#43447、vllm-ascend 的 DSv4 移植）
**全是 hybrid / SWA 模型**（我做了独立检索确认，见 §5）。这正是 FILTER 附录 A.6 说的那类裂缝。

---

## 2. 匹配对照的设计

把被点名的变量**单独拿掉**：**dense Qwen3-4B 逐层 page size 相同 ⇒ lcm 就等于 page size ⇒ 不溢出**。
断言预测：**不该看到崩塌**。
同时把**另一个独立变量单独变动**：**池压力**（`--gpu-memory-utilization` 缩池）。于是 2×2：

| 变量 | 取值 |
|---|---|
| 逐层 block size | **均匀**（dense Qwen3-4B，唯一取值 —— 这正是"拿掉被点名变量"） |
| 池压力 | `0.55`（KV 池 **319,328** token，宽松）／`0.30`（**146,448** token，紧张） |
| 并发 | **1** ／ **8** |

**紧张池的占用率**：24 个 prompt × 4096 = **98,304** token 被缓存 + 并发 8 × 4096 = 32,768 活跃 ≈ **131k / 146k ≈ 90%** ⇒ 是**真实的池压力**，不是象征性的。
**每格两段**：`A_cold`（首次出现，必然 miss）→ `B_warm`（同序重发，应命中）。
**全部量取自 `/metrics` 两次快照做差**：`prefix_cache_hits_total / queries_total`、
`request_prefill_kv_computed_tokens_{sum,count}`（官方定义 *"new KV tokens computed during prefill (**excluding cached tokens**)"*）。

**对照纪律（关键）**：**全臂 nospec**。因为 p21 已实测 **dflash 系前缀缓存 0 命中**（vLLM #47930），
带投机去测命中率会被那个**已知缺陷**污染 —— 这正是 p21 建立的方法（按 drafter 类别分别陈述）。

---

## 3. 结果：四格**完全相同**，没有任何崩塌

| 配置 | KV 池 (token) | 并发 | `A_cold` 命中率 | `B_warm` 命中率 | `B_warm` 每请求重算 KV token |
|---|---|---|---|---|---|
| `loose_c1` | 319,328 | 1 | 0.0000 | **0.9981** | **8** |
| `loose_c8` | 319,328 | **8** | 0.0000 | **0.9981** | **8** |
| `tight_c1` | 146,448 | 1 | 0.0000 | **0.9981** | **8** |
| `tight_c8` | 146,448 | **8** | 0.0000 | **0.9981** | **8** |

**读数**：
1. **并发 8、池占用 ~90% 下，命中率仍是 0.9981**，与并发 1、宽松池**逐位相同**。
2. 0.9981 = 98,304 / 98,496 —— 缺的 192 token = 24 个请求 × **8** token，即每个请求**最后一个不完整 block**。
   `prefill_kv_per_req = 8` 在四格里**完全一致** ⇒ 残余重算是一个**固定的半块**（block_size=16 时 8 token），与并发和池压力**都无关**。
3. 冷态四格也都是 0.0000 / 4104（= 4096 + 8），即完整 prefill —— 说明"热态命中"不是靠什么副作用，而是真的复用了。

**⇒ 匹配对照支持上游归因**：在**逐层 block size 均匀**的 dense 模型上，
**即使 bs=8、池占用 ~90%，B112 式的命中率崩塌也不出现**。
⇒ 该崩塌**需要异构 block size**（hybrid/SWA），**不是**"自由块与已缓存条目混淆"这一般性机制在起作用（至少在本文这个压力区间内不是）。

---

## 4. 这条结果改变了什么

| 之前 | 现在 |
|---|---|
| B112 被我的分诊工具按**关键词**（`hybrid`）判为"够不着" | **机制上不可达** —— 根因（异构 block 的 lcm 溢出）在我们 dense 目标上**结构性地不存在**，且有**匹配对照**支持 |
| B112 报告者的**一般性**机制假设（"引擎把自由块与已缓存条目混为一谈"）未被检验 | **在 dense + ~90% 占用 + bs8 下不成立**（该假设若成立，本实验应当看到崩塌） |
| C44 / C109 / C110 的"放弃理由失效"留下了开放感 | 其**根因特属于异构 block** ⇒ 对本机**不可达**；且结构性修法已被 **RFC #42082 + WIP #42374（均 OPEN）** 占位 |
| 上游那条因果断言**没有匹配对照** | **补上了那个对照**（把"异构 vs 均匀"与"压力"、"并发"三个变量分开） |

**⚠️ 诚实边界（这条结果没有覆盖什么）**：
- 本实验**只变动了并发与池容量**，**没有**制造**请求大小混杂 / 抢占（preemption）/ 极高 churn** 的场景；
  报告者的失败运行是 **budget 受限下的 agentic round-robin**。⇒ 结论的强度是
  **"在均匀 block + 90% 占用 + bs8 下不出现"**，不是"在任何负载下都不会出现"。
- 4 格 = 4 个数据点，每格 1 次冷 + 1 次热，**没有重复**（本实验是因果对照，不是效应量测量）。
  四格热态命中率**逐位相同**（0.9981）这一点本身提供了很强的稳定性证据，但**不构成噪声估计**。
- 池压力用"缩池"实现，**没有**同时增加请求数（prompt 文件只有 32 行 ⇒ 最多 24 个不重复 prompt）。

---

## 5. 独立检索：既有崩塌报告**全是 hybrid / SWA**（这支撑"该对照此前未被做过"）

| artifact | 模型形态 |
|---|---|
| vLLM **#42948**（B112 源） | *"Prefix-cache 0% hit … **DeepSeek-V4-Flash hybrid groups** lose all first-block cache keys"* |
| vLLM **#48435** | *"**hybrid-SWA** prefix caching collapses to zero … Gemma-4-31B; eager-freed SWA tails recycled tail-first"* |
| vLLM **#43447** | *"DeepSeekv4 — selective prefix-cache retention for **sliding-window** KV cache"* |
| vllm-ascend commit `c50e497` | *"port **DSv4 SWA** prefix-cache retention to vllm-ascend"* |

**没有**找到 dense（均匀 page size）模型上的同类命中率崩塌报告 ⇒ 本对照此前**未被做过**。

---

## 6. 纪律说明

1. **本条不是"预登记实验"，而是对上游一条因果断言的复核**。因此**不主张任何效应量**；
   所有数字都是**原始快照差**，只在"是否出现崩塌"这一**定性**判断上使用。
2. **本条属于 FILTER 附录 A.6 那个机会类型的首次实际执行** —— 找"上游说了因果、但没做匹配对照"的裂缝。
   **执行结果：这条裂缝真实存在（引文里确实没有对照），但补上对照之后，上游的因果是对的。**
   ⇒ 这验证了机会类型的**可执行性**（能产出干净的对照实验），同时说明它**不保证出候选**。
3. **仪器侧自我更正（本目标的第 5 处）**：本节依赖的"前缀缓存命中率"读数**全臂避开投机**，
   因为 p21 已证明 dflash 系会把它压到 0 —— 若沿用旧习惯带投机测，本文会得到"崩塌"的**假阳性**。
   **"凡涉及前缀缓存的测量，必须先声明 drafter 类别"** 应作为仪器纪律固定下来。
4. **一处先前的判断被本条加强**：我在 p20 v3 里观察到"nospec 臂在加 1 趟预热后 r1/r2/r3 全暖"（641/651/640 tok/s），
   当时只当作"预热修好了混淆"。本条给出了它的**命中率解释**：dense + nospec 下命中率本来就是 0.9981 且不随并发/压力变化。

---

## 7. 复现命令

```bash
# 全 2x2（4 个 serve；每格 A_cold → B_warm，/metrics 两次快照做差）
cd /root/myCCFA && MODE=full bash probes/p22-dense-hit-collapse/run_hit_concurrency.sh
# 只看采集器自测
/root/ccfa_venv/bin/python probes/p22-dense-hit-collapse/hit_metrics.py --selftest
```

**成本**：≈0.2 GPU·h ｜ **末态**：显存 0 MiB、无残留 API server、无残留 EngineCore。
