# p30 —— PCIe-only 机器上 TP 的代价是**延迟税**，因此是**重叠**决定的（不是带宽决定的）

**日期**：2026-09-15 ｜ **目标**：`goal-e5030ca8`｜ **方向**：思路 ⑥ Decode Dataflow / Memory Traffic
**成本**：§0.6 零卡分析 0 GPU·h（`bound.py`）；stage-1 探针 ≈0.05 GPU·h（仅空闲卡 2,3）

---

## 0. 这条轴从哪来（不是从地图的 B 段，而是从**筛子本身过时**来的）

`pipeline/tools/prescreen_map.py` 的 §0 硬编码的硬件档是**旧机器**：

> `# ---- FILTER §0 的机械化部分：本机（单卡 RTX PRO 6000 96GB sm120 / Qwen3-4B dense）够不着的东西`
> `HARDWARE_PAT` 命中 `multi-node|multi-gpu|tensor-parallel|tp=[2-9]|nvlink|infiniband|…|pcp|dcp|context parallel|pipeline parallel`

而机器事实在两轮之间**换过**（两处均有实测记录，不是我推断的）：

| | 旧机（`candidates/sglang-draft-corpus/00_intake.md:46`，2026-09-13） | 现机（`notes/SCHOOLSERVER_ENV_2026-09-15.md`，2026-09-15） |
|---|---|---|
| 卡 | **1× RTX PRO 6000 Blackwell 96 GB, sm120** | **8× RTX 4090 24 GB, sm89** |
| 驱动 / CUDA | 580.82.09 / CUDA 13 | 550.67 / CUDA 12.4 |
| 引擎 | vLLM 0.29.0 + SGLang 0.5.19 **可用** | **vLLM 不可用**（0.26/0.29 是 cu13 构建；0.21 能跑但输出是垃圾） |

⇒ **筛子在两个方向上同时错了**：
· 它把一切 **multi-GPU / TP / PP / NUMA** 条目判为"够不着"—— 但现机**有 8 张卡**（该桶是 FILTER §1a 意义上的"失败理由依赖已变约束"）；
· 它**不会**筛掉"需要 96 GB 单卡"的条目 —— 而那才是现在真正够不着的。

**这不是一条候选，这是一个系统性偏差**：整池候选是在一台已经不存在的机器上过筛的。下面这条轴就是从这个偏差里掉出来的。

---

## 1. §0.6 零卡量级上界（`bound.py`，只用 p27 已测点）

**输入全部来自既有实测**，除带宽区间外无新假设：
- `results/p27-transport-floor/2026-09-15/`：world=2 同 NUMA 的 all-reduce 中位 —— **1 KB 0.0548 ms / 256 KB 0.0611 ms / 4 MB 0.3138 ms**（4 MB ⇒ 12.4 GB/s）
- Qwen3-4B 几何：hidden 2560，36 层，**每层 2 次 all-reduce**（attention out-proj 与 MLP down-proj 各一次 row-parallel）⇒ **72 次/token**
- 权重 8,044,936,192 B（服务器 `du` 实测 total_size）

| batch | all-reduce 消息 | 每次延迟 | 每 token 税 | 该消息的有效带宽 |
|---|---|---|---|---|
| 1 | 5.0 KB | 0.0549 ms | **3.95 ms** | **0.09 GB/s** |
| 16 | 80 KB | 0.0568 ms | 4.09 ms | 1.44 GB/s |
| 64 | 320 KB | 0.0653 ms | 4.70 ms | 5.02 GB/s |
| 256 | 1280 KB | 0.1285 ms | 9.25 ms | 10.20 GB/s |

**两条结论**：

1. **量级远超门槛**：token 地板 = 权重读一遍 = 8.04–11.49 ms（依赖有效 HBM 带宽 700–1000 GB/s 的假设区间）。
   TP 税占地板 **34–59%** ⇒ **若完全不重叠，端到端慢 29–37%**。这是 2% 杀线的 **17–29 倍**。**§0.6 通过。**
2. **真正的结构事实（比上界本身重要）**：真实 decode 形状（5 KB）下有效带宽 **0.09 GB/s**，比这台机器 4 MB 时能到的
   **12.4 GB/s 低 140×**。⇒ **这里的集合通信不是带宽事件，是纯延迟事件**。而延迟税**几乎不随 batch 摊销**
   （B=1 的 3.95 ms → B=64 的 4.70 ms），因为地板（读权重）同样不随 batch 变。

**⇒ 整条轴的胜负只取决于一件事：这 3.95 ms/token 能否落进计算的阴影里。**
而"能否重叠"是**引擎 dataflow 的属性，不是硬件的属性** —— 这正是它可以被做成一个条件性差异轴（FILTER §3）的原因。

---

## 2. 差异轴（FILTER §3 句式）

> **最近邻**：**BNTU 2026**（DOI 10.21122/2309-4923-2026-1-54-59，**同行评审**）以 2×RTX 3090 + 14B 模型 + vLLM 实测，
> 逐字写：*"The results unequivocally indicate the **unsuitability of Tensor Parallelism for systems without NVLink** due to
> critical synchronization delays. It is proven that **Pipeline Parallelism is the only viable strategy for PCIe clusters**."*
> 另有 **Mobius**（ASPLOS '23，DOI 10.1145/3575693.3575703）：*"traditional pipeline parallelism is more suited for commodity GPUs."*
>
> **条件 C**：**集合通信可以与计算重叠**（或模型小到延迟税只占地板的一小部分）。
>
> **它覆盖不到**：上述结论把"同步延迟"当作**结构性事实**；但在**每层延迟 0.055 ms < 每层计算 0.13 ms** 时，
> 该延迟在原理上可以完全隐藏。它没有测过**重叠/不重叠的 A/B**，因此无法区分
> "PCIe 上 TP 结构性不可用" 与 "vLLM 的那个实现在 PCIe 上没把通信藏起来"。
>
> **我们做 Y**：直接测**重叠与否**这个变量本身。**当 C 成立时结论反向**（TP 可行）；**当 C 不成立时**，
> 我们给出量化的代价（29–37%）并把 BNTU 的定性结论升级为**可判定的条件**。

**为什么这比"再测一遍 TP vs PP"值钱**：TP-vs-PP 的定性结论**已经是同行评审的**（A.6/A.7 已确认），
再测一遍只是基准复现；而**"该结论是延迟税、且延迟税可被重叠消除"** 是一个关于**因果**的断言，
上游写了"due to critical synchronization delays"却**没有做把延迟单独变一次的对照** —— 这正是
`pipeline/tools/causal_gap_scan.py` 定义的机会类型（上游断言因果、无匹配对照）。

---

## 3. stage-1 探针：重叠是真实的还是测量的假象

`probes/p30-tp-overlap/overlap_probe.py`，5 臂同进程交错轮转（抵消共享机漂移）：

| 臂 | 含义 |
|---|---|
| `comm_only` | 36 × all_reduce，纯通信 |
| `comp_only` | 36 × GEMM（110 MiB 权重 ≈ Qwen3-4B 单层权重/2，即 TP=2 每卡每层），**同时给出 Python/launch 开销地板** |
| `serial` | all_reduce → GEMM 同默认流（天然依赖）= **完全不重叠的参照** |
| `dep_streams` | 通信在 `s_comm`，`s_comp.wait_stream(s_comm)` 后 GEMM —— **阴性对照**，依赖不变只换流，必须 ≈ `serial` |
| `indep_streams` | 通信在 `s_comm` ∥ GEMM 在 `s_comp`，无依赖 = **可达重叠上界** |

**内置判伪**：若 `indep_streams ≈ serial` ⇒ 重叠不可能 ⇒ 本候选**当场杀**，成本已沉没不足 0.1 GPU·h。
若 `indep ≈ comp_only` 且 `dep ≈ serial` ⇒ 重叠真实且能到上界 ⇒ 进入 stage-2（**分块依赖版**：
把 row-parallel GEMM 沿输出维切块，块 c 的 GEMM 与块 c−1 的 all-reduce 流水，这才是在真引擎里可实现的形态）。

结果见 `stage1.md`（同目录）。
