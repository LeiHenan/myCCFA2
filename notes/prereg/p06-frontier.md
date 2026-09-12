# 预登记 — Frontier（drafter 算力分配前沿，**主线决胜**）

**登记**：2026-09-11 ｜ **最后一次修订**：2026-09-12（**数据采集之前**，见 §修订记录；本次仅补环境前提，实验设计未改）
**探针**：`probes/p06-frontier/` ｜ **对应方案**：**#6 主线**

## 主假设（Headline claim）

> **drafter capacity（网络层数/宽度）是与 draft length 正交的独立控制维度。**

> 注：本方向在方案内的叫法「投机算力分配 / Speculative Compute Allocation」是**内部标签、非既有术语**（已检索无同名论文）；论文措辞需另定。

即：不存在一个能把两者互相替代的单一配置——在某些 (bs, ctx) 区域，最优组合必须**同时**调整 depth 与 γ，且方向相反。
若该主假设不成立（两者共线），本工作就退化为"另一个 adaptive draft length 实现" ⇒ 见 §杀判据。

| 项 | 内容 |
|---|---|
| **量** | 端到端 `tok/s`、per-step draft 成本 (ms)、平均接受长度、**相对最佳固定配置与相对"已 ship 自适应"的端到端比** |
| **区间** | depth {1..N} × width {窄链式, 默认树, 宽树} × **γ {1,3,5,7}** × batch {1,8,32} × ctx {4k,32k,128k,185k}，每格 ≥3 次重复。**γ 因子是正交性检验的必需项**：至少在 `width=默认树` 子集上与 depth 同扫 |
| **引擎** | **T1/T2 固定 `vLLM`**（只有它同时有"已 ship 的 K 自适应基线"与"多层 drafter"） |
| **基线** | ① spec off（AR）② **最佳固定配置** ③ **DSpark 生产调度器**（置信度头 + STS + 离线 SPS 成本表 + ragged-verify）④ **vLLM `num_speculative_tokens_per_batch_size`**（per-batch K 查表，`dynamic_sd_lookup`）—— **不得省略**。SGLang `adaptive_spec_params` 逐字拒绝多层 worker，**不能**充当 T2 对照（仅可作静态基线） |
| **家族前提** | 必须 ≥1 个**多层并行 drafter**（DFlash 5 层 / DSpark 3 层 MoE）。**EAGLE-3 只有 1 层 ⇒ 无深度旋钮，只能作基线**。取数路径：`vllm-project/speculators`（DSpark/DFlash 上游实现）+ HF 权重（如 `mgoin/GLM-5.2-speculator.dspark-block16`） |
| **截止** | W2 结束（≤2 周） |
| **附带交付物** | `notes/p06-dex-differentiation.md` 的定稿（三轴对照 + 三条判死条件的实测结果） |

## Go 判据（**全部**满足才 Go）

1. **正交性（主假设）**：存在 (bs, ctx) 区域，最优 (depth, γ) 发生**非共线反转**；
2. 存在**内点最优**（不是"越深越好"或"越浅越好"）；
3. 相对**最佳固定配置** ≥8%；
4. **三个必答项**全部通过（见下）。

## 三级判定（**决策级 ≠ 论文级**）

> 2 周是**论文级**产物（全网格 + 自适应对照臂 + 出图）。**决策级**答案分三级，最快 **3–4 天**（纯 GPU 约 1.5–2 天）：

| 级 | 时间 | 网格 | 判据 |
|---|---|---|---|
| **T0 旋钮验证** | 半天（1 卡） | 无网格：加载多层 drafter → 只跑前 `d` 层 → 看接受长度/耗时是否随 `d` 单调可动 | 深度**不是可操作的旋钮** ⇒ 当天杀 |
| **T1 反转探针**（主假设） | 半天–1 天 | `depth{1,3,5} × γ{1,3,7} × ctx{4k,32k} × bs{1}` ≈ **18 格** × ≥3 次重复 | **存在配置反转** ⇒ 主假设存活；**完全无反转** ⇒ 杀 |
| **T2 增量对照**（真门） | 1–2 天 | 在 T1 的反转点上：`adaptive-steps-only` vs `adaptive-steps+depth` | 增量 **<8% ⇒ 杀**；≥8% 且能说清与已 ship 控制器的差异 ⇒ **Go，再做 2 周全网格** |

**T1 的结论是不对称的（必须写进结论）**：用**截断**（同一 drafter 只跑前 `d` 层）得到的是**下界** —— 一个真正按浅层训练的 drafter 可能更好（DSpark 的"2 层胜 5 层"是**训练**出来的，不是截断出来的）。因此：

- 测出**反转 + 增量 ≥8%** ⇒ **决定性正向**；
- **测不出任何东西** ⇒ **提示性但非决定性**（可能是截断的锅）⇒ 必须补一个**已发布的浅层 checkpoint**（如 DSpark 2 层 vs DFlash 5 层；家族不同但层数是真的）再判一次，**不得据此直接杀**。

**环境前提（2026-09-12 追加，数据采集前）**：引擎固定为 vLLM ⇒ 主线所用的 `DFlash2DraftModel` **只从 vLLM 0.28.0 起存在**，而 vLLM ≥0.27 是 **CUDA 13** 构建 ⇒ **机器必须驱动 ≥ 580**。原 8×4090 的 550.67 不满足，已判不可用；选机规格与验收见 `docs/MACHINE_REQUIREMENTS.md`。**此项不改变任何实验设计、判据或网格。**

**硬件与上下文上限（2026-09-12 按实测配置更正）**：T1 = `ctx{4k,32k} × bs{1}` 单卡。以 `Qwen3-4B` + `Qwen3-4B-speculator.dflash2` 为例：**144 KiB/token** ⇒ 32k 的 KV = **4.5 GiB（FP16）/ 2.25 GiB（FP8）**，权重 ≈8 GiB、drafter ≈2.6 GiB。⇒ **T0 只需 ≥16 GB；T1 需单卡 ≥24 GB**，本机 96 GB 用 `bfloat16`（不量化）。**T0/T1 均不需要多卡**。所有对照条件必须**同 KV dtype**，且论文须注明 GPU 型号。详见 `notes/p06-toolchain-check.md` §5。

> ⚠️ **128k 格不可行（2026-09-12 实测，数据采集前更正）**：`Qwen3-4B` 的 `max_position_embeddings = 40960`（原 toolchain check 未核这一项就写了 128k）。vLLM 直接拒绝
> `User-specified max_model_len (140000) is greater than the derived max_model_len (max_position_embeddings=40960)`。
> 要跑 128k 必须加 rope scaling（YaRN）——那会**改变被测量的模型本身**，超出本探针范围。
> ⇒ **T1 的长上下文格定为 32k（`ctx{4096, 32768}`，`--max-model-len 40960`）**，仍保留 8× 的上下文对比。此为**数据采集前**的网格更正，不构成事后放宽。

## 扩展条款（预登记，2026-09-11 决定；**改动即事后调整**）

1. **T1 = 1 家族 × 1 引擎（vLLM）**，先拿主假设的答案。
2. **自动触发扩展**：T1 出现**反转但幅度在噪声边缘**，或**共线且效应量 ≥ 8%** ⇒ 在**同一预登记**下**追加第二家族**（DFlash ↔ DSpark）；**不自动追加第二引擎**（除非 vLLM 格已跑通且 SGLang 已验证多层 drafter 可承载）。
   **bs 扩展条件（补齐 T1 的已知盲区）**：T1 的 `bs{1}` 只为最便宜地看信号；**若反转只在高 bs 出现**，则 Go 判据的"(bs,ctx) 区域反转"无法在 T1 判定 ⇒ **必须按本条在同一预登记下扩到 `bs{1,8,32}` 再判，不得据 T1 阴性直接杀**。
3. **降级形态**：机制主张死亡**且共线在两家族都成立** ⇒ 启动**新的、独立预登记的测量论文实验**（新 prereg、新数据）；**不得**把旧数据重新解读成测量论文。

## 必答项（任一不过即不 Go）

1. **与已 ship 控制器的差异**：它们调的是 `speculative_num_steps` / `num_speculative_tokens_per_batch_size`（长度轴）；本主线主张的是 drafter **网络层数/宽度**（深度轴）。
2. **为什么不能把 vLLM 的 per-batch K 查表（`num_speculative_tokens_per_batch_size`）扩到深度轴** —— 本线跑在 vLLM，这才是审稿人会问的那一问。
   SGLang 控制器逐字拒绝多层 worker（`enable_multi_layer_eagle=True is not supported`）只是**旁证**，不构成答案。
   **必须给出结构性理由**（例如深度轴的"接受率—成本"关系与步数轴不同、最优深度随 (bs, ctx) 反转），否则本工作退化为 plumbing。
3. **增量对照**：每个 (bs, ctx) 格上必须做 **adaptive-steps-only vs adaptive-steps+depth** 的对照；只有后者胜出且 ≥8%，深度轴才算独立贡献。
4. **动机抗辩（新增，针对 MLSys '26 PRISM）**：必须回答"既然 PRISM 主张容量可与推理成本**架构性解耦**，为什么仍需要运行时分配？"
   **唯一站得住的落点**：PRISM 是**训练期静态**设计，不处理**上下文相关**的成本 —— 例如 DFlash 在 ~185k 的**全上下文重扫**（`#54691` 实测 16 vs 71 tok/s），那是"每步扫描范围"问题，不是"参数规模"问题。回答不出来 ⇒ 退化为 plumbing。

## 读数与分析（2026-09-12 增补，**数据采集前**）

1. **ridge 斜率（取代"二值反转"作为主读数）**：逐 `(bs, ctx)` 计算 `γ*(depth) = argmax_γ 平均 tok/s`，并报
   `spread = max γ* − min γ*`。脚本：`probes/p06-frontier/analyze_t1.py`（`--selftest` 已覆盖三种判定）。
   | 观测 | 判定 |
   |---|---|
   | `spread == 0`（γ* 与 depth 无关） | **强可分离 ⇒ 主假设成立** |
   | `spread > 0` 且 `γ*` 随 depth **单调** | **斜 ridge ⇒ 走 H1b** |
   | 其它（非单调） | **不可判定** ⇒ 补重复，或按扩展条款扩 `bs{1,8,32}` 再判 |
2. **H1b 分支（斜 ridge 的正向出口，防止"斜了就杀"）**：若 ridge **系统且稳定**（`L* = f(depth)` 可拟合），
   **不自动判死**，而是转为「**预测式 horizon 策略**」——**但它必须打赢已 ship 的适配器**
   （vLLM `num_speculative_tokens_per_batch_size`；SGLang 侧则是其 acceptance-EMA 控制器）。
   **门槛不变**：相对**最佳固定配置** ≥8% **且** 相对**已 ship 适配器**有增量；否则只是把已有机制换了一种拟合方式，不得成文。

## 杀判据

| 条件 | 动作 |
|---|---|
| **`optimal depth ≈ f(optimal γ)`（共线 / 可互相替代）** | **杀或并入 `#7`/`#12`** —— 主假设不成立 |
| 收益只在 185k 角落 | 杀 |
| 无内点最优 / 无配置反转 / <8% | 杀 |
| 深度自适应相对 `adaptive-steps-only` 无增量 | 杀或改向"步数-深度联合"的窄问题 |
| 必答项 2 给不出结构性理由 | 改写定位或杀 |

## 已知的反对证据 / 拥挤邻居（必须写入结论，不得省略）

**引擎侧（比论文更强，且是真正的基线）**

- SGLang `adaptive_spec_params` 已 ship 运行时自适应（batch 1/8/32/64 → 候选步数 `[1,3,5,7]/[0,1,3]/[0,1]/[0]` + 迟滞 + 多套 CUDA-graph 原子切换），但**明确不支持多层 worker**。
- vLLM 有 per-batch K 查表（`dynamic_sd_lookup` ← `num_speculative_tokens_per_batch_size`；RFC `#48627` 的形态为 `[[1,64,3],[65,128,1],[129,512,0]]`）。
- ⇒ 高 batch 下"少投机"**已被解决**；深度轴必须在这个基线上仍有增量。

**文献侧（长度/树/选择轴，均不占本轴但构成拥挤）**

| 工作 | 实际控制的轴 | 备注 |
|---|---|---|
| AdaEAGLE `2412.18910` | **Adaptive Draft Structures**（含 draft length 建模） | 不是 drafter 网络容量 |
| OPT-Tree（TACL） | draft **tree** 结构（depth / branching） | 树轴 |
| AdaSD `2512.11280` / AdaptiveSD `2607.03876` | generation length / 多策略编排（CPU 受限） | 长度/策略轴 |
| **MemSpec** `2608.10362`（LCTES 2026） | **选哪个 drafter**（residency，内存预算） | 与"当前 drafter 多大"不同；全文 `KV cache`=0 |
| DSpark `2607.05147` | 离线**静态** depth/block 选择（2 层胜 5 层） | 证明该轴"活"，但非运行时 |
| Graft `2605.20104` | draft **树**深度（§4.4 因 CUDA-graph 静态形状放弃动态深度） | 类比而非先例 |
| MLSys '26：ReSpec / Sparse Self-Speculative Decoding / Beat the long tail | RL 训练侧、self-speculation、分布感知 | 均不占本轴 |
| **Compute-vs-Copy** `2511.12031`《Striking the Right Balance between Compute and Copy: Improving LLM Inferencing Under Speculative Decoding》 | **投机解码下 compute 与 copy 的配比** | **未读正文（Tier B）**；与 `#10`（recompute-vs-load）及本线的"预算"叙述相邻，引用前须读 |
| **TLT** `2511.16665`（ASPLOS '26；Tier B） | 训练期训 adaptive drafter + "**select speculative-decoding strategies per input batch**"（RL 长尾） | **不占格**：策略/长度轴 + 训练侧，不是运行时分配 drafter 网络容量 |
| **PRISM** `2602.01762`（MLSys '26 oral） | **训练期架构重构**：把每步计算拆到不同参数集，"**decouple model capacity from inference cost**" | **不占格，但攻击动机**：若容量不必按成本付费，运行时分配还解决什么？见必答项 ④ |
| **HELIOS** `2504.10724`（MLSys '26 oral） | early-exit 家族：**多模型动态切换** + 只加载可能用到的层 + 实时 profiler | **不占格，但挤压叙述**：运行时自适应深度已有人卖（1.48× 吞吐 / 15.14× batch） |
| SpecDiff-2 `2511.00606`（MLSys '26）；Speculative Decoding: Performance or Illusion? `2601.11580`（MLSys '26 oral） | 扩散 drafter 对齐；生产引擎上的系统测量 | 否；后者是"高 batch 下投机收益缩水"的权威佐证 |

## 修订记录

| 日期 | 变更 | 时点 |
|---|---|---|
| 2026-09-11 | 初版 | 数据采集前 |
| 2026-09-11 | 加基线 ④、三个必答项、增量对照、两条杀判据 | 数据采集前 |
| 2026-09-11 | **主假设升级为正交性**；网格加 **γ 因子**（正交性检验必需）；基线补 ⑤ vLLM per-batch K 查表；加"共线即杀"；补拥挤邻居清单（含 MLSys '26 三篇与 DSpark 上游可用性） | **数据采集前**（本工作区尚无任何 frontier 数据） |
| 2026-09-11 | **MLSys '26 全量扫描后**加入 PRISM / HELIOS / SpecDiff-2 / "Performance or Illusion?" 到邻居清单，并新增**必答项 ④（动机抗辩）** | **数据采集前** |
| 2026-09-11 | **引擎固定为 vLLM**（SGLang 的自适应基线对多层不可用）；新增**扩展条款**（优先家族、不自动加引擎；降级须新预登记） | **数据采集前** |
| 2026-09-12 | **补环境前提**：`DFlash2DraftModel` 仅 vLLM ≥0.28 支持，而 ≥0.27 为 CUDA 13 构建 ⇒ **需驱动 ≥580**；原 8×4090（550.67）不可用，机器按 `docs/MACHINE_REQUIREMENTS.md` 重选。**实验设计 / 判据 / 网格一字未改** | **数据采集前**（本工作区尚无任何 frontier 数据） |
| 2026-09-12 | **ctx 网格 128k → 32k**（数据采集前更正）：实测 `Qwen3-4B` 的 `max_position_embeddings = 40960`，vLLM 直接拒绝 `max_model_len=140000`；要跑 128k 需加 rope scaling（会改变被测量的模型）⇒ T1 长上下文格定为 **32k**（`ctx{4096,32768}`，`--max-model-len 40960`） | **数据采集前**（T1 尚无任何数据） |
| 2026-09-12 | **T0 实测结论：结局 A（深度旋钮可操作）** —— `ctx=4096,γ=7`, 真实文本：接受长度 1.303→2.829、tok/s 101.5→180.3，随深度**单调增长**；lossless 检查通过。并修掉三个会让 T0 得出错误结论的陷阱（变体路径须含 `dflash`、数据集须真实文本、变体须 `--prune-weights`） | 数据采集（T0）后，T1 之前 |
| 2026-09-12 | **T1/T2 实测结论（数据采集后）**：T1 斜 ridge（γ* 随 depth 3→7）⇒ 严格正交性否证；T2 增量 **+0.1% ≪ 8%** ⇒ **机制主张死亡，`#6` No-Go**。γ=8/15 实测引擎崩（CUDA cublas）⇒ 可行 γ 上限 = 7 = 训练操作点 ⇒ 网格已覆盖可行域。**本 prereg 至此结题**；按 D1 条款第 3 条，若要沿"capacity 与 length 非正交"继续，须**新的独立 prereg** | 数据采集**之后**（结论登记，非判据改动） |
