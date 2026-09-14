# (B) 交付：为何全部被否 —— `goal-e5030ca8` 发散搜索结果

**日期**：2026-09-15 ｜ **轮次**：7/256 ｜ **目标**：在 LLM 推理 / AI Infra / MLSys 内发散性找到 1 个过全部闸门的课题
**结论**：**无幸存者。** 8 个给定方向全部裁决完毕；候选池 ①② 全量耗尽；自建 5 条候选全部被**事前冻结的判据**杀掉。
**GPU 花费**：本轮 ≈0.6 GPU·h（上限 8）；本目标累计 ≈1.1 GPU·h。

---

## 0. 机器（以实测为准，两台上都做过）

| | schoolserver（`ssh schoolserver`） | 新机（用户提供，`ssh -p 26924 root@connect.weste.seetacloud.com`） |
|---|---|---|
| GPU | **8× RTX 4090 24 GB, sm89, driver 550.67** | **1× RTX 6000D 85,651 MiB, sm120, driver 595.71.05** |
| 引擎 | **vLLM/SGLang 均不可用**（cu13 扩展 vs driver 550；0.21 能跑但**输出是垃圾**） | **vLLM 0.29.0 可用且通过正确性闸门**；SGLang 未装 |
| 可用平台 | HF transformers 4.57.6 + torch 2.11.0+cu128（**逐位确定**，`mode=off` TV=0.000000000） | vLLM 0.29.0 + Qwen3-4B + DFlash2/DSpark 原生支持 |

---

## 1. 八个方向的裁决（S3b，共 5 个子代理，0 GPU·h）

| 方向 | 判决 | 决定性反证（均经 fetch 并读回正文） |
|---|---|---|
| 🥇 Speculation Budget | **OCCUPIED** | 上一个目标已墓碑；本轮把其存活切片 **S-B** 也杀掉（§2.4） |
| 🥈 Verification-aware Attention | **OCCUPIED** | **SpecAttn**（2510.27641）已把**验证器**稀疏化；**QuantSpec**（2502.10424）已用 INT8 验证 |
| 🥉 KV Predictive Management | **OCCUPIED（双向）** | **2601.14279** 做了预测器质量消融：position-only **8.85** 优于学习打分器 8.91；**Belady 计算**显示 LRU 达最优的 **99.4%**，任何预测器头寸 ≤0.5 点 |
| 4 Inference-time Compute Allocation | **OCCUPIED（含匹配对照）** | **2605.10875（ICML 2026）** 对 **best-of-500 随机调度**做匹配对照 |
| 5 Reasoning-aware KV / Attention | **OCCUPIED** | **RaaS**（ACL 2025 Findings, 2502.11147）占 milestone-token 生命周期；**Random Attention**（2609.03430）跑了匹配对照并否定：*"the selection signal contributes almost nothing"* |
| 6 Decode Dataflow / Memory Traffic | **PARTIALLY OCCUPIED** | **2605.30571**（batch-1 decode 归因）+ **2512.01644**（逐算子归因）。存活切片 = no-P2P host-staged 机的字节-时间归因 ⇒ **被本轮 p30 杀掉**（§2.2） |
| 7 Adaptive Precision | **PARTIALLY OCCUPIED** | **MarginGate**（2605.30218）+ **2608.13756 "The Integer Alibi"**：margin 预测翻转 **ROC-AUC 0.94**，且**逐层**已做 |
| 8 Agentic LLM Runtime | **OCCUPIED** | **SAGA**（2605.00528，整个 workflow 作调度单位，KV 重算 38%→8%）+ **ThunderAgent**（2602.13692，recompute 进目标函数） |

**方向 5/8 的杀因是"匹配对照已被做过且为负"**，不是"没人做"——这是本轮最省时间的一类杀。

---

## 2. 自建候选：逐条的现象数字 / 复现命令 / 机制 / 杀因 / 复活条件

### 2.1 C1 —— sm89 上 block-FP8 静默回退：**阻塞（非杀）**

- **现象**：门控已零卡实证（子代理直接调用编译扩展）：`cutlass_scaled_mm_supports_block_fp8(89)=False`、`(90)=True`；量级**无人测过**。
- **复现**：`pipeline/…`；证据 `results/p29-c1-blocked/README.md`。
- **杀因**：**不可测**。需要 ① block-FP8 量化 checkpoint（磁盘没有）② vLLM 的 kernel 选择器本身（schoolserver 上 vLLM 不可用）。"静默回退"是**关于 vLLM 开发过程的断言**，本地实验无法证伪。
- **复活条件**：一台 **sm89 + 可跑 vLLM 0.29 + block-FP8 权重**的机器；或把断言改写成"两种 FP8 格式在 sm89 上的相对代价"（但那已不是 C1）。

### 2.2 p30 —— PCIe 上 TP 的"延迟税"：**杀掉（我的上界用错了承重数字）**

- **§0.6 上界（错）**：p27 的 **0.0548 ms/次** × 72 次/token = 3.95 ms/token ÷ 地板 8.04–11.49 ms = **34–59%**，看似是 2% 杀线的 17–29 倍。
- **实测（对）**：真实依赖链 `x→gemm_i→all_reduce_i→gemm_{i+1}`，`chain/no_comm` = **1.015**（p10 口径 1.045）⇒ 暴露代价 **~1.5–4.5%**。
- **复现**：`probes/p30-tp-overlap/chain_probe.py`（含阴性对照 `chunkC_dep`）；`results/p30-tp-latency-tax/{bound.py,stage1.md,stage2.md}`。
- **机制/根因**：p27 的 0.0548 ms 是**单发带同步**口径；背靠背 **0.0262–0.0270 ms**，链上暴露增量 **≈0.011 ms** ⇒ 上界**高估 5–10×**。且该效应**低于本机同格噪声**（基线自身极差 **132.6%**）。
- **杀因**：量级崩塌 + 低于噪声地板。**S3b 同时判 OCCUPIED**：FlashInfer **PR #4393 `"state":"MERGED"`（本人在 2026-09-15 复核确认）**，标题 `[feat]custom all reduce kernel by qsang-nv`，正文含 `root complex`/`NVLink` 语境 ⇒ 确实是为 PCIe 无 NVLink 做的定制 all-reduce。SGLang #34528（OPEN）在其上 TPOT **21.14 → 13.62 ms** ⇒ 该 fabric 上**已发货的答案是"消除延迟"，不是"重叠"**。
  - ⚠️ **诚实边界（我自己的复核，2026-09-15）**：S3b 引的 **"12 KiB 5.3 µs vs NCCL 205.1 µs = 38.8×" 我无法核验** —— 对该页 HTML 做定串统计得 `205.1` **0 次**、`38.8` **0 次**，页面里唯一的 µs 数字是 **18.7 µs**。（我第一次用 `grep -c "205.1"` 得到"命中 2 次"是**假阳性**：`grep` 把 `.` 当正则通配。**这条已记为 R5 的新实例。**）⇒ **p30 的杀不依赖这个数字**（它由我自己的实测 1.5–4.5% 独立支撑），但**引用时必须改成可核验的表述**。
- **附带确立（可引用）**：decode 形状（5 KB）下本机集合通信有效带宽 **0.09 GB/s**，比同机 4 MB 时的 **12.4 GB/s 低 140×**；因此**分块重叠在本机是反效果的**（`chunk8` 1.459 > `chain` 1.015）。
- **复活条件**：换一台**安静**的机器重测（本机 load≈337/192 核）。

### 2.3 p28 → p31 → p33 → p35 —— "丢弃质量不是货币，支撑结构才是"：**杀掉（被自己的匹配轴缺陷解释掉）**

- **p33 曾测到（错）**：按丢弃质量匹配后，random 的漂移是 window 的 **18–23×**，CI 下界 9.4–15.8×（而已发表稠密方向天花板 3×）；natural 上翻转 **30/30 vs 2/30**，CI 不相交。
- **p35 的自我审计（对）**：匹配量取的是**全 query 平均**的丢弃质量，而被测输出**只有最后一个 query**。因果掩码下三结构早期 query 行为不对称 ⇒ 实测缺陷幅度 **randn 0.23× / topn 0.58–0.63× / window 1.16–1.27×** ⇒ **三结构从未在相同真实质量下比较过**。
- **修正轴后**：合并 mass 0.40 的 TV 比 **4.3×，CI [0.9, 10.3] 含 1**；**oracle vs random 塌到 0.8–2.7×**；natural 翻转率 CI 重叠；**random 上下文上排序反向**（mass 0.40：topn 0.60 比 randn 0.50 更差）⇒ 按事前冻结判据**死亡**。
- **复现**：`probes/p35-fixed-axis/{fixed_axis_probe.py,analyze.py}`；`results/p35-fixed-axis/SUMMARY.md`。
- **升为方法学规则**：**当扰动按 query 位置不均匀时，"全 query 平均"的扰动量不是匹配轴**；必须按**决定被测输出的那个位置**匹配。适用于一切"稀疏注意力 / KV 驱逐"的对照实验。
- **复活条件**（三条须同时满足）：匹配轴是决定输出的那个 query；该轴上 TV 比 **CI 下界 >3×**；翻转率 CI 不重叠**且**结构排序在 ≥2 类上下文一致；并在**真实部署的稀疏/驱逐实现**上复现（非 monkey-patch 的 softmax 重归一化）。

### 2.4 S-B —— 投机"pays/does-not-pay"曲面的匹配对照分解：**杀掉（零卡闸门，未花 1 秒 GPU）**

- **上游因果断言**（vLLM #54749）：*"(2k, B=8) pays …; (32k, B=8) does not pay …; but then (32k, B=1) would be switched off, and it pays."*
- **上游自认缺口**：*"the **2×2 design for that is written and unrun**"*、*"the explanation for its magnitude is open."*
- **量级（上游实测）**：单差异格 **1.29×–1.36×**（c=256，per-arm stdev ≤1.72%）。
- **零卡闸门结果**：`notes/SB_SEPARABILITY_2026-09-15.md`。已发表的 pay/no-pay 矩阵（#54691 的 **15 个 break-even 阈值**，真 3 ctx × 5 batch 网格）被**一个 2 有效参数的可分离评分 15/15 复现**（`pay ⟺ B·ctx^1.568 < 6.3e7`）；自由形式 `f(B)+g(ctx)`（7 参数）与 2 拐点阶梯（3 参数）**也都拟合**；且**那 15 个符号全部是推断、不是测量**——核心标量 `a` **在任何格上都没被测过**。
- **杀因**：`SEPARABLE FORM FITS ⇒ KILL`。**这是 FILTER §0.6 的直接成果：在任何 GPU 花费之前杀掉。**
- **残留（未获许可）**：#54691 自己的**实测**分解显示固定 B=1 时 draft 项 **1.24→7.21 ms（5.8×）**而 target step 只动 **1.097×** ⇒ 成本形式 `F(ctx)+K·M(batch)` 被证伪（这是**函数形式**错误，与 #54749 §2 结论相反）；**但**同作者后来的固定 batch 诊断（#49986 评论 11）**未能复现**上下文依赖，并警告"并发随上下文变化的扫描会把 batch 效应折进 ctx 读数"⇒ 标定问题，**同样不许可**。
- **复活条件**：测 #54749 自称"写好未跑"的 **2×2 四格的可达接受长度 `a`**（非 break-even），现有 harness、无需新代码。**伪证条件：若 `a(32k,B=1) ≤ 1.91`，整个矛盾自行消解。**

### 2.5 C258 —— sm120 上重裁 `enable_qk_norm_rope_fusion`：**杀掉（§0.6 量级不足）**

- **§3**：最近邻把 QK-RMSNorm+RoPE 的三次逐层发射保持未融合；在条件 C=**H100(sm90,132 SM)** 下它测到更慢，故 `enable_qk_norm_rope_fusion` 在 O0–O3 全档为 False（`<V>/config/vllm.py:252,275,298,321`；`<V>/compilation/passes/pass_manager.py:225`）。
- **§0.6 算术**：3 次发射→1 次，净省 2×1.0–1.5 µs×36 层 = **72–108 µs/步**；权重读下界 t_base≈5.3 ms（8.0e9 B ÷ 1.52 TB/s），现实 7–12 ms ⇒ **0.6–1.5% < 2%**。乐观上界（3 µs/层）仍 ≤1.5%。**n*=0**。
- **杀因**：§0.6 量级不足；且上游在 H100 上测到的是**负号**（issue #34391 正文**无任何数字**，已读）。
- **复活条件**：若某台机上 t_base 显著更低（更小的模型 / 更快的显存），占比可能过 2%——但那已不是"重裁符号"的问题。

---

## 3. 候选池状态（本目标新结论）

| 池 | 状态 | 证据 |
|---|---|---|
| ① 失败理由依赖已变约束 | **耗尽** | KV 地图 **185** 条带引文放弃格（上轮，幸存 0，`GOAL3_CLOSE:438,:492`）＋ **INFERENCE 263** 条 `REASON-MAY-HAVE-EXPIRED`（本轮，幸存 0，`notes/POOL1_ENGINE_UNLOCKED_2026-09-15.md`）＋ 投机 C+G（上轮）。KV 与投机地图的 `REASON-MAY-HAVE-EXPIRED` **各为 0**（脚本全量抽取） |
| ② 从未被讨论过 | **耗尽** | `KV_NEVER_DISCUSSED.md` 方法头：就在**本机 `/root/ccfa_venv` 的 vLLM 0.29.0 树**上做的（扫 3952 条 TODO/not-supported，2026-09-14），target 同为 Qwen3-4B |
| ③ 因果裂缝 | **最佳实例已杀** | S-B（§2.4）。`causal_gap_scan.py` 输出噪声大（命中的多是验证者更正与元数据），机械筛选不足以直接立项 |
| ④ 地图 B 段 | 仅用于排除（按 rev8 降权） | — |

**本轮池① 杀因统计（子代理）**：**~120 格**的 R2 条件落在**本机没有的配置**上（量化 / MoE / MLA / 多卡 / ROCm / SGLang / 训练）；**~90 格**的 R1 约束在 0.29.0 源码里**逐字仍在**。**新占位者（地图未记）**：vLLM PR **#47979**「SM120 PCIe serving stack」状态 **`DRAFT`**。

---

## 4. 本目标真正留下的东西：**7 条方法学规则**（比任何单条候选都耐用）

| # | 规则 | 代价（怎么学到的） |
|---|---|---|
| R1 | **"全 query 平均"的扰动量不是匹配轴**（扰动按位置不均匀时）；必须按决定被测输出的那个位置匹配 | p28→p31→p33→p35 **四次**自我更正 |
| R2 | **单发带同步的延迟数字不能当作流水链上的暴露延迟**（本机高估 5–10×） | p30 上界 34–59% vs 实测 1.5–4.5% |
| R3 | **零卡上界必须用正确口径的承重数字**；上界漂亮不等于可立项 | p30；也是 S-B 被 §0.6 杀掉的正面案例 |
| R4 | **筛子本身必须与被测机器绑定**（硬件画像**和**引擎画像），每次复述"够不着"前重判 | §9.1 硬件档写死（旧机单卡 sm120）；§9.9 把 schoolserver 的引擎结论当成整个池子的结论 |
| R5 | **子代理的"范围判断"与它的检索结论一样需要回读原文复核**；**且复核本身要用定串匹配**（`grep -F`），否则 `.` 等元字符会造出假阳性 | 本目标 **4 次**：池① 范围误判（KV 185 其实已过筛）、"C1 不可测"的边界、numerical-drift 的 interim 假阳性、**FlashInfer #4393 的 `205.1`/`38.8` 定串复核（我的 `grep` 正则假阳性）** |
| R6 | **"能 import" ≠ "算得对" ≠ "可对照"**：配置必须**读回断言**，产物必须**验正确性** | schoolserver vLLM 0.21 能跑但输出垃圾；两次"两组配置实际相同"报废实验 |
| R7 | **判据必须事前冻结，且不得事后移动** | 本目标 4 次自我否证（p30、p33、p35、S-B）**全部由事前判据触发**，没有一次是我事后找理由 |

---

## 5. 下一轮测绘结论（哪些面还没查过）

1. **池③ 需要更好的工具**：`causal_gap_scan.py` 的噪声来自它匹配到**验证者评论**而非上游断言。可行的改造：只在**已合并的性能 PR 描述**里找因果断言，并要求同 PR 内**没有**只变该变量的对照。
2. **SGLang 面完全未查**：新机**没装 SGLang**；schoolserver 上 SGLang 同样不可用。三张地图里依赖 SGLang 的格子（INFERENCE 35 格 / KV 42 格）在本目标内**从未被验证过**。
3. **地图自身的硬件杀因审计 —— 已完成，0 幸存者**（`notes/KILLREASON_AUDIT_2026-09-15.md`，0 GPU·h）。
   三张地图里"sm120 缺 X / 被门控 / 回退到 Z"形式的杀因约 **28 条**；逐条实测 **11 条**：**6 HOLD / 4 FALSE-ON-THIS-RIG / 1 UNVERIFIABLE**，**没有任何一条的残留内容过 §0.6 的 2% 门槛** ⇒ 不发 §3 句。
   4 条 FALSE 都是"**以错误理由关闭、但在这里无所谓**"：
   · **Spec-D1.8**（"trtllm-gen decode 被 `is_sm100_supported()` 门控"）——那是 **SGLang** 的闸（本机未装），vLLM 里该属性**不存在**；真实闸 `utils/flashinfer.py:482-486` **包含 sm12x**。残留 = XQA decode 可达，§0.6 上界 **0.199 ms/步 = 1.7–2.8%**（且假设注意力读取归零）⇒ **不过**。
   · **Spec-D1.4**（"FP8 KV 只能当存储"）——子句对 FA 成立（`flash_attn.py:228`；`nm -D | grep -ci fp8` → **0**），但**标题过度断言**：本机 `FlashInferBackend.supports_kv_cache_dtype("fp8_e4m3")=True`。残留 = 被迫离开 FA 后端，非缺功能，量级不可界 ⇒ **不过**。
   · **Spec-D1.5**（W8A8 "不为 sm ≥ 10.0 构建"）——假：`_C_stable_libtorch.abi3.so` 内含 `sm_120`/`sm_120f` 标记，树里 60 个 `*sm120*` 文件。§0.6 = **0**（BF16 dense）。
   · **INF-E18/E19**（B12X MXFP8/MXFP4 "静默丢弃"）——硬件闸**通过**（`is_device_capability_family(120)`，`mxfp8/b12x.py:56`、`mxfp4/b12x.py:59`）；真正的阻塞是 `pip install vllm[b12x]`。§0.6 = **0**。

   **⚠️ 我必须撤回自己上一轮写在这里的一条**：我曾据池① 子代理的升级写下"sm120 **有** split-KV（Qwen3-4B 的 32:8 GQA decode 形状可走到）"。
   审计指出并**经我采纳**：**符号存在 ≠ 可达**——我亲自核到的 561 个 split 符号只是**二进制那一半**；**已装封装**在
   `<V>/vllm_flash_attn/flash_attn_interface.py:311-312` 用 Python 闸掉了它：`if num_splits > 1: raise NotImplementedError("FA2 does not support num_splits > 1")`。
   而上游 `flash_api.cpp` 的 GQA-swap 分支在 Python 传 `>1` 时**根本没机会执行**。
   ⇒ **"sm120 split-KV 是活的"应记为 `UNRESOLVED`，不是 established。** 最便宜的收口：一次短运行打印 decode batch 在 `cudagraph_mode=FULL` 下的 `attn_metadata.max_num_splits`。

   **两条可复用的机制事实（非格子）**：
   · `is_device_capability_family(100)=False` **但** `has_device_capability(100)=True` 在 sm120 上**答案相反**
     ⇒ 任何写成"需要 compute capability 100"的杀因**不构成有效的 sm120 排除**。（地图多数用的是正确的谓词形式，故未追加判 FALSE。）
   · **Spec-D2.6（sm120 上无自适应验证）是一条正确的杀**，且它**从工程上解释了方向 🥇 在本机的处境**：
     `adaptive_verification.py:481` 要求 `AttentionCGSupport.ALWAYS`，而 `flash_attn.py:356-359` 只在 **FA3** 上给 `ALWAYS`，
     **本机的 FA2 ⇒ `UNIFORM_BATCH`**（`get_flash_attn_version()` → 2），FlashInfer 同样封顶在 `UNIFORM_BATCH`（`flashinfer.py:1010-1013`）。
4. **结构性判断（跨 ~700 个地图格 + 263 + 185 + 8 方向 + 5 条自建候选一致）**：
   > **"够大的都被占着，没被占的都不够大。"**
   两条池子给出两种互补的失败：池①/B 段产出"够大但已占位"，池②产出"空闲但够不着门槛"。
   若这个二分在下一轮仍成立，则**本工作区约束下（单卡 sm120 / dense 4B / 无多节点 / 无 SGLang）的可发布空间已被系统性探尽**——
   那本身是一个可交付的结论，而不是失败。
