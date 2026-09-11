# 逐步实验指导（W1–W3）

配套：判据见 `EXECUTION_PLAN.md`，探针细节见 `probes/*/README.md`，预登记见 `notes/prereg/`。
**本文件只讲"怎么做"**：命令、注入点、日志格式、判定与回退。

> ⚠️ 命令模板里的 flag 以你 pin 的版本为准（`vllm serve --help` / 官方 Speculative Decoding 文档）。本机是 macOS，**无 CUDA ⇒ 实验必须在远程 GPU 上跑**。

---

## Phase 0 — 开工前（30 分钟，不占 GPU）

### 0.1 硬件预算

**先看 §0.1b 的显存算术**：80 GB 是"省事档"，不是硬门槛。24 GB 卡（4090/5090）能覆盖 E1 与 C3，frontier 需要 2 张，**#12 不适合**。

| 探针 | 省事配置 | 24 GB 卡路径 | 纯跑时间 | 备注 |
|---|---|---|---|---|
| E1（#1） | 1×80 GB（H100/A100） | **1×4090/5090 可行**（ctx 网格降到 ≤32k，或换 ≤4B target） | ~8 h | 见 §0.1b 的 128 KiB/token 算术 |
| 家族确认 + frontier（#6） | 1×80 GB | **2×4090/5090 可行**（TP=2 覆盖 185k 格）；单张只能在 ≤32k 跑 | 2 周 | **必须能改 drafter 代码** |
| C3 扫描（#3） | 1×80 GB | **1×4090 足够**（瓶颈常在 CPU 的 mask 计算，多核比大卡重要） | 1 周 | 不改代码，可与 frontier 排队共用 |
| #12 | 4–8× NVLink 卡（H100/H800） | **不建议**（4090 无 NVLink，A2A 走 PCIe 会改变结论的适用性） | 2 周 | 见 §0.1b 末段 |

### 0.1b 显存算术（决定你用哪档卡）

**每 token KV 字节** = `2 × layers × kv_heads × head_dim × dtype_bytes`（GQA；非 MLA）。

- 8B 级、8 个 KV head、FP16 ⇒ **≈128 KiB/token**（与本仓库 `docs/GPT.md` 的独立计算一致）。
  于是 **单请求 128k ctx ≈ 16 GiB**，加 ~16 GB 权重 ⇒ 24 GB 卡放不下 128k 格。
- 4B 级 ⇒ 权重 ~8 GB、KV ~56 KiB/token ⇒ 32k ctx 单请求 ~1.8 GiB，24 GB 卡宽裕。

**由此得到的取舍**

1. **E1 不需要 128k**。要测的比值本身是 `≈ γ/ctx`——**ctx 越小比值越大**，最敏感的区间恰好在短上下文。24 GB 卡跑 `ctx{4k,16k,32k}` 反而更能看清信号；若坚持 128k 格，用 2×24 GB 做 TP=2。
2. **frontier 的 185k 格**（4.4× 损失锚点所在）在 24 GB 单卡上做不到，需 **2×24 GB TP=2** 或 1×80 GB。其余格子（≤32k）单卡即可。
3. **C3 的瓶颈在 CPU**（grammar mask 计算）：优先选**高主频多核**的实例，卡只要放得下模型。
4. **#12 不要用 4090 出结论**：EP 的 all-to-all 收益与互联强耦合，PCIe-only 的数字会被质疑不可迁移。4090 上只做 smoke test，正式数据用 NVLink 机器。
5. 优先顺序（性价比）：**2×5090（32 GB×2）> 1×80 GB > 2×4090 > 1×4090**。仓库里既有的三臂消融就是在 dual RTX 5090 上跑的（`docs/V41.md` A2）。

### 0.2 固定引擎版本

- 选 vLLM 近主干；**记录是否包含** `#53614`（MERGED 09-06）、`#55760`（MERGED 09-08）——这两条影响 hybrid/prefix 语义，即使本阶段不跑 hybrid 也要记录。
- 每次实验在 `summary.md` 顶部记录：`git rev-parse HEAD`、`vllm --version`、`pip freeze | head -50`。

### 0.3 三层 smoke test（决定 E1 用哪个 drafter）

```bash
TARGET=meta-llama/Llama-3.1-8B-Instruct      # 或你的 target

# ① 无投机
vllm serve $TARGET --max-model-len 32768 --gpu-memory-utilization 0.85

# ② ngram 投机（不需要 draft 模型，先验证投机路径与记账能跑通）
vllm serve $TARGET --speculative-config '{"method":"ngram","num_speculative_tokens":5}'

# ③ EAGLE-3 / draft model（确认 checkpoint 存在且能加载）
vllm serve $TARGET --speculative-config '{"model":"<eagle3-draft-ckpt>","num_speculative_tokens":5}'
```

- ③ 失败不影响 E1 的**结论有效性**，但必须在 `summary.md` 标注 drafter 家族（ngram ≠ EAGLE-3，回滚/预留行为不同）。
- 记录每个配置的 `--block-size`（默认 16）与 `max_num_seqs`。

### 0.4 冻结测量量（写进 `notes/prereg/e1.md`）

| 量 | 定义 | 为什么要分开 |
|---|---|---|
| `R_byte` | 未提交 token 实际占用的 KV 字节 / 已提交 KV 字节 | #1 的收益上限直接由它决定 |
| `R_reserve` | 为投机**预留**的 slot 数 / 已提交 KV 所占 slot | plan.md #13 已发现"每请求每步预留 8 slot 而实际只需 1" ⇒ 预留口径可能远大于字节口径 |
| `H` | 未提交块从分配到释放之间的**驻留步数** | 若 H≈0，则未提交状态根本不存在，"分级"无从谈起 |

---

## Phase 1 — D1：E1（1 天）

### Step 1.1 环境（60–90 min）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install "vllm==<pin>"        # 或：git clone … upstream/vllm && pip install -e upstream/vllm
pip install pandas
huggingface-cli login            # 若模型是 gated
```

### Step 1.2 基线跑（30 min）

不开投机，跑 `(bs, ctx) ∈ {1,8,32} × {4k,32k,128k}`，记录：块使用量、tok/s、TTFT。
**目的**：确认块记账可用；用实测反推 **per-token KV 字节**（不要用公式，用实测：`Δblocks × block_size × …`）。

### Step 1.3 仪器化（2–3 h）

三个注入点（vLLM V1 路径，以你的 pin 为准）：

| 位置 | 记什么 |
|---|---|
| `vllm/v1/core/kv_cache_manager.py` → `allocate_slots()` / `free()` | 每次分配/释放的块数，以及该请求当前是否处于投机步 |
| `vllm/v1/core/sched/scheduler.py` → `schedule()` | 每请求 `num_scheduled_tokens`、投机 token 数、批次构成 |
| `vllm/v1/worker/gpu_model_runner.py` → `execute_model()` 返回处 | `num_accepted_tokens`（接受长度） |

**日志格式**（JSONL，一步一行）：

```json
{"step":123,"ts":1699,"req_id":"r7","phase":"decode","ctx_len":32768,
 "num_draft":5,"num_accepted":3,"blocks_alloc":2,"blocks_freed":0,
 "blocks_held_spec":2,"bs":8,"gamma":5}
```

**退路（不想改源码）**：开启 vLLM 的 metrics/logging，用 `kv_cache_usage` 与 `spec_decode_num_accepted_tokens` 一类指标做时间序列近似，并明确标注"间接口径"。

### Step 1.4 扫格（3–4 h）

- 网格：`ctx{4k,32k,128k} × bs{1,8,32} × γ{3,5,7}` + **1 个 draft-tree 配置**（若有）。
- 每格：3 次重复 × ≥200 decode steps，取分布而非均值。
- 128k × bs32 可能 OOM：**记录为"不可行"**，不得外推。

### Step 1.5 分析（1 h）

必须回答的三问：

1. `R_byte` 的 p95 是否全部 <5%？
2. `R_reserve / R_byte` 的比值是多少（>2 说明瓶颈在预留而非真实占用）？
3. `H` 的分布（H=0/1/≥2 分别占比）？

并做**解析对照**：`R_byte ≈ γ/ctx`（O(batch×window) vs O(batch×context)）。**若实测远大于解析值，差额本身就是发现**——那说明引擎的保守预留才是可优化对象。

### Step 1.6 判定（30 min）

| 结果 | 动作 |
|---|---|
| 全部 HBM 可行点 p95 <5% | #1 定死，归档（写进 `decision_log`） |
| 存在可行点 p95 ≥10% 且是真实部署配置 | 进入机制设计 |
| 5–10% | 补测 `H` 与峰值占用后再判 |

写 `results/e1_uncommitted_kv/<date>/summary.md`（四段：量/区间/基线、数字、判据对照、结论）。

---

## Phase 2 — W1 D2–D5：#6 前置 + C3 并行

### Step 2.1 drafter 家族确认（半天）——**#6 的生死前提**

1. 找到能跑起来的**多层并行 drafter**（DFlash 5 层 / DSpark 3 层 MoE 类）。
2. 确认**能否截断层数**：**不要重训**。做法：同一 drafter 只执行前 `d` 层（`d ∈ {1..N}`），
   - 若框架暴露层数配置 → 直接用；
   - 否则在 drafter 的 `forward` 里于第 `d` 层后 `return`（或 hook 跳过后续层），确保 KV/缓存路径不炸。
3. 记录：家族、总层数、截断方式、精度影响（接受长度的下降曲线）。

> 若只有 EAGLE-3（1 层）：**#6 降级**，深度旋钮不存在 ⇒ 直接跳到 Phase 3 的 #4 门 + #12。

### Step 2.2 per-step draft 成本仪器（半天）

在每个 (d, bs, ctx) 格记录：draft 阶段耗时、verify 阶段耗时、接受长度、tok/s。
**这是 frontier 的因变量**，必须与 Step 1.3 的记账共用同一套日志。

### Step 2.3 C3 扫描（1 周，与 2.1/2.2 并行）

- 语法类落 `results/c3_constrained_scan/<date>/grammars/`：class-1 JSON schema、class-2 嵌套 JSON+regex、class-3 C++ 子集、**class-4 Python 变体/Bython 类**。
- 网格：`batch{1,8,32,128,256} × 4 类 × spec{on,off}`，每格 3 次。
- 记录相对**无约束**吞吐比、mask 耗时、CPU 占用、TTFT/TPOT。
- 判据与先验（≤6%）按 `notes/prereg/c3.md`。

### Step 2.4 DEX 替换测试（2 h，纯写作）

按 `probes/frontier_drafter/README.md` §6 的模板写两段，结论三选一：继续 / 改措辞 / 改方向。

---

## Phase 3 — W2：frontier（主线决胜）

### Step 3.1 跑 frontier（3–4 天）

- 网格：`d{1..N} × width{窄链式,默认树,宽树} × bs{1,8,32} × ctx{4k,32k,128k,185k}`，spec off 与最佳固定配置作对照。
- 每格 3 次重复；产出 `frontier.csv` + 等高线图。
- 基线纪律：**DSpark 生产调度器（置信度头 + STS + 离线 SPS 成本表 + ragged-verify）是基线**，不是空白；结论必须有一句"与已 ship 调度器的差异"。

### Step 3.2 判定（半天）

Go = **内点最优** ∧ **(bs,ctx) 平面配置反转** ∧ **≥8%**（相对最佳固定配置）∧ 能说清与已 ship 调度器的差异。
任一不满足 ⇒ **杀**（收益只在 185k 角落也算杀）。

### Step 3.3 #4 复核门（2 h，无 GPU）

按 `probes/gate_hybrid_state/README.md`：读 `archive/engine_prs.md`（`#53614`/`#50172`/`#55760`/`#56142`/`#55697`+`#55873-6`/SGLang `#30393`）与 `archive/probe_c/atom.md`，产出 `notes/gap_hybrid_state.md`。
gap 若仍是"checkpoint 边界 / 提交语义" ⇒ **归档**；只有"多分支 SSM 快照内存放大 + 淘汰"才通过。

### Step 3.4 回退表（提前想好，别在失败当天想）

| 若 | 则 |
|---|---|
| #6 死（无反转 / <8% / 只有 EAGLE-3） | 依 Step 3.3 结果切 #4 收窄版；否则切 #12 |
| #4 门不过 | #12 转正；C1/C2 填缝探针补位 |
| #12 时间盒到期不达标 | 回看归档清单，重开 **#11**（需先拿到分支类真实 trace）或 **#10**（需新的 recompute 证据） |

---

## Phase 4 — W3：收敛

1. **决策会**：锁定 1 主线 + 1 备线，条件：量级 ≥8%（自测 vs 部署基线）且机制可归因。
2. **写三件套**：gap 声明（它测哪个量、在哪个区间、对哪个基线）、机制草案、基线清单。
3. **论文骨架**：现象 → 归因 → 机制 → 收益（MLSys 口味：真实栈上的可测机制）。

---

## 每次收工必做

1. `results/<probe>/<date>/summary.md`（四段，含判据逐条对照）；
2. `notes/decision_log.md` 追加一条——**死亡也要写**；
3. 若判据有变：`notes/prereg/<probe>.md` + `decision_log` 同时留痕（不得在看到数据后放宽）。

## 需要我介入的四个点

1. E1 的仪器化补丁（给出你 pin 的 vLLM commit，我按该版本写注入代码）。
2. E1 结果解读（把 `summary.md` 与 `ratio.csv` 发我，我判"死/灰区/活"并给下一步）。
3. frontier 的截断实现与 Go/No-Go 判定。
4. #4 门的 gap 声明终稿。
