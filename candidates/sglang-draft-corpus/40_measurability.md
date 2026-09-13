# S4 可测量性与上界（Measurability & ceiling） — 草案语料生命周期：跨会话/跨副本的冷启动罚金

**候选**：`sglang-draft-corpus` ｜ **成本上限**：≤2 GPU·h（仪器冒烟）
**目标**：在花 GPU 之前算出**最好情况能赚多少**。上界不过 8% 就不要开跑。

> **时点声明**：本文件写于**正式实验数据采集之前**（2026-09-13）。§仪器冒烟里的数字来自
> `probes/p12-draft-corpus/smoke.sh` 的一次 **160 格**运行，它是**仪器验证**（decision #100），
> 不是判据检验；判据检验在 S5 预登记之后。

## 观测量与其恒等式

### 主观测量：接受长度（accept length）

```
spec_accept_length = completion_tokens / spec_verify_ct        （含 bonus token）
```

出处：`sglang/srt/managers/tokenizer_manager.py:2811-2814`（SGLang 0.5.19 已装树，读正文核对）。
**单位**：token / 每 verify step。**取值上界** `= speculative_num_draft_tokens = 8`（本实验 γ=7 ⇒ 每步最多接受 7 个 draft + 1 个 bonus）。

### 吞吐恒等式（把"每步产出"与"每步代价"分开）

```
Δln(吞吐)  =  Δln(接受长度)  −  Δln(每步代价)
```

与 decision #67/#76 的 `tpot ≈ itl / accept_len` 同源。**注意** decision #82 的就地更正：
`mean_itl_ms` 是**每流式 chunk** 而非每步 ⇒ 本式只用**接受长度**与**端到端 wall**，不依赖 `itl` 语义。

### 冷启动罚金与可回收性（本候选的自变量）

```
cold_penalty = ( accept_steady − accept_control_first ) / accept_steady
recovery     = ( accept_treat_first − accept_control_first ) / ( accept_steady − accept_control_first )
```

- **对照首通 `accept_control_first`** = **全新 serve 进程 + 空语料**下的**第一遍** 32 条 prompt；
- **稳态 `accept_steady`** = 同一组 prompt 在同一 serve 内第 4–5 遍（语料已饱和）；
- **干预首通 `accept_treat_first`** = **全新 serve 进程 + 启动后立刻装入预置语料**下的**第一遍**。
- **`recovery` 即"可回收性"的量化定义**：预置语料把冷启动罚金闭合了多少。

### 三条采集路径（交叉验证用，PIPELINE §3.2）

| # | 路径 | 口径 | 状态 |
|---|---|---|---|
| ① | **每请求响应体** `meta_info.spec_accept_length` / `spec_accept_rate` | 见上（每请求、含 bonus） | ✅ 冒烟 **160/160** 采到 |
| ② | `/metrics` 的 `sglang:spec_accept_length` | **batch 级 Gauge**（`observability/metrics_collector.py:424`） | ⚠️ **gauge，跑完归零** ⇒ 只能**运行中高频轮询**（decision #66 的同源教训） |
| ③ | 客户端 wall / `e2e_latency` | 独立于引擎内部计数 | ✅ 冒烟采到 |

### 已排除的一条路径（如实记录，不静默删除）

原计划用 `/get_internal_state` 的**累计计数器** `spec_total_num_accept_tokens / spec_total_num_forward_ct` 做差。
**该端点在 SGLang 0.5.19 上返回 404** —— serve 日志逐条记录：
`"GET /get_internal_state HTTP/1.1" 404 Not Found`（`/root/ccfa_results/2026-09-13/p12_smoke/serve.log`）。
**替代方案**：把路径 ②（Gauge 轮询）与路径 ① 在同一时间窗内对照；**一致（<2%）** 则可当独立读数，
**不一致** 则本实验只保留 ① + ③，并在结论里写明"缺第二条独立路径"。

## 仪器冒烟与口径核对

**冒烟配置**：SGLang 0.5.19 / Qwen3-4B / `--speculative-algorithm NGRAM --speculative-num-steps 7 --speculative-num-draft-tokens 8` / `--disable-radix-cache` / `--context-length 8192` / `--mem-fraction-static 0.7` / ctx=4096 prompts / 32 prompts × 128 out-tokens × 5 reps 串行。

**冒烟结果（160/160 无错）**：

| rep | n | acc_len 均值 | acc_rate 均值 | wall_s 均值 | cached_tokens |
|---|---|---|---|---|---|
| 0 | 32 | **1.8365** | 0.1228 | 0.788 | 0 |
| 1 | 32 | **6.9744** | 0.8633 | 0.300 | 0 |
| 2 | 32 | 7.6005 | 0.9535 | 0.276 | 0 |
| 3 | 32 | 7.7696 | 0.9747 | 0.271 | 0 |
| 4 | 32 | 7.7827 | 0.9772 | 0.271 | 0 |

**口径核对的三条结论**：

1. **接受长度不是进程预热**：rep0 的**前 8 条**均值 **1.713** 与**后 8 条**均值 **1.712** 逐位持平
   ⇒ 第 1 遍**全程**都在 ~1.8，不存在"越跑越快"的进程暖机；第 2 遍才整体跃升
   ⇒ **变量是语料状态，不是进程状态**。这一条把"冷启动"从含糊词变成可辩护的机制主张。
2. **`cached_tokens` 恒为 0**：因为 `--disable-radix-cache` ⇒ 收益**不可能**来自前缀缓存，**只能**来自 NGRAM 语料
   ⇒ 冒烟阶段就把最危险的替代解释（prefix cache）排除。
3. **可达上界 = 8**：稳态 7.78 已贴近 `num_draft_tokens=8` 的天花板 ⇒ 该基准下**接受长度没有更多空间**，
   要再赚只能降**每步代价**（不在本候选范围）。

**冒烟抓到的两条仪器缺陷（已修 / 已绕）**：

- `FileNotFoundError: 'ninja'` —— venv 的 `bin/` 不在 PATH 时 flashinfer JIT 失败，报错会伪装成"GPU 不支持"。
  **已修**：`smoke.sh` 中 `export PATH="$VENV/bin:$PATH"`。
- `/get_internal_state` 404 ⇒ 累计计数器路径不可用，已改用 §采集路径的方案。

## oracle 上界

**定义**：完美决策（此处 = **在任何 serve 上，从第 1 个请求起就有完整语料**）相对**空语料**能赚多少。

**实测数字（冒烟，L2）**：

| 量 | 空语料（rep0） | 饱和语料（rep4） | 比值 |
|---|---|---|---|
| 接受长度 | **1.8365** | **7.7827** | **4.24×**（+323.8%） |
| wall / 32 请求 | 25.22 s | 8.67 s | **2.91×** |
| `spec_verify_ct`（每 128 token 的解码步数） | ~70（均值） | ~16.4 | 4.27× |

**oracle 上界 = Δln(接受长度) = ln(4.2384) = +144.4%**（按吞吐恒等式，假设每步代价不变）。
若按实测 wall 计，**+190.8%**（32 请求窗口）。

**⇒ 上界 ≈ +144%（保守口径）／+191%（wall 口径），远超 8% 闸门 ⇒ 通过。**

**上界的适用范围**：只在**被冒烟的这一个配置**内有效（Qwen3-4B + NGRAM γ=7 + ctx 4096 + 32 条同分布 prompt + 串行），
**不外推**到其他模型 / drafter / 并发 / 上下文长度（TOPIC_METHODOLOGY §4.4 上界有效范围）。

**避免夸大的第二口径**：上式 "steady" 是**同一组 prompt 重复出现**的结果，是语料累积的**最好情形**；
真实流量下每条 prompt 只出现一次，语料靠**跨请求重叠**增长。因此本候选主张的**不是**"每个请求都能到 7.78"，
而是"**冷窗口这一段**可被预置语料提前闭合"，判据用 `recovery ≥ 80%` 这种**相对量**而非绝对接受长度。

## 噪声预算

| 观测量 | 实测噪声（冒烟） | 依据 |
|---|---|---|
| **接受长度（稳态）** | rep3 vs rep4 = 7.7696 vs 7.7827 ⇒ **0.17%**；rep2 vs rep3 ⇒ **2.2%** | 语料饱和后极稳 |
| **接受长度（空语料 rep0）** | 逐请求 1.32–2.51；**前 8 vs 后 8 均值差 0.06%** | 组内稳定，组间（不同 prompt）有差异 |
| **wall（单请求）** | 0.264–1.056 s（跨 prompt） | 受 prompt 难度影响 ⇒ 必须用**同 prompt 集**对照 |

**判据的可分辨性要求（写死）**：

- 主观测量 = **接受长度**（噪声 ≤2.2%）⇒ 判据 `recovery ≥ 80%` 对应效应量 ≥ **0.8 × 5.95 = 4.76 token**，
  是噪声（0.17% × 7.78 ≈ 0.013 token）的 **≈360×** ⇒ 远超"效应 > 3× 噪声"门槛。
- 若用 **wall** 作辅助判据：要求 ≥3× 噪声，且**噪声用正式实验实测的 3 个独立 serve 极差**代入，
  不沿用本节估计。
- **k 值写死**：主判据 k = 3；`recovery` 的 80% 阈值本身已含 ≥300× 余量。

## 环境前提

| 前提 | 值 | 违反的后果 |
|---|---|---|
| 驱动 / CUDA | 580.82.09 / CUDA 13（sm120） | 引擎起不来 |
| 引擎 | **SGLang 0.5.19**（pin；每条结果带版本号） | 指标语义可能变 |
| 模型 | Qwen3-4B（target，bf16）+ **NGRAM draft（无独立 drafter 权重）** | 换 drafter 会改变接受长度语义 |
| locale | 脚本开头 `export LC_ALL=C.UTF-8` | `import readline` 段错误、引擎静默死 |
| `libstdc++` | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` | NGRAM 的 JIT 内核加载失败，报错伪装成"GPU 不支持" |
| `PATH` | 含 `$VENV/bin`（flashinfer JIT 要 `ninja`） | 启动即 `FileNotFoundError: 'ninja'` |
| venv | **独立** `/root/autodl-tmp/venvs/sglang`，与 vLLM 不混装 | 依赖互踩 |
| GPU 独占 | **同一张卡同一时间只跑一个 serve** | 显存与读数互相污染 |
| prompt 集 | **真实文本** `prompts_4096.jsonl` 前 32 条，**固定顺序、不 shuffle** | `--dataset-name random` 或 shuffle 会改变接受长度语义（decision #76） |
| 采样 | `temperature=0`、`ignore_eos=true`、`max_new_tokens=128` 全网格一致 | 输出长度不同 ⇒ 接受长度不可比 |
| 输出长度 | 128（固定） | 早期 EOS 会污染接受长度 |
| 缓存 | 所有臂 `--disable-radix-cache` | 前缀缓存会混淆"语料"这一归因 |
| 结果落点 | 仓库外 `/root/ccfa_results/2026-09-13/p12_corpus/` | 仓库同步会删掉结果 |

## 可采纳前置条件

**本测量在下列任一情况发生时无效，必须显式排除后才能下结论**：

1. **`cached_tokens > 0`** 出现在任何对照臂 ⇒ 前缀缓存介入，接受长度变化不能归给语料（本实验所有臂均 `--disable-radix-cache`）。
2. **路径 ① 与路径 ② 在同一时间窗内不一致 >2%** ⇒ 只保留 ① + ③，并标注"无第二条独立路径"。
3. **confirmation 臂未复现冷启动罚金**（`accept_cold ≪ accept_steady` 不成立）⇒ 该次运行作废，不用于判定。
4. **干预臂语料装载失败或被截断**（`success != true` 或 message 含 `truncated`）⇒ 干预未生效，**不得**当作"干预无效"。
5. **`max_running_requests` 未显式设置**（引擎对投机解码会把默认重置为 48）⇒ 并发口径漂移；本实验为**串行**（同时只有 1 个请求），不受影响，但结果里必须写明。
6. **跨臂污染**（前一个臂的进程/显存残留）⇒ 每个臂必须**独立 serve 进程**，并在结论里给进程 PID 与启动时间。

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] 观测量用**恒等式**定义（写成 `A = B − C` 或等价形式），不是『看起来相关的指标』 —— `spec_accept_length = completion_tokens / spec_verify_ct`（`tokenizer_manager.py:2811-2814`）＋ 吞吐恒等式 `Δln(吞吐) = Δln(接受长度) − Δln(每步代价)` ＋ `cold_penalty` / `recovery` 的比值定义
- [x] 仪器经冒烟验证，并记下**单位/口径/计数器 vs gauge** 的核对结果 —— 160/160 无错；单位 = token/每 verify step、上界 = `speculative_num_draft_tokens` = 8；**`sglang:spec_accept_length` 是 batch 级 Gauge（跑完归零）**、**`/get_internal_state` 在 0.5.19 上 404**（已记录并给替代方案）
- [x] 算出 **oracle 上界**的具体数字（完美决策能拿到多少），并写成百分比 —— 接受长度 **1.8365 → 7.7827 = 4.24×** ⇒ **上界 = +144.4%**（接受长度口径）／**+190.8%**（wall 口径），并写明只在被冒烟配置内有效
- [x] oracle 上界 ≥ 8%（低于此值立即杀，不要指望实验里变大） —— **+144.4% ≫ 8%** ⇒ 通过（+190.8% 为第二口径佐证）
- [x] 噪声预算在**采数之前**估计（同配置重复、极差或置信区间），并写进判据 —— 接受长度稳态同格 **0.17%**、跨 rep ≤2.2%；空语料前 8 vs 后 8 **0.06%**；判据 `recovery ≥ 80%` 对应 ≈360× 噪声；wall 噪声**待实测**并在正式判定时代入
- [x] 环境前提逐条列出（驱动/引擎版本/dtype/串行化开关/数据集必须真实文本等） —— 见 §环境前提 13 条（含 `ninja` / `libstdc++` / locale / 独占 GPU / 固定 prompt 与采样参数）
- [x] 写出结论的**可采纳前置条件**（什么情况下这个测量无效，必须显式排除） —— 见 §可采纳前置条件 6 条（`cached_tokens>0`、两路径不一致>2%、confirmation 未复现、语料装载失败/截断、并发口径、跨臂污染）

> **杀出口**：oracle 上界 < 8% ⇒ 杀（本仓库实证：T2 上界只有 +0.1%）；仪器口径未核 ⇒ 禁止采数
> 本候选**未触发**：上界 +144.4%，仪器口径已核。
