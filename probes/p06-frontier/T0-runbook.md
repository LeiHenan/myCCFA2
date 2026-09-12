# T0 — 深度旋钮验证（半天，1 卡）

**目标**：确认 `#6` 的前提是否成立 —— **同一个 drafter 的网络深度是不是一个"可操作的旋钮"**。
**判据**：见 §4。**时间盒**：半天。**未在真机验证过**：命令模板里的 flag 以 `vllm serve --help` / `vllm bench serve --help` 为准。

---

## 1. 为什么先做它

T1 要测"最优 depth 与最优 γ 是否可分离"。如果深度**根本不可操作**（截断即崩、或截断后行为完全不变），T1 的 18 格就是白跑 —— T0 用半天把这条路先探明。

依据（`notes/p06-toolchain-check.md`）：
- vLLM 侧 `qwen3_dflash.py` 用 **`config.num_hidden_layers`** 建层；
- 权重 `config.json` 含 **`transformer_layer_config`**（逐层配置）；
- speculators 训练侧有 **`--num-layers`**（一等参数）。

⇒ 首选"**改 config**"，不要重训、也先不要改代码。

> ✅ **2026-09-12 按真实权重核对（重要，曾会导致 T0 假阴性）**：`mgoin/Qwen3-4B-speculator.dflash2` 的
> `config.json` **顶层没有** `num_hidden_layers`；层数在 **`transformer_layer_config.num_hidden_layers`（嵌套 dict）**，
> 且同层还有 `layer_types`（逐层 SWA/Full 标记，长度必须与层数一致）。
> vLLM 侧对应 `DFlashQwen3Model.layers = [.. for i in range(self.config.num_hidden_layers)]` 与
> `_dflash_layer_causal()` 的 `layer_types[layer_idx]`，其中 `self.config` 就是 `transformer_layer_config`。
> `make_depth_variants.py` 已据此修好（支持嵌套 dict + 同步截短 `layer_types`），并加了**护栏**：
> 若一处字段都没改动就直接报错退出 —— 否则会产出与源逐字节相同的"变体"，让 T0 得到"所有深度行为一致"的**假阴性**。

## 2. 准备：生成深度变体

```bash
# 1) 取权重（示例：Qwen3-4B 的 dflash2）
huggingface-cli download mgoin/Qwen3-4B-speculator.dflash2 \
  --local-dir upstream/drafters/src

# 2) 生成 d ∈ {1,2,3,4,5} 变体（只改 config；默认不动权重张量）
python probes/p06-frontier/make_depth_variants.py \
  --src upstream/drafters/src --out upstream/drafters --depths 1 2 3 4 5

# 3) 若 vLLM 严格加载报 unexpected keys，再叠 --prune-weights（需 safetensors）
python probes/p06-frontier/make_depth_variants.py \
  --src upstream/drafters/src --out upstream/drafters --depths 1 2 3 4 5 --prune-weights
```

**备选路径（仅当改 config 无效）**：在 draft 模型里对层列表切片（`self.layers = self.layers[:d]`）；做法与撤销方式同 `probes/p01-e1-uncommitted-kv/instrument/e1_patch.py`（幂等 + 备份）。
**最后手段**：用 speculators `--num-layers` 训练不同深度 —— **本阶段不做**（要数据与时间），仅在 T0 结论为"灰区 C"时备案。

## 3. 测什么

固定 `bs=1, ctx=4096, γ=7`，对每个 `d ∈ {1..5}`：

| 记录项 | 来源 |
|---|---|
| 能否加载 / 有无 NaN / 输出是否损坏 | 服务日志 + 固定 prompt 的输出比对 |
| **接受长度**（accepted tokens/step） | `/metrics` 的 `vllm:spec_decode_num_accepted_tokens*` 与 `num_draft_tokens*`，或 `--save-result` 的 JSON |
| per-step draft 耗时、verify 耗时 | 服务日志 / profiler（可选） |
| `tok/s` | `vllm bench serve` |
| 配置事实 | 该变体的 `num_hidden_layers`、`transformer_layer_config` 长度、权重文件大小 |

**lossless 检查**：同一 prompt 在 `d=full` 与基准（未裁剪）下，前 32 token 应一致。

## 4. 判定（T0 的三种结局）

| 结局 | 观测 | 动作 |
|---|---|---|
| **A. 旋钮可操作** | 接受长度/耗时随 `d` **可动**（单调或单峰），且 `d=full` 与基准一致 | **进 T1** |
| **B. 旋钮不可操作** | 加载失败 / NaN 或损坏输出 / 所有 `d` 的行为**完全相同** | **`#6` 当天降级**（深度不是资源 ⇒ 无从分配） |
| **C. 灰区** | 能跑但接受长度几乎不随 `d` 变 | 记录并升级为 T1 的前置问题 —— **重要发现**：说明 draft 质量主要由**目标模型喂入的 aux hidden states** 决定，而非 drafter 深度 ⇒ 对 `#6` 是坏消息，须在 T1 结论中显式写出 |

## 5. 产物

```
results/p06-frontier/<YYYY-MM-DD>/T0/
├── summary.md        # 三结局判定 + 表格（d × {accepted_len, draft_ms, tok/s}）+ lossless 检查
├── d1.json … d5.json # 每档原始数字
└── run.log
```

## 6. 已知失败形态与排查

| 症状 | 可能原因 | 处置 |
|---|---|---|
| `ValueError: There is no module or parameter named 'layers.1'` | **权重文件含全部层**，而 config 只声明 d 层；vLLM 严格加载 | **必须** `make_depth_variants.py --prune-weights`（实测：软链/仅改 config 都会撞这条） |
| 输出损坏 / 重复 token | `mask_token_id` / confidence head 依赖深层 hidden | 记录为 **B**；不要硬修 |
| OOM | `ctx=4096` 不该 OOM；若发生，先查是否误加载了 full-depth 权重 | 降低 `gpu-memory-utilization`（本机用 0.5）或换档 |

> ❌ **已作废的一条**（2026-09-12 实测更正）：早期本表把"所有 `d` 的输出逐字相同"当成失败形态。
> **这是错的** —— 投机解码**无损**，各档 greedy 输出**本就应当逐字相同**，那恰恰是 lossless 检查**通过**。
> 深度是否可操作，只看**接受长度 / tok/s 是否随 d 变动**（见 §7 实测表）。

---

## 7. 实测踩坑清单（2026-09-12，RTX PRO 6000 96 GB + vLLM 0.29.0 + Qwen3-4B + DFlash2）

这一节是**真机跑通 T0 的全部代价**，按"症状 → 根因 → 修法"排列：

| # | 症状 | 根因（已定位到源码/环境） | 修法 |
|---|---|---|---|
| 1 | `Engine core initialization failed ... Failed core proc(s): {}`，**且无 Python traceback** | 容器把 `LC_ALL=en_US.UTF-8` 写进环境但该 locale **未生成** ⇒ `setlocale()` 失败 ⇒ python `import readline` **段错误**（栈：`rl_initialize → _rl_init_locale → PyInit_readline`），日志里只有一行 `!!!!!!! Segfault encountered !!!!!!!` | 任何 python 进程之前 `export LC_ALL=C.UTF-8; export LANG=C.UTF-8` |
| 2 | `RuntimeError: mha_varlen_fwd ... query and key must have the same dtype` | `--kv-cache-dtype float16` 而**模型是 bfloat16** | KV dtype 必须与模型一致：用 `auto` 或 `bfloat16`（合法值只有 auto/float16/bfloat16/fp8*，**没有 `fp16`**） |
| 3 | **接受率恒为 0**（`Mean acceptance length: 1.00`），但模型文件完好 | vLLM 0.29 的 method 推断**只看路径字符串**：`"dflash" in draft_model_config.model.lower()`（`DFlash2DraftModel` **不在**架构白名单）⇒ 变体放在 `.../drafters/d5` 会被当成通用 `draft_model`，DFlash 专属接线失效；日志会出现 `speculative method 'draft_model'; using the V1 model runner instead` | **变体目录路径必须含 "dflash"**，例如 `DRAFTER_ROOT=/root/autodl-tmp/dflash-variants`；`run_t0.sh`/`run_t1.sh` 已有护栏会直接拒绝 |
| 4 | 接受率恒为 0（另一个原因） | `vllm bench serve --dataset-name random` 喂**随机 token id**，drafter 无从预测 | 用真实文本：`make_prompts.py` 造 `prompts_<ctx>.jsonl`，再 `DATASET=custom DATASET_DIR=...`（实测同一 drafter：随机数据 0/7112，真实文本 158/707 = 22%） |
| 5 | 启动慢、显存被占满 | 96 GB 卡沿用 `--gpu-memory-utilization 0.85` ⇒ 预留约 82 GB KV cache | 用 `0.5`（≈48 GB，bs=1 足够），起服务约 65 s |

### T0 实测结果（可作为"结局 A"的参照）

`ctx=4096`、`γ=7`、真实文本（WikiText）、`KV=bfloat16`、`GPU_UTIL=0.5`：

| depth | config 层数 | 平均接受长度 | 接受率 | tok/s |
|---|---|---|---|---|
| 1 | 1 | 1.303 | 0.043 | 101.5 |
| 2 | 2 | 1.523 | 0.075 | 114.4 |
| 3 | 3 | 1.725 | 0.104 | 124.2 |
| 4 | 4 | 2.099 | 0.157 | 144.6 |
| 5 | 5 | **2.829** | 0.261 | **180.3** |

⇒ 接受长度与吞吐**均随深度单调增长**，lossless 检查通过 ⇒ **结局 A（旋钮可操作）**，进 T1。
（并已排除权重因素：变体与源权重 83 个张量**逐位相同**、config 键集相同。）
