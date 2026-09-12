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
| `unexpected keys` / 严格加载报错 | 权重文件含全部层 | `--prune-weights`，或用 vLLM 的非严格加载路径 |
| 输出损坏 / 重复 token | `mask_token_id` / confidence head 依赖深层 hidden | 记录为 **B**；不要硬修 |
| 所有 `d` 完全一致 | 截断未生效（config 未被读取）或质量由 aux states 决定 | 先确认加载的是变体目录；仍一致 ⇒ 归为 **C** |
| OOM | `ctx=4096` 不该 OOM；若发生，先查是否误加载了 full-depth 权重 | 降低 `gpu-memory-utilization` 或换档 |
