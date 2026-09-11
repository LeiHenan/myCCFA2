# `#6` 工具链核对（解冻步骤 ② 的交付物）

**执行**：2026-09-12（无卡）｜ **结论**：**通过 —— 可开工**；卡型建议见 §5
**pin**：vLLM `main` = **`9a35c081e80a94828af6f611525102bb70e3c67f`**（2026-09-12 取自 `git ls-remote`）
**对照材料**：`upstream/vllm-main-check/`（`version.py` / `kv_cache_manager.py` / `scheduler.py`，从该 pin 的 raw 下载）

---

## 1. E1 仪器化锚点：**全部命中，`e1_patch.py` 无需修改**

| 锚点 | 该 pin 上的出现次数 |
|---|---|
| Hook A（`if request.num_output_placeholders > 0:` + `-= num_rejected`） | **1** |
| `num_rejected = num_draft_tokens - num_accepted` | **1** |
| `class KVCacheManager` ／ `def allocate_slots(` | 1 ／ 1 |
| `def get_block_ids`（Hook B 依赖） | 存在（5 处） |

⇒ 打补丁后可直接跑，无需按版本改写。

## 2. 基线 ④ 在该 pin 存在

`scheduler.py` 中 `num_speculative_tokens_per_batch_size` 出现 **2 次** ⇒ "per-batch K 查表"这个对照基线在最新 main 上成立。

## 3. 多层 drafter：**可得**（这是 ② 的核心结论）

**vLLM main 的模型注册表**（`models/registry.py`）：

| 注册名 | 实现文件 | 说明 |
|---|---|---|
| `DFlashDraftModel` | `qwen3_dflash` | DFlash 草稿 |
| `DFlash2DraftModel` | `qwen3_dflash2` | DFlash2 草稿（对应我们的 4B 权重） |
| — | `qwen3_dspark.py`（存在，HTTP 200） | DSpark 草稿 |
| 另有 | `llama_eagle` / `llama4_eagle` / `cohere_eagle` / `minicpm_eagle` / 多种 `*_mtp` | EAGLE 与 MTP 家族 |

**已发布的权重**（HF，`mgoin` 名下 6 个）：

| 权重 | target | 备注 |
|---|---|---|
| **`mgoin/Qwen3-4B-speculator.dflash2`** | **Qwen3-4B** | `block_size: 8`、`speculative_tokens: 7` ⇒ **24 GB 卡可跑** |
| `mgoin/Qwen3-8B-speculator.dspark` / `.dflash` / `.dspark-reasoning` | Qwen3-8B | 8B 更适合 80 GB |
| `mgoin/GLM-5.2-speculator.dspark` / `.dspark-block16` | GLM-5.2-FP8（前沿 MoE） | **需多卡节点，超出本阶段预算** |
| （另有 `RedHatAI/Qwen3-8B-speculator.peagle`） | Qwen3-8B | P-EAGLE |

## 4. 深度旋钮：**是配置，不是硬编码**（T0 因此很便宜）

- vLLM 侧：`qwen3_dflash.py` 逐行 **`_dflash_layer_causal(config, i) for i in range(config.num_hidden_layers)`** ⇒ **草稿层数由权重 config 的 `num_hidden_layers` 驱动**。
- 权重侧：Qwen3-4B-dflash2 的 `config.json` 含 **`transformer_layer_config`**（逐层配置）。
- 训练侧：speculators 的 DSpark 文档逐字 —— *"All DFlash parameters (`--block-size`, `--max-anchors`, **`--num-layers`**, ...) apply unchanged."* ⇒ **`--num-layers` 是一等参数**。

⇒ **T0（旋钮验证）的第一步 = 改 `num_hidden_layers`（并相应裁剪权重）后重载**，无需改代码。**待验风险**：anchored-block 与 Markov/confidence 头在浅层下的行为（可能需要在 T0 中一并观察接受长度是否连续退化）。

## 5. 卡型建议（显存算术）

Qwen3-4B：36 层 / 8 KV heads / head_dim 128 ⇒ **144 KiB/token（FP16）**，权重 ≈ 8 GiB。

| ctx | KV（FP16） | KV（FP8） |
|---|---|---|
| 4k | 0.56 GiB | 0.28 GiB |
| 32k | 4.50 GiB | 2.25 GiB |
| 128k | **18.0 GiB** | **9.0 GiB** |
| 185k | 25.4 GiB | 12.7 GiB |

- **24 GB 卡（推荐起步）**：8（权重）+ ~1（drafter）+ KV ⇒ **4k/32k 宽裕**；**128k 需 `--kv-cache-dtype fp8`**（8+1+9 ≈ 18 GiB，含激活/CUDA graph 偏紧但可行）。`bs{1}` 是 T1 的网格，正好回避了 KV 随 batch 放大。
- **80 GB 卡（省心档）**：128k/185k 与 bs>1 全部宽裕，无需 FP8 KV。
- **注意**：KV dtype 是**配置**而非处理变量，**所有对照条件必须同 dtype**（否则差异不可归因）。

⇒ **① 的决策**：T0 + T1 用 **1×24 GB** 即可（配 FP8 KV 或 ctx 收到 32k）；只有需要 8B target 或 **185k 格/bs>1** 时才升到 80 GB。**GLM-5.2 系 drafter 不在本阶段预算内。**
