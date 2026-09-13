# 零卡预算：混合模型的「状态内存 vs KV 内存」

> 口径：bf16（2 B/elem）；状态形状逐字取自 vLLM `model_executor/layers/mamba/mamba_utils.py:150-300`；`mamba_block_size = 16`（检查点粒度，vLLM 默认 `DEFAULT_BLOCK_SIZE=16`，用户可用 `--mamba-block-size` 调大）。

| 模型 | 架构 | 层数 (状态/注意力/MLP) | KV /token | **状态 /序列** | **交叉点 L\*** | 检查点开销 /KV（= L\*/b） |
|---|---|---|---|---|---|---|
| Qwen_Qwen3-Next-80B-A3B-Instruct | qwen3_next | 48 (36/12/0) | 24.00 KiB | **37.69 MiB** | **1608 tokens** | **100.5×** |
| ibm-fms_Bamba-9B | — | — | — | — | — | ⚠️ 未支持的 model_type='bamba'（本脚本只覆盖 ['falcon_h1', 'mamba2', 'nemotron_h', 'qwen3_next']） |
| nvidia_Nemotron-H-8B-Base-8K | nemotron_h | 52 (24/4/24) | 16.00 KiB | **49.41 MiB** | **3162 tokens** | **197.6×** |
| state-spaces_mamba2-2.7b | mamba2(纯SSM) | 64 (64/0/0) | 0 B | **1.88 MiB** | **inf tokens** | **inf×** |
| tiiuae_Falcon-H1-7B-Instruct | falcon_h1 | 44 (44/44/0) | 44.00 KiB | **66.90 MiB** | **1557 tokens** | **97.3×** |

## 明细（状态占总缓存的比例；检查点按最坏情形 b=16 整段前缀都存）

| 模型 | batch | ctx | KV | 状态(常驻) | 状态占比 | 检查点开销 | 检查点/KV |
|---|---|---|---|---|---|---|---|
| Qwen_Qwen3-Next-80B-A3B-Instruct | 1 | 4096 | 96.00 MiB | 37.69 MiB | **28.2%** | 9.42 GiB | 100.5× |
| Qwen_Qwen3-Next-80B-A3B-Instruct | 1 | 32768 | 768.00 MiB | 37.69 MiB | **4.7%** | 75.38 GiB | 100.5× |
| Qwen_Qwen3-Next-80B-A3B-Instruct | 32 | 4096 | 3.00 GiB | 1.18 GiB | **28.2%** | 301.50 GiB | 100.5× |
| Qwen_Qwen3-Next-80B-A3B-Instruct | 32 | 32768 | 24.00 GiB | 1.18 GiB | **4.7%** | 2412.00 GiB | 100.5× |
| nvidia_Nemotron-H-8B-Base-8K | 1 | 4096 | 64.00 MiB | 49.41 MiB | **43.6%** | 12.35 GiB | 197.6× |
| nvidia_Nemotron-H-8B-Base-8K | 1 | 32768 | 512.00 MiB | 49.41 MiB | **8.8%** | 98.81 GiB | 197.6× |
| nvidia_Nemotron-H-8B-Base-8K | 32 | 4096 | 2.00 GiB | 1.54 GiB | **43.6%** | 395.25 GiB | 197.6× |
| nvidia_Nemotron-H-8B-Base-8K | 32 | 32768 | 16.00 GiB | 1.54 GiB | **8.8%** | 3162.00 GiB | 197.6× |
| state-spaces_mamba2-2.7b | 1 | 4096 | 0 B | 1.88 MiB | **100.0%** | 480.00 MiB | inf× |
| state-spaces_mamba2-2.7b | 1 | 32768 | 0 B | 1.88 MiB | **100.0%** | 3.75 GiB | inf× |
| state-spaces_mamba2-2.7b | 32 | 4096 | 0 B | 60.00 MiB | **100.0%** | 15.00 GiB | inf× |
| state-spaces_mamba2-2.7b | 32 | 32768 | 0 B | 60.00 MiB | **100.0%** | 120.00 GiB | inf× |
| tiiuae_Falcon-H1-7B-Instruct | 1 | 4096 | 176.00 MiB | 66.90 MiB | **27.5%** | 16.73 GiB | 97.3× |
| tiiuae_Falcon-H1-7B-Instruct | 1 | 32768 | 1.38 GiB | 66.90 MiB | **4.5%** | 133.80 GiB | 97.3× |
| tiiuae_Falcon-H1-7B-Instruct | 32 | 4096 | 5.50 GiB | 2.09 GiB | **27.5%** | 535.22 GiB | 97.3× |
| tiiuae_Falcon-H1-7B-Instruct | 32 | 32768 | 44.00 GiB | 2.09 GiB | **4.5%** | 4281.75 GiB | 97.3× |

## 粒度—内存权衡：`检查点开销 / KV = L* / b`（与上下文长度 **无关**）

| 模型 | L\*（= 允许的最细粒度下界） | b=16 | b=256 | b=1024 | b=4096 | b=32768 |
|---|---|---|---|---|---|---|
| Qwen_Qwen3-Next-80B-A3B-Instruct | **1608 tokens** | **100.5×** | **6.3×** | **1.6×** | 0.39× | 0.05× |
| nvidia_Nemotron-H-8B-Base-8K | **3162 tokens** | **197.6×** | **12.4×** | **3.1×** | 0.77× | 0.10× |
| state-spaces_mamba2-2.7b | **inf tokens** | **inf×** | **inf×** | **inf×** | **inf×** | **inf×** |
| tiiuae_Falcon-H1-7B-Instruct | **1557 tokens** | **97.3×** | **6.1×** | **1.5×** | 0.38× | 0.05× |

> **设计约束（本步的核心结论）**：只要检查点粒度 `b < L*`，「缓存状态检查点」就比「缓存 KV」**更贵**。三个模型的 `L*` 都只有 **1.6k–3.2k tokens**，而 KV 的块粒度是 16 ⇒ **想拿到与 KV 同粒度的状态复用，代价是 KV 本身的 ~100–200 倍**。
> ⚠️ `mamba_block_size` 的**默认值**在 config 层是 `None`（构造 KV cache spec 时解析；`abstract.py:68-69` 断言其非空）。`validate_mamba_block_size` 把 `== max_model_len` 当作『未显式设置』（`vllm/config/vllm.py:2839-2841`）⇒ **默认疑似等于 max_model_len（整段序列只存一份）**。**推定，须在 S4 用一次真实 serve 的日志确认**。


## 读法（三条判据）

1. **`L*` = 状态/序列 ÷ KV每token** ⇒ 上下文短于 `L*` 时，**一份状态就比整段 KV 更占内存**。
2. **检查点开销 / KV = `L*` / b**（与 L 无关）⇒ b 每减半，检查点开销翻倍。
3. **P3 判定**：若某模型在 `batch=32, ctx=4096` 下状态占比 >5%，则「状态与 KV 双重占用」是真痛点（L1）；若 ≪1% 则 P3 淘汰。
