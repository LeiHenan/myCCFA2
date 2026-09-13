# aux-c3 结果：混合模型的「状态 vs KV」零卡预算（2026-09-13）

**探针**：`probes/aux-c3-hybrid-state/budget.py`（含 `--selftest`）
**候选**：`candidates/hybrid-state-serving`（S1 取证任务第 1 步，decision #72）
**成本**：**0 GPU·h**（只用 `config.json` + vLLM 自身公式；配置副本存 `candidates/hybrid-state-serving/evidence/configs/`）
**复现**：`python probes/aux-c3-hybrid-state/budget.py --dir candidates/hybrid-state-serving/evidence/configs --out results/aux-c3-hybrid-state/2026-09-13 --block-size 16`

## 结论（一句话）

**P3「状态内存与 KV 双重占用」成立，且已从 L0 升到 L1**：三个真实混合模型的**每序列循环状态 = 1.9–67 MiB**，与「一份 KV 等价的状态」在 **1.6k–3.2k tokens** 处交叉 ⇒ 在 **4k 上下文、bs=32** 这种主流配置下，**状态占每序列缓存内存的 27.5%–43.6%**。

## 主表（bf16；状态形状逐字取自 vLLM `mamba_utils.py:150-300`）

| 模型 | 层数（状态/注意力/MLP） | KV /token | **状态 /序列** | **交叉点 L\*** | 状态占比 @bs32,4k | 状态占比 @bs32,32k |
|---|---|---|---|---|---|---|
| `nvidia/Nemotron-H-8B-Base-8K` | 52 (24/4/24) | 16 KiB | **49.41 MiB** | **3162** | **43.6%** | 8.8% |
| `tiiuae/Falcon-H1-7B-Instruct` | 44 (44/44/0) | 44 KiB | **66.90 MiB** | **1557** | **27.5%** | 4.5% |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 48 (36/12/0) | 24 KiB | **37.69 MiB** | **1608** | **28.2%** | 4.7% |
| `state-spaces/mamba2-2.7b`（纯 SSM） | 64 (64/0/0) | 0 B | 1.88 MiB | ∞ | 100% | 100% |
| `ibm-fms/Bamba-9B` | — | — | — | — | ⚠️ **vLLM 0.29 不支持**（无 `bamba.py`，注册表无该架构） | — |

## 本步最硬的一条：粒度—内存权衡

`检查点开销 / KV = L* / b`（**与上下文长度无关**；b = `mamba_block_size` 检查点粒度）

| 模型 | L\* | b=16 | b=256 | b=1024 | b=4096 |
|---|---|---|---|---|---|
| Nemotron-H-8B | 3162 | **197.6×** | 12.4× | 3.1× | 0.77× |
| Falcon-H1-7B | 1557 | **97.3×** | 6.1× | 1.5× | 0.38× |
| Qwen3-Next-80B | 1608 | **100.5×** | 6.3× | 1.6× | 0.39× |

⇒ **设计约束：只要 `b < L*`，「缓存状态检查点」就比「缓存 KV」更贵。** 而 KV 的块粒度是 16
⇒ 想要**与 KV 同粒度**的状态复用，代价是 **KV 本身的 ~100–200 倍**。这不是"配错了参数"，
而是两个内存池的**粒度—价值曲线形状不同**：KV 的复用单位是块（16 tokens），状态的复用单位
是"整份状态（1.9–67 MiB）"，两者的**最小可复用单位相差 3–4 个数量级**。

## 必须写进结论的限制（诚实边界）

1. **这是内存主张，不是吞吐主张**：状态内存占比高 ≠ 吞吐损失 ≥8%。若这些状态只是"占着但不引发抢占"，
   痛点就不构成可写课题 ⇒ **必须做取证第 2 步（≤1 GPU·h）**：看在真实 serve 下状态是否真的挤压并发/触发抢占。
2. **痛点集中在短—中等上下文**：32k 时状态占比降到 4.5–8.8%。但 4k×bs32 是主流生产配置（对话/agent 步进），
   **不是"极端角落"**（伪问题形态⑦ 不命中）。
3. **`mamba_block_size` 默认值仍是推定**：config 层是 `None`，`validate_mamba_block_size` 把 `== max_model_len`
   当作"未显式设置"（`vllm/config/vllm.py:2839-2841`）⇒ 默认疑似 = `max_model_len`（整段只存一份）。
   **须在 S4 用一次真实 serve 日志确认**（`GPU KV cache size` 那一类日志行）。
4. **`bamba` 不可用**：该架构不在 vLLM 0.29 注册表内 ⇒ 若要用 Bamba 做第二家族，需换引擎版本或换模型。
5. **`L*` 是"一份状态 ↔ 整段 KV"的交叉点**，不是命中率模型：它给的是**内存量级**，不预测 prefix 命中率
   （那是 P2，仍需实测）。

## 下一步（仍属 S1 取证，≤1 GPU·h）

- **第 2 步**：单卡跑 `mamba_cache_mode ∈ {none, align, all}` × prefix cache 开/关，量**吞吐 / 显存 / 同前缀命中率**；
  同时**确认 `mamba_block_size` 的实际默认值**（读 serve 日志）。
- **第 3 步（P2 专项）**：同一 token 前缀走两种批处理路径，看状态检查点命中是否不同。
- **判定线**：三档吞吐差 <8% ⇒ P1/P3 不构成可写课题，直接杀（省下 4 周）。
