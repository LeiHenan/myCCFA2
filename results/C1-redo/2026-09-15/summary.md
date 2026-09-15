# C1-redo / 2026-09-15 — 摘要

**探针**：`instrument/`（重跑入口见文末）｜**预登记**：[`notes/prereg/C1-redo.md`](../../../notes/prereg/C1-redo.md)（采数前冻结）
**完整报告**：[`docs/C1重做_最终报告_2026-09-15.md`](../../../docs/C1重做_最终报告_2026-09-15.md)｜**方案审计**：[`docs/C1重做_方案审计_2026-09-15.md`](../../../docs/C1重做_方案审计_2026-09-15.md)
> **⚠️ 更正**：本目录 `raw/` 与四份 manifest 存在 2 个 run_id 的跨网格碰撞（core-dense 与 k-sens 共用
> `Qwen3-4B_ngram_k3_pc{0,1}`），`derived/analysis_core-dense.json` 的 ngram K=3 一行因此**无法从归档 raw 复算**。
> 影响范围、两轮数字与"结论不变"的论证见 [`CORRECTION.md`](CORRECTION.md)。

**机器**：A800-SXM4-80GB / 18 vCPU / vLLM 0.29.0 / torch 2.12.1+cu130 ｜ **成本**：28 次引擎启动 ≈ 1.6 GPU·h

---

## 1. 量 / 区间 / 基线

| 量 | 口径 | 基线 |
|---|---|---|
| **主量** decode_tps | 输出 token ÷ **解码段**墙钟（排除预填充） | 同架构同 drafter 的 **PC off** 臂 |
| TTFT p50 | 首 token 到达 | 单独报，不并入主量 |
| 命中 | 引擎自带 `vllm:prefix_cache_hits_total` 臂前后差分 | PC off 恒 0 |
| 接受率 | `spec_decode_num_accepted_tokens / draft_tokens`（引擎计数器） | 同 drafter 的 PC off 臂 |
| 噪声底 σ/μ | **同格内 3 次完全相同的暖臂**的离散度 | θ = max(1.05, 1+3σ/μ) |

**参照系**（§4.2）：噪声底 σ/μ 中位 **1.52%**（dense）/ **1.39%**（hybrid）⇒ θ=1.050；
跨配置分歧基线 **10.0%**（dense）/ **18.3%**（hybrid）；
同实例冷/暖基线见判据对照（**两者是不同量，不可互替**）。

## 2. 数字

### 2.1 P3' 解码期吞吐（暖臂，PC on/off）— 主判据

| target | drafter | PC-on | PC-off | on/off | 超 θ | seed 方向一致 |
|---|---|---|---|---|---|---|
| Qwen3-4B | none | 48.24 | 47.51 | 1.0152 | 否 | 否 |
| Qwen3-4B | ngram | 64.26 | 63.53 | 1.0114 | 否 | 否 |
| Qwen3-4B | eagle3 | 97.25 | 93.69 | 1.0381 | 否 | 否 |
| Qwen3-4B | dflash | 125.01 | 123.53 | 1.0120 | 否 | 否 |
| Qwen3.5-4B | none | 31.19 | 31.17 | 1.0008 | 否 | 否 |
| Qwen3.5-4B | ngram | 30.95 | 31.64 | 0.9782 | 否 | 是 |
| Qwen3.5-4B | eagle3 | 57.24 | 59.50 | 0.9620 | 否 | 是 |
| Qwen3.5-4B | dflash | 87.72 | 86.54 | 1.0136 | 否 | 否 |

**按采数前写死的 θ=5%：8 格无一显著。** 但按 seed 级 t 检验（5 seed 比值，df=4）出：

| target | drafter | 比值 | seed 间 SD | t | 95% CI | 结论 |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | **eagle3** | 0.9625 | 0.0179 | **−4.70** | [0.940, 0.985] | **排除 1** |
| Qwen3.5-4B | **ngram** | 0.9783 | 0.0072 | **−6.79** | [0.969, 0.987] | **排除 1** |
| Qwen3.5-4B | none（对照） | 1.0013 | 0.0292 | 0.10 | [0.965, 1.037] | 含 1 |
| Qwen3-4B | 四档全部 | — | — | −0.76…2.35 | 全部含 1 | — |

⇒ **存在小的、drafter 特异的、只在 hybrid 上出现的 PC 解码代价（2–4%）**；dense 上没有。

### 2.2 P2' 命中记账闸门

| grid | PC-on 暖格 | 其中命中>0 | PC-off 暖格 | 其中命中=0 | 判定 |
|---|---|---|---|---|---|
| core-dense | 40 | **40** | 40 | **40** | ✅ |
| core-hybrid | 40 | **40** | 40 | **40** | ✅ |

### 2.3 接受率（上游主张的直接仪器）

| target | drafter | PC-on | PC-off | on/off | 冷臂 | 暖臂 | warm/cold |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | dflash | 0.8367 | 0.8395 | 0.9966 | 0.8385 | 0.8367 | 0.9978 |
| Qwen3-4B | eagle3 | 0.4096 | 0.4124 | 0.9931 | 0.4110 | 0.4096 | 0.9964 |
| Qwen3-4B | ngram | 0.5945 | 0.5947 | 0.9996 | 0.5945 | 0.5945 | 1.0000 |
| Qwen3.5-4B | dflash | 0.7115 | 0.7153 | 0.9947 | 0.7106 | 0.7115 | 1.0013 |
| Qwen3.5-4B | eagle3 | 0.4504 | 0.4532 | 0.9939 | 0.4491 | 0.4504 | 1.0030 |
| Qwen3.5-4B | ngram | 0.5609 | 0.5275 | 1.0632 | 0.5609 | 0.5609 | 1.0000 |

⇒ 全部落在 **0.994–1.003**（唯一例外是 hybrid ngram 的 **+6.3%**，方向与"塌陷"相反）。

### 2.4 同实例冷/暖的贪心输出分歧（正确性）

| 臂 | 分歧数/60（dense / hybrid） | 分歧起点：最早 / 中位 |
|---|---|---|
| none（**对照**） | 6 / 11 | 0.65 / 0.77 ｜ 0.26 / 0.62 |
| ngram | 6 / 6 | 0.64 / 0.77 ｜ 0.65 / 0.95 |
| eagle3 | 6 / 11 | 0.65 / 0.77 ｜ 0.26 / 0.65 |
| dflash | 4 / 13 | 0.77 / 0.95 ｜ 0.26 / 0.65 |

### 2.5 K 敏感性（Qwen3-4B + ngram）

| K | PC-on | PC-off | on/off |
|---|---|---|---|
| 1 | 59.74 | 58.29 | 1.0249 |
| 3 | 62.51 | 62.64 | 0.9979 |
| 5 | 67.39 | 66.51 | 1.0133 |
| 7 | 69.21 | 67.72 | 1.0220 |

PC 效应与 K 无关（无单调趋势）；但 **K 本身敏感**：59.1 → 68.5 tok/s（+16%）。

## 3. 判据对照（Go / No-Go 逐条）

| 预登记判据 | 结果 | 判定 |
|---|---|---|
| **P2'** 暖臂命中>0、冷臂≈0、PC off 恒 0 | 40/40 + 40/40 | ✅ **Go**（有效期闸门通过） |
| **P3'** 某 drafter \|on/off−1\| > θ **且** 5 seed 方向一致 | 无一格超 θ | ⛔ **No-Go**（按 S4 归档） |
| **P3' 补充** seed 级 t 检验 | hybrid eagle3/ngram CI 排除 1；对照含 1 | ⚠ **存在 2–4% 真效应**，但未过预先写死的闸门 |
| **P1'** PC on/off 贪心输出逐字相同；`none` 臂也分歧 ⇒ 装置作废 | **`none` 臂分歧**（dense 6/60、hybrid 11/60） | 🔴 **INSTRUMENT_INVALID**（不是"发现 bug"） |
| **S3** 噪声底 σ/μ > 15% | 1.52% / 1.39% | ✅ 未触发 |
| **S5** 引擎启动失败 ≠ 效应为零 | 主网格 16/16 OK；Falcon 投机臂 2/2 失败并单独记录 | ✅ 已分开 |

**Falcon-H1-3B-Instruct**：`none` 两臂正常（429 s / 354 s），`ngram` 两臂**均在 33 s 内启动失败**，
`kv_cache_interface.py:879 assert self.page_size_padded >= page_size`，**pc=0 与 pc=1 均崩、未经任何 patch**
⇒ 该家族无法提供 PC×spec 对照数据。

## 4. 结论 + 后续

1. **上游机制（"缓存恢复的 token 不经 target ⇒ drafter 读未初始化 KV ⇒ 接受率塌陷"）在本配置不复现**：
   接受率在 PC on/off 与冷/暖之间全在 ±0.7% 内。
2. **工程后果**：按预先写死的 θ=5%，8 格无一显著 ⇒ **归档**；但 seed 级检验显示 hybrid 上有 2–4% 的真效应。
3. **P1' 仪器本身被自己的对照臂证伪**：投机解码是**无损**的，输出一致性原理上测不到"drafter 读到未初始化 KV"；
   且 `none` 对照臂自己就分歧 ⇒ 该不变量在本栈上是**装置级**噪声。
4. **两条口径必须分开**：跨配置分歧基线（10.0% / 18.3%，装置级不可复现）与同实例冷/暖基线是**两个量**。
   混用会把真缺陷判成噪声、也会把噪声判成缺陷。

**后续（不在本轮范围）**：
- **MTP 路径**——该模型**有**内建 MTP head（`mtp_num_hidden_layers=1`、15 个 `mtp.*` 张量、
  引擎已注册对应架构），本轮**未测**是遗漏而非不可测；后续轮已在该路径上复现出缺陷（见 C1 仓 v9 线）。
- 跨引擎检验；`mamba_cache_mode` 对照（需绕过 `enable_prefix_caching` 的强制覆盖）。

---

## 重跑

```bash
cd /root/autodl-tmp
python instrument/c1_verify_models.py          # 先做完整性：尺寸检查会漏掉内容损坏
python instrument/c1r_workload_probe.py        # 确认工作负载对 drafter 有分辨空间
for G in core-dense core-hybrid k-sens falcon; do
  python instrument/c1r_driver.py --grid $G --seeds 0 1 2 3 4 \
    --n-per-set 6 --gen 48 --warm-repeats 3 --concurrencies 1 8 --gpu-util 0.5
done
python instrument/c1r_analysis.py   --grid core-dense
python instrument/c1r_analysis.py   --grid core-hybrid
python instrument/c1r_acceptance.py --grid core-dense
python instrument/c1r_acceptance.py --grid core-hybrid
```
