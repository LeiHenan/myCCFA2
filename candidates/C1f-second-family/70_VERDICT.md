# C-1f：第二 hybrid 家族**跑不起来** —— 崩溃已被上游 OPEN PR 认领

**日期**：2026-09-15 ｜ **平台**：RTX 4080 SUPER 32GB，vLLM 0.29.0
**目的**：给 C-1b/d/e 的 hybrid 结论找**第二个 hybrid 家族**做独立复现（我自己的闸门要求）。
**结果**：**Falcon-H1-3B + prefix caching + ngram 投机 = 引擎启动即崩**，
且**该崩溃已被上游 PR #47635 修复（OPEN 未合并）** ⇒ **不入我的账，但它解释了一件事**。

---

## 1. 崩溃与精确根因

```
[Falcon-H1-3B] enable_prefix_caching=True + speculative_config(ngram, K=3)
  INFO [interface.py:918] Setting attention block size to 2080 tokens ...
  INFO [interface.py:942] Padding mamba page size by 0.24% ...
  ERROR AssertionError: assert self.page_size_padded >= page_size
        at vllm/v1/kv_cache_interface.py:879  (MambaSpec.page_size_bytes)
        via vllm/v1/attention/backends/utils.py:276 resolve_kv_cache_layout
```

**对照（同一进程、同一参数，只改一个开关）**：

| 配置 | 结果 |
|---|---|
| Falcon-H1-3B，prefix caching，**无投机** | ✅ 启动成功 |
| Falcon-H1-3B，**无** prefix caching，无投机 | ✅ 启动成功 |
| Falcon-H1-3B，prefix caching + **ngram K=3** | ❌ **`page_size_padded >= page_size` 断言失败** |

⇒ **必须同时有"投机"与"prefix caching"，且只在 hybrid 模型上**才触发。

**机制（读源码 + 日志推出，未逐行断点验证）**：
`_align_hybrid_block_size` 用 `get_mamba_state_shape_from_config()` 算出 mamba page size，
把 attention page 撑到 ≥ 它，并把差值记为 `mamba_page_size_padded`（Falcon 上只有 **0.24%** 余量）。
而 `MambaSpec.page_size_bytes` 把 **`num_speculative_blocks`**（= `num_speculative_tokens`）
计入每层状态大小。两者不一致 ⇒ 投机把真实 page 撑过 padding 余量 ⇒ 断言崩。

## 2. **这不是我的发现** —— 上游已在修

| PR | 标题 | 状态（2026-09-15 亲验） |
|---|---|---|
| **#47635** | *[Bugfix] Pass **num_spec** to mamba2_state_shape in **FalconH1/GraniteMoeHybrid/Zamba2** (spec decode crashes at init)* | **OPEN / 未合并** |
| #41233 | [Bugfix][Hybrid][NemotronH] Fix mamba_cache_mode=all + speculative decoding crash | **MERGED** |
| #45207 | [Bugfix] Pad Mamba page size instead of scaling block_size | **MERGED** |

**#47635 的标题就是我看到的现象本身**（"Pass num_spec to mamba2_state_shape … crashes at init"）。
⇒ 按 workspace 判据（"有 MERGED PR 或 ≥2 个并行开放提案 ⇒ 被占"），**这条**：
- 没有 MERGED PR（#47635 仍 OPEN）⇒ 严格按判据**未完全被占**；
- 但它的**立论完全落在别人的 PR 上**，我最多只能提供"独立复现 + 定量触发条件"，
  **这不足以支撑一篇论文**。

⇒ **我不把它作为课题。** 记为「已有 OPEN 修复在办」。

## 3. 它对**已有结论**的意义（重要）

1. **Falcon-H1 这条路暂时封死** ⇒ 我的"第二个 hybrid 家族"复现**没做成**。
   C-1b/d/e 的 hybrid 结论**仍然只有 Qwen3.5-4B 一个家族**。
2. **解释了为什么 Qwen3.5 没崩**：Qwen3.5-4B 的 padding 余量是 **1.49%**，Falcon 只有 **0.24%**。
   余量够 ⇒ 断言不触发。**但这意味着 Qwen3.5 上投机状态可能"塞进了别人的 padding 里"**——
   即 **page size 记账可能与实际不符**。
   ⚠️ **这会让 C-1d/C-1e 的 hybrid 成本数字可疑**：如果 Qwen3.5 的 mamba 页被**低估**，
   那么内存记账与 kernel 行为都可能偏离，"hybrid 每步更贵"就可能是**记账/布局错配的症状**，
   而不是"线性注意力架构固有更贵"。
   **这是我目前最大的未排除混淆，也是下一步的第一优先。**
3. **可测的判别**：在 Qwen3.5 上用 `mamba_cache_mode ∈ {align, all, none}` + 投机开关做四格，
   看成本差异是否只在 `align`（即依赖 padding 的那条路径）上出现。

## 4. 下一步（更新后的优先级）

| 优先 | 事项 | 为什么 |
|---|---|---|
| **1** | **验证 Qwen3.5 上 `mamba_page_size_padded` 是否真的 ≥ 投机后的实际页** | 若不等，C-1d/e 的成本差可能是布局错配的症状，**结论要改写** |
| 2 | 找一个**不触发该断言**的第二 hybrid 家族（Zamba2-2.7B / Nemotron-H / GraniteMoeHybrid 也可能中招，需先试） | 我的闸门要求 ≥2 家族 |
| 3 | 拆步计时（草稿 vs 验证） | C-1d/e 的机制归属 |

## 5. 诚实边界

- 崩溃是**我实测**的（两种配置对照、可复现）；**上游 PR 的存在是检索到的**，
  我**没有**读 #47635 的 diff（只读了标题与状态）。
- §1 的机制是**读源码 + 日志推断**，**未**用断点或打印验证 `page_size` 与 `page_size_padded` 的实际数值。
- §3 第 2 点（"C-1d/e 数字可疑"）是**推断**，**未取证**；我把它列为第一优先正是因为它会推翻已有结论。
- Falcon-H1-3B 已完整下载（8.5 GB），脚本与复现命令都已就位，修好后可立即重跑。

## 6. 复现

```bash
# 崩溃（应报 AssertionError: page_size_padded >= page_size）
cd /root/autodl-tmp
/root/miniconda3/bin/python -c "
import os; os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
from vllm import LLM
LLM(model='/root/autodl-tmp/models/Falcon-H1-3B', gpu_memory_utilization=0.42,
    max_model_len=4096, enforce_eager=True, dtype='bfloat16', language_model_only=True,
    enable_prefix_caching=True,
    speculative_config={'method':'ngram','num_speculative_tokens':3,
                        'prompt_lookup_max':4,'prompt_lookup_min':2})"
# 去掉 speculative_config 或 enable_prefix_caching 任一 ⇒ 正常启动
```
