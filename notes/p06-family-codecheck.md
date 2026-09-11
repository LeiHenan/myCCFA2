# #6 代码层家族检查（无卡）

**执行**：2026-09-11 ｜ **结论**：**代码层通过，但定位必须收紧**
**证据**：`archive/prior_art/srcfull/sglang-main/python/sglang/srt/speculative/*`（本地源码树）、`archive/dspark_deep_read.md`

## 一、多层 drafter 存在（不是障碍）

- `speculative/multi_layer_eagle_utils.py` + `multi_layer_eagle_worker_v2.py` + `multi_layer_eagle_draft_extend_cuda_graph_runner.py`：SGLang 已有**多层 EAGLE** drafter 路径（含 widened draft-extend）。
- DFlash：`models/dflash.py:561` 读 `draft_config.num_hidden_layers`，并把 `draft_num_layers=num_layers` 传入 `dflash_utils.py` 的显存估算 ⇒ **层数是配置项**。
- DSpark：生产骨干 3 个 MoE 层，论文消融显示 **2 层胜 5 层**（离线静态）。

## 二、但"运行时自适应"已占一半（关键）

`speculative/adaptive_spec_params.py` 文件头逐字：

> **"Adjusts speculative_num_steps at runtime based on observed acceptance lengths."**

- 带 per-batch-size 档位：batch 1 → `candidate_steps [1,3,5,7]`｜8 → `[0,1,3]`｜32 → `[0,1]`｜64 → `[0]`；另有上下行迟滞（`up_hysteresis` / `down_hysteresis`）与 `ceiling_coeff`。
- `speculative/adaptive_runtime_state.py` 逐字：**"A complete set of runtime resources bound to a specific speculative decoding configuration… Switching adaptive steps swaps the entire state atomically."** —— 即**多套 CUDA-graph / attention state 的原子切换已经实现**。这正是我原先给 #6 记的"多形状 graph 池"机制贡献 ⇒ **不能再把它当卖点**。

## 三、对 #6 的三条硬约束（已写入 `notes/prereg/p06-frontier.md`）

1. **不得自称"投机算力分配"泛称**：draft 长度（`speculative_num_steps` / `speculative_num_draft_tokens`）与 verify 预算的自适应**已 ship**；draft **树**深度已被 Graft / DDD / ECHO 占。
2. **唯一可主张的轴**：drafter **网络自身**的层数 / 宽度在运行时的弹性缩放（DSpark 证明该轴是活的，但只做离线静态选择）。
3. **必须回答**：与 `adaptive_spec_params` 的差异是什么（它换"步数"，我们换"网络深度"；两者耦合时谁更优）。此条作为 frontier 的**必答项**，写进 `summary.md`。

## 四、结论

代码层**通过**（多层 drafter 存在、层数可配置、可截断），因此 W2 的 frontier **值得租卡**。
但能否成文仍由 frontier 的三条判据决定：**内点最优 ∧ (bs,ctx) 配置反转 ∧ ≥8%**。
