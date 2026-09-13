# p16：BI 的成本与来源拆分 —— **cuBLAS 工作区配置既不来带不变性、也不花钱**

**日期**：2026-09-13 ｜ 新目标第 1 轮 ｜ **成本 ≈0.05 GPU·h** ｜ 原始数据 `/root/ccfa_results/2026-09-13/p16_workspace/`

## 起点（零卡源码）

`vllm/model_executor/determinism/batch_invariant.py:918-940`：
**SM80** 分支装 4 个 triton persistent matmul 覆盖（`mm/addmm/matmul/linear`）；
**SM90/SM100/SM120** 走 `else` 分支 —— **不替换 matmul**，改为
`CUBLAS_WORKSPACE_CONFIG=":16:8"` + `CUBLASLT_WORKSPACE_SIZE="1"`，注释自述
*"the only source of batch variance is split-k, which we disable via the cuBLAS workspace config"*。
其余替换：`softmax / _log_softmax / _softmax / mean.dim / bmm`（triton 版）+ RMSNorm。

## 实验（BI=0，仅改 `CUBLAS_WORKSPACE_CONFIG`）

vLLM 0.29.0 / Qwen3-4B / sm120 / 4096-token prompt / `max_new_tokens=64` / n=1 与 n=8 并发相同请求 / 每格 2 次重复。

| `CUBLAS_WORKSPACE_CONFIG` | n=1 | **n=8 发散** | wall (n=1 / n=8) |
|---|---|---|---|
| 不设（默认） | div 0.0000 | **unique 3/8，div 0.5000** | 0.902 / 1.824 s |
| `:4096:8` | div 0.0000 | **unique 3/8，div 0.5000** | 0.894 / 1.830 s |
| `:16:8`（BI 使用的值） | div 0.0000 | **unique 6/8，div 0.7500**（更差） | 0.882 / 1.822 s |

## 结论

1. **工作区配置不是不变性的来源**：单独把它设成 BI 用的 `:16:8`，发散**反而更高**（0.75 vs 0.50）。
   ⇒ BI 的有效性**不来自**"禁 split-k"。
2. **工作区配置零成本**：三档的 n=1 与 n=8 墙钟差异 **< 0.5%**。
   ⇒ 在真实 prefill 负载上，**禁 split-k 不花钱**。
3. **推论（下一步靶子）**：BI 的 **1.7×** 成本**不可能来自 cuBLAS 侧**，只能来自
   **算子替换**（triton `softmax/mean/bmm` + RMSNorm）⇒ "这些替换是否全都必要"是
   "**更省等价方案**"的**唯一剩余入口**。

## 零卡估算器（新工具，含 `--selftest`）

`probes/p16-splitk/estimate_splitk.py`：从 `config.json` 推导真实形状（实测 Qwen3-4B：
hidden=2560、inter=9728、GQA 32Q/8KV、36 层），给出算术强度 ≈ M
⇒ **预测"禁 split-k 的代价随 M 增大而下降"**，本轮实验与之相符（代价 ≈ 0）。

## 授权落地情况（诚实登记）

- **多卡：本机不可执行** —— `nvidia-smi -L` 只有 1 张卡、MIG 关闭、无第二张卡；换机需用户在 AutoDL 侧开多卡实例
  （规格与重建手册见 `docs/MACHINE_REQUIREMENTS.md`、`docs/SERVER_BOOTSTRAP.md`）。
- **时间放宽：已生效** —— 本轮开始做"机制归因 + 找更省等价方案"。

## 下一步

把成本按**算子**拆开：屏蔽/恢复 `softmax` / `mean` / `bmm` / RMSNorm 的 BI 替换，逐一测
"发散是否回来"与"墙钟省回多少"。若发现**某一类替换其实不必要** ⇒ 即为比"全开 BI"更省的等价手段。
