# p29 —— C1（sm89 上 block-FP8 静默回退）**在本机不可测**：如实记为阻塞

**日期**：2026-09-15 ｜ **目标**：`goal-e5030ca8` ｜ **成本**：0 GPU·h

## 为什么记这一条

硬件差异轴地图的候选 **C1** 目前是本方向**最有希望**的一个：门控已被**零卡实证**
（子代理直接调用编译扩展：`cutlass_scaled_mm_supports_block_fp8(89) = False`、`(90) = True`；
DeepGEMM 自述 *"Currently, only Hopper and Blackwell GPUs are supported."*），
而**量级"无人测过"**（子代理明确拒绝估数，说"那正是实验本身"）。

## 但它在本机无法测 —— 两个前提都不满足

| 需要的前提 | 本机状态 |
|---|---|
| **① 一个 block-FP8（block-scaled FP8）量化模型** | ❌ 磁盘上只有 `Qwen3-4B`（bf16）与 `Qwen3-4B-speculator.dflash2`。block-FP8 权重需要**带 block scale 的 checkpoint**（如 DeepSeek 系 FP8 权重 / llm-compressor 产物），**磁盘仅剩 ~67 G，且需要重新下载与格式适配** |
| **② vLLM 的 kernel 选择器本身**（`can_implement` 派发链 → 回退到 `TritonFp8BlockScaledMMKernel`） | ❌ **vLLM 在本机不可用**：0.26/0.29 的扩展是 cu13 构建、driver 550 拒绝执行；0.21 能跑但**输出是垃圾**（见 `notes/SCHOOLSERVER_ENV_2026-09-15.md` §6） |

**没有 ②，"静默回退"这件事本身就不存在**——因为回退逻辑是 vLLM 的代码；
**没有 ①，就没有 block-FP8 的 GEMM 可测**。两者都不是我能在此环境绕开的。

## 能做的替代（以及为什么它们不够）

| 替代 | 为什么不够 |
|---|---|
| 手写 Triton block-FP8 GEMM 并与 torch `_scaled_mm`（per-tensor FP8）比 | 只能说明"两种 FP8 格式在 sm89 上的相对速度"，**测不到 C1 的断言本身**（那个断言是"vLLM 在 sm89 上**静默**改用了一个**从未针对该架构做过性能选型**的 kernel"）。而"从没做过性能选型"是**关于 vLLM 开发过程的断言**，本地实验无法证实或证伪 |
| 等 `apt-get install -y g++` 或驱动 ≥580 | **正确路径**：若驱动 ≥580，原装 vLLM 0.29.0 可用 ⇒ ①仍需下载 block-FP8 权重，但 ②满足 |
| 换机器（H800/H100） | C1 **恰恰是关于 sm89 的**，换到 Hopper 就没了差异轴 |

## 结论与复活条件

**C1 = ⚠ 阻塞（不是被杀）。** 复活条件明确：**驱动 ≥580（或 g++ 解锁 vLLM 0.21 的 flashinfer 路径且输出正确性被验证）+ 下载一个 block-FP8 checkpoint**。
在那之前，本方向的候选池实际收缩到"**能在 HF/torch 平台上独立复现**"的那一类——这也正是本轮转向 p28/p30 那条线的原因。
