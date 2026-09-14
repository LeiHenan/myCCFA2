# `schoolserver` 环境实测档案（2026-09-15）

**用途**：这台机器**装好了但开箱不可用**，根因不显然。本文件固化实测事实、修法、以及**已排除的路径**，避免重复踩坑。
**机器**：`ssh schoolserver`（免密已通）＝ `10.176.60.2:12900`，user `user`，容器 `37179fa47076`。

## 1. 硬件（实测）

| 项 | 实测值 |
|---|---|
| GPU | **8× NVIDIA GeForce RTX 4090，24 GB/卡，sm89（Ada Lovelace）** |
| 驱动 | **550.67 ⇒ nvidia-smi 自报 CUDA 12.4** |
| 互联 | `nvidia-smi topo -m`：**全部 SYS = PCIe，无 NVLink**；GPU 0–3 属 NUMA 0，GPU 4–7 属 NUMA 1 |
| CPU / RAM | 192 核 / **1007 GB** |
| 磁盘 | `/home/user` 7.3 T、**99% 已用、仅剩 ~79 G**；`/` overlay 1.8 T、剩 145 G |
| **共享情况** | **GPU 1 与 GPU 7 被他人占用 23 GB** ⇒ 我们只用 0/2/3/4/5/6 |
| 工具 | `tmux` 在 `/home/user/envs/tools/bin/tmux`；**无 `hf`/`huggingface-cli`/`aria2c`**；有 `setsid`/`git`/`curl` |

## 2. ⛔ 核心结论：**现有驱动跑不了任何现代推理引擎**

**每一个现代引擎的 wheel 都是 CUDA 13 构建**（实测依赖）：

| 版本 | 实测依赖 |
|---|---|
| vLLM 0.22.1 / 0.23.0 / 0.24.0 / 0.25.1 / 0.26.0 | `torch==2.11.0` + **`humming-kernels[cu13]`** |
| vLLM 0.27.1 / 0.28.0 / 0.29.0 | **`torch==2.13.0`**（cu130） |
| SGLang 0.5.19（最新） | `flashinfer_python[cu13]` + **`humming-kernels[cu13]`** + `torch==2.13.0` |

**决定性报错**（vLLM 0.26.0 实际跑起来时）：
```
RuntimeError: get_cuda_view_from_cpu_tensor, /workspace/csrc/libtorch_stable/cuda_view.cu:38,
cudaHostGetDevicePointer failed: CUDA driver version is insufficient for CUDA runtime version
```
⇒ 扩展能 `dlopen`（把 `libcudart.so.13` 加进 loader 路径即可），但**一执行 CUDA 调用就被 550 驱动拒绝**。

**⇒ 这台机器要么升驱动，要么只能用"从源码/旧版针对 CUDA ≤12.8 构建"的引擎。**
**`sudo` 需要密码** ⇒ 容器内**无法自行升级驱动**（实测 `sudo -n true` 失败）。

## 3. 实测可用的部分（重要）

- **torch 2.11.0+cu128 在 550 驱动上完全可用**（实测）：`is_available True`、`device_count 8`、
  `device0 NVIDIA GeForce RTX 4090 (8,9)`、GPU matmul 正常。
  ⇒ **CUDA 小版本兼容（12.8 runtime on 12.4 driver）成立**，被卡住的只是 **CUDA 13** 的那一层。
- 模型已下载：`/home/user/models/Qwen3-4B`（7.6 G）、`/home/user/models/Qwen3-4B-speculator.dflash2`（2.7 G）。
- 仓库 `/home/user/myCCFA` 是 git 仓库，origin 与本地同源。

## 4. 修法 / 路径

### 路 A（推荐，需宿主 root）：驱动升到 ≥580
一步解锁**原本就装好的 vLLM 0.29.0**（也正是本仓库全部地图/台账假定的版本）。这是唯一的"零妥协"解。

### 路 B（我正在试，不需 root）：vLLM ≤0.21.0 + cu128 的 torch
`vllm==0.21.0` 钉 `torch==2.11.0` 且**不依赖 `humming-kernels[cu13]`**；把 torch 家族强制换成 cu128 版即可能可用。
脚本：`/home/user/ccfa_logs/build_v21.sh`（在**全新 venv** `/home/user/venv_v21` 里做，不动已有环境）。
日志：`/home/user/ccfa_logs/v21.log`。
**待验证**：该 wheel 的编译扩展究竟链 cu12 还是 cu13（其打包方式与新版不同，`/vllm/` 下无 `.so`）。

### 路 C：从源码构建（针对 CUDA 12.8）
系统有 `/usr/local/cuda-12.2`；需 CUDA 12.8 toolkit，数小时编译 + ~20 G 磁盘，不保证成功。

### ❌ 已验证无效的做法（别再试）
- 只把 `nvidia/cu13/lib` 加进 `LD_LIBRARY_PATH`：**import 能过、CUDA 调用仍被拒**。
- 在已有 venv 里降级 torch 到 cu128 却保留 vLLM ≥0.22：**vLLM 自己的扩展仍是 cu13**，必失败。
- 装 `torchaudio==2.11.0`（默认 PyPI）而不指定 cu128 索引：拿到的是 **cu130 构建**，与 torch cu128 冲突并硬报错。
  必须 `--index-url https://download.pytorch.org/whl/cu128`。

## 5. 环境固化脚本

`/home/user/myCCFA/env.sh`（用 `source` 载入）：
```bash
export VENV=/home/user/myCCFA/.venv
export PATH=$VENV/bin:$PATH
export LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
export LC_ALL=C.UTF-8
```
（`LD_LIBRARY_PATH` 那一行是**必须**的：`libcudart.so.13` 只在 venv 的 `nvidia/cu13/lib` 里。）

---

## 6. 续测（2026-09-15 第 3 轮）：**逐条实测出的最终可用性矩阵**

前面第 4 节的路径 B 已试到底，结论要更新——**不是"vLLM 跑不起来"，而是"能跑但算错"**，这更难发现也更危险。

| 配置 | 结果 | 证据 |
|---|---|---|
| vLLM 0.29.0 + torch 2.13+cu130（原装） | ❌ `libcudart.so.13` 缺失 / `cudaHostGetDevicePointer failed: CUDA driver version is insufficient` | 第 2 节 |
| vLLM 0.26.0 + torch 2.11+cu128 + cu13 lib 路径 | ❌ 同上（vLLM 自带扩展是 cu13 构建） | 第 4 节 |
| **vLLM 0.21.0 + torch 2.11+cu128 + 整套 cu12 nvidia 运行时** | ⚠️ **引擎能起来、能生成，但输出是垃圾** | 见下 |
| └ 其中 `attention_backend=FLASH_ATTN`（默认） | ❌ `vllm-flash-attn ... CUDA driver version is insufficient`（该扩展仍是 cu13 构建） | `ccfa_logs/e2e.log` |
| └ 其中 `attention_backend=TRITON_ATTN` | ⚠️ 跑通（`[init] 10.8s`、Triton 编译 `kernel_unified_attention` 成功）**但输出 `'!RAD_optional!!!...'`** | 实测 |
| └ 其中 `attention_backend=FLEX_ATTENTION` | ⚠️ 同样跑通、**同样垃圾**（`'时候_wind_HI就是要ẽ不清楚odash...'`） | 实测 |
| **HF transformers 4.57.6 + torch 2.11+cu128（不经 vLLM）** | ✅ **输出正确** | 见下 |

**关键对照（区分"模型坏"与"引擎算错"）——同一批 prompt、同一张卡**：

| 路径 | "The capital of France is" 的输出 |
|---|---|
| HF transformers | `' Paris. The capital of Paris is...?'` ✅ |
| vLLM 0.21 TRITON_ATTN | `'!RAD_optional!!!...'` ❌ |
| vLLM 0.21 FLEX_ATTENTION | `'时候_wind_HI就是要ẽ不清楚odash...'` ❌ |

⇒ **模型完整（3 shard / 8.04 GB）、torch/cu128/GPU 全部正确；错的是 vLLM 这一层**。
两个**互相独立**的 attention 后端都错 ⇒ **不是 attention 后端的问题**，而在更底层（vLLM 0.21 的 `_C` 算子 / 该版本与 torch 2.11 的组合）。
已排除的候选原因：`transformers` 版本（vLLM 0.21 要 `>=4.56`，实测已是 4.57.6，**不是**根因）；flashinfer 采样器 JIT（已用 `VLLM_USE_FLASHINFER_SAMPLER=0` 绕开）。

### ✅ 可用的实验平台：HF transformers（**已验证**）
```
361-token 上下文 + 48 token 贪心解码，Qwen3-4B bf16，单张 4090：
  rep0 1.69 s (28.4 tok/s)   ← 含 warm-up
  rep1 1.24 s (38.6 tok/s)
  rep2 1.24 s (38.6 tok/s)
  同格 3 次输出逐位一致 = True     ⇒ 噪声地板 ≈ 0，适合做前后对照
```
**代价**：比 vLLM 慢一到两个数量级 ⇒ 实验必须按小负载设计。**收益**：确定性极好，且不依赖任何 cu13 扩展。

## 7. 要恢复"快引擎"，只需下面任一条（都需要 root，我做不了）

| 路径 | 一条命令 | 解锁 |
|---|---|---|
| **A（最省事）** | `apt-get install -y g++` | FlashInfer 运行时 JIT 可编译 ⇒ vLLM 0.21/0.26 的 flashinfer 路径可用（**仍需验证输出正确性**） |
| **B（最彻底）** | 驱动升到 **≥580** | 原本装好的 **vLLM 0.29.0 + cu130 直接可用**，即本仓库全部地图/台账假定的版本 |
| C | 授权从源码针对 CUDA 12.8 构建 vLLM | 数小时 + ~20 G，且不保证解决"算错" |

**⚠️ 一条必须记住的教训**：vLLM 0.21 那次是**先跑通、后出垃圾**——如果我只看"进程没报错、有 tokens/s"就宣布环境就绪，后面所有实验都会建立在错误数值上。
**"断言产物正确，而不只是断言它跑完了"** 与既有的"断言配置真的生效"是同一条纪律的延伸。
