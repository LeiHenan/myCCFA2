# 新机器规格书（选机 / 验收 / 降级）

**为什么有这份文件**：原 8×RTX 4090 服务器（驱动 **550.67**）于 2026-09-12 被判定**不能跑主线实验**。
卡住它的**不是显存，是 CUDA 代际**：主线 `#6` 需要的 `DFlash2DraftModel` 检查点族只被 vLLM **≥0.28** 支持，而 vLLM ≥0.27 全部是 **CUDA 13** 构建 ⇒ 要求驱动 **≥580**，本机 550.67 属于 CUDA 12.x 区间（上限 12.4），且拿不到 root 无法升级。

本文件提供四件事：**(1)** 硬性规格与理由；**(2)** 显存/磁盘预算（按实测数字）；**(3)** 命令级验收脚本；**(4)** 只能拿到旧驱动时的降级方案。

> 配套：`docs/SERVER_BOOTSTRAP.md` 是**上机后**的环境初始化与卡分配手册；本文件是**上机前**的选机与验收标准。二者是"买什么"与"怎么用"的关系。

---

## 0. 结论速览

| 项 | 最低要求 | 推荐 | 理由 |
|---|---|---|---|
| **NVIDIA 驱动** | **≥ 580** | R580 或更新（590/595/610/615 皆可） | CUDA 13.x 要求驱动 ≥580；这是**唯一不可妥协**的一条（§1） |
| GPU | **1× ≥24 GB，sm_89+** | **2× ≥24 GB** 或 1× ≥40 GB | 单卡可跑 T0/T1（128k 用 FP8 KV = 9 GiB）；第二张用于并行跑 E1（原计划 GPU0=E1、GPU1=T0/T1） |
| GPU 架构 | **Ada(8.9) / Hopper(9.0)+** | 4090 / L40S / RTX 6000 Ada / H100 | **FP8 KV 需要 ≥sm_89**；A100（sm_80）只能退回 FP16 KV，会违反 T1 网格的 `KV_DTYPE=fp8` 设定 |
| CUDA | 13.0+（**由 wheel 自带，不需要装 toolkit，也不需要 nvcc**） | — | vLLM wheel 内嵌 CUDA 运行时；本机在 cu128 情形下已验证"无编译器也能跑" |
| 数据盘 | ≥ 100 GB 可用 | **≥ 200 GB 可用** | §2 预算表；原机只剩 90 GB，一轮下载就吃掉 29 GB |
| CPU / 内存 | 8 vCPU / 32 GB | 16 vCPU / 64 GB | 128k 上下文 + 两个引擎并行 + safetensors 加载 |
| Python | 3.10–3.12 | **3.11** | vLLM 支持区间；`python -m venv` 足够，**不需要 conda** |
| root / sudo | **不需要**（前提：镜像自带 ≥580 驱动） | — | 只有**装驱动**才需要 root；原机正是卡在这里 |
| 实例类型 | 持久 VM / Pod + `tmux` | 按需计费，**禁用空闲自动关机** | T1 是 18 格连续 sweep，serverless / 会被杀实例不适用 |
| 网络 | PyPI + GitHub + HuggingFace 可达 | 国内：TUNA PyPI + `hf-mirror.com` | §5 有实测注意事项 |

---

## 1. 为什么"驱动 ≥580"是硬门槛（决定成败的一条）

完整事实链（每条都实测或出自官方文档）：

1. 主线 `#6` 的 drafter 检查点是 `mgoin/Qwen3-4B-speculator.dflash2`，架构 **`DFlash2DraftModel`**（配置文件已核对：`block_size=8`、`sliding_window_non_causal=true`、`speculators_version=0.7.0.dev0`）。
2. vLLM 中 `vllm/model_executor/models/qwen3_dflash2.py` **只从 v0.28.0（2026-08-26）起存在**；v0.26.0 / v0.27.0 / v0.27.1 **连模块文件都没有**（HTTP 404），`registry.py` 里也无 `DFlash2DraftModel` 注册项 ⇒ **补一行注册表救不回来**。
3. vLLM 0.27.1 / 0.28.0 / 0.29.0 的 `requires_dist` 全部钉 **`torch==2.13.0`**。实测在干净 venv 装 `vllm==0.29.0` 会拉入 `torch 2.13.0+cu130`、`nvidia_cudnn_cu13`、`cuda_toolkit-13.0.3.0`，其编译扩展需要 **`libcudart.so.13`**。
4. **`torch 2.13.0` 没有 cu128 变体**：PyTorch 官方索引里 cu128 最高只到 `torch 2.11.0`，2.13.0 只有 **cu129 / cu130**。这就是"把 torch 降级到 2.11.0+cu128 后 `import vllm` 报 `libcudart.so.13` 缺失"的根因。
5. NVIDIA 官方驱动对应表（[CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)）：

   | CUDA Toolkit | 对应驱动分支 | 次版本兼容的最低驱动 |
   |---|---|---|
   | 13.x | R580（13.0）→ R615（13.4） | **≥ 580** |
   | 12.x | — | ≥ 525，且 < 580 |
   | 11.x | — | ≥ 450，且 < 525 |

   本机 **550.67** 落在 12.x 区间 ⇒ **任何 CUDA 13 构建都无法加载**（实测报错：`The NVIDIA driver on your system is too old (found version 12040)`）。
6. 替补引擎同样如此：**SGLang 0.5.19** 钉 `torch==2.13.0` + `flashinfer_python[cu13]` ⇒ 同一条死路。两条独立路径都被驱动卡住，这是"换机器"而非"换引擎"的原因。

> **给管理员/厂商的一句话**：我们需要一台 **驱动 ≥580（CUDA 13.0）** 的 GPU 机器；卡型 ≥24 GB、Ada/Hopper 架构即可，不需要 NVLink，不需要 root，不需要 CUDA toolkit。

---

## 2. 显存与磁盘预算（按实测数字，不是估算）

Qwen3-4B 实测配置：36 层 / 8 KV heads / head_dim 128 ⇒ **144 KiB/token**。

### 2.1 显存

| 组成 | 128k 上下文占用 |
|---|---|
| 目标模型 BF16 权重 | ~8 GiB |
| drafter（dflash2，2.7 GB 文件） | ~2.7 GiB |
| KV cache **FP16** | **18.0 GiB** |
| KV cache **FP8** | **9.0 GiB** |
| 激活 + 工作区（bs=1） | ~2–3 GiB |
| **合计（FP8 KV）** | **~22 GiB ⇒ 24 GB 卡可跑** |
| **合计（FP16 KV）** | **~31 GiB ⇒ 需 ≥40 GB 卡** |

⇒ T1 的 18 格网格（含 128k）**在 24 GB 卡上可行**，前提是 `KV_DTYPE=fp8`（这一点 `run_t1.sh` 已默认）。**24 GB 是底线，不是将就**。

### 2.2 磁盘

| 项 | 大小 |
|---|---|
| venv（torch cu13 + nvidia 运行库 + vllm + flashinfer 等） | ~10–12 GB |
| `Qwen/Qwen3-4B` 权重 | 7.6 GB |
| dflash2 drafter | 2.7 GB |
| （可选）8B 目标 + drafter 稳健性臂 | ~17 GB |
| （可选）小目标 0.6B/1.7B 预热网格 | ~4 GB |
| 原始结果 / 引擎日志（18 格 × 128k × 多轮） | ~10–20 GB |
| pip / HF 缓存与重下余量 | ~15 GB |
| **合计** | **~70 GB ⇒ 建议可用空间 ≥200 GB** |

原机 7.3 TB 盘只剩 **90 GB（99% 满）**，每一步下载都在冒险——这是选机时容易被忽略、但会真实打断实验的一项。

---

## 3. 框架与权重清单（精确到版本）

### Tier A —— 推荐（prereg 一字不改）

| 角色 | 名称 |
|---|---|
| 引擎 | `vllm==0.29.0`（或 prereg 钉的 main commit `9a35c081e80a94828af6f611525102bb70e3c67f`） |
| 运行时 | `torch==2.13.0+cu130`（pip 自动解析）、CUDA 13.0 runtime（wheel 自带）、Python 3.11 |
| 目标模型 | `Qwen/Qwen3-4B` |
| drafter | `mgoin/Qwen3-4B-speculator.dflash2`（arch `DFlash2DraftModel`） |
| 基线② | vLLM 内置 `num_speculative_tokens_per_batch_size`（per-batch K 查表；0.26.0 起即存在） |
| 跨引擎旁证 | `sglang==0.5.19`（同为 CUDA 13，非必需） |
| 可选工具 | `speculators`（PyPI 0.8.0，声明 `torch>=2.9,<=2.13`）——用于构造深度变体与对照 |

### Tier B —— 只能拿到 CUDA 12.x 驱动时的降级（**需新增 prereg，见 D1 条款**）

| 角色 | 名称 |
|---|---|
| 引擎 | `vllm==0.26.0`（钉 `torch==2.11.0`，cu128 索引最高即 2.11.0） |
| 运行时 | `torch==2.11.0+cu128`、驱动 ≥525 即可 |
| drafter | `weifanjiang/qwen3-4b.speculators.dflash-ce01tv09-bs8-swa`（arch `DFlashDraftModel`，**5 层 / block 8**，target `Qwen/Qwen3-4B`，`speculators_version=0.7.0.dev98`） |

**Tier B 保留什么**：4B 目标、单卡 24 GB、128k + FP8 KV 的 18 格网格、深度旋钮（0.26.0 的 `qwen3_dflash.py` 同样按 `config.num_hidden_layers` 逐层构建）、基线②。
**Tier B 损失什么**：引擎版本不再是 main 钉点（需修正案 + 论文 threat to validity）；drafter 由 dflash2 变 dflash **v1**（无 token selector / conv / 非因果滑窗），ridge 结论严格说只对 v1 成立。

---

## 4. 验收清单（拿到机器先跑这个，10 分钟）

一键脚本：`bash docs/acceptance_check.sh`（与本文件同目录）。逐条手工版：

```bash
# 4.1 驱动 —— 唯一硬门槛
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv
#   通过：至少一行 driver_version >= 580，且该行 memory.used 很小（空闲卡）

# 4.2 Python
python3 -V                       # 期望 3.10–3.12（3.11 最佳）

# 4.3 环境（venv，不用 conda）
python3 -m venv ~/venv && . ~/venv/bin/activate && pip install -U pip

# 4.4 装引擎（Tier A）
pip install "vllm==0.29.0"

# 4.5 torch 与 CUDA 是否真的可用
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
#   期望：2.13.0+cu130 13.0 <卡名>   ← 若报 "driver too old"，回 4.1

# 4.6 决定性一条：DFlash2 是否被引擎注册
python -c "from vllm.model_executor.models.registry import ModelRegistry as M; a=M.get_supported_archs(); print('DFlash2DraftModel' in a, 'DFlashDraftModel' in a)"
#   期望：True True   ← 原机（550.67）就是在这一步失败的

# 4.7 真算子（排除"只能 import 不能算"）
python - <<'EOF'
import torch
a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
print('matmul ok:', round((a @ a).sum().item(), 2))
EOF

# 4.8 磁盘
df -h ~        # 可用 >= 100 GB（建议 >= 200 GB）

# 4.9 网络（国内机房必看）
python -c "import urllib.request as u;print(u.urlopen('https://pypi.org/simple/', timeout=10).status)"
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python -c "import huggingface_hub as h;print(h.model_info('Qwen/Qwen3-4B').sha[:8])"
```

**只有 4.1 与 4.6 同时通过，这台机器才算可用。**

---

## 5. 网络注意事项（原机实测踩过的坑）

- HuggingFace **直连不可达**（原机 curl 返回 000）；`HF_ENDPOINT=https://hf-mirror.com` 可用（200）。
- **必须设 `HF_HUB_DISABLE_XET=1`**：否则大文件走 Xet CAS 会失败并报
  `RuntimeError: Task error: File reconstruction error: CAS Client Error: HTTP status client error (401 Unauthorized), domain: https://cas-server.xethub.hf.co/...`。设了就正常下载（原机 Qwen3-4B 7.6 GB 与 drafter 2.7 GB 均以此完成）。
- PyPI：TUNA 镜像可用；GitHub 指纹可达（脚本下载与 `git push` 均正常）。

---

## 6. 选机对照表（可直接交给厂商 / 管理员）

| 渠道 | 该怎么提要求 |
|---|---|
| 任意云 | "要 **CUDA 13.0 / 驱动 580+** 的 GPU 实例，卡型 ≥24 GB 且为 Ada 或 Hopper（4090 / L40S / RTX 6000 Ada / H100），数据盘可用 ≥200 GB，按量计费、可长期保持 SSH 会话" |
| 国内（AutoDL / 恒源云 / 揽睿等） | 默认镜像多为 CUDA 12.x ⇒ **必须显式选 CUDA 13 基础镜像**；先开 1 小时按量实例跑 §4 验收，再决定包周 |
| 海外（RunPod / Vast.ai / Lambda 等） | 选 CUDA 13.0 模板；确认是**持久实例/Pod**而非 serverless；确认无空闲自动关机 |
| 学校 / 实验室机器 | 直接把 §1 的六条事实链 + §4 的验收命令发给管理员，并附一句"不需 root，只要镜像自带 ≥580 驱动" |

**成本预期**：主线到出判定（T0 → T1 → T2 → E1）约 **4–6 个工作日**，算上全网格与重跑余量，**建议按 3 周预算**（按需实例可随时释放）。

---

## 7. 若最终只能拿到 12.x 驱动

走 Tier B（§3），并**必须先补一份 prereg 修正案**（引擎降级不属于 `EXECUTION_PLAN.md` §12 的 D1 扩展条款范围："扩展优先家族、不自动加引擎"）。修正案需写明：引擎版本、权重族变更、ridge 结论的适用范围，以及一份 0.26 → main 的投机解码关键路径 diff 作为外推证据。

---

## 8. 原 8×4090 服务器的剩余用途

不作主线实验机，但仍可用作：
1. **Tier B 实验机**（若决定降级）；
2. 不依赖 CUDA 13 的**数据处理与分析**（`analyze.py` / `analyze_t1.py` 的 ridge 判定、结果汇总）；
3. 已下载资产的暂存（`~/myCCFA/upstream/models/Qwen3-4B`、`~/myCCFA/upstream/drafters/src`，共 10.3 GB）——若新机器网络受限，可先从这里转存。

---

**版本**：v1.0（2026-09-12）· 依据：vLLM 源码/tag 逐版本核对、PyPI 与 PyTorch 索引实测、NVIDIA 官方驱动表、原机三轮实测日志。
