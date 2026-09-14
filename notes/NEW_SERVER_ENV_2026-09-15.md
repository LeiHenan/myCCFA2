# 新机环境画像（用户提供，2026-09-15）

**接入**：`ssh -p 26924 root@connect.weste.seetacloud.com`（AutoDL，rebuild 后的那台）
**用途**：用户决定「**换机器**」——不在 schoolserver 上装 g++ / 升驱动，改用这台做验证。

---

## 1. 硬件与软件（实测，非推断）

| 项 | 值 | 来源 |
|---|---|---|
| 容器 | `autodl-container-b9j8ek84q3-c14058ce` | `hostname` |
| GPU | **1× NVIDIA RTX 6000D**，**85,651 MiB (83.7 GiB)** | `nvidia-smi --query-gpu` |
| **compute capability** | **12.0（sm120 / Blackwell）** | 同上 |
| 驱动 | **595.71.05** | 同上 |
| 拓扑 | 单卡（`GPU0 X`），无 P2P/NVLink 议题 | `nvidia-smi topo -m` |
| CPU / RAM | **208 核 / 1007 GB** | `nproc` / `free -g` |
| 磁盘 | `/root/autodl-tmp` **200 G（1% 已用）**；`/` overlay **30 G（21 G 可用）** | `df -h` |
| venv | `/root/ccfa_venv`：Python **3.12.3**，**torch 2.13.0+cu130**，`cuda.is_available()=True` | 实测 |
| **vLLM** | **0.29.0**（可 import，且**引擎可跑**，见 §3） | `import vllm` |
| SGLang | **未安装**（`ModuleNotFoundError`） | 实测 |
| 环境脚本 | `/root/ccfa_env.sh`（见 §2） | — |
| 模型 | **重建时被清空**；`/root/autodl-tmp/models` 原为空 | 实测 |

⇒ **这台正是三张地图（`KV_CACHE_GAP_MAP` / `INFERENCE_ACCEL_GAP_MAP` / `spec-decode-gap-map`）
原始校准的硬件档**：sm120、96 GB 级单卡、driver ≥580、vLLM 可用。
`pipeline/tools/prescreen_map.py` 里被我新增的 `ada-8x4090` 画像**只适用于 schoolserver**；
在**这台**机上原始假设重新成立 ⇒ 复述任何"够不着"结论时必须写明**是哪台机**。

## 2. `/root/ccfa_env.sh`（既有，未改动）

```
export LC_ALL=C.UTF-8        # 容器未生成 en_US.UTF-8 ⇒ 不设会让 python import readline 段错误
export LANG=C.UTF-8
export CCFA=/root/myCCFA
export MODELS=/root/autodl-tmp/models
export TARGET=/root/autodl-tmp/models/Qwen3-4B
export DRAFTER_SRC=/root/autodl-tmp/models/dflash2
export DRAFTER_ROOT=/root/autodl-tmp/dflash-variants
export KV_DTYPE=bfloat16
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_DISABLE_XET=1
export OUT_BASE=/root/ccfa_results/$(date +%F)   # 必须在仓库外
. /root/ccfa_venv/bin/activate
```

## 3. 引擎正确性闸门（**本项目最贵的教训：能 import ≠ 算得对**）

**背景**：schoolserver 上 vLLM 0.21.0 引擎初始化成功、正常吐 token，**但输出是垃圾**
（`'!RAD_optional!!!...'`）。所以"vLLM 0.29.0 可 import"**不构成**可用性证据。

**闸门 v1**（`probes/p34-engine-gate/vllm_correctness.py`）在 Qwen3-4B 上下载完成后运行：

| 检查 | 结果 |
|---|---|
| 模型下载 | 7.6 G，3 个 safetensors 分片，`DOWNLOAD_DONE` |
| vLLM 引擎 | 权重加载 1.65 s；可用 KV 缓存 **60.98 GiB**；`enforce_eager` 下 init 64.27 s |
| **输出可读性** | **连贯英文**：`' was driven by the need to perform complex calculations more efficiently. The first mechanical calculator was the abacus, which dates back to ancient times…'` ⇒ **不是 schoolserver 那种垃圾输出** |
| 与 HF 贪心逐 token 比对 | **第 14 个 token 起分歧**（HF=3881 vs vLLM=22148） |

**闸门 v2 结果（`gate2.py`，三 prompt × 128 token）—— 通过**：

| 判据 | 结果 |
|---|---|
| **T1 配置内自洽性** | 三个 prompt 同配置连跑两次 **全部 IDENTICAL** ⇒ **引擎确定，可用作仪器** |
| **T2 可读性** | 完全连贯英文 |
| **T3 分歧性质** | **3 个 prompt 中 2 个与 HF eager 前 128 token 逐位全同**；1 个在第 **31** token 分歧，分歧后文本**仍连贯**（vLLM 讲 difference engine、HF 讲 analytical engine，两种史实均正确）⇒ 分歧位置**随 prompt 变化** ⇒ 数值路径差异，非故障 |
| **T4 margin** | 分歧步 HF 的 top1−top2 margin = **0.1250**，全序列中位 **2.125** ⇒ **小 17 倍**，与已发表 margin 判据（2605.30218 / 2608.13756）自洽 |

**⇒ 结论：这台机 = 正确的引擎 + 不同的数值路径。可用于实验。**
且它比 schoolserver 上的 monkey-patch 平台强：**真实部署引擎**、84 GB 显存（可上更大模型/更长上下文）、
**sm120（地图原始硬件档）**、以及可控的 kernel 路径轴（`enforce_eager` / cudagraph / `torch.compile`，
正是 `vLLM PR #55944` 记录的那条轴）。

**判读（关键）**：分歧**不等于**故障。vLLM 与 HF 用不同 kernel / 不同 reduction 次序，
**这正是 `vLLM PR #55944` 文档正文写明的现象**：*"Eager execution, `torch.compile` without CUDA graphs,
and the default path with CUDA graphs use different kernels. Their outputs differ by a small amount for
the same input."*
⇒ **对"引擎是否是可用仪器"这个问题，v1 不足以定论**；v2（`gate2.py`）把它拆成四条可判据：
T1 配置内自洽性（同配置连跑两次必须逐 token 相同）／T2 128 token 可读性／
T3 分歧位置随 prompt 变化且分歧后仍连贯（⇒ 数值路径差异，非故障）／
T4 分歧步的 top1−top2 margin 是否偏小（⇒ 与已发表 margin 判据 2605.30218 / 2608.13756 自洽）。

**纪律**：**T1 不通过（引擎非确定）⇒ 这台机不可用于任何测量**，先修引擎。
**T1–T3 通过 ⇒ 这台机是"正确的引擎 + 不同的数值路径"**，而那个差异本身**就是我当前候选的研究对象**。

## 4. 投机解码路径（S-B 曾需要，现 S-B 已被零卡闸门杀掉，但基础设施已验证一半）

| 检查 | 结果 |
|---|---|
| vLLM 0.29.0 原生支持 DFlash/DSpark | **是**（源码实测）：`vllm/config/speculative.py` 有 `DFlashModelTypes = Literal["dflash"]`、`DSparkModelTypes = Literal["dspark"]`；`vllm/model_executor/models/` 下有 `qwen3_dflash.py`、`qwen3_dflash2.py`、`qwen3_dspark.py`、`laguna_dflash.py` |
| **OFF 臂（无投机）** | **通过**：`OFF_OK`，引擎初始化 3.77 s，`GPU KV cache size: 474,288 tokens`（max_model_len 4608） |
| **ON 臂（dflash2, K=3）** | **尚未验证** —— 投机器 `mgoin/Qwen3-4B-speculator.dflash2` **下载未完成**（7 文件中最后 1 个=权重仍在传），引擎因此初始化失败。**这是"下载未完"，不是"配置不可行"。** |
| 历史佐证 | `/root/ccfa_results/2026-09-12/S1_4k/bs1/d3_g3_ctx4096_r1.bench.json` 的启动日志证明**上一个目标已在本机成功跑过**同配置（`Resolved architecture: DFlash2DraftModel`、`FLASH_ATTN v2`、`V2 Model Runner`），并含真实数字：`spec_decode_acceptance_rate 24.19%`、`acceptance_length 1.726`、**`per_position_acceptance_rates [0.484, 0.182, 0.060]`**、`mean_tpot_ms 10.07`、`mean_itl_ms 17.27` |

**纪律提醒（本机已验证的一条）**：`enforce_eager=True` 时 `speculative_config` 与 `num_spec_tokens`
必须**读回断言**（`llm.llm_engine.vllm_config.speculative_config`），不能只断言构造参数——
本工作区已两次因"两组配置实际相同"报废实验。`probes/p36-spec-gate/spec_gate.py` 里已内置该回读断言。

---

## 5. 对目标文本「schoolserver 已有环境」的实测否定（2026-09-15）

目标文本称：*"`/home/user/envs/tools` 是**已装好并验证过**的 conda env —— vLLM 0.29.0 + torch 2.13.0+cu130 +
transformers 5.17.0 + triton 3.7.1 + flashinfer 0.6.18 + xgrammar 0.2.6；其 `verify.log` 记录 `available True`"*。

**逐条实测（`ssh schoolserver`）**：

| 断言 | 实测 |
|---|---|
| `/home/user/envs/tools` 里有 python | **无**。该目录是 conda env 骨架，`bin/` 下只有 `c_rehash/captoinfo/clear/infocmp/infotocap/ncurses6-config/ncursesw6-config/openssl` ⇒ `/home/user/envs/tools/bin/python: No such file or directory` |
| 有 `verify.log` | **空/不存在** |
| 有 vLLM 0.29.0 | **无** |

**其余 env 全查（`ml` / `model` / `quant` / `torch` / `venv_v21`）**：

| env | torch | vLLM |
|---|---|---|
| `/home/user/.conda/envs/ml` | 2.9.0+cu128, `avail=True` | **none** |
| `/home/user/.conda/envs/model` | 2.9.0+cu128, `avail=True` | **none** |
| `/home/user/.conda/envs/quant` | 2.9.0+cu128, `avail=True` | **none** |
| `/opt/anaconda3/envs/torch` | 2.9.0+cu128, `avail=True` | **none** |
| `/home/user/venv_v21` | **import 失败** | none |

⇒ **schoolserver 上不存在可用的 vLLM**（此前结论再次成立，且现在连目标文本点名的那个 env 也排除了）。
叠加 §9.9（AutoDL 机一直有可用 vLLM 0.29.0）⇒ **本目标内唯一可用的引擎在新机（AutoDL）上。**
**且 schoolserver 没有 C++ 编译器**（`cc1plus` 缺失、`sudo` 需密码、`/opt/anaconda3` 不可写）⇒ **无法从内部补齐**。

---

## 6. 投机路径验证的**最终状态**（诚实记录，含我自己的 harness 缺陷）

| 步骤 | 结果 |
|---|---|
| 投机器 `mgoin/Qwen3-4B-speculator.dflash2` 下载 | **完成并通过完整性校验**：`model.safetensors` **2,820,554,168 B**，`sha256 = 459f75b6da6a70b7d5630408196e2203798af0ca33db23bb1d628d8c9212e805`，与该仓 `SHA256SUMS` **逐字符相同** |
| （过程）HF snapshot 下载**停滞**在 1.2 GB | 改为从 `.cache/**/*.incomplete` 播种 + `curl -C -` 续传解决；`.incomplete` 的文件名前缀正是期望哈希，佐证播种起点正确 |
| OFF 臂（无投机） | **通过**（`OFF_OK`，引擎 init 3.77 s，KV 474,288 tokens） |
| **ON 臂（dflash2, K=3）** | **未验证** —— 引擎进程**卡住**：占 **81,291 MiB** 显存、**GPU 利用率 0%**、父进程 `do_wait` 40 分钟无进展 ⇒ 已终止并释放显存（0 MiB） |

**根因（我的 harness 缺陷，不是引擎缺陷）**：`probes/p36-spec-gate/spec_gate.py` 在**同一个进程里顺序加载两个引擎**
（先 OFF 再 ON），每个都设 `gpu_memory_utilization=0.90`。第一个引擎的显存未被可靠回收，第二个 init 因此陷入停滞。
**正确做法**：两个臂必须在**独立进程**里跑（各自 `LLM(...)` 一次）。
**这条不影响本目标**：唯一需要投机路径的候选 **S-B 已被其零卡可分离性闸门杀掉**（`notes/SB_SEPARABILITY_2026-09-15.md`）。
**复活时先做这件事**：把 `spec_gate.py` 拆成 `--arm off|on` 的两个进程调用，再判 ON 臂。

**已确立、与 ON 臂无关的部分**：vLLM 0.29.0 在本机**原生支持 DFlash/DSpark**
（`vllm/config/speculative.py` 的 `DFlashModelTypes`/`DSparkModelTypes`；`models/qwen3_dflash2.py` 等）；
且**上一个目标已在本机成功跑通**同配置（`/root/ccfa_results/2026-09-12/S1_4k/bs1/d3_g3_ctx4096_r1.bench.json` 的启动日志
`Resolved architecture: DFlash2DraftModel`，含 `acceptance_rate 24.19%`、`per_position_acceptance_rates [0.484, 0.182, 0.060]`）。
⇒ **"本机不能跑投机解码"是错的**；只是**我这次的 ON 臂 harness 写错了**。
