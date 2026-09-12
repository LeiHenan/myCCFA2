# 服务器初始化手册（GPU 机器）

> 🧊 **2026-09-12 更新：本手册原针对的 8×RTX 4090 服务器（驱动 550.67 / CUDA 12.4）已判定不可用于主线实验。**
> 原因：vLLM ≥0.28（主线 `#6` 的 `DFlash2DraftModel` 只从该版本起存在）是 **CUDA 13** 构建，要求驱动 **≥ 580**；该机落在 CUDA 12.x 驱动区间且无 root，无法升级。
> ⇒ **上机之前先读 [`MACHINE_REQUIREMENTS.md`](MACHINE_REQUIREMENTS.md)**（选机规格），并跑 `bash docs/acceptance_check.sh` 过验收。
> 本文以下内容（环境初始化、卡分配、smoke test、结果回写纪律）对新机器**同样适用**；"8 卡"字样按实际卡数理解即可。

**目标**：10 分钟内确认环境可用，并把**今天要跑的两个实验**并行起来。
**前提**：一台 **驱动 ≥580（CUDA 13）**、单卡 ≥24 GB（Ada/Hopper，FP8 KV 需 sm_89+）的服务器，SSH 可登录。原 8×4090 的显存与互联特性仍可参考（24 GB/卡、**PCIe、无 NVLink**）。

---

## 阅读顺序（**先读这一段**）

| 顺序 | 文件 | 作用 |
|---|---|---|
| **0** | [`docs/MACHINE_REQUIREMENTS.md`](MACHINE_REQUIREMENTS.md) + `bash docs/acceptance_check.sh` | **上机之前**：这台机器是否合格（两条硬门槛：驱动 ≥580、`DFlash2DraftModel` 被引擎注册）。不合格就别开跑 |
| **1** | **本文** `docs/SERVER_BOOTSTRAP.md` | 环境验收 → 依赖 → 模型 → **smoke test** → 卡分配 → 结果回写纪律 |
| **2** | [`probes/p06-frontier/T0-runbook.md`](../probes/p06-frontier/T0-runbook.md) | **T0**：深度旋钮验证（半天）。**结局 B ⇒ `#6` 当天降级** |
| **3** | [`probes/p06-frontier/README.md`](../probes/p06-frontier/README.md) | **T1** 操作与判据概览；脚本 `run_t1.sh`（18 格 sweep）+ `analyze_t1.py`（ridge 判定） |
| **4** | [`probes/p01-e1-uncommitted-kv/instrument/README.md`](../probes/p01-e1-uncommitted-kv/instrument/README.md) | **E1**：`e1_patch.py --apply` → 压测 → `analyze.py` → `--revert` |
| **5** | 判据（**不要只看命令**）：[`notes/prereg/p06-frontier.md`](../notes/prereg/p06-frontier.md)、[`notes/prereg/p01-e1-uncommitted-kv.md`](../notes/prereg/p01-e1-uncommitted-kv.md) | 门槛 / 杀判据 / 扩展条款 / H1b / D3 支线 |
| 6 | 全局背景：[`EXECUTION_PLAN.md`](../EXECUTION_PLAN.md)（唯一权威）、[`docs/EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md) | 决策记录、W1–W3 规划、命令级细节 |

> ⚠️ **不要**从 `docs/plan.md` / `docs/kimi_plan.md` / `docs/GPT.md` / `docs/V41.md` 取执行指令 —— 它们是**冻结的历史记录**，文首有勘误横幅，执行依据一律以 `EXECUTION_PLAN.md` 为准。

---

## 0. 本次的卡分配（8 卡只用 3 张，其余留空）

| 卡 | 用途 | 备注 |
|---|---|---|
| **GPU0** | **E1**（`#1` 探针，1 天） | 用 `ngram` 或 EAGLE-3 即可；仪器化补丁已在最新 main 上验过锚点 |
| **GPU1** | **T0 → T1**（`#6` 主线，1–1.5 天） | `Qwen3-4B` + `Qwen3-4B-speculator.dflash2`；128k 格需 `--kv-cache-dtype fp8` |
| GPU2–7 | 留空 | 后续 T2 第二臂 / 全网格 / `#7` 探针 |

**必须 pin 卡**：`export CUDA_VISIBLE_DEVICES=0`（或 1）。**同一张卡上不要并发两个 vLLM 实例**（显存与性能互相污染）。

## 1. 环境验收（10 分钟）

```bash
# 1.1 硬件 / 驱动
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
#   期望：8 行 "NVIDIA GeForce RTX 4090", 24564 MiB

# 1.2 计算能力（Ada = (8,9)；FP8 KV 需要 ≥ Ada）
python3 -c "import torch; print(torch.cuda.get_device_capability())"

# 1.3 磁盘（权重 + 编译缓存建议 ≥100 GB 余量）
df -h "$HOME"

# 1.4 互连拓扑（记录用；本阶段 bs=1 单卡不需要，但 T2 若用 TP=2 需要它）
nvidia-smi topo -m | head -12
```

## 2. 环境选择（**venv 优先，不必用 conda**）

```bash
python3 -V
# ≥3.10 → 直接 venv（推荐）
python3 -m venv .venv && . .venv/bin/activate && pip install -U pip
# <3.10 或没有 pip 权限 → 用 conda **只提供 Python**，之后同样用 pip 装包
#   conda create -n myccfa python=3.12 -y && conda activate myccfa && pip install -U pip
```

**为什么不必用 conda、且对 vLLM 有风险**：

| 事实 | 含义 |
|---|---|
| vLLM 的 wheel **自带 CUDA 运行时**（以 `nvidia-*` pip 包形式） | 用 venv 最干净 |
| 若同一 env 里又用 conda 装了 `cudatoolkit` / `pytorch-cuda` | 两套 CUDA 库混在一起 ⇒ 常见 `undefined symbol` / 版本不匹配 |
| `conda install vllm` 版本滞后、ABI 风险更高 | **只用 pip 装 vLLM** |

**一条硬规则**：**一个 env 只用一种包管理器**。可以用 conda 创建环境（拿 Python），但 `vLLM / torch / nvidia-*` 一律用 `pip` 装，**不要**在同一个 env 里 conda 装 torch 或 cuda 工具链。

## 2b. 装 vLLM：先试 wheel，**不要一上来就编译源码**

```bash
pip install vllm safetensors pandas
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
# 关键自检：DFlash/DSpark 是否已注册（这是 #6 的家族前提）
python -c "from vllm.model_executor.models.registry import ModelRegistry as R; \
print([k for k in R.get_supported_archs() if 'DFlash' in k or 'DSpark' in k])"
```

- **输出含 `DFlash2DraftModel`（或 `DFlashDraftModel`）⇒ 直接用 wheel，别编译**（快、稳）。
- **输出为空 ⇒ 才**装我们核对过锚点的那个 commit（**会从源码编译，20–60 分钟，且需要匹配的 nvcc**）：
  ```bash
  pip install "vllm @ git+https://github.com/vllm-project/vllm.git@9a35c081e80a94828af6f611525102bb70e3c67f"
  ```
- 无论哪条路径，**把 `vllm.__version__`（或 commit）写进 `results/<probe>/<日期>/summary.md`**（判据要求记录 pin）。

> 若该机不能访问 GitHub：从本机 `scp -r` 整个目录过去即可（脚本都在 `probes/` 下）。

## 3. 模型下载

```bash
export HF_HOME=$PWD/upstream/hf_cache
hf download Qwen/Qwen3-4B --local-dir upstream/models/Qwen3-4B
hf download mgoin/Qwen3-4B-speculator.dflash2 --local-dir upstream/drafters/src
```

## 4. 三层 smoke test（每层单独起、测完 kill）

```bash
export CUDA_VISIBLE_DEVICES=0
# ① 无投机
vllm serve Qwen/Qwen3-4B --max-model-len 32768 --port 8000
# ② ngram 投机（验证投机路径通）
vllm serve Qwen/Qwen3-4B --speculative-config '{"method":"ngram","num_speculative_tokens":5}' --max-model-len 32768 --port 8000
# ③ 真 drafter（验证 DFlash2 能加载）
vllm serve Qwen/Qwen3-4B \
  --speculative-config '{"model":"'"$PWD"'/upstream/drafters/src","num_speculative_tokens":7}' \
  --max-model-len 32768 --port 8000
```
**③ 成功 ⇒ `#6` 的家族前提在真机上确认**；失败则记录报错（这会直接影响 `#6` 的去留）。

## 5. 生成深度变体并跑 T0/T1

```bash
export CUDA_VISIBLE_DEVICES=1
python probes/p06-frontier/make_depth_variants.py \
  --src upstream/drafters/src --out upstream/drafters --depths 1 2 3 4 5
# 按 T0-runbook.md 跑旋钮验证（结局 A 才继续）

DRY=1 OUT=results/p06-frontier/$(date +%F) bash probes/p06-frontier/run_t1.sh   # 先看命令
KV_DTYPE=fp8 OUT=results/p06-frontier/$(date +%F) bash probes/p06-frontier/run_t1.sh
python probes/p06-frontier/analyze_t1.py --dir results/p06-frontier/$(date +%F) --out results/p06-frontier/$(date +%F)
```

## 6. E1（另开一个 shell，卡 0）

```bash
export CUDA_VISIBLE_DEVICES=0
python probes/p01-e1-uncommitted-kv/instrument/e1_patch.py --apply
E1_LOG=/tmp/e1_metrics.jsonl vllm serve Qwen/Qwen3-4B \
  --speculative-config '{"model":"'"$PWD"'/upstream/drafters/src","num_speculative_tokens":5}' \
  --max-model-len 32768 --port 8100
# 另一个 shell：按 probes/p01-e1-uncommitted-kv/README.md 的网格压测
python probes/p01-e1-uncommitted-kv/instrument/analyze.py \
  --log /tmp/e1_metrics.jsonl --out results/p01-e1-uncommitted-kv/$(date +%F)
python probes/p01-e1-uncommitted-kv/instrument/e1_patch.py --revert   # 收工必做
```

## 7. 结果回写（**结论必须入库**）

- `results/<probe>/<date>/summary.md` **现在会被 git 跟踪**（`.gitignore` 已放开）；`*.csv`/`*.json` 等原始数据仍不入库。
- 每个实验结束：填 `summary.md` 的四段（量/区间/基线、数字、判据对照、结论）→ 在 `notes/decision_log.md` 追加一条（**死亡也要写**）→ `git add -A && git commit && git push`。

## 8. 注意事项（8×4090 特有）

1. **无 NVLink**：本阶段 `bs{1}` 单卡即可，**不要为了"用满 8 卡"上 TP** —— TP 会改变性能基线，且 PCIe 拓扑会让数字不可比。
2. **FP8 KV**：Ada 支持（`(8,9)`）；24 GB 卡跑 **128k 格必须** `--kv-cache-dtype fp8`（或把 ctx 收到 32k）。**所有对照条件必须同 dtype**。
3. **同卡不并发**：两个 vLLM 实例同卡会互相抢显存并污染吞吐数字。
4. **记录 GPU 型号**：论文里必须写明（4090 ≠ H100，接受率与延迟都不同）。
5. **长上下文慢**：128k 格单次可能数分钟；`run_t1.sh` 的 `REPS` 可先设 1 做通跑，再设 3 出正式数。
