# Frontier — drafter 算力分配前沿扫描（**#6 主线决胜实验**）

**时间盒**：2 周 ｜ **硬件**：1–2 卡 ｜ **预登记**：`notes/prereg/p06-frontier.md`

**赢则成文，输则一周半永久关方向。两种结果都值。**

## 1. 家族事实（先确认实验是否成立）

| drafter 家族 | 层数 | 深度旋钮 | 备注 |
|---|---|---|---|
| **EAGLE-3** | **1** | **不存在** | 只能作基线；其树宽/树深旋钮已被 AdaEDL / SpecDec++ / Pacer / SVIP / DDD / ECHO / Graft 占满 |
| **DFlash** | 5 | 存在 | 4.4× 损失锚点来源（`#54691`，~185k ctx） |
| **DSpark** | 生产骨干 **3**（MoE 层，sliding window 128）；"a **2-layer** DSpark outperforms the **5-layer** DFlash baseline" | 存在 | 生产已 ship 置信度调度（见 §4 基线纪律） |
| MTP head | 单头 | 粒度粗 | 可作交叉验证 |

**前置检查（W1）**：确认能跑起 ≥1 个**多层并行 drafter**。若只有 EAGLE-3 可用 ⇒ **#6 直接降级**（深度旋钮不存在），不得用 EAGLE-3 的树深冒充 drafter 深度。
**取数路径**：`vllm-project/speculators`（DSpark/DFlash 上游实现）+ HF 权重（如 `mgoin/GLM-5.2-speculator.dspark-block16`）。没有它，W2 的租卡没有意义。

## 2. 定位

`Speculative Compute Allocation` — 每步决定给这一轮投机投多少算力。`depth × width × draft length × verify budget` 是**同一预算的不同形态**，论文卖的是"分配"而不是"调参"。

## 3. 网格与指标

| 维度 | 取值 |
|---|---|
| drafter depth | {1, 2, 3, 4, 5, 6}（取该家族可训练/可取用的范围）；**T1 只需 {1,3,5}** |
| **draft length γ** | **{1, 3, 5, 7}** —— **正交性检验必需**，至少在 `width=默认树` 子集上与 depth 同扫 |
| drafter width / 树形 | {窄链式, 默认树, 宽树} 三档 |
| batch | 1 / 8 / 32 |
| context | 4k / 32k / 128k / **185k** |
| 对照 | spec off（AR 基线）、最佳**固定**配置 |

指标：`tok/s`、`per-step draft 成本 (ms)`、`平均接受长度`、`相对最佳固定配置的端到端比`。每格 ≥3 次重复。

## 4. 基线纪律（否则会被判成 configuration tuning）

DSpark 生产路径已 ship 的**置信度调度 + STS 校准 + 离线 SPS 成本表 + ragged-verify（cap-accept / compact）**是**基线**，不是空白。结论里必须有一句："本机制与已 ship 的调度器差在哪。"

## 5. 判据

| 条件 | 动作 |
|---|---|
| **正交性（主假设）**：存在 (bs, ctx) 区域，最优 (depth, γ) 发生**非共线反转** | 主假设存活，进入 T2 |
| **共线**：`optimal depth ≈ f(optimal γ)`（两者可互相替代） | **杀或并入 `#7`/`#12`**（退化为另一个 adaptive draft length 实现） |
| 存在内点最优 **且** 配置反转 **且** 相对最佳固定配置 ≥8% **且** 三个必答项通过 | **Go**：进入机制设计（子网络切换 + 与已 ship 控制器的协同） |
| 收益只出现在 185k 角落 | **杀**（"只在角落赢"） |
| 无内点最优 / 无反转 | **杀** |
| 相对最佳固定配置 <8% | **杀** |

## 6. DEX 替换测试（**书面交付物，W1 完成**）

回答：**"把本机制读成『DEX（`2606.29223`）换成一个独立 drafter』，它还成立吗？"**

- 若"能" ⇒ delta 只落在模型替换上 ⇒ 改措辞或改方向。
- 必须写清三条区分轴：
  1. **独立 drafter 网络深度**（不是目标模型 early-exit / self-speculation：DEL / SpecBound / DSSD / LayerSkip）；
  2. **调度器级、跨请求**的分配（不是 DEX 的 per-commit-position）；
  3. **与验证预算耦合**（depth × width × K × verify 是同一预算）。

模板：

```markdown
### DEX 替换测试
- DEX 做什么：目标模型层深轴上的并行深度探索 + commit–collapse（lossless）。
- 若把"探索器"换成独立 drafter：差异剩 ______。
- 差异是否构成新机制：是/否，理由 ______。
- 结论：继续 / 改措辞为 ______ / 改方向。
```

## 7. 产物

```
results/p06-frontier/<YYYY-MM-DD>/
├── summary.md          # 判据三条 + 基线差异句 + DEX 替换测试
├── frontier.csv        # (depth, width, bs, ctx) → tok/s, draft_ms, accept_len
├── figures/            # 等高线 / 反转点图
└── run.log
```

## 8. 本目录工具（2026-09-12 增补）

| 文件 | 用途 | 自测 |
|---|---|---|
| [`T0-runbook.md`](T0-runbook.md) | 深度旋钮验证的操作手册（改 config → 裁权重 → 加载 → 三结局判定 + 失败形态） | —（文档） |
| [`make_depth_variants.py`](make_depth_variants.py) | 从已发布 speculator 权重生成 `d1/ d3/ d5/` 深度变体（只改 config，可选 `--prune-weights`） | `--selftest` ✔ |
| [`run_t1.sh`](run_t1.sh) | 18 格 sweep（`DRY=1` 只打印命令；**`DATASET=custom DATASET_DIR=...` 必填**，见 `make_prompts.py`；长上下文格为 32k，因 Qwen3-4B 上限 40960） | dry-run ✔（117 行命令） |
| [`analyze_t1.py`](analyze_t1.py) | 聚合结果 → `t1.csv`/`ridge.csv`/`summary.md`，判定正交 / 斜 ridge(H1b) / 不可判定 | `--selftest` ✔（三种情形） |

> 三者**均未在真机验证过**（无 GPU 环境）；`vllm serve` / `vllm bench serve` 的 flag 以你 pin 的版本 `--help` 为准。

