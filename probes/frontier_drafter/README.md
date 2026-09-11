# Frontier — drafter 算力分配前沿扫描（**#6 主线决胜实验**）

**时间盒**：2 周 ｜ **硬件**：1–2 卡 ｜ **预登记**：`notes/prereg/frontier.md`

**赢则成文，输则一周半永久关方向。两种结果都值。**

## 1. 家族事实（先确认实验是否成立）

| drafter 家族 | 层数 | 深度旋钮 | 备注 |
|---|---|---|---|
| **EAGLE-3** | **1** | **不存在** | 只能作基线；其树宽/树深旋钮已被 AdaEDL / SpecDec++ / Pacer / SVIP / DDD / ECHO / Graft 占满 |
| **DFlash** | 5 | 存在 | 4.4× 损失锚点来源（`#54691`，~185k ctx） |
| **DSpark** | 生产骨干 **3**（MoE 层，sliding window 128）；"a **2-layer** DSpark outperforms the **5-layer** DFlash baseline" | 存在 | 生产已 ship 置信度调度（见 §4 基线纪律） |
| MTP head | 单头 | 粒度粗 | 可作交叉验证 |

**前置检查（W1）**：确认能跑起 ≥1 个**多层并行 drafter**。若只有 EAGLE-3 可用 ⇒ **#6 直接降级**（深度旋钮不存在），不得用 EAGLE-3 的树深冒充 drafter 深度。

## 2. 定位

`Speculative Compute Allocation` — 每步决定给这一轮投机投多少算力。`depth × width × draft length × verify budget` 是**同一预算的不同形态**，论文卖的是"分配"而不是"调参"。

## 3. 网格与指标

| 维度 | 取值 |
|---|---|
| drafter depth | {1, 2, 3, 4, 5, 6}（取该家族可训练/可取用的范围） |
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
| 存在**内点最优**（不是"越深越好"或"越浅越好"）**且** 在 (bs, ctx) 平面上出现**配置反转** **且** 相对最佳固定配置 **≥8%** | **Go**：进入机制设计（多形状 graph 池 / 子网络切换摊销策略） |
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
results/frontier_drafter/<YYYY-MM-DD>/
├── summary.md          # 判据三条 + 基线差异句 + DEX 替换测试
├── frontier.csv        # (depth, width, bs, ctx) → tok/s, draft_ms, accept_len
├── figures/            # 等高线 / 反转点图
└── run.log
```
