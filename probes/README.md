# probes/ — 实验 runbook

每个探针一个目录，目录内 `README.md` 是**可执行 runbook**（测什么、怎么测、何时杀）。代码与产物在开跑时落到对应目录 / `results/`。

| 探针 | 对应方案 | 时间盒 | 一句话判据 | 定位 |
|---|---|---|---|---|
| [`e1_uncommitted_kv/`](e1_uncommitted_kv/README.md) | #1 投机 KV 分级 | **1 天** | 全部 HBM 可行点上 p95 比值 <5% ⇒ 死 | 判定探针（唯一有效否决权） |
| [`c3_constrained_scan/`](c3_constrained_scan/README.md) | #3 约束解码三交集 | **1 周** | class-4 @batch≥32 >0.85× 且瓶颈非结构性 ⇒ 归档 | 判定探针（低概率彩票） |
| [`frontier_drafter/`](frontier_drafter/README.md) | **#6 主线** | **2 周** | 无内点最优 / 无 (bs,ctx) 配置反转 / <8% ⇒ 杀 | 主线决胜实验 |
| [`gate_hybrid_state/`](gate_hybrid_state/README.md) | #4 收窄版 | **2 小时** | gap 仍落在"checkpoint 边界 / 提交语义" ⇒ 归档 | 复核门（不是实验） |

填缝探针（C1 prefill CUDA-graph padding、C2 FlashInfer radix 被静默关闭）见 `EXECUTION_PLAN.md` §8.3，runbook 在主线等待期补。

## 统一约定

1. **预登记先行**：开跑前在 `notes/prereg/<probe>.md` 写清 `量 / 区间 / 基线 / 杀判据 / 截止时间`，之后再改判据必须留痕。
2. **产物落地**：`results/<probe>/<YYYY-MM-DD>/`，至少包含 `summary.md`（结论 + 原始数字表）、`run.log`、以及生成脚本。
3. **结论回写**：无论生死，`notes/decision_log.md` 追加一条；死亡也要写（避免半年后重开）。
4. **基线纪律**：一律对**部署中实际在跑的配置**比较，不对理想化基线比较；<8% 即止。
5. **家族/区间标注**：每个数字必须能回答"在哪个模型家族、哪个 (bs, ctx) 区间、对哪个基线"（`docs/plan.md` §5.6）。
