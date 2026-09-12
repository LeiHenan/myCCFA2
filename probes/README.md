# probes/ — 实验 runbook

命名规则见 [`docs/GLOSSARY.md`](../docs/GLOSSARY.md)：**`pNN-*` = 候选 `#NN` 的实验；`aux-cN-*` = 辅助探针 C1–C9**。

| 目录 | 候选 | 时间盒 | 一句话判据 | 定位 |
|---|---|---|---|---|
| [`p01-e1-uncommitted-kv/`](p01-e1-uncommitted-kv/README.md) | #1 | **1 天** | 全部 HBM 可行点上 p95 比值 <5% ⇒ 死 | 判定探针（唯一有效否决权） |
| [`p03-grammar-scan/`](p03-grammar-scan/README.md) | #3 | **1 周** | class-4 @batch≥32 >0.85× 且瓶颈非结构性 ⇒ 归档 | 判定探针（低概率彩票） |
| [`p04-hybrid-gate/`](p04-hybrid-gate/README.md) | #4 | **2 小时** | gap 仍落在"checkpoint 边界/提交语义" ⇒ 归档 | 复核门（不是实验） |
| [`p06-frontier/`](p06-frontier/README.md) | **#6 主线** | **3–4 天判定 → 2 周全网格** | 先 T0/T1/T2 三级判定：**无配置反转 / 共线 / 增量 <8% ⇒ 杀** | 主线决胜实验 |
| [`p07-adapter-dispersion/`](p07-adapter-dispersion/README.md) | #7 | **≤1 周** | 离散度 <5 个百分点 ⇒ 归档 | 备线 B 的入场判定 |
| [`aux-c1-prefill-padding/`](aux-c1-prefill-padding/README.md) | — | 1 周 | 记 `padded/real` | 填缝探针（辅助表 C1） |
| [`aux-c2-flashinfer-radix/`](aux-c2-flashinfer-radix/README.md) | — | 1 周 | 静默关闭 radix 的代价 | 填缝探针（辅助表 C2） |

> `#12`（MoE EP 对冲）的实验 runbook 在时间盒启动时创建，命名为 `p12-ep-imbalance/`。
>
> **`p06` 的工具链已就绪**：`T0-runbook.md`（旋钮验证）、`make_depth_variants.py`（深度变体）、`run_t1.sh`（18 格 sweep）、`analyze_t1.py`（ridge 判定）。

## 统一约定

1. **预登记先行**：开跑前写 `notes/prereg/<与实验目录同名>.md`（量 / 区间 / 基线 / 杀判据 / 截止），之后改判据必须留痕。
   **例外**：`p04-hybrid-gate` 是"门"而非探针（不采集数据），判据在执行前即冻结在 `probes/p04-hybrid-gate/README.md` §3 与 `EXECUTION_PLAN.md` §6，故**无** `notes/prereg/` 条目（已执行完毕，结论见 `notes/gap_hybrid_state.md`）。
2. **产物落地**：`results/<实验目录名>/<YYYY-MM-DD>/`，至少含 `summary.md`、原始数字表、`run.log`、生成脚本。
3. **结论回写**：无论生死，`notes/decision_log.md` 追加一条；死亡也要写。
4. **基线纪律**：对**部署中实际在跑的配置**比较，不对理想化基线；<8% 即止。
5. **家族/区间标注**：每个数字都要能回答"哪个模型家族、哪个 (bs, ctx) 区间、对哪个基线"。
