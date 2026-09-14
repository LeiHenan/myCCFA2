# p35 原始数据（从 AutoDL 机取回，关机器前保存）

| 文件 | 来源 | 说明 |
|---|---|---|
| `L512.jsonl` | `/root/ccfa_results/p35/L512.jsonl` | p35 修正轴探针的 **360 格原始记录**（30 上下文 × 3 结构 × 4 目标质量）。每行含 `drop_last`（末-query 丢弃质量，**真正的匹配轴**）、`drop_global`（旧轴，用于量化缺陷）、`tv`、`flip`、`gap`、`mass_ok` |
| `d3_g3_ctx4096_r1.bench.json` | `/root/ccfa_results/2026-09-12/S1_4k/bs1/` | **上一个目标**在同一台机跑通的 `vLLM 0.29.0 + Qwen3-4B + dflash(K=3)` bench：`spec_decode_acceptance_rate 24.19%`、`acceptance_length 1.726`、**`per_position_acceptance_rates [0.484, 0.182, 0.060]`**、`mean_tpot_ms 10.07`、`mean_itl_ms 17.27` |

**为什么保存**：`L512.jsonl` 是 `SUMMARY.md` 里"候选死亡"判定的**原始证据**（`analyze.py` 的全部比值/CI 都由它算出）；
`d3_g3_*.bench.json` 是"这台机器能跑投机解码"的**历史佐证**（用于反驳"引擎不可用"的误判）。
两台机器：`schoolserver`（8×4090 sm89，**vLLM 不可用**）与 AutoDL（RTX 6000D sm120，**vLLM 0.29.0 可用**）。
