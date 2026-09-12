# results/ — 实验产物

**`summary.md` 会入库**（结论必须跟着仓库走）；`*.csv`/`*.json`/`run.log` 等**原始数据不入库**（`.gitignore` 已按此设置，必要时用 `git add -f` 单独加小体积表）。

## 目录约定

```
results/<probe>/<YYYY-MM-DD>/
├── summary.md      # 必填：结论 + 原始数字表 + 与 prereg 判据的逐条对照
├── data.csv|json   # 原始数字（不要只留图）
├── figures/        # 图
├── run.log
└── instrument/     # 仪器化补丁 / 生成脚本（能重跑）
```

## summary.md 必填四段

1. **量 / 区间 / 基线**（对应 `notes/prereg/<probe>.md`）
2. **数字**（每格 p50 与 p95，不只是均值）
3. **判据对照**：Go / No-Go 逐条打勾
4. **结论 + 后续**：写回 `notes/decision_log.md` 的那一条

## 规矩

- 结论与原始数字**同时**保留：只有图没有数据表的产物视为无效。
- 任何"没测到"的结果也要留：写明测了哪些格子、为什么不完整（这比"结果很好"更有用）。
- 生成脚本必须与产物同目录，保证半年后可重跑。
