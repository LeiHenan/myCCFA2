# candidates/ — 候选方向档案

**用途**：每个候选科研方向一个子目录，装着它在**选题阶段**（S0–S5，即"要不要做"）的全部产物。
执行期的代码与结果仍按仓库既有约定放 `probes/` 与 `results/`。

```bash
python pipeline/new_candidate.py --slug <slug> --title "<中文标题>"   # 建档案 + 生成骨架
python pipeline/check.py --dir candidates/<slug> --through S4        # 声称 S0–S4 完成，逐条核
```

| 文件 | 阶段 | 一句话 |
|---|---|---|
| `00_intake.md` | S0 | 约束量化（算力/时间/资产/不做清单/降级路线） |
| `10_pains.md` | S1 | 候选痛点（现象/谁在疼/量级/证据/反证） |
| `20_screen.md` | S2 | 伪问题七形态 + 自检十问 + 存活清单 + 淘汰台账 |
| `30_occupancy.md` | S3 | 最近邻（读正文）/ 引擎已 ship 旋钮 / 差异轴 / 结构性理由 |
| `40_measurability.md` | S4 | 观测量恒等式 / 仪器冒烟 / **oracle 上界** / 噪声预算 / 可采纳前置条件 |
| `50_prereg.md` | S5 | 预登记（并提升为 `notes/prereg/<slug>.md`） |
| `60_decision.md` | S6 | T0/T1/T2 三级判定与与预登记的差异 |
| `70_mechanism.md` | S7 | 效应分解 + 替代解释排除 + 适用范围 |
| `80_close.md` | S8 | 结论 / 适用范围与反证 / 产物清单 / 后续 |

**规矩**：
1. 骨架内容来自 [`pipeline/gates.json`](../pipeline/gates.json)，**不要手改小节名与清单文本**（校验器按文本比对）。
2. 判据只改 `gates.json`，并在 `notes/decision_log.md` 记一行。
3. 被淘汰的候选**不要删目录**——在 `20_screen.md` 的淘汰台账里写明理由，负面结论同样是资产（`#6` 的结题报告就是这么留下来的）。
4. 标注 `STATUS: killed|paused|active` 在该目录 `00_intake.md` 的第一行，便于一眼看出存亡。
