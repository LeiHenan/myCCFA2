# T0 — 深度旋钮验证（结果）

**判定**：A. 旋钮可操作（进 T1）

- 平均接受长度 1.303–2.829（相对跨度 80.5%，展示阈值 5%）
- tok/s 101.5–180.3
- lossless 检查通过：各档 greedy 输出逐字相同（投机解码无损，这是**预期**行为）
- 接受长度随深度单调不减

| depth | config 层数 | 平均接受长度 | 接受率 | tok/s | greedy 输出（前 40 字） |
|---|---|---|---|---|---|
| 1 | 1 | 1.303 | 0.043 | 101.5 |  Paris. The capital of Germany is Berlin |
| 2 | 2 | 1.523 | 0.075 | 114.4 |  Paris. The capital of Germany is Berlin |
| 3 | 3 | 1.725 | 0.104 | 124.2 |  Paris. The capital of Germany is Berlin |
| 4 | 4 | 2.099 | 0.157 | 144.6 |  Paris. The capital of Germany is Berlin |
| 5 | 5 | 2.829 | 0.261 | 180.3 |  Paris. The capital of Germany is Berlin |

> 口径：**平均接受长度 = 1 + `num_accepted_tokens / num_drafts`**（与 vLLM 日志的 `Mean acceptance length` 一致；1.00 = 一个草稿都没被接受）；接受率 = `num_accepted_tokens / num_draft_tokens`。
> 判据见 `probes/p06-frontier/T0-runbook.md` §4；上表的展示阈值**不是**预登记判据，仅用于标出'几乎不动'。
