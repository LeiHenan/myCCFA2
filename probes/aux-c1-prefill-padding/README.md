# aux-C1 — prefill CUDA-graph padding 探针

**填缝探针**（主线等待期跑）｜ 1 周 / 1 卡 ｜ 详见 `EXECUTION_PLAN.md` §8.3

- **记什么**：每次 replay 的 `padded/real` 比值。
- **已知事实**：`_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2`。
- **产物**：`results/aux-c1-prefill-padding/<日期>/{summary.md, padding.csv}`
