# aux-C2 — FlashInfer radix 探针

**填缝探针**（主线等待期跑）｜ 1 周 / 1 卡 ｜ 详见 `EXECUTION_PLAN.md` §8.3

- **已知事实**：Blackwell 上 FlashInfer 是默认后端，而 `attention_hook.py:570` 对其**静默** `disable_radix_cache=True`（vLLM `attention.py` L370 有同样的镜像）。
- **要回答**：这个静默关闭在真实负载上损失多少命中率/吞吐。
- **产物**：`results/aux-c2-flashinfer-radix/<日期>/{summary.md, hit_rate.csv}`
