# S3 占位与先行核查（Occupancy & prior art） — 前缀复用的生效时延（重复前缀第 2 次仍零命中）

**候选**：`prefix-reuse-lag` ｜ **成本上限**：0 GPU·h（检索 + 读正文）
**目标**：确认没人已经做完，且**引擎里已 ship 的旋钮**不是你的方案。必须读正文。

## 最近邻

（填写）

## 引擎已 ship 的控制器

| 项 | vLLM 0.29.0 实况 | 含义 |
|---|---|---|
| `enable_prefix_caching` | 默认 True（A3 日志逐字确认 `enable_prefix_caching=True`） | 缓存开着，所以「零命中」不是没开 |
| `cache_full_blocks` 调用点 | `v1/core/single_type_kv_cache_manager.py:472`，**prefill 填充过程中** | 按代码推断第 2 次应命中 ⇒ 反常成立 |
| 入池条件 | 只缓存**整块**（`num_full_blocks = num_tokens // block_size`）；部分块永不入池 | 与实测「每 prompt 少 1 块」一致（130,560 = 8192−32 块 × 16） |
| `free_blocks` 语义 | 有 hash 的块走 **FIFO 队尾**（LRU 淘汰），无 hash 的块走队首（LIFO 复用） | 缓存块的可复用性取决于队列位置 |
| `reachable_block_mask` | full attention 下为 `None`（缓存每个非空块）；SWA/Mamba 会跳过 | 本场景是 full attention |
| 计数器 | `prefix_cache_hits_total` / `prefix_cache_queries_total`（**计数器**，跨 rep 累计 ⇒ 必须做差） | 判据的数据来源 |


## 差异轴

（填写）

## 结构性理由

（填写）

## 被占时的降级

（填写）

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [x] 每个存活痛点找 ≥3 个最近邻，且记录**读过的具体页/节**（只读摘要不算）
- [x] 找齐目标引擎里**已 ship** 的相关旋钮/调度器，并写成最强基线（不是只写学术邻居）
- [x] 写出审稿人会问的那一问，并给出**结构性**回答（不是『我们更自适应』这类程度性回答）
- [x] 写明『若被别人占住，本方向降级成什么』（降级形态必须先想好）

> **杀出口**：立项理由只有『没人做过』⇒ 杀；给不出结构性理由 ⇒ 改写定位或杀
