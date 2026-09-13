# S7 归因与机制（Attribution） — 前缀复用的生效时延（重复前缀第 2 次仍零命中）

**候选**：`prefix-reuse-lag` ｜ **成本上限**：≤0.5 天（多数可复用 S6 已采的数据）
**目标**：把效应拆成可命名的因子，并排除显存/排队/顺序/冷启动的替代解释。

## 效应分解

（填写）


## 插桩定位（2026-09-13，sitecustomize 注入，未改 vLLM 源码）

方法：把 `sitecustomize.py` 放进 venv 的 site-packages（每个子进程自动加载，含 EngineCore），
在 `KVCacheManager.get_computed_blocks`（**查询**）与 `BlockPool.cache_full_blocks`（**填充**）打点；
`CCFA_TRACE_PREFIX=1` 时生效。原始证据：`results/p10-prefix-reuse-lag/2026-09-13/trace_evidence.txt`。

### 结果一：**填充路径两边完全一致** ⇒ 块确实被缓存了

| | 投机关（off） | 投机开（d5γ3） |
|---|---|---|
| `TRACE-CACHE` 行数 | 559 | 607 |
| `full_blocks` 最大值 | **264** | **264** |
| 首行模式 | `cached_before=0 full_blocks=256 mask=None` | `cached_before=0 full_blocks=256 mask=None` |
| `full_blocks >= 250` 的行数 | 549 | 581 |

⇒ **"投机下不缓存"被排除**：两边都把整段 prompt（256 块）连同生成部分（到 264 块）写进了前缀缓存。

### 结果二：**差异在查询侧**

| rep | 投机关 `computed` | 投机开 `computed` |
|---|---|---|
| 第 1 次 | 0（应当） | 0（应当） |
| 第 2 次 | **4096**（= 256 块，命中） | **0（未命中）** |

### 结果三：已排除的闸门

`get_computed_blocks` 的第一个分支是 `if not self.prefix_cache_lookup_enabled(request)`，
而该函数是 `enable_caching and not request.skip_reading_prefix_cache`。
逐行核对 `skip_reading_prefix_cache` 的赋值点：它**只在 `prompt_logprobs is not None`（或 pooling 模型）时为 True**
（`sampling_params.py:543`、`pooling_params.py:94/125`）—— **我们没用 prompt_logprobs ⇒ 该闸门不是原因**。

### 结果四：找到投机专属的机制成分（但不完整）

`kv_cache_coordinator.py:63`：`drop_eagle_block = 0 in self.eagle_group_ids` ——
**当 KV 组属于 EAGLE/MTP（即带 draft 模型的投机）时，命中计算会丢掉最后一块**
（`SpecGroup` 注释亦云 "the EAGLE last-block drop"）。
这**正好解释**命中量差异：投机开 **255 块/prompt**（130,560 tokens）vs 投机关 **256 块/prompt**（131,072 tokens）。

⚠️ **但它解释不了"第 2 次整体零命中"** —— 丢一块只会让命中从 256 变 255，不会变成 0。

## 尚未解释的部分与下一步

**未解释**：投机开时第 2 次查询为什么返回 0（块已在缓存里、闸门未触发、只应少 1 块）。
**下一步（按优先级）**：
1. 在 `find_longest_cache_hit`（`single_type_kv_cache_manager.py:552/692/…`）打点：记录
   `block_hashes[0]` 的前若干位、`max_length`、`hit_length`、以及命中的 manager 类名
   ⇒ 判定是"哈希不匹配"还是"多组取 min 导致 0"；
2. 在同一插桩里统计 `_maybe_evict_cached_block` 的调用次数（排除"查之前已被驱逐"）；
3. 若确认是 **多 KV 组（drafter 组）拖累**（例如 hybrid 协调器对多组取最小值、而 drafter 组在第 2 次尚未缓存），
   则干预方向明确：**让命中不再被 drafter 组拖累/或让 drafter 组同步缓存**。

## 替代解释与排除证据

（填写）

## 适用范围

（填写）

## 上界复核

（填写）

## 完成清单

> 校验器逐条比对下面的文本；全部 `[x]` 才算该阶段完成。

- [ ] 把总效应**分解**为可命名的因子（例：`Δln(吞吐) = Δln(接受长度) − Δln(每步代价)`），并给出各因子占比
- [ ] 逐个排除替代解释（显存挤占 / 排队 / 顺序 / 冷启动 / 跨机器差异），且排除证据用**计数器**而非 gauge
- [ ] 写明结论的**适用范围**（哪些轴、族、上下文长度、并发区间没测）
- [ ] 复核 S4 的 oracle 上界是否仍成立（实测有没有超过上界 ⇒ 说明上界算错了）

> **杀出口**：效应无法归因 ⇒ 降级为现象记录，禁止写成机制主张
