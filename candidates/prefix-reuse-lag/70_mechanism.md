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


## 插桩 v2（按 KV 组）：**组结构本身就是差异所在**

同一套插桩加了 `kv_cache_group_id` 与每组的命中长度（`[T-FLCH]` / `[T-HIT]` / `[T-CACHE] gid=`）：

| 配置 | 协调器 | **KV 组数** | 第 1 次命中 | **第 2 次命中** |
|---|---|---|---|---|
| **投机关** | `UnitaryKVCacheCoordinator` | **1** | `per_group=[0]` | **`per_group=[256]` ⇒ `computed=4096` ✅** |
| **投机开 d5γ3** | `HybridKVCacheCoordinator` | **9** | `per_group=[0,0,0,0,0,0,0,0,0]` | **`per_group=[0×9]` ⇒ `computed=0` ❌** |

**填充侧**（同一实验）：投机开时 `[T-CACHE] gid=0/1/2 … cached_before=0 full_blocks=256 block_size=16`
⇒ **每个组都把整段 prompt（256 块）写进了缓存**。

### ⇒ 这一轮把机制压缩成一个明确的问题

1. 投机解码把 KV 缓存从 **1 组**变成 **9 组**（`HybridKVCacheCoordinator`）——
   推测是 target 的注意力 KV + drafter 的多类状态（注意力/卷积/SSM 等）各成一组；
2. 混合协调器要对多组**取一致**（`find_longest_cache_hit` 返回单一 `hit_len`）⇒ **任一组为 0，整体命中即为 0**；
3. 投机关时唯一那组在第 2 次就命中 256 块；投机开时 **9 个组在第 2 次全部为 0** ——
   而它们在第 1 次明明都填充过 ⇒ 只剩两种可能：
   **(a) 缓存在两次之间被驱逐了**（投机下 9 组共享显存 ⇒ 压力大得多）；
   **(b) 查表时哈希不匹配**（组相关的哈希/键差异）。
4. ⚠️ **本轮的一个仪器缺陷**（如实记录）：我的驱逐计数器（`_maybe_evict_cached_block`）**只打印前 8 次**，
   于是 `grep -c T-EVICT` 得到的两边都是 8 —— **这是打印上限，不是真实次数** ⇒ 驱逐量目前**未知**，
   而它恰是区分 (a)/(b) 的关键。

### 下一步（唯一且明确）

把驱逐统计改成"进程退出时输出一次汇总"，并在查表处对**第一个块哈希**直接探测
`block_pool.get_cached_block(hash)`，从而判定：
- 若第 1 个块已不在池中 ⇒ **(a) 驱逐**所致（干预方向：让缓存块不被投机负载挤出，或提高缓存优先级）；
- 若第 1 个块在池中但查表返回 0 ⇒ **(b) 哈希/组键不匹配**（干预方向：对齐组间的哈希或放宽"多组必须一致"的约束）。


## 插桩 v3：**机制确认 —— 命中在 9 个 KV 组之间是"全有或全无"，而 target 组本来已经命中**

判别器：`BlockPool.get_cached_block(hash, group_ids)` **只要任一 group 缺失就返回 `None`**
（`block_pool.py:198`，逐组查 `make_block_hash_with_group_id(hash, gid)`）。
于是对请求的**第一个块哈希**做两次探测：`[0]`（只看 target 组）与 `list(range(9))`（引擎实际语义）。

| 配置 | 第 1 次（lookup #1–3） | **第 2 次（lookup #33–35）** | `computed` |
|---|---|---|---|
| **投机关**（1 组） | `group0=MISS` / `allgroups=MISS` | **`group0=HIT` / `allgroups=HIT`**，`per_group=[256]` | **4096** ✅ |
| **投机开 d5γ3**（9 组） | `group0=MISS` / `allgroups=MISS` | **`group0=HIT` 但 `allgroups=MISS`**，`per_group=[0×9]` | **0** ❌ |

**驱逐量 = 0**（进程退出汇总：`calls_to_evict=0 actual_evictions=0`，本次统计**无打印上限**）
⇒ **排除驱逐**。

### ⇒ 机制（已确认，非推测）

1. 投机解码把 KV 缓存拆成 **9 组**（target 的注意力 KV + drafter 的多类状态各成一组）；
2. **命中判定要求 9 组同时命中**（`get_cached_block` 任一组缺失即返回 `None`）；
3. 第 2 次出现时，**target 组（group 0）已经命中**，但 **8 个非 target 组没有该块** ⇒ 整体判为 0
   ⇒ **target 已缓存的 256 个块被白白浪费**，整段 prefill 重做；
4. 到第 3 次时，其余组也积累了对应的块 ⇒ 整体命中 ⇒ 吞吐 2.63× 跳变。

**这是一个可修复的引擎内部设计问题**：drafter 的状态组**本可以重算**（代价远小于重做整段 target prefill），
但当前"全组一致"的语义让 target 的复用被 drafter 组绑架。

### 干预方向（C4 判据的直接候选）

- **最小干预**：让命中长度由**可前缀缓存的组**（至少 group 0）决定，而非全组一致；drafter 组缺失时
  按"重算 drafter 状态"处理（drafter 只有几层，代价 << 重做 4k prompt 的 target prefill）；
- **验证方式**：打补丁后重跑 onset 实验，看 **onset 是否从 3 回到 2**、该 rep 吞吐是否 **≥+8%**（C4 判据）。
- ⚠️ **必须同时验证正确性**：drafter 状态重算后输出是否与非投机路径一致（lossless），否则不能主张。


## 干预实验 #1（2026-09-13）：**未达成 C4** —— "只由 group 0 决定命中"没有把 onset 从 3 提前到 2

**干预**：`CCFA_GROUP0_ONLY=1` 时把 `HybridKVCacheCoordinator.find_longest_cache_hit` 替换为只用
group 0（full-attention 组）的 manager 计算命中，其余组返回空（引擎会为它们重算）。
**⚠️ 更正**：中途我曾据 trace 误判"干预成功"，实际那 32 条 `computed=4080` 是**第 3 次**的
（`computed=0` 共 64 条 = 前两次各 32 条）。

| 配置 | 命中增量（逐次） | 吞吐（逐次） | TTFT（逐次 ms） | onset |
|---|---|---|---|---|
| 基线（投机开，无干预） | 0, 0, 130560 | 751 / 751 / 2000 | 2119 / 2084 / 252 | **3** |
| **干预（group0-only，投机开）** | 0, 0, 130560 | 751 / 751 / 2000 | 2119 / 2084 / 252 | **3** |

⇒ **干预无效**：即便把命中判定缩到 group 0，第 2 次仍然 `hit_len=0`。

### 由此得到的新证据（与 v3 的探测对比后相当关键）

v3 的探测显示：第 2 次时 **`get_cached_block(h0, [0])` = HIT**（group 0 的**第一个块在池中**）。
而干预后**只查 group 0** 时 `hit_len` 仍然是 0 ⇒ 说明链式查找在第 2 个块之前就停了，或者
**命中的块数被 `drop_eagle_block` 吃掉了**（若链上只命中 1 块，`drop_eagle_block=True` 会把它丢掉 ⇒ 0）。
两种可能与"只有第一个块在 group 0 里"一致 —— 即**第 1 次运行只把每个 prompt 的首块留在了 group 0**，
其余块（虽然填充日志显示 `full_blocks=256`）在两次之间**不在池中**。

### 下一步（二选一，都是零/低成本）

1. **探测链深度**：对第 2 次请求，逐个块哈希探测 group 0 的命中情况
   （`get_cached_block(h_i, [0])` for i=0..k），直接数出"连续命中几个块"
   ⇒ 若只有 1 ⇒ 查"为什么只留下首块"（释放顺序 / 驱逐策略 / 块队列语义）；
2. **同时**统计 `drop_eagle_block` 的作用：把该参数强制为 False 再测一次
   ⇒ 若命中数从 0 变成 1 块，说明**首块命中被 EAGLE 丢块规则抹掉**，那修复点就非常具体。


## 干预实验 #2（2026-09-13）：**块明明都在缓存里，查找仍返回 0；且与 EAGLE 丢块无关**

**探测方法**：在 `get_computed_blocks` 外层对**请求自己的 `block_hashes[:10]`** 逐块调
`block_pool.get_cached_block(h_i, [0])`；另加开关 `CCFA_NO_DROP_EAGLE=1` 强制 `use_eagle=False`。

| lookup | `computed` | group 0 前 10 块探测 |
|---|---|---|
| #001（第 1 次） | 0 | `0000000000`（尚未缓存，正常） |
| **#033（第 2 次）** | **0** | **`1111111111` —— 十个块全部命中** |
| #065（第 3 次） | 4080 | `1111111111` |

| 配置 | 命中增量（逐次） | 吞吐（逐次） | onset |
|---|---|---|---|
| 基线 | 0, 0, 130560 | 751 / 743 / 1950 | 3 |
| **关掉 EAGLE 丢块** | 0, 0, 130560 | 746 / 755 / 1936 | **3（无变化）** |

### 三个被本轮排除的假设

1. ❌ **"链在第 2 块前断了"** —— 第 2 次时前 10 块**全部命中**（探测直接打脸）；
2. ❌ **"唯一命中被 EAGLE 丢块抹掉"** —— `use_eagle=False` 后逐位不变；
3. ❌ **"非 target 组拖累"** —— 上一轮（decision #89）只查 group 0 也返回 0。

### ⇒ 问题被逼到唯一位置：**命中长度的内部计算**

块在池中、哈希能命中、组已隔离、丢块规则关掉 —— 但 `find_longest_cache_hit` 仍返回 0。
剩下的可能（下一轮逐个打点）：
- **对齐/取整**：`alignment_tokens` 与 `max_cache_hit_length` 的组合把长度取整到 0
  （注意第 3 次返回的是 **4080** 而不是 4096 ⇒ 内部确有非 16 的对齐/截断在起作用）；
- **`num_prefill_lookahead`**：混合协调器构造参数里有这一项，可能改变候选长度；
- **每步偏移**：请求的 `num_computed_tokens` 在第 2 次调度时非 0，导致传入的哈希列表被偏移。

**下一轮**：直接在 `FullAttentionManager.find_longest_cache_hit` 内部打点（进入时的 `max_length`、
逐块命中计数、返回前的 `hit_length`），即可定位是"取整归零"还是"偏移导致的空表"。

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
