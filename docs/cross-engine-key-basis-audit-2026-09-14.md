# 跨引擎键基础审计（E0 扩展）：vLLM vs SGLang

**日期**：2026-09-14 ｜ **成本**：0 GPU·h ｜ **出处**：源码逐行阅读（本机 + 服务器上的 SGLang main 检出）

---

## 0. 一句话结论

**同一个问题，两个引擎解决了不同的子集，且没有任何一方是 sound 的。**

- **vLLM 0.29.0**：把身份分量**拼接进一个无类型元组再哈希** ⇒ 域不分隔（#44701 的根因）；但它**建模了 prompt_embeds**，唯独**漏了 mask**。
- **SGLang main**：把身份分量**作为独立的结构化字段**（`RadixKey` 的 `__slots__`）⇒ **域天然分隔**，且用带版本号、长度前缀的命名空间摘要（比 vLLM 更严谨）；但它**完全没有建模 embeddings**。

⇒ 这不是"某个引擎的一个 bug"，而是**一个尚未有正确答案的设计空间**。这正是课题的立论基础。

---

## 1. 证据对照表（逐行源码）

| 身份分量 | vLLM 0.29.0 | SGLang main | 证据 |
|---|---|---|---|
| prompt token ids | ✅ 在键里 | ✅ 在键里 | vLLM `kv_cache_utils.py:838`；SGLang `radix_cache.py:62` |
| **cache_salt / extra_key** | ⚠️ **在键里，但是裸字符串拼接** | ✅ **独立字段 + 版本化 + 长度前缀** | 见 §2 |
| **LoRA / adapter 身份** | ✅ 在键里（`_gen_lora_extra_hash_keys`） | ➖ **RadixKey 无该字段** | vLLM `kv_cache_utils.py:603`；SGLang `radix_cache.py:62` |
| **prompt_embeds** | ✅ 在键里（逐块 sha256，`_gen_prompt_embeds_extra_hash_keys`） | ❌ **RadixKey 无该字段** | vLLM `kv_cache_utils.py:557-580`；SGLang `radix_cache.py:62` |
| **prompt_is_token_ids mask** | ❌ **完全不在键里（已实测碰撞）** | ❌ **无该概念** | 我的 `probes/repro_cache_key_soundness.py` 与 `key_coverage_census.py` |
| 多模态 | ✅（`mm_extra_keys`） | ➖ 未见 | vLLM `kv_cache_utils.py:526` |

---

## 2. 关键对照：同一个 `cache_salt`，两种设计

**vLLM 0.29.0**（`vllm/v1/core/kv_cache_utils.py:611-613`）：

```python
extra_keys: list[Any] = (
    lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
)
return tuple(extra_keys), new_start_mm_idx
```

⇒ 一个**无类型元组**。LoRA 名与 salt 都是 `str`，所以
`("COLLIDE_SALT",) == ("COLLIDE_SALT",)` ⇒ **同一块哈希**。
**我已实测：sha256 / sha256_cbor / xxhash 三种算法全部碰撞。**

**SGLang main**（`python/sglang/srt/mem_cache/radix_cache.py:62`）：

```python
__slots__ = ("token_ids", "extra_key", "cache_salt", "is_bigram", "limit")
...
if self.cache_salt != other.cache_salt:
    raise ...  # RadixKey 操作要求 cache_salt 匹配
```

⇒ **结构化字段**：`extra_key` 与 `cache_salt` 在类型层面就不可互换。
`RadixKey` 还带 `__repr__` 与显式的不变量检查（:172-175）。

更进一步，SGLang 的命名空间摘要是**版本化 + 长度前缀 + 显式存在性**的
（`python/sglang/srt/mem_cache/utils.py:123-137`）：

```python
digest = hashlib.sha256(b"sglang-cache-namespace-v1")   # ← 版本号
for part in (extra_key, cache_salt):
    if part is None:
        digest.update(b"\x00")                          # ← 区分"缺失"
        continue
    encoded = part.encode("utf-8")
    digest.update(b"\x01" + len(encoded).to_bytes(8, "little") + encoded)  # ← 长度前缀
```

**这三点（版本 / 长度前缀 / 存在性编码）恰好就是 #44701 与 #49125 需要的全部东西。**
也就是说：**vLLM 缺的修复，SGLang 已经独立发明了**——但**没有跨引擎传播**。

---

## 3. 这个对照为什么把课题从"修 bug"提升为"真问题"

1. **不存在"正确的实现"可抄**：
   - 抄 SGLang 的 `RadixKey` 结构化设计 ⇒ 修好 #44701，但 vLLM 会**丢掉** embeddings 的建模；
   - 抄 vLLM 的 embedding 哈希 ⇒ **丢掉** SGLang 的域分隔；
   - 两者都**没有** mask 的概念。
   ⇒ 需要的是**一个统一的身份模型**，而不是从任一引擎移植代码。

2. **两个引擎的键空间彼此不兼容** ⇒ 任何**跨引擎 KV 复用 / 路由 / 分层存储**
   （P/D 分离、LMCache/Mooncake/HiCache、cache-aware router）在键层面就是**没有契约**的。
   这与已观测到的 TRT-LLM #18156（路由侧 `match_count` 恒 0）是同一类问题。

3. **"在键里"与"在键里且 sound"是两件事**：
   vLLM 把 `cache_salt` 放进键里，却在**同一个无类型元组**里，
   所以"在键里"并**不**保证安全。这否证了"把字段加进键就完事"的朴素修补路线。

---

## 4. 对课题方案的修订

**原方案**说"给 KV 缓存补上条目标识这个缺失的抽象"。**跨引擎证据把它加强为**：

> 键基础（key basis）在**主流引擎之间已经分叉**，且没有一个是 sound 的。
> 本文给出 ① 一个可自动化的**键基础审计**；② 一个**引擎无关的身份模型**（类型化分量 +
> 版本 + 域分隔 + fail-closed）；③ 在 **≥2 个引擎**上验证并量化"缓存正确性的代价"。

**新增的可发表内核**：§1 的对照表本身就是一张**没有任何论文画过**的图——
把"哪些身份分量进入了缓存键"作为**跨引擎的一等比较维度**。

---

## 4b. 版本时效性：缺陷在 vLLM **main** 上仍然存在（2026-09-14 实测）

课题的"现行性"是整个立论的前提。我克隆了 vLLM main 并逐行核对
（commit `23cfaad49701c497def53552b23317335431f72a`，日期 **2026-09-14 07:59:51 UTC**，
即**本报告当天**）：

```
$ grep -c "prompt_is_token_ids" vllm/v1/core/kv_cache_utils.py
0                                      # ← mask 依然完全不在缓存键路径里

$ grep -n "extra_keys: list\[Any\]" -A 12 vllm/v1/core/kv_cache_utils.py
628:    extra_keys: list[Any] = (
629:        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
                                       # ← 依然是无类型裸拼接（行号从 611→628 漂移，
                                       #    结构与 0.29.0 逐字相同）
```

⇒ **两条缺陷都不是历史遗留**：截至今天，vLLM 主线仍未把 mask 纳入键，
且 extra keys 仍是**未分隔的拼接**。这使课题的"未被解决"判定有当日证据支撑。

---

## 5. 诚实边界

- 本文对 SGLang 的结论来自**其 main 分支检出的源码阅读**，**未运行** SGLang 做动态验证；
  vLLM 侧则是**静态 + 动态双重验证**（§1 的 ❌ 项均由探针实测）。
- 表中 ➖ 表示"我未在该引擎中找到对应机制"，**不等于"不存在"**——我的搜索范围是该仓库的
  `mem_cache/`、`managers/`、以及 `kv_cache_utils.py`；若该分量在别处被合并进 token 序列，
  我的结论会偏严。
- 我**没有**审计 TRT-LLM（本机无检出）；按方案这属于 E0 的剩余工作。
