# C-1 判定：**预登记假设被证伪** —— prefix caching 对 dflash 不是有害的，而是**最有益的**

**日期**：2026-09-14 ｜ **平台**：RTX 4080 SUPER 32GB，vLLM 0.29.0，Qwen3-4B（真实文本，多轮对话，1024-token system prefix）
**预登记**：`prereg_C1_spec_prefix_2026-09-14.md`（**采数前写下**，含杀判据）
**成本**：≈0.6 GPU·h

---

## 0. 一句话

**我的预登记假设被自己的数据证伪，而且方向是反的。**
我预测 `dflash2` 会被前缀缓存破坏；实测 **`dflash2` 是四档里吞吐最高的，且从 PC 中获益不比任何 drafter 少**。

---

## 1. 实测结果（4 drafter × PC 开/关，每格 3 遍，48 请求/遍）

| drafter | PC **on** (tok/s) | PC **off** (tok/s) | **off/on**（越大=PC 越有用） |
|---|---|---|---|
| 无投机 | 1196.9 | 230.5 | 0.193 |
| **dflash2** | **1461.6** | 270.0 | **0.185** |
| eagle3 | 1431.3 | 259.1 | 0.181 |
| ngram | 1231.9 | 253.1 | 0.205 |

（取第 3 遍；第 2、3 遍差异 <2%，即稳态口径）

### 逐条对照预登记判据

| 预测 | 内容 | 实测 | 判定 |
|---|---|---|---|
| **P1** | dflash2 在热缓存遍 `cached_tokens ≈ 0`（全量重算） | **未见**：dflash2 PC-on 三遍 **728 → 1172 → 1462 tok/s**，第 2/3 遍显著快于第 1 遍 ⇒ **缓存确实在命中** | ❌ **证伪** |
| **P2** | eagle3 / ngram 正常命中 | ✅ 两者同样呈第 2/3 遍变快（eagle3 1342→1431；ngram 411→1232） | ✅ 成立（但无区分度） |
| **P3**（**主判据**） | dflash2 的 off/on **> 1.15**，且**大于** eagle3/ngram 至少 15% | dflash2 off/on = **0.185**，**远小于 1.15**；且 **0.185 < 0.181×1.15**，也**不高于** eagle3 | ❌ **证伪（且方向相反）** |

**⇒ 按预登记 §4 的杀判据：P1 失败 + P3 失败 ⇒ 本候选归档。**

---

## 2. 与上游材料的冲突（必须记下来）

本工作区的侦察报告记载：
> vLLM issue **#47930**（OPEN，我核过：**零 linked PR**）标题即
> *"DFlash/DSpark draft acceptance collapses with automatic prefix caching enabled"*；
> PR #47926（我核过：**仍是 DRAFT**）声称 dflash 需要 target 辅助隐藏状态，
> 而前缀缓存恢复的 token 从不经过 target ⇒ 读到未初始化 KV。

**我的实测与它冲突。** 可能的原因（我**没有**验证，仅列出）：

1. **方法名差异**：vLLM 0.29 的方法名是 `dflash`，而我下载的 checkpoint 叫 `*.dflash2`。
   两者可能是不同实现路径（`DFlashModelTypes = Literal["dflash"]`、`DSparkModelTypes = Literal["dspark"]`，
   **`dflash2` 不是合法值**）——我踩过这个坑，见 §3。
2. **模型族差异**：上游报的是 hybrid GDN / Qwen3.8 系，我用的是 dense Qwen3-4B。
3. **版本差异**：上游报的引擎版本可能早于 0.29.0。
4. **可能已被修**（尽管 issue 仍 OPEN、PR 仍 DRAFT——但或许部分修复已随别处合并）。

**⇒ 我不断言上游是错的。** 我断言的是：**在我这个配置下，预登记的两条预测都不成立。**

---

## 3. 本轮顺带得到的硬事实（对做投机解码的人有用）

### 3.1 **前缀缓存对投机解码是巨幅正收益，不是风险**

无投机时 PC 给 **5.19×**；有投机时 **4.9–5.5×**。**四档全部同向、量级一致。**
之前我把这条方向记为"上游在修 = 危险区"，实测显示**它在我的配置下是纯粹的好事**。

### 3.2 `ngram` 的冷启动特征（新观察）

```
ngram  PC=on  : pass0 410.6  → pass1 1173.0 → pass2 1231.9   （第 1 遍慢 3×）
ngram  PC=off : pass0 181.9  → pass1  252.5 → pass2  253.1
其余三档      : 三遍差异 < 10%
```

**只有 ngram 在第 1 遍显著慢**。合理机制：ngram 的草稿来自 prompt 内已有 n-gram，
冷缓存时第 1 遍没有可复用前缀，草稿命中率低，但**它仍然付了投机开销**。
⇒ **"ngram 在冷缓存首遍是负收益"** 是可测的，但**本次未设对照**（第 1 遍没有 no-spec 同条件比较），
**不能作为结论**，只登记为观察。

### 3.3 一个可复用的工程坑（值得记档）

`speculative_config={"method": "dflash2"}` 会被 pydantic 拒绝，合法值是
`ngram / medusa / mlp_speculator / draft_model / suffix / custom_class / eagle / eagle3 /
extract_hidden_states / mtp / ... / dflash / dspark / ngram_gpu`。
**`mgoin/Qwen3-4B-speculator.dflash2` 这个 checkpoint 要用 `method="dflash"` 加载。**

---

## 4. 归档理由

| 判据 | 结果 |
|---|---|
| 有可命名的观测轴 | ✅ 吞吐 / cached_tokens |
| **有量级** | ❌ **效应方向与预测相反**；dflash2 off/on = 0.185（预登记要求 >1.15） |
| 有受害者 | ❌ dflash2 是**受益最大**的一档，没有受害者 |
| 未治理 | ❌ 无需治理 |

⇒ **归档。这是本会话第 11 个被杀的候选，且是第一个"预登记判据先写好、再被自己数据打掉"的。**

---

## 5. 诚实边界

- 单模型（Qwen3-4B dense）、单卡、`enforce_eager`、`max_model_len=4096`、并发 1（`llm.generate` 批量）。
  **未测并发 >1**（预登记网格里有，本轮的 harness 没实现并发臂——**这是我的执行缺口，不是数据**）。
- **没有直接读 `cached_tokens`**：P1/P2 是靠"第 2/3 遍是否变快"推断的。
  预登记要求直接读 per-request cached tokens，**我没做到**（离线 `LLM` API 未暴露该字段）。
  ⇒ 若要用更硬的证据重开此案，应改用 OpenAI server + `/metrics` 的 `prefix_cache_hits_total`。
- 我**没有**验证 §2 列出的四个冲突原因；那需要额外的对照实验。
- 未测 **hybrid（GDN/Mamba）模型**——而上游报的正是 hybrid。**这是最可能的调和点**，
  也是若要重开此案的第一优先实验。
