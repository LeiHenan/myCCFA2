# Gate — hybrid 状态一致性复核门（#4 收窄版）

**时间盒**：**2 小时**（不是实验）｜ **交付物**：`notes/gap_hybrid_state.md`（1 页 gap 声明）

**这个门的作用**：在投入任何 GPU 时间之前，判断 #4 还剩不剩一篇论文的空间。**不过门就直接归档，不留恋。**

## 1. 要读什么（全在冻结证据库里，均为本轮已核实证据）

| 文件 | 读什么 |
|---|---|
| `archive/engine_prs.md` | `#53614`（MERGED）、`#50172`（OPEN）、`#55760`（MERGED）、`#56142`（OPEN）、RFC `#55697` + `#55873`/`#55875`/`#55876`（OPEN）、SGLang `#30393`（MERGED） |
| `archive/probe_c/atom.md` | checkpoint 语义：rung / `successor_room` / fork vs copy / **spec 排除**（spec step commits `1 + accepted_drafts`，其余 rollback + re-forward；fork 会让状态裂到两个 group，no single read index spans） |

## 2. 已被占的（**不得重做**）

| 证据 | 状态 | 内容 |
|---|---|---|
| vLLM `#53614` | MERGED 2026-09-06 | Kimi-K3 checkpoint 扩展到 **spec decoding / EAGLE block rewind + partial prefix caching**；"re-keyed from its **provisional boundary**" |
| vLLM `#50172` | OPEN | GDN `mamba_cache_mode="all"` × MTP，**"declares functional correctness as the deliverable"** |
| vLLM `#55760` | MERGED 2026-09-08 | Mamba + EAGLE 的 `prefix_cache_retention_interval` 默认值修复 |
| vLLM `#56142` | OPEN 2026-09-09 | hybrid shared-prefix checkpoint 边界修复 |
| vLLM RFC `#55697` + 3 PR | OPEN 2026-09-09 | 应用声明式 checkpoint 边界 + 调度状态机 + 批量 GDN kernel（L40S 7.58×） |
| SGLang `#30393` | MERGED | draft-cache packed/sidecar HiCache 集成 |

⇒ **"tentative state ≠ committed state" 不是新抽象，已是引擎自己的词汇。** 任何以"提交语义 / checkpoint 边界一致性"为卖点的写法都已被覆盖。

## 3. 门（写入 `notes/gap_hybrid_state.md` 的判定）

**通过**：gap 可写成下面三个方向之一，且**不与上表重叠**：

1. **多分支 SSM 快照的内存放大**：树验证/多分支下 N 个分支 ↔ N 份 recurrent 快照。引擎常量：约 **512 KB/层**，每序列现实上只能留 **1–2 个** checkpoint。要回答：放大倍数、能否共享/分层、对 batch 的挤压。
2. **checkpoint 淘汰策略**：留哪个 rung、留多久、淘汰谁——与 prefix/agent 复用模式耦合。
3. **跨请求 / agent 复用的调度**：谁的状态值得跨请求保留（与 #2 已归档的"保留准入"不同：这里问的是**状态快照**而非 prefix KV 块，且必须与已 ship 的 checkpoint 语义区分）。

**不通过**：gap 的表述仍落在"checkpoint 边界 / 状态提交语义 / 正确性对齐"⇒ **#4 归档**，资源全部转向 #6 与 #12。

## 4. 产出模板

```markdown
# gap 声明 — hybrid 状态（#4 收窄版）
## 一、上表已覆盖了什么（逐条一句话）
## 二、我主张的 gap（一句话，必须可证伪）
## 三、为什么它不是上表的子集（逐条对照）
## 四、第一个可信数字是什么、怎么在 ≤2 周 / ≤2 卡拿到
## 五、结论：通过 / 不通过（不通过 ⇒ 归档）
```

**通过后**：在 `notes/decision_log.md` 登记，并新建 `probes/hybrid_state_*/` runbook（届时再写）。
