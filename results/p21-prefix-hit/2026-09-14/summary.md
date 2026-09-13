# p21 —— 前缀缓存的「命中仍要重算」与「缓存状态改变输出」（KV 地图 B107 / B116 / B109）

**日期**：2026-09-14 ｜ **目标**：`goal-a729588f`（推理加速 / 投机解码 / KV Cache 三方向）
**预登记**：无独立预登记文件（本轮为**诊断性探测**，见 §5 的纪律说明）｜ **成本**：≈0.15 GPU·h（4 次 serve，全部清理，末显存 0 MiB）
**原始数据**：`/root/ccfa_results/2026-09-14/p21_hit/`（本文件是派生物，raw 不入库）
**仪器**：`probes/p21-prefix-hit-recompute/probe_prefix_hit.py`（含 `--selftest`）+ `run_hit_grid.sh`

---

## 1. 问题来自哪里（三张地图的交叉点）

| 格 | 地图标注 | 原话 |
|---|---|---|
| **B107** | OPEN，**untested** | *"EAGLE/MTP prefix-cache last-block drop causes a **1,648-token recompute per hit**"*；姊妹 issue #51771：*"EAGLE/MTP block drop + prefix caching is **untested**"* |
| **B116** | OPEN（2026-08-31 新开） | *"The minimal accepted prefix-cache configuration produces **different text for two identical long-prompt requests**, while the no-prefix baseline passes."* |
| **B109** | OPEN（bug） | *"Identical prompts at `temperature=0` produce **_completely_ different output sequences across runs**"*（记者称基础调度器 block 生命周期 bug） |

**设计**：同一批 prompt、同一顺序、前缀缓存**全臂默认开启**（vLLM 0.29.0 的 `config/cache.py:138` 默认 `True`，服务日志实测有 `enable_prefix_caching=True`），
走**三段对照**：`A_cold`（首次出现 / 必然 miss）→ `B_warm`（同样顺序再发 / 应命中）→ `C_warm2`（再发一次 / 同状态）。
**一次运行同时给出**：B107 = 阶段 B 的 `vllm:request_prefill_kv_computed_tokens`（官方定义 *"new KV tokens computed during prefill (**excluding cached tokens**)"*）每请求均值；
B116 = A 与 B 的输出是否相同；B109 = B 与 C 的输出是否相同（纯重复性）。

**为什么用这个量而不是 TTFT**：`prefix_cache_hits/queries` 与 `request_prefill_kv_computed_tokens` 是**引擎自报的因果量**（命中多少、白算了多少），
TTFT 只是它的下游表现。历史教训：绝对值不可用，**计数器必须做差**。

---

## 2. 结果（Qwen3-4B / 8 个 4096-token prompt / max_tokens=32 / T=0 / ignore_eos / sm120 / `--enforce-eager`）

| 臂 | 阶段 B 命中 / 查询 | 阶段 B 每请求**重算** KV token | 冷 vs 热 输出相同 | 热 vs 热 输出相同 |
|---|---|---|---|---|
| `S_nospec`（不投机） | 32,640 / 32,768 | **16**（1 个 block） | 7 / 8 | 8 / 8 |
| `S_eagle3_k3` | 32,512 / 32,768 | **32**（2 个 block） | 7 / 8 | 8 / 8 |
| `S_ngram7` | 32,640 / 32,768 | **16**（1 个 block） | **5 / 8** | 8 / 8 |
| **`S_dflash2_k3`** | **0 / 32,768** | **4096 = 100%** | 8 / 8 | **7 / 8** |

**逐条读数**：

1. **`S_dflash2_k3`（DFlash2，MTP/DFlash 类）前缀缓存完全不命中**：阶段 B 的 `prompt_tokens_cached=0`、`prefix_cache_hits=0/32768`、
   `prefill_kv_computed` 每请求 **4096**（= 整个 prompt 重算），命中时重算占 prompt 比例 = **1.0**。
   对照 `S_nospec` 命中时只重算 **16** token（block 边界的最后一块）⇒ **同一操作白算的 prefill 相差 256×**。
2. **这是方法特异的，不是"投机解码破坏缓存"**：`eagle3`（32 token）与 `ngram`（16 token）**与不投机同样正常命中**。
   三张地图里"EAGLE/MTP 前缀缓存"被放在一起讲，但**实测只有 dflash 系失败**。
3. **不是一次性、而是延迟一拍**：阶段 C（第三遍）dflash2 **命中 32,512 / 32,768**（重算降到 32 token）。
   ⇒ 「第 1 遍不命中、**第 2 遍仍不命中**、第 3 遍才命中」。服务日志自报的累计命中率与此吻合
   （dflash2 **33.1%** ≈ 1/3 三阶段中只有第三阶段命中；nospec **57.7%**）。
4. **它同时解释了 p20 网格的异常**：p20 v2 里 A1（不投机）第 2 遍就暖、而 **A2–A6（全部 dflash）到第 3 遍才暖**
   （吞吐 537→532→**919**、TTFT 368→328→**74 ms**）。v2 的"热 p50"因此是**冷热状态不匹配**的伪比较 —— 已在 v3 用预热趟修正（见 §5）。
5. **B116 在三个臂上复现**：`nospec` 7/8、`eagle3` 7/8、`ngram` **5/8** —— 同一 prompt、T=0，**只因为前缀缓存冷/热不同就给出不同文本**。
   且 **冷 vs 热 的 token_logprobs 在四个臂上全部 8/8 不同**（位级不稳定），而**热 vs 热**在三个非 dflash 臂上 8/8 稳定。
6. **B109 只在 dflash2 臂出现**：`S_dflash2_k3` 的热 vs 热出现 **7/8**（1 个 prompt 两次同状态请求给出不同文本）。
   与 §3 的机制一致 —— 读取未初始化/陈旧的 draft context KV 本就会带来不确定性。

---

## 3. ⛔ S3b 判定：**已被上游占位并正在修复 → 不立项**（本轮最重要的结论）

在把上述数字写成候选**之前**做 S3b 检索，结果**直接命中同题**，而且比我测得更大规模、更早、正在修：

| artifact | 状态 | 关键内容（引文） |
|---|---|---|
| **vLLM issue #47930** | **OPEN** | 标题即我的结论：*"[Bug]: **DFlash/DSpark draft acceptance collapses with automatic prefix caching enabled**"* |
| **vLLM PR #47926** | OPEN/Draft（等 code-owner） | 机制原文：*"DFlash/DSpark drafters build their context KV from the target's auxiliary hidden states… Tokens whose KV is restored at request (re)admission — **automatic prefix caching hits**, KV-connector restores, resumption after preemption — **never run through the target, so their draft context KV slots are never written**… so it **reads uninitialized/stale KV for the whole restored region**. The impact scales with the cache-hit length… while **MTP/EAGLE-style drafters on the same workload are unaffected**… **Existing public DSpark benchmark recipes work around it by serving with `--no-enable-prefix-caching`**"* |
| **vLLM PR #54163** | **OPEN**（`Fixes #53477`） | *"DFlash/DSpark no longer drop the last prefix-cache block"*；*"the next turn's fixed-point prefix-cache lookup converged to 0 → **the whole context was recomputed on every reply**"* |
| **vLLM issue #54094** | **OPEN**（取代已 CLOSED 的 #54027，后者 `stateReason: COMPLETED`，原文 *"Superseded by #54094"*） | *"An identical 1,040,011-token prompt gets near-complete prefix-cache reuse with target-only static YaRN, but **zero reuse after enabling a DFlash2 K=7 drafter**"*；环境栏：**`GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`**、*"Resolved attention/Mamba block size: **1,648 tokens**"*（正是 B107 标题里那个 1,648） |
| **#47930 评论中的第三个部署** | — | *"We hit this too, on **Kimi-K3 with Inferact/Kimi-K3-DSpark**… vLLM 0.28.1, TP8 on B300, k=3… **prefix caching on, real agent traffic**… Aggregate acceptance is **1.70 tokens/step over 10 days**… the shape tracks **'damage scales with the restored region'**"*（按 prompt 长度分桶：0–16k 2.98 → 256k+ **1.25**） |

**⇒ 按 S3b v1.1.2 三问**：② **有未关闭的同类失败报告**（#47930、#54094 均 OPEN）✔命中；
③ **上游正在修**（两个 OPEN PR）✔命中 ⇒ **记 🔴，不立项**。
这正是 `notes/FILTER_3AXIS.md` §1（C 类优先）与 §6 想拦下的情形，而它**在花掉任何 GPU·h 之前就被拦下了**。

### 3.1 我这次测量**仍然有**的独立价值（记录，但不据此立项）

三张地图与上游报告里的既有案例**全部落在特殊形态上**：#54094 是 **YaRN + 1.04M 上下文 + mamba align**、
#40624 是 **Gemma4 混合注意力**、#54163 的推导链是 **mamba block**、#47930 的复现是 **Kimi-K3 DSpark / TP8 / 长 agent 会话**。
**我的是最小条件**：**dense Qwen3-4B、4096 token、无 YaRN、无 mamba、无混合注意力、bf16 KV、sm120、vLLM 0.29.0**。
⇒ 我把「需要 mamba/长上下文/YaRN 才能触发」这一隐含量级**去掉**了：**在普通 dense 模型 + 4k prompt 上就已 100% 失效**。
另外我观察到上游报告都没写的**时间结构**（第 2 遍仍 0 命中、第 3 遍才命中），它与 #54163 的
*"the **next turn's** fixed-point prefix-cache lookup converged to 0"* 互为独立佐证。

**⇒ 结论**：这是一个**可引用的复现**，但**不是未被占位的缺口**，因此不作为候选。若将来要主张，
只能走"**最小条件下失效 + 跨引擎**"这条窄差异轴，且必须与 #47926/#54163 **同时**比较 —— 按 §1 判据价值很低。

---

## 4. 与我此前结论的关系（必须更正的地方）

- **decision #117–#119 关闭"批不变性"时的前提被本次结果收窄**：我当时说"开关有效 ⇒ 无可回收空间"，
  但那批实验**全部在关闭前缀缓存下做**。本次显示**默认配置（前缀缓存开启）下确定性本身就不成立**
  （B116 在 nospec 上也复现：冷 vs 热 7/8、logprobs 8/8 不同）。这**不改变**"不立项"的结论
  （B108 的实现 PR #46592 是他人 OPEN 提案），但**改变了理由**：不是"已解决"，而是"**未解决且已被他人占位**"。
- **B107 的措辞被实测修正**：地图把"EAGLE/MTP"合写；实测 **dflash 系失效、eagle3/MTP 类正常**。
  上游 #47926 的机制说明与我的实测**完全一致**（dflash 需要 target 隐藏状态建 context KV，eagle 不需要）。

---

## 5. 纪律与自我更正

1. **本轮是诊断性探测，不是预登记实验** —— 它由 p20 v2 里观察到的**异常**（spec 臂第 3 遍才暖）触发，
   目的是**找机制**而不是验证假设。因此**不据此宣布任何"效应"**；凡涉及效应大小处均只报原始数字。
   由此产生的**协议变更已按"采数前修订"写进** `notes/prereg/p20-dsd-concurrency.md`（v3 加预热趟）。
2. **仪器侧的一处自我更正（连带影响 decision #120）**：`override_envs_for_invariance()` 设的是
   `CUBLAS_WORKSPACE_CONFIG=":4096:8"`（`batch_invariant.py:983`），**不是**我 #120 里记的 `":16:8"`
   （那是 `:932` 的另一个分支）。**#120 的实验结论不受影响**（我当时两个值都测了：`:4096:8` 与不设同为 `div 3/8`，
   `:16:8` 反而更差 `6/8`）⇒ "工作区配置不是不变性的来源、也不花钱"这一结论**反而被加强**，但**归因须更正**。
3. **p21 的 `--logprobs 1` 只用于灵敏度**：位级比较（`|Δ| < 1e-9`）必然把"不同 prefill 路径的浮点差异"也算作不等，
   所以"logprobs 8/8 不同"**不能**单独当作"语义不同"的证据；**文本不同（5–7/8）才是**。
4. **未取证**：本轮没有做跨引擎（SGLang）对照，也没有验证 #47926/#54163 的修复是否在我们的 commit 上生效
   （它们未 merge，故不可能生效）。§3 的上游引文来自检索到的页面正文，**我未逐条在本地复现其环境**。

---

## 6. 复现命令

```bash
# 服务（每个臂一条；前缀缓存保持默认开启）
/root/ccfa_venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen3-4B --served-model-name q3 --port 32090 \
  --max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization 0.55 --enforce-eager \
  --speculative-config '{"method":"dflash","model":"/root/autodl-tmp/models/dflash2","num_speculative_tokens":3}'

# 三段冷/热对照（同时给 B107 / B116 / B109）
/root/ccfa_venv/bin/python probes/p21-prefix-hit-recompute/probe_prefix_hit.py \
  --base-url http://127.0.0.1:32090 --prompts /root/autodl-tmp/prompts/prompts_4096.jsonl \
  --n-prompts 8 --max-tokens 32 --logprobs 1 --out /root/ccfa_results/.../S_dflash2_k3
```

**成本**：4 次 serve ×（起 ~40 s + 24 请求）≈ 0.15 GPU·h ｜ 末态：显存 0 MiB、无残留 API server、无残留 EngineCore。
