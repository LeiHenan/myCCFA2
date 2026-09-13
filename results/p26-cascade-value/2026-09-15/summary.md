# p26 —— cascade attention 值多少？（同后端内的干净 A/B）｜ **结论：≈0，候选死于量级**

**日期**：2026-09-15 ｜ **目标**：`goal-a729588f` ｜ **成本**：≈0.2 GPU·h（含 2 次被断言拦下的无效 A/B + 1 次手工验证）
**仪器**：`probes/p26-cascade-value/{make_shared_prefix.py,run_cascade_ab.sh}`（均含/经 `--selftest`）
**raw**：`/root/ccfa_results/2026-09-14/p26_cascade/`

## 1. 候选的来源与全部亲验事实

来自第三张地图 **E1 段（18 行，轴 = attention backend + compile/cudagraph）**，逐条**我在安装树亲验**：

| 事实 | 出处（亲验） |
|---|---|
| cascade attention **默认关闭**：`disable_cascade_attn: bool = True`，docstring 明写 *"users must opt in to cascade attention by setting this to False"* | `config/model.py:280` |
| **有用 flag**：`--no-disable-cascade-attn` ⇒ argparse 实测 `False`，且 `EngineArgs:1840` 会传进 `ModelConfig`；**端到端验证**：带该 flag 起服务后 `non-default args` 里出现 `'disable_cascade_attn': False` | `engine/arg_utils.py:944`、手工起服务 |
| **FlashInfer 上它被硬关**：`use_cascade_attention()` **无条件 `return False`**（`# TODO: Cascade attention doesn't work, disable it for now`），而 `flash_attn.py:1665` 有真正的启发式 | `v1/attention/backends/flashinfer.py` |
| **启发式在我们负载上返回 True**：24 请求共享 4096 前缀 ⇒ True；对照 <256 前缀 / <8 请求 / SWA / alibi / DCP>1 ⇒ 全 False | `probes/p25-cascade-flashinfer/check_cascade_heuristic.py` |
| **触发前提**：`common_prefix_len = min(common_prefix_len, num_computed_tokens.min())` ⇒ 必须有**已缓存**的公共前缀 ⇒ 实验保持前缀缓存开启 | `gpu_model_runner.py:2764` |
| **S3b**：禁用 PR **#26130 MERGED**；其原因 **#25679**（正文：*"there's a correctness issue with cascade attention on the FlashInfer backend"*）**CLOSED / NOT_PLANNED**；更早给 FlashInfer 加 cascade 的 **#8132 CLOSED 未 merge** ⇒ **没人正在修** | 独立检索 + HTML 直取 |

**⇒ 形状上这是我这条目标里最强的一个候选**：一个**已实现、有文档、可 opt-in、启发式对本负载返回 True** 的共享前缀 prefill 优化，
却在 **sm120 上被自动选中的那个后端**里被硬关，且其阻塞原因是一个**已被放弃**的正确性 bug。

## 2. 实验设计（唯一变量 = 那个 flag，同一后端）

- **同一后端 FLASH_ATTN**（避免把"后端速度差"和"cascade 效果"混在一起）、前缀缓存**保持开启**（`common_prefix_len` 需要已缓存的前缀）。
- **共享前缀 workload**（必须构造，`prompts_4096.jsonl` 是 32 篇不同文章 ⇒ 批内公共前缀≈0）：
  24 条请求**共享前 17,655 字符（≈4096 token）**，每条再带**约 6000 字符（≈1400 token）的唯一后缀**
  —— 后缀必须够长，否则"在共享 KV 上做 prefill"这件事本身没活干，cascade 会被人为测成 0。
- 并发 24（≥8 的阈值）、`--output-len 8`（让指标由 prefill/TTFT 主导）、1 趟预热丢弃 + 3 次测量。

## 3. 结果：**无论吞吐还是 TTFT，都测不出差别**

| 臂 | 3 次暖态 (tok/s) | 暖态 p50 | med TTFT (ms) | TTFT p50 |
|---|---|---|---|---|
| `off`（默认：cascade 关闭） | 561.62 / 551.13 / 544.02 | **551.13** | 168.07 / 173.03 / 178.95 | **173.03** |
| `on`（`--no-disable-cascade-attn`） | 546.40 / 548.58 / 565.44 | **548.58** | 174.02 / 171.68 / 161.99 | **171.68** |
| **Δ** | — | **−0.46%** | — | **−0.8%** |

重复极差：`off` 3.2%、`on` 3.5% ⇒ **Δ 远小于噪声**。

**⇒ 判定：杀（死于量级）。** 并且由此得到一条更强的推论：
**FlashInfer 上那条硬关（`return False`）在我们负载上≈零代价** —— 也就是说，即使把 #25679 那个正确性 bug 修好、
把 cascade 在 FlashInfer 上重新打开，**也拿不到可测收益**。

## 4. 诚实边界（这条"零"该怎么读）

1. 我测的是**"带上 flag 与不带 flag 的端到端差"**，**不是"cascade 确实被执行的差"** —— vLLM **没有**任何 cascade 的日志/指标（我 grep 过 `cascade` 在 `gpu_model_runner.py` / `scheduler.py` 里的 logger 调用：**零**）。
   所以"cascade 真的生效了但收益≈0"与"cascade 因为某个我没看到的内部条件没被选中"这两种解释，**本条无法区分**。
   要区分需要**给引擎加一处 cascade 计数/日志**（引擎改动，属下一轮可做的半径）。
2. 负载是"共享 4096 前缀 + 每请求 1400 token 唯一后缀 + 输出 8"。cascade 的收益**应当随"后缀/前缀比"和"共享请求数"变化**；
   本实验只测了**一个**配置点，**没有扫**这两个维度 ⇒ "在别的比例下是否有收益"**未取证**。
3. 并发上限 24（`--max-num-seqs 32`）；更大批次的收益未测。

**复活条件**：① 先给引擎加 cascade 命中计数，确认它真的被执行；② 扫"后缀长度 × 请求数"；③ 只有当某配置点 Δ ≥3× 噪声时才重开。

## 5. 本轮两次"无效 A/B 被拦下"（仪器纪律的实际价值）

运行器里有一条断言：从服务日志回读 `disable_cascade_attn` 的实际取值，与臂的意图不符就**拒绝该臂、判 A/B 无效**。
它**连续拦下两次**，两次都是**我自己的错**：

1. 第一次：我用 `--hf-overrides '{"disable_cascade_attn": false}'` 当开关。**实测无效** ——
   `hf_overrides` 只喂给 **HF 的 config**（`config/model.py:561-575`），不作用于 vLLM 的 `ModelConfig`；
   进程内实测 `ModelConfig(hf_overrides={"disable_cascade_attn": False}).disable_cascade_attn` **仍是 True**。
   我此前把 `compute_hash()` 的 `ignored_factors` 列表（`model.py:430-444`）**误读成**"可被 hf_overrides 覆盖的字段表"。
2. 第二次：改用正确的 `--no-disable-cascade-attn` 后，**我的两步 sed 替换把这一行整个删掉了** ——
   于是 `on` 臂其实**没带任何 flag**、与 `off` 臂完全相同。断言再次拒绝。

**⇒ 若没有这条断言，我会跑出"cascade 收益 = 0"的结论，而实际上两组配置一模一样 —— 那是一个假结论。**
**"断言产物/断言配置实际生效，而不是断言意图"** 这条纪律在本目标里已经救回三次（前两次见 p20 的零产出与 p21 的冷热不匹配）。
