# S5 预登记（子阶段续）— 语料恢复的成本与等价性（R 阶段）

**候选**：`sglang-draft-corpus` ｜ **时点**：2026-09-13 —— **数据采集前**（与 R 系列实验脚本同 commit 冻结）
**母预登记**：`notes/prereg/sglang-draft-corpus.md`（commit `207de66`）+ 其修订记录（commit `f4f8d70`）。
**本文件是母预登记的续章**，按 E5 条款要求**独立登记**：只回答下面两个新问题，不复用母实验数据做新主张。

## 为什么有这一阶段（问题重述）

语料是**文本**、不是引擎状态（`external_corpus_manager.py` 无 save/restore，但语料**本来就是客户端持有的文本**）。
因此"持久化"的真实对手**不是** `save()`，而是**重新生成那一遍**：
母实验测得「生成 32 条续写」= **25.04 s GPU**，而它占全链路 **73.9%**。
⇒ 本阶段的问题不是"能不能存"，而是：

> **R1**：在新实例上，从**已落盘的语料文本**恢复，是否**比重新生成更便宜**？
> **R2**：恢复出来的语料，效果是否与"重新生成"**等价**？
> **R3**：把 prompt 原文一起持久化（即恢复"累积后 SAM 的全部内容"）是否**更好**？

## 主假设

- **H-R1**：恢复代价（读盘 + 传输 + 服务端 encode/SAM 构建）**< 15%** 的生成代价（即 < 3.8 s vs 25.04 s）。
- **H-R2**：恢复后的首通接受长度与"重新生成"路径**相对差 < 1%**（同一 token 内容 ⇒ 期望**完全一致**）。
- **H-R3**：`30 条 prompt 原文 + 32 条续写` 的合并语料，接受长度 **≥** 仅续写语料（即 prompt 原文在有同分布打底时不再有害）。

## 观测量与判据

| 量 | 定义 | 判据 |
|---|---|---|
| `cost_gen` | 生成 32 条续写的 32 请求总 `e2e_latency`（GPU 忙时） | 参照值（母实验 25.04 s） |
| `cost_restore` | `POST /add_external_corpus` 的客户端墙钟（含服务端 encode + SAM 构建） | **R1 判据：`cost_restore < 0.15 × cost_gen`** |
| `accept_restored` | 恢复后**首通** 32 条的 `spec_accept_length` 均值 | **R2 判据：`|accept_restored − accept_regen| / accept_regen < 1%`** |
| `accept_regen` | 母实验 6.6787 / 本轮 `R_regen` 复测 | 参照值 |
| `accept_combo` | 合并语料恢复后的首通接受长度 | **R3 判据：`accept_combo ≥ accept_restored − 1%`** |
| `accept_xproc` | **另一个新 serve 进程**恢复同一落盘文件后的首通接受长度 | **跨进程判据：与 `accept_restored` 相对差 < 1%** |

## 网格（每臂 = 一个**全新 serve 进程**；串行；同 prompt 集同序）

| 臂 | 语料来源 | 用途 |
|---|---|---|
| `R_gen` | 无 → 跑一遍并落盘续写 | 测 `cost_gen`；产出 `corpus_out.jsonl` |
| `R_cold` | 无 | 冷启动对照（复现 1.8360） |
| `R_regen` | **重新生成**的续写（`build_corpus.py` 从 `R_gen.jsonl`） | `accept_regen`（与母实验 D_treat 独立复测） |
| `R_restore_docs` | **落盘的续写 documents** | `cost_restore` + `accept_restored` |
| `R_restore_combo` | **落盘的 prompt 原文 + 续写** documents | `accept_combo`（H-R3） |
| `R_restore_xproc` | `R_restore_docs` 的**同一个落盘文件**，在**另一个全新进程**再恢复一次 | 跨进程可移植性 |

**固定量**：与母实验完全一致（SGLang 0.5.19、Qwen3-4B、NGRAM γ=7、ctx 8192、`--disable-radix-cache`、
`--max-running-requests 32`、`--speculative-ngram-external-sam-budget 7`、`temperature=0`、`ignore_eos=true`、
`max_new_tokens=128`、`prompts_4096.jsonl` 前 32 条同序、串行、每臂独立进程）。

**逐轴"翻转会不会发生在极值处"论证**：

| 轴 | 范围 | 极值处会翻转吗 |
|---|---|---|
| 恢复类别 | documents（需 encode） ↔ tokens（免 encode） | **不会翻转结论，只影响 `cost_restore` 的绝对量**：tokens 路径更快但要求 payload 更大。**本轮测 documents（现实路径，语料本来就是文本）**，tokens 路径由 `restore_client.py --tokens-out` 支持但**列为未测**。 |
| 语料规模 | 4127（仅续写）↔ 135230（prompt+续写，33×） | **可能翻转 R3**：母实验 T1-b 显示"仅 prompt 131103"有害，但那时**没有同分布打底**。合并语料正是这一格的极值 ⇒ **必须测**（已列入网格）。 |
| 进程 | 同进程 ↔ 跨进程 | **不应翻转**（语料是纯文本，可移植）⇒ 用 `R_restore_xproc` 证伪机会检验。 |
| 恢复次数 | 1 次 | 多次恢复的累积成本未测（明写未测）。 |

## 杀判据（区分"杀"与"补测后再判"）

| 结果 | 处置 |
|---|---|
| **`cost_restore ≥ 0.15 × cost_gen`**（恢复不比生成便宜） | **杀 R 方向**：持久化没有经济价值 ⇒ 本候选到此结题（不再主张生命周期） |
| **`accept_restored` 与 `accept_regen` 相对差 ≥ 1%** | **补测后再判**：先查是否 token 内容不一致（比对 `loaded_token_count`）；若内容一致而接受长度不同 ⇒ 说明存在非确定性，按 3 次重复重判 |
| **`R_cold` 未复现 1.8360** | **本轮作废**（前提失效） |
| **`accept_combo < accept_restored − 1%`** | **不杀方向**，但把 H-R3 判为否证（prompt 原文即使有打底也无益），并写入结论 |
| 任一臂 serve 未就绪 / 装载 `success != true` / message 含 `truncated` | 该臂作废；若关键臂（`R_restore_docs`、`R_gen`）作废 ⇒ 整轮重跑 |

## 噪声与可分辨性

沿用母预登记：接受长度在同进程内噪声 ≤2.2%（稳态）、`temperature=0` + 固定种子下**实测 σ=0.0000**（母实验）。
⇒ `R2` 的 1% 阈值远宽于可分辨下限（0），不存在"噪声淹没"风险；`R1` 的 15% 阈值针对的是**秒级**量（非采样量）。
**k 值**：与母预登记一致 **k = 3**（效应 ≥3× 噪声）。

## 扩展条款（先登记，后触发）

- **E-R1**：若 `cost_restore` 落在 `0.15–0.5 × cost_gen` ⇒ 追加 `tokens` 路径（免 encode）测一次，判定 encode 是否是主成本。
- **E-R2**：若 `accept_combo > accept_restored + 1%` ⇒ 追加"仅 prompt 原文 + 有打底"的对照，确认增益来自 prompt 还是来自语料总量。
- **E-R3（明写不做）**：**不**做跨机型/跨副本网络的真实传输（本轮是同一台机器上的第二个进程）；
  **不**做引擎内序列化（`NgramCorpus` 无该能力，属另一半径）。

## 修订记录

| 日期 | 变更 | 时点 |
|---|---|---|
| 2026-09-13 | 首次登记 R1/R2/R3、判据、6 臂网格、杀判据、E-R1–E-R3 | **数据采集前** |
