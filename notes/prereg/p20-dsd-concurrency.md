# 预登记 — DSD（动态投机调度）在**并发 + 前缀缓存开启**下的代价

**登记时点**：2026-09-14，**数据采集前** ｜ 目标 `goal-a729588f`（三方向）｜ 成本上限 ≤0.3 GPU·h

## 为什么是这一格（来自投机解码地图 D4/B1.6）

上游自己的 merged benchmark（PR #45953）**只测 BS1 且关闭前缀缓存**（命令里全是 `--no-enable-prefix-caching`，
只报 base/EAGLE3 的 TPOT `2.91/2.97/2.91` ms）⇒ 地图原话：*"DSD at concurrency on this class is essentially uncharacterized."*
同时有多条 **OPEN** 报告称：
- vLLM #48494：*"[DSD] incur[s] a **12–25% throughput penalty** vs no-spec … even with an all-K=0 table
  producing nearly zero draft tokens"* + *"**The presence of the batch-size table is the trigger**; K=0 tiers are not required."*
- vLLM #49548：*"catastrophic aggregate-throughput collapse under concurrency"*（8 并发 ~232 → 24–157 tok/s）

## 主假设（可证伪）

> **H1**：在**并发 8 + 前缀缓存开启**下，**"表的存在"本身**（而非"表里 K 的值"）会造成可测的吞吐下降。
> 即：`all-K=0 表` 与 `静态 K=0`（等价于不投机）在**相同请求集**下吞吐显著不同。

**若 H1 成立** ⇒ 这是一个**结构性**开销（调度器/图/元数据路径），不是调参问题，且**上游自己的基准没覆盖**。
**若 H1 不成立**（两者在噪声内相等）⇒ 我方负载上不存在该开销，**本条按否证记录**。

## 观测轴与判据（数字写死）

**固定**：vLLM 0.29.0 / Qwen3-4B / sm120 / `--max-model-len 8192` / `--gpu-memory-utilization 0.55` /
**并发 8**（客户端并发，`vllm bench serve` 同一 prompt 集与顺序）/ 输出长度固定（`ignore_eos`）/
**前缀缓存开启**（与上游基准相反，这是本格的条件 C）/ 每格独立 serve / 每格重复 **3** 次。

**六臂（并发全部 = 8；drafter 统一用 dflash2 保证跨臂唯一变量是"表"）**：

| 臂 | 投机配置 | 意图 |
|---|---|---|
| A1 | 无 | 绝对基线 |
| A2 | dflash2, `num_speculative_tokens=3`，**无表** | 静态基线 |
| A3 | dflash2, `=0`，**无表** | 静态 K=0 → **应与 A1 等价**（H1 的对照零点） |
| A4 | dflash2, `=0`，**表 `[[1,8192,0]]`** | **H1 的核心测试**：表存在但全 K=0 |
| A5 | dflash2, `=3`，**表 `[[1,3,3],[4,8192,0]]`** | 生产式跳变表 |
| A6 | dflash2, `=3`，**表 `[[1,8192,3]]`** | 表 = 常数 K |

**主判据 P1（H1）**：`throughput(A3) − throughput(A4)` 的**相对差** ≥ **8%** 且方向为 A4 更慢，
且超过噪声（见 §噪声）；符号相反或 <8% ⇒ **H1 否证**。
**副判据 P2**：`throughput(A2) / throughput(A1)`（投机净收益）；若 <1 则记录为"投不划算"的独立复现。
**副判据 P3**：A2 的 `accept_len`（由 `/metrics` 的 `vllm:spec_decode*_total` **做差**得到）应 ≈2–3；
A3/A4 的 `num_drafts` 增量应为 **0**（证明"K=0 表确实没出 draft"——否则 H1 的对照无效）。
**副判据 P4（口径）**：A1 与 A3 的吞吐应在噪声内相等（若无 ⇒ 说明 K=0 路径本身有额外开销，须单列）。

## 噪声与可分辨性

同格 3 次重复，报 p50 与极差；**k = 3**（效应 ≥3× 重复极差才称效应）。
若噪声 >8% ⇒ 提高重复到 5 次（扩展条款 E1），并如实标注。

## 杀判据

| 结果 | 处置 |
|---|---|
| **P3 失败**（A3/A4 的 `num_drafts` 增量 ≠ 0） | **本轮作废**（对照不成立：表未生效） |
| **P4 失败**且差 ≥8% | 记录为**另一条**结构性开销（K=0 路径），H1 改用 A3 作对照重判 |
| **P1 <8% 或符号相反** | **杀 H1**（记为否证），不追加实验 |
| 任一臂 serve 未就绪 | 该臂作废；关键臂（A3/A4）缺 ⇒ 整轮重跑 |

## 扩展条款

- **E1**：噪声 >8% ⇒ 每格重复提到 5 次。
- **E2**：若 P1 成立 ⇒ 追加**并发 {1, 32}** 两点，判断开销是否随并发变化（**先登记后触发**）。
- **E3（明写不做）**：不做多卡/DP（上游文档：DSD 与 DP 不兼容，会自动关闭表）；不做 Eagle/DFlash 以外的方法（文档：只测过 Eagle/Eagle-3/DFlash）。

## 环境前提

驱动 580 / sm120；`VLLM_USE_V2_MODEL_RUNNER` 不设（auto ⇒ 实测选 V2）；`--enforce-eager`（文档：Full Cudagraph 只在 V2 上工作，eager 下无图差异）；
前缀缓存**开启**（本格的条件 C）；prompt 集固定顺序；`temperature=0`。

## 修订记录

| 日期 | 变更 | 时点 |
|---|---|---|
| 2026-09-14 | 首次登记：H1、P1–P4、A1–A6 六臂、杀判据、E1–E3 | **数据采集前** |
| 2026-09-14 | **采数前修订 #1**：A3 由"静态 K=0"改为"静态 K=1"；**原 P1 作废**，拆为 P1a/P1b；P4 改口径；新增 P5。另更正"`--result-filename` 不存在"这一**错误结论** | **数据采集前**（尚无任何可用吞吐数据） |

## 采数前修订 #1（2026-09-14）—— 原 P1 设计不可实现，改用 K 匹配对照

**触发事由：两次"零产出"网格。** 网格 #1（`p20_dsd_console.log`）与 #2（`p20_dsd_console2.log`）六臂**全部 serve 就绪**
（`Using V2 Model Runner`），但 **18 份 `.bench.log` 全为 0 字节、0 份结果 JSON**。根因**已定位并证实**：

- 运行器调用 `python -m vllm.benchmarks.serve`；而 `vllm/benchmarks/serve.py` 在本环境的 0.29.0 里**没有
  `if __name__ == "__main__":` 块**（`grep -n "__main__"` 无输出）⇒ **import 后静默退出、退出码 0**。
  实测：`$PY -m vllm.benchmarks.serve --help` **无任何输出且 `EXIT=0`**。
  ⇒ `|| echo "bench 失败"` **永不触发**（退出码是 0），失败被完全吞掉。
  正确入口：`$PY -m vllm.entrypoints.cli.main bench serve`（或 `$VENV/bin/vllm bench serve`）。
- **连带更正**：我此前记录"`--result-filename` 不存在"**是错的**。该结论由**同一个静默退出**推出（命令从未真正运行，
  任何参数都"失败"）。实测 `bench serve --help=all` 第 185 行确有 `--result-filename RESULT_FILENAME`。已启用。

**由实现约束引出的设计变更**：顶层 `num_speculative_tokens: 0` 被 schema 拒绝
（`Input should be greater than 0`；K=0 **只在表内**合法）⇒ **原 A3（"静态 K=0 且无表"）无法构造**，
而它正是原 **P1** 的对照零点。故：

| 原判据 | 处置 | 替代 |
|---|---|---|
| **P1**：`thr(A3_static_k0) − thr(A4_table_allk0)` | **作废**（A3 不可构造） | 见 **P1a / P1b** |
| **P4**：`thr(A1) ≈ thr(A3)`（K=0 路径开销） | 改口径 | A3 现为 K=1 ⇒ P4 变为"**最小投机（K=1）净效应**" |

**新主判据（仍对准 H1"表的存在本身"）**：

- **P1a（K 匹配对照，最干净）**：`thr(A2_static_k3)` vs `thr(A6_table_const3)` —— **两者 K 都是 3**，
  唯一差异是**有没有表**。相对差 ≥8% 且 A6 更慢 ⇒ 表的存在本身有代价。
- **P1b（零草稿表 vs 不投机）**：`thr(A1_nospec)` vs `thr(A4_table_allk0)` —— 表存在但全 K=0（应零草稿）vs 完全不投机。
- **P5（新增，表分层是否真生效的可证伪预测）**：A5 的表是 `[[1,3,3],[4,8192,0]]`，并发 8 时解码批应落在 **bs≥4 档 ⇒ K=0**
  ⇒ 预测 **A5 的 `num_drafts` 增量 ≈ 0**，吞吐应贴近 A1/A4 而非 A2/A6。若 A5 反而 ≈ A2 ⇒ **表未被真正查询**（对照不成立，本轮作废）。

**主指标**：`output_throughput`。**逐次重复上报**（rep1 = 冷缓存；rep2–3 = 前缀已缓存 ⇒ 本格条件 C 的稳态），
主判据取 **rep2–3 的 p50**，rep1 单列（条件 C 下冷/热必须分开看，否则把缓存效应误读成臂间差异）。

**判据不变部分**：P2（`thr(A2)/thr(A1)` 投机净收益）、P3（A4 的 `num_drafts` 增量 = 0，证明"全 K=0 表确实零草稿"）、
噪声 k=3（效应 ≥3× 重复极差）、杀判据（P3 失败 ⇒ 本轮作废；P1a/P1b <8% 或反向 ⇒ 杀 H1）。

**运行器加固（防止再次零产出）**：每次 bench 后**断言**结果 JSON 存在且非空、能解析出 `output_throughput`，
并把数字写入 TSV；serve 用 `setsid` 起、按**进程组**回收，回收后轮询 `nvidia-smi` 直到 ~0 MiB 才进下一臂
（网格 #2 结束后曾出现一个**孤儿 `VLLM::EngineCore`**（PPID=1、31.8 GB，来自 `spec-c6/venv` 0.28.0），已手工清理）。

**采数前新证实的环境事实**：`enable_prefix_caching` 在 0.29.0 的默认值就是 **`True`**（`config/cache.py:138`），
且 A4 的 EngineCore 配置转储里明确出现 `enable_prefix_caching=True, enable_chunked_prefill=True`
⇒ **条件 C（前缀缓存开启）成立且可审计**（不是"我没传这个参数所以大概开着"）。
另：A4（全 K=0 表）的配置被完整接受并解析（`Resolved architecture: DFlash2DraftModel`，
`speculative_config=SpeculativeConfig(method='dflash', ..., num_spec_tokens=1)`），**没有任何关于表的告警或拒绝**。
per-position 计数器的**键集合**随 K 变化（K=1 臂键为 `{0}`；K=3 臂键为 `{0,1,2}`）⇒ 投机配置按臂生效的**独立旁证**。
