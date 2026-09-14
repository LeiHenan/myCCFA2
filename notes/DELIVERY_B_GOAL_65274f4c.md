# (B) 交付：单卡机制创新搜索 —— 为何全部被否

**目标**：`goal-65274f4c` ｜ **轮次**：9/200 ｜ **机器**：RTX PRO 6000 Blackwell 95.6 GB / sm120 / driver 580.82.09 / vLLM 0.29.0 / Qwen3-4B
**GPU 花费**：≈2.5 GPU·h（上限 8/轮）
**结论**：**18 条自建机制，0 存活。** 每条都死在**具名的占位者**或**具名的闸门**上，无一条死因是"没测出来"。

---

## §1 判死台账（每条：机制 · 死因 · 决定性证据）

| # | 机制 | 死因 | 决定性证据 |
|---|---|---|---|
| 1 | 并发搜索式推理的 KV 准入/跨程序复用（候选 A） | **实测否证** | `beam_search` 包装层 2.75× 贪心、2.60× 于原生 n=4；而引擎本身 4 个独立请求只要 **1.05×** ⇒ 属引擎 API 实现缺陷 |
| 2 | 把稀疏索引构建从 decode 挪到写入时刻 | 占位 | PIVOT (2607.24593，**标题即"索引开销"**)、MInference 1.0、SGLang #31790 合入 Quest kernel 原语 |
| 3 | 能量作为一等调度目标 | 占位 | VoltanaLLM (2509.04827)、MLSys'26 layered prefill、**专门的能量刻画论文 2608.28044**（连我那张表都有论文版） |
| 4 | 投机天花板 = 带宽/算力之比、与接受率无关 | 占位 | **MoESD (NeurIPS'25 spotlight, 2505.19645)** + "target efficiency" |
| 5 | cudagraph 捕获边界填充浪费 | **实测否证** | 捕获尺寸 32 以上每 8 一档（非 2 的幂）；n=32→33 为 14.280→14.277 ms，无跳变 |
| 6 | 算力受限区间感知 | 占位 | **TileSparse (ICML 2026)** *"Arithmetic-Intensity-Aware Sparse Attention for **Compute-Bound** LLM Decoding"* |
| 7 | 跨精度 KV 复用 | 占位/在修 | omlx #2487（缓存签名含 TurboQuant 深度）、#2272（按 payload 格式重建前缀块） |
| 8 | 带宽感知 KV 淘汰 | **算术否证** | 每 block 每步省下的带宽是**常数** ⇒ 判据退化为"距下次使用多少步" ⇒ 正是 LRU/Belady 已优化的 |
| 9 | prefill/decode 阶段的优化选择 | 占位 | DuetServe (2511.04791, *"attention-aware roofline model"* 选 SM 切分)、Nexus (2507.06608)、DOPD (2511.20982) |
| 10 | 收割低批空闲张量核 | 占位 | Lookahead Decoding (2402.02057) + **vLLM 自己发货的 Adaptive Verification**（*"The crossover moves with load…"*，K 表末档 `[129,512,0]`） |
| 11 | 不靠草稿减步数 | 占位 | Lookahead Decoding 摘要逐字：*"without needing auxiliary models or data stores… reduce the number of total decoding steps"*；§3.3 *"Verify in The Same Step"* |
| 12 | 输出长度作为控制变量 | 占位 | LASER (2606.31580)：*"treats reasoning depth itself as a controllable serving variable"* |
| 13 | 准入：长度与 KV 共同未知 | 占位 | 2504.11320：*"a job's memory requirement is not fixed at admission"* |
| 14 | 负载定价的 KV 驱逐（含跨翻转点重算价） | 占位 | **KVLearn (SYSTOR '26)**：`CARS(b)=P̂(b)(R(b)−T(b))−U(b,Δt)`，`R(b)` 按 `L×` 分段 |
| 15 | 投机 × 能量 | 占位 | ACM battery-aware speculative decoding (10.1145/3832810.3832825) |
| 16 | 调度 × 量化/精度 | 占位 | FineServe (2509.06261) |
| 17 | 稀疏 × 调度/内存 | 占位 | HiSparse (2608.07009)、ICML'26 stochastic sparse attention |
| 18 | 在线硬件自适应（用户清单 #10） | 占位 | Autopoiesis (2604.07144)：*"Continuous adaptation to shifting runtime trade-offs is necessary"* |
| — | **按会话驻留 KV** | **双杀** | ① 可达性：session-LRU 比 LRU **更差**（补缺口 −2%~−25%）；② 占位：CacheWise (2606.16824) |
| — | **用真实轨迹改进前缀缓存（整族）** | **实测 §0.6 判死** | 真实速率下未命中 prefill 仅 **13.3 ms vs e2e 1061 ms = 1.2% < 2%** |

## §2 本目标真正留下的资产（比任何单条候选耐用）

**仪器（这是 9 轮里唯一从 0 变成可用的东西）**
- **真机回放 + 专卡同格噪声地板 0.4–1.3%**（对比 schoolserver 共享机的 132%）⇒ **>4% 的效应即可判定**。
  `probes/p43-replay/{replay,replay2,replay3}.py`
- **结构忠实的轨迹回放**（本轮自己发明）：按 `hash_ids` 逐块把块 id 展开成 16 个 token、
  顺序拼接、以 token id 列表直发 ⇒ **轨迹里共享的块在 prompt 里也共享**。
  实测共享块 37,481/79,863 = 53%，与零卡仿真一致。**没有这一步，回放测不出任何缓存效应。**
- **每臂指标取差**（`/metrics` 是累计值，同进程多臂会混在一起——这是上轮踩的坑）。

**数据**
- Alibaba `qwen-bailian-usagetraces-anon` 四条真实生产轨迹，**269,681 请求**，含
  `hash_ids` 块级前缀哈希、`parent_chat_id` 会话链、`type`。`data/traces/*.jsonl`

**零卡仿真工具**
- `cache_optimal.py`（LRU/LFU/ARC/**Belady** 精确命中率）、`policy_gap.py`（可达性）、
  `session_policy.py`（会话感知）、`bailian_stats.py`、`session_analysis.py`

**本卡代价模型（第一手）**
- prefill 30,700 tok/s；decode 单步 n=1 时 7.30 ms、n=64 时 18.85 ms；
- 每步字节只涨 36% 而时间涨 2.58×，多出的是**随批线性的权重算力项**（0.158 ms/序列 ≈ 50 TFLOPS）
  ⇒ **带宽→算力翻转点在 batch ≈ 35**；
- **实测达到带宽 1,442–1,515 GB/s（名义 1792 的 81–85%）**；
- cudagraph 省下 ≈4.3–4.7 ms/步的发射开销（已发货，无可捡余量）。

## §3 为什么是 0/18（结构判断，附元观察）

1. **可推导的机制都已被拿走。** 18 条里 15 条死于具名占位者，且**多数是 2025–2026 的会议论文或引擎已发货特性**。
2. **元观察（值得沿用）**：这些框架**被"引擎 release notes + 系统会议（SYSTOR/NSDI/ICML/MLSys）"关掉的速度快于被 arXiv 关掉**。
   以后筛新颖性**不能只搜 arXiv**。本目标内：候选 10/11 由 vLLM 文档关闭、候选 14 由 SYSTOR'26 关闭。
3. **另 3 条死在我自己的算术或实测上**（#1 #5 #8），**另有 2 条死在我自己的 §0.6 实测上**——这不是浪费：
   它们把"这台卡 + 这类负载上什么不值钱"钉死了。
4. **用户给的分母（公开真实轨迹）已用满**，产出四个真实发现（前缀复用 45–64%、显存超额认购 2–9×、
   LRU-vs-Belady 缺口只在大缓存结论外成立、会话感知驱逐为负），**但每一条都在可达性或占位上出局**。

## §4 复活条件（换哪个分母/门槛就会活）

| 换什么 | 会让什么活 |
|---|---|
| **prefill 占比高的负载**（输入 ≫ 输出：长文档 RAG、代码库问答） | 缓存/prefill 整族（#9 #14，以及"LRU 在大缓存近最优"的反例） |
| **带宽/算力比更低的卡**（decode 更贵） | 同上，且 §0.6 的分母变大 |
| **私有业务轨迹（哪怕只有统计量）** | 把任一被杀机制变成**负载特异系统论文**而非重发通用结论 |
| **真实故障/事故记录** | 本会话唯一高产的工作类型就是"推翻既有承重数字"（p30/p33/p35 都是），事故记录最容易长出这种题 |
| **门槛降为"复现+改正+工程落地"** | 立刻可做：vLLM 0.29 新发货的 ARC 在四条真实轨迹上**不优于 LRU**（thinking 10% 处 ARC 51.71 < LRU 52.16）；LFU 在 traceB 上领先两者 ~9 点；"LRU 近最优"只在**大缓存**成立 |
| **SGLang 面**（两台机器都没装） | 三张地图里依赖 SGLang 的 **77 格**从未被验证过 |

## §5 本目标内的自我更正（按重要性）

1. **§0.6 里用 batch=1 的 decode 速率**——**同一个错误犯了两次**（p40 会话驻留、p41 缓存缺口）。
   该负载显存超额认购 2–9×，必然高并发；用 batch=1 会**系统性低估 prefill 占比约 10 倍**。
2. **Belady 实现 bug**——首版只在 `nxt[i] ≤ n` 时入堆，"以后再也不用的块"永不淘汰 ⇒ 给出 **Belady < LRU 的物理不可能结果**。
3. **轨迹回放用固定文本造 prompt** ⇒ 所有请求前缀相同，**完全没复现轨迹的前缀共享结构**，测了等于没测。
4. **把到达率降采样 7×** ⇒ 把卡变成 99% 空闲；真实速率（23.88 req/s）本来就装得下。
5. **session-LRU 两个 bug**（堆键不匹配导致驱逐从不发生＝缓存无界；弹出后未放回导致堆抽干）——
   两者分别被"物理不可能"与**容量断言**抓住。**工具的正确性必须由不变式守，不能由"跑完了"守。**
6. **用户清单 #1（Joint Speculation–KV–Scheduler）在我方证据里是最拥挤的一条**，与其推荐位次相反；
   其切片 S-B 已被零卡可分离性闸门杀掉。

## §6 下一轮测绘结论

- **先查引擎 release notes 与系统会议**，再查 arXiv（见 §3.2）。
- **SGLang 面完全未查**（77 格）；**TensorRT-LLM / llama.cpp 未搜**。
- **`api.github.com` 限流为 0** 在本环境无法绕过 ⇒ vLLM 源码只能从**已装树**读（本目标一直如此做，可行）。
- **本卡是可用的**：回放仪器已建成、噪声 0.4–1.3%、真实轨迹已就位。**缺的只有"一个没被占的问题"。**
