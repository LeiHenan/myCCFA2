# p38 —— 本轮两条候选的 S3b 判决（均杀）＋ 一处自我更正

## 1. 「把索引构建从 decode 挪到写入时刻」→ **占位，杀**
检索面（外部来源，未逐篇读正文，只作"未找到空白"的证据）：
[PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention](https://ar5iv.labs.arxiv.org/html/2607.24593)
（**直接就是讲稀疏注意力索引开销的**）、[MInference 1.0](https://huggingface.co/papers/2407.02490)、
SGLang [PR #31790](https://git.codeproxy.net/sgl-project/sglang/pull/31790)（合入 Quest GPU kernel 原语）、
Sparse-vLLM 已把 `quest-pattern` 文档化、[统一稀疏注意力与分层内存](https://arxiv-org.ezproxy.obspm.fr/html/2604.26837v1)（专比 per-token decode 延迟）。
叠加用户已排除"单纯 Sparse Attention" ⇒ **杀**。

## 2. 「能量作为一等目标」→ **占位，杀**
[VoltanaLLM](https://ar5iv.labs.arxiv.org/html/2509.04827)（能量+SLO 的分离式服务与频率控制）、
[MLSys 2026 layered prefill](https://browse-export.arxiv.org/pdf/2510.08055)（报 energy benefit）、
以及 **《Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms》**（2608.28044）
—— **最后一篇正是本文 §3 那张表的论文版** ⇒ 连"刻画"这半都已被发表 ⇒ **杀**。

## 3. 能量实测（第一手，Qwen3-4B，T=64，同 prompt，单卡）
| n | 墙钟 | tokens | 平均功率 | J/token |
|---|---|---|---|---|
| 1 | 0.704 s ⚠️预热污染 | 64 | 141.1 W | 1.5530 ⚠️ |
| 2 | 0.494 | 128 | 220.4 | 0.8502 |
| 4 | 0.504 | 256 | 214.2 | 0.4215 |
| 8 | 0.544 | 512 | 232.1 | 0.2468 |
| 16 | 0.623 | 1024 | 271.5 | 0.1653 |
（idle 87.1 W）

**⚠️ 自我更正（同轮内）**：n=1 是首跑、未取中位，被预热污染。干净 A/B 显示 n=1/T=64 的真实 ms/step 是 **7.22**（非 11），
故 n=1 的真实 J/token ≈ **1.02**，**J/token 跨度 ≈ 6.2×（非 9.4×）**。
**我"采样占 50% 关键路径"的假设同时被否**：greedy 7.22 / top_p=1 7.24 / top_p=0.9 7.38 / top_k=50 7.32 ms ⇒ 采样 **≤2%**。

## 4. 本轮的开放观察（未立项，仅记录）
`n=2` 与 `n=16` 构成 Pareto 前沿（0.494 s / 0.623 s），**n=1 被两端同时支配**；
在带宽受限的 decode 里 compute 侧确实几乎全被隐藏（采样 ≤2%）⇒ **没有"SM 空闲可用来做别的"的余量可捡**。
