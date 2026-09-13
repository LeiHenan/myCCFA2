# p12-draft-corpus 结果 — 冷启动罚金与「预置语料」的可回收性

**判定**：**决策级 Go（条件性）** —— `recovery = 86.2% ≥ 80%`（预登记 C2），C0/C1/C3/C4 全过。
**日期**：2026-09-13 ｜ **引擎**：SGLang 0.5.19 ｜ **模型**：Qwen3-4B（bf16）｜ **卡**：RTX PRO 6000 Blackwell（sm120, driver 580.82.09）
**预登记**：`notes/prereg/sglang-draft-corpus.md`（冻结于 commit `207de66`，**采数前**）
**成本**：**≈0.75 GPU·h**（预算 ≤8 GPU·h）
**原始数据（仓库外）**：`/root/ccfa_results/2026-09-13/p12_{smoke,corpus,budget,outcorpus,validate}/`

## 1. 主表：唯一自变量 = 语料（全部串行、同 prompt 集同序、radix cache 关闭）

| 臂 | serve 进程 | `sam_budget` | 语料 | 语料 token | 首通 accept | wall/32 | 逐请求范围 |
|---|---|---|---|---|---|---|---|
| `A_control` ×3 | 3（独立） | 默认 **0** | 无 | 0 | **1.8360** | 25.04 s | [1.23, 3.05] |
| `B_treat` ×3 | 3（独立） | 默认 **0** | prompt 原文 | 131103 | **1.8360** | 24.94 s | 同左 |
| `C_steady`（第 4 遍） | 1 | 默认 0 | 引擎自积累 | — | **7.7696** | 8.69 s | — |
| `D_gen` | 1 | 7 | 无（**生成语料用**） | 0 | 1.8360 | 24.98 s | — |
| `D_ctrl` | 1 | 7 | 无 | 0 | **1.8360** | 24.98 s | [1.23, 3.05] |
| **`D_treat`** | 1 | **7** | **模型续写（output-as-corpus）** | **4127** | **6.6787** | **10.22 s** | [2.29, 8.00] |
| `V_treat_r2` | 1 | 7 | 同上 | 4127 | **6.6787** | 10.26 s | 同左 |
| `V_treat_r3` | 1 | 7 | 同上 | 4127 | **6.6787** | 10.23 s | 同左 |
| `B0`（预算极值） | 1 | **0** | prompt 原文 | 131103 | 1.8360 | 25.07 s | — |
| `B7`（预算极值） | 1 | **7** | prompt 原文 | 131103 | **1.5337** | 28.57 s | — |
| `V_budget3` | 1 | 3 | 模型续写 | 4127 | **3.7576** | 14.11 s | — |

## 2. 判据对照（预登记 C0–C4）

| 判据 | 阈值 | 实测 | 结果 |
|---|---|---|---|
| C0 现象可复现 | `control ≤ 0.5 × steady`，对照同向 | 1.8360 ≤ 3.8848；**4/4 独立进程 = 1.8360**（罚金 **76.4%**） | ✅ |
| C1 口径自洽 | 违例 0、`cached_tokens` 全 0 | 704 行：`accept ≤ 8` 违例 **0**、`rate` 越界 **0**、`cached_tokens ≠ 0` **0** | ✅ |
| C2 可回收性 | `recovery ≥ 80%` | **86.2%** | ✅ |
| C3 可分辨性 | 效应 ≥ 3σ（k=3） | **+4.8427 token**；σ = 0.0000（A 臂 4 点、处理臂 3 点）；两臂**区间不重叠** | ✅ |
| C4 wall 方向（辅助） | 与 C2 同向 | 24.98 s → 10.22 s（**2.45×**） | ✅ |

## 3. 关键派生量

| 量 | 值 |
|---|---|
| 冷启动罚金 | **76.4%**（accept 1.8360 vs 稳态 7.7696） |
| 接受长度增益 | **1.8360 → 6.6787 = 3.64×**（**+263.8%**） |
| `recovery` | **86.2%**（相对"引擎自带会话内累积"这一已 ship 控制器） |
| 冷窗口 wall | **24.98 → 10.22 s = 2.45×** |
| 全链路 wall（生成语料 1 遍 + 恢复后首通 + 重放） | **33.89 → 8.69 s = 3.90×** |
| oracle 上界复核 | 上界 7.7827 / 实测 6.6787 = 上界的 **85.8%**（**未越界** ⇒ 上界成立） |
| `spec_verify_ct`（每 128 output token） | ≈70（冷）→ ≈19.2（恢复）= **3.64×** |

## 4. 两条结构性发现（比性能数字更有迁移价值）

1. **`sam_budget` 默认 0 ⇒ 外置语料静默失效**：`kernels/jit/csrc/ngram_corpus/ngram.cpp:173` 取
   `min(external_sam_budget, total_draft_token_num)`；而三条校验（`arg_groups/speculative_hook.py:867-880`）
   **只在设了 `--…-corpus-path` 时触发** ⇒ `POST /add_external_corpus` 可以**成功装载 131103 token 却完全不生效，且无任何警告**
   （本实验 9 次装载 `success:true`，其中 6 次实际无效）。**这也解释了本仓库上一轮 decision #94 的"预置语料看不出效果"**（当时未打开配额）。
2. **语料语义决定收益符号**：同为 budget=7，**prompt 原文 −16.5%** vs **模型续写 +263.8%**。
   ⇒ 外置语料**不是免费杠杆**：语料必须落在目标模型自己的续写分布里。

## 5. 偏差与失败（如实记录）

| 事件 | 处置 |
|---|---|
| 预登记"固定量"漏写 `--speculative-ngram-external-sam-budget` ⇒ 主实验 B 臂干预被静默禁用（`recovery = 0%`） | 登记为**数据采集后**修订（commit `f4f8d70`）；该轮**不计入否证**；补测后重判 |
| `FileNotFoundError: 'ninja'`（venv `bin/` 不在 PATH，flashinfer JIT 失败） | 冒烟阶段抓到并修（`export PATH="$VENV/bin:$PATH"`） |
| `/get_internal_state` 在 0.5.19 上 **404** ⇒ 累计计数器路径不可用 | 改用响应体 + 客户端 wall（`/metrics` 的 batch 级 gauge 未纳入判定）；已在 S4 登记 |
| σ = 0.0000（`temperature=0` + 固定种子 ⇒ 完全确定性） | 已在 S6 说明"σ=0 是设置性质，非效应无穷大"；正面证据为 360× 同进程噪声余量与区间不重叠 |

## 6. 复现

```bash
# 1) 同步（不要把结果放 /root/myCCFA）
git archive --format=tar HEAD | ssh -p 11640 root@connect.westd.seetacloud.com \
  'rm -rf /root/myCCFA && mkdir -p /root/myCCFA && tar -x -C /root/myCCFA'
# 2) 一轮完整复现（tmux 内，含生成语料 → 装载 → 恢复测量）
tmux new-session -d -s p12 -c /root/myCCFA \
  "bash /root/myCCFA/probes/p12-draft-corpus/run_output_corpus.sh 2>&1 | tee /root/ccfa_results/repro.log"
# 3) 离线判据重算（可对任意一臂目录）
/root/autodl-tmp/venvs/sglang/bin/python probes/p12-draft-corpus/verdict.py --dir <数据目录>
# 4) 零 GPU 自测（5 个工具全含 --selftest）
for t in probe analyze verdict corpus_client build_corpus; do
  python3 probes/p12-draft-corpus/$t.py --selftest; done
```

## 7. 适用范围（不外推）

仅 **Qwen3-4B + NGRAM（γ=7）+ ctx 4096 + 串行 + radix cache 关闭 + 同模型续写语料**。
**未测**：并发 > 1、ctx ≠ 4096、其他模型/家族、EAGLE/MTP/dflash、外部分布/跨模型语料、语料规模中间点、
配额 1/2/4/5/6、**真正的跨进程/跨副本状态恢复**（下一阶段）、多卡/跨机。

---

# R 阶段（同日晚追加）：语料恢复的成本与等价性

**预登记**：`notes/prereg/sglang-draft-corpus-r.md`（采数前冻结，commit `3c590b4`）｜**成本 ≈0.15 GPU·h**
**原始数据**：`/root/ccfa_results/2026-09-13/p12_restore/`（`R_*.jsonl`、`*.serve.log`、`restore_cost.jsonl`）
**判定**：**R1 ✅ / R2 ✅ / R3 ✅ / 跨进程 ✅**（冷启动对照逐位复现 1.836007）

| 臂 | 首通 accept | 语料 | cost_restore |
|---|---|---|---|
| `R_gen` / `R_cold` | 1.836007 | 无 | — （`cost_gen` = **25.058 s**） |
| `R_regen` | 6.678743 | 重新生成 4127 tok | — |
| `R_restore_docs` | **6.678743** | 落盘 documents 4127 tok | **0.0539 s** |
| `R_restore_combo` | **6.946652** | prompt 原文 + 续写 = 135231 tok | **0.4067 s** |
| `R_restore_xproc` | **6.678743** | 同一落盘文件，**另一全新进程** | **0.0625 s** |

**派生量**：恢复/生成成本比 = **1.62%**（**60×** 便宜）；恢复等价性 = **0.000%**；
合并语料增益 = **+4.0%**；恢复后达稳态 **93.0%**（母阶段 86.2%）。

**源码级边界**：`NgramCorpus` 的 14 个方法中**无任何导出/序列化接口**
（`srt/speculative/cpp_ngram/ngram_corpus.py:15-153`）⇒ **trie 不可导出，跨副本只能共享语料文本**。

**未测**：跨机/网络传输、并发 >1、真实流量下的整机收益（"从 0 学" vs "继承 warm 副本"）。
