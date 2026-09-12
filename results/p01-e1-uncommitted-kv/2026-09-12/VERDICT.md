# E1 结论（#1 未提交投机 KV 探针）—— 2026-09-12

**机器**：RTX PRO 6000 Blackwell 96 GB ｜ **引擎**：vLLM 0.29.0（main 附近）｜ **驱动**：580.82.09
**配置**：`Qwen3-4B` + DFlash2 drafter，`γ=3`，`ctx{4096,32768} × 并发{1,8} × 3 reps`（主判据 12 格）+ 正确口径复测 4 格，真实文本 prompt（WikiText），`KV=bfloat16`，`--enforce-eager`

---

## 一、主判据：`#1` **判死**（有效测量）

| 指标 | 实测 p95 | 解析对照 |
|---|---|---|
| `R_byte` = rejected / committed | **0.0716% – 0.0723%** | `γ/ctx = 3/4112 = 0.0727%` ✔ 吻合 |

- 12 格（γ=3 × ctx{4k,32k} × 并发{1,8} × 3 reps）**全部落在 0.0716%–0.0723%**，两次独立运行**逐位一致**。
- 预登记杀判据是"全部 HBM 可行点上 p95 < 5%"：实测值比门槛低 **70×**；且整网格内解析上界 `γ/ctx ≤ 7/4096 = 0.17%` 也远低于 5%
  ⇒ **`#1`（"未提交投机 KV"）按预登记判死**。
- 机制原因（已在 prereg 中作为源码事实登记）：被拒 token 的 KV 在同一步就被 `num_computed_tokens -= num_rejected` 回滚 ⇒ 驻留步数 `H ≈ 0` ⇒ 真实占用只有 `γ/ctx` 量级。

## 二、次要终点 D3：**不成立**（原触发是**指标定义错误**，已更正后重测）

1. **旧口径的假信号**：`analyze.py` 原以 `R_reserve = 新分配块数 × block_size / committed`。
   但 vLLM `KVCacheManager.allocate_slots` 的返回值语义是 **"A list of new allocated blocks"**（**新增**块，非总块；原始记录印证：预填充 `new_tokens=4104/blocks=257/ctx=0`，解码步 `blocks=0或1`）。
   ⇒ 旧 `R_reserve ≈ block_size/ctx`，而 `R_byte ≈ γ/ctx`，二者之比**恒等于 `block_size/γ`**（γ<8 时必然 >2）
   ⇒ 预登记的 D3 触发条件被**结构性假信号**点亮，**不能**作为"存在保守预留"的证据。
2. **更正后重测**（Hook B 增记 `total_blocks = KVCacheManager.get_block_ids(req)`，即请求**持有**的全部块，含 lookahead 预留）：

| 格 | `R_byte` p95 | **`R_reserve_total / committed` p95** |
|---|---|---|
| ctx=4096, bs=1 | 0.0723% | **1.00533** |
| ctx=4096, bs=8 | 0.0723% | **1.00534** |
| ctx=32768, bs=1 | 0.0720% | **1.00529** |
| ctx=32768, bs=8 | 0.0716% | **1.00526** |

   ⇒ KV 持有容量只比已提交 token 多 **0.53%**（就是最后一个 block 的取整），跨 ctx 与并发一致
   ⇒ **不存在可消除的保守预留 ⇒ D3 支线关闭**（无 headroom，不值得立项）。

## 三、判据逐条对照（`notes/prereg/p01-e1-uncommitted-kv.md`）

- [x] 全部 HBM 可行点 p95 < 5% ⇒ **杀** —— 命中（0.072%，70× 余量）
- [ ] 某可行点 p95 ≥ 10% ⇒ 活 —— 未命中
- [x] D3 次要终点（`R_reserve/R_byte > 2`）—— 按**原口径**命中，但经查为指标定义假象；按**正当口径**（总持有/已提交 = 1.0053）⇒ **不成立**

## 四、产物与可复用手艺

- 正确口径的 4 格：`g3_ctx{4096,32768}_bs{1,8}_r1.analyze.log`（含 `R_byte` 与 `R_reserve_total_over_committed`）
- 仪器：`probes/p01-e1-uncommitted-kv/instrument/{e1_patch.py,analyze.py}`（Hook A/B 锚点在 vLLM 0.29.0 上核对通过；Hook B 已自包含并加自检）
- 执行器：`probes/p01-e1-uncommitted-kv/run_e1.sh`（含插桩自检；`DRY=1` 预览）
- **教训（写进工具的注释里）**：① Hook 的 helper 必须与被 wrap 的模块同名字空间（跨模块会 NameError 且被 `except: pass` 吞掉）；② 指标的"返回值语义"必须先读源码（`allocate_slots` 返回新增块）再定义口径，否则会得到结构性假信号。
