# C-1b 判定：**hybrid（GDN/线性注意力）目标上投机解码是单调负收益**

**日期**：2026-09-14 ｜ **平台**：RTX 4080 SUPER 32GB，vLLM 0.29.0
**目标模型**：dense `Qwen3-4B` vs **hybrid** `Qwen3.5-4B`（24 linear_attention + 8 full_attention）
**Drafter**：ngram（prompt-lookup）｜ **语料**：真实文本多轮对话，1024-token 共享前缀，48 请求，输出长度钉死
**关联**：本结论源于 C-1 证伪时留下的**最大未解混淆**（"上游报的是 hybrid，我测的是 dense"）

---

## 0. 一句话

**在 hybrid 目标上，投机解码在 K=1 就已造成 2.44× 的净损失，且随 K 单调恶化到 4.55×。**
dense 目标上同一配置**不呈单调、且可正可负**。**两族差异是结构性的，且方向对 hybrid 不利。**

---

## 1. 实测（K 扫描，前缀缓存开启，每格 3 遍取中位）

| K（draft tokens） | dense tok/s | hybrid tok/s |
|---|---|---|
| 0（无投机，零对照） | **1468.2** | **628.7** |
| 1 | 1181.3 | 257.4 |
| 3 | 1192.5 | 222.6 |
| 5 | 1240.6 | 209.1 |
| 7 | 1186.9 | **138.2** |
| **单调递减步数** | **2/4**（非单调） | **4/4（严格单调）** |

**相对 K=0 的净效应**：

| K | dense | hybrid |
|---|---|---|
| 1 | −19.5% | **−59.1%** |
| 3 | −18.8% | −64.6% |
| 5 | −15.5% | −66.7% |
| 7 | −19.2% | **−78.0%** |

**⇒ hybrid 上投机在**所有** K 都是负收益，最好的情况（K=1）也是 2.44× 慢。**

---

## 2. 为什么这个形状是有信息量的（而不是噪声）

1. **dense 是非单调且平坦的**（K=1..7 都在 1181–1241 之间，波动 <5%）：
   说明 dense 上 ngram 的接受率与草稿开销大致抵消 —— 与 C-1 的发现一致（ngram ≈ 无投机）。
2. **hybrid 是严格单调的**（4/4 步递减，K=1→7 再降 46%）：
   每多一个草稿 token 都让吞吐更差 ⇒ **代价按 K 线性累加，而收益没有跟上**。
   这是"**草稿/验证路径在 hybrid 上每 token 成本异常高**"的特征，
   而不是"接受率低"的特征（接受率低会让 K 增大趋于平台，而非持续下降）。
3. **K=1 的巨大落差（628.7 → 257.4）**：即使只加 1 个草稿 token 也付 2.44× 代价
   ⇒ 开销不是按 token 线性摊的，**有一段固定的、与 K 无关的惩罚**
   （可能是 per-step 的草稿上下文构建 / graph 重捕获 / 状态回滚）。

**三条证据合起来指向**：hybrid 目标的投机路径存在**与 K 无关的固定开销 + 与 K 线性的额外开销**，
两者叠加使投机在 hybrid 上**结构性地**不划算。

---

## 3. 这解释了 C-1 的证伪（重要）

C-1 中我测 **dense** Qwen3-4B，得到"prefix caching 对投机是正收益、dflash2 最好"，
因此判 dflash 无害。**但那是在 dense 上**。
本轮测 **hybrid** Qwen3.5-4B，得到**同一配置族下投机大幅负收益**。

⇒ **上游 issue #47930 报的是 hybrid，我 C-1 测的是 dense —— 我的证伪不覆盖 upstream 的条件。**
**C-1 的"已证伪"结论必须收窄为：在 dense Qwen3-4B 上，预登记的两条预测不成立。**
它**不能**推广为"投机×前缀缓存无害"。这条更正已写回 C-1 的判定文件。

---

## 4. 尚未验证的（诚实边界 —— 必须做完才能立项）

| 项 | 状态 |
|---|---|
| **接受率/接受长度** | ❌ **没拿到**。我尝试抓 vLLM 的 `SpecDecoding metrics` 日志行，10 格全部 `None`（logger 名不对或未开 `log_stats`）。**因此"低接受率"只是推测，未取证。** |
| **单步延迟分解**（草稿/验证/状态回滚各占多少） | ❌ 未测。这是定位"固定开销"来源的关键 |
| **并发 >1** | ❌ 未测（预登记 C-1 就列了这一臂，两轮都没实现——执行缺口） |
| **其它 drafter**（eagle3/dflash on hybrid） | ❌ 未测。dflash2/eagle3 的 draft head 词表与 Qwen3.5 不匹配（151936/32000 vs **248320**），**不能直接复用** |
| **其它 hybrid 家族**（Nemotron-H / Jamba / Falcon-H1 / Qwen3-Next） | ❌ 未测。**只有一个 hybrid 模型就下结论是危险的** |
| **是否 upstream 已知** | ⚠️ 未查。需检索 "hybrid GDN speculative decoding slow" / "linear attention draft overhead" |

## 5. 判据（下一步，先写后做）

- **立项判据**：① 在 **≥2 个 hybrid 家族**上复现单调负收益；② 给出**单步延迟分解**，
  指出固定开销与线性开销各自的来源（file:line）；③ 与 upstream 已知问题划清界限。
- **杀判据**：若换一个 hybrid 模型后效应消失 ⇒ 这是该 checkpoint 的问题，归档。
- **当前状态**：**观察到强效应（−59% 到 −78%）+ 机制形状清楚（严格单调）+ 单一模型 ⇒ 不足以立项，但值得继续。**

## 6. 复现

```bash
# K 扫描（dense vs hybrid），~10 分钟
cd /root/autodl-tmp
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  /root/miniconda3/bin/python c1b_hybrid_verify.py \
    --ks 0,1,3,5,7 --targets dense,hybrid \
    --n-convs 8 --prefix 1024 --gen 32 --passes 3 --gpu-util 0.42 \
    --out /root/autodl-tmp/c1b.json
```
原始数据：`results/C1b-hybrid-spec/2026-09-14/c1b.json`、`c1_hybrid.json`


---

## 7. 跨运行可复现性（补做，2026-09-14 晚）

同一效应在 **3 次独立运行**中复现：

| 运行 | hybrid K=0 | hybrid K=3 | **K0/K3** | dense K0 | dense K3 | **K0/K3** |
|---|---|---|---|---|---|---|
| run2（K 扫描） | 628.7 | 222.6 | **2.82×** | 1468.2 | 1192.5 | 1.23× |
| run3 | 559.4 | 231.8 | **2.41×** | 1201.5 | 1198.1 | 1.00× |

**⇒ hybrid 上投机代价稳定在 2.4–2.8×；dense 上稳定在 1.0–1.2×。**
两族差异**大于任何一次运行内部的波动**，且方向一致 ⇒ **不是单次运行的产物**。

（这正是上一轮 `quantile-speedup` 缺的那一步：那条只做了一次就立项。
本次**先做跨运行复现**再考虑立项。）

## 8. 仪器缺口（诚实记录）

- **接受率/接受长度始终没拿到**。已试过四种方式，全部失败：
  1. 在本进程挂 `logging.Handler` → 引擎核心在**独立进程**，收不到；
  2. `disable_log_stats=False` → 必须在 `LLM(...)` 显式传入，且**仍无输出**；
  3. `VLLM_LOG_STATS_INTERVAL=0.5` → 无输出；
  4. `VLLM_CONFIGURE_LOGGING=1` + 抓 stderr → `grep -c "SpecDecoding metrics"` = **0**。
- 根因（已定位一处）：`vllm/entrypoints/llm.py:228-229` **默认把 `disable_log_stats` 设为 True**。
  覆盖它仍无输出 ⇒ 还有第二处闸门未找到。
- **⇒ 要用更硬的机制证据，必须改用 OpenAI server + `/metrics`**
  （`vllm:spec_decode_*` 计数器）或 `--per-request-spec-decode-metrics`。
  这是下一步的第一优先，**不是可选项**：没有接受率，"代价来自草稿路径"就仍是推测。
