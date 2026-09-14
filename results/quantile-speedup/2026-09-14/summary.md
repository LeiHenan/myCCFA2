# quantile-speedup — 受控实验结果（2026-09-14）

**平台**：<SSH_HOST>，RTX 4080 SUPER 32GB，vLLM 0.29.0，Qwen3-4B
**脚本**：`probes/e4_final.py` ｜ **原始数据**：本目录 `e4_final_s1.json`

## 设计

| 项 | 设置 |
|---|---|
| prompt | 256 token × 40 条（尾 12 token 各异） |
| 输出 | `max_tokens=16 + ignore_eos` **钉死** ⇒ 两臂解码成本对齐 |
| 引擎 | 同进程两实例：`baseline`（无投机）/ `ngram-spec`（`num_speculative_tokens=5`, lookup 4/2） |
| 交错 | 逐请求交错，**交替先后顺序** ⇒ 消除系统漂移 |
| 噪声底 | 同引擎**二次通过**（`ratios_noise`） |
| 其他 | `max_model_len=1024`、`enforce_eager`、无前缀缓存 |

## 结果

| 指标 | 基线 | 投机 | 比值 | 噪声底 |
|---|---|---|---|---|
| mean | 551.2 ms | 395.7 ms | **0.718** | 0.942 |
| p50 | — | — | **0.672** | 0.963 |
| p95 | 577.7 ms | 576.1 ms | **0.997** | 1.002 |
| p99 | 589.3 ms | 585.9 ms | **0.994** | 1.025 |
| CV | — | — | **8.95×** | 3.72× |

**判据**：`tail_vs_mean = 1.389`，噪声底 `1.064` ⇒ **效应超出噪声 33 个百分点**（`tail_effect_exceeds_noise = true`）

**逐请求 spec/baseline 延迟比**（<1 = 投机更快）：

```
min 0.395   p05 0.514   p50 0.648   p95 1.071   max 1.089
被拖慢的请求比例：17.5%
```

## 三句话

1. 中位数请求快 **35%**，最快的快 **2.5×**；
2. 最慢的请求反而慢 **9%**，**17.5% 的请求被拖慢**；
3. **p95/p99 与基线无差别**——被拖慢的那批请求正好落在尾部。

## 复现命令（含全部环境坑）

```bash
<SSH_HOST>
pkill -9 -f e4_final; pkill -9 -f vllm; sleep 4
cd /root/autodl-tmp
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  timeout 520 /root/miniconda3/bin/python e4_final.py \
    --n-prompts 40 --gpu-util 0.42 --max-len 1024 --gen 16 --prompt 256 \
    --seed 7919 --tag s1
```

- `--max-len 1024`：2048 会让两引擎 KV 合计超卡
- `--gpu-util 0.42`：单进程**最多两个** `LLM`；三个必然失败
- `VLLM_USE_FLASHINFER_SAMPLER=0`：该机缺 flashinfer
- **单次命令窗口约 10 分钟 ⇒ n=40 是上限**（n=140 已实测超时）
- 更大样本须用多个 `--seed` 独立批次；`screen`/`setsid nohup` 在该机均不持久化

## 诚实边界

- n=40、单模型、单一加速技术、无并发。**扩样前不得声称普适。**
- 噪声底是"同引擎二次通过"，**不是** `docs/TOPIC_METHODOLOGY.md` §4.1 要求的"同配置 5 次重复"。
- 未测真实 SLO 违约率（需要真实负载与 deadline），只测了延迟分位数。
