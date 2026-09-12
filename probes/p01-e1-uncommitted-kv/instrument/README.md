# E1 仪器化（#1 · `p01-e1-uncommitted-kv`）

两个脚本 + 一条源码事实。**用途**：把"未提交投机 KV 占比"从一个没人数过的量，变成 D1 当天就能拿到的数字。

> ✅ **锚点核对（2026-09-12，新增）**：在 **vLLM `v0.29.0`**（即新机器上实际要跑的版本）上逐项验过，**补丁无需修改**：
> ① Hook A 锚点（`if request.num_output_placeholders > 0:` / `request.num_output_placeholders -= num_rejected`）
> 在 `vllm/v1/core/sched/scheduler.py` 中**恰好出现 1 次**（第 1916–1917 行），缩进与模板逐字节一致；
> ② `KVCacheManager.allocate_slots(self, request, num_new_tokens, ...)` 与包装器的位置参数签名一致（第 343 行起）；
> ③ 其返回值 `KVCacheBlocks | None` 具备 `get_block_ids()`。
> ⇒ D1 的"引擎固定 vLLM"在 0.29.0 上成立，插桩可直接 `--apply`。

## 怎么用

```bash
# 0) 环境（在 GPU 机器上，装好 vllm 之后）
source .venv/bin/activate

# 1) 打补丁（幂等；锚点基于本地快照 vllm-main，若你的 pin 不同会在锚点处报错并打印期望片段）
python probes/p01-e1-uncommitted-kv/instrument/e1_patch.py --status
python probes/p01-e1-uncommitted-kv/instrument/e1_patch.py --apply

# 2) 起服务，开日志（E1_LOG 未设置时探针零开销）
export E1_LOG=/tmp/e1_metrics.jsonl
vllm serve <TARGET> \
  --speculative-config '{"model":"<eagle3-draft-ckpt>","num_speculative_tokens":5}' \
  --max-model-len 32768 --gpu-memory-utilization 0.85

# 3) 按网格压测（另开一个 shell）。ctx × batch × γ{3,5,7} + 1 个 draft-tree 配置
vllm bench serve --model <TARGET> --base-url http://localhost:8000 \
  --dataset-name random --random-input-len 4096 --random-output-len 256 \
  --num-prompts 64 --max-concurrency 8          # 逐格改参数，每格 ≥3 次

# 4) 分析
python probes/p01-e1-uncommitted-kv/instrument/analyze.py \
  --log /tmp/e1_metrics.jsonl --block-size 16 \
  --out results/p01-e1-uncommitted-kv/$(date +%F) --tag "ctx4k_bs8_g5"

# 5) 收工后撤销（保持环境干净）
python probes/p01-e1-uncommitted-kv/instrument/e1_patch.py --revert
```

## 记录格式（JSONL）

| 事件 | 字段 | 含义 |
|---|---|---|
| `spec` | `req` `ctx` `draft` `accepted` `rejected` `invalid` | 每个 decode 步每条请求：提出的草稿数、被接受数、被拒数、回滚**之后**的已提交上下文长度、语法无效草稿数 |
| `alloc` | `req` `new_tokens` `blocks` `ctx` | 每次 `allocate_slots`：请求的 token 数与实际拿到的块数（**预留口径**的来源） |

## 一条决定"该测什么"的源码事实

`vllm/v1/core/sched/scheduler.py`（本地快照，约 1970–1985 行）逐字：

```python
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
...
if not output_is_stale:
    if request.num_computed_tokens > 0:
        request.num_computed_tokens -= num_rejected
```

⇒ **被拒 token 在同一步就被回滚**，位置计数退回并被下一步覆写。推论：

1. **`H`（未提交块的驻留步数）预期 ≈ 0** —— "未提交状态"不是一个持续存在的内存类别，而是每一步的瞬时窗口；
2. 因此 `R_byte` 的解析值就是 **`γ / ctx`**（随上下文缩小），长上下文下必然很小；
3. **真正可能大的量是 `R_reserve`**：引擎为投机**预留**的槽位按块粒度向上取整，而每请求每步只需 1 个位置（`docs/plan.md` #13 记的"预留 8 个 slot 而实际只需 1"就是这一类）。

⇒ **D1 的结论不该只看 `R_byte`**。分析器因此同时输出三个量，并在 `summary.md` 的第五节强制回答"`R_reserve / R_byte` 是否 >2"。

## 注意事项

- 仅适用于 **vLLM V1** 引擎；`--speculative-config` 必须真的生效（用 `spec` 记录数确认，为 0 直接报错）。
- 插入点在 `if not output_is_stale:` **内部**，因此 drop-mode / 陈旧输出不会污染统计。
- TP>1 时多个进程都会写同一个 `E1_LOG`：记录带 `pid` 可区分，但**建议先用 TP=1** 取数。
- `--apply` 会在同目录留下 `*.e1orig` 原文件备份；`--revert` 从备份**字节级还原**并删除备份。升级 vllm 前请先 revert。
- 若你的 pin 与快照差异较大：`--apply` 会打印缺失的锚点片段，照着插一行 `_e1_emit(...)` 即可；`--revert` 按标记删除，不会误删你的改动。
- 判据与网格以 `notes/prereg/p01-e1-uncommitted-kv.md` 为准（含 24 GB 卡的 ctx 分支）。
