# p23 —— 「带 seed vs 不带 seed」匹配对照：逐请求采样循环的每步代价

**日期**：2026-09-15 ｜ **目标**：`goal-a729588f`（引擎执行代价 / 投机解码 / KV Cache 优化）
**成本**：**≈0.15 GPU·h**（3 个 serve × (1 趟预热丢弃 + 3 次测量 × 2 臂)）｜ 末态：显存 0 MiB、无残留进程
**仪器**：`probes/p23-seeded-sampling/{seeded_ab.py,run_seeded_ab.sh}`（`seeded_ab.py --selftest` 通过）
**raw**：`/root/ccfa_results/2026-09-14/p23_seeded/`

## 1. 被检验的引擎自认缺口（**vLLM 0.29.0 与 `main` 同一份代码、同一个 TODO**）

`vllm/v1/sample/ops/topk_topp_sampler.py::random_sample`
```python
q = empty_exponential_noise_like(probs, use_fp64_gumbel)
# NOTE(woosuk): To batch-process the requests without their own seeds, which is the common case,
# we first assume that every request does not have its own seed. Then we overwrite the values
# for the requests that have their own seeds.
if len(generators) != probs.shape[0]:
    q.exponential_()                      # ← 整批一次 kernel（快路径）
if generators:
    # TODO(woosuk): This can be slow because we handle each request
    #      one by one. Optimize this.
    for i, generator in generators.items():
        q[i].exponential_(generator=generator)   # ← 每个带 seed 的请求一次 kernel
```
触发条件（`gpu_model_runner.py`）：`sampling_type == SamplingType.RANDOM_SEED` ⇒ 建 `torch.Generator`；否则 `None`。
⇒ **`temperature>0` 不带 seed ⇒ RANDOM ⇒ 走快路径；`temperature>0` 带 seed ⇒ RANDOM_SEED ⇒ 走逐请求循环。**
⇒ A/B 的**唯一变量**就是"请求里有没有 seed"，两边都真的在采样 —— 这是一个**教科书式的条件性差异轴**（快路径就在引擎里，只是被条件绕开）。

## 2. 结果：**候选死于量级**（3 个配置全部不可分辨）

| 配置 | 并发 | unseeded 暖态 p50 (tok/s) | seeded 暖态 p50 | Δ | 3× 重复噪声 | 可分辨？ |
|---|---|---|---|---|---|---|
| `eager_c8` | 8 | 617.20 | 608.80 | **−1.36%** | 10.21% | **否** |
| `eager_c32` | 32 | 1003.11 | 967.34 | **−3.57%** | 15.93% | **否** |
| `graph_c32`（CUDA graph） | 32 | 1066.51 | 1039.30 | **−2.55%** | 14.56% | **否** |

- **符号一致**（三格 seeded 都更慢）⇒ 缺口在方向上**是真的**；但量级 **1.4–3.6%**，远低于 3× 噪声（10–16%），
  也远低于本工作区惯用的 8% 门槛 ⇒ **按目标判据（效应量 >3× 同格重复噪声）不成立**。
- **副产品（有价值）**：`graph_c32` 仍有 −2.55% ⇒ **CUDA graph 并没有把这些逐请求 launch 吸收掉**
  （要么采样不在图内，要么逐请求 launch 落在图外）—— 这条否掉了我"图会吃掉它"的先验。
- 边界：并发只到 32（受 `--max-num-seqs 32` 限制）。理论上代价 ∝ 批内 seeded 请求数，
  ⇒ 在 **并发数百** 的生产形态下量级会更大；但本机达不到那个批大小，**该外推未取证**。

## 3. 判定

**杀（死于量级，不是死于被占位）。** 这是本目标里**第一个通过 S3b、却死在测量上的池② 候选** ——
它与前 9 条（全部死于"上游在办"）性质不同：**缺口真实、无人讨论、可达，但太小，不值得立项。**

**复活条件**：若本机/换机能跑到并发 ≥128 且仍观测到 ≥8% 的差距，则量级论证翻转。

## 4. 本轮顺带修掉的两个仪器坑（都属于"本该早就发现"）

1. **预热被重定向到 `/dev/null`，把错误信息一起丢了** —— 我因此**调试了四轮**都看不到失败原因
   （症状：脚本在调用函数处静默退出、退出码 1、无任何报错）。
   **这是"丢弃证据"这一类错误的第二次**（第一次是决策 #123 的 bench 日志 0 字节）。
   ⇒ 规则：**任何被丢弃的输出必须落盘**；预热只丢弃**结果**，不丢弃**日志**。
2. **bash 5 + `set -u` 陷阱（实测 bash 5.1.16）**：同一条 `local` 里，**后面的赋值不能引用前面的** ——
   `local a=$1 b=$2 c="$a-$b"` ⇒ 直接 `a: unbound variable` 并中止脚本。
   `p20/p22` 的运行器**没踩到纯属侥幸**（它们的 `tag`/`r` 恰好与**全局**变量重名、已被绑定）。
   ⇒ 规则：**一条 `local` 只声明、不互相引用**。
