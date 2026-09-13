#!/usr/bin/env python3
"""零卡估算：**禁用 cuBLAS split-k** 对 Qwen3-4B 各阶段 GEMM 的影响量级。

动机（p15 第 8 轮的源码发现）：`vllm/model_executor/determinism/batch_invariant.py:918-940` 显示
**SM90/SM100/SM120 走 `else` 分支**：不替换 matmul 算子，而是
`CUBLAS_WORKSPACE_CONFIG=":16:8"` + `CUBLASLT_WORKSPACE_SIZE="1"` ⇒ **靠禁用 split-k 消除批方差**。
⇒ 单卡成本的主要嫌疑 = split-k 被禁。本脚本用**真实 config** 估这块的上下界，避免手写形状出错。

判据背景：split-k 把 K 维切分并行，主要收益在 **M 小（decode / 低并发）** 时；
M 很大（prefill）时 GPU 已被 M×N 填满，split-k 收益趋零。
⇒ 可检验预测：**"禁 split-k 的代价"应随 M（= 并发×token 数）增大而减小。**

用法：python estimate_splitk.py [--model /root/autodl-tmp/models/Qwen3-4B] [--selftest]
"""
import argparse
import json
import os
import sys

# Qwen3-4B 的层形状从 config 推导（不手写）
def load_shapes(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    h = cfg["hidden_size"]
    i = cfg.get("intermediate_size")
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim", h // heads)
    n_layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    shapes = {
        "q_proj": (heads * head_dim, h),
        "kv_proj": (2 * kv_heads * head_dim, h),
        "o_proj": (h, heads * head_dim),
        "gate_up_proj": (2 * i, h),
        "down_proj": (h, i),
        "lm_head": (vocab, h),
    }
    return cfg, shapes, n_layers


def gemm_bytes_flops(m, n, k, dt=2):
    flops = 2 * m * n * k                      # MAC*2
    w_bytes = n * k * dt                       # 权重读取（下界）
    return flops, w_bytes


def estimate(model_dir, m_values):
    cfg, shapes, L = load_shapes(model_dir)
    print(f"模型 {os.path.basename(model_dir)}：hidden={cfg['hidden_size']} "
          f"inter={cfg.get('intermediate_size')} layers={L} "
          f"heads={cfg['num_attention_heads']} kv_heads={cfg.get('num_key_value_heads')}")
    print(f"{'proj':<14}{'N×K':>14}  " + "".join(f"{'M='+str(m):>16}" for m in m_values))
    print("-" * (16 + 16 * len(m_values)))
    for name, (n, k) in shapes.items():
        row = f"{name:<14}{f'{n}x{k}':>14}  "
        for m in m_values:
            flops, wb = gemm_bytes_flops(m, n, k)
            row += f"{flops/1e9:>10.2f} GF{wb/1e6:>5.1f}MB"
        print(row)
    print()
    print("读法：GF = 该阶段总 FLOPs；MB = 权重字节（固定）。")
    print("  · M=1（decode 单序列）⇒ 算术强度 = 2M·N·K/(N·K·2) ≈ M ⇒ **严重 memory-bound** ⇒ split-k 收益最大")
    print("  · M 很大（prefill）⇒ 计算受限 ⇒ split-k 收益趋零")
    print("  ⇒ **可检验预测：禁 split-k 的墙钟代价随 M 增大而下降**")
    return 0


def selftest():
    import tempfile
    td = tempfile.mkdtemp()
    cfg = {"hidden_size": 2560, "intermediate_size": 9728, "num_attention_heads": 32,
           "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 36,
           "vocab_size": 151936}
    json.dump(cfg, open(os.path.join(td, "config.json"), "w"))
    c, shapes, L = load_shapes(td)
    assert L == 36 and shapes["q_proj"] == (4096, 2560), shapes
    assert shapes["kv_proj"] == (2048, 2560), shapes
    assert shapes["gate_up_proj"] == (19456, 2560), shapes
    f, wb = gemm_bytes_flops(1, 2560, 4096)
    assert f == 2 * 2560 * 4096 and wb == 2560 * 4096 * 2, (f, wb)
    print("selftest ✔ 形状从 config 推导（含 GQA 的 kv 头）+ FLOPs/字节口径")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--m", nargs="*", type=int, default=[1, 8, 64, 512, 4096, 32768])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return estimate(a.model, a.m)


if __name__ == "__main__":
    sys.exit(main())
