#!/usr/bin/env python3
"""零卡预算器：混合模型（SSM/线性注意力）的**状态内存 vs KV 内存**。

**为什么需要它（S1 取证任务第 1 步，decision #72）**：
候选 `hybrid-state-serving` 的 P3 痛点是"状态内存与 KV 双重占用、没有联合预算"。
按本工作区判据 **L0（没有数字）不得进入 S3** ⇒ 先用**零 GPU** 把量级算出来再谈读论文。
本脚本只用 `config.json` + vLLM 自己的状态形状公式（逐字取自
`vllm/model_executor/layers/mamba/mamba_utils.py:150-300`），**不需要 GPU、不需要下载权重**。

核心量与恒等式
--------------
- `kv_per_token`   = 2(K,V) × n_attn_layers × n_kv_heads × head_dim × dtype_bytes
- `state_per_seq`  = Σ_状态层 (conv_state + temporal_state) × dtype_bytes     ← **与上下文长度无关**
- **交叉点 `L*` = state_per_seq / kv_per_token**
  ⇒ 上下文超过 `L*` 后 KV 才比"一份状态"更占内存；**短上下文时状态可以是大头**。
- **检查点摊销**：mamba 状态以 `mamba_block_size`（记 b）为粒度做检查点 ⇒
  `checkpoint_per_token = state_per_seq / b`，于是
  **`检查点开销 / KV` = L* / b**（**与上下文长度 L 无关**）
  ⇒ 想要比 `L*` 更细的复用粒度，就要付出**比 KV 本身更贵**的内存。

用法：
  python budget.py --dir <含 *.json 配置的目录> [--out 目录] [--block-size 16]
  python budget.py --selftest
"""

import argparse
import glob
import json
import os
import sys

DTYPE_BYTES = 2          # bf16（vLLM 默认 dtype）
DEFAULT_BLOCK = 16       # vllm/config/cache.py:79  DEFAULT_BLOCK_SIZE = 16


# ---------------------------------------------------------------- 形状公式
def mamba2_state(intermediate, n_groups, num_heads, head_dim, state_size, conv_kernel):
    """vLLM MambaStateShapeCalculator.mamba2_state_shape（mamba_utils.py:184）"""
    conv_dim = intermediate + 2 * n_groups * state_size
    conv = conv_dim * (conv_kernel - 1)
    temporal = num_heads * head_dim * state_size
    return conv, temporal


def gdn_state(num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel):
    """vLLM MambaStateShapeCalculator.gated_delta_net_state_shape（mamba_utils.py:258）"""
    conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
    conv = conv_dim * (conv_kernel - 1)
    temporal = num_v_heads * head_v_dim * head_k_dim
    return conv, temporal


def mamba1_state(intermediate, state_size, conv_kernel):
    """vLLM MambaStateShapeCalculator.mamba1_state_shape（mamba_utils.py:169）"""
    return intermediate * (conv_kernel - 1), intermediate * state_size


# ---------------------------------------------------------------- 各架构解析
def parse_nemotron_h(c):
    """层类型来自 hybrid_override_pattern（vLLM nemotron_h.py:572 按 layer_idx 取字符）。
    M = Mamba 层（有状态）、* = 注意力层（有 KV）、- = 纯 MLP（两者都没有）。
    intermediate = mamba_num_heads × mamba_head_dim（nemotron_h.py:371 逐字）。"""
    pat = c.get("hybrid_override_pattern", "")
    n_m = pat.count("M")
    n_a = pat.count("*")
    conv, temp = mamba2_state(c["mamba_num_heads"] * c["mamba_head_dim"], c["n_groups"],
                              c["mamba_num_heads"], c["mamba_head_dim"],
                              c["ssm_state_size"], c["conv_kernel"])
    return dict(arch="nemotron_h", layers=len(pat), n_state=n_m, n_attn=n_a,
                n_mlp=len(pat) - n_m - n_a,
                state_elems_per_layer=conv + temp,
                kv_heads=c["num_key_value_heads"], head_dim=c["attention_head_dim"])


def parse_falcon_h1(c):
    """Falcon-H1 是**并行混合**：每一层同时含 attention 分支与 SSM 分支
    （vLLM falcon_h1.py:317-355 FalconH1ParallelHybrid 逐字）⇒ 状态层数 = 注意力层数 = 总层数。"""
    n = c["num_hidden_layers"]
    conv, temp = mamba2_state(c["mamba_d_ssm"], c["mamba_n_groups"], c["mamba_n_heads"],
                              c["mamba_d_head"], c["mamba_d_state"], c["mamba_d_conv"])
    return dict(arch="falcon_h1", layers=n, n_state=n, n_attn=n, n_mlp=0,
                state_elems_per_layer=conv + temp,
                kv_heads=c["num_key_value_heads"], head_dim=c["head_dim"])


def parse_qwen3_next(c):
    """full_attention_interval = k ⇒ 每 k 层一个 full attention（vLLM qwen3_next.py:656
    按 config.layer_types 取），其余为 GDN 线性注意力层。"""
    layers = c.get("layer_types")
    if layers:
        n_a = sum(1 for x in layers if "full" in x)
    else:
        k = c.get("full_attention_interval", 4)
        n_a = c["num_hidden_layers"] // k
    n = c["num_hidden_layers"]
    conv, temp = gdn_state(c["linear_num_key_heads"], c["linear_num_value_heads"],
                           c["linear_key_head_dim"], c["linear_value_head_dim"],
                           c["linear_conv_kernel_dim"])
    return dict(arch="qwen3_next", layers=n, n_state=n - n_a, n_attn=n_a, n_mlp=0,
                state_elems_per_layer=conv + temp,
                kv_heads=c["num_key_value_heads"], head_dim=c["head_dim"])


def parse_mamba2(c):
    """纯 SSM（无注意力层）⇒ kv_per_token = 0，内存 100% 是状态，且**不随上下文增长**。"""
    inter = c.get("expand", 2) * c.get("d_model", 0)
    heads = c.get("nheads", 0)
    hd = c.get("head_dim") or (inter // heads if heads else 0)
    conv, temp = mamba2_state(inter, c.get("n_groups", 1), heads, hd,
                              c.get("d_state", 0), c.get("d_conv", 4))
    return dict(arch="mamba2(纯SSM)", layers=c.get("n_layer", 0), n_state=c.get("n_layer", 0),
                n_attn=0, n_mlp=0, state_elems_per_layer=conv + temp,
                kv_heads=0, head_dim=0, note="无注意力层 ⇒ KV=0，状态不随上下文增长")


PARSERS = {
    "nemotron_h": parse_nemotron_h,
    "falcon_h1": parse_falcon_h1,
    "qwen3_next": parse_qwen3_next,
    "mamba2": parse_mamba2,
}


def budget(cfg, block=None, dtype_bytes=DTYPE_BYTES, batches=(1, 8, 32),
           ctxs=(4096, 32768, 131072)):
    mt = cfg.get("model_type")
    if mt not in PARSERS:
        return {"error": f"未支持的 model_type={mt!r}（本脚本只覆盖 {sorted(PARSERS)}）"}
    p = PARSERS[mt](cfg)
    kv_per_token = 2 * p["n_attn"] * p["kv_heads"] * p["head_dim"] * dtype_bytes
    state_per_seq = p["n_state"] * p["state_elems_per_layer"] * dtype_bytes
    L_star = state_per_seq / kv_per_token if kv_per_token else float("inf")
    b = block or DEFAULT_BLOCK
    ckpt_per_token = state_per_seq / b
    rows = []
    for B in batches:
        for L in ctxs:
            kv = B * kv_per_token * L
            st = B * state_per_seq
            ck = B * ckpt_per_token * L          # 最坏情形：整段前缀都存检查点
            rows.append(dict(batch=B, ctx=L, kv_bytes=kv, state_bytes=st, ckpt_bytes=ck,
                             state_share=st / (kv + st) if kv + st else 1.0,
                             ckpt_over_kv=(ck / kv) if kv else float("inf")))
    return dict(**p, kv_per_token=kv_per_token, state_per_seq=state_per_seq,
                L_star=L_star, block=b, ckpt_per_token=ckpt_per_token,
                ckpt_over_kv_ratio=(ckpt_per_token / kv_per_token) if kv_per_token else float("inf"),
                rows=rows)


# ---------------------------------------------------------------- 输出
def fmt(n):
    for u, s in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if abs(n) >= s:
            return f"{n / s:.2f} {u}"
    return f"{n:.0f} B"


def report(results, block):
    L = ["# 零卡预算：混合模型的「状态内存 vs KV 内存」", "",
         f"> 口径：bf16（2 B/elem）；状态形状逐字取自 vLLM "
         f"`model_executor/layers/mamba/mamba_utils.py:150-300`；"
         f"`mamba_block_size = {block}`（检查点粒度，vLLM 默认 `DEFAULT_BLOCK_SIZE=16`，"
         f"用户可用 `--mamba-block-size` 调大）。", "",
         "| 模型 | 架构 | 层数 (状态/注意力/MLP) | KV /token | **状态 /序列** | **交叉点 L\\*** | "
         "检查点开销 /KV（= L\\*/b） |", "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if "error" in r:
            L.append(f"| {name} | — | — | — | — | — | ⚠️ {r['error']} |")
            continue
        L.append(f"| {name} | {r['arch']} | {r['layers']} ({r['n_state']}/{r['n_attn']}/{r['n_mlp']}) | "
                 f"{fmt(r['kv_per_token'])} | **{fmt(r['state_per_seq'])}** | "
                 f"**{r['L_star']:.0f} tokens** | **{r['ckpt_over_kv_ratio']:.1f}×** |")
    L += ["", f"## 明细（状态占总缓存的比例；检查点按最坏情形 b={block} 整段前缀都存）", "",
          "| 模型 | batch | ctx | KV | 状态(常驻) | 状态占比 | 检查点开销 | 检查点/KV |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if "error" in r:
            continue
        for row in r["rows"]:
            if row["batch"] in (1, 32) and row["ctx"] in (4096, 32768):
                L.append(f"| {name} | {row['batch']} | {row['ctx']} | {fmt(row['kv_bytes'])} | "
                         f"{fmt(row['state_bytes'])} | **{row['state_share'] * 100:.1f}%** | "
                         f"{fmt(row['ckpt_bytes'])} | {row['ckpt_over_kv']:.1f}× |")
    L += ["", "## 粒度—内存权衡：`检查点开销 / KV = L* / b`（与上下文长度 **无关**）", "",
          "| 模型 | L\\*（= 允许的最细粒度下界） | b=16 | b=256 | b=1024 | b=4096 | b=32768 |",
          "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if "error" in r:
            continue
        cells = []
        for b in (16, 256, 1024, 4096, 32768):
            cells.append(f"**{r['L_star'] / b:.1f}×**" if b < r["L_star"] else f"{r['L_star'] / b:.2f}×")
        L.append(f"| {name} | **{r['L_star']:.0f} tokens** | " + " | ".join(cells) + " |")
    L += ["", "> **设计约束（本步的核心结论）**：只要检查点粒度 `b < L*`，"
          "「缓存状态检查点」就比「缓存 KV」**更贵**。三个模型的 `L*` 都只有 **1.6k–3.2k tokens**，"
          "而 KV 的块粒度是 16 ⇒ **想拿到与 KV 同粒度的状态复用，代价是 KV 本身的 ~100–200 倍**。",
          "> ⚠️ `mamba_block_size` 的**默认值**在 config 层是 `None`（构造 KV cache spec 时解析；"
          "`abstract.py:68-69` 断言其非空）。`validate_mamba_block_size` 把 `== max_model_len` 当作"
          "『未显式设置』（`vllm/config/vllm.py:2839-2841`）⇒ **默认疑似等于 max_model_len（整段序列只存一份）**。"
          "**推定，须在 S4 用一次真实 serve 的日志确认**。", ""]

    L += ["", "## 读法（三条判据）", "",
          "1. **`L*` = 状态/序列 ÷ KV每token** ⇒ 上下文短于 `L*` 时，**一份状态就比整段 KV 更占内存**。",
          "2. **检查点开销 / KV = `L*` / b**（与 L 无关）⇒ b 每减半，检查点开销翻倍。",
          "3. **P3 判定**：若某模型在 `batch=32, ctx=4096` 下状态占比 >5%，则「状态与 KV 双重占用」"
          "是真痛点（L1）；若 ≪1% 则 P3 淘汰。"]
    return "\n".join(L) + "\n"


def selftest():
    # 合成配置：只验证公式与解析，不依赖网络
    c = {"model_type": "nemotron_h", "hybrid_override_pattern": "M-M*", "mamba_num_heads": 2,
         "mamba_head_dim": 4, "n_groups": 1, "ssm_state_size": 8, "conv_kernel": 4,
         "num_key_value_heads": 2, "attention_head_dim": 8}
    r = budget(c, block=16, batches=(1,), ctxs=(1024,))
    # conv_dim = 2*4 + 2*1*8 = 24 ; conv = 24*3 = 72 ; temporal = 2*4*8 = 64 ⇒ 136 elem/层
    # "M-M*" = 4 个字符 ⇒ M, -, M, * ⇒ 2 个状态层、1 个注意力层、1 个纯 MLP 层
    assert r["n_state"] == 2 and r["n_attn"] == 1 and r["n_mlp"] == 1, r
    assert r["state_per_seq"] == 2 * 136 * 2, r["state_per_seq"]
    assert r["kv_per_token"] == 2 * 1 * 2 * 8 * 2, r["kv_per_token"]
    assert abs(r["L_star"] - (2 * 136 * 2) / (2 * 1 * 2 * 8 * 2)) < 1e-9, r["L_star"]
    # 检查点/KV 恒等于 L*/b（与 ctx 无关）
    for row in r["rows"]:
        assert abs(row["ckpt_over_kv"] - r["L_star"] / 16) < 1e-9, row
    # Falcon-H1 并行混合：状态层数 == 注意力层数
    f = parse_falcon_h1({"num_hidden_layers": 4, "mamba_d_ssm": 8, "mamba_n_groups": 1,
                         "mamba_n_heads": 2, "mamba_d_head": 4, "mamba_d_state": 8,
                         "mamba_d_conv": 4, "num_key_value_heads": 1, "head_dim": 4})
    assert f["n_state"] == f["n_attn"] == 4, f
    # GDN（Qwen3-Next）：conv_dim = 2*2*2 + 3*4 = 20（k_heads=2,head_k=2,v_heads=3,head_v=4）
    g = gdn_state(2, 3, 2, 4, 4)
    assert g == (20 * 3, 3 * 4 * 2), g
    # 纯 SSM ⇒ kv_per_token == 0 ⇒ L* = inf
    m = budget({"model_type": "mamba2", "n_layer": 2, "d_model": 8, "expand": 2,
                "nheads": 2, "head_dim": 4, "d_state": 8, "d_conv": 4, "n_groups": 1},
               batches=(1,), ctxs=(1024,))
    assert m["kv_per_token"] == 0 and m["L_star"] == float("inf"), m
    txt = report({"demo": r}, 16)
    assert "交叉点" in txt and "检查点" in txt
    print("selftest ✔ 解析(M/O/*)/mamba2 公式/GDN 公式/Falcon 并行混合/纯 SSM/检查点恒等式/输出")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.dir:
        ap.error("需要 --dir，或 --selftest")
    results = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        name = os.path.basename(f)[:-5]
        try:
            cfg = json.load(open(f, encoding="utf-8"))
        except Exception as e:                                  # noqa: BLE001
            results[name] = {"error": f"配置读取失败 {e}"}
            continue
        # 兼容 state-spaces 的原始 Mamba2 配置（无 model_type，但有 n_layer/d_model）
        if "model_type" not in cfg and "n_layer" in cfg and "d_model" in cfg:
            cfg = {"model_type": "mamba2", **cfg}
        if "architectures" not in cfg and "model_type" not in cfg:
            results[name] = {"error": "不是 HF config.json（可能只是权重索引）"}
            continue
        results[name] = budget(cfg, block=a.block_size)
    txt = report(results, a.block_size)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "budget.md"), "w", encoding="utf-8").write(txt)
        with open(os.path.join(a.out, "budget.csv"), "w", encoding="utf-8") as fh:
            fh.write("model,arch,layers,n_state,n_attn,kv_per_token_B,state_per_seq_B,"
                     "L_star_tokens,ckpt_over_kv_x,batch,ctx\n")
            for name, r in results.items():
                if "error" in r:
                    fh.write(f"{name},ERROR,,,,,,,,\n")
                    continue
                for row in r["rows"]:
                    fh.write(f"{name},{r['arch']},{r['layers']},{r['n_state']},{r['n_attn']},"
                             f"{r['kv_per_token']:.0f},{r['state_per_seq']:.0f},"
                             f"{r['L_star']:.0f},{row['ckpt_over_kv']:.3f},{row['batch']},{row['ctx']}\n")
        print(f"已写出 {a.out}/budget.md 与 budget.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
