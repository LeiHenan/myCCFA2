#!/usr/bin/env python
"""Build a minimal PEFT-format LoRA adapter for Qwen3-4B, with no downloads.

We need a REAL loaded adapter so vLLM puts a real LoRA name into `extra_keys`
and we can test whether a client-supplied `cache_salt` collides with it.

Qwen3-4B target modules: q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj.
Rank 8, random (non-zero) weights so the adapter actually changes the model.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import save_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="adapterA")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2560)   # Qwen3-4B
    ap.add_argument("--num-layers", type=int, default=36)
    ap.add_argument("--intermediate", type=int, default=9728)
    ap.add_argument("--num-q-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--scale", type=float, default=0.01,
                    help="std of A/B init; the LoRA delta is ~ r*scale^2*alpha/r")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    r, h = a.rank, a.hidden
    q_out = a.num_q_heads * a.head_dim
    kv_out = a.num_kv_heads * a.head_dim
    g = torch.Generator().manual_seed(1234)

    targets = {
        "q_proj": (h, q_out), "k_proj": (h, kv_out), "v_proj": (h, kv_out),
        "o_proj": (q_out, h), "gate_proj": (h, a.intermediate),
        "up_proj": (h, a.intermediate), "down_proj": (a.intermediate, h),
    }

    tensors = {}
    for layer in range(a.num_layers):
        for mod, (in_f, out_f) in targets.items():
            base = f"base_model.model.model.layers.{layer}.self_attn.{mod}" \
                if mod in ("q_proj", "k_proj", "v_proj", "o_proj") else \
                f"base_model.model.model.layers.{layer}.mlp.{mod}"
            # LoRA: A is (r, in), B is (out, r)
            tensors[f"{base}.lora_A.weight"] = (
                torch.randn(r, in_f, generator=g, dtype=torch.float32) * a.scale
            ).to(torch.bfloat16)
            tensors[f"{base}.lora_B.weight"] = (
                torch.randn(out_f, r, generator=g, dtype=torch.float32) * a.scale
            ).to(torch.bfloat16)

    save_file(tensors, os.path.join(a.out, "adapter_model.safetensors"))

    cfg = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": "Qwen/Qwen3-4B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": r,
        "target_modules": list(targets.keys()),
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(a.out, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"wrote adapter to {a.out}")
    print(f"  tensors: {len(tensors)}  rank={r}")
    print(f"  sample keys: {list(tensors)[:2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
