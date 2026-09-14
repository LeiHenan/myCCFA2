#!/usr/bin/env python
"""C-1g: is the Qwen3.5 hybrid page accounting CORRECT under speculation?

Why this matters (from C-1f):
  Falcon-H1 crashes on spec+prefix-caching with
      assert self.page_size_padded >= page_size
  because MambaSpec.page_size_bytes counts num_speculative_blocks, while the
  padding was computed from a shape that did not. Qwen3.5-4B does NOT crash --
  its padding headroom is bigger (1.49% vs 0.24%).

  But "does not crash" is not "is correct". If Qwen3.5's speculative state is
  being squeezed into padding that was sized for a different shape, then the
  page layout / memory accounting is wrong, and my C-1d/C-1e conclusion
  ("hybrid costs more per speculative step") could be a SYMPTOM OF THE
  ACCOUNTING MISMATCH rather than a property of linear attention.

This probe answers it directly by reading the engine's own numbers:
    page_size        = sum(prod(shape)*dtype_size)   over the real MambaSpec shapes
    page_size_padded = the value alignment installed
    headroom         = padded - real
and by comparing the effective mamba state size WITH and WITHOUT speculation.

Gate:
  * headroom >= 0 with spec on  -> accounting is at least self-consistent
  * headroom <  0               -> the crash condition, just not reached
  * headroom shrinking to ~0    -> speculative state is eating the padding;
                                   treat the hybrid cost numbers as suspect
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=(
        "/root/autodl-tmp/models/Qwen3.5-4B,/root/autodl-tmp/models/Falcon-H1-3B"))
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    from vllm import LLM
    from vllm.config import VllmConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.platforms.interface import Platform

    for path in [m.strip() for m in a.models.split(",") if m.strip()]:
        name = os.path.basename(path.rstrip("/"))
        for spec_on in (False, True):
            kw = dict(model=path, gpu_memory_utilization=a.gpu_util,
                      max_model_len=a.max_len, enforce_eager=True,
                      dtype="bfloat16", language_model_only=True,
                      enable_prefix_caching=True)
            if spec_on:
                kw["speculative_config"] = {
                    "method": "ngram", "num_speculative_tokens": a.k,
                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
            tag = f"{name} spec={'on' if spec_on else 'off'}"
            try:
                llm = LLM(**kw)
            except Exception as e:  # noqa: BLE001
                print(f"[{tag}] ENGINE FAILED: {type(e).__name__}: {str(e)[:120]}",
                      flush=True)
                continue

            # read the engine's own resolved KV cache config
            try:
                cfg = llm.llm_engine.vllm_config
                groups = cfg.cache_config
                kvc = getattr(llm.llm_engine, "kv_cache_config", None)
                if kvc is None:
                    kvc = getattr(getattr(llm.llm_engine, "engine_core", None),
                                  "kv_cache_config", None)
                specs = None
                if kvc is not None:
                    specs = []
                    for g in getattr(kvc, "kv_cache_groups", []) or []:
                        for s in getattr(g, "kv_cache_specs", []) or []:
                            specs.append(s)
                info = {"block_size": getattr(cfg.cache_config, "block_size", None),
                        "mamba_block_size": getattr(cfg.cache_config,
                                                    "mamba_block_size", None),
                        "mamba_cache_mode": getattr(cfg.cache_config,
                                                    "mamba_cache_mode", None),
                        "mamba_page_size_padded": getattr(cfg.cache_config,
                                                          "mamba_page_size_padded", None)}
                if specs:
                    mambas = [s for s in specs if isinstance(s, MambaSpec)]
                    if mambas:
                        m = mambas[0]
                        real = sum(1 for _ in ())  # placeholder
                        from math import prod
                        from vllm.utils.torch_utils import get_dtype_size
                        real = sum(prod(sh) * get_dtype_size(dt)
                                   for sh, dt in zip(m.shapes, m.dtypes))
                        info.update({
                            "mamba_num_speculative_blocks":
                                getattr(m, "num_speculative_blocks", None),
                            "mamba_page_size_real": real,
                            "mamba_page_size_padded": m.page_size_padded,
                            "headroom_bytes": (m.page_size_padded - real)
                            if m.page_size_padded is not None else None,
                        })
                print(f"[{tag}] {info}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{tag}] INTROSPECT FAILED: {type(e).__name__}: {str(e)[:140]}",
                      flush=True)
            del llm
            import torch
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
