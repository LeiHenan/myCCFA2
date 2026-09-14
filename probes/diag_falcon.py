#!/usr/bin/env python
"""Diagnose why Falcon-H1-3B fails to initialise in this vLLM build."""
import os
import traceback

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

from vllm import LLM

MODELS = [
    ("Falcon-H1-3B", "/root/autodl-tmp/models/Falcon-H1-3B"),
]

for name, path in MODELS:
    for eager in (True,):
        try:
            llm = LLM(model=path, gpu_memory_utilization=0.42,
                      max_model_len=2048, enforce_eager=eager,
                      dtype="bfloat16", language_model_only=True)
            print(f"[{name} eager={eager}] OK")
            del llm
        except Exception as e:  # noqa: BLE001
            print(f"[{name} eager={eager}] {type(e).__name__}: {str(e)[:400]}")
            tb = traceback.format_exc()
            # surface the innermost real cause
            for line in tb.strip().splitlines():
                if any(k in line for k in ("Error", "assert", "raise",
                                           "not supported", "Unsupported")):
                    print("   >>", line.strip()[:200])
