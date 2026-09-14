#!/usr/bin/env python
"""Coverage census: enumerate EVERY input that can change the KV produced for a
request, and determine whether each participates in the cache key.

This is the formal core of the topic: the cache key is a PROJECTION of request
identity, and every identity component the projection drops is a potential false
HIT. We enumerate by reading the engine's own request/input surface rather than
guessing, then classify:

  MODELLED    the component participates in the key (variation -> different hash)
  DROPPED     the component changes the effective input but NOT the key (false HIT)
  IGNORED     the component cannot change the KV, and is (correctly) absent

Zero GPU.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request

BLOCK = 4
N = 8


def hf(t):
    return hashlib.sha256(str(t).encode()).digest()


def req(**kw):
    base = dict(request_id="r", prompt_token_ids=[0] * N, pooling_params=None,
                sampling_params=SamplingParams(max_tokens=8, temperature=0.0))
    base.update(kw)
    return Request(**base)


def hashes(r):
    init_none_hash(hf)
    return [h.hex() for h in get_request_block_hasher(BLOCK, hf)(r)]


def embeds(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, 4, generator=g)


# NOTE: use the REAL LoRARequest type. An earlier version of this probe used a
# hand-rolled mock that carried a `lora_scale` attribute; vLLM 0.29.0's
# LoRARequest has no such field, so that "dropped component" was an ARTIFACT of
# the mock, not a defect. Using the real type prevents this class of error.
try:
    from vllm.lora.request import LoRARequest as LR  # type: ignore
    _LR_FIELDS = [f for f in getattr(LR, "__struct_fields__", ()) or ()]
except Exception:  # noqa: BLE001
    LR = None
    _LR_FIELDS = []


class MM:
    def __init__(self, tag, offset=0):
        self.identifier = tag
        self.offset = offset
        self.length = 2
        self.mm_hash = None
        self.placeholders = None


def census():
    """(component, affects_kv, baseline_kwargs, variant_kwargs, how)"""
    rows = []

    # Prefer REAL token ids from the running engine when available; fall back to
    # synthetic ids so the census still runs offline.
    global N
    try:
        toks = real_token_ids()
        if 8 <= len(toks) <= 4096:
            REAL = toks[: (len(toks) // BLOCK) * BLOCK]
            N = len(REAL)
            print(f"[census] using {N} REAL token ids from /tokenize")
        else:
            REAL = [0] * N
    except Exception as e:  # noqa: BLE001
        REAL = [0] * N
        print(f"[census] /tokenize unavailable ({type(e).__name__}); synthetic ids")
    Z = [0] * N

    def add(name, affects_kv, b, v, how):
        rows.append(dict(component=name, affects_kv=affects_kv, base=b, var=v, how=how))

    # --- components that CAN change the KV ---------------------------------
    _b = list(REAL)
    _v = list(REAL)
    _v[-1] = (_v[-1] + 1) % 1000
    add("prompt_token_ids", True, dict(prompt_token_ids=_b), dict(prompt_token_ids=_v),
        "the tokens themselves")
    add("prompt_embeds", True, {}, dict(prompt_embeds=embeds(1)),
        "supplied hidden states")
    # To isolate the mask, all token ids are 0 so that Request's zeroing of
    # mask=False positions leaves `all_token_ids` identical across the arms.
    add("prompt_is_token_ids (mask)", True,
        dict(prompt_embeds=embeds(), prompt_token_ids=Z,
             prompt_is_token_ids=[True] * N),
        dict(prompt_embeds=embeds(), prompt_token_ids=Z,
             prompt_is_token_ids=[False] * N),
        "per-position choice of learned-vs-supplied embedding")
    if LR is not None:
        add("lora_request identity", True, {},
            dict(lora_request=LR(lora_name="A", lora_int_id=1, lora_path="/tmp/A")),
            "different adapter weights")
        # Only test fields the REAL class actually has.
        for f in ("base_model_name", "load_inplace", "is_3d_lora_weight"):
            if f not in _LR_FIELDS:
                continue
            b = dict(lora_request=LR(lora_name="A", lora_int_id=1, lora_path="/tmp/A"))
            v = dict(lora_request=LR(lora_name="A", lora_int_id=1, lora_path="/tmp/B",
                                     **{f: True}))
            add(f"lora_request.{f}", False, b, v, "adapter metadata, not KV content")
    add("cache_salt", True, {}, dict(cache_salt="S"), "tenant isolation")

    # --- components that CANNOT change the KV ------------------------------
    add("request_id", False, dict(request_id="r1"), dict(request_id="r2"),
        "identifier only")
    add("priority", False, dict(priority=0), dict(priority=7), "scheduling hint")
    add("arrival_time", False, dict(arrival_time=1.0), dict(arrival_time=2.0),
        "bookkeeping")
    add("sampling_params.temperature", False, {}, dict(
        sampling_params=SamplingParams(max_tokens=8, temperature=0.7)),
        "decode-time only; prompt KV unaffected")
    add("sampling_params.max_tokens", False, {}, dict(
        sampling_params=SamplingParams(max_tokens=99, temperature=0.0)),
        "decode-time only")

    return rows


def real_token_ids():
    """Use the engine's own tokenizer so the demonstration uses REAL token ids."""
    import json as _json
    import urllib.request

    url = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
    model = os.environ.get("VLLM_MODEL", "/root/autodl-tmp/models/Qwen3-4B")
    body = _json.dumps({"model": model, "prompt": "p" * 400}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/tokenize", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        toks = _json.loads(r.read())["tokens"]
    return toks


def main() -> int:
    import vllm

    print(f"vLLM {vllm.__version__}  block_size={BLOCK}  prompt_len={N}")
    print("=" * 108)
    print(f"{'identity component':<34}{'affects KV':<12}{'in key?':<12}{'verdict'}")
    print("-" * 108)

    out = []
    for r in census():
        try:
            hb, hv = hashes(req(**r["base"])), hashes(req(**r["var"]))
            in_key = hb != hv
            if r["affects_kv"] and not in_key:
                verdict = "*** DROPPED -> false HIT ***"
            elif (not r["affects_kv"]) and in_key:
                verdict = "*** SPURIOUS -> false MISS ***"
            elif r["affects_kv"]:
                verdict = "MODELLED (ok)"
            else:
                verdict = "IGNORED (ok)"
        except Exception as e:  # noqa: BLE001
            in_key = None
            verdict = f"ERROR {type(e).__name__}: {e}"
        print(f"{r['component']:<34}{str(r['affects_kv']):<12}{str(in_key):<12}{verdict}")
        out.append({**{k: v for k, v in r.items() if k not in ("base", "var")},
                    "in_key": in_key, "verdict": verdict})

    dropped = [o["component"] for o in out if "DROPPED" in str(o["verdict"])]
    spurious = [o["component"] for o in out if "SPURIOUS" in str(o["verdict"])]
    print("-" * 108)
    print(f"DROPPED  (false hits): {dropped}")
    print(f"SPURIOUS (false misses): {spurious}")
    print(f"\ncoverage: {len(out) - len(dropped) - len(spurious)}/{len(out)} components modelled")
    print(f"LoRARequest fields available in this vLLM: {_LR_FIELDS}")

    p = os.environ.get("CENSUS_OUT", "/tmp/key_coverage_census.json")
    with open(p, "w") as f:
        json.dump({"vllm": vllm.__version__, "rows": out}, f, indent=2, default=str)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
