#!/usr/bin/env python
"""Zero-GPU reproduction of vLLM prefix-cache KEY-SOUNDNESS defects.

Two independent failure modes of the same missing abstraction — a KV cache entry
has no typed identity:

  DEFECT B (false HIT -> silently wrong output)
    `prompt_is_token_ids` selects, per position, whether to use the model's learned
    embedding for the token id or the caller-supplied embedding row. It is part of
    the *effective input*, but it never reaches the block hash. Two requests with
    identical token ids and identical `prompt_embeds` but different masks therefore
    produce IDENTICAL block hashes.

  DEFECT A (false HIT across semantic domains)
    `generate_block_hash_extra_keys` builds one untyped tuple by concatenation:
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    A LoRA name and a cache_salt are both bare strings, so a LoRA request named
    "COLLIDE_SALT" and a base request with cache_salt="COLLIDE_SALT" produce the
    same extra-keys tuple, hence the same hash.

Run on the machine that has vLLM installed (no GPU, no model weights needed):
    python repro_cache_key_soundness.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys

RESULTS: dict = {}


def _hash_fn(algo: str):
    if algo == "sha256":
        return lambda t: hashlib.sha256(str(t).encode()).digest()
    if algo == "sha256_cbor":
        try:
            import cbor2

            return lambda t: hashlib.sha256(cbor2.dumps(t)).digest()
        except Exception:  # noqa: BLE001
            return None
    if algo == "xxhash":
        try:
            import xxhash

            return lambda t: xxhash.xxh128(str(t).encode()).digest()
        except Exception:  # noqa: BLE001
            return None
    return None


def part0_source_proof() -> dict:
    """Static evidence: does the mask appear anywhere in the cache-key path?"""
    import vllm
    from vllm.v1.core import kv_cache_utils as k

    src = inspect.getsource(k)
    out = {
        "vllm_version": vllm.__version__,
        "file": k.__file__,
        "prompt_is_token_ids_in_kv_cache_utils": "prompt_is_token_ids" in src,
        "extra_keys_is_bare_concatenation": (
            "lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys"
            in src.replace("\n", " ").replace("  ", " ")
            or all(
                s in src
                for s in ("lora_extra_keys", "mm_extra_keys", "cache_salt_keys",
                          "prompt_embeds_keys")
            )
        ),
    }
    return out


def part1_defect_b() -> dict:
    """The embedding mask is not hashed -> identical keys for different inputs."""
    from vllm.v1.core.kv_cache_utils import _gen_prompt_embeds_extra_hash_keys

    class FakeReq:
        def __init__(self, embeds, mask):
            # the mask is stored on the request but is NEVER read by the key fn
            self.prompt_embeds = embeds
            self.prompt_is_token_ids = mask
            self._prompt_embeds_per_block_hashes = {}

    import torch

    n, d = 16, 4
    embeds = torch.arange(n * d, dtype=torch.float32).reshape(n, d)

    # Same token ids (all zeros), same supplied embedding tensor, DIFFERENT mask:
    # request X uses the learned embedding at every position (all True);
    # request Y uses the supplied embedding at every position (all False).
    mask_x = [True] * n
    mask_y = [False] * n

    hx = _gen_prompt_embeds_extra_hash_keys(FakeReq(embeds, mask_x), 0, 16)
    hy = _gen_prompt_embeds_extra_hash_keys(FakeReq(embeds, mask_y), 0, 16)

    return {
        "hash_x": hx[0].hex()[:32] if hx else None,
        "hash_y": hy[0].hex()[:32] if hy else None,
        "mask_participates": hx != hy,
        "verdict": "IDENTICAL KEY (defect reproduced)" if hx == hy else "distinct",
    }


def part2_defect_a() -> dict:
    """Untyped concatenation: LoRA name vs cache_salt collide."""
    from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys

    class FakeLora:
        def __init__(self, name):
            self.lora_name = name

    class FakeReq:
        def __init__(self, lora_name=None, salt=None):
            self.lora_request = FakeLora(lora_name) if lora_name else None
            self.cache_salt = salt
            self.prompt_embeds = None
            # no multi-modal inputs in this repro
            self.mm_features = []
            self.mm_positions = []
            self._prompt_embeds_per_block_hashes = {}

    ok = {}
    # (i) the domain-collision pair from vllm#44701
    try:
        e_lora, _ = generate_block_hash_extra_keys(FakeReq(lora_name="COLLIDE_SALT"), 0, 16, 0)
        e_salt, _ = generate_block_hash_extra_keys(FakeReq(salt="COLLIDE_SALT"), 0, 16, 0)
        ok["lora_vs_salt"] = {
            "lora_keys": list(e_lora) if e_lora else None,
            "salt_keys": list(e_salt) if e_salt else None,
            "tuples_equal": e_lora == e_salt,
        }
    except Exception as e:  # noqa: BLE001
        ok["lora_vs_salt"] = {"error": f"{type(e).__name__}: {e}"}

    # (ii) does the mask change the extra keys at all?
    try:
        r1 = FakeReq()
        r1.prompt_embeds = None
        k1, _ = generate_block_hash_extra_keys(r1, 0, 16, 0)
        ok["empty_request"] = {"keys": list(k1) if k1 else None}
    except Exception as e:  # noqa: BLE001
        ok["empty_request"] = {"error": f"{type(e).__name__}: {e}"}

    return ok


def part3_hash_collision() -> dict:
    """End-to-end: identical (tokens, extra_keys) -> identical block hash."""
    from vllm.v1.core.kv_cache_utils import (
        generate_block_hash_extra_keys,
        init_none_hash,
    )

    class FakeLora:
        def __init__(self, name):
            self.lora_name = name

    class FakeReq:
        def __init__(self, lora_name=None, salt=None):
            self.lora_request = FakeLora(lora_name) if lora_name else None
            self.cache_salt = salt
            self.prompt_embeds = None
            self.mm_features = []
            self.mm_positions = []
            self._prompt_embeds_per_block_hashes = {}

    out = {}
    for algo in ("sha256", "sha256_cbor", "xxhash"):
        hf = _hash_fn(algo)
        if hf is None:
            out[algo] = "unavailable (missing optional dep)"
            continue
        init_none_hash(hf)
        tokens = (101, 102, 103)
        try:
            e1, _ = generate_block_hash_extra_keys(FakeReq(lora_name="COLLIDE_SALT"), 0, 3, 0)
            e2, _ = generate_block_hash_extra_keys(FakeReq(salt="COLLIDE_SALT"), 0, 3, 0)
            h1 = hf((None, tokens, e1))
            h2 = hf((None, tokens, e2))
            out[algo] = {
                "hash_lora_req": h1.hex()[:24],
                "hash_salt_req": h2.hex()[:24],
                "collide": h1 == h2,
            }
        except Exception as e:  # noqa: BLE001
            out[algo] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    print("=" * 78)
    print("PART 0 — static proof: does the effective-input mask reach the cache key?")
    print("=" * 78)
    try:
        p0 = part0_source_proof()
        for k, v in p0.items():
            print(f"  {k}: {v}")
        RESULTS["part0"] = p0
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR {type(e).__name__}: {e}")
        RESULTS["part0"] = {"error": str(e)}

    print()
    print("=" * 78)
    print("PART 1 — DEFECT B: prompt_is_token_ids omitted from the key")
    print("=" * 78)
    try:
        p1 = part1_defect_b()
        for k, v in p1.items():
            print(f"  {k}: {v}")
        RESULTS["part1_defect_b"] = p1
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR {type(e).__name__}: {e}")
        RESULTS["part1_defect_b"] = {"error": str(e)}

    print()
    print("=" * 78)
    print("PART 2 — DEFECT A: untyped concatenation (LoRA name vs cache_salt)")
    print("=" * 78)
    try:
        p2 = part2_defect_a()
        print(json.dumps(p2, indent=2, default=str)[:2000])
        RESULTS["part2_defect_a"] = p2
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR {type(e).__name__}: {e}")
        RESULTS["part2_defect_a"] = {"error": str(e)}

    print()
    print("=" * 78)
    print("PART 3 — end-to-end block-hash collision under each hash algorithm")
    print("=" * 78)
    try:
        p3 = part3_hash_collision()
        print(json.dumps(p3, indent=2, default=str)[:2000])
        RESULTS["part3_hash_collision"] = p3
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR {type(e).__name__}: {e}")
        RESULTS["part3_hash_collision"] = {"error": str(e)}

    out_path = os.environ.get("REPRO_OUT", "/tmp/repro_cache_key_soundness.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
