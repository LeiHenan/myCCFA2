#!/usr/bin/env python
"""E2-substantive: does the salt<->adapter-name collision change WHAT IS SERVED?

E2 established that a base request carrying cache_salt == <adapter name> and a
request using that adapter land on the SAME prefix-cache entries (both report
prefix-cache hits). Counting hits only proves shared bookkeeping. This script
asks the question that matters:

  If the salted base request runs FIRST, does the adapter request then get served
  the BASE model's continuation (i.e. the adapter silently not applied), or
  vice versa?

We compare against cold references for each domain, so every claim has a baseline.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


def gen(model, prompt, salt=None, mx=12):
    b = {"model": model, "prompt": prompt, "max_tokens": mx,
         "temperature": 0.0, "seed": 0}
    if salt:
        b["cache_salt"] = salt
    r = urllib.request.Request("http://127.0.0.1:8500/v1/completions",
                               data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=180) as f:
            d = json.loads(f.read())
    except urllib.error.HTTPError as e:
        msg = e.read()[:200].decode(errors="replace")
        return None, "HTTP %d: %s" % (e.code, msg)
    return d["choices"][0].get("text", ""), None


def main() -> int:
    P = "The capital of France is"
    ADAPTER = "adapterS"

    print("=== cold references (each domain on its own) ===")
    base_ref, _ = gen("qwen3-4b", P)
    lora_ref, e = gen(ADAPTER, P)
    print("  base   :", repr(base_ref))
    print("  adapter:", repr(lora_ref), e or "")
    differ = base_ref != lora_ref
    print("  adapter changes output:", differ)

    print("\n=== order 1: salted-base first, then adapter ===")
    s1, _ = gen("qwen3-4b", P, salt=ADAPTER)
    l_after, _ = gen(ADAPTER, P)
    print("  salted base #1 :", repr(s1))
    print("  adapter after  :", repr(l_after))
    got_base = (l_after == s1)

    print("\n=== order 2: adapter first, then salted-base ===")
    l_first, _ = gen(ADAPTER, P)
    s_after, _ = gen("qwen3-4b", P, salt=ADAPTER)
    print("  adapter first  :", repr(l_first))
    print("  salted after   :", repr(s_after))
    got_lora = (s_after == l_first)

    print("\n=== verdict ===")
    v = {
        "adapter_changes_output": differ,
        "adapter_served_base_output": differ and got_base,
        "saltedbase_served_adapter_output": differ and got_lora,
        "collision_has_behavioral_effect": differ and (got_base or got_lora),
    }
    print(json.dumps(v, indent=2))
    if not differ:
        print("\nVACUOUS: adapter does not change output; cannot detect mis-serving.")
    elif v["collision_has_behavioral_effect"]:
        print("\n*** BEHAVIORAL EFFECT CONFIRMED: one domain was served the other's")
        print("    continuation because their cache keys collided. ***")
    else:
        print("\nno behavioral effect: the collision affects cache accounting only;")
        print("the model still applied the correct weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
