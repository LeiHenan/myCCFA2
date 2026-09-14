#!/usr/bin/env python
"""C-3b: a BETTER implicit-invariant detector.

My first detector (probes/enumerate_aliasing.py) used 4 regex families over the
whole tree and found only 1 key-path hit -- because it was looking for textual
"aliasing" patterns rather than for INVARIANTS THAT ARE ASSUMED BUT NOT CHECKED.

This one is aimed at the pattern I actually found BY HAND in C-1f/C-1g:

    Falcon-H1 crashed because its `get_mamba_state_shape_from_config` computed a
    state shape WITHOUT reading `speculative_config.num_speculative_tokens`,
    while the consumer (MambaSpec.page_size_bytes) counted
    `num_speculative_blocks`.  Qwen3.5 does read it (qwen3_5.py:389-407), which is
    why block_size moved 528 -> 544 instead of asserting.

That is a general, checkable invariant:

    INV-SPEC-AWARE:  every function that sizes a cache structure must account for
                     the speculative draft depth if any consumer of that size does.

So this detector statically walks every model's
`get_mamba_state_shape_from_config` (+ the generic interfaces) and classifies:

    SPEC_AWARE      reads num_speculative_tokens and forwards it
    SPEC_UNAWARE    sizes state but never reads the spec depth   <-- RISK
    N/A             not a hybrid / does not size mamba state

and then cross-checks whether the consumer side counts speculative blocks.
A SPEC_UNAWARE producer + a spec-counting consumer is exactly the Falcon crash
condition, so each hit is a *candidate* defect with a known failure mode.

Zero GPU. Run against any vLLM source tree.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass, asdict, field

SPEC_TOKENS = ("num_speculative_tokens", "num_spec")
STATE_SIZERS = (
    "get_mamba_state_shape_from_config",
    "get_state_shape",
)
# module-level helper names that receive the draft depth
CALC_HELPERS = (
    "gated_delta_net_state_shape",
    "mamba2_state_shape",
    "mamba1_state_shape",
    "kda_state_shape",
    "linear_attention_state_shape",
    "short_conv_state_shape",
)


@dataclass
class Finding:
    model_file: str
    func: str
    kind: str                      # SPEC_AWARE | SPEC_UNAWARE
    reads_spec_token: bool
    forwards_spec: bool
    callee: str | None = None
    lineno: int = 0
    note: str = ""


def walk_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def src_of(node) -> str:
    return ast.unparse(node)


def detect_in_file(path: str) -> list[Finding]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    out: list[Finding] = []
    for fn in walk_functions(tree):
        if fn.name not in STATE_SIZERS:
            continue
        body = src_of(fn)
        reads = any(t in body for t in SPEC_TOKENS)
        callee = next((c for c in CALC_HELPERS if c in body), None)
        forwards = bool(callee) and reads
        if not callee:
            # some models build shapes inline; treat "sizes a state at all" as the
            # trigger, detected via mamba-ish vocabulary in the body
            mambaish = any(k in body for k in ("conv_state", "ssm_state",
                                               "temporal_state", "state_shape"))
            if not mambaish:
                continue
        kind = "SPEC_AWARE" if (reads and (forwards or callee is None)) else "SPEC_UNAWARE"
        out.append(Finding(
            model_file=os.path.basename(path), func=fn.name, kind=kind,
            reads_spec_token=reads, forwards_spec=forwards, callee=callee,
            lineno=fn.lineno,
            note=("forwards spec depth to " + callee) if forwards
                 else ("sizes state but never reads the spec depth")))
    return out


def consumer_counts_spec(vllm_root: str) -> dict:
    """Does the consumer side count speculative blocks?"""
    f = os.path.join(vllm_root, "v1", "kv_cache_interface.py")
    res = {"counts_spec_blocks": False, "evidence": []}
    try:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if "num_speculative_blocks" in line:
                    res["counts_spec_blocks"] = True
                    res["evidence"].append(f"v1/kv_cache_interface.py:{i}: {line.strip()[:110]}")
    except OSError:
        pass
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vllm_root", help="path to a vllm package dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    models_dir = os.path.join(a.vllm_root, "model_executor", "models")
    findings: list[Finding] = []
    if os.path.isdir(models_dir):
        for fn in sorted(os.listdir(models_dir)):
            if fn.endswith(".py"):
                findings.extend(detect_in_file(os.path.join(models_dir, fn)))

    cons = consumer_counts_spec(a.vllm_root)

    aware = [f for f in findings if f.kind == "SPEC_AWARE"]
    unaware = [f for f in findings if f.kind == "SPEC_UNAWARE"]

    print(f"vllm root: {a.vllm_root}")
    print(f"state-sizing functions found: {len(findings)}")
    print(f"  SPEC_AWARE  : {len(aware)}")
    print(f"  SPEC_UNAWARE: {len(unaware)}")
    print(f"\nconsumer side counts speculative blocks: {cons['counts_spec_blocks']}")
    for e in cons["evidence"][:4]:
        print("   ", e)

    print("\n=== SPEC-UNAWARE producers (each is a Falcon-style crash candidate) ===")
    print("   (only a defect if the consumer counts spec blocks -- see above)")
    for f in sorted(unaware, key=lambda x: x.model_file):
        print(f"  {f.model_file:<34} {f.func:<40} line {f.lineno}")
        print(f"      {f.note}")

    print("\n=== SPEC-AWARE producers (the safe pattern, for contrast) ===")
    for f in sorted(aware, key=lambda x: x.model_file):
        print(f"  {f.model_file:<34} callee={f.callee}")

    verdict = {
        "n_state_sizers": len(findings),
        "n_spec_aware": len(aware),
        "n_spec_unaware": len(unaware),
        "consumer_counts_spec_blocks": cons["counts_spec_blocks"],
        "unaware_models": [f.model_file for f in unaware],
    }
    if cons["counts_spec_blocks"] and len(unaware) >= 3:
        print(f"\n*** {len(unaware)} producer(s) fail to account for the draft depth while "
              f"the consumer counts it ***")
        print("    => this is the invariant class C-1f found by hand; each is a "
              "crash/mismatch candidate.")
        verdict["meets_C3_gate"] = len(unaware) >= 3
    else:
        verdict["meets_C3_gate"] = False

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"findings": [asdict(f) for f in findings],
                       "consumer": cons, "verdict": verdict}, fh, indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
