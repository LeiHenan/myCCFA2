#!/usr/bin/env python
"""Where does grammar-mask cost actually bite? Sweep concurrency x schema bound.

Hypothesis from the offline microbenchmark: fill cost is ~18us unconstrained but
~280us with a `maxLength` bound, paid once per batched step per request. At batch 1
that is ~3% of an 8ms/token step (invisible). At batch 32 it is ~9ms of host work
per step -- i.e. it should become the bottleneck. This script tests that.

Also measures the cost of grammar at all (structured vs unstructured), which is the
"grammar tax" an operator actually pays.

No server is started or restarted; we only send HTTP requests.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request


def schema_with(bound, structured=True):
    if not structured:
        return None
    content = {"type": "string", "description": "the file body"}
    if bound is not None:
        content["maxLength"] = bound
    return {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": content},
        "required": ["path", "content"],
        "additionalProperties": False,
    }


def call(url, model, schema, max_tokens, prompt, timeout=300):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "write_file", "schema": schema, "strict": True},
        }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    dt = time.perf_counter() - t0
    return dt, (payload.get("usage", {}) or {}).get("completion_tokens") or 0


PROMPT = (
    "Write a JavaScript function that reads a quantity from an input event, "
    "validates it, POSTs it to an inventory API, parses the JSON response and "
    "navigates to the inventory list page. Return only JSON matching the schema."
)


def run_arm(url, model, schema, conc, max_tokens, prompt):
    """Fire `conc` identical requests simultaneously; report aggregate tok/s."""
    out = []
    lock = threading.Lock()

    def worker():
        try:
            dt, ntok = call(url, model, schema, max_tokens, prompt)
            with lock:
                out.append((dt, ntok))
        except Exception as e:  # noqa: BLE001
            with lock:
                out.append((float("nan"), 0, f"{type(e).__name__}: {e}"))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    toks = sum(x[1] for x in out)
    lat = [x[0] for x in out if x[0] == x[0]]
    return {
        "wall_s": wall,
        "total_tokens": toks,
        "agg_tps": toks / wall if wall > 0 else 0.0,
        "lat_p50": statistics.median(lat) if lat else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concs", default="1,2,4,8,16")
    a = ap.parse_args()
    concs = [int(x) for x in a.concs.split(",")]

    arms = [
        ("unstructured", None),
        ("schema no bound", schema_with(None)),
        ("schema maxLength=100000", schema_with(100000)),
    ]

    print(f"server={a.url} max_tokens={a.max_tokens}")
    hdr = f"{'concurrency':>12} | " + " | ".join(f"{n:>22}" for n, _ in arms)
    print(hdr)
    print(f"{'':>12} | " + " | ".join(f"{'agg tok/s':>10}{'lat_p50':>12}" for _ in arms))
    print("-" * len(hdr))

    table = {}
    for c in concs:
        row = {}
        cells = []
        for name, sch in arms:
            r = run_arm(a.url, a.model, sch, c, a.max_tokens, PROMPT)
            row[name] = r
            cells.append(f"{r['agg_tps']:>10.1f}{r['lat_p50']:>12.3f}")
        table[c] = row
        print(f"{c:>12} | " + " | ".join(cells))

    print("\nratios vs 'schema no bound' (aggregate throughput):")
    for c in concs:
        nb = table[c]["schema no bound"]["agg_tps"]
        ml = table[c]["schema maxLength=100000"]["agg_tps"]
        uns = table[c]["unstructured"]["agg_tps"]
        if nb > 0:
            print(f"  conc={c:<3} maxLength/no_bound={ml/nb:6.3f}x   "
                  f"grammar_tax(no_bound vs unstructured)={nb/uns if uns else 0:6.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
