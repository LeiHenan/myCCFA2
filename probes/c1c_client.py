#!/usr/bin/env python
"""C-1c client: run a real-text multi-turn workload against a vLLM OpenAI server and
read BOTH throughput and the engine's own speculative-decoding acceptance counters.

Counters scraped from /metrics before and after:
    vllm:spec_decode_num_accepted_tokens_total
    vllm:spec_decode_num_draft_tokens_total
    vllm:spec_decode_num_drafts
Derived (per the engine's own documented PromQL):
    accept_rate        = d_accepted / d_draft_tokens
    mean_accept_len    = 1 + d_accepted / d_drafts

Output length is pinned with ignore_eos so throughput is not confounded by length.
The workload is issued as concurrent requests (ThreadPool) so we also cover the
"concurrency > 1" arm that the C-1 pre-registration listed but was never run.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import threading
import time
import urllib.request

SYSTEM = (
    "You are a meticulous software engineering assistant embedded in a large "
    "codebase. Always answer concisely and cite the relevant module. Follow the "
    "team conventions: prefer explicit names, avoid global state, and never "
    "silently swallow errors. "
)
TURNS = [
    "Summarise what the inventory module is responsible for.",
    "Which function would I change to add a retry on the POST?",
    "What could go wrong if the quantity is negative?",
    "Suggest a unit test for the navigation step.",
    "Rewrite the error handling to log and re-raise.",
    "What is the risk of calling the API twice for the same event?",
]

SPEC_KEYS = (
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_drafts",
)


def scrape(url: str) -> dict:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=20) as r:
            txt = r.read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k in SPEC_KEYS:
            if line.startswith(k):
                try:
                    out[k] = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    pass
    return out


def post(url, body, timeout=300):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600")
    ap.add_argument("--model", default="m")
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--n-convs", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", default="/root/autodl-tmp/c1c_cell.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    filler = tok.encode(
        "The inventory service exposes a small HTTP API and a set of pure "
        "helpers, described below in detail for onboarding purposes. ",
        add_special_tokens=False)
    prompts = []
    for c in range(a.n_convs):
        sys_ids = tok.encode(SYSTEM, add_special_tokens=False)
        while len(sys_ids) < a.prefix:
            sys_ids = sys_ids + filler
        sys_ids = sys_ids[:a.prefix]
        acc = list(sys_ids)
        for k, t in enumerate(TURNS):
            acc = acc + tok.encode(f"\nUser: {t}\nAssistant:",
                                   add_special_tokens=False)
            prompts.append(list(acc))
            acc = acc + tok.encode(
                " Understood: the module validates input, posts to the API, "
                f"parses the response and navigates (turn {k}, variant {c}).",
                add_special_tokens=False)
    print(f"requests={len(prompts)} concurrency={a.concurrency}", flush=True)

    def one(p):
        return post(a.url, {"model": a.model, "prompt": p,
                            "max_tokens": a.gen, "temperature": 0.0,
                            "ignore_eos": True, "seed": 0})

    def run_pass():
        if a.concurrency <= 1:
            return [one(p) for p in prompts]
        lock, res = threading.Lock(), []
        chunks = [prompts[i::a.concurrency] for i in range(a.concurrency)]

        def worker(ch):
            for p in ch:
                try:
                    r = one(p)
                except Exception:  # noqa: BLE001
                    r = None
                with lock:
                    res.append(r)
        ts = [threading.Thread(target=worker, args=(c,)) for c in chunks]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return res

    tps_runs, accs = [], []
    for pi in range(a.passes):
        before = scrape(a.url)
        t0 = time.perf_counter()
        outs = run_pass()
        dt = time.perf_counter() - t0
        after = scrape(a.url)
        lens = [(o or {}).get("usage", {}).get("completion_tokens", 0)
                for o in outs if o]
        ntok = sum(lens)
        tps = ntok / dt if dt > 0 else 0.0
        print(f"  [len] n={len(lens)} min={min(lens) if lens else 0} "
              f"max={max(lens) if lens else 0} "
              f"median={st.median(lens) if lens else 0}", flush=True)
        tps_runs.append(tps)
        d = {k: after.get(k, 0) - before.get(k, 0) for k in SPEC_KEYS}
        acc = (d[SPEC_KEYS[0]] / d[SPEC_KEYS[1]]
               if d.get(SPEC_KEYS[1]) else None)
        mal = (1 + d[SPEC_KEYS[0]] / d[SPEC_KEYS[2]]
               if d.get(SPEC_KEYS[2]) else None)
        accs.append((acc, mal))
        print(f"  pass{pi}: {dt:.2f}s {tps:7.1f} tok/s  "
              f"draft={d.get(SPEC_KEYS[1],0):.0f} accepted={d.get(SPEC_KEYS[0],0):.0f} "
              f"drafts={d.get(SPEC_KEYS[2],0):.0f}  acc_rate={acc} mean_acc_len={mal}",
              flush=True)

    ar = [x for x, _ in accs if x is not None]
    ml = [y for _, y in accs if y is not None]
    res = {
        "tok_per_s_median": round(st.median(tps_runs), 2) if tps_runs else None,
        "tok_per_s_runs": [round(x, 1) for x in tps_runs],
        "accept_rate_median": round(st.median(ar), 4) if ar else None,
        "mean_accept_len_median": round(st.median(ml), 4) if ml else None,
        "n_accept_samples": len(ar),
        "concurrency": a.concurrency,
    }
    print(json.dumps(res, indent=2))
    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "result": res}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
