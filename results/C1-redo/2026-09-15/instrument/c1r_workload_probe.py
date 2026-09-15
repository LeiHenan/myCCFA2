#!/usr/bin/env python
"""C1 redo -- workload validity probe.

Why this exists: the first core-dense run produced acceptance_rate = 0.0000 for
BOTH neural drafters across 22,320 draft tokens, while ngram sat at 1.0000.  The
cause is the workload, not the drafter: the prompts were random word salads and
greedy decoding collapsed into a repeated single token ("IIIIIIII...").  ngram
predicts that trivially by copying; a neural drafter has no reason to.

A workload on which a drafter can never be accepted has no headroom to reveal a
prefix-cache x drafter interaction, so it cannot answer the pre-registered
question.  This probe measures, per candidate workload and per drafter:

  * acceptance rate / mean accept length (from the engine's own counters)
  * an output sample, so degeneracy is visible rather than inferred

It is an INSTRUMENT CHECK, not a result: nothing here enters the final numbers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c1r_experiment as X  # noqa: E402

ROOT = "/root/autodl-tmp"

# --- candidate workloads -----------------------------------------------------
DEGENERATE_SYS = (
    "You are a meticulous software engineering assistant embedded in a large "
    "codebase. Always answer concisely and cite the relevant module. "
)

NATURAL_SYS = (
    "You are a senior engineer writing internal design documentation. Explain "
    "trade-offs concretely, name the mechanisms involved, and prefer prose over "
    "lists. Assume the reader knows Python and HTTP but not this codebase. "
    "Ground every recommendation in an observable consequence. "
)

NATURAL_ASKS = [
    "Explain how a client should implement retries with exponential backoff and jitter, and why jitter matters.",
    "Describe the failure modes of caching an HTTP response keyed only by URL, and how a validator fixes them.",
    "Walk through what happens when a queue consumer crashes midway through processing a message.",
    "Explain why an idempotency key must be stored transactionally with the effect it guards.",
    "Describe how a connection pool can deadlock under nested acquisition and how to detect it.",
    "Explain the trade-off between read replicas and sharding for a write-heavy service.",
]


SENTENCES = [
    "The scheduler admits a request only after it has reserved space for every token it may generate.",
    "A block table maps logical positions to physical pages, so a forked sequence can share pages until it writes.",
    "Copy-on-write keeps the sharing cheap: the first divergent token triggers a private copy of just that block.",
    "Chunked prefill splits a long prompt into pieces so that decoding requests are never starved by a large prefill.",
    "The scheduler prefers to batch requests whose prompt lengths are similar, which keeps the batch shapes stable.",
    "Speculative decoding proposes several tokens and verifies them in one pass, so the target runs once per group.",
    "Acceptance depends on how well the drafter models the target, not on how large the target happens to be.",
    "A rejected proposal costs the same verification work as an accepted one, so a bad drafter is pure overhead.",
    "Prefix caching reuses the key and value blocks of a shared prefix, which turns repeated prefill into a lookup.",
    "Cache hits are counted in tokens, so a partial block match still saves most of the prefill computation.",
    "Eviction is driven by recency, which means a long idle period can quietly discard the blocks a workload depends on.",
    "The hash of a block covers its tokens and the blocks before it, so a match implies the whole chain matches.",
    "Metrics are exported per engine iteration, and an iteration may contain prefill, decode, or neither.",
    "A counter that only observes drafted steps will undercount the total work whenever drafting is skipped.",
    "Backpressure appears as a growing queue rather than as an error, so latency degrades before throughput does.",
    "A retry without jitter synchronises clients into bursts that arrive exactly when the service is least able to serve them.",
    "An idempotency key is only useful if it is written in the same transaction as the effect it is meant to guard.",
    "Connection pools fail under nested acquisition because the outer call holds a slot while waiting for an inner one.",
    "Read replicas scale reads but not writes, and replication lag turns into stale reads that look like logic bugs.",
    "Sharding scales writes but makes cross-shard transactions expensive, which is usually the real constraint.",
    "A lease makes progress possible without a leader, at the cost of requiring every participant to agree on time.",
    "Timeouts should be shorter than the caller's deadline, otherwise the caller gives up while the work continues.",
    "Structured logging is only useful when the correlation identifier crosses every process boundary.",
    "Validation belongs at the edge, but invariants belong next to the data they constrain.",
    "A cache keyed only by URL will serve one user's response to another whenever the response varies by header.",
    "A validator turns a conditional request into a cheap comparison, which is why it survives cache invalidation.",
    "The cheapest request is the one the client never sends, which is why an explicit freshness lifetime matters.",
    "Batching amortises fixed costs, but it also couples the latency of unrelated requests.",
    "A queue consumer that crashes midway must either be idempotent or must acknowledge only after the effect is durable.",
    "Visibility timeouts are a lease on a message, and a lease that is too short produces duplicate work.",
    "Dead letter queues are not a failure mode, they are a decision about which failures a human must see.",
    "A histogram of tokens per iteration is the only cheap way to see batching efficiency without sampling.",
    "Throughput and latency are not opposites; a saturated system loses both at the same time.",
    "The measurement instrument is part of the result, and an instrument that is wrong is worse than no instrument.",
    "A pre-registered prediction is only useful if the metric it names can actually detect the mechanism it describes.",
    "Two quantities that share a name but not a definition will eventually be divided by each other by mistake.",
    "Warm-up effects are easily mistaken for cache effects when every configuration starts a fresh process.",
    "A control arm that is never run cannot rule anything out, no matter how confident the reasoning feels.",
    "Comparing arms measured in different processes is not a comparison, it is two measurements and a story.",
    "The direction of an effect is easier to get wrong than its magnitude, because magnitude at least has units.",
    "If a threshold is chosen before the noise floor is known, the threshold encodes an assumption rather than a criterion.",
]


def build_natural(tok, seeds, n_per_set, prefix_len):
    """Long SHARED prefix per seed (varied prose, no sentence repeats), plus a
    per-request chat-templated question.

    The earlier version padded by repeating ONE sentence ~11 times.  That is a
    repetition attractor: greedy decoding latched onto it and emitted a constant
    token, which is why every workload looked degenerate.
    """
    import random as _random
    sets = {}
    for s in seeds:
        rng = _random.Random(4242 + s)
        order = SENTENCES[:]
        rng.shuffle(order)
        doc = ""
        i = 0
        while len(tok.encode(doc, add_special_tokens=False)) < prefix_len:
            doc += order[i % len(order)] + " "
            i += 1
            if i % len(order) == 0:      # reshuffle before any sentence repeats
                rng.shuffle(order)
        pref = tok.encode(doc, add_special_tokens=False)[:prefix_len]
        prompts = []
        for j in range(n_per_set):
            ask = NATURAL_ASKS[j % len(NATURAL_ASKS)]
            msgs = [{"role": "system", "content": "You are a senior engineer writing internal design documentation."},
                    {"role": "user", "content": f"Context: {doc[:1200]}\n\nQuestion: {ask}"}]
            try:
                full = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=True)
                # newer transformers returns a BatchEncoding, not a list
                if hasattr(full, "keys"):
                    full = full["input_ids"]
                elif isinstance(full, dict):
                    full = full["input_ids"]
                if full and isinstance(full[0], list):
                    full = full[0]
            except Exception as e:  # noqa: BLE001
                print(f"  chat_template failed ({type(e).__name__}: {e}); using plain text")
                full = tok.encode(f"\n\nQuestion {j}: {ask}\n\nAnswer:",
                                  add_special_tokens=False)
            prompts.append(list(pref) + list(full))
        sets[s] = {"prefix_len": len(pref), "prompts": prompts}
    return sets


async def probe(args, target_path, tok, drafter, k, sets, pc=1):
    import aiohttp
    a = argparse.Namespace(**vars(args))
    a.paths = {}
    if drafter == "eagle3":
        a.paths["eagle3"] = args.eagle3
    if drafter == "dflash":
        a.paths["dflash"] = args.dflash
    log = os.path.join(ROOT, "results", "raw", f"workloadprobe_{drafter}.serverlog")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    srv = X.Server(a, target_path, drafter, k, bool(pc), log)
    srv.start()
    ok, why = srv.wait_health(900)
    if not ok:
        print(f"[{drafter}] ENGINE FAILED {why}")
        print(srv.log_tail(12))
        srv.stop()
        return None
    try:
        sem = asyncio.Semaphore(1)
        before = X.scrape_metrics(srv.base)
        res = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
            await asyncio.gather(*[X.stream_one(sess, sem, srv.base, p, args.gen, res, i)
                                   for i, p in enumerate(sets[0]["prompts"])])
        after = X.scrape_metrics(srv.base)
        d = X.metric_delta(before, after)
        # sample text: one non-streaming call
        txt = None
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as sess:
                async with sess.post(srv.base + "/v1/completions", json={
                        "model": "t", "prompt": sets[0]["prompts"][0],
                        "max_tokens": args.gen, "temperature": 0.0,
                        "ignore_eos": True, "stream": False}) as r:
                    obj = await r.json()
                    txt = (obj.get("choices") or [{}])[0].get("text")
        except Exception as e:  # noqa: BLE001
            txt = f"<sample failed: {e}>"
        dt = d.get("vllm:spec_decode_num_draft_tokens_total")
        at = d.get("vllm:spec_decode_num_accepted_tokens_total")
        dr = d.get("vllm:spec_decode_num_drafts_total")
        dec = [r for r in res if r.get("decode_s")]
        tps = (sum(r["chunks"] for r in dec) / sum(r["decode_s"] for r in dec)) if dec else None
        out = {"drafter": drafter, "k": k, "draft_tokens": dt, "accepted_tokens": at,
               "drafts": dr,
               "acceptance_rate": (at / dt) if dt else None,
               "mean_accept_len": (1 + at / dr) if (at is not None and dr) else None,
               "decode_tps": tps, "sample": (txt or "")[:220]}
        errs = [r for r in res if "error" in r]
        if errs:
            print(f"[{drafter:7s}] REQUEST ERRORS ({len(errs)}/{len(res)}): "
                  f"{str(errs[0].get('error'))[:220]}")
        print(f"[{drafter:7s}] n_ok={sum(1 for r in res if 'error' not in r)} "
              f"acc={out['acceptance_rate']} len={out['mean_accept_len']} "
              f"dec_tps={tps and round(tps,1)} sample={out['sample'][:90]!r}")
        return out
    finally:
        srv.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-path", default=f"{ROOT}/models/Qwen3-4B")
    ap.add_argument("--tokenizer", default=f"{ROOT}/models/Qwen3-4B")
    ap.add_argument("--eagle3", default=f"{ROOT}/models/eagle3-qwen3-4b")
    ap.add_argument("--dflash", default=f"{ROOT}/models/dflash-qwen3-4b")
    ap.add_argument("--drafters", nargs="+", default=["none", "ngram", "eagle3", "dflash"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--out", default=f"{ROOT}/results/workload_probe.json")
    # Server() reads these off args:
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", type=int, default=1)
    ap.add_argument("--language-model-only", type=int, default=0)
    ap.add_argument("--python", default="/root/miniconda3/bin/python")
    ap.add_argument("--port", type=int, default=8612)
    ap.add_argument("--step-file", default=f"{ROOT}/results/steps.txt")
    ap.add_argument("--step-patch", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    sets = build_natural(tok, [0], 6, a.prefix)

    out = []
    for d in a.drafters:
        r = asyncio.run(probe(a, a.target_path, tok, d, a.k, sets))
        if r:
            out.append(r)
    json.dump(out, open(a.out, "w"), indent=2, default=str)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
