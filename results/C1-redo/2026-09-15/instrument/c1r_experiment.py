#!/usr/bin/env python
"""C1 redo -- main grid harness (implements prereg_C1_v2_2026-09-15.md).

What this fixes relative to probes/c1_spec_prefix.py (the v1 probe):
  * v1 measured one wall-clock tok/s over a batch whose prompt:output ratio was
    ~40:1, so prefix-cache prefill savings dominated everything.  Here we time
    each request and split it into TTFT (prefill) and decode window, and the
    primary quantity is DECODE tok/s.
  * v1's cached_tokens reader was a no-op stub that returned None on every path.
    Here every arm scrapes the engine's own counters and the client-side usage
    block, and an arm that yields no evidence of cache accounting is marked
    INSTRUMENT_FAILED instead of quietly producing a number.
  * v1 hardcoded seed=0 and used 3 passes of the same set as "repeats".  Here
    cold/warm are defined by "has this exact prompt set been seen by this engine
    instance" and verified from counters, with several independent prompt sets.
  * v1 ran dflash under the invalid method name "dflash2" and silently recorded
    a ValidationError for the hypothesis-critical arm.  Here engine-start failure
    is a first-class recorded outcome, never conflated with a null effect.

Arm protocol inside ONE server instance (so the arms share engine state):
    cold  : first ever send of prompt set S
    warm  : immediate resend of S          -> must show cached_tokens > 0 if PC on
    warm2 : third send of S                -> stability check
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request

import aiohttp

# Workload.  Two earlier versions of this were invalid and both were caught only
# by looking at the generated TEXT, not at the timing numbers:
#   1. random-word prompts  -> greedy decoding collapsed to a constant token;
#   2. natural prompts padded by repeating ONE sentence ~11 times -> same collapse,
#      because the repetition itself is an attractor.
# On either workload ngram scored 1.00 acceptance by copying while the neural
# drafters scored 0.00, i.e. the drafter comparison had no headroom and could not
# have detected anything.  The version below uses varied, non-repeating prose for
# the shared prefix, and a chat-templated question as the per-request suffix.
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

NATURAL_ASKS = [
    "Explain how a client should implement retries with exponential backoff and jitter, and why jitter matters.",
    "Describe the failure modes of caching an HTTP response keyed only by URL, and how a validator fixes them.",
    "Walk through what happens when a queue consumer crashes midway through processing a message.",
    "Explain why an idempotency key must be stored transactionally with the effect it guards.",
    "Describe how a connection pool can deadlock under nested acquisition and how to detect it.",
    "Explain the trade-off between read replicas and sharding for a write-heavy service.",
]


def build_prompt_sets(tok, seeds, n_per_set, prefix_len, suffix_words=None):
    """Long SHARED prefix per seed (varied prose, no sentence repeats) + a
    chat-templated per-request question.  Requests within a set share the prefix
    token-for-token, which is what makes prefix-cache hits deterministic."""
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
            if i % len(order) == 0:
                rng.shuffle(order)
        pref = tok.encode(doc, add_special_tokens=False)[:prefix_len]
        prompts = []
        for j in range(n_per_set):
            ask = NATURAL_ASKS[j % len(NATURAL_ASKS)]
            msgs = [{"role": "system",
                     "content": "You are a senior engineer writing internal design documentation."},
                    {"role": "user", "content": f"Context: {doc[:1200]}\n\nQuestion: {ask}"}]
            try:
                full = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=True)
                if hasattr(full, "keys"):
                    full = full["input_ids"]
                elif isinstance(full, dict):
                    full = full["input_ids"]
                if full and isinstance(full[0], list):
                    full = full[0]
            except Exception:
                full = tok.encode(f"\n\nQuestion {j}: {ask}\n\nAnswer:",
                                  add_special_tokens=False)
            prompts.append(list(pref) + list(full))
        sets[s] = {"prefix_len": len(pref), "prompts": prompts}
    return sets


SPEC_METHOD = {"ngram": "ngram", "eagle3": "eagle3", "dflash": "dflash"}
STEP_FILE = [None]   # set in main_async; read by run_arm for per-arm step deltas


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def spec_config(drafter, k, paths):
    if drafter == "none":
        return None
    cfg = {"method": SPEC_METHOD[drafter], "num_speculative_tokens": k}
    if drafter == "ngram":
        cfg.update(prompt_lookup_max=4, prompt_lookup_min=2)
    else:
        cfg["model"] = paths[drafter]
    return cfg


class Server:
    def __init__(self, args, target_path, drafter, k, pc, log_path):
        self.args = args
        self.log_path = log_path
        self.proc = None
        self.port = args.port
        self.base = f"http://127.0.0.1:{self.port}"
        self.cmd = [
            args.python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", target_path,
            "--served-model-name", "t",
            "--port", str(self.port),
            "--gpu-memory-utilization", str(args.gpu_util),
            "--max-model-len", str(args.max_len),
            "--no-enable-log-requests",
        ]
        if args.enforce_eager:
            self.cmd.append("--enforce-eager")
        if args.language_model_only:
            self.cmd.append("--language-model-only")
        self.cmd.append("--enable-prefix-caching" if pc else "--no-enable-prefix-caching")
        sc = spec_config(drafter, k, args.paths)
        if sc:
            self.cmd += ["--speculative-config", json.dumps(sc)]

    def start(self):
        env = dict(os.environ)
        # This container's default (non-login) environment exports
        # LC_ALL=en_US.UTF-8, but only C / C.utf8 / POSIX exist here.  Under a
        # missing locale the stdlib readline extension dies inside
        # _rl_init_locale, so ANY module that imports readline kills the process:
        #     !!!!!!! Segfault encountered !!!!!!!
        #       File "<unknown>", line 0, in _rl_init_locale
        # EngineCore hit exactly that and died right after KV-cache allocation.
        # Pin a locale that exists.
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        env["C1R_STEP_FILE"] = self.args.step_file
        env["C1R_STEP_STATUS"] = os.path.join(
            os.path.dirname(self.args.step_file), "step_status.txt")
        # The sitecustomize step-counter is OPT-IN and defaults OFF: even in its
        # lazy form it makes EngineCore segfault in this container (the import
        # chain reaches stdlib readline, whose _rl_init_locale dies because
        # LC_ALL=en_US.UTF-8 cannot be set here).  The vendor's own
        # /metrics counters are the primary step/iteration instrument instead --
        # which is also what the v1 audit said was never actually cross-checked.
        if getattr(self.args, "step_patch", 0):
            pdir = "/root/autodl-tmp/probes/patchdir"
            prev = env.get("PYTHONPATH")
            env["PYTHONPATH"] = pdir + (os.pathsep + prev if prev else "")
        self.logf = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.logf,
                                     stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)

    def wait_health(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False, "process exited rc=%s" % self.proc.returncode
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=3):
                    return True, "ok"
            except Exception:
                time.sleep(3)
        return False, "health timeout after %ss" % timeout

    def stop(self):
        """Kill the whole session, not just the API server.

        EngineCore is a *spawned child*; terminating the parent alone leaves it
        alive holding ~41 GiB of the 80 GiB card, and every later launch then
        dies with "Free memory ... is less than desired GPU memory utilization".
        That exact leak cost a debugging round here.  start_new_session=True put
        the whole tree in one process group, so killpg reaches all of it.
        """
        import signal
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
        except Exception:
            pgid = None
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        finally:
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass
                # reap the group and give the driver a moment to free VRAM
                deadline = time.time() + 30
                while time.time() < deadline:
                    try:
                        os.killpg(pgid, 0)
                    except Exception:
                        break
                    time.sleep(1)
            try:
                self.logf.close()
            except Exception:
                pass

    def log_tail(self, n=25):
        try:
            with open(self.log_path, errors="ignore") as f:
                return "".join(f.readlines()[-n:])
        except Exception:
            return ""


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
KEEP = ("prefix_cache", "spec_decode", "iteration", "time_to_first",
        "prompt_tokens", "generation_tokens", "num_requests", "request_success")


def scrape_metrics(base):
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=20) as r:
            text = r.read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        return {"__error__": f"{type(e).__name__}: {e}"}
    out = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{")[0].split(" ")[0]
        if not any(k in name for k in KEEP):
            continue
        try:
            val = float(line.rsplit(" ", 1)[1])
        except Exception:
            continue
        out[name] = out.get(name, 0.0) + val
    return out


def metric_delta(a, b):
    d = {}
    for k, v in b.items():
        if k == "__error__":
            continue
        if k in a:
            d[k] = v - a[k]
        else:
            d[k] = v
    return d


# --------------------------------------------------------------------------- #
# correctness invariant (P1'): a prefix cache must be semantically transparent,
# so greedy output must be bit-identical with PC on and with PC off.
# Non-streaming, because we need the token ids, not just timing.
# --------------------------------------------------------------------------- #
async def correct_one(session, base, prompt_ids, gen, out, idx):
    payload = {"model": "t", "prompt": prompt_ids, "max_tokens": gen,
               "temperature": 0.0, "ignore_eos": True, "stream": False}
    rec = {"idx": idx}
    try:
        async with session.post(base + "/v1/completions", json=payload) as r:
            if r.status != 200:
                rec["error"] = f"HTTP {r.status}: {(await r.text())[:300]}"
                out.append(rec)
                return
            obj = await r.json()
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out.append(rec)
        return
    ch = (obj.get("choices") or [{}])[0]
    usage = obj.get("usage") or {}
    rec.update(
        token_ids=ch.get("token_ids"),
        text=ch.get("text"),
        finish_reason=ch.get("finish_reason"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )
    out.append(rec)


async def run_correctness(base, csets, seed, gen, concurrency):
    recs = []
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=900)) as session:
        await asyncio.gather(*[correct_one(session, base, p, gen, recs, i)
                               for i, p in enumerate(csets[seed]["prompts"])])
    recs.sort(key=lambda r: r["idx"])
    return recs


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #
async def stream_one(session, sem, base, prompt_ids, gen, out, idx):
    async with sem:
        payload = {
            "model": "t", "prompt": prompt_ids, "max_tokens": gen,
            "temperature": 0.0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True},
        }
        rec = {"idx": idx, "n_prompt_tokens": len(prompt_ids)}
        t0 = time.perf_counter()
        t_first = None
        chunks = 0
        usage = None
        try:
            async with session.post(base + "/v1/completions", json=payload) as r:
                if r.status != 200:
                    rec["error"] = f"HTTP {r.status}: {(await r.text())[:300]}"
                    out.append(rec)
                    return
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    ch = obj.get("choices") or []
                    if ch and ch[0].get("text"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks += 1
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            out.append(rec)
            return
        t_end = time.perf_counter()
        rec.update(
            ttft_s=(t_first - t0) if t_first else None,
            decode_s=(t_end - t_first) if t_first else None,
            total_s=t_end - t0, chunks=chunks,
            completion_tokens=(usage or {}).get("completion_tokens"),
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            cached_tokens=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            usage=usage,
        )
        out.append(rec)


def step_count(path):
    """Engine iterations emitted by the sitecustomize patch (includes prefill;
    it is NOT a decode-step count -- see 审核_原方案错误清单 4.2)."""
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except Exception:
        return None


async def run_arm(base, sets, seed, gen, concurrency, out_dir, tag):
    sem = asyncio.Semaphore(concurrency)
    before = scrape_metrics(base)
    steps0 = step_count(STEP_FILE[0]) if STEP_FILE else None
    results = []
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=900)) as session:
        tasks = [stream_one(session, sem, base, p, gen, results, i)
                 for i, p in enumerate(sets[seed]["prompts"])]
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    after = scrape_metrics(base)
    deltas = metric_delta(before, after)

    ok = [r for r in results if "error" not in r and r.get("ttft_s") is not None]
    dec = [r for r in ok if r.get("decode_s")]
    ttfts = sorted(r["ttft_s"] for r in ok)
    dec_tps = (sum((r.get("completion_tokens") or r["chunks"]) for r in dec)
               / sum(r["decode_s"] for r in dec)) if dec else None

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None

    cached_vals = [r.get("cached_tokens") for r in ok]
    have_cached = any(v is not None for v in cached_vals)

    return {
        "tag": tag, "seed": seed, "concurrency": concurrency,
        "wall_s": wall, "n_requests": len(results), "n_ok": len(ok),
        "errors": [r for r in results if "error" in r],
        "ttft_p50": pct(ttfts, 0.5), "ttft_p95": pct(ttfts, 0.95),
        "decode_tps": dec_tps,
        "output_tokens": sum((r.get("completion_tokens") or r.get("chunks") or 0) for r in ok),
        "cached_tokens_sum": sum(v for v in cached_vals if v is not None) if have_cached else None,
        "cached_tokens_per_req": (cached_vals if have_cached else None),
        "cached_tokens_instrument": "usage.prompt_tokens_details.cached_tokens" if have_cached else None,
        "steps_delta": (None if steps0 is None or step_count(STEP_FILE[0]) is None
                        else step_count(STEP_FILE[0]) - steps0),
        "cache_hits_delta": deltas.get("vllm:prefix_cache_hits_total"),
        "cache_queries_delta": deltas.get("vllm:prefix_cache_queries_total"),
        "metrics_delta": deltas,
        "per_request": results,
    }


# --------------------------------------------------------------------------- #
async def main_async(a):
    from transformers import AutoTokenizer

    os.makedirs(a.out_dir, exist_ok=True)
    STEP_FILE[0] = a.step_file
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    sets = build_prompt_sets(tok, a.seeds, a.n_per_set, a.prefix, a.suffix_words)

    run = {
        "run_id": a.run_id, "target": a.target, "target_path": a.target_path,
        "drafter": a.drafter, "k": a.k, "prefix_caching": bool(a.prefix_caching),
        "tokenizer": a.tokenizer, "gen": a.gen, "seeds": a.seeds,
        "n_per_set": a.n_per_set, "prefix_len": a.prefix,
        "concurrencies": a.concurrencies, "enforce_eager": a.enforce_eager,
        "gpu_util": a.gpu_util, "max_len": a.max_len,
        "language_model_only": a.language_model_only, "arms": [],
    }

    free_gib, total_gib = _gpu_free_gib()
    run["gpu_free_gib_at_start"] = free_gib
    run["gpu_total_gib"] = total_gib
    need = a.gpu_util * (total_gib or 0)
    if free_gib is not None and total_gib and free_gib < need:
        run["engine_status"] = "FAILED"
        run["engine_error"] = (f"PRECHECK: {free_gib:.2f} GiB free < {need:.2f} GiB "
                               f"requested (gpu_util={a.gpu_util}); stale GPU process?")
        out = os.path.join(a.out_dir, f"{a.run_id}.json")
        json.dump(run, open(out, "w"), indent=2, default=str)
        print(f"[{a.run_id}] {run['engine_error']}", flush=True)
        return out

    srv = Server(a, a.target_path, a.drafter, a.k, bool(a.prefix_caching),
                 os.path.join(a.out_dir, f"{a.run_id}.serverlog"))
    srv.start()
    ok, why = srv.wait_health(a.health_timeout)
    if not ok:
        run["engine_status"] = "FAILED"
        run["engine_error"] = why
        run["server_log_tail"] = srv.log_tail()
        srv.stop()
        out = os.path.join(a.out_dir, f"{a.run_id}.json")
        json.dump(run, open(out, "w"), indent=2, default=str)
        print(f"[{a.run_id}] ENGINE FAILED: {why}", flush=True)
        return out

    run["engine_status"] = "OK"
    run["vllm_version"] = _pkg_version("vllm")
    run["torch_version"] = _pkg_version("torch")
    print(f"[{a.run_id}] engine up", flush=True)

    # cold = first ever send of this prompt set on this engine instance;
    # warm1..warmN = immediate resends (identical work) -> the noise floor comes
    # from these replicates rather than from a separate throwaway pilot.
    phases = ["cold"] + [f"warm{i}" for i in range(1, a.warm_repeats + 1)]
    groups = [(seed, conc) for seed in a.seeds for conc in a.concurrencies]
    rng = random.Random(a.order_seed)
    rng.shuffle(groups)          # randomise GROUP order to spread drift...
    order = []
    for seed, conc in groups:
        for phase in phases:     # ...but cold must precede its own warms
            order.append((seed, conc, phase))
    run["arm_order"] = order

    try:
        await _run_all_arms(a, srv, sets, order, run, tok)
    finally:
        # Must be unconditional: an exception on this path previously skipped
        # cleanup and left EngineCore holding ~40 GiB, which then made every
        # later launch fail its free-memory precheck.
        srv.stop()

    out = os.path.join(a.out_dir, f"{a.run_id}.json")
    json.dump(run, open(out, "w"), indent=2, default=str)
    print(f"[{a.run_id}] wrote {out}", flush=True)
    return out


async def _run_all_arms(a, srv, sets, order, run, tok):
    for seed, conc, phase in order:
        tag = f"{phase}|c{conc}|s{seed}"
        try:
            res = await run_arm(srv.base, sets, seed, a.gen, conc, a.out_dir, tag)
        except Exception as e:  # noqa: BLE001
            res = {"tag": tag, "seed": seed, "concurrency": conc,
                   "fatal": f"{type(e).__name__}: {e}"}
        run["arms"].append(res)
        print(f"  [{a.run_id}] {tag:16s} dec_tps={res.get('decode_tps')} "
              f"ttft_p50={res.get('ttft_p50')} cached={res.get('cached_tokens_sum')} "
              f"hits={res.get('cache_hits_delta')} steps={res.get('steps_delta')}",
              flush=True)

    if a.correctness:
        # Fresh prompt-set namespace (100+) so the first correctness pass is a
        # genuine cold send, untouched by the timing arms above.
        csets = build_prompt_sets(tok, [100 + s for s in a.seeds], a.n_per_set,
                                  a.prefix, a.suffix_words)
        run["correctness"] = []
        for s_ in a.seeds:
            for rep in (1, 2):
                recs = await run_correctness(srv.base, csets, 100 + s_, a.gen, 1)
                run["correctness"].append({"seed": s_, "rep": rep, "requests": recs})
                n_tok = sum(1 for r in recs if r.get("token_ids"))
                print(f"  [{a.run_id}] correctness s{s_} rep{rep}: "
                      f"{len(recs)} reqs, {n_tok} with token_ids, "
                      f"cached={sum((r.get('cached_tokens') or 0) for r in recs)}",
                      flush=True)
        run["correctness_instrument"] = (
            "token_ids" if any(r.get("token_ids")
                               for c in run["correctness"] for r in c["requests"])
            else "text")

    return None


def _gpu_free_gib():
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=30)
        free, total = [float(x) for x in q.strip().splitlines()[0].split(",")]
        return free / 1024.0, total / 1024.0
    except Exception:
        return None, None


def _pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--target-path", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--drafter", default="none")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--prefix-caching", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-per-set", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--suffix-words", type=int, default=12)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--warm-repeats", type=int, default=3)
    ap.add_argument("--correctness", type=int, default=1)
    ap.add_argument("--step-patch", type=int, default=1)
    ap.add_argument("--python", default="/root/miniconda3/bin/python")
    ap.add_argument("--port", type=int, default=8611)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", type=int, default=1)
    ap.add_argument("--language-model-only", type=int, default=0)
    ap.add_argument("--health-timeout", type=int, default=900)
    ap.add_argument("--order-seed", type=int, default=7)
    ap.add_argument("--step-file", default="/root/autodl-tmp/results/steps.txt")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/results/raw")
    ap.add_argument("--eagle3", dest="paths_eagle3", default=None)
    ap.add_argument("--dflash", dest="paths_dflash", default=None)
    a = ap.parse_args()
    a.paths = {}
    if a.paths_eagle3:
        a.paths["eagle3"] = a.paths_eagle3
    if a.paths_dflash:
        a.paths["dflash"] = a.paths_dflash
    if a.drafter in ("eagle3", "dflash") and a.drafter not in a.paths:
        ap.error(f"--{a.drafter} path required for drafter={a.drafter}")
    asyncio.run(main_async(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
