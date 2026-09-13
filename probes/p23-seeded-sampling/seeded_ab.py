#!/usr/bin/env python3
"""p23 客户端 —— **带 seed vs 不带 seed** 的匹配对照（唯一变量 = SamplingType 是否 RANDOM_SEED）。

被检验的引擎自认缺口（vLLM 0.29.0 与 `main` **同一份代码、同一个 TODO**）：
  vllm/v1/sample/ops/topk_topp_sampler.py::random_sample
      q = empty_exponential_noise_like(probs, use_fp64_gumbel)
      # NOTE(woosuk): To batch-process the requests without their own seeds, which is the common case,
      # we first assume that every request does not have its own seed. Then, we overwrite the values
      # for the requests that have their own seeds.
      if len(generators) != probs.shape[0]:
          q.exponential_()                      # ← 整批一次 kernel（快路径）
      if generators:
          # TODO(woosuk): This can be slow because we handle each request
          # one by one. Optimize this.
          for i, generator in generators.items():
              q[i].exponential_(generator=generator)   # ← 每个带 seed 的请求一次 kernel
触发条件（gpu_model_runner.py）：
      if sampling_params.sampling_type == SamplingType.RANDOM_SEED: generator = torch.Generator(...)
      else: generator = None
⇒ **temperature>0 且不带 seed ⇒ RANDOM ⇒ 走快路径；temperature>0 且带 seed ⇒ RANDOM_SEED ⇒ 走逐请求循环。**
   所以 A/B 的**唯一变量**就是"请求里有没有 seed"，两边都真的在采样。

本客户端：固定 prompt 集、固定并发、固定输出长度，只改 seed 的有无；
用**同一批工作量的墙钟**作为吞吐口径，并用 /metrics 的 generation_tokens 确认两边做了一样多的活。
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def post(base, model, prompt, max_tokens, temperature, seed=None, timeout=900.0):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 1.0, "ignore_eos": True, "stream": False}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(base.rstrip("/") + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # 把服务端返回的正文带出来 —— 只统计"错了几个"而不打印原因，是上一版真正的缺陷
        detail = e.read().decode("utf-8", "replace")[:600]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
    return {"wall_s": time.perf_counter() - t0,
            "gen": (payload.get("usage") or {}).get("completion_tokens"),
            "finish": payload["choices"][0].get("finish_reason")}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:32110")
    ap.add_argument("--model", default="q3")
    ap.add_argument("--prompts", required=False)
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seeded", action="store_true", help="每个请求带一个不同的 seed")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        # 只测 body 构造与参数传递，不联网
        import io
        called = {}
        def fake_urlopen(req, timeout=None):
            called["body"] = json.loads(req.data.decode())
            class R:
                def __enter__(self_): return self_
                def __exit__(self_, *x): return False
                def read(self_): return json.dumps({"choices":[{"finish_reason":"length"}],
                                                    "usage":{"completion_tokens":7}}).encode()
            return R()
        urllib.request.urlopen = fake_urlopen  # type: ignore
        r = post("http://x", "m", "p", 5, 0.7, seed=42)
        assert called["body"]["seed"] == 42, called
        assert called["body"]["temperature"] == 0.7, called
        assert r["gen"] == 7, r
        r2 = post("http://x", "m", "p", 5, 0.7, seed=None)
        assert "seed" not in called["body"], called
        print("selftest OK")
        return 0
    if not a.prompts or not a.out:
        print("需要 --prompts 与 --out", file=sys.stderr); return 2

    prompts = []
    with open(a.prompts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
            if len(prompts) >= a.n_prompts:
                break
    print(f"  arm={'SEEDED' if a.seeded else 'unseeded'} n={len(prompts)} conc={a.concurrency} "
          f"T={a.temperature} max_tokens={a.max_tokens}", flush=True)

    t0 = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(post, a.base_url, a.model, p, a.max_tokens, a.temperature,
                          (i if a.seeded else None)): i for i, p in enumerate(prompts)}
        for fu in as_completed(futs):
            try:
                rows.append(fu.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"error": f"{type(exc).__name__}: {exc}"})
    wall = time.perf_counter() - t0
    ok = [r for r in rows if "error" not in r]
    if not ok:
        for r in rows[:2]:
            print("  ERR:", str(r.get("error"))[:400], flush=True)
    gen = sum(r.get("gen") or 0 for r in ok)
    res = {"arm": "seeded" if a.seeded else "unseeded", "n": len(prompts),
           "concurrency": a.concurrency, "temperature": a.temperature,
           "max_tokens": a.max_tokens, "wall_s": wall, "completed": len(ok),
           "errors": len(rows) - len(ok), "generation_tokens": gen,
           "output_throughput": (gen / wall) if wall else None,
           "request_throughput": (len(ok) / wall) if wall else None,
           "mean_latency_s": (sum(r["wall_s"] for r in ok) / len(ok)) if ok else None}
    json.dump(res, open(a.out, "w"), indent=1)
    print("  RESULT " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in res.items() if k != "arm"}), flush=True)
    if not ok:
        print("  ❌ 零成功请求 ⇒ 结果不可用"); return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
