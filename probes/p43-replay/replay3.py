#!/usr/bin/env python3
"""按**真实到达率**回放 + 每臂读取指标增量（修掉上轮两处缺陷）。

上轮两处缺陷：
 ① 我把到达率降采样 7× 到 3.65 req/s，但按实测重算原速率 23.66 req/s 本就能装进一张卡
    （在飞约 10 请求；1940 tok/s vs 卡上 ~3400 tok/s）⇒ 那次降采样把卡变成 99% 空闲，测的不是真实负载。
 ② /metrics 是**累计值**，两条不同配置的回放跑在同一进程 ⇒ 命中率无法按臂拆开。
    本版每臂前后各读一次计数器取差。
"""
import argparse, json, os, statistics as st, threading, time
from concurrent.futures import ThreadPoolExecutor

VOCAB_SAFE = 150000
def blk_tokens(b, bs=16):
    return [1000 + ((b * 7919 + k * 104729) % VOCAB_SAFE) for k in range(bs)]

def load_trace(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n: break
            try: rows.append(json.loads(line))
            except Exception: pass
    rows.sort(key=lambda r: float(r.get("timestamp", 0)))
    return rows

def read_counters(base):
    import requests
    try:
        txt = requests.get(base + "/metrics", timeout=8).text
    except Exception:
        return {}
    out = {}
    for ln in txt.splitlines():
        if ln.startswith("#"): continue
        for k in ("prefix_cache_queries_total", "prefix_cache_hits_total"):
            if ln.startswith("vllm:" + k):
                try: out[k] = float(ln.rsplit(" ", 1)[1])
                except Exception: pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--scales", default="1,3", help="逗号分隔的多个时间轴倍数（每臂一个）")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default="/root/ccfa_results/p43")
    a = ap.parse_args()
    import requests
    assert requests.get(a.base + "/v1/models", timeout=10).status_code == 200

    rows = load_trace(a.trace, a.n)
    t0 = float(rows[0].get("timestamp", 0))
    prompts = []
    for r in rows:
        ids = []
        for b in (r.get("hash_ids") or []): ids.extend(blk_tokens(int(b)))
        prompts.append(ids)
    print("trace %s: %d 条，平均 %d token，原始跨度 %.1f s"
          % (os.path.basename(a.trace), len(rows), sum(len(p) for p in prompts)/len(prompts),
             float(rows[-1]["timestamp"]) - t0), flush=True)

    results = []
    for scale in [float(x) for x in a.scales.split(",")]:
        for rep in range(a.reps):
            sem = threading.Semaphore(a.conc); lat, ttft = [], []
            def one(ids, O):
                with sem:
                    t = time.perf_counter()
                    try:
                        rr = requests.post(a.base + "/v1/completions", json={
                            "model": a.model, "prompt": ids, "max_tokens": max(1, O),
                            "temperature": 0.0, "ignore_eos": True, "stream": True},
                            stream=True, timeout=1800)
                        first = None
                        for line in rr.iter_lines():
                            if line and line.startswith(b"data: ") and b"[DONE]" not in line:
                                if first is None: first = time.perf_counter()
                        end = time.perf_counter()
                        if first: ttft.append((first - t) * 1000)
                        lat.append((end - t) * 1000)
                    except Exception:
                        pass
            c0 = read_counters(a.base)
            wall0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=a.conc) as ex:
                futs = []
                for i, r_ in enumerate(rows):
                    due = wall0 + (float(r_.get("timestamp", 0)) - t0) * scale
                    d = due - time.perf_counter()
                    if d > 0: time.sleep(min(d, 5.0))
                    futs.append(ex.submit(one, prompts[i], int(r_["output_length"])))
                for f in futs: f.result()
            wall = time.perf_counter() - wall0
            c1 = read_counters(a.base)
            q = c1.get("prefix_cache_queries_total",0) - c0.get("prefix_cache_queries_total",0)
            h = c1.get("prefix_cache_hits_total",0) - c0.get("prefix_cache_hits_total",0)
            d = dict(scale=scale, rep=rep, offered_rps=round(len(rows)/max(wall,1e-9),2),
                     wall_s=round(wall,1), done=len(lat),
                     ttft_p50=round(st.median(ttft),2) if ttft else None,
                     ttft_p95=round(sorted(ttft)[int(0.95*(len(ttft)-1))],2) if ttft else None,
                     e2e_p50=round(st.median(lat),1) if lat else None,
                     tok_per_s=round(sum(int(r["output_length"]) for r in rows)/max(wall,1e-9),0),
                     cache_hit_pct=round(100*h/q,2) if q>0 else None)
            results.append(d); print("  %s" % d, flush=True)

    os.makedirs(a.out, exist_ok=True)
    json.dump({"args": vars(a), "arms": results}, open(os.path.join(a.out, "load_response.json"), "w"), indent=1)
    print("REPLAY3_DONE")

if __name__ == "__main__":
    main()
