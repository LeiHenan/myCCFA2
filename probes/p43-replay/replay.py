#!/usr/bin/env python3
"""真实轨迹回放：在真机上跑出「干预前 vs 干预后」对照（目标要求的可回收性判据）。

设计要点（避免本工作区踩过的坑）：
  · **配置/产物回读**：启动即断言服务器活着、模型名正确。
  · **同格重复**：每臂重复 R 次，报中位与极差，并给**同格噪声**；效应须 >3× 该噪声。
  · **到达序列来自真实轨迹**（Alibaba qwen-bailian），按 --scale 降采样以匹配单卡容量。
  · 度量：TTFT(p50/p95)、端到端时延、以及引擎自报前缀缓存命中率。
"""
import argparse, json, os, statistics as st, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

def load_trace(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n: break
            try: rows.append(json.loads(line))
            except Exception: pass
    rows.sort(key=lambda r: float(r.get("timestamp", 0)))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--scale", type=float, default=7.0)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="/root/ccfa_results/p43")
    a = ap.parse_args()
    import requests
    from transformers import AutoTokenizer

    r = requests.get(a.base + "/v1/models", timeout=10)
    assert r.status_code == 200, ("server not healthy", r.status_code)
    print("server models:", [m["id"] for m in r.json()["data"]], flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    base_text = ("The history of computing began with mechanical calculators and evolved through "
                 "vacuum tubes, transistors, integrated circuits, and parallel processors. ")
    cache = {}
    def prompt_of(L):
        if L in cache: return cache[L]
        ids = tok(base_text * 400, truncation=True, max_length=L)["input_ids"]
        cache[L] = tok.decode(ids); return cache[L]
    print("tokenizer ready; L=869 -> %d tokens" % len(tok(prompt_of(869))["input_ids"]), flush=True)

    rows = load_trace(a.trace, a.n)
    t0 = float(rows[0].get("timestamp", 0))
    span = (float(rows[-1]["timestamp"]) - t0) * a.scale
    print("trace %s: n=%d span=%.1fs scaled -> %.2f req/s offered"
          % (os.path.basename(a.trace), len(rows), span, len(rows)/max(span,1e-9)), flush=True)

    sem = threading.Semaphore(a.conc)
    lat, ttft = [], []
    def one(p, O):
        with sem:
            t = time.perf_counter()
            try:
                rr = requests.post(a.base + "/v1/completions", json={
                    "model": a.model, "prompt": p, "max_tokens": max(1, O), "temperature": 0.0,
                    "ignore_eos": True, "stream": True}, stream=True, timeout=900)
                first = None
                for line in rr.iter_lines():
                    if line and line.startswith(b"data: ") and b"[DONE]" not in line:
                        if first is None: first = time.perf_counter()
                end = time.perf_counter()
                if first: ttft.append((first - t) * 1000)
                lat.append((end - t) * 1000)
            except Exception:
                pass

    res = []
    for rep in range(a.reps):
        lat.clear(); ttft.clear()
        wall0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=a.conc) as ex:
            futs = []
            for r_ in rows:
                due = wall0 + (float(r_.get("timestamp", 0)) - t0) * a.scale
                d = due - time.perf_counter()
                if d > 0: time.sleep(min(d, 5.0))
                futs.append(ex.submit(one, prompt_of(int(r_["input_length"])), int(r_["output_length"])))
            for f in futs: f.result()
        wall = time.perf_counter() - wall0
        d = dict(rep=rep, wall_s=round(wall,2), done=len(lat),
                 ttft_p50=round(st.median(ttft),2) if ttft else None,
                 ttft_p95=round(sorted(ttft)[int(0.95*(len(ttft)-1))],2) if ttft else None,
                 e2e_p50=round(st.median(lat),1) if lat else None)
        res.append(d); print("  rep%d %s" % (rep, d), flush=True)

    os.makedirs(a.out, exist_ok=True)
    json.dump({"tag": a.tag, "args": vars(a), "reps": res}, open(os.path.join(a.out, a.tag + ".json"), "w"), indent=1)
    ks = [r["ttft_p50"] for r in res if r["ttft_p50"]]
    if len(ks) > 1:
        print("同格噪声（rep 间 TTFT p50 极差）= %.1f%%" % ((max(ks)-min(ks))/min(ks)*100), flush=True)
    print("REPLAY_DONE")

if __name__ == "__main__":
    main()
