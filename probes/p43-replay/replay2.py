#!/usr/bin/env python3
"""结构忠实的真实轨迹回放（修正 replay.py 的致命缺陷）。

**缺陷**：replay.py 用固定文本造 prompt ⇒ 所有请求前缀相同 ⇒ 完全没复现轨迹的前缀共享结构，
而结构正是本实验的全部意义（于是 TTFT 就等于"完全未命中"的 prefill，毫无信息）。

**修法**：按 `hash_ids` 逐块构造 token 序列——每个块 id 映射到一个确定的 16-token 模式，
prompt = 该请求各块模式顺序拼接。于是**轨迹里共享的块在 prompt 里也共享**，
前缀缓存看到的结构与真实一致。prompt 以 token id 列表直接发送（OpenAI 兼容接口支持）。
"""
import argparse, json, os, statistics as st, threading, time
from concurrent.futures import ThreadPoolExecutor

VOCAB_SAFE = 150000

def blk_tokens(b, bs=16):
    """把块 id 确定性地展开成 bs 个 token id（避开 special token 区间）。"""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--scale", type=float, default=7.0)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="structured")
    ap.add_argument("--out", default="/root/ccfa_results/p43")
    a = ap.parse_args()
    import requests

    assert requests.get(a.base + "/v1/models", timeout=10).status_code == 200
    rows = load_trace(a.trace, a.n)
    t0 = float(rows[0].get("timestamp", 0))

    # 预先构造每条请求的 token 序列（结构忠实）
    prompts, nblk = [], []
    for r in rows:
        h = r.get("hash_ids") or []
        ids = []
        for b in h: ids.extend(blk_tokens(int(b)))
        prompts.append(ids); nblk.append(len(h))
    tot = sum(len(p) for p in prompts)
    print("结构忠实 prompt: %d 条 | 平均 %d token (块 %d) | 合计 %d token"
          % (len(prompts), tot/len(prompts), sum(nblk)/len(nblk), tot), flush=True)
    print("共享块数（唯一 id）: %d / %d" % (len(set(x for r in rows for x in (r.get('hash_ids') or []))), sum(nblk)), flush=True)

    sem = threading.Semaphore(a.conc)
    lat, ttft = [], []
    def one(ids, O):
        with sem:
            t = time.perf_counter()
            try:
                rr = requests.post(a.base + "/v1/completions", json={
                    "model": a.model, "prompt": ids, "max_tokens": max(1, O),
                    "temperature": 0.0, "ignore_eos": True, "stream": True}, stream=True, timeout=900)
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
            for i, r_ in enumerate(rows):
                due = wall0 + (float(r_.get("timestamp", 0)) - t0) * a.scale
                d = due - time.perf_counter()
                if d > 0: time.sleep(min(d, 5.0))
                futs.append(ex.submit(one, prompts[i], int(r_["output_length"])))
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
    print("REPLAY2_DONE")

if __name__ == "__main__":
    main()
