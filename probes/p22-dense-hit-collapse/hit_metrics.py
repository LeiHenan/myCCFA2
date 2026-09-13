#!/usr/bin/env python3
"""p22 采集器 —— 从 /metrics 抓**前缀缓存命中**相关量并做差（p21 的 collect_spec_metrics 只抓投机三键，不够用）。

抓的量（都做**两次快照做差**，绝对值不可用）：
  · `vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total`  ⇒ **命中率**
  · `vllm:prompt_tokens_total` / `vllm:prompt_tokens_cached_total`
  · `vllm:request_prefill_kv_computed_tokens_{sum,count}`（官方定义 *"excluding cached tokens"*）⇒ **每请求实际重算的 KV token**
  · `vllm:generation_tokens_total`（确认这批请求真的跑了）

用法：
  hit_metrics.py --base http://127.0.0.1:32100 fetch --out s0.json
  hit_metrics.py --before s0.json --after s1.json --out d.json diff
  hit_metrics.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

WANT = [
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:request_prefill_kv_computed_tokens_count",
]


def parse_prometheus(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            labels, _, value = rest.partition("}")
            value = value.strip()
            if "quantile=" in labels:
                continue
        else:
            parts = line.split()
            if len(parts) != 2:
                continue
            name, value = parts[0], parts[1]
        try:
            v = float(value.split()[0])
        except (ValueError, IndexError):
            continue
        name = name.strip()
        out[name] = out.get(name, 0.0) + v
        if name.endswith("_total"):
            k = name[: -len("_total")]
            out[k] = out.get(k, 0.0) + v
    return out


def fetch(base: str) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as r:  # noqa: S310
        raw = parse_prometheus(r.read().decode("utf-8", "replace"))
    snap = {k: raw.get(k) for k in WANT}
    snap["_found"] = sorted(k for k in WANT if snap.get(k) is not None)
    return snap


def diff(before: dict, after: dict) -> dict:
    d = {}
    for k in WANT:
        b, a = before.get(k), after.get(k)
        d[k] = None if (b is None or a is None) else a - b
    q, h = d.get("vllm:prefix_cache_queries_total"), d.get("vllm:prefix_cache_hits_total")
    d["hit_rate"] = (h / q) if (q and h is not None) else None
    s, c = (d.get("vllm:request_prefill_kv_computed_tokens_sum"),
            d.get("vllm:request_prefill_kv_computed_tokens_count"))
    d["prefill_kv_computed_per_request"] = (s / c) if (s is not None and c) else None
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:32100")
    ap.add_argument("cmd", nargs="?", choices=["fetch", "diff"])
    ap.add_argument("--out")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd not in ("fetch", "diff") or not a.out:
        print("需要 cmd(fetch|diff) 与 --out", file=sys.stderr)
        return 2
    if a.cmd == "fetch":
        snap = fetch(a.base)
        json.dump(snap, open(a.out, "w"), indent=1)
        print(f"  fetched: found {len(snap['_found'])}/{len(WANT)} keys -> {a.out}")
        if len(snap["_found"]) < 4:
            print(f"  ⚠️ 只找到 {snap['_found']} —— 前缀缓存指标名可能不对，**结果不可用**")
            return 1
        return 0
    d = diff(json.load(open(a.before)), json.load(open(a.after)))
    json.dump(d, open(a.out, "w"), indent=1)
    hr = d.get("hit_rate")
    print(f"  queries={d.get('vllm:prefix_cache_queries_total')} hits={d.get('vllm:prefix_cache_hits_total')} "
          f"hit_rate={'%.4f' % hr if hr is not None else 'NA'} "
          f"prefill_kv_per_req={d.get('prefill_kv_computed_per_request')}")
    return 0


def selftest() -> int:
    txt = """# HELP x
vllm:prefix_cache_queries_total{model_name="q3"} 100.0
vllm:prefix_cache_hits_total{model_name="q3"} 75.0
vllm:prompt_tokens_total{model_name="q3"} 1000.0
vllm:request_prefill_kv_computed_tokens_sum{model_name="q3"} 250.0
vllm:request_prefill_kv_computed_tokens_count{model_name="q3"} 5.0
vllm:request_prefill_kv_computed_tokens_bucket{model_name="q3",le="1.0"} 4.0
"""
    m = parse_prometheus(txt)
    assert m["vllm:prefix_cache_hits_total"] == 75.0, m
    snap = {k: m.get(k) for k in WANT}
    after = dict(snap)
    after["vllm:prefix_cache_queries_total"] = 200.0
    after["vllm:prefix_cache_hits_total"] = 175.0
    after["vllm:request_prefill_kv_computed_tokens_sum"] = 300.0
    after["vllm:request_prefill_kv_computed_tokens_count"] = 15.0
    d = diff(snap, after)
    assert d["vllm:prefix_cache_queries_total"] == 100.0, d
    assert d["vllm:prefix_cache_hits_total"] == 100.0, d
    assert abs(d["hit_rate"] - 1.0) < 1e-12, d["hit_rate"]
    assert abs(d["prefill_kv_computed_per_request"] - 5.0) < 1e-12, d
    # 全 miss ⇒ hit_rate 0；零查询 ⇒ None（不能除零）
    z = diff({"vllm:prefix_cache_queries_total": 0.0, "vllm:prefix_cache_hits_total": 0.0},
             {"vllm:prefix_cache_queries_total": 0.0, "vllm:prefix_cache_hits_total": 0.0})
    assert z["hit_rate"] is None, z
    print("selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
