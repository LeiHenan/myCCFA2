#!/usr/bin/env python3
"""p21 —— 前缀缓存命中的"仍要重算"量 + 缓存状态是否改变输出（KV 地图 B107 / B116 / B109）。

设计（同一个冷/热对照同时回答三个问题）：

  阶段 A（冷）：每个 prompt 各发一次 —— 这是它们第一次出现 ⇒ 前缀缓存必然 miss。
  阶段 B（热）：同样顺序再发一次 ⇒ 应命中完整前缀。
  阶段 C（热）：再发一次（与 B 同状态）⇒ B vs C 是"纯重复性"，A vs B 是"缓存状态效应"。

因此一次运行给出三件事：
  · **B107**：热阶段里 `vllm:request_prefill_kv_computed_tokens`（官方定义 *"new KV tokens computed
    during prefill (excluding cached tokens)"*）的**每请求均值** —— 完整命中应当 ≈1（只有最后一个 token），
    上游称 EAGLE/MTP 下每次命中仍要重算上千 token。**这就是可回收量**（白算的 prefill）。
  · **B116**：A 与 B 的**输出 token 文本是否相同**（*"cache reuse changes deterministic output"*）。
  · **B109**：B 与 C 的**输出是否相同**（纯重复性；T=0 下不同即为不确定性）。

指标口径：`/metrics` 是 Prometheus 文本；计数器和直方图都做**两次快照做差**（历史教训：绝对值和不可用）。
直方图用 Δsum/Δcount 得均值，并保留 Δsum/Δcount/buckets。

用法：
  probe_prefix_hit.py --base-url http://127.0.0.1:32080 --prompts /path/prompts_4096.jsonl \
      --n-prompts 8 --max-tokens 32 --out /root/ccfa_results/.../S_nospec
  probe_prefix_hit.py --selftest        # 不需要服务器
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

# ---------------------------------------------------------------- metrics 解析

# 我们需要做差的量。计数器以 _total 结尾；直方图暴露 _sum/_count/_bucket。
WANT_COUNTERS = [
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:generation_tokens_total",
]
WANT_HISTO = [
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:request_prefill_time_seconds",
]


def parse_prometheus(text: str) -> dict:
    """把 Prometheus 文本解析成 {name: value}，同名多标签**求和**。

    直方图：`name_bucket{le="X"}` → 键 `name_bucket`（累加），`name_sum` / `name_count` 原样。
    分位数/摘要行（带 `quantile=`）跳过。
    """
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # name{labels} value  或  name value
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
        # 归一化计数器名：Prometheus 客户端可能输出 _total，也可能不带
        out[name] = out.get(name, 0.0) + v
        if name.endswith("_total"):
            out.setdefault(name[: -len("_total")], 0.0)
            out[name[: -len("_total")]] += v
    return out


def fetch_metrics(base_url: str) -> dict:
    url = base_url.rstrip("/") + "/metrics"
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
        return parse_prometheus(r.read().decode("utf-8", "replace"))


def delta(after: dict, before: dict) -> dict:
    keys = set(after) | set(before)
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys}


def counter_delta(d: dict, name: str) -> float | None:
    """计数器做差：优先 _total，退回无后缀名。"""
    for k in (name, name[: -len("_total")] if name.endswith("_total") else name + "_total"):
        if k in d:
            return d[k]
    return None


def hist_delta(d: dict, name: str) -> dict:
    s, c = d.get(f"{name}_sum"), d.get(f"{name}_count")
    return {
        "sum": s,
        "count": c,
        "mean": (s / c) if (s is not None and c) else None,
    }


# ---------------------------------------------------------------- HTTP 客户端


def completion(base_url: str, model: str, prompt: str, max_tokens: int,
               timeout: float = 600.0, logprobs: int | None = None) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    if logprobs:
        body["logprobs"] = logprobs
    req = urllib.request.Request(  # noqa: S310
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        payload = json.loads(r.read().decode())
    wall = time.perf_counter() - t0
    ch = payload["choices"][0]
    lp = None
    if logprobs and ch.get("logprobs", {}).get("token_logprobs"):
        vals = [v for v in ch["logprobs"]["token_logprobs"] if v is not None]
        lp = vals
    return {
        "text": ch.get("text"),
        "finish_reason": ch.get("finish_reason"),
        "usage": payload.get("usage"),
        "wall_s": wall,
        "logprobs": lp,
    }


# ---------------------------------------------------------------- 主流程


def run_phase(base_url: str, model: str, prompts: list[str], max_tokens: int,
              tag: str, logprobs: int | None) -> list[dict]:
    rows = []
    for i, p in enumerate(prompts):
        try:
            out = completion(base_url, model, p, max_tokens, logprobs=logprobs)
        except Exception as exc:  # noqa: BLE001
            out = {"text": None, "error": f"{type(exc).__name__}: {exc}", "wall_s": None}
        out["prompt_idx"] = i
        rows.append(out)
        print(f"    [{tag}] prompt {i + 1}/{len(prompts)} wall={out.get('wall_s')} "
              f"chars={len(out.get('text') or '')}", flush=True)
    return rows


def same_outputs(a: list[dict], b: list[dict]) -> dict:
    """逐 prompt 比较两阶段的输出文本（与 token_logprobs）。"""
    n = min(len(a), len(b))
    text_eq = sum(1 for i in range(n) if a[i].get("text") == b[i].get("text"))
    lp_eq = lp_tot = 0
    for i in range(n):
        la, lb = a[i].get("logprobs"), b[i].get("logprobs")
        if la and lb:
            lp_tot += 1
            if all(abs(x - y) < 1e-9 for x, y in zip(la, lb)):
                lp_eq += 1
    return {"n": n, "text_equal": text_eq, "text_total": n, "logprob_equal": lp_eq, "logprob_total": lp_tot}


def verdict_block(phases: dict[str, list[dict]], deltas: dict[str, dict]) -> dict:
    cold, warm, warm2 = phases["A_cold"], phases["B_warm"], phases["C_warm2"]
    if not cold or not warm:
        return {"error": "阶段缺失"}
    prompt_len = (cold[0].get("usage") or {}).get("prompt_tokens")

    def per_req(d, name):
        h = hist_delta(d, name)
        return h

    kv_cold, kv_warm, kv_warm2 = (per_req(deltas[k], "vllm:request_prefill_kv_computed_tokens")
                                  for k in ("A_cold", "B_warm", "C_warm2"))
    pf_cold, pf_warm = (per_req(deltas[k], "vllm:request_prefill_time_seconds")
                        for k in ("A_cold", "B_warm"))

    def cd(k, name):
        return counter_delta(deltas[k], name)

    return {
        "prompt_len": prompt_len,
        "n_requests_per_phase": len(cold),
        "A_cold": {
            "prompt_tokens": cd("A_cold", "vllm:prompt_tokens_total"),
            "prompt_tokens_cached": cd("A_cold", "vllm:prompt_tokens_cached_total"),
            "prefix_cache_queries": cd("A_cold", "vllm:prefix_cache_queries_total"),
            "prefix_cache_hits": cd("A_cold", "vllm:prefix_cache_hits_total"),
            "prefill_kv_computed": kv_cold,
            "prefill_time_s": pf_cold,
        },
        "B_warm": {
            "prompt_tokens": cd("B_warm", "vllm:prompt_tokens_total"),
            "prompt_tokens_cached": cd("B_warm", "vllm:prompt_tokens_cached_total"),
            "prefix_cache_queries": cd("B_warm", "vllm:prefix_cache_queries_total"),
            "prefix_cache_hits": cd("B_warm", "vllm:prefix_cache_hits_total"),
            "prefill_kv_computed": kv_warm,
            "prefill_time_s": pf_warm,
        },
        "C_warm2": {
            "prefix_cache_hits": cd("C_warm2", "vllm:prefix_cache_hits_total"),
            "prefill_kv_computed": kv_warm2,
        },
        # ---- B107：命中后仍重算了多少
        "B107_recompute_per_hit_tokens": kv_warm["mean"],
        "B107_cold_compute_per_req_tokens": kv_cold["mean"],
        "B107_fraction_of_prompt_recomputed_on_hit": (
            kv_warm["mean"] / prompt_len if (kv_warm["mean"] is not None and prompt_len) else None
        ),
        # ---- B116 / B109
        "B116_cold_vs_warm_output": same_outputs(cold, warm),
        "B109_warm_vs_warm_output": same_outputs(warm, warm2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:32080")
    ap.add_argument("--model", default="q3")
    ap.add_argument("--prompts")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", help="输出前缀（生成 <out>.json 与 <out>.requests.json）")
    ap.add_argument("--logprobs", type=int, default=0, help=">0 时记录 token_logprobs（更敏感的不确定性探针）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if not a.prompts or not a.out:
        print("需要 --prompts 与 --out", file=sys.stderr)
        return 2

    prompts = []
    with open(a.prompts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
            if len(prompts) >= a.n_prompts:
                break
    print(f"装载 {len(prompts)} 个 prompt，每个约 {len(prompts[0])} 字符；max_tokens={a.max_tokens}")

    lp = a.logprobs or None
    snaps: dict[str, dict] = {}
    phases: dict[str, list[dict]] = {}
    snaps["start"] = fetch_metrics(a.base_url)
    for tag in ("A_cold", "B_warm", "C_warm2"):
        print(f"  == 阶段 {tag} ==", flush=True)
        phases[tag] = run_phase(a.base_url, a.model, prompts, a.max_tokens, tag, lp)
        snaps[tag] = fetch_metrics(a.base_url)
    deltas = {}
    order = ["start", "A_cold", "B_warm", "C_warm2"]
    for prev, cur in zip(order, order[1:]):
        deltas[cur] = delta(snaps[cur], snaps[prev])

    report = {
        "base_url": a.base_url,
        "model": a.model,
        "prompts_file": a.prompts,
        "n_prompts": len(prompts),
        "max_tokens": a.max_tokens,
        "logprobs": lp,
        "raw_deltas": deltas,
        "verdict": verdict_block(phases, deltas),
    }
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(a.out + ".requests.json", "w", encoding="utf-8") as f:
        json.dump(phases, f, indent=2, ensure_ascii=False)

    v = report["verdict"]
    print("\n================ 判定 ================")
    print(f"prompt 长度（token）        : {v['prompt_len']}")
    for ph in ("A_cold", "B_warm"):
        d = v[ph]
        kv = d["prefill_kv_computed"]
        print(f"{ph:8s} prompt_tokens={d['prompt_tokens']} cached={d['prompt_tokens_cached']} "
              f"hits={d['prefix_cache_hits']}/{d['prefix_cache_queries']} "
              f"prefill_kv_computed mean={kv['mean']}")
    print(f"【B107】冷阶段每请求重算 KV token : {v['B107_cold_compute_per_req_tokens']}")
    print(f"【B107】热阶段(命中)每请求重算     : {v['B107_recompute_per_hit_tokens']}")
    print(f"【B107】命中时重算占 prompt 比例   : {v['B107_fraction_of_prompt_recomputed_on_hit']}")
    print(f"【B116】冷 vs 热 输出相同数        : {v['B116_cold_vs_warm_output']}")
    print(f"【B109】热 vs 热 输出相同数        : {v['B109_warm_vs_warm_output']}")
    return 0


def selftest() -> int:
    text = """# HELP vllm:prompt_tokens Number of prefill tokens processed.
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{model_name="q3"} 100.0
vllm:request_prefill_kv_computed_tokens_bucket{model_name="q3",le="1.0"} 3.0
vllm:request_prefill_kv_computed_tokens_bucket{model_name="q3",le="+Inf"} 4.0
vllm:request_prefill_kv_computed_tokens_sum{model_name="q3"} 40.0
vllm:request_prefill_kv_computed_tokens_count{model_name="q3"} 4.0
vllm:request_prefill_time_seconds_sum{model_name="q3"} 2.0
vllm:request_prefill_time_seconds_count{model_name="q3"} 4.0
"""
    m = parse_prometheus(text)
    assert m["vllm:prompt_tokens_total"] == 100.0, m.get("vllm:prompt_tokens_total")
    assert m["vllm:request_prefill_kv_computed_tokens_sum"] == 40.0
    assert m["vllm:request_prefill_kv_computed_tokens_count"] == 4.0
    d = delta({"vllm:prompt_tokens_total": 250.0,
               "vllm:request_prefill_kv_computed_tokens_sum": 50.0,
               "vllm:request_prefill_kv_computed_tokens_count": 5.0},
              {"vllm:prompt_tokens_total": 100.0,
               "vllm:request_prefill_kv_computed_tokens_sum": 40.0,
               "vllm:request_prefill_kv_computed_tokens_count": 4.0})
    assert counter_delta(d, "vllm:prompt_tokens_total") == 150.0
    h = hist_delta(d, "vllm:request_prefill_kv_computed_tokens")
    assert h["sum"] == 10.0 and h["count"] == 1.0 and h["mean"] == 10.0, h
    # 冷热对照：A=[x,x], B=[x,y] ⇒ text_equal=1；B vs C 全等 ⇒ 2
    ph = {"A_cold": [{"text": "x", "usage": {"prompt_tokens": 100}}] * 2,
          "B_warm": [{"text": "x"}, {"text": "y"}],
          "C_warm2": [{"text": "x"}, {"text": "y"}]}
    dl = {k: {"vllm:prompt_tokens_total": 0.0} for k in ("A_cold", "B_warm", "C_warm2")}
    dl["A_cold"] = {"vllm:prompt_tokens_total": 200.0,
                    "vllm:request_prefill_kv_computed_tokens_sum": 200.0,
                    "vllm:request_prefill_kv_computed_tokens_count": 2.0}
    dl["B_warm"] = {"vllm:prompt_tokens_total": 2.0,
                    "vllm:request_prefill_kv_computed_tokens_sum": 2.0,
                    "vllm:request_prefill_kv_computed_tokens_count": 2.0}
    v = verdict_block(ph, dl)
    assert v["B107_recompute_per_hit_tokens"] == 1.0, v["B107_recompute_per_hit_tokens"]
    assert v["B107_fraction_of_prompt_recomputed_on_hit"] == 0.01
    assert v["B116_cold_vs_warm_output"]["text_equal"] == 1
    assert v["B109_warm_vs_warm_output"]["text_equal"] == 2
    print("selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
