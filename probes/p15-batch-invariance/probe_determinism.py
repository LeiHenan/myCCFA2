#!/usr/bin/env python3
"""p15 批组成不变性探针：同一请求、贪心、固定种子，在不同批组成/重复下是否产出同一结果。

**判定用文本级**（对输出模态无关，避免解析器差异）：
  · 同一 batch 内放 n 份**完全相同**的请求 ⇒ 若不变性成立，n 份输出必须逐字相同；
  · `pairwise_divergence = 1 - (最常见输出出现次数 / 总数)`（0 = 完美一致）；
  · `unique` = 不同输出的个数。

**两种 HTTP 协议**（必须按引擎选，探针不写死单一路由）：
  --protocol openai    → vLLM / SGLang 的 `/v1/completions`（prompt + n + temperature + seed）
  --protocol generate  → SGLang 原生 `/generate`（text + sampling_params）
  ⚠️ 事故记录：初版写死 `/generate`，而 vLLM 的 OpenAI server **没有**该路由 ⇒ 12 格 × 5 次 = 60 次请求全 404。

用法：
  python probe_determinism.py --tag vllm_bi1_n8 --protocol openai --model q3 \
      --base http://127.0.0.1:32010 --prompt "..." --n 8 --max-tokens 64 --repeats 5 --out x.jsonl
  python probe_determinism.py --selftest
"""
import argparse
import collections
import json
import sys
import time
import urllib.error
import urllib.request

MODEL = ["q3"]  # openai 协议用；由 CLI 覆盖


def _post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_request(base, prompt, n, max_tokens, seed, protocol):
    if protocol == "openai":
        return base + "/v1/completions", {
            "model": MODEL[0], "prompt": prompt, "n": n,
            "temperature": 0.0, "max_tokens": max_tokens, "ignore_eos": True, "seed": seed}
    return base + "/generate", {
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens,
                            "ignore_eos": True, "seed": seed}}


def parse_response(d, protocol):
    """→ [{text, n_out, ids}]，两种协议统一形态。"""
    if protocol == "openai":
        if isinstance(d, list):          # 某些实现批量时回 list
            choices = [c for item in d for c in (item.get("choices") or [])]
        else:
            choices = d.get("choices") or []
        out = []
        for c in choices:
            ids = c.get("token_ids") or []
            out.append({"text": c.get("text"), "n_out": len(ids) if ids else None,
                        "ids": ids[:8]})
        return out
    items = d if isinstance(d, list) else [d]
    out = []
    for it in items:
        ids = it.get("output_ids") or []
        out.append({"text": it.get("text"), "n_out": len(ids), "ids": ids[:8]})
    return out


def one_batch(base, prompt, n, max_tokens, seed, timeout, protocol="generate"):
    url, body = build_request(base, prompt, n, max_tokens, seed, protocol)
    t0 = time.perf_counter()
    try:
        d = _post(url, body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:200]!r}", None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", None
    wall = time.perf_counter() - t0
    return parse_response(d, protocol), None, wall


def one_batch_concurrent(base, prompt, n, max_tokens, seed, timeout, protocol):
    """并发提交 n 个**相同**请求（单请求 n=1），让调度器把它们合进同一批。

    为什么不用 `n`: vLLM 在贪心采样下拒绝 `n>1`（`n must be 1 when using greedy sampling`）。
    """
    import concurrent.futures as cf
    url, _ = build_request(base, prompt, 1, max_tokens, seed, protocol)
    def _one(_):
        body = dict(_)
        if protocol == "openai":
            body["n"] = 1
        t0 = time.perf_counter()
        try:
            d = _post(url, body, timeout=timeout)
        except urllib.error.HTTPError as e:
            return None, f"HTTP {e.code}: {e.read()[:200]!r}", None
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}", None
        return parse_response(d, protocol), None, time.perf_counter() - t0
    _, body = build_request(base, prompt, 1, max_tokens, seed, protocol)
    outs, errs, walls = [], [], []
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        for o, e, w in ex.map(_one, [body] * n):
            if e:
                errs.append(e)
            else:
                outs.extend(o)
            if w:
                walls.append(w)
    if errs and not outs:
        return None, errs[0], None
    return outs, (f"{len(errs)} 个子请求失败" if errs else None), (max(walls) if walls else None)


def summarize(outs):
    texts = [o["text"] for o in outs]
    c = collections.Counter(texts)
    top, cnt = c.most_common(1)[0]
    lens = [o["n_out"] for o in outs if o["n_out"]]
    return {"unique": len(c), "total": len(texts),
            "pairwise_divergence": 1.0 - cnt / len(texts),
            "modal_len": collections.Counter(lens).most_common(1)[0][0] if lens else None}


def run(a):
    rows = []
    with open(a.out, "w", encoding="utf-8") as out:
        for rep in range(a.repeats):
            if a.mode == "concurrent":
                outs, err, wall = one_batch_concurrent(a.base, a.prompt, a.n, a.max_tokens,
                                                       a.seed, a.timeout, a.protocol)
            else:
                outs, err, wall = one_batch(a.base, a.prompt, a.n, a.max_tokens, a.seed,
                                            a.timeout, a.protocol)
            row = {"tag": a.tag, "rep": rep, "n": a.n, "seed": a.seed,
                   "max_tokens": a.max_tokens, "protocol": a.protocol, "mode": a.mode,
                   "err": err, "wall_s": wall}
            if outs:
                row.update(summarize(outs))
                row["first_ids"] = outs[0]["ids"]
                row["n_choices"] = len(outs)
                row["sample_texts"] = [o["text"][:50] for o in outs[:4]]
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({k: row.get(k) for k in ("tag", "rep", "n_choices", "unique", "total",
                                                      "pairwise_divergence", "modal_len", "wall_s")},
                             ensure_ascii=False))
    ok = [r for r in rows if not r.get("err")]
    if ok:
        div = sum(r["pairwise_divergence"] for r in ok) / len(ok)
        ws = sorted(r["wall_s"] for r in ok)
        print(f"[{a.tag}] 平均 divergence={div:.4f}  最大 unique={max(r['unique'] for r in ok)}"
              f"  wall 中位={ws[len(ws)//2]:.3f}s")
    else:
        print(f"[{a.tag}] 全部失败：{rows[0].get('err') if rows else 'no rows'}")
    return 0


def selftest():
    # 纯函数级自测：两种协议的请求构造与响应解析
    u, b = build_request("http://x", "p", 3, 10, 7, "openai")
    assert u.endswith("/v1/completions") and b["n"] == 3 and b["seed"] == 7 and b["temperature"] == 0.0, (u, b)
    u2, b2 = build_request("http://x", "p", 3, 10, 7, "generate")
    assert u2.endswith("/generate") and b2["sampling_params"]["max_new_tokens"] == 10, (u2, b2)
    # openai 批量响应
    s = summarize(parse_response({"choices": [{"text": "A", "token_ids": [1, 2]},
                                              {"text": "B", "token_ids": [1, 2]}]}, "openai"))
    assert s["unique"] == 2 and abs(s["pairwise_divergence"] - 0.5) < 1e-9 and s["modal_len"] == 2, s
    # generate 单条 dict
    s2 = summarize(parse_response({"text": "Z", "output_ids": [1, 2, 3]}, "generate"))
    assert s2["unique"] == 1 and s2["modal_len"] == 3, s2
    # 全一致
    s3 = summarize(parse_response({"choices": [{"text": "A", "token_ids": [1]}] * 4}, "openai"))
    assert s3["unique"] == 1 and s3["pairwise_divergence"] == 0.0, s3
    print("selftest ✔ 两种协议的请求构造/响应解析 · divergence · unique · 长度")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--base", default="http://127.0.0.1:32010")
    ap.add_argument("--protocol", choices=["generate", "openai"], default="generate")
    ap.add_argument("--mode", choices=["concurrent", "n"], default="concurrent",
                    help="concurrent=并发 N 个相同请求（推荐，vLLM 贪心下禁用 n>1）")
    ap.add_argument("--model", default="q3")
    ap.add_argument("--prompt", default="The capital of the state containing New York City is")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default="/tmp/p15.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    MODEL[0] = a.model
    if a.selftest:
        return selftest()
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
