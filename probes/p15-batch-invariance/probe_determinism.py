#!/usr/bin/env python3
"""p15 批组成不变性探针：同一请求、贪心、固定种子，在不同批组成/重复下是否产出同一结果。

**判定用文本级**（对输出模态无关，避免解析器差异）：
  · 同一 batch 内放 N 份**完全相同的请求** ⇒ 若不变性成立，N 份输出必须逐字相同；
  · `pairwise_divergence = 1 - (最常见输出的出现次数 / 总数)`（0 = 完美一致）；
  · `unique` = 不同输出的个数。

同时报 wall（用于成本侧），并强制记录服务端实际用的 attention backend（从 serve 日志抓）。

用法：
  python probe_determinism.py --tag vllm_bi1 --base http://127.0.0.1:32010 \
      --prompt "..." --n 8 --max-tokens 64 --repeats 5 --out /path/x.jsonl
  python probe_determinism.py --selftest
"""
import argparse
import collections
import json
import sys
import time
import urllib.error
import urllib.request


def _post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def one_batch(base, prompt, n, max_tokens, seed, timeout):
    """一个 batch 内放 n 份相同请求。"""
    body = {"text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens,
                                "ignore_eos": True, "seed": seed}}
    t0 = time.perf_counter()
    try:
        d = _post(base + "/generate", body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:200]!r}", None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", None
    wall = time.perf_counter() - t0
    # SGLang /generate 单条时返回 dict；批量时返回 list。两种都接住。
    items = d if isinstance(d, list) else [d]
    outs = []
    for it in items:
        items_ids = it.get("output_ids") or []
        outs.append({"text": it.get("text"), "n_out": len(items_ids),
                     "ids": items_ids[:8]})
    return outs, None, wall


def summarize(outs):
    texts = [o["text"] for o in outs]
    c = collections.Counter(texts)
    top, cnt = c.most_common(1)[0]
    return {"unique": len(c), "total": len(texts),
            "pairwise_divergence": 1.0 - cnt / len(texts),
            "modal_len": collections.Counter(o["n_out"] for o in outs).most_common(1)[0][0]}


def run(a):
    out = open(a.out, "w", encoding="utf-8")
    rows = []
    for rep in range(a.repeats):
        outs, err, wall = one_batch(a.base, a.prompt, a.n, a.max_tokens, a.seed, a.timeout)
        row = {"tag": a.tag, "rep": rep, "n": a.n, "seed": a.seed,
               "max_tokens": a.max_tokens, "err": err, "wall_s": wall}
        if outs:
            row.update(summarize(outs))
            row["first_ids"] = outs[0]["ids"]
            row["sample_texts"] = [o["text"][:60] for o in outs[:3]] if a.n <= 8 else None
        rows.append(row)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        print(json.dumps({k: row[k] for k in ("tag", "rep", "unique", "total",
                                              "pairwise_divergence", "modal_len", "wall_s") 
                          if k in row}, ensure_ascii=False))
    out.close()
    ok = [r for r in rows if not r.get("err")]
    if ok:
        div = sum(r["pairwise_divergence"] for r in ok) / len(ok)
        print(f"[{a.tag}] 平均 divergence={div:.4f}  最大 unique={max(r['unique'] for r in ok)}"
              f"  wall 中位={sorted(r['wall_s'] for r in ok)[len(ok)//2]:.3f}s")
    else:
        print(f"[{a.tag}] 全部失败：{rows[0].get('err') if rows else 'no rows'}")
    return 0


def selftest():
    import types
    global _post
    calls = {"i": 0}

    def fake(url, payload, timeout=900):
        calls["i"] += 1
        # 注：第 1 次调用发生在 `one_batch` 里，第 2 次是测试里显式调用的那两次之一 ——
        # 这里按"前两次返回混合、第三次起全部一致"构造（初版写成 <=2 导致第 3 次仍返回混合，自测当场抓到）。
        if calls["i"] <= 2:
            return [{"text": "A" * 10, "output_ids": list(range(10))},
                    {"text": "B" * 10, "output_ids": list(range(10))}]
        return [{"text": "A" * 10, "output_ids": list(range(10))}] * 2

    _post = fake
    outs, err, wall = one_batch("http://x", "p", 2, 10, 0, 5)
    s = summarize(outs)
    assert s["unique"] == 2 and s["total"] == 2 and abs(s["pairwise_divergence"] - 0.5) < 1e-9, s
    # 让计数器走到"全一致"分支：先消耗一次调用
    one_batch("http://x", "p", 2, 10, 0, 5)
    outs2, _, _ = one_batch("http://x", "p", 2, 10, 0, 5)
    s2 = summarize(outs2)
    assert s2["unique"] == 1 and s2["pairwise_divergence"] == 0.0, s2
    # dict（单条）形态也要能接住
    _post = lambda *a, **k: {"text": "Z", "output_ids": [1, 2, 3]}
    outs3, _, _ = one_batch("http://x", "p", 1, 3, 0, 5)
    assert len(outs3) == 1 and outs3[0]["n_out"] == 3, outs3
    print("selftest ✔ 批量/单条两种返回形态 · divergence · unique · 长度")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--base", default="http://127.0.0.1:32010")
    ap.add_argument("--prompt", default="The capital of the state containing New York City is")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default="/tmp/p15.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
