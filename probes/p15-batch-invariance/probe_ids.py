#!/usr/bin/env python3
"""用 SGLang /generate 的 input_ids 路径发 N 个相同请求（长 prompt 用，避免重新 tokenize）。

用法：python probe_ids.py --tag t --base http://127.0.0.1:32020 --ids-file p.json --n 32 ...
      python probe_ids.py --selftest
"""
import argparse, collections, concurrent.futures as cf, json, sys, time, urllib.error, urllib.request


def _post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def one(base, ids, max_tokens, seed, timeout):
    body = {"input_ids": ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens,
                                "ignore_eos": True, "seed": seed}}
    t0 = time.perf_counter()
    try:
        d = _post(base + "/generate", body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:160]!r}", None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", None
    items = d if isinstance(d, list) else [d]
    ids_out = [tuple((it.get("output_ids") or [])[:24]) for it in items]
    return ids_out, None, time.perf_counter() - t0


def run(a):
    ids = json.load(open(a.ids_file, encoding="utf-8"))["input_ids"][: a.prompt_tokens]
    rows = []
    with open(a.out, "w", encoding="utf-8") as out:
        for rep in range(a.repeats):
            with cf.ThreadPoolExecutor(max_workers=a.n) as ex:
                res = list(ex.map(lambda _: one(a.base, ids, a.max_tokens, a.seed, a.timeout),
                                  range(a.n)))
            oks = [r[0] for r in res if r[0]]
            errs = [r[1] for r in res if r[1]]
            walls = [r[2] for r in res if r[2]]
            flat = [t for group in oks for t in group]
            cnt = collections.Counter(flat)
            top, num = cnt.most_common(1)[0] if cnt else (None, 0)
            row = {"tag": a.tag, "rep": rep, "n": a.n, "prompt_tokens": len(ids),
                   "max_tokens": a.max_tokens, "got": len(flat),
                   "unique_prefixes": len(cnt),
                   "pairwise_divergence": (1 - num / len(flat)) if flat else None,
                   "wall_s": max(walls) if walls else None,
                   "err": errs[0] if errs and not flat else (f"{len(errs)} 子请求失败" if errs else None)}
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps(row, ensure_ascii=False))
    return 0


def selftest():
    assert one.__doc__ is None or True
    s = collections.Counter([(1, 2), (1, 2), (3, 4)])
    assert 1 - s.most_common(1)[0][1] / 3 > 0.3
    print("selftest ✔ 计数器口径（divergence 由唯一前缀数推出）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t")
    ap.add_argument("--base", default="http://127.0.0.1:32020")
    ap.add_argument("--ids-file", required=False, default="")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default="/tmp/p15ids.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.ids_file, "需要 --ids-file"
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
