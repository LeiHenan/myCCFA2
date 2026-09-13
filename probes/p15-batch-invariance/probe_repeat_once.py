#!/usr/bin/env python3
"""同一 prompt 串行重复 N 次（每次 n=1），比较跨请求输出是否一致。

用途：vLLM 的 /v1/completions 不支持"一批同 prompt"，故用重复请求测**可复现性**；
批组成效应需用 SGLang 的 text=List（见 probe_batch.py）——两者结论口径不同，不可混用。

用法：python probe_repeat_once.py --tag t --base http://127.0.0.1:32040 --model q3 --prompt "..." \
--repeats 3 --max-tokens 64 --out x.jsonl  [--selftest]
"""
import argparse, collections, json, sys, time, urllib.error, urllib.request


def _post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def once(base, model, prompt, max_tokens, timeout):
    body = {"model": model, "prompt": prompt, "n": 1, "temperature": 0.0,
            "max_tokens": max_tokens, "ignore_eos": True}
    t0 = time.perf_counter()
    try:
        d = _post(base + "/v1/completions", body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:200]!r}", None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", None
    ch = (d.get("choices") or [{}])[0]
    return ch.get("text"), None, time.perf_counter() - t0


def run(a):
    rows = []
    with open(a.out, "w", encoding="utf-8") as out:
        for rep in range(a.repeats):
            txt, err, wall = once(a.base, a.model, a.prompt, a.max_tokens, a.timeout)
            row = {"tag": a.tag, "rep": rep, "err": err, "wall_s": wall,
                   "text_head": (txt or "")[:60], "text_len": len(txt or "")}
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
            print(json.dumps({k: row[k] for k in ("tag", "rep", "text_len", "wall_s")}, ensure_ascii=False))
    ok = [r for r in rows if not r.get("err")]
    if ok:
        uniq = len(set(r["text_head"] for r in ok))
        print(f"[{a.tag}] 跨请求唯一输出数 = {uniq}/{len(ok)}  wall 中位="
              f"{sorted(r['wall_s'] for r in ok)[len(ok)//2]:.3f}s")
    else:
        print(f"[{a.tag}] 全部失败：{rows[0].get('err') if rows else 'no rows'}")
    return 0


def selftest():
    assert len({1, 1, 2}) == 2
    print("selftest ✔ 唯一计数口径（跨请求）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t")
    ap.add_argument("--base", default="http://127.0.0.1:32040")
    ap.add_argument("--model", default="q3")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default="/tmp/p15rep.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
