#!/usr/bin/env python3
"""把 p15 探针改成支持两种 HTTP 协议（幂等）：vLLM=completions，SGLang=generate。

事故记录：初版探针写死 SGLang 的 `/generate`，而 vLLM 的 OpenAI server 没有该路由 ⇒ 两臂全部 HTTP 404
（12 个格 × 5 次 = 60 次请求全废）。教训：**换引擎必须换协议**，探针不该写死单一路由。
"""
import io, os, sys

P = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "probes", "p15-batch-invariance", "probe_determinism.py")

OLD_SIG = 'def one_batch(base, prompt, n, max_tokens, seed, timeout):'
NEW = '''def one_batch(base, prompt, n, max_tokens, seed, timeout, protocol="generate"):
    """一个 batch 内放 n 份相同请求。

    protocol="generate"  → SGLang 原生 `/generate`（text + sampling_params）
    protocol="openai"    → vLLM/SGLang 的 `/v1/completions`（prompt + 标准字段）
    （初版写死 generate，导致 vLLM 侧 60 次请求全 404 —— 见本文件头的事故记录。）
    """
    if protocol == "openai":
        url = base + "/v1/completions"
        body = {"model": _MODEL[0], "prompt": prompt, "n": n,
                "temperature": 0.0, "max_tokens": max_tokens,
                "ignore_eos": True, "seed": seed}
    else:
        url = base + "/generate"
        body = {"text": prompt,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens,
                                    "ignore_eos": True, "seed": seed}}
    t0 = time.perf_counter()
    try:
        d = _post(url, body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:200]!r}", None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", None
    wall = time.perf_counter() - t0
    if protocol == "openai":
        ch = (d.get("choices") or [])
        outs = [{"text": c.get("text"), "n_out": len(c.get("token_ids") or []) or None,
                 "ids": (c.get("token_ids") or [])[:8]} for c in ch]
        return outs, None, wall
    items = d if isinstance(d, list) else [d]
    outs = []
    for it in items:
        items_ids = it.get("output_ids") or []
        outs.append({"text": it.get("text"), "n_out": len(items_ids), "ids": items_ids[:8]})
    return outs, None, wall


_MODEL = ["q3"]


def _old_one_batch_removed(base, prompt, n, max_tokens, seed, timeout):'''

OLD_BODY_MARK = '    body = {"text": prompt,\n            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens,\n                                "ignore_eos": True, "seed": seed}}\n    t0 = time.perf_counter()'


def main():
    s = io.open(P, encoding="utf-8").read()
    if "protocol=\"openai\"" in s:
        print("已是最新（幂等）")
        return 0
    assert OLD_SIG in s, "锚点未命中"
    head, rest = s.split(OLD_SIG, 1)
    # 去掉旧实现（到 summarize 之前）
    tail = rest.split("def summarize(", 1)[1]
    s2 = head + NEW + "\n\n\ndef summarize(" + tail
    # run() 里传 protocol；--protocol 参数；_MODEL 由 CLI 设置
    s2 = s2.replace('outs, err, wall = one_batch(a.base, a.prompt, a.n, a.max_tokens, a.seed, a.timeout)',
                    'outs, err, wall = one_batch(a.base, a.prompt, a.n, a.max_tokens, a.seed,\n'
                    '                                    a.timeout, a.protocol)')
    s2 = s2.replace('    ap.add_argument("--out", default="/tmp/p15.jsonl")',
                    '    ap.add_argument("--protocol", choices=["generate", "openai"], default="generate")\n'
                    '    ap.add_argument("--model", default="q3", help="openai 协议下的 model 名")\n'
                    '    ap.add_argument("--out", default="/tmp/p15.jsonl")')
    s2 = s2.replace("    a = ap.parse_args()\n    if a.selftest:",
                    "    a = ap.parse_args()\n    _MODEL[0] = a.model\n    if a.selftest:")
    io.open(P, "w", encoding="utf-8").write(s2)
    print("探针已改为协议可切换")
    return 0


if __name__ == "__main__":
    sys.exit(main())
