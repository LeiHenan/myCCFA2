#!/usr/bin/env python3
"""把 p15 探针从「单请求 n=N」改成「并发提交 N 个相同请求」（幂等）。

两处事故（都是引擎 API 约束，不是我的设计错）：
  ① vLLM 无 `/generate` 路由 → 404（已由 --protocol 解决）
  ② vLLM 拒绝贪心下的 n>1：`n must be 1 when using greedy sampling, got 8`（本次）
⇒ 正确做法：**并发**发 N 个相同请求，让调度器自然把它们合进同一批；这比 `n` 更贴近"批组成"这一自变量。
"""
import io, os, sys

P = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "probes", "p15-batch-invariance", "probe_determinism.py")

FUNC = '''

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
'''


def main():
    s = io.open(P, encoding="utf-8").read()
    if "one_batch_concurrent" in s:
        print("已是最新（幂等）")
        return 0
    anchor = "\n\ndef summarize(outs):"
    assert anchor in s, "锚点未命中"
    s = s.replace(anchor, FUNC + anchor, 1)
    # run() 改调并发版；并加 --mode
    s = s.replace(
        'outs, err, wall = one_batch(a.base, a.prompt, a.n, a.max_tokens, a.seed,\n'
        '                                        a.timeout, a.protocol)',
        'if a.mode == "concurrent":\n'
        '                outs, err, wall = one_batch_concurrent(a.base, a.prompt, a.n, a.max_tokens,\n'
        '                                                       a.seed, a.timeout, a.protocol)\n'
        '            else:\n'
        '                outs, err, wall = one_batch(a.base, a.prompt, a.n, a.max_tokens, a.seed,\n'
        '                                            a.timeout, a.protocol)')
    s = s.replace('    ap.add_argument("--protocol", choices=["generate", "openai"], default="generate")',
                  '    ap.add_argument("--protocol", choices=["generate", "openai"], default="generate")\n'
                  '    ap.add_argument("--mode", choices=["concurrent", "n"], default="concurrent",\n'
                  '                    help="concurrent=并发 N 个相同请求（推荐，vLLM 贪心下禁用 n>1）")')
    s = s.replace('            row = {"tag": a.tag, "rep": rep, "n": a.n, "seed": a.seed,\n'
                  '                   "max_tokens": a.max_tokens, "protocol": a.protocol,',
                  '            row = {"tag": a.tag, "rep": rep, "n": a.n, "seed": a.seed,\n'
                  '                   "max_tokens": a.max_tokens, "protocol": a.protocol, "mode": a.mode,')
    io.open(P, "w", encoding="utf-8").write(s)
    print("探针已加并发模式")
    return 0


if __name__ == "__main__":
    sys.exit(main())
