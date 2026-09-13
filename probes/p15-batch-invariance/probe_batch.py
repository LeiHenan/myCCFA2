#!/usr/bin/env python3
"""按构造合批的探针：一次 /generate 调用里放 n 个**相同** prompt（SGLang 的 text 支持 List）。

为什么这样设计（两轮事故的结论）：
  ① 用 `ThreadPoolExecutor` 发并发 HTTP **无法证明**服务端把它们放进同一批（上一轮因此把结论降级为未取证）；
  ② vLLM 贪心下禁用 `n>1`；SGLang 的 `sampling_params` **不接受 `seed`**。
  ⇒ SGLang 的 `GenerateReqInput.text` 支持 `List[str]`（`io_struct.py:182`："It can be a single prompt
     or a batch of prompts"）⇒ **一次调用 = 一个批**，合批**按构造成立**，且不需要 seed。

判定：同一调用内 n 份相同 prompt 的输出是否逐字相同；再跨不同 n（批规模）比较。
另可取 token 级指纹（output_ids 前缀）作为不依赖文本的对照。

用法：
  python probe_batch.py --tag sgl_n8 --base http://127.0.0.1:32020 --prompt-file /root/autodl-tmp/prompts/prompts_4096.jsonl \
      --n 8 --max-tokens 64 --repeats 2 --out x.jsonl
  python probe_batch.py --selftest
"""
import argparse
import collections
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request


def _post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load_prompt(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                o = json.loads(line)
                return o["prompt"] if isinstance(o, dict) else o
    raise ValueError("空文件")


def call_batch(base, prompt, n, max_tokens, timeout, temperature=0.0):
    """一个请求 = 一个批（n 份相同 prompt）。不传 seed（SGLang 不接受）。"""
    # ⚠️ 关键（2026-09-13 实测教训）：SGLang 的 `SamplingParams.normalize()` 把
    #    `0 <= temperature < _SAMPLING_EPS` **改写成 1.0**（`sampling_params.py:150-152`），
    #    即 **只设 temperature=0 得到的是标准采样，不是贪心**；真正的贪心是 `top_k=1`（:151 注释）。
    #    初版漏了 top_k ⇒ 两臂都在采样 ⇒ 把随机性误读成"批组成效应"。
    body = {"text": [prompt] * n,
            "sampling_params": {"temperature": temperature, "top_k": 1,
                                "max_new_tokens": max_tokens, "ignore_eos": True}}
    t0 = time.perf_counter()
    try:
        d = _post(base + "/generate", body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:200]!r}", None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", None
    wall = time.perf_counter() - t0
    items = d if isinstance(d, list) else [d]
    outs = []
    for it in items:
        ids = it.get("output_ids") or []
        outs.append({"text": it.get("text"),
                     "n_out": len(ids),
                     "fp": hashlib.sha1(json.dumps(ids[:32]).encode()).hexdigest()[:12]})
    return outs, None, wall


def summarize(outs):
    texts = [o["text"] for o in outs]
    fps = [o["fp"] for o in outs]
    ct, cf = collections.Counter(texts), collections.Counter(fps)
    top_t, n_t = ct.most_common(1)[0]
    top_f, n_f = cf.most_common(1)[0]
    # ⚠️ 指标教训：`all_fp_same` 用"众数出现次数 == 总数"来判，**当每个输出都唯一时它恒为 True**，
    #    会把"全不同"误报成"全相同"（初版即犯此错）。唯一性必须用 `unique_fp`/`unique_text` 看。
    all_fp_same = (len(fps) == n_f)
    return {"total": len(outs), "unique_text": len(ct), "unique_fp": len(cf),
            "divergence_text": 1 - n_t / len(texts),
            "divergence_fp": 1 - n_f / len(fps),
            "modal_fp": top_f,
            "all_fp_same": all_fp_same, "modal_len": collections.Counter(o["n_out"] for o in outs).most_common(1)[0][0],
            "sample_text": (top_t or "")[:60]}


def run(a):
    prompt = load_prompt(a.prompt_file)
    rows = []
    with open(a.out, "w", encoding="utf-8") as out:
        for rep in range(a.repeats):
            outs, err, wall = call_batch(a.base, prompt, a.n, a.max_tokens, a.timeout, a.temperature)
            row = {"tag": a.tag, "rep": rep, "n": a.n, "max_tokens": a.max_tokens,
                   "err": err, "wall_s": wall}
            if outs:
                row.update(summarize(outs))
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({k: row.get(k) for k in ("tag", "rep", "n", "total", "unique_text",
                                                      "unique_fp", "divergence_text", "divergence_fp",
                                                      "modal_len", "wall_s")}, ensure_ascii=False))
    ok = [r for r in rows if not r.get("err")]
    if ok:
        print(f"[{a.tag}] 批内 divergence(text)={sum(r['divergence_text'] for r in ok)/len(ok):.4f}"
              f"  (fp)={sum(r['divergence_fp'] for r in ok)/len(ok):.4f}"
              f"  wall 中位={sorted(r['wall_s'] for r in ok)[len(ok)//2]:.3f}s")
    else:
        print(f"[{a.tag}] 全部失败：{rows[0].get('err') if rows else 'no rows'}")
    return 0


def selftest():
    # 响应解析 + 指纹 + 两种返回形态（list / dict）
    outs = []
    for txt, ids in (("A", [1, 2, 3]), ("A", [1, 2, 3]), ("B", [9, 9])):
        outs.append({"text": txt, "n_out": len(ids), "fp": hashlib.sha1(
            json.dumps(ids[:32]).encode()).hexdigest()[:12]})
    s = summarize(outs)
    assert s["total"] == 3 and s["unique_text"] == 2 and s["unique_fp"] == 2, s
    assert abs(s["divergence_text"] - 1 / 3) < 1e-9, s
    # 全部一致
    outs2 = [{"text": "A", "n_out": 3, "fp": "x"} for _ in range(4)]
    s2 = summarize(outs2)
    assert s2["unique_text"] == 1 and s2["divergence_text"] == 0.0 and s2["all_fp_same"], s2
    # **全不同**时 all_fp_same 必须为 False（初版此处误报 True，是本次抓到的指标 bug）
    outs3 = [{"text": f"A{i}", "n_out": 3, "fp": f"x{i}"} for i in range(4)]
    s3 = summarize(outs3)
    assert s3["unique_fp"] == 4 and not s3["all_fp_same"], s3
    print("selftest ✔ 指纹/唯一计数/divergence/两种一致情形")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t")
    ap.add_argument("--base", default="http://127.0.0.1:32020")
    ap.add_argument("--prompt-file", default="/root/autodl-tmp/prompts/prompts_4096.jsonl")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default="/tmp/p15batch.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
