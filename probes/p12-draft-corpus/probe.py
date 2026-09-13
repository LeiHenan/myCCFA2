#!/usr/bin/env python3
"""p12 冷启动/累积罚金探针 —— 逐请求串行打 SGLang /generate，同时采三条独立路径。

为什么不用 `python -m sglang.benchmark.serving`：① 它把统计聚合成整轮口径，看不到「第几次请求」；
② 我们需要把 **rep 边界与 trace/计数器对齐**（decision #93 的教训）；
③ 本探针要同时采三条路径以做交叉验证（PIPELINE.md §3.2）。

三条采集路径（口径已对着 SGLang 0.5.19 源码核过，见 decision #100）：
  ① 每请求响应体 `meta_info.spec_accept_length` = `completion_tokens / spec_verify_ct`（含 bonus token）
  ② 每请求响应体 `meta_info.spec_accept_rate`   = `num_correct_drafts / (verify_ct × (γ−1))`
  ③ 服务端累计计数器 `/get_internal_state.avg_spec_accept_length`
     = `spec_total_num_accept_tokens / spec_total_num_forward_ct`（**做差**后才是本 rep 的值）
  外加客户端自测的 wall time / TTFT，作为独立的第四条。

用法：
  python probe.py --tag cold_r1 --base http://127.0.0.1:31001 \
      --prompts /root/autodl-tmp/prompts/prompts_4096.jsonl --n 32 --out-len 128 \
      --reps 1 --out /root/ccfa_results/2026-09-13/p12_corpus/cold_r1.jsonl
  python probe.py --selftest
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def _post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def internal_state(base):
    """累计计数器；缺失时返回 None（不伪造 0 —— 0 会被误读成『没有投机』）。"""
    try:
        d = _get(base + "/get_internal_state")
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}
    st = d.get("internal_state", d) if isinstance(d, dict) else {}
    return st


def one_request(base, prompt, out_len, temperature, timeout):
    """串行单请求；返回 (记录, 错误)。"""
    body = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": out_len,
            "ignore_eos": True,  # 固定输出长度 ⇒ 接受长度可比（否则早期 EOS 会污染）
        },
    }
    t0 = time.perf_counter()
    try:
        d = _post(base + "/generate", body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    meta = d.get("meta_info", {}) or {}
    rec = {
        "wall_s": wall,
        # 路径 ① / ②
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
        "spec_num_correct_drafts": meta.get("spec_num_correct_drafts"),
        "spec_num_proposed_drafts": meta.get("spec_num_proposed_drafts"),
        # 通用
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "e2e_latency": meta.get("e2e_latency"),
        "ttft": meta.get("ttft"),
        # 缓存（用于把「冷启动」与「前缀缓存」区分开）
        "cached_tokens": meta.get("cached_tokens"),
        "meta_keys": sorted(meta.keys()),
    }
    return rec, None


def run(args):
    prompts = []
    with open(args.prompts, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"] if isinstance(obj, dict) and "prompt" in obj else obj)
    if not prompts:
        print("空 prompt 集", file=sys.stderr)
        return 2
    prompts = prompts[: args.n]

    pre = internal_state(args.base)
    prev_acc = pre.get("spec_total_num_accept_tokens")
    prev_fwd = pre.get("spec_total_num_forward_ct")

    out = open(args.out, "w", encoding="utf-8")
    rows = []
    for rep in range(args.reps):
        for i, p in enumerate(prompts):
            rec, err = one_request(args.base, p, args.out_len, args.temperature, args.timeout)
            row = {"tag": args.tag, "rep": rep, "idx": i, "err": err}
            if rec:
                row.update(rec)
            # 路径 ③：累计计数器做差（**必须做差**，decision #66）
            st = internal_state(args.base)
            a, f = st.get("spec_total_num_accept_tokens"), st.get("spec_total_num_forward_ct")
            if None not in (a, f) and prev_acc is not None and prev_fwd is not None:
                da, df = a - prev_acc, f - prev_fwd
                row["cum_accept_tokens_delta"] = da
                row["cum_forward_ct_delta"] = df
                row["accept_len_from_counters"] = (da / df) if df else None
            if a is not None:
                prev_acc, prev_fwd = a, f
            row["internal_state_keys"] = sorted(st.keys())
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    out.close()

    ok = [r for r in rows if not r.get("err")]
    print(f"[{args.tag}] rows={len(rows)} ok={len(ok)} err={len(rows) - len(ok)}")
    for key in ("spec_accept_length", "spec_accept_rate", "accept_len_from_counters",
                "completion_tokens", "wall_s", "cached_tokens"):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if vals:
            print(f"  {key:26s} n={len(vals):3d} mean={statistics.fmean(vals):.4f} "
                  f"min={min(vals):.4f} max={max(vals):.4f}")
        else:
            print(f"  {key:26s} **未采到**")
    if not ok:
        print("  ⚠️ 全部失败；首条错误：", rows[0].get("err") if rows else "无行")
    return 0


def selftest():
    """离线自测：不连服务器，用假响应验证解析与计数器做差逻辑。"""
    import types
    global _post, _get
    state = {"a": 100, "f": 10}

    def fake_post(url, payload, timeout=600):
        assert url.endswith("/generate"), url
        assert payload["sampling_params"]["max_new_tokens"] == 8
        state["a"] += 5
        state["f"] += 2
        return {"meta_info": {"spec_accept_length": 2.5, "spec_accept_rate": 0.5,
                              "completion_tokens": 5, "prompt_tokens": 16,
                              "spec_verify_ct": 2, "cached_tokens": 0}}

    def fake_get(url, timeout=60):
        assert url.endswith("/get_internal_state"), url
        return {"internal_state": {"spec_total_num_accept_tokens": state["a"],
                                   "spec_total_num_forward_ct": state["f"]}}

    _post, _get = fake_post, fake_get
    prompts = "/tmp/_p12_selftest.jsonl"
    with open(prompts, "w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({"prompt": f"p{i}"}) + "\n")
    a = types.SimpleNamespace(tag="st", base="http://x", prompts=prompts, n=3, out_len=8,
                              reps=2, temperature=0.0, timeout=10,
                              out="/tmp/_p12_selftest_out.jsonl")
    run(a)
    rows = [json.loads(x) for x in open(a.out, encoding="utf-8")]
    assert len(rows) == 6, len(rows)
    # 第一次请求：pre 基线 (100,10) → post (105,12) ⇒ delta 5/2 = 2.5
    assert abs(rows[0]["accept_len_from_counters"] - 2.5) < 1e-9, rows[0]
    assert rows[0]["spec_accept_length"] == 2.5
    assert all(r["err"] is None for r in rows)
    print("selftest ✔ 请求解析/计数器做差/字段齐全/两 reps 边界")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--base", default="http://127.0.0.1:31001")
    ap.add_argument("--prompts", default="/root/autodl-tmp/prompts/prompts_4096.jsonl")
    ap.add_argument("--n", type=int, default=32, help="用前 n 条 prompt")
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", required=False, default="/tmp/p12.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
