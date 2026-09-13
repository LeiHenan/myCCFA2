#!/usr/bin/env python3
"""p24（零 GPU）—— 直接调用 vLLM 的 `batch_propose_numba`，量出 **单线程 vs 多线程** 的提议器成本。

要回答的问题：`ngram_proposer.py` 把可用线程数硬限成 `min(1, cpu_count // 2)`（注释却写 "Cap ... to 8"，
TODO 写 "bump up the cap from 1 to 8"）。在本机（**208 核、TP=1**）这条 `prange` 并行路径**恒为 1 线程**。
⇒ **可回收量的上界 = t(1 线程) − t(8 线程) 的提议器墙钟**。

**这不能替代整机账**：它量的是**组件成本**。若提议器与 GPU 步重叠，代价可能被隐藏 ——
所以下一步还要用一次服务测它是否在关键路径上。本脚本只给**上界**。

用法（零 GPU）：
  bench_proposer.py --prompts /root/autodl-tmp/prompts/prompts_4096.jsonl \
      --tokenizer /root/autodl-tmp/models/Qwen3-4B --n-reqs 32 --ctx 4096 --k 3 \
      --threads 1,2,4,8,16,32 --reps 5 --out /root/ccfa_results/ngram_prop.json
  bench_proposer.py --selftest
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

def build_inputs(prompts_path: str, tokenizer_path: str, n_reqs: int, ctx: int, seed: int = 0):
    """用**真实文本**填上下文：n-gram 命中率决定提议器的工作量，随机 token 会严重低估它。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    texts = []
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["prompt"])
    ids_pool = []
    for t in texts:
        ids_pool.extend(tok.encode(t, add_special_tokens=False))
    if not ids_pool:
        raise SystemExit("空 prompt 集")
    rng = np.random.default_rng(seed)
    max_model_len = ctx + 8
    tok_cpu = np.zeros((n_reqs, max_model_len), dtype=np.int32)
    for r in range(n_reqs):
        start = int(rng.integers(0, max(1, len(ids_pool) - ctx)))
        chunk = ids_pool[start:start + ctx]
        if len(chunk) < ctx:                      # 环绕补齐
            chunk = (chunk * (ctx // max(1, len(chunk)) + 1))[:ctx]
        tok_cpu[r, :len(chunk)] = np.asarray(chunk, dtype=np.int32)
    num_tokens = np.full(n_reqs, ctx, dtype=np.int32)
    return tok_cpu, num_tokens, max_model_len

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts"); ap.add_argument("--tokenizer")
    ap.add_argument("--n-reqs", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--threads", default="1,2,4,8,16,32")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    from numba import get_num_threads, set_num_threads
    from vllm.v1.spec_decode.ngram_proposer import batch_propose_numba

    tok_cpu, num_tokens, max_model_len = build_inputs(a.prompts, a.tokenizer, a.n_reqs, a.ctx)
    valid_reqs = list(range(a.n_reqs))
    draft = np.zeros((a.n_reqs, a.k), dtype=np.int32)
    ndraft = np.zeros((a.n_reqs,), dtype=np.int32)
    total_tokens = int(num_tokens.sum())
    print(f"  输入：{a.n_reqs} 请求 × {a.ctx} token = {total_tokens} token"
          f"（引擎的启用多线程阈值 num_tokens_threshold=8192 ⇒ 本输入{'会' if total_tokens>=8192 else '不会'}走并行分支）")
    print(f"  可用核 os.cpu_count()={os.cpu_count()}；引擎实算线程数 min(1, cpu//2) = {min(1, (os.cpu_count() or 2)//2)}")

    orig = get_num_threads()
    results = {}
    try:
        seen_drafts = None
        for t in [int(x) for x in a.threads.split(",")]:
            set_num_threads(t)
            batch_propose_numba(valid_reqs, num_tokens, tok_cpu, a.min_n, a.max_n,
                                max_model_len, a.k, draft, ndraft)   # 预热/JIT
            times = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                batch_propose_numba(valid_reqs, num_tokens, tok_cpu, a.min_n, a.max_n,
                                    max_model_len, a.k, draft, ndraft)
                times.append((time.perf_counter() - t0) * 1e3)
            times.sort()
            med = times[len(times)//2]
            nz = int((ndraft > 0).sum())
            drafts = int(ndraft.sum())
            results[t] = {"median_ms": med, "min_ms": times[0], "max_ms": times[-1],
                          "requests_with_draft": nz, "total_drafts": drafts}
            print(f"  threads={t:3d}  中位 {med:8.3f} ms   (min {times[0]:.3f} / max {times[-1]:.3f})"
                  f"  有草稿的请求 {nz}/{a.n_reqs}  草稿总数 {drafts}")
            if seen_drafts is None:
                seen_drafts = (nz, drafts)
            elif (nz, drafts) != seen_drafts:
                print(f"  ⚠️ 不同线程数下产出不同（{seen_drafts} vs {(nz,drafts)}）⇒ 该路径**不确定**，结论不可用")
    finally:
        set_num_threads(orig)

    base = results.get(1, {}).get("median_ms")
    best = min((v["median_ms"] for v in results.values()), default=None)
    out = {"n_reqs": a.n_reqs, "ctx": a.ctx, "k": a.k, "min_n": a.min_n, "max_n": a.max_n,
           "total_tokens": total_tokens, "cpu_count": os.cpu_count(),
           "engine_thread_cap": min(1, (os.cpu_count() or 2)//2),
           "results": results, "t_1thread_ms": base, "t_best_ms": best,
           "recoverable_ms": (base - best) if (base and best) else None}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  ⇒ 单线程 {base:.3f} ms ｜ 最优 {best:.3f} ms ｜ **可回收 {out['recoverable_ms']:.3f} ms/次调用**")
    print("  ⚠️ 这是**组件上界**，不等于端到端收益：还需测提议器是否在关键路径上（可能与 GPU 步重叠）。")
    return 0

def selftest() -> int:
    # 用一个自带的小规模输入验证：线程数改变不应改变产出（正确性），且计时单调性不作断言（机器相关）
    import tempfile, types
    tok_cpu = np.zeros((4, 64), dtype=np.int32)
    pat = np.array([11, 22, 33, 44, 55, 66], dtype=np.int32)
    for r in range(4):
        tok_cpu[r, :64] = np.tile(pat, 11)[:64]      # 高度重复 ⇒ 必然有 n-gram 命中
    num_tokens = np.full(4, 64, dtype=np.int32)
    from numba import set_num_threads
    from vllm.v1.spec_decode.ngram_proposer import batch_propose_numba
    outs = []
    for t in (1, 4):
        set_num_threads(t)
        d = np.zeros((4, 3), dtype=np.int32); n = np.zeros((4,), dtype=np.int32)
        batch_propose_numba(list(range(4)), num_tokens, tok_cpu, 2, 4, 64, 3, d, n)
        outs.append((d.copy(), n.copy()))
    assert np.array_equal(outs[0][0], outs[1][0]), "线程数改变了产出（内容）"
    assert np.array_equal(outs[0][1], outs[1][1]), "线程数改变了产出（长度）"
    assert outs[0][1].sum() > 0, "自测输入没有产生任何草稿 ⇒ 自测无效"
    print("selftest OK —— 1 线程与 4 线程产出逐位一致；草稿数 =", int(outs[0][1].sum()))
    return 0

if __name__ == "__main__":
    sys.exit(main())
