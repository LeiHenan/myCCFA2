#!/usr/bin/env python3
"""构造「共享长前缀」的 prompt 集：cascade attention 只有在批内请求共享 **≥256 token** 的公共前缀时才可能启用。

为什么必须构造：我们现成的 `prompts_4096.jsonl` 是 32 篇**不同**文章 ⇒ 批内公共前缀 ≈ 0 ⇒ cascade 永远不触发。
本脚本用**一个** 4096-token 前缀 + 每请求一段**互不相同**的短后缀，使批内公共前缀 = 4096。

⚠️ 关键机制（我在 `gpu_model_runner.py` 亲验）：`common_prefix_len = min(common_prefix_len, num_computed_tokens.min())`
⇒ **公共前缀只在"已被计算/命中缓存"的那部分才算数** ⇒ 本实验必须**开启前缀缓存**（默认即开启）。
"""
from __future__ import annotations
import argparse, json, sys

def main() -> int:
    ap = argparse.ArgumentParser()
    # 不用 required=True —— 否则 --selftest 会在校验前就被 argparse 拒绝（同类坑第 2 次）
    ap.add_argument("--src", help="源 prompt 集（取其第一条做共享前缀）")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--suffix-chars", type=int, default=0,
                    help="每条请求的**唯一后缀**字符数（用源文件里其它文章拼）。0=只加短标记。")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.src or not a.out:
        print("需要 --src 与 --out", file=sys.stderr); return 2
    with open(a.src, encoding="utf-8") as f:
        src = [json.loads(l)["prompt"] for l in f if l.strip()]
    prefix = src[0]
    rows = []
    for i in range(a.n):
        # 唯一后缀：用**另一篇**文章的开头，保证既不与公共前缀重合、又足够长（cascade 需要有活干）
        if a.suffix_chars:
            body = src[(i % (len(src) - 1)) + 1]
            tail = "\n\n" + body[: a.suffix_chars]
        else:
            tail = ""
        # 再补一个短的唯一标记，确保每条请求彼此不同
        tail += (f"\n\n[Request {i:03d}] case {i:03d} is uniquely numbered {i*7919 % 100000:05d}.\n")
        rows.append({"prompt": prefix + tail})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写入 {len(rows)} 条到 {a.out}；共享前缀 ≈{len(prefix)} 字符 + 每条唯一后缀 ≈{a.suffix_chars} 字符")
    return 0

def selftest() -> int:
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.jsonl"); out = os.path.join(d, "o.jsonl")
        open(src, "w").write(json.dumps({"prompt": "A" * 500}) + "\n")
        sys.argv = ["x", "--src", src, "--out", out, "--n", "5"]
        assert main() == 0
        rows = [json.loads(l) for l in open(out)]
        assert len(rows) == 5
        pre = rows[0]["prompt"][:500]
        assert all(r["prompt"][:500] == pre for r in rows), "公共前缀不一致"
        assert len({r["prompt"] for r in rows}) == 5, "请求不唯一"
        print("selftest OK —— 5 条请求共享 500 字符前缀且彼此唯一")
    return 0

if __name__ == "__main__":
    sys.exit(main())
