#!/usr/bin/env python3
"""构造 (a2) 实验用的 prompt 集：带/不带共享前缀的两组「租户」prompt。

为什么需要它：要判定「新副本该从 0 学、还是从 warm 副本继承」，必须让
**源分布（warm 副本看到的租户）** 与 **目标副本的分布** 可控地相等或不等。
本脚本用「共享前缀」来制造真实世界里最常见的重合形态（系统提示 / few-shot 头 / RAG 模板）。

产物：
  tenantA_prefix.jsonl  源租户 prompt = 共享前缀 + 原 prompt    （warm 副本预热用）
  tenantB_prefix.jsonl  目标租户 prompt = **同一共享前缀** + 原 prompt（目标副本要服务的）
  tenantB_noprefix.jsonl 目标租户（**没有**该前缀）—— 用来证明差异确实来自前缀

用法：
  python build_tenant_prompts.py --src /root/autodl-tmp/prompts/prompts_4096.jsonl \
      --corpus /root/autodl-tmp/prompts/corpus.txt --n 32 --prefix-tokens 512 --outdir /tmp/tenants
  python build_tenant_prompts.py --selftest
"""
import argparse
import json
import os
import sys


def tokenize_prefix(corpus_path, want_tokens, model):
    """从 corpus.txt 取前 want_tokens 个 token，再解码成文本（保持真实文本形态）。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    with open(corpus_path, encoding="utf-8") as fh:
        text = fh.read()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < want_tokens:
        raise ValueError(f"corpus 只有 {len(ids)} token，不足 {want_tokens}")
    return tok.decode(ids[:want_tokens])


def load_prompts(path, n):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append(o["prompt"] if isinstance(o, dict) else o)
            if len(out) >= n:
                break
    return out


def write_jsonl(path, prompts):
    with open(path, "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")


def selftest():
    import tempfile
    td = tempfile.mkdtemp()
    src = os.path.join(td, "p.jsonl")
    with open(src, "w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(json.dumps({"prompt": f"base{i}"}) + "\n")
    ps = load_prompts(src, 2)
    assert ps == ["base0", "base1"]
    write_jsonl(os.path.join(td, "o.jsonl"), ["x"])
    assert json.loads(open(os.path.join(td, "o.jsonl"), encoding="utf-8").read())["prompt"] == "x"
    print("selftest ✔ prompt 读取/条数截取/JSONL 写出（tokenizer 部分在服务器上跑）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/autodl-tmp/prompts/prompts_4096.jsonl")
    ap.add_argument("--corpus", default="/root/autodl-tmp/prompts/corpus.txt")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--prefix-tokens", type=int, default=512)
    ap.add_argument("--outdir", default="/root/ccfa_results/2026-09-13/p12_tenant")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    os.makedirs(a.outdir, exist_ok=True)
    prompts = load_prompts(a.src, a.n)
    prefix = tokenize_prefix(a.corpus, a.prefix_tokens, a.model)
    print(f"共享前缀：{a.prefix_tokens} token，前 80 字符：{prefix[:80]!r}")
    write_jsonl(os.path.join(a.outdir, "tenantB_noprefix.jsonl"), prompts)
    write_jsonl(os.path.join(a.outdir, "tenantA_prefix.jsonl"), [prefix + p for p in prompts])
    write_jsonl(os.path.join(a.outdir, "tenantB_prefix.jsonl"), [prefix + p for p in prompts])
    for f in ("tenantA_prefix", "tenantB_prefix", "tenantB_noprefix"):
        p = os.path.join(a.outdir, f + ".jsonl")
        print(f"  {p}  lines={sum(1 for _ in open(p, encoding='utf-8'))}")
    print("说明：")
    print("  · 本轮**目标副本**要服务的 prompt 集 = tenantB_prefix（共享前缀 + 原 prompt）。")
    print("  · 「同分布源」= 用 tenantB_prefix 自己生成的续写（上界情形）。")
    print("  · 「异租户源」= 用 tenantB_noprefix 生成的续写（与目标只共享 512-token 前缀，"
          "其余不同）⇒ 用于测量跨租户迁移到底能迁移什么。")
    print("  · 「原文源」= 直接把 tenantB_prefix 的 prompt 文本当语料（零生成成本）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
