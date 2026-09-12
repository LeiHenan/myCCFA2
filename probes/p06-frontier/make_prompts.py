#!/usr/bin/env python3
"""为 frontier 探针构建**固定长度的真实文本 prompt**（JSONL，供 `vllm bench serve --dataset-name custom`）。

为什么需要它（2026-09-12 实测）：
  `vllm bench serve --dataset-name random` 喂的是**随机 token id**。drafter 是在自然文本上训练的，
  对随机序列无从预测 ⇒ **每一档 depth 的接受长度都是 1.00（零接受）**，脊线被整体压平、失去信号。
  改用连续真实文本后才有可比的接受率。

用法：
  # 从 WikiText（datasets 库，需网络/HF 镜像）构建
  python make_prompts.py --ctx 4096  --num 8 --out prompts_4k.jsonl   --tokenizer <模型目录>
  python make_prompts.py --ctx 131072 --num 8 --out prompts_128k.jsonl --tokenizer <模型目录>

  # 或从本地纯文本文件构建（离线可用；可给多个文件，按顺序拼接）
  python make_prompts.py --ctx 4096 --num 8 --out p.jsonl --tokenizer <dir> --text-file a.md b.md

输出：每行 {"prompt": "<恰好约 ctx token 的文本>", "output_tokens": 128}
"""

import argparse
import json
import sys
from pathlib import Path


def load_corpus_text(text_files: list[str]) -> str:
    if text_files:
        parts = []
        for f in text_files:
            parts.append(Path(f).read_text(encoding="utf-8", errors="ignore"))
        return "\n\n".join(parts)
    try:
        from datasets import load_dataset
    except Exception:
        sys.exit("需要 datasets（pip install datasets）或改用 --text-file")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    if len(text) < 3_000_000:  # test split 偏小，补 train
        try:
            tr = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:20000]")
            text += "\n\n" + "\n\n".join(r["text"] for r in tr if r["text"].strip())
        except Exception as e:
            print(f"  (补充 train 失败，仅用 test：{type(e).__name__})", file=sys.stderr)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True, help="每条 prompt 的目标 token 数")
    ap.add_argument("--num", type=int, default=8, help="prompt 条数")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="HF 模型目录（用其 tokenizer 计数）")
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--text-file", nargs="*", default=[], help="本地纯文本文件（离线路径）")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    text = load_corpus_text(a.text_file)
    print(f"  语料字符数: {len(text):,}")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"  语料 token 数: {len(ids):,}")
    need = a.ctx * a.num
    if len(ids) < need:
        print(f"  ⚠️ 语料不足：需 {need:,} tokens，只有 {len(ids):,}（可增加语料或减少 --num）", file=sys.stderr)

    # 均匀取 num 段，每段恰好 ctx 个 token（等距错开，避免各条重叠）
    stride = max(a.ctx, (len(ids) - a.ctx) // max(1, a.num))
    rows = []
    for i in range(a.num):
        s = i * stride
        e = s + a.ctx
        if e > len(ids):
            break
        rows.append({
            "prompt": tok.decode(ids[s:e]),
            "output_tokens": a.output_tokens,
        })
    if not rows:
        sys.exit("一条 prompt 都构造不出来（语料太短）")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 校验：重新编码后的实际长度
    lens = [len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) for r in rows]
    print(f"  已写 {len(rows)} 条 → {a.out}")
    print(f"  实际 token 长度: min={min(lens)} max={max(lens)} (目标 {a.ctx})")


if __name__ == "__main__":
    main()
