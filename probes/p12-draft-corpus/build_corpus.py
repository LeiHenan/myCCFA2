#!/usr/bin/env python3
"""把某个 /generate 响应里的**模型续写**导出成语料 JSONL（上游 output-as-corpus 语义）。

用法：
  python build_corpus.py --in <probe.jsonl 或含 output 的 jsonl> --out corpus_out.jsonl
  python build_corpus.py --selftest
说明：本探针的原始 jsonl 只存了 meta_info；若没有 output 字段，用 --from-text 从
      `--print-requests` 之外的方式拿不到 ⇒ 此时应改用 `probe.py --save-output`。
"""
import argparse, json, sys


def selftest():
    import tempfile, os
    rows = [{"tag": "t", "rep": 0, "idx": 0, "output": "hello world"},
            {"tag": "t", "rep": 0, "idx": 1, "output": "foo bar"}]
    td = tempfile.mkdtemp()
    p = os.path.join(td, "in.jsonl")
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n = build(p, os.path.join(td, "out.jsonl"))
    assert n == 2, n
    got = [json.loads(l)["prompt"] for l in open(os.path.join(td, "out.jsonl"), encoding="utf-8")]
    assert got == ["hello world", "foo bar"], got
    print("selftest ✔ 导出 output 为语料 JSONL")
    return 0


def build(inp, outp):
    n = 0
    with open(inp, encoding="utf-8") as fi, open(outp, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            txt = r.get("output")
            if not txt:
                continue
            fo.write(json.dumps({"prompt": txt}, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in"); ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.__dict__["in"] and a.out):
        ap.error("需要 --in 与 --out")
    print("导出", build(a.__dict__["in"], a.out), "条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
