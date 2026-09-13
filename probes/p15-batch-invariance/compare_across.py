#!/usr/bin/env python3
"""对比"不同并发度/不同 BI 设置"下的输出是否逐字相同 —— 批组成依赖的直接判据。

为什么必需：`divergence=0`（同批内 n 份一致）**不能**证明批不变性，因为可能这 n 份根本没合批。
真正的判据是**跨批组成**对比：同一请求在 n=1 / n=16 下是否给出同一串文本。

用法：
  python compare_across.py --dir /root/ccfa_results/2026-09-13/p15_quick
  python compare_across.py --selftest
"""
import argparse
import collections
import glob
import json
import os
import re
import sys


def load_first_text(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    ok = [r for r in rows if not r.get("err")]
    if not ok:
        return None, (rows[0].get("err") if rows else "empty")
    r = ok[0]
    st = r.get("sample_texts") or []
    return (st[0] if st else None), None


def parse_name(p):
    m = re.search(r"bi(\d+)_n(\d+)\.jsonl$", os.path.basename(p))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def analyze(d, verbose=True):
    files = sorted(glob.glob(os.path.join(d, "bi*_n*.jsonl")))
    if not files:
        print("没有找到 bi*_n*.jsonl")
        return None
    by_bi = collections.defaultdict(dict)
    for f in files:
        bi, n = parse_name(f)
        if bi is None:
            continue
        t, err = load_first_text(f)
        by_bi[bi][n] = {"text": t, "err": err, "file": os.path.basename(f)}
    out = {"by_bi": by_bi, "across_n": {}, "across_bi": None}
    for bi, d2 in sorted(by_bi.items()):
        texts = {n: v["text"] for n, v in d2.items() if v["text"] is not None}
        uniq = set(texts.values())
        out["across_n"][bi] = {"n_levels": sorted(texts), "unique_texts": len(uniq),
                               "same_across_n": len(uniq) == 1 and len(texts) > 1}
        if verbose:
            print(f"BI={bi}: 并发度 {sorted(texts)} ⇒ 不同输出数 = {len(uniq)}"
                  f" ⇒ 跨批组成{'一致 ✅' if len(uniq) == 1 and len(texts) > 1 else '不一致 ❌'}")
            for n in sorted(texts):
                print(f"    n={n:<3} {(texts[n] or '')[:70]!r}")
    if 0 in by_bi and 1 in by_bi:
        t0 = {n: v["text"] for n, v in by_bi[0].items() if v["text"] is not None}
        t1 = {n: v["text"] for n, v in by_bi[1].items() if v["text"] is not None}
        common = sorted(set(t0) & set(t1))
        if common:
            n0 = common[0]
            same = t0[n0] == t1[n0]
            out["across_bi"] = {"n": n0, "same": same}
            if verbose:
                print(f"\nBI=0 vs BI=1（同 n={n0}）⇒ {'输出相同' if same else '输出不同 ⚠️（开关改变了结果）'}")
    return out


def selftest():
    import tempfile
    td = tempfile.mkdtemp()

    def w(name, text, err=None):
        with open(os.path.join(td, name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"err": err, "sample_texts": [text] if text else []}) + "\n")
    w("bi0_n1.jsonl", "AAA"); w("bi0_n8.jsonl", "AAA"); w("bi0_n16.jsonl", "AAA")
    w("bi1_n8.jsonl", "BBB")
    o = analyze(td, verbose=False)
    assert o["across_n"][0]["same_across_n"] is True, o["across_n"]
    assert o["across_n"][1]["unique_texts"] == 1
    assert o["across_bi"]["same"] is False, "BI=0 与 BI=1 文本不同 ⇒ same 应为 False"
    w("bi0_n16.jsonl", "ZZZ")
    o2 = analyze(td, verbose=False)
    assert o2["across_n"][0]["same_across_n"] is False, o2["across_n"]
    print("selftest ✔ 跨并发度一致性 / 跨 BI 对比 / 不一致能被抓到")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/ccfa_results/2026-09-13/p15_quick")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    analyze(a.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
