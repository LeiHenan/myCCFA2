#!/usr/bin/env python3
"""p12 语料恢复客户端 —— 测「从持久化恢复」的成本，并与「重新生成」对比。

**为什么单独写**：`corpus_client.py` 的 `add` 只报装载成功，不报**耗时**；
而本阶段的核心量就是**代价对比**（恢复 vs 重新生成），所以必须把
「编码 + HTTP + 服务端 SAM 构建」整段墙钟单独测出来。

两类恢复（故意都测，因为它们的代价结构不同）：
  --docs-file <jsonl>   从落盘的 **documents** 恢复 ⇒ 服务端每次都要重新 encode（CPU 成本）
  --tokens-file <jsonl> 从落盘的 **token id 块**恢复 ⇒ 无 encode，但要求引擎支持 token_chunks
                        （`AddExternalCorpusReqInput.token_chunks`，见 io_struct.py:1674）

用法：
  python restore_client.py --base http://127.0.0.1:31010 --docs-file /path/d.jsonl --n 32 --id r1 --out rec.jsonl
  python restore_client.py --base ... --tokens-file /path/t.jsonl --n 32 --id r2 --out rec2.jsonl
  python restore_client.py --selftest
"""
import argparse
import json
import statistics as st
import sys
import time
import urllib.error
import urllib.request


def _post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:  # noqa: BLE001
            return e.code, {"_raw": body[:400]}


def load_docs(path, n):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append(o["prompt"] if isinstance(o, dict) and "prompt" in o else o)
            if len(out) >= n:
                break
    return out


def load_token_chunks(path, n):
    """token 块文件：每行是 JSON 数组（token id 序列）。"""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if isinstance(o, dict):
                o = o.get("input_ids") or o.get("tokens") or o.get("token_ids")
            if not isinstance(o, list):
                raise ValueError("token 块文件的每行必须是 JSON 数组，或含 input_ids/tokens 的对象")
            out.append(o)
            if len(out) >= n:
                break
    return out


def restore(base, payload, timeout):
    t0 = time.perf_counter()
    st_, d = _post(base + "/add_external_corpus", payload, timeout)
    wall = time.perf_counter() - t0
    return st_, d, wall


def run_tokens(args):
    """A 类：先由客户端把 docs 编码成 token 块并落盘，再多次从 token 块恢复（测无-encode 路径）。"""
    from transformers import AutoTokenizer  # 延迟导入：只有本子命令需要
    docs = load_docs(args.docs_file, args.n)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.perf_counter()
    chunks = [tok.encode(d, add_special_tokens=False) for d in docs]
    enc_wall = time.perf_counter() - t0
    with open(args.tokens_out, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    print(f"[encode] docs={len(docs)} tokens={sum(len(c) for c in chunks)} "
          f"encode_wall={enc_wall:.3f}s -> {args.tokens_out}")
    st_, d, wall = restore(args.base, {"corpus_id": args.id, "token_chunks": chunks}, args.timeout)
    rec = {"mode": "tokens", "stage": "restore", "http": st_, "success": d.get("success"),
           "loaded_token_count": d.get("loaded_token_count"), "wall_s": wall,
           "encode_wall_s": enc_wall, "message": d.get("message")}
    _emit(rec, args.out)
    return 0 if d.get("success") else 1


def run_docs(args):
    docs = load_docs(args.docs_file, args.n)
    st_, d, wall = restore(args.base, {"corpus_id": args.id, "documents": docs}, args.timeout)
    rec = {"mode": "documents", "stage": "restore", "http": st_, "success": d.get("success"),
           "loaded_token_count": d.get("loaded_token_count"), "wall_s": wall,
           "n_docs": len(docs), "message": d.get("message")}
    _emit(rec, args.out)
    return 0 if d.get("success") else 1


def _emit(rec, out):
    print(json.dumps(rec, ensure_ascii=False))
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def selftest():
    import os
    import tempfile
    global _post
    seen = {}

    def fake(url, payload, timeout=1800):
        seen["payload"] = payload
        tok = sum(len(c) for c in payload.get("token_chunks", [])) or 42
        return 200, {"success": True, "corpus_id": payload["corpus_id"],
                     "loaded_token_count": tok, "message": "ok"}

    _post = fake
    td = tempfile.mkdtemp()
    dpath = os.path.join(td, "d.jsonl")
    with open(dpath, "w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(json.dumps({"prompt": f"doc{i}"}) + "\n")
    tpath = os.path.join(td, "t.jsonl")
    with open(tpath, "w", encoding="utf-8") as fh:
        fh.write(json.dumps([1, 2, 3]) + "\n")
        fh.write(json.dumps({"input_ids": [4, 5]}) + "\n")
    assert load_token_chunks(tpath, 5) == [[1, 2, 3], [4, 5]]
    assert load_docs(dpath, 2) == ["doc0", "doc1"]
    import types
    rc = run_docs(types.SimpleNamespace(docs_file=dpath, n=3, base="http://x", id="c",
                                        timeout=5, out=os.path.join(td, "rec.jsonl")))
    assert rc == 0 and seen["payload"]["documents"] == ["doc0", "doc1", "doc2"]
    rows = [json.loads(l) for l in open(os.path.join(td, "rec.jsonl"), encoding="utf-8")]
    assert len(rows) == 1 and rows[0]["mode"] == "documents" and rows[0]["wall_s"] >= 0
    print("selftest ✔ documents 恢复载荷/条数截取/token_chunks 两种格式/计时落盘")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:31010")
    ap.add_argument("--docs-file")
    ap.add_argument("--tokens-file")
    ap.add_argument("--tokens-out", default="")
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--id", default="restore")
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.tokens_out:
        return run_tokens(a)
    if a.docs_file:
        return run_docs(a)
    ap.error("需要 --docs-file 或 --tokens-file，或 --selftest")


if __name__ == "__main__":
    sys.exit(main())
