#!/usr/bin/env python3
"""p12 语料客户端：把一组文档装进 SGLang 的外置语料（动态 HTTP API）。

**为什么用 `documents` 而不是 `file_path`**（读源码的结论，decision #101）：
`srt/managers/tokenizer_control_mixin.py:209-243` 显示两条路径：
  - `file_path` → `iter_external_corpus_chunks(path, tokenizer, max_tokens)`，**由 loader 决定文件格式**
    （上轮 decision #95/#96 撞到的 JSONL / JSON 字符串约束即来自这里）；
  - `documents: List[str]` → 引擎**自己** `tokenizer.encode(doc, add_special_tokens=False)`，
    文档之间插 `SEPARATOR_TOKEN`，并在超过 `speculative_ngram_external_corpus_max_tokens` 时**截断**
    （返回 message 里带 `(truncated: exceeded N token limit)`）。
⇒ 用 `documents` 可以完全绕开文件格式问题，且**截断行为显式可见**。

用法：
  python corpus_client.py --base http://127.0.0.1:31001 add --docs-file <prompts.jsonl> --n 32 --id prewarm
  python corpus_client.py --base http://127.0.0.1:31001 list
  python corpus_client.py --base http://127.0.0.1:31001 remove --id prewarm
  python corpus_client.py --selftest
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def _req(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:  # noqa: BLE001
            return e.code, {"_raw": body[:400]}


def load_docs(path, n, field="prompt"):
    docs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(obj[field] if isinstance(obj, dict) else obj)
            if len(docs) >= n:
                break
    return docs


def cmd_add(a):
    docs = load_docs(a.docs_file, a.n, a.field)
    payload = {"corpus_id": a.id, "documents": docs}
    st, d = _req(a.base + "/add_external_corpus", payload, timeout=a.timeout)
    print(f"HTTP {st}  success={d.get('success')}  corpus_id={d.get('corpus_id')}  "
          f"loaded_token_count={d.get('loaded_token_count')}")
    print("message:", d.get("message"))
    if d.get("message") and "truncated" in str(d.get("message")):
        print("⚠️ 被截断：语料超过 --speculative-ngram-external-corpus-max-tokens")
    return 0 if d.get("success") else 1


def cmd_list(a):
    st, d = _req(a.base + "/list_external_corpora")
    print(f"HTTP {st}  {json.dumps(d, ensure_ascii=False)[:600]}")
    return 0


def cmd_remove(a):
    st, d = _req(a.base + "/remove_external_corpus", {"corpus_id": a.id})
    print(f"HTTP {st}  {json.dumps(d, ensure_ascii=False)[:400]}")
    return 0 if d.get("success") else 1


def selftest():
    global _req
    seen = {}

    def fake(url, payload=None, timeout=600):
        seen["url"], seen["payload"] = url, payload
        if url.endswith("/add_external_corpus"):
            return 200, {"success": True, "corpus_id": payload["corpus_id"],
                         "message": "Loaded corpus 'x' with 42 tokens.",
                         "loaded_token_count": 42}
        if url.endswith("/list_external_corpora"):
            return 200, {"success": True, "corpus_token_counts": {"x": 42}}
        return 200, {"success": True}

    _req = fake
    p = "/tmp/_p12_corpus_selftest.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for i in range(5):
            fh.write(json.dumps({"prompt": f"doc{i}"}) + "\n")
    import types
    assert cmd_add(types.SimpleNamespace(base="http://x", docs_file=p, n=3, field="prompt",
                                         id="c1", timeout=5)) == 0
    assert seen["payload"]["documents"] == ["doc0", "doc1", "doc2"], seen["payload"]
    assert cmd_list(types.SimpleNamespace(base="http://x")) == 0
    assert cmd_remove(types.SimpleNamespace(base="http://x", id="c1")) == 0
    print("selftest ✔ documents 载荷/条数截取/list/remove")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:31001")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p_add = sub.add_parser("add")
    p_add.add_argument("--docs-file", required=True)
    p_add.add_argument("--n", type=int, default=32)
    p_add.add_argument("--field", default="prompt")
    p_add.add_argument("--id", default="prewarm")
    p_add.add_argument("--timeout", type=float, default=600)
    p_add.set_defaults(func=cmd_add)
    p_ls = sub.add_parser("list")
    p_ls.set_defaults(func=cmd_list)
    p_rm = sub.add_parser("remove")
    p_rm.add_argument("--id", required=True)
    p_rm.set_defaults(func=cmd_remove)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not getattr(a, "func", None):
        ap.error("需要子命令 add/list/remove，或 --selftest")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
