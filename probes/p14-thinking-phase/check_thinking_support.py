#!/usr/bin/env python3
"""检查模型/引擎是否支持 thinking（reasoning）模式 —— S3b 前置事实，零 GPU。

为什么先查这个：「新特性刚 ship、其行为尚未被刻画」是唯一还没试过的窗口。
若模型根本不开 thinking，这条线当场作废，不必花任何 GPU。

用法：python check_thinking_support.py [--model /root/autodl-tmp/models/Qwen3-4B] [--selftest]
"""
import argparse
import json
import os
import sys

KWS = ("enable_thinking", "thinking", "reasoning", "<|think|>", "think")


def scan(model_dir):
    out = {"model_dir": model_dir, "exists": os.path.isdir(model_dir)}
    if not out["exists"]:
        return out
    tc_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tc_path):
        tc = json.load(open(tc_path, encoding="utf-8"))
        ct = tc.get("chat_template", "") or ""
        out["chat_template_len"] = len(ct)
        out["chat_template_flags"] = {k: (k in ct) for k in KWS}
        # 抽出与 thinking 有关的模板片段，便于人眼核
        frag = []
        for line in ct.splitlines():
            if "think" in line.lower():
                frag.append(line.strip()[:160])
        out["thinking_fragments"] = frag[:6]
    gc_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gc_path):
        out["generation_config"] = json.load(open(gc_path, encoding="utf-8"))
    return out


def scan_engine(vllm_pkg=None, sglang_pkg=None):
    res = {}
    for name, pkg, files in (
        ("vllm", vllm_pkg, ["entrypoints/openai/serving_engine.py",
                            "entrypoints/openai/protocol.py",
                            "reasoning/__init__.py"]),
        ("sglang", sglang_pkg, ["srt/server_args.py"]),
    ):
        if not pkg or not os.path.isdir(pkg):
            res[name] = {"pkg_found": False}
            continue
        hits = {}
        for f in files:
            fp = os.path.join(pkg, f)
            if not os.path.exists(fp):
                continue
            txt = open(fp, encoding="utf-8", errors="replace").read()
            hits[f] = {k: txt.count(k) for k in ("enable_thinking", "reasoning_parser", "thinking") if k in txt}
        res[name] = {"pkg_found": True, "hits": hits}
    return res


def selftest():
    import tempfile
    td = tempfile.mkdtemp()
    json.dump({"chat_template": "{% if enable_thinking %}...think...{% endif %}"},
              open(os.path.join(td, "tokenizer_config.json"), "w"))
    json.dump({"eos_token_id": 1}, open(os.path.join(td, "generation_config.json"), "w"))
    o = scan(td)
    assert o["exists"] and o["chat_template_flags"]["enable_thinking"] is True, o
    assert o["generation_config"]["eos_token_id"] == 1
    assert scan("/nonexistent")["exists"] is False
    e = scan_engine("/nonexistent", "/nonexistent")
    assert e["vllm"]["pkg_found"] is False
    print("selftest ✔ 模板 flag 扫描/片段抽取/generation_config/缺失路径")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--vllm", default="/root/ccfa_venv/lib/python3.12/site-packages/vllm")
    ap.add_argument("--sglang", default="/root/autodl-tmp/venvs/sglang/lib/python3.12/site-packages/sglang")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    print("=== 模型侧 ===")
    print(json.dumps(scan(a.model), ensure_ascii=False, indent=2)[:2500])
    print("\n=== 引擎侧 ===")
    print(json.dumps(scan_engine(a.vllm, a.sglang), ensure_ascii=False, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
