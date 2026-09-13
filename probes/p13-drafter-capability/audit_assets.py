#!/usr/bin/env python3
"""p13 资产审计：本地 drafter 的能力声明 + 公开 deeper drafter 的可取性（零 GPU）。

为什么需要：A1 归档数据显示"每步提议数"轴（d1→d5）有 57.5–92.6% 的吞吐差，
而 vLLM 的约束是 `num_speculative_tokens` 默认取 **draft 模型 config 的 `n_predict`**
且必须整除（`config/speculative.py:1407-1427`）⇒ **能吃到多少收益取决于 checkpoint 自己声明的能力**。
本脚本把"我们手上有什么 / 外面有什么"变成可核事实。

用法：python audit_assets.py [--repo <hf repo id>] [--selftest]
"""
import argparse
import json
import os
import sys
import urllib.request

LOCAL = ["/root/autodl-tmp/models/dflash2", "/root/autodl-tmp/models/Qwen3-4B"]
KEYS = ("num_hidden_layers", "n_predict", "num_lookahead_tokens", "num_nextn_predict_layers",
        "block_size", "architectures", "model_type", "speculative_num_draft_tokens")


def local_audit(path):
    out = {"path": path, "exists": os.path.isdir(path)}
    if not out["exists"]:
        return out
    out["files"] = sorted(os.listdir(path))[:20]
    cfg = os.path.join(path, "config.json")
    if os.path.exists(cfg):
        d = json.load(open(cfg, encoding="utf-8"))
        out["declared"] = {k: d[k] for k in KEYS if k in d}
    else:
        out["declared"] = None
    # 权重文件与大小（用于估算下载量）
    tot = 0
    for f in os.listdir(path):
        p = os.path.join(path, f)
        if os.path.isfile(p) and f.endswith((".safetensors", ".bin", ".pt")):
            tot += os.path.getsize(p)
    out["weights_bytes"] = tot
    return out


def remote_files(repo, endpoint="https://hf-mirror.com"):
    url = f"{endpoint}/api/models/{repo}"
    req = urllib.request.Request(url, headers={"User-Agent": "ccfa-audit"})
    with urllib.request.urlopen(req, timeout=45) as r:
        d = json.loads(r.read().decode())
    return [s.get("rfilename") for s in d.get("siblings", [])], d


def selftest():
    a = local_audit("/nonexistent")
    assert a["exists"] is False
    import tempfile
    td = tempfile.mkdtemp()
    json.dump({"n_predict": 3, "num_hidden_layers": 1}, open(os.path.join(td, "config.json"), "w"))
    open(os.path.join(td, "model.safetensors"), "wb").write(b"x" * 10)
    b = local_audit(td)
    assert b["exists"] and b["declared"]["n_predict"] == 3 and b["weights_bytes"] == 10, b
    print("selftest ✔ 本地 config 关键字段提取 / 权重体积统计 / 缺失目录处理")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="huluhuluu/Qwen3-4B-Instruct-2507-EAGLE3-ShareGPT-NoWindow-epoch-1-step-30000")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    print("=== 本地资产 ===")
    for p in LOCAL:
        print(json.dumps(local_audit(p), ensure_ascii=False, indent=2))
    print("\n=== 公开 deeper drafter 仓库文件清单 ===")
    try:
        files, d = remote_files(a.repo)
        for f in files[:30]:
            print("  ", f)
        print("  文件数：", len(files))
    except Exception as e:  # noqa: BLE001
        print(f"  取不到：{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
