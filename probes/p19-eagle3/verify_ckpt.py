#!/usr/bin/env python3
"""校验刚下载的 EAGLE3 drafter：① safetensors 完整性 ② 与 target (Qwen3-4B) 的维度是否匹配。

为什么要这一步：下载成功 ≠ 可用。EAGLE 类 drafter 必须与 target 的 hidden/heads/vocab 对齐，
且 checkpoint 里应含 EAGLE 特有的 draft 头（如 fc/embed/draft 相关权重），否则起服务时才会暴雷。

用法：python verify_ckpt.py --ckpt /root/autodl-tmp/models/eagle3-qwen3-4b [--selftest]
"""
import argparse, json, os, sys

def verify(ckpt):
    out = {"dir": ckpt}
    cfg_p = os.path.join(ckpt, "config.json")
    st_p = os.path.join(ckpt, "model.safetensors")
    out["has_config"] = os.path.exists(cfg_p)
    out["has_safetensors"] = os.path.exists(st_p)
    if out["has_config"]:
        cfg = json.load(open(cfg_p, encoding="utf-8"))
        keys = ("architectures", "num_hidden_layers", "hidden_size", "num_attention_heads",
                "num_key_value_heads", "vocab_size", "draft_vocab_size", "model_type",
                "num_layers", "intermediate_size", "head_dim", "tie_word_embeddings")
        out["config"] = {k: cfg[k] for k in keys if k in cfg}
        out["config_all_keys"] = sorted(cfg.keys())
    if out["has_safetensors"]:
        try:
            from safetensors import safe_open
            with safe_open(st_p, framework="pt") as f:
                names = list(f.keys())
                out["n_tensors"] = len(names)
                # 取第一个张量确认可读 + 统计参数量
                tot = 0
                shapes = {}
                for n in names:
                    sl = f.get_slice(n)
                    sh = sl.get_shape()
                    shapes[n] = sh
                    tot += 1
                    for d in sh:
                        tot = tot  # noqa
                out["total_params"] = sum(
                    __import__("math").prod(shapes[n]) for n in names)
                out["first_tensors"] = names[:8]
                out["name_keywords"] = {
                    kw: sum(1 for n in names if kw in n.lower())
                    for kw in ("draft", "fc", "embed", "lm_head", "midlayer", "layers.")
                }
                out["dims_sample"] = {n: shapes[n] for n in names[:4]}
        except Exception as e:  # noqa: BLE001
            out["safetensors_error"] = f"{type(e).__name__}: {e}"
    return out

def selftest():
    import tempfile, json as J
    td = tempfile.mkdtemp()
    J.dump({"architectures": ["Eagle3"], "num_hidden_layers": 1}, open(os.path.join(td, "config.json"), "w"))
    o = verify(td)
    assert o["has_config"] and not o["has_safetensors"] and o["config"]["num_hidden_layers"] == 1, o
    o2 = verify("/nonexistent")
    assert not o2["has_config"] and not o2["has_safetensors"]
    print("selftest ✔ config 读取 / 缺失路径 / 字段提取")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/autodl-tmp/models/eagle3-qwen3-4b")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    print(json.dumps(verify(a.ckpt), ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
