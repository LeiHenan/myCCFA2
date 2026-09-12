#!/usr/bin/env python3
"""从已发布的 speculator 权重生成"深度变体"（T0 用）。

只改 config 的层数相关字段（默认不动权重张量）：
  num_hidden_layers / n_layer / num_layers        -> 设为 d
  transformer_layer_config（若为 list）            -> 截短到 d

用法：
  python make_depth_variants.py --src SRC --out OUT --depths 1 3 5
  python make_depth_variants.py --src SRC --out OUT --depths 1 3 5 --prune-weights
  python make_depth_variants.py --selftest

注意：**未在真机验证过**；--prune-weights 需要 `safetensors`。
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers")
LIST_KEYS = ("transformer_layer_config",)


def rewrite_config(cfg: dict, d: int) -> tuple[dict, list[str]]:
    """把层数相关字段设为 d，返回 (新 config, 改动说明)。"""
    notes, out = [], dict(cfg)
    for k in LAYER_COUNT_KEYS:
        if k in out and isinstance(out[k], int):
            notes.append(f"{k}: {out[k]} -> {d}")
            out[k] = d
    for k in LIST_KEYS:
        v = out.get(k)
        if isinstance(v, list) and len(v) >= d:
            notes.append(f"{k}: len {len(v)} -> {d}")
            out[k] = v[:d]
    return out, notes


def prune_safetensors(path: Path, d: int, notes: list[str]) -> None:
    """丢弃 index >= d 的层权重（按 'layers.<i>.' 命名启发式）。"""
    try:
        from safetensors.torch import load_file, save_file
    except Exception:
        sys.exit("--prune-weights 需要 safetensors：pip install safetensors")
    import re

    tensors = load_file(str(path))
    pat = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")
    kept, dropped = {}, 0
    for name, t in tensors.items():
        m = pat.search(name)
        if m and int(m.group(1)) >= d:
            dropped += 1
            continue
        kept[name] = t
    save_file(kept, str(path))
    notes.append(f"{path.name}: dropped {dropped} tensor(s)")


def make_variants(src: Path, out: Path, depths: list[int], prune: bool) -> None:
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        sys.exit(f"缺 {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for d in depths:
        dst = out / f"d{d}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        new_cfg, notes = rewrite_config(cfg, d)
        (dst / "config.json").write_text(
            json.dumps(new_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if prune:
            for st in sorted(dst.glob("*.safetensors")):
                prune_safetensors(st, d, notes)
        print(f"  d{d}: " + ("; ".join(notes) if notes else "无字段可改（检查 config 键名）"))


def selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "src", Path(td) / "out"
        src.mkdir(parents=True)
        (src / "config.json").write_text(json.dumps({
            "architectures": ["DFlash2DraftModel"],
            "num_hidden_layers": 5,
            "transformer_layer_config": [{"i": i} for i in range(5)],
            "block_size": 8,
        }), encoding="utf-8")
        (src / "model.safetensors").write_bytes(b"stub")
        make_variants(src, out, [1, 3, 5], prune=False)
        for d in (1, 3, 5):
            c = json.loads((out / f"d{d}" / "config.json").read_text(encoding="utf-8"))
            assert c["num_hidden_layers"] == d, c
            assert len(c["transformer_layer_config"]) == d, c
            assert c["block_size"] == 8, "非层数字段必须保持不变"
        print("selftest ✔ 配置改写正确且未动其它字段")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src"); ap.add_argument("--out")
    ap.add_argument("--depths", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--prune-weights", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.src and a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        make_variants(Path(a.src), Path(a.out), a.depths, a.prune_weights)
    else:
        ap.error("需要 --src/--out，或 --selftest")
