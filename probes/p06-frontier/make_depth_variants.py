#!/usr/bin/env python3
"""从已发布的 speculator 权重生成"深度变体"（T0 用）。

只改 config 的层数相关字段（默认不动权重张量）。

**真实 dflash / dflash2 配置的形态（2026-09-12 按 `mgoin/Qwen3-4B-speculator.dflash2` 实测核对）**：
  - 顶层**没有** `num_hidden_layers`；
  - 层数在 **`transformer_layer_config`（dict，Qwen3Config 形态）** 里的 `num_hidden_layers`（=5）；
  - 同层还有 `layer_types`（逐层 SWA/Full 标记，长度须与层数一致），**必须同步截短**。
  vLLM 侧对应（`vllm/model_executor/models/qwen3_dflash.py`，v0.29.0）：
  `self.layers = ModuleList([... for layer_idx in range(self.config.num_hidden_layers)])`
  与 `_dflash_layer_causal()` 里的 `layer_types[layer_idx]`，其中 `self.config` 即 `transformer_layer_config`。

用法：
  python make_depth_variants.py --src SRC --out OUT --depths 1 3 5
  python make_depth_variants.py --src SRC --out OUT --depths 1 3 5 --prune-weights
  python make_depth_variants.py --selftest

注意：`--prune-weights` 需要 `safetensors`；不裁剪权重时由 vLLM 的非严格加载忽略多余层。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers")
NESTED_KEYS = ("transformer_layer_config",)
PER_LAYER_LIST_KEYS = ("layer_types", "layers")


def rewrite_config(cfg: dict, d: int) -> tuple[dict, list[str]]:
    """把层数相关字段设为 d，返回 (新 config, 改动说明)。"""
    notes: list[str] = []
    out = dict(cfg)

    # (1) 顶层键（部分 checkpoint 有）
    for k in LAYER_COUNT_KEYS:
        v = out.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            notes.append(f"{k}: {v} -> {d}")
            out[k] = d

    # (2) transformer_layer_config：dict（dflash/dflash2 真实形态）或 list（旧形态）
    for k in NESTED_KEYS:
        v = out.get(k)
        if isinstance(v, dict):
            new_v = dict(v)
            for kk in LAYER_COUNT_KEYS:
                vv = new_v.get(kk)
                if isinstance(vv, int) and not isinstance(vv, bool):
                    notes.append(f"{k}.{kk}: {vv} -> {d}")
                    new_v[kk] = d
            for kk in PER_LAYER_LIST_KEYS:
                vv = new_v.get(kk)
                if isinstance(vv, list) and len(vv) > d:
                    notes.append(f"{k}.{kk}: len {len(vv)} -> {d}")
                    new_v[kk] = vv[:d]
            out[k] = new_v
        elif isinstance(v, list) and len(v) >= d:
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


def make_variants(src: Path, out: Path, depths: list[int], prune: bool,
                  symlink_weights: bool = False) -> None:
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        sys.exit(f"缺 {cfg_path}")
    if prune and symlink_weights:
        sys.exit("--prune-weights 与 --symlink-weights 互斥：裁剪会改到共享的源权重文件")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for d in depths:
        dst = out / f"d{d}"
        new_cfg, notes = rewrite_config(cfg, d)
        # 护栏：一处都没改 ⇒ 变体与源**逐字节相同**，T0 会得到"所有深度行为一致"的假阴性。
        if not notes:
            sys.exit(
                f"d{d}: 没有任何字段被改动 —— config 键名与预期不符。真实 dflash/dflash2 的层数在 "
                "transformer_layer_config.num_hidden_layers。**不要**拿这种“变体”去跑 T0。"
            )
        if dst.exists():
            shutil.rmtree(dst)
        if symlink_weights:
            # 权重只存一份，各档用软链引用（省 13+ GB；vLLM 按路径读文件，软链可用）
            dst.mkdir(parents=True)
            for f in sorted(src.iterdir()):
                if f.name == "config.json":
                    continue
                (dst / f.name).symlink_to(f.resolve())
            notes.append("权重=软链（共享源文件）")
        else:
            shutil.copytree(src, dst)
        (dst / "config.json").write_text(
            json.dumps(new_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if prune:
            for st in sorted(dst.glob("*.safetensors")):
                prune_safetensors(st, d, notes)
        print(f"  d{d}: " + "; ".join(notes))


def selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "src", Path(td) / "out"
        src.mkdir(parents=True)
        # 形态与真实 dflash2 配置一致：层数嵌在 transformer_layer_config 里
        (src / "config.json").write_text(json.dumps({
            "architectures": ["DFlash2DraftModel"],
            "block_size": 8,
            "aux_hidden_state_layer_ids": [1, 9, 17, 25, 33],
            "transformer_layer_config": {
                "num_hidden_layers": 5,
                "layer_types": ["sliding_attention"] * 5,
                "hidden_size": 2560,
                "vocab_size": 151936,
            },
        }), encoding="utf-8")
        (src / "model.safetensors").write_bytes(b"stub")
        make_variants(src, out, [1, 3, 5], prune=False)
        for d in (1, 3, 5):
            c = json.loads((out / f"d{d}" / "config.json").read_text(encoding="utf-8"))
            tlc = c["transformer_layer_config"]
            assert tlc["num_hidden_layers"] == d, tlc
            assert len(tlc["layer_types"]) == d, "layer_types 必须与层数同步截短"
            assert c["block_size"] == 8, "非层数字段必须保持不变"
            assert c["aux_hidden_state_layer_ids"] == [1, 9, 17, 25, 33], \
                "aux_hidden_state_layer_ids 指向**目标模型**层，绝不能改"
            assert "num_hidden_layers" not in c, "不得凭空增加顶层键"
        # 护栏自测：键名不匹配时必须报错而不是静默产出相同副本
        bad = Path(td) / "bad"
        bad.mkdir()
        (bad / "config.json").write_text(json.dumps({"unknown_key": 1}), encoding="utf-8")
        try:
            make_variants(bad, Path(td) / "badout", [1], prune=False)
        except SystemExit as e:
            assert "没有任何字段被改动" in str(e), e
        else:
            raise AssertionError("键名不匹配时本应报错")
        print("selftest ✔ 嵌套层数/layer_types 正确改写；非层数字段未动；护栏生效")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--depths", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--prune-weights", action="store_true")
    ap.add_argument("--symlink-weights", action="store_true",
                    help="各档只写 config，权重用软链共享（省磁盘；与 --prune-weights 互斥）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.src and a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        make_variants(Path(a.src), Path(a.out), a.depths, a.prune_weights, a.symlink_weights)
    else:
        ap.error("需要 --src/--out，或 --selftest")
