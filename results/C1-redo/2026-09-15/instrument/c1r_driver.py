#!/usr/bin/env python
"""C1 redo -- grid driver.  Runs c1r_experiment.py once per (target, drafter, K, PC)
cell, in a recorded order, alternating the PC on/off order between blocks so that a
monotonic drift cannot land systematically on one arm.

Emits one JSON per launch into --out-dir (written by c1r_experiment.py) plus a
driver manifest recording the exact launch order and each launch's outcome.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

# Only C / C.utf8 / POSIX exist in this container, but the default environment
# exports LC_ALL=en_US.UTF-8, under which `import readline` segfaults.  Pin it
# for every child (see c1r_experiment.Server.start).
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

ROOT = "/root/autodl-tmp"
PY = "/root/miniconda3/bin/python"

MODELS = {
    "Qwen3-4B": dict(
        path=f"{ROOT}/models/Qwen3-4B", tokenizer=f"{ROOT}/models/Qwen3-4B",
        lmo=0,
        drafters={"eagle3": f"{ROOT}/models/eagle3-qwen3-4b",
                  "dflash": f"{ROOT}/models/dflash-qwen3-4b"}),
    "Qwen3.5-4B": dict(
        path=f"{ROOT}/models/Qwen3.5-4B", tokenizer=f"{ROOT}/models/Qwen3.5-4B",
        lmo=1,
        drafters={"eagle3": f"{ROOT}/models/eagle3-qwen3.5-4b",
                  "dflash": f"{ROOT}/models/dflash-qwen3.5-4b"}),
    "Falcon-H1-3B-Instruct": dict(
        path=f"{ROOT}/models/Falcon-H1-3B-Instruct",
        tokenizer=f"{ROOT}/models/Falcon-H1-3B-Instruct",
        lmo=0, drafters={}),
}

DRAFTERS = ["none", "ngram", "eagle3", "dflash"]


def build_grid(name):
    """-> list of (target, drafter, k, pc)"""
    cells = []
    if name == "core-dense":
        for d in DRAFTERS:
            cells.append(("Qwen3-4B", d, 3))
    elif name == "core-hybrid":
        for d in DRAFTERS:
            cells.append(("Qwen3.5-4B", d, 3))
    elif name == "core":
        for d in DRAFTERS:
            cells.append(("Qwen3-4B", d, 3))
        for d in DRAFTERS:
            cells.append(("Qwen3.5-4B", d, 3))
    elif name == "falcon":
        for d in ["none", "ngram"]:
            cells.append(("Falcon-H1-3B-Instruct", d, 3))
    elif name == "k-sens":
        for k in (1, 3, 5, 7):
            cells.append(("Qwen3-4B", "ngram", k))
    else:
        raise SystemExit(f"unknown grid {name}")

    out = []
    for i, (t, d, k) in enumerate(cells):
        # alternate which PC arm runs first, so drift cancels across blocks
        pcs = (1, 0) if i % 2 == 0 else (0, 1)
        for pc in pcs:
            out.append((t, d, k, pc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out-dir", default=f"{ROOT}/results/raw")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-per-set", type=int, default=6)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--warm-repeats", type=int, default=2)
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--max-launches", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8611)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    cells = build_grid(a.grid)
    if a.max_launches:
        cells = cells[:a.max_launches]

    manifest = {"grid": a.grid, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cmd_args": vars(a), "launches": []}
    mpath = os.path.join(a.out_dir, f"manifest_{a.grid}.json")

    for i, (target, drafter, k, pc) in enumerate(cells):
        spec = MODELS[target]
        # NOTE: run_id 必须带 grid 前缀。原实现只用 (target,drafter,k,pc)，导致
        # k-sens 网格重跑了 core-dense 已有的 (Qwen3-4B,ngram,K=3) 两格并**覆盖**同名文件，
        # 使 analysis_core-dense.json 里那一行无法从归档 raw 复算（见 CORRECTION.md）。
        run_id = f"{a.grid}_{target}_{drafter}_k{k}_pc{pc}"
        cmd = [PY, f"{ROOT}/probes/c1r_experiment.py",
               "--run-id", run_id, "--target", target,
               "--target-path", spec["path"], "--tokenizer", spec["tokenizer"],
               "--drafter", drafter, "--k", str(k), "--prefix-caching", str(pc),
               "--seeds", *[str(s) for s in a.seeds],
               "--n-per-set", str(a.n_per_set), "--gen", str(a.gen),
               "--warm-repeats", str(a.warm_repeats),
               "--concurrencies", *[str(c) for c in a.concurrencies],
               "--gpu-util", str(a.gpu_util), "--max-len", str(a.max_len),
               "--port", str(a.port), "--language-model-only", str(spec["lmo"]),
               "--out-dir", a.out_dir]
        if drafter in spec["drafters"]:
            cmd += [f"--{drafter}", spec["drafters"][drafter]]
        elif drafter not in ("none", "ngram"):
            print(f"[{i+1}/{len(cells)}] SKIP {run_id}: no checkpoint for {drafter} on {target}",
                  flush=True)
            manifest["launches"].append({"run_id": run_id, "skipped": "no checkpoint"})
            continue

        print(f"\n[{i+1}/{len(cells)}] LAUNCH {run_id}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        dt = time.time() - t0
        jpath = os.path.join(a.out_dir, f"{run_id}.json")
        status = "missing"
        if os.path.exists(jpath):
            try:
                status = json.load(open(jpath)).get("engine_status", "?")
            except Exception:
                status = "unreadable"
        manifest["launches"].append({"run_id": run_id, "rc": rc, "seconds": round(dt, 1),
                                     "engine_status": status, "json": jpath})
        json.dump(manifest, open(mpath, "w"), indent=2)
        print(f"[{i+1}/{len(cells)}] done {run_id} rc={rc} {dt:.0f}s engine={status}",
              flush=True)

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"\nDRIVER DONE -> {mpath}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
