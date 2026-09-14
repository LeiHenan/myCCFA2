#!/usr/bin/env python3
"""p30 stage-1：PCIe-only / host-staged 机器上，decode 形状的 TP all-reduce 能否落进计算阴影？

对照设计（5 臂，同一进程内交错轮转以抵消漂移）：
  1 comm_only     36 x all_reduce                      —— 纯通信
  2 comp_only     36 x GEMM                            —— 纯计算（同时给出 Python/launch 开销地板）
  3 serial        all_reduce -> GEMM（同默认流，天然依赖）—— 完全不重叠的参照
  4 dep_streams   all_reduce 在 s_comm；s_comp.wait_stream(s_comm) 后 GEMM
                  —— **阴性对照**：依赖关系不变，只换成两条流。它必须 ≈ serial，
                     否则说明"重叠"是测量假象而非真并发。
  5 indep_streams all_reduce 在 s_comm  ∥  GEMM 在 s_comp，无依赖
                  —— 可达重叠的**上界**

判定：overlap_gain = 1 - indep/serial。
  若 indep ≈ serial  ⇒ 本机集合通信**无法**与计算重叠 ⇒ 候选轴当场杀（零成本）
  若 indep ≈ comp_only 且 dep ≈ serial ⇒ 重叠真实且可达上界；进入 stage-2（分块依赖版）

纪律：父进程绝不碰 CUDA（本项目踩过 forked subprocess 重初始化）；结果写每 rank JSON，不用 mp.Queue。
"""
import argparse, json, os, statistics, sys, time

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--hid", type=int, default=2560)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--n", type=int, default=21504, help="GEMM 的 N：2560*21504*2 = 110 MB ≈ Qwen3-4B 单层权重/2")
    p.add_argument("--ms", type=int, nargs="+", default=[1, 64])
    p.add_argument("--reps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--outdir", default="/home/user/ccfa_logs/p30")
    return p

def run_rank(rank, world, args, devnull):
    import torch, torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = torch.device("cuda", rank)

    # ---- 如实记录 NCCL 实况（本项目纪律：断言配置真的生效，而不是意图）
    facts = {
        "rank": rank,
        "torch": torch.__version__,
        "nccl": "".join(f"{c}" for c in torch.cuda.nccl.version()) if isinstance(torch.cuda.nccl.version(), tuple) else str(torch.cuda.nccl.version()),
        "dev_name": torch.cuda.get_device_name(rank),
        "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "<unset>"),
        "NCCL_ALGO": os.environ.get("NCCL_ALGO", "<unset>"),
        "NCCL_PROTO": os.environ.get("NCCL_PROTO", "<unset>"),
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", "<unset>"),
        "peer_access": None,
    }
    if world == 2:
        try:
            facts["peer_access"] = bool(torch.cuda.can_device_access_peer(0, 1))
        except Exception as e:
            facts["peer_access"] = f"err:{e}"

    s_comm = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()
    assert s_comm.cuda_stream != s_comp.cuda_stream, "两条流必须是不同流"

    results = {"facts": facts, "cells": []}

    for M in args.ms:
        msg_bytes = M * args.hid * 2
        t = torch.ones(M, args.hid, dtype=torch.bfloat16, device=dev)
        x = torch.ones(M, args.hid, dtype=torch.bfloat16, device=dev)
        W = torch.ones(args.hid, args.n, dtype=torch.bfloat16, device=dev) * 0.01

        def arm_comm():
            for _ in range(args.layers):
                dist.all_reduce(t)
        def arm_comp():
            for _ in range(args.layers):
                torch.mm(x, W)
        def arm_serial():
            for _ in range(args.layers):
                dist.all_reduce(t)
                torch.mm(x, W)
        def arm_dep():
            for _ in range(args.layers):
                with torch.cuda.stream(s_comm):
                    dist.all_reduce(t)
                s_comp.wait_stream(s_comm)
                with torch.cuda.stream(s_comp):
                    torch.mm(x, W)
        def arm_indep():
            for _ in range(args.layers):
                with torch.cuda.stream(s_comm):
                    dist.all_reduce(t)
                with torch.cuda.stream(s_comp):
                    torch.mm(x, W)

        arms = {"comm_only": arm_comm, "comp_only": arm_comp, "serial": arm_serial,
                "dep_streams": arm_dep, "indep_streams": arm_indep}
        times = {k: [] for k in arms}

        for i in range(args.warmup + args.reps):
            for name, fn in arms.items():          # 交错轮转，抵消机器漂移
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1e3
                if i >= args.warmup:
                    times[name].append(dt)

        cell = {"M": M, "msg_bytes": msg_bytes, "world": world, "layers": args.layers,
                "gemm": {"M": M, "K": args.hid, "N": args.n, "weight_bytes": args.hid * args.n * 2}}
        med = {}
        for k, v in times.items():
            v = sorted(v)
            med[k] = statistics.median(v)
            cell[k] = {"median_ms": statistics.median(v),
                       "p10_ms": v[int(0.1 * (len(v) - 1))],
                       "p90_ms": v[int(0.9 * (len(v) - 1))],
                       "spread_pct": (v[int(0.9 * (len(v) - 1))] - v[int(0.1 * (len(v) - 1))]) / v[int(0.1 * (len(v) - 1))] * 100}
        cell["overlap_gain_vs_serial_pct"] = (1 - med["indep_streams"] / med["serial"]) * 100
        cell["negative_control_dep_vs_serial_pct"] = (med["dep_streams"] / med["serial"] - 1) * 100
        cell["indep_vs_compute_only_pct"] = (med["indep_streams"] / med["comp_only"] - 1) * 100
        # 本机是否 host-staged：用 4 MB 大消息的有效带宽做旁证（与 p27 的 12.2-13.4 GB/s 对照）
        results["cells"].append(cell)

    dist.barrier()
    dist.destroy_process_group()
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f"rank{rank}.json"), "w") as f:
        json.dump(results, f, indent=2)

def main():
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    import torch.multiprocessing as mp
    world = 2
    mp.spawn(run_rank, args=(world, args, None), nprocs=world, join=True)

    cells = None
    for r in range(world):
        with open(os.path.join(args.outdir, f"rank{r}.json")) as f:
            d = json.load(f)
        if r == 0:
            print("== 实况 ==")
            for k, v in d["facts"].items():
                print(f"  {k}: {v}")
            cells = d["cells"]
    print("\n== 结果（36 层 = 一个 token 的 TP 通信/计算量）==")
    for c in cells:
        print(f"\n--- M={c['M']}  msg={c['msg_bytes']}B  GEMM {c['gemm']['M']}x{c['gemm']['K']}x{c['gemm']['N']} "
              f"(权重 {c['gemm']['weight_bytes']/2**20:.0f} MiB) ---")
        for k in ("comm_only", "comp_only", "serial", "dep_streams", "indep_streams"):
            v = c[k]
            print(f"  {k:<14} {v['median_ms']:8.3f} ms  [p10 {v['p10_ms']:7.3f} p90 {v['p90_ms']:7.3f}]  极差 {v['spread_pct']:5.1f}%")
        print(f"  => overlap_gain(vs serial) = {c['overlap_gain_vs_serial_pct']:+.1f}%")
        print(f"  => 阴性对照 dep_streams vs serial = {c['negative_control_dep_vs_serial_pct']:+.1f}%  (应 ≈0)")
        print(f"  => indep vs comp_only = {c['indep_vs_compute_only_pct']:+.1f}%  (≈0 = 重叠到上界)")

if __name__ == "__main__":
    main()
