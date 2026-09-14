#!/usr/bin/env python3
"""p30 stage-2：**真实 TP 依赖链**下，PCIe(host-staged) 的集合通信延迟能否被隐藏。

stage-1 失败的原因（如实记录）：我把 dataflow 建错了。真实 TP 的链是
    x -> gemm_i -> all_reduce_i -> gemm_{i+1} -> ...
即 comm_i 依赖 gemm_i，而 gemm_{i+1} 依赖 comm_i —— **严格串行链**。
stage-1 的 dep/indep 两臂里 comm_{i+1} 不依赖 gemm_i，于是两臂都变成了同一条流水线
（实测 4.627 vs 4.606，差 0.5%，而格内极差 8-44%）⇒ **阴性对照失败 ⇒ 该探针无效**，不是发现。

本探针把臂建在真实依赖上，并用**配对比值**（每 rep 内除以 no_comm 基线）抵消共享机漂移。

臂：
  no_comm        36 x [h=mm(x,W1); y=mm(h,W2); x=y]                      基线（纯计算链）
  chain          36 x [h=mm(x,W1); y=mm(h,W2); all_reduce(y); x=y]       真实 TP 链，串行
  comm_only      36 x all_reduce(5KB)                                    背靠背排队（无依赖）
  comm_only_sync 36 x [all_reduce(5KB); sync]                            **无处可藏时**的暴露延迟
  chunkC         36 x [h=mm(x,W1); for c: y_c=mm(h,W2c); all_reduce(y_c)@s_comm; wait; x=y]
                 —— 分块：块 c 的 gemm 可与块 c-1 的 comm 重叠
  chunkC_dep     同上，但每块都先等 comm 再做下一块 —— **阴性对照**，必须 ≈ chain

判据（开跑前冻结）：
  · 若 comm_only_sync ≈ 36*0.055=1.98ms 且 comm_only ≈ 0.94ms ⇒ 排队能摊销、串行链不能（延迟结构性暴露）
  · chunkC_dep ≈ chain（阴性对照成立）才采信 chunkC 的任何增益
  · 若所有 chunkC ≥ chain ⇒ **本机无可利用重叠** ⇒ 该轴按"结构性延迟税"结题，不做"能省"的主张
"""
import argparse, json, os, statistics, time

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--hid", type=int, default=2560)
    p.add_argument("--inter", type=int, default=21504, help="MLP 中间维；2560*21504*2=105MiB")
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--outdir", default="/home/user/ccfa_logs/p30")
    return p

def run_rank(rank, world, args, _):
    import torch, torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
    dev = torch.device("cuda", rank)
    s_comm = torch.cuda.Stream()

    M, H, I, L = args.m, args.hid, args.inter, args.layers
    x0 = torch.randn(M, H, dtype=torch.bfloat16, device=dev) * 0.02
    W1 = torch.randn(H, I, dtype=torch.bfloat16, device=dev) * 0.02
    W2 = torch.randn(I, H, dtype=torch.bfloat16, device=dev) * 0.02
    y = torch.zeros(M, H, dtype=torch.bfloat16, device=dev)
    wbytes = H * I * 2 * 2

    def arm_no_comm():
        x = x0
        for _ in range(L):
            h = torch.mm(x, W1); x = torch.mm(h, W2)
        return x
    def arm_chain():
        x = x0
        for _ in range(L):
            h = torch.mm(x, W1); y.copy_(torch.mm(h, W2)); dist.all_reduce(y); x = y
        return x
    def arm_comm_only():
        for _ in range(L):
            dist.all_reduce(y)
    def arm_comm_only_sync():
        for _ in range(L):
            dist.all_reduce(y); torch.cuda.synchronize()
    def make_chunk(C, dependent):
        W2c = list(torch.chunk(W2, C, dim=0))     # 沿中间维切，输出仍是 (M,H) 的分块累加
        yc = [torch.zeros(M, H, dtype=torch.bfloat16, device=dev) for _ in range(C)]
        def arm():
            x = x0
            for _ in range(L):
                h = torch.mm(x, W1)
                for c in range(C):
                    yc[c].copy_(torch.mm(h[:, c*(I//C):(c+1)*(I//C)], W2c[c]))
                    # **必须**显式让 s_comm 等到本块的 gemm 入队之后再发集合通信。
                    # 缺这一行 => all_reduce 可能读到未写完的 yc[c]：既算错，又给出**假阳性**的分块增益。
                    # 事件记录在"本块 gemm 已入队"这一时刻 => 块 c 的 comm 仍可与块 c+1 的 gemm 重叠，
                    # 所以这条 wait 不会把重叠吃掉（这正是要测的形态）。
                    s_comm.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s_comm):
                        dist.all_reduce(yc[c])
                    if dependent:
                        torch.cuda.current_stream().wait_stream(s_comm)
                torch.cuda.current_stream().wait_stream(s_comm)
                x = torch.stack(yc, 0).sum(0, dtype=torch.bfloat16)
            return x
        return arm

    # ---- 正确性自检：分块实现必须与 no_comm 数值一致，否则测的不是同一件事
    def _correctness():
        r_nc = arm_no_comm(); torch.cuda.synchronize()
        arms_probe = []
        for C in args.chunks:
            r_c = make_chunk(C, False)(); torch.cuda.synchronize()
            d_ = ((r_c.float()-r_nc.float()).norm()/r_nc.float().norm()).item()
            arms_probe.append((C, d_))
        return r_nc, arms_probe

    arms = {"no_comm": arm_no_comm, "chain": arm_chain,
            "comm_only": arm_comm_only, "comm_only_sync": arm_comm_only_sync}
    for C in args.chunks:
        arms[f"chunk{C}"] = make_chunk(C, False)
        arms[f"chunk{C}_dep"] = make_chunk(C, True)

    times = {k: [] for k in arms}
    for i in range(args.warmup + args.reps):
        for name, fn in arms.items():
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            if i >= args.warmup:
                times[name].append(dt)

    base = sorted(times["no_comm"])                      # 每 rep 的配对比值
    rep_base = times["no_comm"]
    out = {"rank": rank, "M": M, "layer_weight_bytes": wbytes, "layers": L, "arms": {}}
    for k, v in times.items():
        sv = sorted(v)
        ratios = sorted(a / b for a, b in zip(v, rep_base) if b > 0)
        out["arms"][k] = {
            "median_ms": statistics.median(sv), "p10_ms": sv[int(0.1*(len(sv)-1))],
            "ratio_median": statistics.median(ratios),
            "ratio_p10": ratios[int(0.1*(len(ratios)-1))],
            "ratio_p90": ratios[int(0.9*(len(ratios)-1))],
            "spread_pct": (sv[int(0.9*(len(sv)-1))]-sv[int(0.1*(len(sv)-1))])/sv[int(0.1*(len(sv)-1))]*100,
            "n": len(sv)}
    try:
        r_nc, cprof = _correctness()
        out["chunk_rel_err_vs_no_comm"] = {str(C): d for C, d in cprof}
    except Exception as e:
        out["chunk_rel_err_vs_no_comm"] = f"err:{e}"
    dist.barrier(); dist.destroy_process_group()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(args.outdir, f"chain_rank{rank}.json"), "w"), indent=2)

def main():
    args = build_parser().parse_args(); os.makedirs(args.outdir, exist_ok=True)
    import torch.multiprocessing as mp
    mp.spawn(run_rank, args=(2, args, None), nprocs=2, join=True)
    d = json.load(open(os.path.join(args.outdir, "chain_rank0.json")))
    print(f"== M={d['M']} 每层权重 {d['layer_weight_bytes']/2**20:.0f} MiB, {d['layers']} 层, 每 rep = 一个 token 的量 ==")
    print(f"{'arm':<14} {'中位(ms)':>9} {'p10(ms)':>9} {'比值中位':>9} {'比值p10':>8} {'比值p90':>8} {'极差%':>7}")
    order = ["no_comm", "chain", "comm_only", "comm_only_sync"] + \
            [f"{p}{c}" for c in args.chunks for p in ("chunk", "chunk") if False] + \
            [k for k in d["arms"] if k.startswith("chunk")]
    for k in order:
        if k not in d["arms"]: continue
        v = d["arms"][k]
        print(f"{k:<14} {v['median_ms']:9.3f} {v['p10_ms']:9.3f} {v['ratio_median']:9.3f} "
              f"{v['ratio_p10']:8.3f} {v['ratio_p90']:8.3f} {v['spread_pct']:7.1f}")
    print("\n== 判读（比值 = 相对 no_comm 基线，配对）==")
    ch, co, cos_ = d["arms"]["chain"], d["arms"]["comm_only"], d["arms"]["comm_only_sync"]
    print(f"  chain/no_comm        = {ch['ratio_median']:.3f}  (串行链把 comm 完全暴露 -> 应 >1)")
    print(f"  comm_only/no_comm    = {co['ratio_median']:.3f}  (背靠背排队 -> 摊销)")
    print(f"  comm_only_sync/no_comm = {cos_['ratio_median']:.3f}  (无处可藏 -> 应为 max)")
    for C in args.chunks:
        c, cd = d["arms"][f"chunk{C}"], d["arms"][f"chunk{C}_dep"]
        gain = 1 - c["ratio_median"]/cd["ratio_median"] if cd["ratio_median"] else float("nan")
        print(f"  chunk{C}: 比值 {c['ratio_median']:.3f} vs 阴性对照 {cd['ratio_median']:.3f} "
              f"=> 相对增益 {gain*100:+.1f}%   [阴性对照应 ≈ chain={ch['ratio_median']:.3f}]")

if __name__ == "__main__":
    main()
