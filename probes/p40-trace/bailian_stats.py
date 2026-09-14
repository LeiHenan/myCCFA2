#!/usr/bin/env python3
"""Bailian（阿里百炼）生产轨迹分析：真实到达率、长度分布、会话结构、**块级真实前缀复用**。

为什么用真实轨迹：本会话 14 条自建机制全部因"在通用负载上已有人做过"而被否，
没有一条是因为"在真实负载上没用"。真实轨迹是把任何一条被杀机制变成**负载特异系统论文**的分母。

hash_ids 语义（先验证再用）：数据集文档称 "Salted SipHash block IDs for KV cache simulation (16 tokens per block)"。
若为**逐块内容哈希** ⇒ 共享前缀表现为**共享的前导 id** ⇒ 用"最长公共前导"即真实可复用块数。
本脚本先做该语义检验（统计 id 的重复率与请求间前导共享率），再据此算复用。
"""
import argparse, json, statistics as st, sys
from collections import defaultdict, deque

def load(path, limit=None):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            rows.append(r)
    rows.sort(key=lambda r: r.get("timestamp", 0.0))
    return rows

def analyze(rows, tag):
    n = len(rows)
    ts = [float(r.get("timestamp", 0.0)) for r in rows]
    inp = [int(r.get("input_length", 0)) for r in rows]
    out = [int(r.get("output_length", 0)) for r in rows]
    span = ts[-1] - ts[0] if n > 1 else 0.0
    # 会话结构
    turns = defaultdict(int)
    for r in rows:
        cid = r.get("parent_chat_id", -1)
        if cid not in (-1, None): turns[cid] += 1
    # hash_ids 语义检验 + 真实前缀复用
    seen = set(); reuse_blocks = []; reuse_frac = []; newblock_bytes = 0
    id_repeat = 0; id_total = 0
    ids_seen_count = defaultdict(int)
    for r in rows:
        h = r.get("hash_ids") or []
        if not isinstance(h, list): h = []
        id_total += len(h)
        lead = 0
        for x in h:                      # 前导连续命中 = 可复用的前缀块数
            if x in seen: lead += 1
            else: break
        reuse_blocks.append(lead)
        reuse_frac.append(lead / len(h) if h else 0.0)
        newblock_bytes += (len(h) - lead) * 16 * 147000   # 16 token/block × 147 KB/token
        for x in h:
            ids_seen_count[x] += 1
            seen.add(x)
    id_repeat = sum(1 for v in ids_seen_count.values() if v > 1)
    def pct(v, p): 
        s = sorted(v); return s[min(len(s)-1, int(p*(len(s)-1)))] if s else 0
    print("=== %s ===" % tag, flush=True)
    print("  请求数 %d | 时间跨度 %.1f s | 到达率 %.2f req/s" % (n, span, n/max(span,1e-9)), flush=True)
    print("  input  token: mean %.0f p50 %d p95 %d max %d | 合计 %.3f M"
          % (st.mean(inp), pct(inp,.5), pct(inp,.95), max(inp), sum(inp)/1e6), flush=True)
    print("  output token: mean %.0f p50 %d p95 %d max %d | 合计 %.3f M"
          % (st.mean(out), pct(out,.5), pct(out,.95), max(out), sum(out)/1e6), flush=True)
    print("  会话: 有父请求的 turns %d 个 | 多轮会话数 %d | 最大 turn %d"
          % (sum(turns.values()), len(turns), max(turns.values()) if turns else 0), flush=True)
    print("  hash_ids: 总块 %d | 去重后 %d (重复率 %.1f%%) | 平均每请求 %.1f 块"
          % (id_total, len(seen), 100*id_repeat/max(len(ids_seen_count),1), id_total/max(n,1)), flush=True)
    print("  真实前缀复用: 平均可复用块 %.1f (占 %.1f%%) | 复用率>0 的请求 %.1f%%"
          % (st.mean(reuse_blocks), 100*st.mean(reuse_frac),
             100*sum(1 for x in reuse_blocks if x>0)/max(n,1)), flush=True)
    print("  新块所需的 KV 字节合计 ≈ %.2f GB（按 147 KB/token、16 token/块）" % (newblock_bytes/1e9), flush=True)
    return dict(n=n, span=span, inp_mean=st.mean(inp), out_mean=st.mean(out),
                reuse_frac=st.mean(reuse_frac), reuse_gt0=sum(1 for x in reuse_blocks if x>0)/max(n,1),
                turns=len(turns))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/Users/leihenan/Desktop/myProject/data/traces")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    import os
    for f in sorted(os.listdir(a.dir)):
        if not f.endswith(".jsonl"): continue
        rows = load(os.path.join(a.dir, f), a.limit)
        if not rows: continue
        analyze(rows, f)
        print(flush=True)

if __name__ == "__main__":
    main()
