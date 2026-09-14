#!/usr/bin/env python3
"""用真实轨迹的**精确块引用串**（hash_ids）算 LRU vs Belady 最优命中率（零卡）。

要检验的已发表结论：*"Recency Is Near-Optimal for LLM Prefix Caches"* —— LRU 达 Belady 的 99.4–100%，
故任何预测器头寸 ≤0.5 点。**但那是他们的负载。**

本机真实生产轨迹的特征相反：轮间间隔 p50 37–141 s、p99 达 4000 s，同时活跃 KV 是缓存的 2–9 倍
⇒ 复用是**区间型**而非**近因型** ⇒ LRU 的最优性可能不成立。

判据（开跑前冻结）：若在任一真实轨迹上 `Belady命中率 − LRU命中率` 的**绝对百分点差 >2**，
则"LRU 近最优"在该负载上不成立，该轴**有空间**；否则该轴**杀**。
"""
import json, os, sys
from collections import OrderedDict, defaultdict
import heapq

D = "/Users/leihenan/Desktop/myProject/data/traces"

def refs_of(path, max_req):
    refs = []
    with open(path, encoding="utf-8") as f:
        rows = []
        for i, line in enumerate(f):
            if i >= max_req: break
            try: rows.append(json.loads(line))
            except Exception: pass
    rows.sort(key=lambda r: r.get("timestamp", 0.0))
    for r in rows:
        h = r.get("hash_ids") or []
        if isinstance(h, list): refs.extend(h)
    return refs

def lru_hits(refs, cap):
    c = OrderedDict(); hit = 0
    for x in refs:
        if x in c:
            hit += 1; c.move_to_end(x)
        else:
            c[x] = 1
            if len(c) > cap: c.popitem(last=False)
    return hit

def belady_hits(refs, cap):
    """Belady 最优（MIN）。必须满足 belady >= lru 对任意容量。
    关键：**每次引用都入堆**（含"以后再也不引用"的哨兵 n），否则死块永不淘汰、缓存被占满。"""
    n = len(refs)
    nxt = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        x = refs[i]
        nxt[i] = last.get(x, n)
        last[x] = i
    cache = set(); cur = {}; heap = []; hit = 0
    for i in range(n):
        x = refs[i]
        if x in cache:
            hit += 1
        else:
            if len(cache) >= cap:
                while heap:
                    neg, y = heapq.heappop(heap)
                    if y in cache and cur.get(y) == -neg:
                        cache.discard(y); break
            cache.add(x)
        cur[x] = nxt[i]
        heapq.heappush(heap, (-nxt[i], x))
    return hit

def main():
    maxreq = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    for fn in sorted(os.listdir(D)):
        if not fn.endswith(".jsonl"): continue
        refs = refs_of(os.path.join(D, fn), maxreq)
        if not refs: continue
        distinct = len(set(refs))
        print("=== %s === 引用数 %d | 去重块 %d" % (fn, len(refs), distinct), flush=True)
        print("  %-10s %-14s %-14s %-12s" % ("缓存占比", "LRU 命中率", "Belady 命中率", "绝对差(点)"), flush=True)
        for frac in (0.01, 0.05, 0.10, 0.25, 0.50, 1.00):
            cap = max(1, int(distinct * frac))
            h1 = lru_hits(refs, cap) / len(refs) * 100
            h2 = belady_hits(refs, cap) / len(refs) * 100
            flag = "  <== >2 点" if (h2 - h1) > 2.0 else ""
            print("  %-10s %-14.2f %-14.2f %-12.2f%s" % ("%.0f%%" % (frac*100), h1, h2, h2-h1, flag), flush=True)
        print(flush=True)

if __name__ == "__main__":
    main()
