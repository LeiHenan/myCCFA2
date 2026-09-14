#!/usr/bin/env python3
"""缺口可不可达？在线策略 vs Belady（零卡，用真实轨迹的精确块引用串）。

Belady 是离线最优、不可实现。真正的问题：**在线策略能拿到缺口的多少？**
特别测 vLLM 0.29 自己发货的 **ARC**（CachePolicyFactory: LRUCachePolicy / ARCCachePolicy）。
判据（事前冻结）：若在 1–10% 缓存区间，**最佳在线策略**相对 LRU 的增益 < 缺口的 1/3，
则缺口"真实但不可达" ⇒ 这条轴不足以为题。
"""
import json, os, sys
from collections import OrderedDict, defaultdict
import heapq

D = "/Users/leihenan/Desktop/myProject/data/traces"

def refs_of(path, max_req):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_req: break
            try: rows.append(json.loads(line))
            except Exception: pass
    rows.sort(key=lambda r: r.get("timestamp", 0.0))
    refs = []
    for r in rows:
        h = r.get("hash_ids") or []
        if isinstance(h, list): refs.extend(h)
    return refs

def lru(refs, cap):
    c = OrderedDict(); hit = 0
    for x in refs:
        if x in c: hit += 1; c.move_to_end(x)
        else:
            c[x] = 1
            if len(c) > cap: c.popitem(last=False)
    return hit

def lfu(refs, cap):
    """LFU + LRU tie-break（近似：按引用计数淘汰最小）"""
    cnt = defaultdict(int); c = OrderedDict(); hit = 0
    import heapq as hq
    heap = []
    for x in refs:
        cnt[x] += 1
        if x in c:
            hit += 1; c.move_to_end(x)
            hq.heappush(heap, (cnt[x], x))
        else:
            c[x] = 1; hq.heappush(heap, (cnt[x], x))
            while len(c) > cap:
                k, y = hq.heappop(heap)
                if y in c and cnt[y] == k:
                    del c[y]
    return hit

def arc(refs, cap):
    """ARC：T1(recency)+T2(frequency)，B1/B2 ghost，自适应目标 p"""
    t1, t2 = OrderedDict(), OrderedDict()
    b1, b2 = OrderedDict(), OrderedDict()
    p = 0; hit = 0
    def repl(x):
        nonlocal p
        if t1 and ((x in b2 and len(t1) == p) or len(t1) > p):
            y, _ = t1.popitem(last=False); b1[y] = 1
        else:
            y, _ = t2.popitem(last=False); b2[y] = 1
    for x in refs:
        if x in t1:
            hit += 1; del t1[x]; t2[x] = 1
        elif x in t2:
            hit += 1; t2.move_to_end(x)
        else:
            if x in b1:
                p = min(cap, p + max(1, len(b2)//max(len(b1), 1)))
                repl(x); b1.pop(x, None); t2[x] = 1
            elif x in b2:
                p = max(0, p - max(1, len(b1)//max(len(b2), 1)))
                repl(x); b2.pop(x, None); t2[x] = 1
            else:
                if len(t1) + len(b1) == cap:
                    if len(t1) < cap:
                        b1.popitem(last=False); repl(x)
                    else:
                        t1.popitem(last=False)
                elif len(t1) + len(b1) < cap:
                    tot = len(t1)+len(t2)+len(b1)+len(b2)
                    if tot >= cap:
                        if tot == 2*cap: b2.popitem(last=False)
                        repl(x)
                t1[x] = 1
            while len(t1)+len(t2) > cap: repl(x)
    return hit

def belady(refs, cap):
    n = len(refs); nxt = [n]*n; last = {}
    for i in range(n-1, -1, -1):
        x = refs[i]; nxt[i] = last.get(x, n); last[x] = i
    cache = set(); cur = {}; heap = []; hit = 0
    for i in range(n):
        x = refs[i]
        if x in cache: hit += 1
        else:
            if len(cache) >= cap:
                while heap:
                    neg, y = heapq.heappop(heap)
                    if y in cache and cur.get(y) == -neg: cache.discard(y); break
            cache.add(x)
        cur[x] = nxt[i]; heapq.heappush(heap, (-nxt[i], x))
    return hit

def main():
    maxreq = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    for fn in sorted(os.listdir(D)):
        if not fn.endswith(".jsonl"): continue
        refs = refs_of(os.path.join(D, fn), maxreq)
        if not refs: continue
        distinct = len(set(refs)); N = len(refs)
        print("=== %s === 引用 %d 去重 %d" % (fn, N, distinct), flush=True)
        print("  %-6s %-9s %-9s %-9s %-9s %-12s" % ("缓存%","LRU","LFU","ARC","Belady","ARC补缺口"), flush=True)
        for frac in (0.01, 0.05, 0.10, 0.25):
            cap = max(1, int(distinct*frac))
            h = {k: f(refs, cap)/N*100 for k, f in
                 (("lru",lru),("lfu",lfu),("arc",arc),("bel",belady))}
            gap = h["bel"] - h["lru"]
            fill = (h["arc"]-h["lru"])/gap*100 if gap > 1e-9 else float("nan")
            print("  %-6s %-9.2f %-9.2f %-9.2f %-9.2f %-12s" % (
                "%.0f%%"%(frac*100), h["lru"], h["lfu"], h["arc"], h["bel"],
                ("%.0f%%" % fill) if gap > 1e-9 else "n/a"), flush=True)
        print(flush=True)

if __name__ == "__main__":
    main()
