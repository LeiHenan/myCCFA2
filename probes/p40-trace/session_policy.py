#!/usr/bin/env python3
"""会话感知驱逐能否补上 Belady 缺口？（零卡，真实轨迹精确块引用串）

动机：上一轮我用 batch=1 的 decode 速率得出"会话重算只占请求成本 2.2%" ⇒ 判死。
**那是错的**：该负载显存超额认购 2–9×，真实并发 n≈64（0.295 ms/token）下，
coder 请求成本 ≈0.42 s 而会话 KV 中位重算 0.124 s ⇒ **29%**。判决需重做。

本脚本测一个**只用引擎已有信息**（session_id + 观测到的历史轮间间隔）的策略：
  session-LRU —— 驱逐时优先丢"其所属会话最近活动最久远"的块（块级 LRU 的会话化版本）。
对照：LRU / LFU / ARC / Belady。
判据（事前冻结）：若 session-LRU 补上的缺口 <1/3 ⇒ 不可达 ⇒ 杀。
"""
import json, os, sys
from collections import OrderedDict, defaultdict
import heapq

D = "/Users/leihenan/Desktop/myProject/data/traces"

def stream(path, max_req):
    """返回 (refs, sess_of_ref)：refs 是按 timestamp 排序后的块引用；sess_of_ref 是每引用的会话键。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_req: break
            try: rows.append(json.loads(line))
            except Exception: pass
    rows.sort(key=lambda r: r.get("timestamp", 0.0))
    refs, sess = [], []
    for r in rows:
        k = r.get("chat_id") if r.get("parent_chat_id", -1) == -1 else r.get("parent_chat_id")
        h = r.get("hash_ids") or []
        if not isinstance(h, list): continue
        for x in h:
            refs.append(x); sess.append(k)
    return refs, sess

def hit_lru(refs, cap):
    c = OrderedDict(); hit = 0
    for x in refs:
        if x in c: hit += 1; c.move_to_end(x)
        else:
            c[x] = 1
            if len(c) > cap: c.popitem(last=False)
    return hit

def hit_session_lru(refs, sess, cap):
    """会话化 LRU（按**会话**建堆 + 惰性失效 + 容量断言）。
    优先级 = 该块所属会话的最近引用序号；驱逐时从『最近活动最久远』的会话里丢一个块。"""
    from collections import defaultdict
    last_sess = {}
    blocks = {}                      # block -> session
    in_sess = defaultdict(set)       # session -> set(block)
    heap = []                        # (session_last_ref, session)
    hit = 0
    for i, x in enumerate(refs):
        s = sess[i]; last_sess[s] = i
        if x in blocks:
            hit += 1
        else:
            if len(blocks) >= cap:
                victim = None
                while heap:
                    key, ss = heapq.heappop(heap)
                    if last_sess.get(ss, -1) == key and in_sess.get(ss):
                        victim = ss; break
                if victim is None:
                    raise AssertionError("驱逐失败：缓存将无界增长（cap=%d, size=%d）" % (cap, len(blocks)))
                y = next(iter(in_sess[victim]))
                in_sess[victim].discard(y); del blocks[y]
                heapq.heappush(heap, (key, victim))   # 放回：该会话可能还有别的块在缓存里
            blocks[x] = s; in_sess[s].add(x)
        heapq.heappush(heap, (i, s))
        assert len(blocks) <= cap, "容量被突破: %d > %d" % (len(blocks), cap)
    return hit

def hit_belady(refs, cap):
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
    maxreq = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    for fn in sorted(os.listdir(D)):
        if not fn.endswith(".jsonl"): continue
        refs, sess = stream(os.path.join(D, fn), maxreq)
        if not refs: continue
        distinct = len(set(refs)); N = len(refs)
        print("=== %s === 引用 %d 去重 %d 会话 %d" % (fn, N, distinct, len(set(sess))), flush=True)
        print("  %-6s %-9s %-12s %-9s %-10s" % ("缓存%","LRU","session-LRU","Belady","补缺口"), flush=True)
        for frac in (0.01, 0.05, 0.10):
            cap = max(1, int(distinct*frac))
            a = hit_lru(refs, cap)/N*100
            b = hit_session_lru(refs, sess, cap)/N*100
            c = hit_belady(refs, cap)/N*100
            gap = c - a
            fill = (b-a)/gap*100 if gap > 1e-9 else float("nan")
            print("  %-6s %-9.2f %-12.2f %-9.2f %-10s" % ("%.0f%%"%(frac*100), a, b, c,
                  ("%.0f%% %s" % (fill, "PASS" if fill>=33 else "fail")) if gap>1e-9 else "n/a"), flush=True)
        print(flush=True)

if __name__ == "__main__":
    main()
