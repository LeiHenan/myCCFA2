#!/usr/bin/env python3
"""会话结构与「轮间间隔 × 会话 KV 足迹」——决定会话感知驻留是否有空间的两个量（零卡）。

背景：CacheWise (2606.16824) 在**编码 agent**轨迹上得出「LRU 不顾工具调用返回 ⇒ 会话感知很重要」。
本脚本用**另外三类真实生产轨迹**（reasoning / 通用 text+search+image+file / 高并发短请求 API）
量同一件事，看该结论是否推广，以及需要多少容量才能按会话驻留。
"""
import json, os, statistics as st
from collections import defaultdict

D = "/Users/leihenan/Desktop/myProject/data/traces"
KV_PER_TOKEN = 147_000   # Qwen3-4B bf16, 字节/token

def main():
    for fn in sorted(os.listdir(D)):
        if not fn.endswith(".jsonl"): continue
        sess = defaultdict(list)
        with open(os.path.join(D, fn), encoding="utf-8") as f:
            for line in f:
                try: r = json.loads(line)
                except Exception: continue
                key = r.get("chat_id") if r.get("parent_chat_id", -1) == -1 else r.get("parent_chat_id")
                sess[key].append(r)
        multi = {k: v for k, v in sess.items() if len(v) > 1}
        gaps, foot, ncum = [], [], []
        for k, v in multi.items():
            v.sort(key=lambda r: r.get("timestamp", 0.0))
            cum = 0
            for i, r in enumerate(v):
                if i > 0:
                    gaps.append(float(r["timestamp"]) - float(v[i-1]["timestamp"]))
                cum += int(r.get("input_length", 0)) * KV_PER_TOKEN
                # 会话在第 i 轮结束后常驻的 KV ≈ 最后一轮 input+output
                last = (int(r.get("input_length", 0)) + int(r.get("output_length", 0))) * KV_PER_TOKEN
            foot.append(last); ncum.append(cum)
        if not multi:
            print("%-28s 无多轮会话" % fn); continue
        def p(v, q):
            s = sorted(v); return s[min(len(s)-1, int(q*(len(s)-1)))] if s else 0
        print("=== %s ===" % fn)
        print("  多轮会话 %d 个（占全部会话 %.1f%%）" % (len(multi), 100*len(multi)/len(sess)))
        print("  轮间间隔 s: p50 %.1f  p90 %.1f  p99 %.1f  max %.0f"
              % (p(gaps,.5), p(gaps,.9), p(gaps,.99), max(gaps)))
        print("  会话末轮 KV 足迹: p50 %.2f GB  p90 %.2f GB  max %.2f GB"
              % (p(foot,.5)/1e9, p(foot,.9)/1e9, max(foot)/1e9))
        print("  会话累计写入 KV: p50 %.2f GB  p90 %.2f GB  max %.2f GB"
              % (p(ncum,.5)/1e9, p(ncum,.9)/1e9, max(ncum)/1e9))
        # 若要按会话驻留：同时活跃会话的 KV 总量上限
        ev = []
        for r in (x for v in sess.values() for x in v):
            ev.append((float(r.get("timestamp", 0.0)), +1, (int(r.get("input_length",0))+int(r.get("output_length",0)))*KV_PER_TOKEN))
            ev.append((float(r.get("timestamp", 0.0))+max(1.0, float(r.get("output_length",0))/30.0), -1,
                       (int(r.get("input_length",0))+int(r.get("output_length",0)))*KV_PER_TOKEN))
        ev.sort(); live = 0; peak = 0
        for _, d, b in ev:
            live += d*b
            peak = max(peak, live)
        print("  同时活跃的 KV 峰值（粗估，按 30 tok/s 解码）≈ %.1f GB" % (peak/1e9))
        print(flush=True)

if __name__ == "__main__":
    main()
