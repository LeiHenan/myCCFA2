#!/usr/bin/env python3
"""真实负载轨迹的载入 / 规范化 / 统计（零卡）。

**为什么先建这个**：本会话 14 条自建机制全部因"在通用负载上已有人做过"而被否，
没有一条是因为"在真实负载上没用"。⇒ 真实轨迹是把任何一条被杀机制变成
**负载特异的系统论文**的分母。工具先就位，轨迹一到即可开跑。

支持的输入（字段名宽松匹配，一行一条 JSON 或 CSV）：
  arrival  : arrival_ms | arrival | t | ts | timestamp
  prompt   : prompt_tokens | prompt_len | input_len | in_tokens | context_len
  output   : output_tokens | output_len | out_tokens | completion_len | max_tokens
  session  : session_id | session | conv_id | conversation_id | tenant   （可选，决定前缀复用结构）
缺失 arrival 时按顺序以 0 间隔到达；缺失 output 时按 prompt 的固定比例估（并在报告里标 EST）。

用法：
  trace_load.py --trace T.jsonl --dry-run          # 只解析 + 出统计（零卡）
  trace_load.py --selftest                          # 内置合成轨迹自测
"""
import argparse, json, os, statistics as st, sys

ALIASES = {
    "arrival": ["arrival_ms", "arrival", "t", "ts", "timestamp", "time_ms"],
    "prompt":  ["prompt_tokens", "prompt_len", "input_len", "in_tokens", "context_len", "prompt_length"],
    "output":  ["output_tokens", "output_len", "out_tokens", "completion_len", "max_tokens", "completion_tokens"],
    "session": ["session_id", "session", "conv_id", "conversation_id", "tenant", "user_id"],
}
NORM = {"arrival": "arrival_ms", "prompt": "prompt_tokens", "output": "output_tokens", "session": "session_id"}

def pick(rec):
    out, used = {}, {}
    low = {str(k).lower().strip(): v for k, v in rec.items()}
    for key, names in ALIASES.items():
        for n in names:
            if n in low and low[n] not in (None, ""):
                out[NORM[key]] = low[n]; used[key] = n; break
    return out, used

def load(path):
    rows, est_out, missing = [], 0, set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                parts = line.split(",")
                rec = {"arrival_ms": parts[0], "prompt_tokens": parts[1] if len(parts) > 1 else None,
                       "output_tokens": parts[2] if len(parts) > 2 else None}
            r, used = pick(rec)
            if "prompt_tokens" not in r:
                missing.add("prompt_tokens"); continue
            r["prompt_tokens"] = int(float(r["prompt_tokens"]))
            if "output_tokens" not in r:
                r["output_tokens"] = max(1, int(0.25 * r["prompt_tokens"])); est_out += 1
            else:
                r["output_tokens"] = int(float(r["output_tokens"]))
            r["arrival_ms"] = float(r.get("arrival_ms", 0.0))
            rows.append(r)
    rows.sort(key=lambda x: x["arrival_ms"])
    return rows, est_out, missing

def stats(rows, est_out, missing):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    span = rows[-1]["arrival_ms"] - rows[0]["arrival_ms"]
    p = [r["prompt_tokens"] for r in rows]; o = [r["output_tokens"] for r in rows]
    # 会话内前缀复用潜力：同一 session 的相邻请求共享前缀的下界估计（取 min(prompt) 之和）
    sess = {}
    for r in rows:
        sess.setdefault(r.get("session_id"), []).append(r["prompt_tokens"])
    share = sum(min(v) * (len(v) - 1) for v in sess.values() if len(v) > 1)
    tot_p = sum(p)
    return {
        "n_requests": n, "span_s": round(span / 1000.0, 2),
        "arrival_rate_rps": round(n / max(span / 1000.0, 1e-9), 3),
        "prompt_tokens": {"mean": round(st.mean(p), 1), "p50": st.median(p), "p95": sorted(p)[int(0.95*(n-1))], "max": max(p)},
        "output_tokens": {"mean": round(st.mean(o), 1), "p50": st.median(o), "p95": sorted(o)[int(0.95*(n-1))], "max": max(o)},
        "total_prompt_tokens": tot_p,
        "total_output_tokens": sum(o),
        "n_sessions": len([k for k in sess if k is not None]),
        "prefix_reuse_potential_pct": round(100.0 * share / max(tot_p, 1), 2),
        "output_estimated_rows": est_out, "rows_missing_prompt": sorted(missing),
    }

def selftest():
    import tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for i in range(20):
        tmp.write(json.dumps({"arrival_ms": i * 50, "prompt_len": 1000 + 10 * i,
                              "completion_len": 50, "conv_id": "s%d" % (i // 5)}) + "\n")
    tmp.close()
    rows, est, miss = load(tmp.name)
    s = stats(rows, est, miss)
    assert s["n_requests"] == 20, s
    assert s["span_s"] == 0.95, s
    assert s["n_sessions"] == 4, s
    assert s["prefix_reuse_potential_pct"] > 0, s
    assert est == 0 and not miss, (est, miss)
    os.unlink(tmp.name)
    print("selftest OK")
    print(json.dumps(s, ensure_ascii=False, indent=1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.trace:
        print("需要 --trace 或 --selftest", file=sys.stderr); return 2
    rows, est, miss = load(a.trace)
    s = stats(rows, est, miss)
    print(json.dumps(s, ensure_ascii=False, indent=1))
    if miss:
        print("!! 缺字段:", sorted(miss), file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
