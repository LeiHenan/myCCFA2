#!/usr/bin/env python3
"""Extended engine-self-admitted-gap scanner for the ACCELERATION axes.

Wraps /Users/leihenan/Desktop/myProject/probes/p17-open-cells/scan_engine_todos.py
(reuses its PATTERNS verbatim) and replaces TOPIC_KWS with acceleration-axis topics,
so the same "the engine itself says it did not do this" signal is harvested for the
axes this map owns instead of spec-decode / KV.

Usage:
  python3 scan_accel_todos.py --root /tmp/vllm-0.29.0 --tag vllm_accel
  python3 scan_accel_todos.py --root /tmp/sglang-main/python --tag sglang_accel
  python3 scan_accel_todos.py --selftest
"""
import argparse, importlib.util, os, re, sys
from collections import Counter, defaultdict

ORIG = "/Users/leihenan/Desktop/myProject/probes/p17-open-cells/scan_engine_todos.py"
_spec = importlib.util.spec_from_file_location("orig_scan", ORIG)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
PATTERNS = _m.PATTERNS
MAX_LINE = 200

TOPIC_KWS = {
    "attn_backend": ["attention", "attn", "flashinfer", "flash_attn", "flash-attn", "trtllm",
                     "cutlass", "triton", "flex_attention", "flexattention", "backend",
                     "head_dim", "sliding_window", "kv_cache_dtype", "xqa", "fmha"],
    "compile_cudagraph": ["compile", "inductor", "dynamo", "cudagraph", "cuda_graph", "capture",
                          "piecewise", "torch.compile", "pass_config", "fusion", "fuse",
                          "splitting_ops", "dynamic_shape", "guard"],
    "weight_act_quant": ["quant", "fp8", "int4", "int8", "awq", "gptq", "nvfp4", "mxfp",
                         "marlin", "machete", "w8a8", "w4a16", "scale", "smoothquant",
                         "bitsandbytes", "torchao", "modelopt"],
    "batching_sched": ["schedul", "chunked", "prefill", "decode", "batching", "preempt",
                       "max_num_batched_tokens", "max_num_seqs", "continuous_batch",
                       "priority", "queue", "admission"],
    "sampling_logits": ["sampl", "logit", "top_k", "top_p", "min_p", "penalty", "seed",
                        "grammar", "xgrammar", "outlines", "llguidance", "structured",
                        "guided", "regex", "json_schema"],
    "tokenizer_detok": ["tokeniz", "detokeniz", "detoken", "tokenizer_manager", "mistral"],
    "host_cpu_overhead": ["cpu", "host", "overhead", "pinned", "omp_", "num_threads", "thread",
                          "synchron", "launch", "zmq", "ipc", "serializ", "msgpack", "pickle",
                          "numpy", "copy_", "block_table", "slot_mapping"],
    "serve_api": ["metrics", "prometheus", "api_", "http", "frontend", "uvicorn", "fastapi",
                  "request", "middleware", "stream", "sse", "openai", "grpc"],
    "longctx_attn": ["rope", "rotary", "context_parallel", "ring", "window", "sparse",
                     "sink", "yarn", "ntk", "max_model_len", "chunked_local"],
    "memory_alloc": ["alloc", "memory_pool", "expandable", "cudaMalloc", "workspace",
                     "graph_pool", "sleep", "wake", "offload", "prefetch"],
}
SKIP_DIRS = {"__pycache__", "test", "tests", "csrc", "3rdparty", "benchmark", "benchmarks",
             "docs", "doc", "examples", "assets", "scripts", "build"}


def scan(root, topics=None):
    hits = defaultdict(list)
    counts = Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith((".py", ".cu", ".cuh", ".h", ".cpp", ".yaml", ".json")):
                continue
            p = os.path.join(dirpath, fn)
            try:
                fh = open(p, encoding="utf-8", errors="replace")
            except Exception:
                continue
            for ln, line in enumerate(fh, 1):
                low = line.lower()
                for pat, label in PATTERNS:
                    if re.search(pat, line, re.IGNORECASE):
                        counts[label] += 1
                        # score by number of distinct keyword hits, tie-break by declaration
                        # order, so an ambiguous line lands in its dominant topic
                        scored = [(sum(1 for k in kws if k in low), -i, t)
                                  for i, (t, kws) in enumerate(TOPIC_KWS.items())]
                        scored = [s for s in scored if s[0] > 0]
                        key = max(scored)[2] if scored else "other"
                        if topics and key not in topics:
                            break
                        hits[key].append((os.path.relpath(p, root), ln, label,
                                          line.strip()[:MAX_LINE]))
                        break
    return hits, counts


def selftest():
    import tempfile
    td = tempfile.mkdtemp()
    os.makedirs(os.path.join(td, "srt"))
    open(os.path.join(td, "srt", "a.py"), "w").write(
        "# TODO: cudagraph capture is not supported for this model\n"
        "raise NotImplementedError('nvfp4 weight quant for now')\n"
        "# TODO: the tokenizer does not support this mode\n"
        "# unrelated\n")
    hits, counts = scan(td)
    tot = sum(len(v) for v in hits.values())
    assert tot == 3, (tot, dict(hits))
    assert "compile_cudagraph" in hits, hits.keys()
    assert "weight_act_quant" in hits, hits.keys()
    assert "tokenizer_detok" in hits, hits.keys()
    print("selftest OK — patterns match, accel topic routing works, 3 hits")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--tag", default="accel")
    ap.add_argument("--topics")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.root, "--root required"
    topics = set(a.topics.split(",")) if a.topics else None
    hits, counts = scan(a.root, topics)
    lines = [f"# {a.tag} engine-self-admitted gaps (root={a.root})", "", "## pattern hit totals"]
    for k, v in counts.most_common():
        lines.append(f"- {k}: {v}")
    lines += ["", "## by acceleration topic"]
    for t in sorted(hits):
        lines += [f"### {t} ({len(hits[t])})", ""]
        for f, ln, label, txt in hits[t][:200]:
            lines.append(f"- `{f}:{ln}` **[{label}]** {txt}")
        lines.append("")
    out = a.out or f"/tmp/{a.tag}_todos.md"
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{a.tag}: {sum(len(v) for v in hits.values())} hits across {len(hits)} topics -> {out}")
    for t in sorted(hits):
        print(f"  {t}: {len(hits[t])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
