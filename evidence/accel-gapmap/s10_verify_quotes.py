#!/usr/bin/env python3
"""Verify every quoted string in S10_gh_sub.md against the fetched source text."""
import re, sys

MD = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/S10_gh_sub.md"
F = "/tmp/s10_fetch/"
DOCS = "/tmp/vllm_bench_cli.md"

# quote-substring -> source file
MAP = {
 "Anyone building a concurrency ladder": F + "issu52884.txt",
 "Repeating the same vllm bench serve workload": F + "pull53920.txt",
 "Verified against current": F + "issu54227.txt",
 "Repeating `vllm bench serve` against the same server": DOCS,
 "When --max-concurrency is set, a request": F + "issu54101.txt",
 "Looks like the issue comes from the default value": F + "issu24684.txt",
 "Providing a --dataset-path was forcing": F + "issu24684.txt",
 "Your conclusion is based on your model": F + "issu998.txt",
 "Additionally, I noticed that you have very few": F + "issu998.txt",
 "You can add `--enable-torch-compile`": F + "issu998.txt",
 "The reason it's called bench one batch": F + "issu18712.txt",
 "sglang.bench_serving can print misleading": F + "issu23949.txt",
 "The current output looks like valid benchmark data": F + "issu23949.txt",
 "It measures one phase of the L-phase cycle": F + "issu38398.txt",
 "I think the remaining difference here": F + "issu6294.txt",
 "I'm also working on updating our performance document": F + "issu6294.txt",
 "Your config only has 128 for the cuda graph": F + "issu6294.txt",
 "in both of the rerun tests it looks like": F + "issu6294.txt",
 "The test_perf.py regression test shows large variance": F + "issu8391.txt",
 "Is inclusive one-second bucket occupancy intentional": F + "issu56653.txt",
 "One thing I'd push on in your protocol": F + "issu42484.txt",
 "For English that distinction rarely matters": F + "issu51963.txt",
 "vllm bench serve reports latency and throughput": F + "pull46938.txt",
 "An HTTP 200 SSE response can report a generation error": F + "pull38868.txt",
 "Running the generated-shared-prefix benchmark": F + "pull38917.txt",
 "My \"fixed\" harness counted": F + "issu54677.txt",
 "I was reporting per-stream": F + "issu54677.txt",
 "My 0.26 baseline was running": F + "issu54677.txt",
 "The original 4.3": F + "issu42484.txt",
 "New Finding 5": F + "issu42484.txt",
 "So the benchmark output is right": F + "issu1630.txt",
}

def norm(s):
    s = re.sub(r"\s+", " ", s)
    return s.strip()

qcache = {}
def src(path):
    if path not in qcache:
        qcache[path] = norm(open(path, encoding="utf-8", errors="replace").read())
    return qcache[path]

md = open(MD, encoding="utf-8").read()
ok = bad = 0
for key, path in MAP.items():
    if key in src(path):
        ok += 1
    else:
        bad += 1
        print(f"MISSING in {path.split('/')[-1]}: {key!r}")

# also verify every QUOTE:/ASKING QUOTE:/STATED REASON: line's leading fragment resolves
for m in re.finditer(r"^(?:QUOTE|ASKING QUOTE|STATED REASON): \"(.{20,90})", md, re.M):
    frag = norm(m.group(1))
    frag = frag.replace('\\"', '"')
    hit = any(frag in src(p) for p in set(MAP.values()))
    if not hit:
        print("UNRESOLVED QUOTE LINE:", frag)
print(f"\n{ok} ok / {bad} missing")
