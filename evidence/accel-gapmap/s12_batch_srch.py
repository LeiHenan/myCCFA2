#!/usr/bin/env python3
"""S12: batch GitHub search via the repo-scoped issues endpoint (works; global /search hangs)."""
import sys, subprocess, urllib.parse, os
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
from fetch import curl
import s12_list3

REPO = "NVIDIA/TensorRT-LLM"
BASE = "https://github.com/" + REPO + "/issues?q="

QUERIES = [
    # --- closed-unmerged perf PRs ---
    'is:pr is:closed is:unmerged cuda graph',
    'is:pr is:closed is:unmerged attention',
    'is:pr is:closed is:unmerged plugin',
    'is:pr is:closed is:unmerged quant',
    'is:pr is:closed is:unmerged quantization',
    'is:pr is:closed is:unmerged kernel',
    'is:pr is:closed is:unmerged scheduler',
    'is:pr is:closed is:unmerged overlap',
    'is:pr is:closed is:unmerged fusion',
    'is:pr is:closed is:unmerged MMHA',
    'is:pr is:closed is:unmerged FMHA',
    'is:pr is:closed is:unmerged revert',
    'is:pr is:closed is:unmerged XQA',
    'is:pr is:closed is:unmerged perf',
    'is:pr is:closed is:unmerged flashinfer',
    'is:pr is:closed is:unmerged int4',
    'is:pr is:closed is:unmerged fp8',
    'is:pr is:closed is:unmerged nvfp4',
    'is:pr is:closed is:unmerged coalesced',
    'is:pr is:closed is:unmerged moe',
    'is:pr is:closed is:unmerged graph',
    'is:pr is:closed is:unmerged triton',
    'is:pr is:closed is:unmerged memory',
    'is:pr is:closed is:unmerged executor',
    'is:pr is:closed is:unmerged deprecated',
    # --- deprecation / not planned text ---
    '"has been deprecated"',
    '"is deprecated"',
    '"not planned"',
    '"no plans to support"',
    '"we do not plan"',
    '"we will not support"',
    '"out of scope"',
    '"no longer supported"',
    '"removing the"',
    '"we are removing"',
    '"remove the" plugin',
    '"will be removed"',
    '"is no longer"',
    '"not supported" plug',
    'XQA',
    '"attention plugin"',
    'deprecated plugin',
    '"no immediate plans"',
    '"we have no plans"',
    '"will not be supported"',
    '"drop support"',
    '"deprecate"',
    'is:issue is:closed label:wontfix',
    'is:issue is:closed label:"not a bug"',
    'is:issue is:closed label:"known limitation"',
    'is:pr is:closed is:unmerged "not planned"',
    'is:issue is:closed "we recommend" plugin',
    'is:issue is:closed "removed" plugin',
]

if __name__ == "__main__":
    outdir = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/.s12scratch/srch"
    os.makedirs(outdir, exist_ok=True)
    allrows = {}
    for q in QUERIES:
        url = BASE + urllib.parse.quote(q, safe="")
        try:
            h = curl(url, timeout=60)
        except Exception as e:
            print("ERR", q, e)
            continue
        ns = s12_list3.nodes(h)
        seen = set()
        rows = []
        for n in ns:
            r = s12_list3.row(n)
            if r in seen:
                continue
            seen.add(r)
            rows.append(r)
        fn = os.path.join(outdir, "q_" + str(abs(hash(q)) % 10**8) + ".txt")
        with open(fn, "w") as f:
            f.write("QUERY: " + q + "\nURL: " + url + f"\nROWS: {len(rows)}\n")
            for r in rows:
                f.write(r + "\n")
        print(f"{len(rows):3d}  {q}")
    print("done")
