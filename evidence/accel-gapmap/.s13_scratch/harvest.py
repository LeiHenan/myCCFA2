#!/usr/bin/env python3
"""Background arXiv API discovery harvester for D.2 / D.4 buckets."""
import subprocess, sys, os, re, html, time, urllib.parse

OUT = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/.s13_scratch/raw"
os.makedirs(OUT, exist_ok=True)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

QUERIES = [
    # --- D.2 candidates: stated >96 GB footprints ---
    ('d2_h100_8x', 'abs:"8x H100" OR abs:"8 H100" OR abs:"eight H100"'),
    ('d2_h100_80gb', 'abs:"H100 GPUs" AND abs:"80 GB"'),
    ('d2_640gb', 'abs:"640 GB"'),
    ('d2_gpu_mem_multi', 'abs:"GPU memory" AND abs:"nodes" AND abs:"inference"'),
    ('d2_405b', 'abs:"Llama-3.1-405B" OR abs:"Llama 3.1 405B"'),
    ('d2_671b', 'abs:"671B" OR abs:"DeepSeek-V3" AND abs:"inference"'),
    ('d2_h200_8x', 'abs:"H200" AND abs:"GPUs"'),
    ('d2_multi_gpu_serve', 'abs:"multi-GPU" AND abs:"inference" AND abs:"memory"'),
    ('d2_tensor_parallel', 'abs:"tensor parallelism" AND abs:"inference" AND abs:"H100"'),
    ('d2_kv_offload_node', 'abs:"KV cache" AND abs:"host memory"'),
    # --- D.4 candidates: distributed fabric / remote store / cluster service ---
    ('d4_petals', 'abs:"Petals" OR abs:"distributed inference" AND abs:"volunteer"'),
    ('d4_s3_weight', 'abs:"S3" AND abs:"model" AND abs:"loading"'),
    ('d4_etcd', 'abs:"etcd" OR abs:"Redis" AND abs:"inference"'),
    ('d4_serverless', 'abs:"serverless" AND abs:"LLM" AND abs:"inference"'),
    ('d4_disagg_store', 'abs:"weight loading" AND abs:"distributed"'),
    ('d4_cluster_serve', 'abs:"cluster" AND abs:"LLM serving"'),
    ('d4_remote_memory', 'abs:"remote memory" AND abs:"LLM"'),
    ('d4_k8s', 'abs:"Kubernetes" AND abs:"LLM"'),
    ('d4_ckpt_store', 'abs:"checkpoint" AND abs:"distributed" AND abs:"loading" AND abs:"LLM"'),
    ('d4_model_store', 'abs:"model store" OR abs:"weight store"'),
]


def run(q):
    url = ('http://export.arxiv.org/api/query?search_query='
           + urllib.parse.quote(q, safe=':+"()')
           + '&start=0&max_results=60&sortBy=relevance')
    p = subprocess.run(["curl", "-sL", "--retry", "3", "--retry-delay", "2",
                        "--retry-all-errors", "--max-time", "55", "-A", UA, url],
                       capture_output=True)
    return p.stdout.decode("utf-8", "replace")


def parse(x):
    ents = re.findall(r'(?s)<entry>(.*?)</entry>', x)
    rows = []
    for e in ents:
        i = re.search(r'<id>(.*?)</id>', e)
        t = re.search(r'(?s)<title>(.*?)</title>', e)
        s = re.search(r'(?s)<summary>(.*?)</summary>', e)
        rows.append((html.unescape(i.group(1)) if i else '?',
                     re.sub(r'\s+', ' ', html.unescape(t.group(1))).strip() if t else '?',
                     re.sub(r'\s+', ' ', html.unescape(s.group(1))).strip() if s else ''))
    return rows


for name, q in QUERIES:
    dest = os.path.join(OUT, name + ".txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        continue
    for attempt in range(3):
        try:
            x = run(q)
        except Exception as ex:
            x = ""
        rows = parse(x)
        if rows:
            with open(dest, "w") as f:
                f.write("QUERY: " + q + "\n")
                f.write("N: %d\n\n" % len(rows))
                for i, t, s in rows:
                    f.write("ID: %s\nTITLE: %s\nABS: %s\n\n" % (i, t, s[:900]))
            print("OK %-22s %d" % (name, len(rows)), flush=True)
            break
        time.sleep(3)
    else:
        print("FAIL %-22s" % name, flush=True)
    time.sleep(1.5)
print("DONE")
