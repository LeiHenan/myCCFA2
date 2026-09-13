"""列出 SGLang server_args 中与 ngram / speculative / cache / mem 相关的参数（权威取值）。

用法：python list_serve_args.py <path-to>/sglang/srt/server_args.py
"""
import re
import sys

SRC = open(sys.argv[1], encoding="utf-8").read()
PAT = re.compile(pattern=r"^    ([a-z_0-9]+): A\[(.*?)\n    \]", flags=re.S | re.M)

WANT_PREFIX = (
    "speculative_ngram", "speculative_num", "speculative_eagle", "speculative_algorithm",
    "speculative_draft", "speculative_accept", "disable_radix", "chunked_prefill",
    "page_size", "mem_fraction_static", "max_running_requests", "attention_backend",
    "context_length", "schedule_policy", "dtype", "tp_size", "max_total_tokens",
)

for m in PAT.finditer(SRC):
    name = m.group(1)
    body = " ".join(m.group(2).split())
    if not name.startswith(WANT_PREFIX):
        continue
    d = re.search(pattern=r"\]\s*=\s*(.*)$", string=body)
    print("%-46s = %s" % (name, (d.group(1) if d else "?")[:60]))
