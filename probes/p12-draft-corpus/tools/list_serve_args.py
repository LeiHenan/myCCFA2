"""列出 SGLang server_args 中与 ngram / speculative / cache / mem 相关的参数（权威取值）。"""
import re, sys
src = open(sys.argv[1]).read()
pat = re.compile(r"^    ([a-z_0-9]+): A\[(.*?)\n    \]", src, re.S | re.M)
want = ("ngram", "speculative", "cache", "mem_fraction", "max_running", "page_size",
        "attention_backend", "context_length", "tp_size", "dtype", "disable_radix",
        "chunked_prefill", "schedule_policy")
for m in pat.finditer(src):
    name, body = m.group(1), " ".join(m.group(2).split())
    if not any(w in name for w in want):
        continue
    if name.startswith(("speculative_ngram", "speculative_num", "speculative_eagle",
                        "speculative_algorithm", "speculative_draft", "disable_radix",
                        "chunked_prefill", "page_size", "mem_fraction_static",
                        "max_running_requests", "attention_backend", "context_length",
                        "schedule_policy", "dtype", "tp_size")) or name == "speculative_algorithm":
        d = re.search(r"\]\s*=\s*(.*)$", body)
        print("%-46s = %s" % (name, (d.group(1) if d else "?")[:60]))
