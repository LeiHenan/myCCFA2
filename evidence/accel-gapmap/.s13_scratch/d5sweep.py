#!/usr/bin/env python3
"""Sequential arXiv API sweep with long backoff; writes one file per query."""
import os, sys, time, hashlib, subprocess, re, html

OUT = "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/.s13_scratch"
CACHE = "/tmp/accelsrc"
os.makedirs(CACHE, exist_ok=True)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

QUERIES = [
    'abs:%22single+GPU%22+AND+abs:%22tensor+parallelism%22',
    'abs:%22single+GPU%22+AND+abs:%22tensor+parallel%22',
    'abs:%22single+GPU%22+AND+abs:%22multi-GPU%22',
    'abs:%22single+GPU%22+AND+abs:%22expert+parallelism%22',
    'abs:%22single+GPU%22+AND+abs:%22pipeline+parallelism%22',
    'abs:%22one+GPU%22+AND+abs:%22tensor+parallel%22',
    'abs:%22single+GPU%22+AND+abs:%22disaggregated%22',
    'abs:%22single+GPU%22+AND+abs:%22LLM+inference%22',
    'abs:%22single+device%22+AND+abs:%22LLM%22',
    'abs:%22single-GPU%22+AND+abs:%22inference%22',
    'abs:%228+GPUs%22+AND+abs:%22inference%22',
    'abs:%22tensor+parallelism%22+AND+abs:%22serving%22',
    'abs:%22multi-GPU%22+AND+abs:%22LLM+inference%22',
    'abs:%22multi-node%22+AND+abs:%22LLM+inference%22',
    'abs:%22context+parallelism%22+AND+abs:%22inference%22',
    'abs:%22sequence+parallelism%22+AND+abs:%22inference%22',
    'abs:%22offloading%22+AND+abs:%22multi-GPU%22',
]


def unesc(s):
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def main():
    for i, q in enumerate(QUERIES):
        url = ("http://export.arxiv.org/api/query?search_query=" + q +
               "&start=0&max_results=60&sortBy=submittedDate&sortOrder=descending")
        key = hashlib.sha1(("API::" + url).encode()).hexdigest()
        path = os.path.join(CACHE, key + ".atom")
        if os.path.exists(path) and os.path.getsize(path) > 800:
            data = open(path, encoding="utf-8").read()
        else:
            data = ""
            for attempt in range(12):
                p = subprocess.run(["curl", "-sL", "--max-time", "60", "-A", UA, url],
                                   capture_output=True)
                data = p.stdout.decode("utf-8", "replace")
                if "<entry" in data:
                    break
                time.sleep(20 + 15 * attempt)
            if len(data) > 800 and "<entry" in data:
                open(path, "w", encoding="utf-8").write(data)
        entries = re.findall(r"(?s)<entry>(.*?)</entry>", data)
        lines = [f"### QUERY: {q} -> entries: {len(entries)}"]
        for e in entries:
            aid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
            t = re.search(r"(?s)<title>(.*?)</title>", e)
            pub = re.search(r"<published>([^<]+)</published>", e)
            ab = re.search(r"(?s)<summary>(.*?)</summary>", e)
            lines.append("-" * 90)
            lines.append("ID: " + (aid.group(1) if aid else "?"))
            lines.append("DATE: " + (pub.group(1)[:10] if pub else "?"))
            lines.append("TITLE: " + (unesc(t.group(1)) if t else "?"))
            lines.append("ABS: " + (unesc(ab.group(1)) if ab else "?"))
        open(os.path.join(OUT, f"api_q{i:02d}.txt"), "w", encoding="utf-8").write("\n".join(lines))
        print(f"q{i:02d}: {len(entries)} entries  {q}", flush=True)
        time.sleep(6)


if __name__ == "__main__":
    main()
