#!/usr/bin/env python3
"""S3b 已解决性核查的取证工具：**跨引擎 × 跨社区**的前置检索。

为什么需要它（2026-09-13 的实证教训）：上一轮只查了 SGLang 的 roadmap 与源码就继续了，
漏掉 vLLM 早已默认 ship 同一问题类（prompt-lookup decoding）⇒ 在 0 GPU 时本可拦下的候选
一路走到实验结束才被整机账杀死。**手工检索不可复现、易漏**，所以把检索固化成工具 + 产物。

用法：
  # 1) 手工检索（把结论填进 35_solved_check.md 的判定矩阵）
  python pipeline/solved_scan.py --keywords "prompt lookup decoding" "ngram draft corpus" \
      --repos vllm-project/vllm sgl-project/sglang NVIDIA/TensorRT-LLM --out out.md
  # 2) GitHub API（需 token 才稳；无 token 走 --min-interval 限速）
  python pipeline/solved_scan.py --keywords "..." --mode api --cache-dir .cache/solved_scan --out api.md
  # 3) 离线：把已保存的 API JSON 解析成同一张表（可复现，不联网）
  python pipeline/solved_scan.py --mode offline --cache-dir .cache/solved_scan --out offline.md
  python pipeline/solved_scan.py --selftest
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

DEFAULT_REPOS = ["vllm-project/vllm", "sgl-project/sglang", "NVIDIA/TensorRT-LLM",
                 "ggml-org/llama.cpp"]


def queries(keywords, repos):
    """→ [(scope, query)]；scope = 'global' 或具体 repo。"""
    out = []
    for kw in keywords:
        out.append(("global", kw))
        for r in repos:
            out.append((r, f"{kw} repo:{r}"))
    return out


def table(rows, title):
    L = [f"## {title}", "",
         "| 关键字 | 范围 | 命中数 | 最近条目（日期 / 编号 / 标题 / 状态 / 链接） |",
         "|---|---|---|---|"]
    for r in rows:
        items = r.get("items", [])[:5]
        cell = "<br>".join(f"{i['date']} · #{i['number']} · {i['title'][:70]} · "
                           f"{i['state']} · [{i['number']}]({i['url']})" for i in items) or "—"
        L.append(f"| `{r['keyword']}` | {r['scope']} | {r.get('total', len(items))} | {cell} |")
    L.append("")
    return L


def flatten_search(payload):
    items = []
    for it in payload.get("items", []) or []:
        items.append({
            "number": it.get("number"),
            "title": (it.get("title") or "").strip(),
            "state": ("merged" if it.get("pull_request", {}).get("merged_at") else it.get("state")),
            "date": (it.get("created_at") or "")[:10],
            "updated": (it.get("updated_at") or "")[:10],
            "url": it.get("html_url"),
            "score": it.get("score"),
        })
    return items


def api_search(q, cache_dir, min_interval, token=None, force=False):
    import urllib.parse
    import urllib.request
    import urllib.error

    os.makedirs(cache_dir, exist_ok=True)
    key = urllib.parse.quote_plus(q)[:180].replace("%", "_")
    cp = os.path.join(cache_dir, key + ".json")
    if os.path.exists(cp) and not force:
        with open(cp, encoding="utf-8") as fh:
            return json.load(fh)
    url = ("https://api.github.com/search/issues?q=" + urllib.parse.quote_plus(q)
           + "&per_page=20&sort=created&order=desc")
    hdrs = {"Accept": "application/vnd.github+json", "User-Agent": "ccfa-solved-scan"}
    if token:
        hdrs["Authorization"] = "Bearer " + token
    delay = max(min_interval, 6.5)  # 未认证搜索 API 约 10 次/分钟
    last = os.path.join(cache_dir, ".last_call")
    if os.path.exists(last):
        try:
            elapsed = time.time() - float(open(last, encoding="utf-8").read().strip())
            if elapsed < delay:
                time.sleep(delay - elapsed)
        except Exception:  # noqa: BLE001
            pass
    for attempt in range(4):
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                d = json.loads(r.read().decode())
            open(last, "w", encoding="utf-8").write(str(time.time()))
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False)
            return d
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            wait = 20 * (attempt + 1)
            print(f"  ! HTTP {e.code}（{body[:80]}）⇒ 退避 {wait}s", file=sys.stderr)
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {type(e).__name__}: {e} ⇒ 退避 15s", file=sys.stderr)
            time.sleep(15)
    return {"items": [], "_error": "failed after retries"}


def manual_template(keywords, repos, out):
    L = ["# S3b 已解决性核查 — 检索记录（手工模式）", "",
         "> ⚠️ 手工模式**不产生证据**：下面的每一行都必须由人/Agent 实际打开链接、读过正文后填写，",
         "> 并把结论抄进 `35_solved_check.md` 的判定矩阵。**未读正文的行标『未取证』**。", "",
         "**检索关键字**：" + "、".join(f"`{k}`" for k in keywords), "",
         "**必查对象**：" + "、".join(f"`{r}`" for r in repos), "",
         "## 待查清单", "",
         "| # | 对象 | 状态（已 ship / RFC / 论文 / 无人做） | URL | 查证深度（读了哪一节 / file:line） | 与我们的差异 |",
         "|---|---|---|---|---|---|"]
    for i, r in enumerate(repos, 1):
        L.append(f"| {i} | `{r}` | 待填 | 待填 | 待填 | 待填 |")
    L.append(f"| {len(repos)+1} | 会议论文（把关键字丢进检索式） | 待填 | 待填 | 待填 | 待填 |")
    L.append(f"| {len(repos)+2} | 生产实践（工程博客 / 论坛 / 用户抱怨） | 待填 | 待填 | 待填 | 待填 |")
    L += ["", "## 自检（提交前逐条核）", "",
          "- [ ] 判定矩阵 ≥5 行，且覆盖 ≥3 个引擎/实现与 ≥2 个研究社区",
          "- [ ] 每行都有**可点开的 URL** 与**查证深度**（未读正文的必须写『未取证』）",
          "- [ ] 写了一条**反证**：主动找证据说明这个方向可能已被解决",
          "- [ ] 写明了『若被占，降级成什么』且降级不需要新半径"]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return out


def offline(cache_dir, out):
    rows = []
    for p in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        name = os.path.basename(p)[:-5]
        rows.append({"keyword": name, "scope": "(cache)", "total": d.get("total_count"),
                     "items": flatten_search(d)})
    body = ["# S3b 已解决性核查 — 检索记录（离线解析缓存）", "",
            f"缓存目录：`{cache_dir}`（{len(rows)} 条查询）", ""]
    body += table(rows, "全部缓存查询")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body) + "\n")
    return out, rows


def run_api(keywords, repos, cache_dir, min_interval, token, out, force=False):
    qs = queries(keywords, repos)
    rows = []
    for scope, q in qs:
        print(f"[{scope}] {q}")
        d = api_search(q, cache_dir, min_interval, token, force=force)
        rows.append({"keyword": q.split(" repo:")[0], "scope": scope,
                     "total": d.get("total_count"), "items": flatten_search(d)})
    body = ["# S3b 已解决性核查 — 检索记录（GitHub API）", "",
            f"查询数：{len(qs)}；缓存：`{cache_dir}`；关键字：" + "、".join(f"`{k}`" for k in keywords), ""]
    body += table(rows, "按范围汇总")
    body += ["## 使用方式", "",
             "把上表里 **状态=merged 且日期最近**的条目逐条打开读正文，抄进 `35_solved_check.md`：",
             "命中面 ≥3 条才算完成核查；任何一条『已 ship 同一问题类』都要写进降级路径。", ""]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body) + "\n")
    return out, rows


def selftest():
    import tempfile
    td = tempfile.mkdtemp()
    # queries 覆盖：每个关键字 1 次全局 + 每 repo 1 次
    qs = queries(["kw1", "kw2"], ["a/b", "c/d"])
    assert len(qs) == 2 * (1 + 2), qs
    assert ("global", "kw1") in qs and ("a/b", "kw1 repo:a/b") in qs
    # 手工模板
    o = manual_template(["kw1"], ["a/b", "c/d"], os.path.join(td, "man.md"))
    t = open(o, encoding="utf-8").read()
    assert "待查清单" in t and t.count("待填") >= 4 and "反证" in t
    # 离线解析：伪造一个搜索缓存
    cd = os.path.join(td, "cache")
    os.makedirs(cd)
    fake = {"total_count": 2, "items": [
        {"number": 11, "title": "feat: x", "state": "open", "created_at": "2026-08-01T00:00:00Z",
         "updated_at": "2026-08-02T00:00:00Z", "html_url": "https://x/11", "score": 1.0,
         "pull_request": {"merged_at": None}},
        {"number": 12, "title": "feat: y", "state": "closed", "created_at": "2026-07-01T00:00:00Z",
         "updated_at": "2026-07-02T00:00:00Z", "html_url": "https://x/12", "score": 2.0,
         "pull_request": {"merged_at": "2026-07-05T00:00:00Z"}}]}
    json.dump(fake, open(os.path.join(cd, "q1.json"), "w", encoding="utf-8"))
    o2, rows = offline(cd, os.path.join(td, "off.md"))
    assert len(rows) == 1 and rows[0]["items"][1]["state"] == "merged", rows
    assert "#12" in open(o2, encoding="utf-8").read()
    # 全局搜索：**故意不把结果写成表**（避免用户不读链接就套用）
    print("selftest ✔ 查询展开/手工模板/离线解析(merged 识别)/产物含编号与链接")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", nargs="*", default=[])
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    ap.add_argument("--mode", choices=["manual", "api", "offline"], default="manual")
    ap.add_argument("--cache-dir", default=".cache/solved_scan")
    ap.add_argument("--min-interval", type=float, default=6.5)
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    ap.add_argument("--out", default="solved_scan.md")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.mode == "manual":
        if not a.keywords:
            ap.error("manual 模式需要 --keywords（问题类的多种叫法）")
        print("已写出：", manual_template(a.keywords, a.repos, a.out))
        print("⇒ 下一步：逐行打开链接、读正文、填进 35_solved_check.md（未读正文标『未取证』）")
        return 0
    if a.mode == "offline":
        o, rows = offline(a.cache_dir, a.out)
        print(f"已写出：{o}（{len(rows)} 条查询）")
        return 0
    if not a.keywords:
        ap.error("api 模式需要 --keywords")
    o, rows = run_api(a.keywords, a.repos, a.cache_dir, a.min_interval, a.token, a.out, a.force)
    print(f"已写出：{o}（{len(rows)} 条查询）")
    print("⚠️ API 只给**标题级**信息；S3b 要求读正文 ⇒ 逐条打开再填判定矩阵。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
