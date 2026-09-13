# Researcher brief — Inference Acceleration Gap Map (NOT KV cache, NOT speculative decoding)

## Target rig (decides relevance)
ONE GPU: **RTX PRO 6000 Blackwell, 96 GB, sm120** (compute capability 12.0; note
`torch.cuda.is_device_capability_family(100)` is **FALSE** on sm120 — sm120 is *not* sm100).
driver 580, 208 CPU cores, single node, **no NVLink / no InfiniBand / no cluster**.
Software: **vLLM 0.29.0**, **SGLang 0.5.19**. Target model **Qwen3-4B**: dense, full attention,
NOT MoE, NOT hybrid Mamba, NOT MLA/DeepSeek. Drafters: dflash2 (DFlash/MTP-class) + EAGLE3 Qwen3-4B.

**OUT OF SCOPE (owned by two sibling maps in this repo — do NOT re-map):**
1. KV-cache management (paging, prefix caching, eviction, KV quant, offload/tiering, disaggregated P/D).
2. Speculative decoding (drafters, draft training, n-gram/suffix, verification, acceptance length).
You MAY cite them as *adjacent context* in one clause, but they must not be a row's subject.

## Fetching — use the helper, not the raw tools
`web_fetch` is unreliable here. Bare `curl` to arxiv.org / api.github.com often returns HTTP 000
through the local proxy. **`api.github.com` is rate-limited to zero — do not use it.**

```bash
cd /Users/leihenan/Desktop/myProject/evidence/accel-gapmap
python3 fetch.py gh    https://github.com/vllm-project/vllm/pull/12345   # title,state,comments
python3 fetch.py arxiv 2506.12345                                        # title/abstract/dates
python3 fetch.py get   https://docs.vllm.ai/en/latest/...                # any page as text
python3 fetch.py raw   <url>                                             # raw bytes
```
Raw HTML is cached in `/tmp/accelsrc/`. If you need a different UA / a site that blocks:
`curl -sL --retry 4 --retry-delay 2 --retry-all-errors -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36" <url>`

**GitHub issue/PR comment bodies ARE present in the fetched HTML** (`<div class="comment-body">`
and `"body":"` inside the embedded JSON payload). A previous researcher wrongly claimed they do not
render, and that hid real maintainer closure reasons on ~5 rows. `fetch.py gh` extracts them. Verify
before you ever conclude a body is unavailable.

**Every arXiv ID must be read back** with `fetch.py arxiv <id>` and its title confirmed — `web_search`
has previously returned fabricated arXiv IDs in this project. Never cite an unverified ID.

## Evidence rules (non-negotiable)
1. **No verbatim quote ⇒ drop the row.** Paraphrase is not a quote. If a quote is a composite of
   fragments, label it `STATED REASON (composite: quoted fragments interleaved with retrieved framing text)`.
2. **Automated stale-bot closures with no human statement ⇒ DROP the item** (e.g. `github-actions`
   "This issue has been automatically marked as stale"). A `stateReason: not_planned` with zero human
   comments is *not* a closure reason — drop it, or record it only if a human comment states a reason.
3. Label every item `READ BODY` / `TITLE ONLY` / `TITLE + STATE ONLY` / `APPROXIMATE`.
   **TITLE ONLY must never be the sole basis for a claim.**
4. Prefer primary sources (the PR/issue/commit/doc page). If you cite a blog, say "blog".
5. Do **not** invent rows to fill a quota. An honest "found nothing, queries tried: ..." beats padding.
6. Do **not** give recommendations or prioritisation. This is an inventory, not a proposal.
7. Record the **exact queries you issued** (GitHub search URLs, web_search strings) — they go in the file.

## Useful search entry points (HTML, no API key)
- `https://github.com/search?q=repo%3Avllm-project%2Fvllm+%22is%3Aunmerged+is%3Apr%22+<terms>&type=issues`
  (URL-encode; `type=issues` covers PRs too). Add `is%3Aclosed`, `label%3A"not+planned"`, `sort%3Aupdated-desc`.
- `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+is%3Aclosed+label%3Awontfix`
- `https://github.com/sgl-project/sglang/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged`
- `https://github.com/NVIDIA/TensorRT-LLM/pulls?q=is%3Apr+is%3Aclosed+is%3Aunmerged`
- docs: `https://docs.vllm.ai/en/latest/...`, `https://docs.sglang.io/...`,
  raw docs at `https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/...` (0.29.0 is the rig's version).
- Local repo checkouts may exist under `/tmp` for grepping source/config flags — check before fetching.

## Output format — write ONE file: `S<n>.md` in this directory
Return only a short summary (counts + the 3 most consequential findings) to the caller; the file is
the deliverable. Use exactly these block shapes so the aggregator can parse them.

```
### A<n>.<k> | CLOSED
QUESTION: <the question this answers>
WHO: <who closed it>
ARTIFACT: <paper/PR number/flag/config key>
URL: <primary url>            # multiple allowed, separated by ' ; '
STATUS: <merged|shipped|published> <date if known>
EVIDENCE: READ BODY | TITLE ONLY | TITLE + STATE ONLY | APPROXIMATE
QUOTE: "<verbatim, continuous, from the retrieved page>"
```

```
### B<n>.<k> | OPEN
WHO IS STILL ASKING: <person/org + venue>
URL: <url>
ASKING QUOTE: "<verbatim, continuous>"
WHAT IS MISSING: <one or two sentences, concrete>
EVIDENCE: READ BODY | TITLE ONLY | APPROXIMATE
```
An OPEN item is only valid if you have a **verbatim quote of someone asking** — a question, a
"we could not", a "TBD", a "not supported yet", a limitation paragraph, a maintainer "no ETA".

```
### C<n>.<k> | ABANDONED   (<thematic subsection name>)
WHO: <who tried>
ARTIFACT: <PR/issue/paper id + title>
URL: <url>
STATED REASON: <verbatim closure/failure reason>
EVIDENCE: READ BODY | TITLE ONLY | APPROXIMATE
```

```
### D<n>.<k> | HARDWARE-RULED-OUT   (D.1 >1 GPU | D.2 >96 GB VRAM | D.3 multi-node/NVLink/IB |
                                     D.4 datacenter fabric/distributed service | D.5 borderline)
WHO: <authors/maintainers>
ARTIFACT: <paper/PR/issue>
URL: <url>
HARDWARE QUOTE: "<verbatim, including any GPU-count words — DO NOT trim 'with four GPUs' etc.>"
EVIDENCE: READ BODY | TITLE ONLY | APPROXIMATE
```
**Hardware quotes must be exact and complete.** A previous verifier caught a quote that had been
silently edited inside the quotation marks to delete "with four GPUs". Quote the whole sentence.

At the end of your file add:
```
## SEARCH LOG (queries actually issued)
- <query or URL>
## NEGATIVE FINDINGS
- <subtopic where nothing was found> — queries tried: <...>
```
