# Section E brief — NEVER-DISCUSSED (negative evidence)

Section E is now the highest-value section of this map. It is the one thing a
discussion-derived inventory structurally cannot produce: **gaps the engine itself admits to,
for which no human anywhere appears to have ever raised the issue.**

## Candidate source (already generated for you)
`/tmp/ecand/<E-id>.md` — engine-self-admitted gaps harvested from the *installed* source trees
by `evidence/accel-gapmap/scan_accel_todos.py`, which extends this repo's existing
`probes/p17-open-cells/scan_engine_todos.py` to the acceleration axes. Totals: **3583 hits in
vLLM 0.29.0, 4941 in SGLang** across 11 acceleration topics. Each line has the form:

```
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:127` **[not supported]** # PIECEWISE cudagraphs are not supported for dflash.
```

Your file lists STRONG markers (`TODO`, `FIXME`, `XXX`, `not supported`, `not implemented`,
`NotImplementedError`, `does not support`, `future work`, `raise:not supported`) first — work
those. WEAK markers (`fallback`, `workaround`, `for now`) are a sample; use them only if a gap
is genuinely implied.

Local source trees: `/tmp/vllm-0.29.0` (vLLM 0.29.0) and `/tmp/sglang-main/python` (SGLang).
You may `grep -n -A6 -B6` around any cited line to read the surrounding code and understand what
the gap actually is. **Do not cite local line numbers as the deliverable URL** — cite a
GitHub blob/raw URL (e.g.
`https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/<path>` or
`https://github.com/vllm-project/vllm/blob/v0.29.0/<path>#L<line>`) and say the local mirror was
used only for corroboration.

## Your job — the negative-evidence test
For each candidate gap, take the *specific technical claim* (e.g. "piecewise CUDA graphs are not
supported for the dflash speculator") and search **whether any human has ever raised it**:

- GitHub issue/PR search HTML across the relevant repos, e.g.
  `https://github.com/search?q=repo%3Avllm-project%2Fvllm+piecewise+cudagraph+dflash&type=issues`
  (also try `is%3Aissue`, `is%3Apr`, `is%3Aopen`, and the same on `sgl-project/sglang`,
  `NVIDIA/TensorRT-LLM`, `ggml-org/llama.cpp`, `Dao-AILab/flash-attention`,
  `flashinfer-ai/flashinfer`, `pytorch/pytorch`).
- Plain GitHub repo search pages, e.g.
  `https://github.com/vllm-project/vllm/issues?q=is%3Aissue+piecewise+cudagraph+dflash`.
- `web_search` (use it, but verify every hit at its primary URL).
- arXiv: `https://arxiv.org/search/?query=...&searchtype=all`.

Then classify:

* **NEVER-DISCUSSED** — you found no issue, PR, paper, blog or mailing-list thread raising it.
  This is the Section E row.
* **DISCUSSED** — you found a thread. Then it is NOT a Section E row. (Optionally: if the thread
  is an OPEN PR, the item is `OCCUPIED`; if merged, `SOLVED`. Report these separately, they are
  not Section E.)

## Output block format — write `E<n>.md`

```
### E<k> | NEVER-DISCUSSED
GAP (engine's own words): "<verbatim source line or error string, from the installed tree>"
SOURCE: <repo path:line> @ <version>   URL: <github raw/blob url>
WHAT IS MISSING: <one or two sentences: what capability does the engine not have?>
QUERIES TRIED: <every query string / URL you issued, separated by ' ; '>
ABSENCE CLAIM: I did not find it (searches above) | I claim it does not exist (state why the
  search space is exhaustive — e.g. the term is unique enough that any discussion would surface)
CONFIDENCE THE ABSENCE IS REAL: HIGH | MEDIUM | LOW — <one line why>
TARGET-RIG RELEVANCE: <does this bite a single RTX PRO 6000 Blackwell sm120 / vLLM 0.29.0 /
  SGLang 0.5.19 / Qwen3-4B dense setup? yes/no + one line>
EVIDENCE: READ BODY
```

**Be rigorous about the difference between "I did not find it" and "it does not exist."** State
which one you are claiming. An honest LOW-confidence absence is worth more than a bold false one.
If a gap is trivially searchable and obviously never discussed, say HIGH.

## Also allowed in your file
A short `## DISCUSSED (therefore NOT Section E)` list — items you checked and found already
discussed, with the URL. This is useful because it proves you ran the test rather than skipping it.
Include their S3b label: `OCCUPIED (open PR #N)` / `SOLVED (merged #N)`.

## Rules (unchanged)
- No verbatim quote ⇒ drop the row. Verbatim here means the source line or the doc string.
- No recommendations, no prioritisation. Inventory only.
- Do not re-map KV-cache management or speculative decoding as *subjects*. (A CUDA-graph or
  compilation gap that merely *mentions* dflash is in scope; speculative-decoding algorithm
  questions are not.)
- End the file with `## SEARCH LOG (queries actually issued)` and `## NEGATIVE FINDINGS`
  (searches that returned nothing usable — with the exact query).
- Target **12-20 NEVER-DISCUSSED rows per researcher**. Do not pad: a short honest file beats
  invented absences.
