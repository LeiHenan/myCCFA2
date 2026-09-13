# Verifier brief — adversarial verification layer (inference-accel gap map)

You are an **adversarial verifier**. You did NOT write the evidence blocks you are checking.
Your job is to try to **falsify** them. A verifier that confirms everything has failed.

## What you receive
One or more `S<n>.md` files in `/Users/leihenan/Desktop/myProject/evidence/accel-gapmap/`
(the subtopic researchers' raw evidence blocks). Read them with the `read` tool.

## What you must produce
Write `S<n>.verify.md` for each file you are assigned, in this shape:

```
# Verification report — S<n>
## Corrections (rows I changed)
- ROW: <block id> | FIELD: <field> | WAS: <old text, truncated to 200 chars> | NOW: <new text or DROP>
  | REASON: <what I actually saw at the URL> | EVIDENCE: <url> | <fetch method + http status>
## Confirmations (rows I checked and could not falsify)
- <block id> — quote re-located at source; state token re-read as <state>
## Could not verify
- <block id> — <why: 404, paywall, PDF-only, login-gated, ...>
## Tally
checked=N corrected=M dropped=K verbatim_relocated=P unverifiable=Q
```

## Method — for EVERY block
1. **Re-fetch the URL yourself.** Use `python3 fetch.py gh|arxiv|get|raw <url>` (see
   `RESEARCHER_BRIEF.md` for the fetch notes). `api.github.com` is rate-limited to zero — do not use
   it. `web_fetch` is unreliable — do not use it. Bare `curl` to arxiv.org often returns HTTP 000
   through the local proxy; if `fetch.py` fails, retry with the explicit curl line from the brief.
2. **Re-locate the claimed verbatim quote in the retrieved page text.** Use the mechanical checker:
   ```bash
   cd /Users/leihenan/Desktop/myProject/evidence/accel-gapmap
   python3 verify.py quote "<url>" "<the claimed quote>"
   ```
   `MATCH` = found as one continuous run. `MATCH-FRAGMENTS` = all fragments present but separated by
   an ellipsis. `NEAR`/`MISMATCH` = **not** found; read the `missing_tail` field — that is exactly
   where the quote diverges, and it is how a previous audit caught a hardware quote that had been
   silently edited to delete the words "with four GPUs".
   A `MISMATCH` is not automatically a defect — it may be a composite. But then the block **must**
   carry the label `(composite: quoted fragments interleaved with retrieved framing text)`.
3. **Try to falsify the STATUS claim.** Specifically:
   - A row marked **CLOSED/merged** — is the PR actually merged, or merely closed-unmerged? Read the
     state token from the page (`fetch.py gh` prints `STATE-TOKEN`, `MERGED_AT`, `CLOSED_AT`,
     `STATE_REASON`). A row claiming "3 PRs merged" must have **each** PR verified merged.
   - A row marked **OPEN** — has a later merged PR or a comment already fixed it? Search the repo for
     a follow-up. **A previous audit found an OPEN row that a merged PR had already fixed.**
   - A row marked **ABANDONED** — is the artifact actually still open? **A previous audit found an
     ABANDONED row whose artifact was still open**; it belonged in OPEN.
   - A row marked **HARDWARE-RULED-OUT** — does the quoted sentence actually say what the bucket
     claims, with no GPU-count words removed? Quote the whole sentence.
4. **Apply the drop rules yourself.** If the only closure text is an automated stale-bot notice with
   no human statement, the row must be DROPPED. If a row has no verbatim quote at all, DROP it.
5. **Check evidence labels.** `TITLE ONLY` must never be the sole basis for a claim. If a row's only
   support is a search-result title, either find the body or downgrade/drop the row.
6. **Check every arXiv ID by reading it back**: `python3 fetch.py arxiv <id>` and confirm the title
   matches what the row claims. `web_search` has returned fabricated arXiv IDs in this project.

## Local source mirrors you may use for source-level checks
- `/tmp/vllm-0.29.0` — vLLM 0.29.0 source tree (no git metadata).
- `/tmp/sglang-main` — SGLang source tree (no git metadata).
- `/tmp/kvsrc`, `/tmp/vllm_src` — older snapshots from a sibling map.
Grepping these is good corroboration but **not** a substitute for a URL: cite the GitHub raw/blob
URL, and say the local mirror was used for corroboration only. Line numbers drift — do not present
them as line-precise on live `main`.

## Rules
- Report **every** correction you apply, with the reason and what you actually saw.
- Do NOT silently fix a row. The correction must be visible in your report so the orchestrator can
  apply it and so the final file's correction log is honest.
- If you cannot verify something, say so — that is a result, not a failure.
- Do not add new rows. You are not a researcher. If you find a glaring gap while verifying, note it
  under `## Could not verify` or a short `## Notes` section; do not pad the map.
