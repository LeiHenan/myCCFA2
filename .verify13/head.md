# Verification of S13

Independent adversarial verification of `evidence/kv-gapmap/S13.md`.

- **Method**: every URL cited in S13.md was re-fetched live through the supplied `fetch.sh` proxy helper (HTTP status, effective URL and page `<title>` recorded). Every quoted string was then checked against the retrieved text with whitespace/unicode/code-span-tolerant normalisation, plus full-text HTML (`arxiv.org/html/<id>v1`) for the arXiv items and the embedded page JSON for GitHub items (rendered HTML alone omits issue/PR bodies).
- **Scope**: 49 URLs, 19 ABANDONED rows, 13 CLOSED rows, 9 OPEN rows, 8 HW_RULED_OUT rows.
- **Date of verification**: same session as the file; no content was modified in S13.md.
- **Note on 2607.27187**: arXiv serves no full-text HTML for this paper ("No HTML for '2607.27187v1'", HTTP 404). Its PDF was fetched instead and its text extracted for quote checking. The agent was right to use the abs page, but `arxiv.org/html/2607.27187v1` is **not** readable.

## URL check

All 49 cited URLs returned HTTP 200, none redirected, and every effective URL is identical to the cited URL. No DEAD and no MISMATCH. "Title seen" is the page `<title>` as rendered.

| URL | HTTP | Effective URL | Title seen | Verdict (OK / MISMATCH / DEAD) |
|---|---|---|---|---|
