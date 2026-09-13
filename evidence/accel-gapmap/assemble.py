#!/usr/bin/env python3
"""Assemble INFERENCE_ACCEL_GAP_MAP.md from S*.md / E*.md evidence blocks.

  python3 assemble.py

Reads  _index.jsonl    (produced by `python3 verify.py parse`)
       corrections.json (verification-forced edits: drops, reclassifications, field rewrites)
       HEADER.md        (prose header + search log; written by hand at the end)
Writes ../INFERENCE_ACCEL_GAP_MAP.md
"""
import json, os, re, sys
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "INFERENCE_ACCEL_GAP_MAP.md"))

CORR = {}
cpath = os.path.join(HERE, "corrections.json")
if os.path.exists(cpath):
    CORR = json.load(open(cpath, encoding="utf-8"))

HEADER = ""
_hp = os.path.join(HERE, "HEADER.md")
if os.path.exists(_hp):
    HEADER = open(_hp, encoding="utf-8").read()


def load():
    recs = []
    for line in open(os.path.join(HERE, "_index.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def apply_corrections(recs):
    out, dropped, log = [], [], []
    for r in recs:
        c = CORR.get(r["id"])
        if c:
            if c.get("drop"):
                dropped.append((r["id"], c.get("why", "")))
                log.append(f"DROP {r['id']} — {c.get('why','')}")
                continue
            for k, v in c.items():
                if k in ("drop", "why", "note"):
                    continue
                r[k] = v
            if c.get("note"):
                log.append(f"EDIT {r['id']} — {c['note']}")
        if r.get("_DROP"):
            dropped.append((r["id"], r.get("_DROP")))
            continue
        out.append(r)
    return out, dropped, log


def g(r, *names, default=""):
    for n in names:
        for k in (n, n.replace(" ", "_"), n.replace("-", "_"), n.upper()):
            if r.get(k):
                return r[k]
    return default


def clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def md_cell(s):
    return clean(s).replace("|", "\\|")


def urls(r):
    u = g(r, "URL", "URLS")
    return [x.strip() for x in re.split(r"\s*;\s*", u) if x.strip()]


def url_html(r):
    us = urls(r)
    if not us:
        return ""
    if len(us) == 1:
        return f"[link]({us[0]})"
    return " ; ".join(f"[{i+1}]({u})" for i, u in enumerate(us))


def ev(r):
    return clean(g(r, "EVIDENCE", "READ", default="READ BODY"))


def section_a(recs, n_start=1):
    rows, i = [], n_start
    for r in recs:
        if r["section"] != "CLOSED":
            continue
        rows.append("| A{} | {} | {} | {} | {} | {} | {} |".format(
            i, md_cell(g(r, "QUESTION")), md_cell(g(r, "WHO")),
            md_cell(g(r, "ARTIFACT")), url_html(r), md_cell(g(r, "STATUS")), ev(r)))
        i += 1
    hdr = ("| # | Question it answers | Who closed it | Artifact (paper/PR/flag/config) | URL | "
           "Status (merged/shipped/published + date) | READ BODY or TITLE ONLY |\n"
           "|---|---|---|---|---|---|---|")
    return hdr + "\n" + "\n".join(rows), i - n_start


def section_b(recs):
    parts, n = [], 0
    for r in recs:
        if r["section"] != "OPEN":
            continue
        n += 1
        parts.append(
            f"### B{n}. {clean(g(r,'QUESTION','WHO_IS_STILL_ASKING'))[:170]}\n\n"
            f"* **WHO is still asking:** {clean(g(r,'WHO_IS_STILL_ASKING','WHO'))}\n"
            f"* **URL:** {' ; '.join(urls(r))}\n"
            f"* **Verbatim quote of the asking:** {clean(g(r,'ASKING_QUOTE','QUOTE'))}\n"
            f"* **What specifically is missing:** {clean(g(r,'WHAT_IS_MISSING','MISSING'))}\n"
            f"* **S3b status:** {clean(g(r,'S3B', default='NOT RUN'))}\n"
            f"* **Evidence:** {ev(r)}\n")
    return "\n".join(parts), n


def section_e(recs):
    """NEVER-DISCUSSED (negative evidence) — what a discussion-derived map cannot produce."""
    parts, n = [], 0
    for r in recs:
        if r["section"] != "NEVER-DISCUSSED":
            continue
        n += 1
        parts.append(
            f"### E{n}. {clean(g(r,'WHAT_IS_MISSING','GAP'))[:180]}\n\n"
            f"* **GAP (engine's own words):** {clean(g(r,'GAP'))}\n"
            f"* **SOURCE:** {clean(g(r,'SOURCE'))} — {url_html(r)}\n"
            f"* **What is missing:** {clean(g(r,'WHAT_IS_MISSING'))}\n"
            f"* **Queries tried:** {clean(g(r,'QUERIES_TRIED'))}\n"
            f"* **Absence claim:** {clean(g(r,'ABSENCE_CLAIM'))}\n"
            f"* **Confidence the absence is real:** "
            f"{clean(g(r,'CONFIDENCE_THE_ABSENCE_IS_REAL','CONFIDENCE'))}\n"
            f"* **Target-rig relevance:** {clean(g(r,'TARGET_RIG_RELEVANCE','TARGET_RIG'))}\n"
            f"* **Evidence:** {ev(r)}\n")
    return "\n".join(parts), n


C_GROUPS = OrderedDict([
    ("C.1", "Closed-unmerged PRs (a maintainer or author ended the attempt)"),
    ("C.2", "Wontfix / not-planned / by-design"),
    ("C.3", "Stale-closed (kept only where a human, not a bot, stated a reason)"),
    ("C.4", "Reverted / withdrawn / retracted / removed"),
    ("C.5", "Measured negative results (the attempt ran and did not pay off)"),
    ("C.6", "Papers undercut by their own stated limitations"),
])


def guess_group(r):
    if r.get("GROUP"):
        return clean(r["GROUP"])
    sub = (r.get("subsection") or "").lower()
    for key, pat in [("C.4", r"revert|withdraw|retract|rolled back|remov"),
                     ("C.2", r"wontfix|not.planned|by.design|not a bug|won't fix|will not"),
                     ("C.3", r"stale"),
                     ("C.6", r"paper|limitation|arxiv|undercut"),
                     ("C.5", r"negative|did not|no improvement|slower|regress|no benefit|no gain"),
                     ("C.1", r"unmerged|closed")]:
        if re.search(pat, sub):
            return key
    if urls(r) and "arxiv.org" in urls(r)[0]:
        return "C.6"
    return "C.1"


def _c_row(n, r):
    """One Section-C row. REASON-MAY-HAVE-EXPIRED and S3b ride in the reason cell so the
    'expired reason' pool stays greppable straight from the table."""
    reason = md_cell(g(r, "STATED_REASON", "REASON"))
    exp = clean(g(r, "REASON-MAY-HAVE-EXPIRED", "REASON_MAY_HAVE_EXPIRED"))
    s3b = clean(g(r, "S3B"))
    extra = []
    if exp:
        extra.append(f"**REASON-MAY-HAVE-EXPIRED:** {md_cell(exp)}")
    if s3b:
        extra.append(f"**S3b:** {md_cell(s3b)}")
    tail = (" " + " ".join(extra)) if extra else ""
    return "| C{} | {} | {} | {} | {}{} | {} |".format(
        n, md_cell(g(r, "WHO")), md_cell(g(r, "ARTIFACT")), url_html(r), reason, tail, ev(r))


def section_c(recs):
    groups = OrderedDict((k, []) for k in C_GROUPS)
    other = []
    for r in recs:
        if r["section"] != "ABANDONED":
            continue
        gk = guess_group(r)
        (groups[gk] if gk in groups else other).append(r)
    parts, n, expired_rows = [], 0, []
    for k, title in C_GROUPS.items():
        rs = groups[k]
        parts.append(f"\n### {k} {title}  ({len(rs)} items)\n")
        if not rs:
            parts.append("_No genuinely-evidenced item found in this subsection._\n")
            continue
        parts.append("| # | Who tried | Artifact | URL | STATED REASON (verbatim) | Evidence |")
        parts.append("|---|---|---|---|---|---|")
        for r in rs:
            n += 1
            parts.append(_c_row(n, r))
            if clean(g(r, "REASON-MAY-HAVE-EXPIRED", "REASON_MAY_HAVE_EXPIRED")):
                expired_rows.append((n, r))
        parts.append("")
    if other:
        parts.append(f"\n### C.7 Other attempted-and-abandoned  ({len(other)} items)\n")
        parts.append("| # | Who tried | Artifact | URL | STATED REASON (verbatim) | Evidence |")
        parts.append("|---|---|---|---|---|---|")
        for r in other:
            n += 1
            parts.append(_c_row(n, r))
            if clean(g(r, "REASON-MAY-HAVE-EXPIRED", "REASON_MAY_HAVE_EXPIRED")):
                expired_rows.append((n, r))
    exp = ("\n**Items whose stated reason may have expired** (the constraint the abandonment "
           "depended on has since changed — highest-priority pool):\n\n")
    if expired_rows:
        for num, r in expired_rows:
            exp += (f"- **C{num}** {md_cell(g(r,'ARTIFACT'))} — "
                    f"{md_cell(g(r,'REASON-MAY-HAVE-EXPIRED','REASON_MAY_HAVE_EXPIRED'))} "
                    f"({url_html(r)})\n")
    else:
        exp += "_None identified._\n"
    parts.insert(0, exp)
    return "\n".join(parts), n


D_GROUPS = OrderedDict([
    ("D.1", "Requires >1 GPU"),
    ("D.2", "Requires >96 GB VRAM"),
    ("D.3", "Requires multi-node / NVLink / InfiniBand"),
    ("D.4", "Requires a datacenter storage fabric or distributed service"),
    ("D.5", "Borderline (published multi-GPU but with a stated single-GPU mode)"),
])


def guess_d(r):
    sub = (r.get("subsection") or "")
    for k in D_GROUPS:
        if sub.startswith(k):
            return k
    txt = " ".join([sub, clean(g(r, "HARDWARE_QUOTE"))[:400]]).lower()
    if re.search(r"nvlink|infiniband|multi-node|multinode|rdma|nvswitch|roce", txt):
        return "D.3"
    if re.search(r"storage fabric|s3|gcs|redis|etcd|distributed (store|service|cluster)|object store", txt):
        return "D.4"
    if re.search(r"\b(8|16|32|64)\s*[x×]\s*(h100|a100|h200|b200|gb200|v100|rtx)", txt):
        return "D.2"
    if re.search(r"single.gpu mode|one gpu|1 gpu|single gpu", txt):
        return "D.5"
    return "D.1"


def section_d(recs):
    groups = OrderedDict((k, []) for k in D_GROUPS)
    for r in recs:
        if r["section"] != "HARDWARE-RULED-OUT":
            continue
        groups[guess_d(r)].append(r)
    parts, n = [], 0
    for k, title in D_GROUPS.items():
        rs = groups[k]
        parts.append(f"\n### {k} {title}  ({len(rs)} items)\n")
        if not rs:
            parts.append("_No genuinely-evidenced item found in this subsection._\n")
            continue
        parts.append("| # | Who | Artifact | URL | HARDWARE QUOTE (verbatim, complete) | Evidence |")
        parts.append("|---|---|---|---|---|---|")
        for r in rs:
            n += 1
            exp = clean(g(r, "REASON-MAY-HAVE-EXPIRED", "REASON_MAY_HAVE_EXPIRED"))
            s3b = clean(g(r, "S3B"))
            tail = ""
            if exp:
                tail += f" **REASON-MAY-HAVE-EXPIRED:** {md_cell(exp)}"
            if s3b:
                tail += f" **S3b:** {md_cell(s3b)}"
            parts.append("| D{} | {} | {} | {} | {}{} | {} |".format(
                n, md_cell(g(r, "WHO")), md_cell(g(r, "ARTIFACT")), url_html(r),
                md_cell(g(r, "HARDWARE_QUOTE")), tail, ev(r)))
        parts.append("")
    return "\n".join(parts), n


if __name__ == "__main__":
    recs = load()
    recs, dropped, log = apply_corrections(recs)
    a, na = section_a(recs)
    b, nb = section_b(recs)
    c, nc = section_c(recs)
    d, nd = section_d(recs)
    e, ne = section_e(recs)
    doc = (HEADER +
           "\n---\n\n## A. CLOSED cells (someone shipped or published a working answer)\n\n" + a +
           "\n\n---\n\n## B. OPEN cells (explicitly unsolved, with evidence someone is STILL asking)"
           "\n\n" + b +
           "\n\n---\n\n## C. ATTEMPTED-AND-ABANDONED cells\n" + c +
           "\n\n---\n\n## D. HARDWARE-RULED-OUT\n" + d +
           "\n\n---\n\n## E. NEVER-DISCUSSED (negative evidence)\n\n"
           "_Gaps the engine itself admits to, for which no human discussion could be found. "
           "Each row carries the exact queries tried and a confidence that the absence is real. "
           "Read the absence claim before treating any row as unoccupied ground._\n\n" + e +
           "\n\n---\n\n## Verification correction log\n\n" +
           ("\n".join("- " + x for x in log) if log else "_No corrections applied._") +
           "\n\n## Items dropped under the no-verbatim-quote rule\n\n" +
           ("\n".join(f"- {i} — {w}" for i, w in dropped) if dropped else "_None._") + "\n")
    open(OUT, "w", encoding="utf-8").write(doc)
    print(f"A={na} B={nb} C={nc} D={nd} E={ne}  ->  {OUT}")
