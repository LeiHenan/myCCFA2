#!/usr/bin/env python3
"""Mechanical verbatim-quote checker + S*.md parser for the accel gap map.

  python3 verify.py quote <url> "<quote text>"      # MATCH / NEAR / MISMATCH
  python3 verify.py parse                            # S*.md -> _index.jsonl ; prints counts
  python3 verify.py audit <sample_n> [seed]          # blind sample of Section C reason fields

Matching is whitespace-normalised and HTML-entity-normalised, because arXiv/GitHub rendering
introduces line breaks, non-breaking spaces and ligature artefacts that are not author edits.
A MISMATCH here is not automatically a defect: it may be a composite quote. The verifier's job
is to distinguish "rendering artefact" from "text that is not on the page".
"""
import json, os, random, re, sys, html, difflib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch import curl, strip_html, gh  # noqa: E402

ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)


def norm(s):
    if s is None:
        return ""
    s = html.unescape(s)
    s = s.translate(ZW)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014", "--").replace("\u2013", "-")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def corpus(url):
    """Return (variant_name, normalised_text) pairs for a URL."""
    out = []
    if "github.com" in url:
        try:
            g = gh(url)
            out.append(("gh-extract", norm(g)))
        except Exception as e:
            out.append(("gh-extract-ERR", norm(str(e))))
    try:
        out.append(("page-text", norm(strip_html(curl(url)))))
    except Exception as e:
        out.append(("page-text-ERR", norm(str(e))))
    if "arxiv.org/abs/" in url:
        aid = re.search(r"abs/([0-9.]+)", url).group(1)
        for alt in (f"https://arxiv.org/html/{aid}v1", f"https://arxiv.org/html/{aid}",
                    f"https://ar5iv.labs.arxiv.org/html/{aid}"):
            try:
                t = norm(strip_html(curl(alt)))
                if len(t) > 2000:
                    out.append(("html:" + alt.rsplit("/", 2)[-1][:24], t))
            except Exception:
                pass
    return out


def check(url, quote, verbose=True):
    q = norm(quote)
    if not q:
        return {"verdict": "EMPTY", "best": None}
    best = {"verdict": "MISMATCH", "variant": None, "ratio": 0.0}
    for name, text in corpus(url):
        if not text:
            continue
        if q in text:
            return {"verdict": "MATCH", "variant": name, "ratio": 1.0}
        # try the quote cut at an ellipsis (composite marker)
        parts = [p for p in re.split(r"\s*(?:\.\.\.|\[\.\.\.\]|…)\s*", q) if len(p) > 12]
        if len(parts) > 1 and all(p in text for p in parts):
            return {"verdict": "MATCH-FRAGMENTS", "variant": name, "ratio": 1.0,
                    "n_fragments": len(parts)}
        # approximate: longest matching window
        sm = difflib.SequenceMatcher(None, q, text, autojunk=False)
        m = sm.find_longest_match(0, len(q), 0, len(text))
        ratio = m.size / max(1, len(q))
        if ratio > best["ratio"]:
            best = {"verdict": "NEAR" if ratio > 0.75 else "MISMATCH",
                    "variant": name, "ratio": round(ratio, 3),
                    "missing_tail": q[m.a + m.size:][:160],
                    "matched_head": q[:80]}
    return best


BLOCK = re.compile(r"^### ([ABCDE]\d+\.\d+)\s*\|\s*(CLOSED|OPEN|ABANDONED|HARDWARE-RULED-OUT|NEVER-DISCUSSED)\s*(.*)$")
FIELD = re.compile(r"^([A-Z][A-Z0-9 _\(\)/\.\->\+]*?):\s?(.*)$")


def parse(verbose=True):
    recs = []
    for fn in sorted(os.listdir(HERE)):
        if not re.match(r"^[SE]\d+\.md$", fn):
            continue
        cur = None
        for line in open(os.path.join(HERE, fn), encoding="utf-8", errors="replace"):
            m = BLOCK.match(line.rstrip("\n"))
            if m:
                if cur:
                    recs.append(cur)
                cur = {"id": m.group(1), "section": m.group(2), "subsection": m.group(3).strip(" ()"),
                       "file": fn}
                continue
            if cur is None:
                continue
            f = FIELD.match(line)
            if f:
                k = f.group(1).strip().replace(" ", "_")
                cur[k] = (cur.get(k, "") + " " + f.group(2).strip()).strip()
            elif line.strip() and not line.startswith("#"):
                cur["_extra"] = (cur.get("_extra", "") + " " + line.strip()).strip()
        if cur:
            recs.append(cur)
    with open(os.path.join(HERE, "_index.jsonl"), "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs


def url_of(rec):
    for k in ("URL", "URLS"):
        if rec.get(k):
            return rec[k].split(";")[0].strip()
    return None


def quote_of(rec):
    if rec["section"] == "OPEN":
        return rec.get("ASKING_QUOTE") or rec.get("QUOTE")
    if rec["section"] == "ABANDONED":
        return rec.get("STATED_REASON")
    if rec["section"] == "HARDWARE-RULED-OUT":
        return rec.get("HARDWARE_QUOTE")
    if rec["section"] == "NEVER-DISCUSSED":
        return rec.get("GAP")
    return rec.get("QUOTE")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "quote":
        r = check(sys.argv[2], sys.argv[3])
        print(json.dumps(r, ensure_ascii=False))
    elif cmd == "parse":
        recs = parse()
        from collections import Counter
        print(json.dumps(Counter([r["section"] for r in recs]), indent=1))
        print(json.dumps(Counter([r["file"] for r in recs]), indent=1))
    elif cmd == "audit":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20260914
        recs = [r for r in parse(verbose=False)
                if r["section"] == "ABANDONED" and quote_of(r) and url_of(r)]
        random.seed(seed)
        sample = random.sample(recs, min(n, len(recs)))
        print(f"# blind sample n={len(sample)} of {len(recs)} ABANDONED rows with a reason+URL (seed {seed})")
        tally = {"MATCH": 0, "MATCH-FRAGMENTS": 0, "NEAR": 0, "MISMATCH": 0, "EMPTY": 0}
        for r in sample:
            res = check(url_of(r), quote_of(r))
            tally[res["verdict"]] = tally.get(res["verdict"], 0) + 1
            print(f"{res['verdict']:16s} {res.get('ratio')!s:6s} {r['id']:7s} {r['file']:6s} "
                  f"{url_of(r)[:78]}")
            if res["verdict"] in ("NEAR", "MISMATCH"):
                print(f"    missing_tail: {res.get('missing_tail','')[:150]!r}")
        print("TALLY", json.dumps(tally))
        print(f"VERBATIM RATE = {(tally['MATCH']+tally['MATCH-FRAGMENTS'])}/{len(sample)}"
              f" = {100*(tally['MATCH']+tally['MATCH-FRAGMENTS'])/max(1,len(sample)):.0f}%")
    else:
        sys.exit("unknown cmd")
