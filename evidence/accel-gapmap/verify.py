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
from collections import Counter

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


RAW_SUFFIXES = (".py", ".txt", ".md", ".json", ".cu", ".cuh", ".h", ".cpp", ".yaml",
                 ".yml", ".toml", ".cfg", ".rst", ".sh")


def corpus(url):
    """Return (variant_name, normalised_text) pairs for a URL.

    The raw (un-HTML-stripped) text is ALWAYS included for source-like URLs: strip_html treats
    any `<...>` span as a tag, which silently deletes real source between a stray `<` and a later
    `>` (a verifier measured ~14 KB lost from vLLM's CMakeLists.txt). Without this variant the
    checker reports false MISMATCHes on exactly the source quotes that matter most.
    """
    out = []
    blob = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+?)(?:#.*)?$", url)
    if blob:
        o, rp, ref, path = blob.groups()
        try:
            rawt = curl(f"https://raw.githubusercontent.com/{o}/{rp}/{ref}/{path}")
            if rawt and len(rawt) > 50:
                out.append(("raw-of-blob", norm(rawt)))
        except Exception:
            pass
    raw = None
    try:
        raw = curl(url)
    except Exception:
        raw = None
    if raw and (url.split("?")[0].endswith(RAW_SUFFIXES)
                or "raw.githubusercontent.com" in url
                or "raw.githubusercontent.com" in url):
        out.append(("raw-text", norm(raw)))
    if "github.com" in url:
        try:
            g = gh(url)
            out.append(("gh-extract", norm(g)))
        except Exception as e:
            out.append(("gh-extract-ERR", norm(str(e))))
    if raw is not None:
        try:
            stripped = norm(strip_html(raw))
            # only use the stripped variant if it did not lose most of the document
            if raw and len(stripped) > 0.5 * len(norm(raw)):
                out.append(("page-text", stripped))
            else:
                out.append(("page-text-raw", norm(raw)))
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


TRAIL = re.compile(r'^(\s*"[^"]*")\s*(?:\(|\[|--|\u2014|$)')

def first_quoted_span(s):
    """If the value starts with a double-quoted span, return just that span. Researchers often
    append a locator or a note after the closing quote; the quote itself is what must match."""
    s = s.strip()
    if not s.startswith('"'):
        return s
    end = s.find('"', 1)
    while end != -1:
        if end + 1 >= len(s) or s[end + 1] in ' \t.,;:)]}':
            return s[:end + 1]
        end = s.find('"', end + 1)
    return s


def strip_wrapping_quotes(s):
    """Block-format fields wrap the quote in double quotes or backticks; strip one layer so the
    checker tests the quoted text, not the delimiters."""
    s = s.strip()
    if len(s) >= 2 and s[0] in '"\u201c`' and s[-1] in '"\u201d`':
        s = s[1:-1]
    return s


def check(url, quote, verbose=True):
    q = norm(strip_wrapping_quotes(first_quoted_span(quote)))
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


BLOCK = re.compile(r"^###\s+([ABCDE][0-9.]*?)\s*\|\s*(CLOSED|OPEN|ABANDONED|HARDWARE-RULED-OUT|NEVER-DISCUSSED)\s*(.*)$")
FIELD = re.compile(r"^([A-Z][A-Z0-9 _\.\->\+/]*(?:\s*\([^)]*\))?)\s*:\s?(.*)$")


def parse(verbose=True):
    recs = []
    for fn in sorted(os.listdir(HERE)):
        if not re.match(r"^[SE]\d+\.md$", fn):
            continue
        cur = None
        last_key = None
        open_key = None      # a quote-bearing field whose closing quote has not been reached
        for line in open(os.path.join(HERE, fn), encoding="utf-8", errors="replace"):
            m = BLOCK.match(line.rstrip("\n"))
            if m:
                if cur:
                    recs.append(cur)
                stem = fn[:-3]
                cur = {"id": f"{stem}:{m.group(1)}", "section": m.group(2),
                       "subsection": m.group(3).strip(" ()"), "file": fn}
                last_key = open_key = None
                continue
            if cur is None:
                continue
            f = FIELD.match(line)
            if f and open_key is None:
                k = f.group(1).strip().replace(" ", "_")
                cur[k] = (cur.get(k, "") + " " + f.group(2).strip()).strip()
                last_key = k
                v = cur[k]
                # Many engine quotes are multi-line source snippets. If the value opens a
                # double quote that never closes on this line, keep consuming continuation
                # lines until the quotes balance, so the row is not truncated mid-sentence.
                if v.startswith('"') and v.count('"') % 2 == 1:
                    open_key = k
            elif open_key is not None:
                cur[open_key] = (cur[open_key] + "\n" + line.rstrip()).strip()
                if cur[open_key].count('"') % 2 == 0:
                    open_key = None
            elif line.strip() and not line.startswith("#"):
                cur["_rawblock"] = (cur.get("_rawblock", "") + "\n" + line.rstrip()).strip()
                cur["_extra"] = (cur.get("_extra", "") + " " + line.strip()).strip()
        if cur:
            recs.append(cur)
    with open(os.path.join(HERE, "_index.jsonl"), "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs


URLRX = re.compile(r"https?://[^\s;,|)\]]+")


def url_of(rec):
    for k in ("URL", "URLS"):
        if rec.get(k):
            m = URLRX.search(rec[k])
            if m:
                return m.group(0).rstrip(".")
            return rec[k].split(";")[0].strip()
    # Section E blocks carry "SOURCE: <path> URL: <url>" on one line, so the URL lands
    # inside the SOURCE value; fall back to scanning every field for the first URL.
    for k in ("SOURCE", "ARTIFACT", "URLS", "_extra"):
        if rec.get(k):
            m = URLRX.search(rec[k])
            if m:
                return m.group(0).rstrip(".")
    for k, v in rec.items():
        if isinstance(v, str):
            m = URLRX.search(v)
            if m:
                return m.group(0).rstrip(".")
    return None


def quote_of(rec):
    """Return the item's claim-bearing quotation. Field names in the evidence files are
    inconsistently suffixed (e.g. `STATED_REASON_(verbatim,_arXiv_Comments_field)`), so match
    by prefix, longest-prefix-first, before falling back to any key containing QUOTE/REASON."""
    sec = rec["section"]
    if sec == "OPEN":
        pref = ("ASKING_QUOTE", "QUOTE")
    elif sec == "ABANDONED":
        pref = ("STATED_REASON", "REASON", "CLOSURE_REASON", "QUOTE")
    elif sec == "HARDWARE-RULED-OUT":
        pref = ("HARDWARE_QUOTE", "QUOTE")
    elif sec == "NEVER-DISCUSSED":
        pref = ("GAP", "QUOTE")
    else:  # CLOSED
        pref = ("QUOTE", "VERBATIM_QUOTE", "DOC_QUOTE", "STATUS_QUOTE")
    keys = list(rec.keys())
    for p in pref:
        for k in keys:
            if k == p and rec.get(k):
                return rec[k]
        for k in keys:
            if k.upper().startswith(p) and rec.get(k):
                return rec[k]
    for k in keys:
        if ("QUOTE" in k.upper() or "REASON" in k.upper()) and rec.get(k):
            return rec[k]
    return None

def checkall(only=None):
    """Mechanically re-locate EVERY claimed quote at its URL. Writes checkall.jsonl."""
    recs = parse(verbose=False)
    out = open(os.path.join(HERE, "checkall.jsonl"), "w", encoding="utf-8")
    tally = Counter()
    for i, r in enumerate(recs):
        if only and r["section"] != only:
            continue
        u, q = url_of(r), quote_of(r)
        if not u or not q:
            res = {"verdict": "NO-URL-OR-QUOTE"}
        else:
            try:
                res = check(u, q)
            except Exception as e:
                res = {"verdict": "FETCH-ERROR", "err": str(e)[:200]}
        res.update({"id": r["id"], "section": r["section"], "url": u,
                    "quote": (q or "")[:300], "file": r["file"]})
        tally[res["verdict"]] += 1
        out.write(json.dumps(res, ensure_ascii=False) + "\n")
        out.flush()
        if i % 10 == 0:
            print(f"[{i}/{len(recs)}] {dict(tally)}", flush=True)
    print("FINAL", json.dumps(tally), flush=True)
    return tally


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
    elif cmd == "checkall":
        checkall(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit("unknown cmd")

