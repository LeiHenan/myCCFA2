#!/usr/bin/env python3
"""From page dumps, extract the strongest 'why it was dropped' statements:
   - author/maintainer comments matching negative-result language
   - close events with state reason
   - label-add events for negative labels
"""
import re, glob, os, sys

KEY = re.compile(
    r"(clos(e|ing|ed)|withdraw|supersed|dropp|abandon|revert|not pursu|"
    r"won't|will not|no longer|not worth|no benefit|no measurable|"
    r"does not help|doesn't help|didn't help|by design|out of scope|"
    r"not planned|reopen|premature|deprioriti|decided against|"
    r"no plans to|not going to|regression|not merge|do not merge|"
    r"closed-as-slop|low effort|agent generated|invalid|stale)", re.I)


def blocks(txt):
    """Yield (author, date, body) for rendered + graphql comments."""
    for m in re.finditer(
            r"\[RENDERED BLOCK author=([^ ]+) date=([^\]]+)\]\n(.*?)(?=\n\[RENDERED|\n#####|\Z)",
            txt, re.S):
        yield m.group(1), m.group(2), m.group(3)
    for m in re.finditer(
            r"\[GRAPHQL COMMENT by ([^ ]+) assoc=([^ ]+) at ([^\]]+)\]\n(.*?)(?=\n\[GRAPHQL|\n#####|\n########|\Z)",
            txt, re.S):
        yield m.group(1), m.group(3), m.group(4)


def events(txt):
    out = []
    for m in re.finditer(r"\[GRAPHQL CLOSED by ([^ ]+) at ([^ \]]+) reason=([^\]]+)\]", txt):
        out.append(f"CLOSED by {m.group(1)} at {m.group(2)} reason={m.group(3)}")
    for m in re.finditer(r"\[GRAPHQL MERGED at ([^\]]+)\]", txt):
        out.append(f"MERGED at {m.group(1)}")
    # rendered label / close events
    for m in re.finditer(r"EV: ([^|]*(?:added|closed|reopened|merged)[^|]*\|[^|]*\|[^|]*)", txt):
        s = " ".join(m.group(1).split())
        if re.search(r"label|closed this|merged this|reopened", s, re.I):
            out.append("EV: " + s[:200])
    return out


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    for f in sorted(glob.glob("/Users/leihenan/Desktop/myProject/kv_discovery/pages/*.txt")):
        base = os.path.basename(f)[:-4]
        if filt and filt not in base:
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        hits = []
        for who, when, body in blocks(txt):
            if KEY.search(body):
                one = " ".join(body.split())
                hits.append((who, when, one))
        evs = events(txt)
        if not hits and not evs:
            continue
        print(f"\n{'='*90}\n### {base}\n{'='*90}")
        # metadata line
        for line in txt.split("\n")[:14]:
            if re.match(r"\s*(FETCHED|title|state|stateReason|closedTime|mergedTime|labels|number|author)", line):
                print("   " + line.strip())
        if evs:
            print("  EVENTS:")
            for e in dict.fromkeys(evs):
                print("   - " + e)
        for who, when, one in hits[-6:]:
            print(f"  [{who} @ {when}] {one[:520]}")


main()
