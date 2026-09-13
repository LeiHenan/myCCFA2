import re, sys
src = open(sys.argv[1]).read()
for m in re.finditer(r"parser\.add_argument\((.*?)\)\n", src, re.S):
    block = " ".join(m.group(1).split())
    names = re.findall(r'"(--[a-z0-9-]+)"', block)
    if not names:
        continue
    d = re.search(r"default=([^,)]+)", block)
    h = re.search(r'help="(.*?)"', block)
    dv = d.group(1) if d else "-"
    hv = h.group(1)[:55] if h else ""
    print("%-30s default=%-20s %s" % (names[0], dv, hv))
