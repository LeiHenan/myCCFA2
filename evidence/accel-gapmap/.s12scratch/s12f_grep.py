import subprocess, sys, re
pat = re.compile(r"(no longer|not.{0,20}maintain|deprecat|archiv|sunset|discontinu|unmaintain|end of life|EOL|stepping back|step back|abandon|ceased|superseded|not actively)", re.I)
for r in sys.argv[1:]:
    got=None
    for br in ("main","master"):
        u=f"https://raw.githubusercontent.com/{r}/{br}/README.md"
        out=subprocess.run(["python3","fetch.py","get",u],capture_output=True,text=True,timeout=180,cwd="/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
        t=out.stdout
        if t and not t.startswith("404") and len(t)>300:
            got=(u,t); break
    if not got:
        print(f"### {r}: NO README"); continue
    u,t=got
    hits=[(i+1,l.strip()) for i,l in enumerate(t.splitlines()) if pat.search(l)]
    print(f"### {r} :: {u} :: {len(hits)} hits")
    for i,l in hits[:25]:
        print(f"   L{i}: {l[:400]}")
