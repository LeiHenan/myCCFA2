import subprocess, sys
repos = sys.argv[1:]
for r in repos:
    got=None
    for br in ("main","master"):
        for fn in ("README.md",):
            u=f"https://raw.githubusercontent.com/{r}/{br}/{fn}"
            out=subprocess.run(["python3","fetch.py","get",u],capture_output=True,text=True,timeout=180,cwd="/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
            t=out.stdout
            if t and "404" not in t[:200] and len(t)>200:
                got=(u,t); break
        if got: break
    if not got:
        print(f"=========== {r} :: NO README FOUND"); continue
    u,t=got
    print(f"=========== {r} :: {u} :: len={len(t)}")
    print(t[:2500])
