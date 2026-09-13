import subprocess, sys, re
repos = sys.argv[1:]
for r in repos:
    url = "https://github.com/"+r
    try:
        html = subprocess.run(["python3","fetch.py","raw",url],capture_output=True,text=True,timeout=180,cwd="/Users/leihenan/Desktop/myProject/evidence/accel-gapmap").stdout
    except Exception as e:
        print(f"{r}\tERR\t{e}"); continue
    arch = re.search(r'"isArchived":(\w+)', html)
    ban  = re.search(r'archived by the owner on ([^"<]*)', html)
    disp = re.search(r'"disabled":(\w+)', html)
    print(f"{r}\tarch={arch.group(1) if arch else '?'}\tbanner={ban.group(1) if ban else '-'}")
