import re,sys,json,html,subprocess
def get(u):
    return subprocess.run(["curl","-sL","--max-time","50","-A","Mozilla/5.0",u],capture_output=True,text=True).stdout
h=get(sys.argv[1])
# find all "body":"..." string values
bodies=re.findall(r'"body":"((?:[^"\\]|\\.)*)"',h)
seen=set()
for b in bodies:
    try: t=json.loads('"'+b+'"')
    except Exception: t=b
    t=t.strip()
    if len(t)<15 or t in seen: continue
    seen.add(t)
    print('-----')
    print(t[:2500])
