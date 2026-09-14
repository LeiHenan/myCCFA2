import re,sys
path=sys.argv[1]; pat=sys.argv[2]; width=int(sys.argv[3]) if len(sys.argv)>3 else 260
t=open(path,encoding='utf-8',errors='ignore').read()
t=re.sub(r'\s+',' ',t)
n=0
for m in re.finditer(pat,t,re.I):
    s=max(0,m.start()-width); e=min(len(t),m.end()+width)
    print('---',m.start())
    print(t[s:e])
    n+=1
    if n>=int(sys.argv[4]) if len(sys.argv)>4 else n>=8: break
