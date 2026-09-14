import json,re,sys
gens={
 'Volta':['V100','Volta'],
 'Turing':['T4 ','Turing'],
 'Ampere':['A100','RTX 3090','3090','A30','A10','Ampere'],
 'Ada':['RTX 4090','4090','L40S','L40','L4 ','Ada Lovelace','Ada'],
 'Hopper':['H100','H200','H800','GH200','Hopper'],
 'Blackwell':['B200','GB200','RTX 5090','5090','RTX 5080','5080','B100','Blackwell','GB10','DGX Spark'],
}
d=json.load(open(sys.argv[1]))
rows=[]
for pid,p in d.items():
    txt=(p['title'] or '')+' '+(p['abs'] or '')
    hits={g:[k for k in ks if re.search(re.escape(k),txt)] for g,ks in gens.items()}
    hits={g:v for g,v in hits.items() if v}
    if len(hits)>=2:
        rows.append((len(hits),pid,p,hits))
rows.sort(key=lambda r:-r[0])
print('papers mentioning >=2 generations:',len(rows),'of',len(d))
for n,pid,p,hits in rows[:45]:
    print(f"\n[{n} gens] {pid} {p['pub']}  {p['title']}")
    print('   hits:',{g:v[:3] for g,v in hits.items()})
