import urllib.parse
import sys, json, urllib.request, time
def api(u):
    for a in range(4):
        try:
            req = urllib.request.Request(u, headers={'User-Agent':'research','Accept':'application/vnd.github+json'})
            return json.load(urllib.request.urlopen(req, timeout=45))
        except Exception as e:
            time.sleep(10)
    print("FAIL", u); return None
for q in sys.argv[1:]:
    u = "https://api.github.com/search/issues?q=" + urllib.parse.quote(q) + "&per_page=50&sort=created&order=desc"
    d = api(u)
    if not d: continue
    print(f"\n##### QUERY: {q}  total={d.get('total_count')}")
    for it in d.get('items',[]):
        print(f"{it['number']} {it['state']}/{it.get('state_reason')} lab={[l['name'] for l in it.get('labels',[])]} | {it['title'][:140]} | {it['html_url']}")
    time.sleep(7)
