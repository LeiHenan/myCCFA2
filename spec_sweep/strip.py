import re,sys,html
def strip(f):
    t=open(f,encoding='utf-8',errors='ignore').read()
    t=re.sub(r'(?is)<(script|style|svg|head)[^>]*>.*?</\1>',' ',t)
    # keep markdown-ish bodies
    t=re.sub(r'(?is)<br\s*/?>','\n',t)
    t=re.sub(r'(?is)</(p|div|li|h[1-6]|tr|pre)>','\n',t)
    t=re.sub(r'(?s)<[^>]+>',' ',t)
    t=html.unescape(t)
    t=re.sub(r'[ \t\xa0]+',' ',t)
    t=re.sub(r'\n\s*\n+','\n',t)
    return t
if __name__=='__main__':
    print(strip(sys.argv[1]))
