import re,html,sys
def txt(p):
    h=open(p,encoding='utf-8',errors='ignore').read()
    h=re.sub(r'(?is)<(script|style|svg|noscript)[^>]*>.*?</\1>',' ',h)
    h=re.sub(r'(?is)<br\s*/?>','\n',h)
    h=re.sub(r'(?is)</(p|div|li|h[1-6]|tr|pre|section)>','\n',h)
    t=re.sub(r'(?s)<[^>]+>',' ',h)
    t=html.unescape(t)
    t=re.sub(r'[ \t\xa0]+',' ',t)
    t=re.sub(r'\n[ \t]*','\n',t)
    t=re.sub(r'\n{3,}','\n\n',t)
    return t
if __name__=='__main__':
    print(txt(sys.argv[1]))
