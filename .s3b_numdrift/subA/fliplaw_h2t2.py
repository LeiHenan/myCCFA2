import re,html,sys
h=open(sys.argv[1],encoding='utf-8',errors='replace').read()
h=re.sub(r'(?is)<(script|style).*?</\1>',' ',h)
# keep math alttext
h=re.sub(r'(?is)<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>',lambda m:' $'+html.unescape(m.group(1))+'$ ',h)
h=re.sub(r'(?is)<math.*?</math>',' [MATH] ',h)
h=re.sub(r'(?s)<[^>]+>',' ',h)
h=html.unescape(h)
h=re.sub(r'[ \t]+',' ',h)
h=re.sub(r'\n\s*\n+','\n',h)
open(sys.argv[2],'w',encoding='utf-8').write(h)
