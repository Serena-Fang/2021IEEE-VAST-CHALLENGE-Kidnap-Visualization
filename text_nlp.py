# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:45:30 2021

@author: 81916
"""
import json
#from gensim.models import LdaModel
with open('news_info_10.json', encoding='utf-8') as f:
    text = f.read()
    text = json.loads(text)
word_dict = {'name':'overall','children':[]} 
for k,v in text.items():
    #if i > 0:
        #break
    pro_all = 0 
    for t in text[k]['topics']:
        pro = 0
        t.append([])
        word = t[1].split('+')
        for w in word:
            pro_dict = {}
            p = w.split('*')
            pro_dict['name'] = p[1].split('"')[1].lower()
            pro_dict['pro'] = float(p[0])
            t[2].append(pro_dict)
            pro += float(p[0])
            pro_all += float(p[0])
        t.append(pro)
    for t in text[k]['topics']:
        t[3] = round(t[3]/pro_all,2)
    #i += 1
b = json.dumps(text)
f2 = open('news_info.json', 'w')
f2.write(b)
f2.close()
