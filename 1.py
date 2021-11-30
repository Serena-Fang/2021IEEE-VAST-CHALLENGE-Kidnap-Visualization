import os
import nltk
import spacy
import json 
from textblob import TextBlob
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nlp = spacy.load("en_core_web_sm")
news_info = {}
newspaper_name =  os.listdir('./News Articles')
ti_dict = {}
#newspaper_name = ['All News Today']
month = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,
         'August':8,'September':9,'October':10,'November':11,'December':12}
for n in newspaper_name:
    articles_name = os.listdir('./News Articles/{}'.format(n))
    #articles_name = ['77.txt']
    for art in articles_name:
        info = {}
        
        info['source'] = n
        info['title'] = 'None'
        info['loc_city'] = 'None'
        info['loc_con'] = 'None'
        f = open('./News Articles/{}/{}'.format(n,art),encoding = 'cp1252')
        text = f.read()
        text = "".join([s for s in text.splitlines(True) if s.strip()])
        ti = text.split('\n')[1].split(' ')[1:]
        f.close()
#print(text[24])
        blob = TextBlob(text)
        info['polarity'] = blob.sentiment.polarity
        info['subjectivity'] = blob.sentiment.subjectivity
        tokenized = nltk.word_tokenize(text)
        p_index = tokenized.index('PUBLISHED')
        if 'TITLE' in tokenized:
           t_index = tokenized.index('TITLE')        
           tag = pos_tag(word_tokenize(' '.join(ti)), tagset='universal')
           title_nonu = [word[0] for word in tag if word[1] == 'NOUN']
           info['title'] = title_nonu
        if '/' in tokenized[p_index+2]:
            info['published'] = tokenized[p_index+2]
            info['timestamp'] = int(info['published'].split("/")[0]) * 366\
            + int(info['published'].split("/")[1]) * 31\
            + int(info['published'].split("/")[2])
        else :
            info['published'] = tokenized[p_index+2:p_index+5]
            info['timestamp']  = int(info['published'][0])\
            + month[info['published'][1]] * 31\
            + int(info['published'][2])*366
        if 'LOCATION' in tokenized:
            l_index = tokenized.index('LOCATION')
            info['loc_city'] = tokenized[l_index+2]
            info['loc_con'] = tokenized[l_index+4]
        doc = nlp(text)
        noun_list = [chunk.text for chunk in doc.noun_chunks]
       
        info['noun_list'] = noun_list[7:]
        news_info['{}_{}'.format(n,art)] = info
        if text.split('\n')[1] in ti_dict:
            ti_dict[text.split('\n')[1]] += 1
        else :
            ti_dict[text.split('\n')[1]] = 1
        print('{}_{}'.format(n,art))


b = json.dumps(news_info)
f2 = open('news_info.json', 'w')
f2.write(b)
f2.close()

#%%
word_index = {}
for k,v in news_info.items():
    for key in v['title']:
        if key in word_index:
            word_index[key].append(k)
        else:
            word_index[key] = [k]
word_index['MINOR'].extend(word_index["'MINOR"] )           
#c = json.dumps(word_index)
#f3 = open('word_index.json', 'w')
#f3.write(c)
#f3.close()    
