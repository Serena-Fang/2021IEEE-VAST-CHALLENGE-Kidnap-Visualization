# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:26:35 2021

@author: 81916
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import os
import json
data = []
newspaper_name =  os.listdir('./News Articles')
for n in newspaper_name:
    articles_name = os.listdir('./News Articles/{}'.format(n))
    for art in articles_name:
        f = open('./News Articles/{}/{}'.format(n,art),encoding = 'cp1252')
        text = f.read()
        text = "".join([s for s in text.splitlines(True) if s.strip()])
        data.append(text)
        f.close()
#%%
# -------------Let’s prepare data for training our doc2vec model----------------------

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
#%%
# -------------Lets start training our model----------------------
'''
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

'''
'''
Note: dm defines the training algorithm. If dm=1 means‘distributed memory’(PV-DM) and dm =0 means‘distributed bag of words’(PV-DBOW). 
Distributed Memory model preserves the word order in a document whereas Distributed Bag of words just uses the bag of words approach, 
which doesn’t preserve any word order.
'''
'''
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

#model.save("d2v.model")
#print("Model Saved")

# -------------Lets play with it----------------------
#from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load("d2v.model")
# to find the vector of a document which is not in training data
test_data = word_tokenize(data[0].lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)

# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])
'''
#%%word2vec
from gensim.models import Word2Vec
#from gensim.models.word2vec import LineSentence
import string
import re 
word_data = [re.split(r' |\n',w) for w in data ]
i = 0
for i in range(len(word_data)):
    word_data[i] =  [x.strip() for x in word_data[i] if x.strip()!='']
    for j in range(len(word_data[i])):
        for c in string.punctuation:
            word_data[i][j] = word_data[i][j].replace(c,'').lower()
        
#%%
model = Word2Vec(word_data, sg=1,size=100, window=10, min_count=0, workers=15,sample=1e-3)
model.save('w2v.model')
model[word_data[0][0]]
model[word_data[1][0]]
model.most_similar(word_data[0][0],topn=10)
#%%

from cvxopt import matrix, solvers
import numpy as np

# 加载训练好的词向量
def load_embedding():
    
    return model


# 计算两个向量的欧式距离
def get_word_embedding_distance(emb1, emb2):
    if (len(emb1) != len(emb2)):
        print('error input,x and y is not in the same space')
        return
    result1 = 0.0
    for i in range(len(emb1)):
        result1 += (emb1[i] - emb2[i]) * (emb1[i] - emb2[i])
    distance = result1 ** 0.5
    return distance

# 把list转成map，key为list中的元素，value为在原句中出现的次数
def word_count(words):
    word_map = {}
    for word in words:
        if word in word_map.keys():
            word_map[word] += 1.0
        else:
            word_map[word] = 1.0
    return word_map

# TODO: 编写WMD函数来计算两个句子之间的相似度

def WMD (sent1, sent2):
    """
    这是主要的函数模块。参数sent1是第一个句子， 参数sent2是第二个句子，可以认为没有经过分词。
    在英文里，用空格作为分词符号。
    
    在实现WMD算法的时候，需要用到LP Solver用来解决Transportation proboem. 请使用http://cvxopt.org/examples/tutorial/lp.html
    也可以参考blog： https://scaron.info/blog/linear-programming-in-python-with-cvxopt.html
    
    需要做的事情为：
    
    1. 对句子做分词： 调用 .split() 函数即可
    2. 获取每个单词的词向量。这需要读取文件之后构建embedding matrix. 
    3. 构建lp问题，并用solver解决
    
    可以自行定义其他的函数，但务必不要改写WMD函数名。测试时保证WMD函数能够正确运行。
    """
    
    # 分词
    words1 = sent1.split()
    words2 = sent2.split()

    # 去重、统计词频
    word_map1 = word_count(words1)
    word_map2 = word_count(words2)

    # 将分词转为词向量
    word_embs1 = [word_emds[word] for word in word_map1.keys()]
    word_embs2 = [word_emds[word] for word in word_map2.keys()]

    # 准备遍历
    len1 = len(word_embs1)
    len2 = len(word_embs2)
    # 获得两个句子之间的词汇距离ci,j,存储在二维数组cc中
    c_ij = []
    for i in range(len1):
        for j in range(len2):
            c_ij.append(get_word_embedding_distance(word_embs1[i], word_embs2[j]))

    # 设计A矩阵
    a = []
    #   句子1对应到句子2部分
    for i in range(len1):
        line_ele = []
        for ii in range(i * len2):
            line_ele.append(0.0)
        for j in range(len2):
            line_ele.append(1.0)
        for ii in range((len1 - i - 1) * len2):
            line_ele.append(0.0)
        a.append(line_ele)

    #   句子2对应到句子1
    for i in range(len2):
        line_ele = []
        for ii in range(len1):
            for jj in range(i):
                line_ele.append(0.0)
            line_ele.append(1.0)
            for jj in range(0, len2 - i - 1):
                line_ele.append(0.0)
        a.append(line_ele)

    # 获得出入量之和 这部分逻辑是跟A的设计连在一起的
    b = [ele / len(words1) for ele in list(word_map1.values())] + \
        [ele / len(words2) for ele in list(word_map2.values())]

    # 列出线性规划问题
    A_matrix = matrix(a).trans()
    b_matrix = matrix(b)
    c_matrix = matrix(c_ij)
    num_of_T = len(c_ij)
    G = matrix(-np.eye(num_of_T))
    h = matrix(np.zeros(num_of_T))

#     求解器求解，注意这里必须指定solver，否则会报错，蛋疼
    sol = solvers.lp(c_matrix, G, h, A=A_matrix, b=b_matrix, solver='glpk')
    return sol['primal objective']


# 读取glove的预训练词向量
word_emds = load_embedding()
print("读取完成")

sent1 = " ".join(word_data[0])
sent2 = " ".join(word_data[0])
sent3 = " ".join(word_data[2])
print(WMD(sent1, sent2))
print(WMD(sent1, sent3))
