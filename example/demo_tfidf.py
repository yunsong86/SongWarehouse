#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time     : 2018/10/23 22:25
# @Author   : YunSong
# @File     : demo_tfidf.py
# @Software : PyCharm
# @version  : python3
"""

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["我 来到  北京 清华大学",
          "他 来到 了 网易 杭研  清华大学",
          " 硕士 毕业 毕业  与 中国 科学院 清华大学",
          "  科学院 ",
          ]

print("\n============================训练tfidf模型 ============================")

tfidfVectorizer = TfidfVectorizer()
vec = tfidfVectorizer.fit_transform(corpus)
word = tfidfVectorizer.get_feature_names()  # 获取词袋模型中的所有词语
weight = vec.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
    for j in range(len(word)):
        print(word[j], weight[i][j])

print("============================保存tfidf模型 ============================")

import pickle

with open('tfidf.model', 'wb') as file:
    pickle.dump(tfidfVectorizer, file)

with open('tfidf.model', 'rb') as file:
    tfidfVectorizer = pickle.load(file)

print("\n============================tfidf 模型预测 ============================")
print("我 来到 北京 清华大学 运送")

tfidf_vec = tfidfVectorizer.transform(["我 来到 北京 清华大学", "我  北京 清华大学"])
print(tfidf_vec)
print(tfidf_vec.data)
print(tfidf_vec[0])

voc = dict((i, w) for w, i in tfidfVectorizer.vocabulary_.items())

for idx in tfidf_vec[0].indices:
    print("%s:%s " % (idx, voc[idx]))
print()
features = zip([voc[j] for j in tfidf_vec.indices], tfidf_vec.getrow(0).data)
for f in features:
    print(f)
print()

features = zip([voc[j] for j in tfidf_vec.indices], tfidf_vec[0].data)
for f in features:
    print(f)
print()
features = zip([j for j in tfidf_vec.indices], [voc[j] for j in tfidf_vec.indices], tfidf_vec[0].data)
for f in features:
    print(f)
