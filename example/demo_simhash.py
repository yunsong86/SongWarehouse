#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-10-24 上午9:21
# @Author  : YunSong
# @File    : demo_simhash.py
# @Software: PyCharm

from utils.simhash import *
from sklearn.feature_extraction.text import TfidfVectorizer


def example1():
    data = {
        1: 'How are you? I Am fine. blar blar blar blar blar Thanks.',
        2: 'How are you i am fine. blar blar blar blar blar than',
        3: 'This is simhash test.',
        4: 'How are you i am fine. blar blar blar blar blar thank1',
    }

    objs = [(str(k), Simhash(v)) for k, v in data.items()]
    index = SimhashIndex(objs, k=10)
    s1 = Simhash(u'How are you i am fine.ablar ablar xyz blar blar blar blar blar blar blar thank')
    dups = index.get_near_dups(s1)
    print(dups)

    index.delete('1', Simhash(data[1]))
    dups = index.get_near_dups(s1)
    print(dups)

    index.delete('1', Simhash(data[1]))
    dups = index.get_near_dups(s1)
    print(dups)
    index.add('1', Simhash(data[1]))
    dups = index.get_near_dups(s1)
    print(dups)

    index.add('1', Simhash(data[1]))
    dups = index.get_near_dups(s1)
    print(dups)


def example2():
    sh1 = Simhash(u'你好　世界！　　呼噜。')
    sh2 = Simhash(u'你好，世界　呼噜')
    print(sh1.distance(sh2))


def example3():
    data = [
        'How are you? I Am fine. blar blar blar blar blar Thanks.',
        'How are you i am fine. blar blar blar blar blar than',
        'This is simhash test.',
        'How are you i am fine. blar blar blar blar blar thank1'
    ]
    vec = TfidfVectorizer()
    D = vec.fit_transform(data)
    voc = dict((i, w) for w, i in vec.vocabulary_.items())

    shs = []
    for i in range(D.shape[0]):
        Di = D.getrow(i)
        # features as list of (token, weight) tuples)
        # for j in Di.indices:
        #     print(voc[j])
        features = zip([voc[j] for j in Di.indices], Di.data)
        shs.append(Simhash(features))

    D0 = D.getrow(0)
    dict_features = dict(zip([voc[j] for j in D0.indices], D0.data))
    print(Simhash(dict_features).value)
    print(Simhash(data[0]).value)


def example4():
    data = [
        'How are you? I Am fine. blar blar blar blar blar Thanks.',
        'How are you i am fine. blar blar blar blar blar than',
        'This is simhash test.',
        'How are you i am fine. blar blar blar blar blar thank1'
    ]
    vec = TfidfVectorizer()
    D = vec.fit_transform(data)
    voc = dict((i, w) for w, i in vec.vocabulary_.items())

    shs = []
    for i in range(D.shape[0]):
        Di = D.getrow(i)
        features = zip([voc[j] for j in Di.indices], Di.data)
        shs.append(Simhash(features))

    objs = [(str(k), sh) for k, sh in enumerate(shs)]
    index = SimhashIndex(objs, k=10)

    vec = vec.transform([u'How are you i am fine. blar blar blar blar blar than'])
    dict_features = dict(zip([voc[j] for j in vec[0].indices], vec[0].data))

    print(dict_features)
    dups = index.get_near_dups(Simhash(dict_features))
    print(dups)


if __name__ == '__main__':
    example1()
    example2()
    example3()
    example4()
