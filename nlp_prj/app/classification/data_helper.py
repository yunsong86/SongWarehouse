#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 11:20
# @Author  : PanYunSong
# @File    : data_helper.py
import re
import jieba
import numpy as np
import pandas as pd


def prepare_data(neg_file, pos_file):
    neg = pd.read_excel(neg_file, header=None)
    pos = pd.read_excel(pos_file, header=None)
    neg['label'] = 0
    pos['label'] = 1
    data = neg.append(pos)
    data = data.sample(frac=1).reset_index(drop=True)
    data.rename(columns={0: "comm"}, inplace=True)
    return data


def clean_content(content):
    content = re.sub(r'[ #]', '', content)
    # cut = ' '.join(jieba.cut(content, HMM=False))
    cut = ' '.join(list(content))

    return cut


def build_vocab(data):
    vocab = set()
    for sent in data:
        vocab = vocab.union(set([word for word in sent.split()]))
    vocab = list(vocab)
    vocab.extend(['PAD', 'UN'])
    word_id = {}
    id_word = {}
    for i, w in enumerate(vocab):
        word_id[w] = i
        id_word[i] = w
    return word_id, id_word


def get_x(sent, word_id, max_len):
    sent_id = [word_id.get(word, "UN") for word in sent.split()]
    if len(sent_id) < max_len:
        sent_id.extend([word_id['PAD']] * (max_len - len(sent_id)))
    return np.asarray(sent_id[:max_len])


def generate_batch(x, y, batch_size=64, output_size=2, word_id=None, max_len=50):
    for i in range(0, len(x), batch_size):
        batch_x = x[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        id_batch_x = []
        id_batch_y = []

        for sen in batch_x:
            x_id = get_x(sen, word_id, max_len)
            id_batch_x.append(x_id)

        for label in batch_y:
            id_batch_y.append(np.eye(output_size)[label])

        yield id_batch_x, id_batch_y


def get_data(x, y, output_size=2, word_id=None, max_len=50):
    id_batch_x = []
    id_batch_y = []
    for sen in x:
        x_id = get_x(sen, word_id, max_len)
        id_batch_x.append(x_id)

    for label in y:
        id_batch_y.append(np.eye(output_size)[label])

    return id_batch_x, id_batch_y


if __name__ == '__main__':
    neg_file = r"F:\share_ubuntu\my_prj\data\nlp\classification/neg.xls"
    pos_file = r"F:\share_ubuntu\my_prj\data\nlp\classification/pos.xls"

    data = prepare_data(neg_file, pos_file)

    data['comm'] = data['comm'].apply(clean_content)
    word_id, id_word = build_vocab(list(data['comm']))
    print(word_id)
    import pickle
    with open("word_id.pkl", 'wb') as f:
        pickle.dump(word_id,f)

    train_frac = 0.7
    train_size = int(train_frac * data.shape[0])
    train_x, test_x, train_y, test_y = list(data['comm'][:train_size]), list(data['comm'][train_size:]), \
                                       list(data['label'][:train_size]), list(data['label'][train_size:])

    print(len(train_x))
    # for item in generate_batch(train_x, train_y, word_id=word_id):
    #     print(item[0])
    test_x = test_x[0:2]
    test_y = test_y[0:2]
    get_data(test_x,test_y,word_id=word_id)