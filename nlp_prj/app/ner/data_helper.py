#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 19-9-24 上午10:18
# @Author  : YunSong
# @File    : data_helper.py
# @Software: PyCharm

import numpy as np
from app.ner.config import CLASS_DICT


class DataHelper(object):
    def __init__(self):
        pass

    @staticmethod
    def get_data(corpus_path):
        # "/mnt/hgfs/share_ubuntu/my_prj/data/nlp/ner/train.txt"
        datas = []
        sample_x = []
        sample_y = []
        vocabs = {'UNK'}
        for line in open(corpus_path, encoding='utf-8'):
            line = line.rstrip().split('\t')
            if not line:
                continue
            char = line[0]
            if not char:
                continue
            cate = line[-1]
            sample_x.append(char)
            sample_y.append(cate)
            vocabs.add(char)
            if char in ['。', '?', '!', '！', '？']:
                datas.append([sample_x, sample_y])
                sample_x = []
                sample_y = []
        word_dict = {wd: index for index, wd in enumerate(list(vocabs))}

        x_train = [[word_dict[char] for char in data[0]] for data in datas]
        y_train = [[CLASS_DICT[label] for label in data[1]] for data in datas]

        return x_train, y_train, word_dict

    @staticmethod
    def get_data2(corpus_path):
        # "/mnt/hgfs/share_ubuntu/my_prj/data/nlp/ner/train.txt"
        datas = []
        sample_x = []
        sample_y = []
        vocabs = {'UNK'}
        for line in open(corpus_path, encoding='utf-8'):
            line = line.rstrip().split('\t')
            if not line:
                continue
            char = line[0]
            if not char:
                continue
            cate = line[-1]
            sample_x.append(char)
            sample_y.append(cate)
            vocabs.add(char)
            if char in ['。', '?', '!', '！', '？']:
                datas.append([sample_x, sample_y])
                sample_x = []
                sample_y = []
        word_dict = {wd: index for index, wd in enumerate(list(vocabs))}

        x_train = [np.asarray([word_dict[char] for char in data[0][0:4]]) for data in datas]
        y_train = [np.asarray([CLASS_DICT[label] for label in data[1][0:4]]) for data in datas]

        return x_train, y_train, word_dict

    @staticmethod
    def get_pretrained_embedding(pretrain_vec_path):
        embeddings_dict = {}
        with open(pretrain_vec_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        return embeddings_dict

    @staticmethod
    def get_embedding_matrix(embeddings_dict, word_dict, embed_dim):
        embedding_matrix = np.zeros((len(word_dict) + 1, embed_dim))  # keras : +1
        for word, i in word_dict.items():
            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    @staticmethod
    def get_embedding_matrix_tf(embeddings_dict, word_dict, embed_dim):
        embedding_matrix = np.zeros((len(word_dict), embed_dim))
        for word, i in word_dict.items():
            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    @staticmethod
    def pad_sequence(sequences, pad_mark=0):
        max_len = max(map(lambda x: len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list

    @staticmethod
    def batch_data(x, y, seq_len, size):
        data_len = len(x)
        for i in range(0, data_len, size):
            batch_x = x[i:min(i + size, data_len)]
            batch_y = y[i:min(i + size, data_len)]
            batch_seq_len = seq_len[i:min(i + size, data_len)]
            yield batch_x, batch_y, batch_seq_len
