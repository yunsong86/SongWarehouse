#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 19-9-24 上午10:06
# @Author  : song
# @File    : app_lstm_crf_keras.py
# @Software: PyCharm
from app.ner.config import *
from app.ner.data_helper import DataHelper
from model.lstm_crf_keras import LstmCrfKeras

if __name__ == '__main__':
    x_train, y_train, word_dict = DataHelper.get_data("/mnt/hgfs/share_ubuntu/my_prj/data/nlp/ner/train.txt")

    embeddings_dict = DataHelper.get_pretrained_embedding(
        "/mnt/hgfs/share_ubuntu/my_prj/data/nlp/ner/token_vec_300.bin")

    embedding_matrix = DataHelper.get_embedding_matrix(embeddings_dict, word_dict, EMBED_DIM)
    # embedding_matrix = None
    config = {'vocab_size': len(word_dict),
              'embed_dim': EMBED_DIM,
              'num_classes': len(CLASS_DICT),
              'time_stamps': TIME_STAMPS,
              'batch_size': BATCH_SIZE,
              'epochs': EPOSHS,
              "embedding_matrix": embedding_matrix}

    # ====================train==========================#

    inst = LstmCrfKeras(**config)
    inst.train(x_train, y_train, "./ner_blstm.h5")

    # ====================infer==========================#

    model_path = "./ner_blstm.h5"
    inst = LstmCrfKeras(is_infer=True, model_path=model_path, **config)
    res = inst.infer(word_dict, TIME_STAMPS, CLASS_DICT, text="他最近头痛,流鼻涕,估计是发烧了")
    print(res)
    res = inst.infer(word_dict, TIME_STAMPS, CLASS_DICT, text="口腔溃疡可能需要多吃维生素")
    print(res)
