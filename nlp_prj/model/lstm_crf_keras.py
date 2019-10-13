#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 19-9-23 下午9:30
# @Author  : YunSong
# @File    : lstm_crf_keras.py
# @Software: PyCharm

import numpy as np
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF


class LstmCrfKeras():
    def __init__(self, is_infer=False,model_path=None, **kwargs, ):
        self.vocab_size = kwargs.get('vocab_size', None)
        self.embedding_dim = kwargs.get('embed_dim', None)
        self.num_classes = kwargs.get('num_classes', None)
        self.embedding_matrix = kwargs.get('embedding_matrix', None)
        self.time_stamps = kwargs.get('time_stamps', None)
        self.batch_size = kwargs.get('batch_size', None)
        self.epochs = kwargs.get('epochs', None)
        self.word_dict = kwargs.get('word_dict', None)
        self.class_dict = kwargs.get('class_dict', None)
        self.model = None
        self._build_network()
        if is_infer is True:
            self.model.load_weights(model_path)

    def _build_network(self):
        model = Sequential()
        if self.embedding_matrix is not None:
            print('****************************')
            embedding_layer = Embedding(self.vocab_size + 1,
                                        self.embedding_dim,
                                        weights=[self.embedding_matrix],
                                        input_length=self.time_stamps,
                                        trainable=False,
                                        mask_zero=True)
        else:
            embedding_layer = Embedding(self.vocab_size + 1,
                                        self.embedding_dim,
                                        input_length=self.time_stamps,
                                        mask_zero=True)
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.num_classes)))
        crf_layer = CRF(self.num_classes, sparse_target=True)
        model.add(crf_layer)
        model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
        model.summary()
        self.model = model

    def train(self, x_train, y_train, model_path):
        x_train = pad_sequences(x_train, self.time_stamps)
        y = pad_sequences(y_train, self.time_stamps)
        y_train = np.expand_dims(y, 2)
        history = self.model.fit(x_train, y_train, validation_split=0.2, batch_size=self.batch_size,
                                 epochs=self.epochs)
        self.model.save(model_path)

    def infer(self,  word_dict, time_stamps, class_dict, text="他最近头痛,流鼻涕,估计是发烧了"):
        x = []
        for char in text:
            if char not in word_dict:
                char = 'UNK'
            x.append(word_dict.get(char))
        x = pad_sequences([x], time_stamps)
        raw = self.model.predict(x)[0][-time_stamps:]
        result = [np.argmax(row) for row in raw]
        chars = [i for i in text]
        label_dict = {j: i for i, j in class_dict.items()}
        tags = [label_dict[i] for i in result][len(result) - len(text):]
        res = list(zip(chars, tags))
        return res
