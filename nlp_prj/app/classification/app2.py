#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 20:07
# @File    : app2.py

import numpy as np
import pandas as pd
import re
import tensorflow as tf
import numpy as np
from sklearn import metrics
import tensorflow as tf


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


def prepare_data(neg_file, pos_file):
    neg = pd.read_excel(neg_file, header=None)
    pos = pd.read_excel(pos_file, header=None)
    neg['label'] = 0
    pos['label'] = 1
    data = neg.append(pos)
    data = data.sample(frac=1).reset_index(drop=True)
    data.rename(columns={0: "comm"}, inplace=True)
    return data


def get_x(sent, word_id, max_len):
    sent_id = [word_id.get(word, "UN") for word in sent.split()]
    if len(sent_id) < max_len:
        sent_id.extend([word_id['PAD']] * (max_len - len(sent_id)))
    return np.asarray(sent_id[:max_len])


def get_data(x, y, output_size=2, word_id=None, max_len=50):
    id_batch_x = []
    id_batch_y = []
    for sen in x:
        x_id = get_x(sen, word_id, max_len)
        id_batch_x.append(x_id)

    for label in y:
        id_batch_y.append([label])

    return id_batch_x, id_batch_y


class TextCnn(object):
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size")
        self.embed_size = kwargs.get("embed_size")
        self.kernel_size = kwargs.get("kernels", [3, 4, 5])
        self.n_filters = kwargs.get("n_filters", 50)
        self.top_k = kwargs.get("top_k", None)
        self.lr = kwargs.get("lr")
        self._init_cls()
        self.build_graph()
        self.saver = tf.train.Saver()

    def _init_cls(self):
        if self.top_k is None:
            self.top_k = 1

    def build_graph(self):
        self.global_step = tf.Variable(-1, trainable=False, name='global_step')

        self.x = tf.placeholder(tf.int32, [None, 50], name='input_x')
        self.y = tf.placeholder(tf.float32, [None, 1], name='input_y')

        with tf.variable_scope("intput_emd"):
            embeddings = tf.Variable(tf.random.uniform([self.vocab_size, self.embed_size]))
            x_embeded = tf.nn.embedding_lookup(embeddings, self.x)

        with tf.variable_scope("mul_filter_cnn"):
            covres = []
            for size in self.kernel_size:
                cov = tf.layers.conv1d(inputs=x_embeded, filters=self.n_filters, kernel_size=size, strides=1,
                                       use_bias=True,
                                       activation=tf.nn.relu, padding='valid', name="cov_%s" % size)

                outputs = tf.reduce_max(cov, reduction_indices=[1], name='reduce_max')
                covres.append(outputs)
            covres = tf.concat(covres, axis=1)
            covres = tf.reshape(covres, [-1, self.top_k * self.n_filters * len(self.kernel_size)])

        with tf.variable_scope("fc"):
            self.logits = tf.layers.dense(covres, 1, name='logits')
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            print("================================================")
            print(self.logits)
            print(self.y)
            print("================================================")

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        with tf.variable_scope("optimizer"):
            # lr: 1e-3  f1_test:0.9133 step:2000
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

        with tf.variable_scope("acc"):
            self.pred = tf.arg_max(self.logits, 1, name="pred")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.arg_max(self.y, 1)), tf.float32),
                                           name='accuracy')

        with tf.variable_scope("summary"):
            tf.summary.scalar("acc", self.accuracy)
            tf.summary.scalar("loss", self.loss)
            self.summary = tf.summary.merge_all()

    def train(self, sess, batch_x, batch_y):
        fd = {
            self.x: batch_x,
            self.y: batch_y,
        }
        global_step, loss, acc, _, summary = sess.run(
            [self.global_step, self.loss, self.accuracy, self.optimizer, self.summary], feed_dict=fd)
        return global_step, loss, acc, summary

    def evaluate(self, sess, batch_x, batch_y):
        fd = {
            self.x: batch_x,
            self.y: batch_y,
        }
        acc, pred, loss, global_step, summary = sess.run(
            [self.accuracy, self.pred, self.loss, self.global_step, self.summary], feed_dict=fd)

        print("*************pred",pred)
        print("*************batch_y",batch_y)

        y_true = np.argmax(batch_y, 1)
        print("*************batch_y",batch_y)

        f1 = metrics.f1_score(y_true, pred)
        return acc, f1, loss, global_step, summary



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
            id_batch_y.append([label])

        yield id_batch_x, id_batch_y

if __name__ == '__main__':
    config = {
        'n_class': 2,
        'embed_size': 128,
        'kernel_size': [3, 4, 5],
        'n_filters': 50,
        'top_k': 1,
        'lr': 1e-3
    }
    neg_file = r"F:\share_ubuntu\my_prj\data\nlp\classification/neg.xls"
    pos_file = r"F:\share_ubuntu\my_prj\data\nlp\classification/pos.xls"
    data = prepare_data(neg_file, pos_file)
    data['comm'] = data['comm'].apply(clean_content)
    word_id, id_word = build_vocab(list(data['comm']))
    print(word_id)
    train_frac = 0.7
    train_size = int(train_frac * data.shape[0])

    train_x, test_x = list(data['comm'][:train_size]), list(data['comm'][train_size:])
    train_y, test_y = list(data['label'][:train_size]), list(data['label'][train_size:])
    config['vocab_size'] = len(word_id)

    epoch = 14000
    with tf.Session() as sess:
        model = TextCnn(**config)
        sess.run(tf.global_variables_initializer())
        f1_max = 0
        f1_count = 0
        for i in range(epoch):
            for item in generate_batch(train_x, train_y, word_id=word_id):
                global_step, loss_train, acc_train, summary_train = model.train(sess, batch_x=item[0], batch_y=item[1])
                # print(global_step,loss_train,acc_train)

                if global_step % 100 == 0:
                    batch_x, batch_y = get_data(test_x, test_y, word_id=word_id)
                    acc_test, f1, loss_test, global_step, summary_dev = model.evaluate(sess, batch_x=batch_x,
                                                                                       batch_y=batch_y)

                    print(
                        'step:{}\tloss_train:{:.4f} acc_train:{:.4f} loss_test:{:.4f} acc_test:{:.4f} f1_test:{:.4f}'.format(
                            global_step,
                            loss_train, acc_train, loss_test, acc_test, f1))

