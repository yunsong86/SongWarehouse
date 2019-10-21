# !/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 20:27
# @Author  : song
# @File    : text_cnn.py

import numpy as np
from sklearn import metrics
import tensorflow as tf


class TextCnn(object):
    def __init__(self, **kwargs):
        self.n_class = kwargs.get("n_class")
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
        self.y = tf.placeholder(tf.int32, [None, self.n_class], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope("intput_emd"):
            embeddings = tf.Variable(tf.random.uniform([self.vocab_size, self.embed_size]))
            x_embeded = tf.nn.embedding_lookup(embeddings, self.x)

        with tf.variable_scope("mul_filter_cnn"):
            covres = []
            for size in self.kernel_size:
                cov = tf.layers.conv1d(inputs=x_embeded, filters=self.n_filters, kernel_size=size, strides=1,
                                       use_bias=True,
                                       activation=tf.nn.relu, padding='valid', name="cov_%s" % size)
                if self.top_k > 1:
                    # cov: batchsize,(step-size)/strides +1,filters
                    cov = tf.transpose(cov, [0, 2, 1])
                    topk = tf.nn.top_k(cov, self.top_k).values
                    # cov transpose的目的是为了获取filters结果的topk的值
                    topk = tf.transpose(topk, [0, 2, 1])
                    # 得到topk的值之后，再返回回来，转变为conv卷积完成之后的形式
                    covres.append(topk)
                    # covres shape是3，batchsize，topk,filters
                else:
                    outputs = tf.reduce_max(cov, reduction_indices=[1], name='reduce_max')
                    covres.append(outputs)
            covres = tf.concat(covres, axis=1)
            covres = tf.reshape(covres, [-1, self.top_k * self.n_filters * len(self.kernel_size)])

        with tf.variable_scope("fc"):
            # n_hdid_dim = 0.5 * (self.top_k * self.n_filters * len(self.kernel_size))
            # covres = tf.layers.dense(covres, n_hdid_dim, name='fc1')
            # covres = tf.contrib.layers.dropout(covres, self.keep_prob)
            # covres = tf.nn.relu(covres)

            self.logits = tf.layers.dense(covres, self.n_class, name='logits')
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        with tf.variable_scope("optimizer"):
            # lr: 1e-3  f1_test:0.9133 step:2000
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

            # lr: 1e-3 f1_test:0.8739 step:323300
            # lr: 1e-2 f1_test:0.8847 step:193300
            # self.optimizer = tf.train.AdagradOptimizer(0.01).minimize(self.loss, global_step=self.global_step)

            # f1_test:0.8801 step:3070500
            # self.optimizer = tf.train.AdadeltaOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

            # f1_test:0.8672 step:3070500
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.9).minimize(self.loss,self.global_step)

            # 0.911913912375096  step:7700
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss,self.global_step)

            # lr: 1e-3 f1_test:0.8521 step:114500
            self.optimizer_sgd = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss,
                                                                                                   self.global_step)

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
            self.keep_prob: 0.5
        }
        global_step, loss, acc, _, summary = sess.run(
            [self.global_step, self.loss, self.accuracy, self.optimizer, self.summary], feed_dict=fd)
        return global_step, loss, acc, summary

    def evaluate(self, sess, batch_x, batch_y):
        fd = {
            self.x: batch_x,
            self.y: batch_y,
            self.keep_prob: 1
        }
        acc, pred, loss, global_step, summary = sess.run(
            [self.accuracy, self.pred, self.loss, self.global_step, self.summary], feed_dict=fd)
        y_true = np.argmax(batch_y, 1)
        f1 = metrics.f1_score(y_true, pred)
        return acc, f1, loss, global_step, summary
