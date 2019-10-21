#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 14:47
# @Author  : song
# @File    : freeze_graph.py

#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 20:27
# @Author  : song
# @File    : text_cnn.py

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

    def _init_cls(self):
        if self.top_k is None:
            self.top_k = 1

    def build_graph(self):
        self.x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.y = tf.placeholder(tf.int32, [None, self.n_class], name='input_y')
        embeddings = tf.Variable(tf.random.uniform([self.vocab_size, self.embed_size]))
        x_embeded = tf.nn.embedding_lookup(embeddings, self.x)
        covres = []
        for size in self.kernel_size:
            cov = tf.layers.conv1d(inputs=x_embeded, filters=self.n_filters, kernel_size=size, strides=1, use_bias=True,
                                   activation=tf.nn.relu, padding='valid')
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

        self.logits = tf.layers.dense(covres, self.n_class)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
        self.pred = tf.arg_max(self.logits, 1)
        true_y = tf.arg_max(self.y, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, true_y), tf.float32))

    def train(self, sess, batch_x, batch_y):
        fd = {
            self.x: batch_x,
            self.y: batch_y,
        }
        loss, acc, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict=fd)
        return loss, acc
