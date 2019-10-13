#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 14:57
# @Author  : PanYunSong
# @File    : lstm_crf.py

import tensorflow as tf


class LstmCrf(object):
    def __init__(self, **kwargs):
        self.dropout_rate = kwargs.get("dropout_rate")
        self.num_layers = kwargs.get("num_layers")
        self.num_classes = kwargs.get("num_classes")
        self.hidden_dim = kwargs.get("hidden_dim")
        self.word_emb_dim = kwargs.get("word_emb_dim")
        self.word_vocab_size = kwargs.get("word_vocab_size")
        self.learning_rate = kwargs.get("learning_rate")
        self.optimizer = kwargs.get("optimizer")
        self.clip_grad = kwargs.get("optimizer", 5)
        # pretrain_emding_matrix
        self.pretrain_emding_matrix = kwargs.get("pretrain_emding_matrix")

        self._init_cls()

    def _init_cls(self):
        self.seq_len = None
        self.train_op = None
        self.loss_op = None
        self.accuracy_op = None
        self.intput_x = None
        self.input_y = None
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.build_graph()
        self.saver = tf.train.Saver()

    def build_graph(self):
        self.intput_x = tf.placeholder(tf.int32, shape=[None, None], name="intput_x")
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name="input_y")
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")

        with tf.variable_scope("intput_emd"):
            if self.pretrain_emding_matrix is not None:
                print('**********************pretrain_emding_matrix************************')
                word_embedding = tf.get_variable("emb-word", [self.word_vocab_size, self.word_emb_dim],
                                                 initializer=tf.constant_initializer(self.pretrain_emding_matrix),
                                                 trainable=False)
            else:
                print('**********************no pretrain_emding_matrix************************')

                word_embedding = tf.get_variable("emb-word", [self.word_vocab_size, self.word_emb_dim])

            intput_x_emb = tf.nn.embedding_lookup(word_embedding, self.intput_x)

        with tf.variable_scope("bilstm"):
            stack_cell_fw = []
            stack_cell_bw = []
            for _ in range(self.num_layers):
                lstm_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
                lstm_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
                stack_cell_fw.append(
                    tf.nn.rnn_cell.DropoutWrapper(cell=lstm_fw, output_keep_prob=(1 - self.dropout_rate)))
                stack_cell_bw.append(
                    tf.nn.rnn_cell.DropoutWrapper(cell=lstm_bw, output_keep_prob=(1 - self.dropout_rate)))

            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell(stack_cell_fw)
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell(stack_cell_bw)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell_fw,
                cell_bw=lstm_cell_bw,
                inputs=intput_x_emb,
                sequence_length=self.seq_len,
                dtype=tf.float32,
            )
            output = tf.concat([output_fw, output_bw], axis=-1)

        with tf.variable_scope("dense"):
            W = tf.get_variable("W", shape=[self.hidden_dim * 2, self.num_classes], dtype=tf.float32)
            b = tf.get_variable("b", shape=[self.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits_op = tf.reshape(pred, [-1, nsteps, self.num_classes])

        with tf.variable_scope("CRF"):
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits_op, self.input_y, self.seq_len)
            self.loss_op = tf.reduce_mean(-log_likelihood)

        with tf.variable_scope("acc"):
            mask = tf.sequence_mask(self.seq_len)
            viterbi_seq, viterbi_score = tf.contrib.crf.crf_decode(self.logits_op, trans_params, self.seq_len)
            viterbi_output = tf.boolean_mask(viterbi_seq, mask)
            viterbi_y = tf.boolean_mask(self.input_y, mask)
            correct_pred = tf.equal(tf.cast(viterbi_output, tf.int32), viterbi_y)
            self.accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        with tf.variable_scope("optimizer"):
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            else:
                optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            grads_and_vars = optim.compute_gradients(self.loss_op)
            grads_and_vars_clip = [[tf.clip_by_value(g, -5, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
            # self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op,
            #                                                                                   global_step=self.global_step)

        with tf.variable_scope("summary"):
            tf.summary.scalar("acc", self.accuracy_op)
            tf.summary.scalar("loss", self.loss_op)
            self.summary_op = tf.summary.merge_all()

    def train(self, sess, data):
        fd = {
            self.intput_x: data['intput_x'],
            self.input_y: data['input_y'],
            self.seq_len: data['seq_len']
        }
        _, loss, acc, sma, gstep = sess.run(
            [self.train_op, self.loss_op, self.accuracy_op, self.summary_op, self.global_step], feed_dict=fd)
        return loss, acc, sma, gstep
