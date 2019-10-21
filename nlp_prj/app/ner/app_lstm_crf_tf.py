#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/9/29 10:40
# @Author  : song
# @File    : app_lstm_crf_tf.py

import os
import tensorflow as tf
from app.ner.config import *
from app.ner.data_helper import DataHelper
from model.lstm_crf import LstmCrf

tf.logging.set_verbosity(tf.logging.INFO)

data_path = r"F:\share_ubuntu\my_prj\data\nlp\ner/train.txt"
x, y, word_dict = DataHelper.get_data(data_path)
pad_x, x_len = DataHelper.pad_sequence(x)
pad_y, y_len = DataHelper.pad_sequence(y)

embeddings_dict = DataHelper.get_pretrained_embedding(
    r"F:\share_ubuntu\my_prj\data\nlp\ner/token_vec_300.bin")

embed_dim = 300
embedding_matrix = DataHelper.get_embedding_matrix_tf(embeddings_dict, word_dict, embed_dim)
# embedding_matrix = None

optimizer = 'Adam'  # Adam  Adadelta Adagrad  RMSProp  Momentum  SGD

config = {
    "dropout_rate": 0.5,
    "num_layers": 2,
    "num_classes": len(CLASS_DICT),
    "hidden_dim": 120,
    "word_emb_dim": embed_dim,
    "word_vocab_size": len(word_dict),
    "learning_rate": 0.002,
    "pretrain_emding_matrix": embedding_matrix,
}

epochs = 5
cpkt_dir = "./output/checkpoints"
cpkt_name = os.path.join(cpkt_dir, "ner_lstmcrf")

model = LstmCrf(**config)
with tf.Session() as sess:

    # cpkt = tf.train.get_checkpoint_state(cpkt_dir)
    # if cpkt and cpkt.model_checkpoint_path:
    #     model.saver.restore(sess, cpkt.model_checkpoint_path)

    summary_writer = tf.summary.FileWriter("./output/summary", sess.graph)
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for item in DataHelper.batch_data(pad_x, pad_y, y_len, size=128):
            data = {"intput_x": item[0], "input_y": item[1], "seq_len": item[2]}
            loss, acc, sma, gstep = model.train(sess, data)
            if gstep % 50 == 0:
                print("step:%s\tacc:%s" % (gstep, acc))
                save_name = model.saver.save(sess, cpkt_name, global_step=gstep)
                summary_writer.add_summary(sma, gstep)
