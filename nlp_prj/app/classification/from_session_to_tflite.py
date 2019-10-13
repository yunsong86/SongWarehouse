#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 19:47
# @Author  : PanYunSong
# @File    : tmp3.py

import pickle
import tensorflow as tf
from model.text_cnn import TextCnn
from classification.data_helper import *

config = {
    'n_class': 2,
    'embed_size': 128,
    'kernel_size': [3, 4, 5],
    'n_filters': 50,
    'top_k': 1,
    'lr': 1e-3
}

if __name__ == '__main__':
    word_id = pickle.load(open("./word_id.pkl", 'rb'))

    config['vocab_size'] = len(word_id)

    ckpt_dir = "./output/ckpt/txt_clf"

    with tf.Session() as sess:
        model = TextCnn(**config)
        cpkt = tf.train.get_checkpoint_state(ckpt_dir)
        if cpkt and cpkt.model_checkpoint_path:
            model.saver.restore(sess, cpkt.model_checkpoint_path)
        sess.run(tf.global_variables_initializer())

        converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors=[model.x], output_tensors=[model.pred])
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)
