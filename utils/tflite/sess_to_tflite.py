#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-10-21 下午12:57
# @Author  : song
# @File    : sess_to_tflite.py
# @Software: PyCharm


import tensorflow  as tf


DL_MODEL = None
ckpt_dir = None
with tf.Session() as sess:
    model = DL_MODEL
    cpkt = tf.train.get_checkpoint_state(ckpt_dir)
    model.saver.restore(sess, cpkt.model_checkpoint_path)
    # sess.run(tf.global_variables_initializer())#一定不能要这句
    converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors=[model.x], output_tensors=[model.pred])
    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)
