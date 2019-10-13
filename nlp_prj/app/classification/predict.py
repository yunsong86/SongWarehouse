#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 20:28
# @Author  : PanYunSong
# @File    : app.py
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
    ckpt_dir = "./output/ckpt"
    word_id = pickle.load(open("./word_id.pkl", 'rb'))

    config['vocab_size'] = len(word_id)
    with tf.Session() as sess:
        model = TextCnn(**config)
        cpkt = tf.train.get_checkpoint_state(ckpt_dir)
        if cpkt and cpkt.model_checkpoint_path:
            print(cpkt.model_checkpoint_path)
            model.saver.restore(sess, cpkt.model_checkpoint_path)

        content = "超级好的卖家！之前不小心拍错了，客服非常耐心的帮我解答问题，快递也非常给力，必须赞！！！"
        cut = ' '.join(list(content))

        sent = [get_x(cut, word_id, max_len=50)]
        fd = {
            model.x: sent,
            model.keep_prob: 1
        }
        pred, logits = sess.run([model.pred, model.logits], feed_dict=fd)

        print(pred)
        print(logits)
