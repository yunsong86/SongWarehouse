#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 14:22
# @Author  : song
# @File    : ckpt_pd.py


import tensorflow as tf
import pickle
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

    # =============================to pd===================================#

    ckpt_dir = "./output/ckpt"
    word_id = pickle.load(open("./word_id.pkl", 'rb'))

    config['vocab_size'] = len(word_id)
    with tf.Session() as sess:
        model = TextCnn(**config)
        cpkt = tf.train.get_checkpoint_state(ckpt_dir)
        if cpkt and cpkt.model_checkpoint_path:
            print(cpkt.model_checkpoint_path)
            model.saver.restore(sess, cpkt.model_checkpoint_path)

        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                      ['acc/pred', "fc/logits/BiasAdd"])

        # 写入序列化的 PB 文件
        with tf.gfile.FastGFile("./output/pd/" + 'txt_clf.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
    #
    # =============================predict===================================#

    sess = tf.Session()
    with tf.gfile.FastGFile("./output/pd/" + 'txt_clf.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    # 需要有一个初始化的过程
    sess.run(tf.global_variables_initializer())

    # 输入
    input_x = sess.graph.get_tensor_by_name('input_x:0')

    pred = sess.graph.get_tensor_by_name('acc/pred:0')
    BiasAdd = sess.graph.get_tensor_by_name('fc/logits/BiasAdd:0')

    content = "超级好的卖家！之前不小心拍错了，客服非常耐心的帮我解答问题，快递也非常给力，必须赞！！！"

    cut = ' '.join(list(content))

    sent = [get_x(cut, word_id, max_len=50)]

    pred, BiasAdd = sess.run([pred, BiasAdd], feed_dict={input_x: sent})
    print(pred)
    print(BiasAdd)
