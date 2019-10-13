#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 16:14
# @Author  : PanYunSong
# @File    : sava_load_pd.py


import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

pb_file_path = "./pd/"


def save_pd():
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, name='x')
        y = tf.placeholder(tf.int32, name='y')
        print(y)
        b = tf.Variable(1, name='b')
        xy = tf.multiply(x, y)
        # 这里的输出需要加上name属性

        op = tf.add(xy, b, name='op_to_store')

        sess.run(tf.global_variables_initializer())

        # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        # 测试 OP
        feed_dict = {x: 10, y: 3}
        print(sess.run(op, feed_dict))

        # 写入序列化的 PB 文件
        with tf.gfile.FastGFile(pb_file_path + 'model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        # 输出
        # INFO:tensorflow:Froze 1 variables.
        # Converted 1 variables to const ops.
        # 31


def load_pd():
    sess = tf.Session()
    with gfile.FastGFile(pb_file_path + 'model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    # 需要有一个初始化的过程
    sess.run(tf.global_variables_initializer())

    # 需要先复原变量
    # print(sess.run('b:0'))
    # 1

    # 输入
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op, feed_dict={input_x: 5, input_y: 5})
    print(ret)


if __name__ == "__main__":
    save_pd()
    load_pd()
