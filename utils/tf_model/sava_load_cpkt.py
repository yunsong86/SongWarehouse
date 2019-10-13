#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 17:37
# @Author  : PanYunSong
# @File    : sava_load_cpkt.py

# !/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 16:14
# @Author  : PanYunSong
# @File    : sava_load_pd.py


import tensorflow as tf

ckpt_dir = "./ckpt/"
model_name = "model"


def save_pd():
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, name='x')

        y = tf.placeholder(tf.int32, name='y')

        b = tf.Variable(1, name='b')

        xy = tf.multiply(x, y)
        # 这里的输出需要加上name属性

        op = tf.add(xy, b, name='op_to_store')

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.save(sess, "./ckpt/model",global_step=20000)

        # 测试 OP
        feed_dict = {x: 10, y: 3}
        print(sess.run(op, feed_dict))


def load_ckpt():
    sess = tf.Session()

    # 《《《 加载模型结构 》》》    # 只需要指定目录就可以恢复所有变量信息

    saver = tf.train.import_meta_graph('./ckpt/model.meta')

    saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))

    # 直接获取保存的变量
    print(sess.run('b:0'))

    # 获取placeholder变量
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')
    # 获取需要进行计算的operator
    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op, {input_x: 5, input_y: 5})
    print(ret)

    # 加入新的操作
    add_on_op = tf.multiply(op, 2)

    ret = sess.run(add_on_op, {input_x: 5, input_y: 5})
    print(ret)


if __name__ == "__main__":
    save_pd()
    load_ckpt()
