#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 10:07
# @Author  : PanYunSong
# @File    : freeze_graph.py

import tensorflow as tf


def freeze_graph(checkpoint_dir, output_node_names, freeze_graph_path, freeze_graph_name):
    checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
    input_checkpoint = checkpoint_state.model_checkpoint_path
    print("=======checkpoint_state:", checkpoint_state)
    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = False
    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # import the meta graph and restore checkpoint
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        saver.restore(sess, input_checkpoint)
        graph = tf.get_default_graph().as_graph_def()
        print('================= Op List Begin=================')
        for node in graph.node:
            print(node.name)
        print('================= Op List End =================')
        # use a built-in TF helper to export variables to constants
        output_node_names = output_node_names
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            graph,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )
        # Finally we serialize and dump the output graph to the filesystem
        tf.train.write_graph(output_graph_def, freeze_graph_path, freeze_graph_name, as_text=False)
        # tf.train.write_graph(output_graph_def, freeze_graph_path, 'freeze_graph.pbtxt', as_text=True)
        print("%d ops in the final graph." % len(output_graph_def.node))
    return output_graph_def


def freeze_graph_test():
    from tensorflow.python.platform import gfile

    sess = tf.Session()
    with gfile.FastGFile('./freeze_graph_res/freeze_graph.pb', 'rb') as f:
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


if __name__ == '__main__':
    args = {'checkpoint_dir': './ckpt',
            'output_node_names': 'op_to_store',
            'freeze_graph_path': './freeze_graph_res',
            'freeze_graph_name': 'freeze_graph.pb'
            }
    freeze_graph(args['checkpoint_dir'], args['output_node_names'], args['freeze_graph_path'],
                 args['freeze_graph_name'])

    freeze_graph_test()

