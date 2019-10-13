#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 14:33
# @Author  : PanYunSong
# @File    : freeze_graph.py
import pickle
import tensorflow as tf
from classification.data_helper import *

if __name__ == '__main__':

    # =============================freeze_graph===================================#
    word_id = pickle.load(open("./word_id.pkl", 'rb'))

    checkpoint_dir = "./output/ckpt"

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
        output_node_names = ["acc/pred", 'fc/logits/BiasAdd']
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            graph,  # The graph_def is used to retrieve the nodes
            output_node_names  # The output node names are used to select the usefull nodes
        )
        # Finally we serialize and dump the output graph to the filesystem
        freeze_graph_path = "./output/freeze_graph"
        freeze_graph_name = "freeze_graph.pb"
        tf.train.write_graph(output_graph_def, freeze_graph_path, freeze_graph_name, as_text=False)
        # tf.train.write_graph(output_graph_def, freeze_graph_path, 'freeze_graph.pbtxt', as_text=True)
        print("%d ops in the final graph." % len(output_graph_def.node))

        # =============================predict===================================#

        from tensorflow.python.platform import gfile

        sess = tf.Session()
        with gfile.FastGFile('./output/freeze_graph/freeze_graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图

        # 需要有一个初始化的过程
        sess.run(tf.global_variables_initializer())

        input_x = sess.graph.get_tensor_by_name('input_x:0')
        pred = sess.graph.get_tensor_by_name('acc/pred:0')

        BiasAdd = sess.graph.get_tensor_by_name('fc/logits/BiasAdd:0')
        content = "超级好的卖家！之前不小心拍错了，客服非常耐心的帮我解答问题，快递也非常给力，必须赞！！！"

        cut = ' '.join(list(content))

        sent = [get_x(cut, word_id, max_len=50)]

        pred, biasAdd = sess.run([pred, BiasAdd], feed_dict={input_x: sent})
        print(pred)
        print(biasAdd)
