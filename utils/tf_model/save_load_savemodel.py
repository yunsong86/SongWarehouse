#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 17:14
# @Author  : PanYunSong
# @File    : save_load_savemodel.py


import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

pb_file_path = "./"


def save():
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, name='x')
        y = tf.placeholder(tf.int32, name='y')
        b = tf.Variable(1, name='b')
        xy = tf.multiply(x, y)
        # 这里的输出需要加上name属性
        op = tf.add(xy, b, name='op_to_store')
        sess.run(tf.global_variables_initializer())

        # # 测试 OP
        feed_dict = {x: 10, y: 3}
        print(sess.run(op, feed_dict))

        builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path + 'savemodel')
        x_info = tf.saved_model.utils.build_tensor_info(x)
        y_info = tf.saved_model.utils.build_tensor_info(y)
        op_info = tf.saved_model.utils.build_tensor_info(op)

        signature_def_map = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"input_1": x_info, "input_2": y_info},
            outputs={"output": op_info},
            method_name=signature_constants.CLASSIFY_METHOD_NAME
        )

        # 构造模型保存的内容，指定要保存的 session，特定的 tag,
        # 输入输出信息字典，额外的信息
        builder.add_meta_graph_and_variables(sess, tags=['saved_model_tag_set'],
                                             signature_def_map={
                                                 signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def_map})

    builder.save()  # 保存 PB 模型


def load():
    with tf.Session(graph=tf.Graph()) as sess:
        metaGraphDef = tf.saved_model.loader.load(sess, tags=['saved_model_tag_set'], export_dir=pb_file_path + 'savemodel')
        signatureDef = metaGraphDef.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        x_info = signatureDef.inputs['input_1']
        y_info = signatureDef.inputs['input_2']
        op_info = signatureDef.outputs['output']

        input_x = tf.saved_model.utils.get_tensor_from_tensor_info(x_info, sess.graph)
        input_y = tf.saved_model.utils.get_tensor_from_tensor_info(y_info, sess.graph)
        op = tf.saved_model.utils.get_tensor_from_tensor_info(op_info, sess.graph)

        ret = sess.run(op, feed_dict={input_x: 5, input_y: 5})
        print(ret)


if __name__ == '__main__':
    save()
    load()
