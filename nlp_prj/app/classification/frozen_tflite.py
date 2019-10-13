#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 15:10
# @Author  : PanYunSong
# @File    : frozen_tflite.py

import tensorflow  as tf

# =========================== Converting a GraphDef from file. =========================== #
graph_def_file = r"F:\share_ubuntu\my_prj\nlp_prj\app\classification\output\freeze_graph/freeze_graph.pb"

input_arrays = ["input_x"]
output_arrays = ["acc/pred"]
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file=graph_def_file, input_arrays=input_arrays,
                                                      output_arrays=output_arrays,
                                                      input_shapes={"input_x": [1, 50]})

# converter.post_training_quantize = True
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# ===========================test =========================== #

import numpy as np
import pickle

word_id = pickle.load(open("./word_id.pkl", 'rb'))


def get_x(sent, word_id, max_len=50):
    sent_id = [word_id.get(word, "UN") for word in sent.split()]
    if len(sent_id) < max_len:
        sent_id.extend([word_id['PAD']] * (max_len - len(sent_id)))
    return np.asarray(sent_id[:max_len])


content = "超级好的卖家！之前不小心拍错了，客服非常耐心的帮我解答问题，快递也非常给力，必须赞！！！"
cut = ' '.join(list(content))
sent = np.asarray(get_x(cut, word_id, max_len=50))
print(len(sent))
input_data = sent.reshape(1, 50)
input_data = input_data.astype(np.int32)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print('output_data shape:', output_data.shape)
print(output_data)
