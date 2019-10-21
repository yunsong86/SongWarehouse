#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-10-21 下午12:57
# @Author  : YunSong
# @File    : pd_to_tflite.py
# @Software: PyCharm


import tensorflow  as tf

# =========================== Converting a GraphDef from file. =========================== #
graph_def_file = r"./freeze_graph/k12_text_clf_v1.pb"

input_arrays = ["input_x"]
output_arrays = ["acc/pred"]
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file=graph_def_file, input_arrays=input_arrays,
                                                      output_arrays=output_arrays,
                                                      input_shapes={"input_x": [1, 15000]})

# converter.post_training_quantize = True
tflite_model = converter.convert()
open("k12_text_clf_v1_pd.tflite", "wb").write(tflite_model)
