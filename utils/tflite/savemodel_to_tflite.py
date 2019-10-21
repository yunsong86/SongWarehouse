#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-10-21 下午12:57
# @Author  : YunSong
# @File    : savemodel_to_tflite.py
# @Software: PyCharm



import tensorflow as tf

saved_model_dir = "./savemodel"

input_arrays = ["input_x"]
output_arrays = ["acc/pred"]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, tag_set={tf.saved_model.tag_constants.TRAINING},
                                                     input_arrays=input_arrays,
                                                     output_arrays=output_arrays,
                                                     input_shapes={"input_x": [1, 15000]})
tflite_model = converter.convert()
open("k12_text_clf_v1_savemodel.tflite", "wb").write(tflite_model)
