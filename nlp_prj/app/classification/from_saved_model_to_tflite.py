#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 20:12
# @Author  : PanYunSong
# @File    : from_saved_model_to_tflite.py


"""
saved_model_cli show --dir ./savemodel --all



tflite_convert   --output_file=foo.tflite   --saved_model_dir=/home/panyunsong/test_savemodel/savemodel  --saved_model_tag_set='saved_model_tag_set' --input_shapes=1 --input_arrays=x

MetaGraphDef with tag-set: 'train' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['dropout_input'] tensor_info:
        dtype: DT_FLOAT
        shape: unknown_rank
        name: dropout_keep_prob:0
    inputs['label_input'] tensor_info:
        dtype: DT_INT32
        shape: (-1)
        name: label_input:0
    inputs['text_input'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 30000)
        name: text_input:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['prediction_output'] tensor_info:
        dtype: DT_INT64
        shape: (-1)
        name: output/predictions:0
    outputs['probability_output'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 7)
        name: output/scores:0
  Method name is: tensorflow/serving/classify

"""

import tensorflow as tf

saved_model_dir = "/home/panyunsong/textcnn/experiments/text_CNN_exp/models/1"

input_arrays = ["text_input"]
output_arrays = ["output/predictions"]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, tag_set={'train'}, input_arrays=input_arrays,
                                                     output_arrays=output_arrays,
                                                     input_shapes={"text_input": [1, 30000]})
tflite_model = converter.convert()
open("textcnn.tflite", "wb").write(tflite_model)
