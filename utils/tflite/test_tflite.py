#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-10-21 下午12:59
# @Author  : YunSong
# @File    : test_tflite.py
# @Software: PyCharm



content = None

input_data = content.reshape(1, 15000)
input_data = input_data.astype(np.int32)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print('output_data shape:', output_data.shape)
print(output_data)
