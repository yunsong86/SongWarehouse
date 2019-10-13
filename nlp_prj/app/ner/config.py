#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 19-9-24 上午10:29
# @Author  : YunSong
# @File    : config.py
# @Software: PyCharm


# ========================lstm-crf-keras==================================#

CLASS_DICT = {
    'O': 0,
    'TREATMENT-I': 1,
    'TREATMENT-B': 2,
    'BODY-B': 3,
    'BODY-I': 4,
    'SIGNS-I': 5,
    'SIGNS-B': 6,
    'CHECK-B': 7,
    'CHECK-I': 8,
    'DISEASE-I': 9,
    'DISEASE-B': 10}
EPOSHS = 10
EMBED_DIM = 300
BATCH_SIZE = 128
TIME_STAMPS = 100
