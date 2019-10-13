#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
"""
# @Time     : 2018/10/24 21:53
# @Author   : YunSong
# @File     : jaccard.py
# @Software : PyCharm
# @version  : python3
"""
import re


def jaccard(news1_str=None, news2_str=None):
    news1_str = re.sub(r'[^\u4e00-\u9fa5]+', "", news1_str)
    news2_str = re.sub(r'[^\u4e00-\u9fa5]+', "", news2_str)
    data1_set = set(news1_str)
    data2_set = set(news2_str)
    a = data1_set & data2_set
    b = data1_set | data2_set
    try:
        sim_value = float(len(a)) / len(b)
    except ZeroDivisionError:
        sim_value = 0
    return sim_value

if __name__ == "__main__":
    news1_str = '把自由贸易试验区建设成为放新高地'
    news2_str = '把自由贸易试验区建设成为改革开放新高地 '
    sim_value = jaccard(news1_str,news2_str)
    print(sim_value)