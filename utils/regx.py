#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 18-10-15 上午11:43
# @Author  : song
# @File    : regx.py
# @Software: PyCharm
import re


class Regx(object):
    """
    正则表达式
    """
    regx_chinese = re.compile(r'[\u4e00-\u9fa5]+')
    regx_brackets = re.compile(r'《[^《》]+》|【[^【】]+】|（[^（）]+）')


if __name__ == '__main__':
    content = '假啊里，jflaj减肥啦放假啊里'
    for item in re.findall(Regx.regx_chinese, content):
        print(item)

    content = '假《啊里，jfla》j减【肥啦放假】啊里（肥啦放假）啊里'
    for item in re.findall(Regx.regx_brackets, content):
        print(item)
