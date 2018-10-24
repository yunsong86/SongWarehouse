#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time     : 2018/10/24 22:02
# @Author   : YunSong
# @File     : nlp_sim.py
# @Software : PyCharm
# @version  : python3
"""

from nlp.sim.jaccard import jaccard

if __name__ == "__main__":
    news1_str = '把自由贸易试验区建设成为放新高地'
    news2_str = '把自由贸易试验区建设革开放新高地 '
    sim_value = jaccard(news1_str, news2_str)
    print(sim_value)

