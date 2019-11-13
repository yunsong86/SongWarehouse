#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/13 9:29
# @Author  : PanYunSong
# @File    : nms.py

import numpy as np


def nms(dets, thresh):
    # 首先为x1,y1,x2,y2,score赋值
    x1 = dets[:, 0]  # 取所有行第一列的数据
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 按照score的置信度将其排序,argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
    order = scores.argsort()[::-1]
    # 计算面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 保留最后需要保留的边框的索引
    keep = []
    while order.size > 0:
        # order[0]是目前置信度最大的，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他窗口的交叠的面积，此处的maximum是np中的广播机制
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        # 计算相交框的面积,左上右下，画图理解。注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算IOU：相交的面积/相并的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr < thresh)[0]  # np.where就可以得到索引值(3,0,8)之类的，再取第一个索引
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1（因为计算inter时是少了1的），所以要把这个1加回来
        order = order[inds + 1]

    return keep


# test
if __name__ == "__main__":
    dets = np.array([[30, 20, 230, 200, 1],
                     [50, 50, 260, 220, 0.9],
                     [210, 30, 420, 5, 0.8],
                     [430, 280, 460, 360, 0.7]])
    thresh = 0.35
    keep_dets = nms(dets, thresh)
    print(keep_dets)
    print(dets[keep_dets])
